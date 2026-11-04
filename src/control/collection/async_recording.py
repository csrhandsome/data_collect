"""Serialize episode I/O outside the servo loop, including reopen and publication."""

import logging
import queue
import threading
import time
from dataclasses import replace

from control.recording_writer import FreshCameraPair

logger = logging.getLogger(__name__)


class AsyncEpisodeRecorder:
    def __init__(self, recorder, *, max_pending=1024):
        self.recorder = recorder
        self.root = recorder.root
        self.active = False
        self.started_ns = 0
        self._gate = FreshCameraPair()
        self._queue = queue.Queue(maxsize=max_pending)
        self._error = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="episode-io", daemon=True)
        self._thread.start()

    @property
    def dataset(self):
        # Only read after close/drain; the worker owns mutable dataset metadata.
        return self.recorder.dataset

    def check(self):
        if self._error is not None:
            raise RuntimeError("Episode I/O failed; acquisition stopped") from self._error

    def _submit(self, method, *args, **kwargs):
        self.check()
        if self._closed:
            raise RuntimeError("Recorder is closed")
        try:
            self._queue.put_nowait((method, args, kwargs))
        except queue.Full as exc:
            raise RuntimeError("Episode I/O backlog is full; acquisition stopped without dropping frames") from exc

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                method, args, kwargs = item
                if self._error is None:
                    getattr(self.recorder, method)(*args, **kwargs)
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
                    logger.exception("Episode I/O worker failed")
            finally:
                self._queue.task_done()

    def start(self):
        if self.active:
            raise RuntimeError("Episode is already recording")
        self.started_ns = time.monotonic_ns()
        self._submit("start", started_ns=self.started_ns)
        self._gate.reset()
        self.active = True

    def enter_critical_phase(self, timestamp_ns):
        if self.active:
            self._submit("enter_critical_phase", timestamp_ns)

    def trace(self, state, sample, position, quaternion, **kwargs):
        if self.active:
            # Mapping/VR arrays may be reused on subsequent control ticks.
            sample = replace(sample, position=sample.position.copy(),
                             quaternion_xyzw=sample.quaternion_xyzw.copy())
            kwargs.setdefault("trace_sample_ns", time.monotonic_ns())
            self._submit("trace", state, sample, position.copy(), quaternion.copy(), **kwargs)

    def frame(self, state, pair, tactile=(None, None)):
        if self.active and self._gate.accept(pair.front_ns, pair.wrist_ns):
            pair = replace(pair, front=pair.front.copy(), wrist=pair.wrist.copy())
            tactile = tuple(image.copy() if image is not None else None for image in tactile)
            self._submit("frame", state, pair, tactile)

    def finish(self, state, **kwargs):
        if not self.active:
            return
        started = time.perf_counter()
        self._submit("finish", state, stopped_ns=time.monotonic_ns(), **kwargs)
        self.active = False
        logger.info("Episode completion queued in %.3f ms (pending=%s)",
                    (time.perf_counter() - started) * 1000, self._queue.qsize())

    def close(self):
        if self._closed:
            self.check()
            return
        self._closed = True
        started = time.perf_counter()
        logger.info("Waiting for episode I/O (pending=%s)", self._queue.qsize())
        self._queue.put(None)
        self._thread.join()
        try:
            self.recorder.close()
        finally:
            self.check()
        logger.info("Episode I/O closed in %.3f s", time.perf_counter() - started)
