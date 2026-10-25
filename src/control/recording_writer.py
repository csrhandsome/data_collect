"""Keep image encoding and LeRobot frame insertion off the 100 Hz control loop."""

from __future__ import annotations

import queue
import threading
from typing import Any


class FreshCameraPair:
    """Accept each pair only after both physical camera frames have advanced."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last = (0, 0)

    def accept(self, external_capture_ns: int, wrist_capture_ns: int) -> bool:
        current = (int(external_capture_ns), int(wrist_capture_ns))
        if any(value <= old for value, old in zip(current, self._last)):
            return False
        self._last = current
        return True


class AsyncDatasetFrames:
    def __init__(self, dataset: Any, max_pending: int = 120) -> None:
        self._dataset = dataset
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=max_pending)
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="lerobot-frames", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            kind, payload = self._queue.get()
            try:
                if kind == "stop":
                    return
                if kind == "frame" and self._error is None:
                    self._dataset.add_frame(payload)
                if kind == "barrier":
                    payload.set()
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
                if kind == "barrier":
                    payload.set()
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("LeRobot frame writer failed") from self._error

    def submit(self, frame: dict[str, Any]) -> None:
        self._raise_if_failed()
        try:
            self._queue.put_nowait(("frame", frame))
        except queue.Full as exc:
            raise RuntimeError(
                "LeRobot writer fell more than 4 seconds behind"
            ) from exc

    def drain(self) -> None:
        event = threading.Event()
        self._queue.put(("barrier", event))
        event.wait()
        self._raise_if_failed()

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(("stop", None))
            self._thread.join()
        self._raise_if_failed()
