"""Recording adapters for the pinned LeRobot 0.6.1 writer.

Keep the upstream v3 shard/index rules, but avoid embedding image bytes into
numeric Parquet rows when every camera is already stored in an MP4.
"""

import logging
import queue
import time

import av
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.dataset_writer import DatasetWriter
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import get_file_size_in_mb
from lerobot.datasets.utils import update_chunk_file_indices
from lerobot.datasets.video_utils import StreamingVideoEncoder

logger = logging.getLogger(__name__)


class LosslessStreamingEncoder(StreamingVideoEncoder):
    """Back pressure is handled by the recording worker, never by dropping frames."""

    def feed_frame(self, video_key, image):
        if not self._episode_active:
            raise RuntimeError("No active video episode")
        image = image.copy()
        while self._threads[video_key].is_alive():
            try:
                self._frame_queues[video_key].put(image, timeout=0.1)
                return
            except queue.Full:
                continue
        raise RuntimeError(f"Video encoder failed: {video_key}")

    def finish_episode(self):
        # Upstream logs timeouts/missing results and may return incomplete files.
        # Check liveness while enqueueing sentinels too: a failed encoder with a
        # full queue must not hang shutdown forever.
        started = time.perf_counter()
        deadline = time.monotonic() + 120
        for key, frames in self._frame_queues.items():
            while True:
                if not self._threads[key].is_alive() or time.monotonic() >= deadline:
                    raise RuntimeError(f"Video encoder failed before flush: {key}")
                try:
                    frames.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
        result = {}
        for key, thread in self._threads.items():
            thread.join(timeout=max(0, deadline - time.monotonic()))
            if thread.is_alive():
                raise RuntimeError(f"Video flush timed out: {key}")
            try:
                status, stats = self._result_queues[key].get_nowait()
            except queue.Empty as exc:
                raise RuntimeError(f"Video encoder returned no result: {key}") from exc
            if status != "ok":
                raise RuntimeError(f"Video encoder failed: {key}: {stats}")
            result[key] = self._video_paths[key], stats
        self._cleanup()
        self._episode_active = False
        logger.info("Video flush finished in %.3f s", time.perf_counter() - started)
        return result


class RecordingDatasetWriter(DatasetWriter):
    def _save_episode_video(self, video_key, episode_index, temp_path=None):
        if temp_path is not None:
            with av.open(str(temp_path)) as container:
                count = container.streams.video[0].frames
            expected = len(self.episode_buffer["frame_index"])
            if count != expected:
                raise RuntimeError(f"Video frame mismatch: {video_key}: {count} != {expected}")
        return super()._save_episode_video(video_key, episode_index, temp_path)

    def _save_episode_data(self, episode_buffer):
        if self._meta.image_keys:
            return super()._save_episode_data(episode_buffer)
        started = time.perf_counter()
        hf_features = get_hf_features_from_features(self._meta.features)
        ep_dict = {key: episode_buffer[key] for key in hf_features}
        # Preserve Hugging Face feature metadata and fixed-size array types.
        # encode_batch handles nested numeric shapes just as Dataset.from_dict.
        table = pa.Table.from_pydict(hf_features.encode_batch(ep_dict), schema=hf_features.arrow_schema)
        length = table.num_rows

        # Same rollover rules as DatasetWriter._save_episode_data in 0.6.1.
        if self._latest_episode is None:
            chunk, file, first = 0, 0, 0
            self._current_file_start_frame = 0
            if self._meta.episodes is not None and len(self._meta.episodes):
                latest = self._meta.episodes[-1]
                first = latest["dataset_to_index"]
                chunk, file = update_chunk_file_indices(
                    latest["data/chunk_index"], latest["data/file_index"], self._meta.chunks_size
                )
                self._current_file_start_frame = first
        else:
            latest = self._latest_episode
            chunk, file = latest["data/chunk_index"], latest["data/file_index"]
            first = int(latest["index"][-1]) + 1
            path = self._root / self._meta.data_path.format(chunk_index=chunk, file_index=file)
            size = get_file_size_in_mb(path)
            count = first - self._current_file_start_frame
            if size + (size / count if count else 0) * length >= self._meta.data_files_size_in_mb:
                chunk, file = update_chunk_file_indices(chunk, file, self._meta.chunks_size)
                self.close_writer()
                self._current_file_start_frame = first
        ep_dict.update({"data/chunk_index": chunk, "data/file_index": file})
        path = self._root / self._meta.data_path.format(chunk_index=chunk, file_index=file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._pq_writer is None:
            self._pq_writer = pq.ParquetWriter(
                path, schema=table.schema, compression="snappy", use_dictionary=True
            )
        self._pq_writer.write_table(table)
        metadata = {
            "data/chunk_index": chunk, "data/file_index": file,
            "dataset_from_index": first, "dataset_to_index": first + length,
        }
        self._latest_episode = {**ep_dict, **metadata}
        self._recorded_frames += length
        logger.info("Parquet: %s rows written in %.3f s", length, time.perf_counter() - started)
        return metadata


def configure_recording_writer(dataset):
    """Adapt this instance only; do not patch LeRobot globally or fork live threads."""
    writer = dataset.writer
    writer.__class__ = RecordingDatasetWriter
    if writer._streaming_encoder is not None:
        writer._streaming_encoder.__class__ = LosslessStreamingEncoder
    return dataset
