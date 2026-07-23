from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data_analysis.delete_latest_episode import delete_latest_episode


class DeleteLatestEpisodeTest(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        if text:
            text += "\n"
        path.write_text(text, encoding="utf-8")

    def test_delete_latest_episode_removes_audio_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            meta_dir = dataset_dir / "meta"
            chunk_dir = dataset_dir / "data" / "chunk-000"
            audio_dir = dataset_dir / "audio"

            meta_dir.mkdir(parents=True)
            chunk_dir.mkdir(parents=True)
            audio_dir.mkdir(parents=True)

            self._write_json(
                meta_dir / "info.json",
                {
                    "chunks_size": 1000,
                    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                    "total_episodes": 2,
                    "total_frames": 12,
                    "total_chunks": 1,
                    "splits": {"train": "0:2"},
                },
            )
            self._write_jsonl(
                meta_dir / "episodes.jsonl",
                [
                    {"episode_index": 0, "length": 5},
                    {"episode_index": 1, "length": 7},
                ],
            )
            self._write_jsonl(
                meta_dir / "episodes_stats.jsonl",
                [
                    {"episode_index": 0, "mean": 0.0},
                    {"episode_index": 1, "mean": 1.0},
                ],
            )

            older_episode = chunk_dir / "episode_000000.parquet"
            latest_episode = chunk_dir / "episode_000001.parquet"
            older_episode.write_bytes(b"older")
            latest_episode.write_bytes(b"latest")

            older_audio = audio_dir / "episode_000000.wav"
            latest_wav = audio_dir / "episode_000001.wav"
            latest_meta = audio_dir / "episode_000001.audio.json"
            latest_sync = audio_dir / "episode_000001.sync.json"
            older_audio.write_bytes(b"older audio")
            latest_wav.write_bytes(b"latest audio")
            latest_meta.write_text("{}", encoding="utf-8")
            latest_sync.write_text("{}", encoding="utf-8")

            exit_code = delete_latest_episode(dataset_dir)

            self.assertEqual(exit_code, 0)
            self.assertTrue(older_episode.exists())
            self.assertFalse(latest_episode.exists())
            self.assertTrue(older_audio.exists())
            self.assertFalse(latest_wav.exists())
            self.assertFalse(latest_meta.exists())
            self.assertFalse(latest_sync.exists())

            episodes_rows = [
                json.loads(line)
                for line in (meta_dir / "episodes.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            stats_rows = [
                json.loads(line)
                for line in (meta_dir / "episodes_stats.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))

            self.assertEqual(episodes_rows, [{"episode_index": 0, "length": 5}])
            self.assertEqual(stats_rows, [{"episode_index": 0, "mean": 0.0}])
            self.assertEqual(info["total_episodes"], 1)
            self.assertEqual(info["total_frames"], 5)
            self.assertEqual(info["total_chunks"], 1)
            self.assertEqual(info["splits"]["train"], "0:1")


if __name__ == "__main__":
    unittest.main()
