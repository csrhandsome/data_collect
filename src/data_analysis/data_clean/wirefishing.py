"""WireFishing-M: one 7z per trial group, with global, local, and left tactile JPG sequences."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from data_analysis.data_clean.base import BaseAligner
from data_analysis.data_clean.lerobot_v3 import GRIPPER_IMAGE_LEFT, AlignedEpisode

_VIEWS = {
    "globel_view": ("global_image_", "observation.exterior_image"),
    "local_view": ("color_image_", "observation.wrist_image_left"),
    "tactile_view": ("left_image_", GRIPPER_IMAGE_LEFT),
}


class WireFishingAligner(BaseAligner):
    name = "wirefishing"
    robot_type = "wirefishing"

    def episodes(self, source: Path) -> list[str]:
        archives = sorted(source.rglob("*.7z"))
        if not archives:
            raise FileNotFoundError(f"WireFishing-M has no 7z archives in {source}")
        return [str(path.relative_to(source)) for path in archives]

    def load(self, source: Path, episode_id: str, work: Path) -> AlignedEpisode:
        raise RuntimeError("WireFishing-M is converted archive by archive")

    def iter_episodes(self, source: Path, work: Path, done: set[str] | None = None):
        finished = done or set()
        import py7zr

        for archive in sorted(source.rglob("*.7z")):
            archive_id = str(archive.relative_to(source))
            extract_root = work / "extract"
            if extract_root.exists():
                shutil.rmtree(extract_root)
            extract_root.mkdir(parents=True)
            print(f"[wirefishing] extract {archive_id}", flush=True)
            with py7zr.SevenZipFile(archive, "r") as handle:
                handle.extract(path=extract_root)
            trials = sorted(path for path in extract_root.rglob("trial_*") if path.is_dir())
            for trial in trials:
                episode_id = f"{archive_id}::{trial.name}"
                if episode_id in finished:
                    continue
                yield self._trial(trial, episode_id, work / "encoded")
            shutil.rmtree(extract_root, ignore_errors=True)

    def _trial(self, trial: Path, episode_id: str, work: Path) -> AlignedEpisode:
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        videos = {}
        counts = []
        for folder_name, (prefix, feature) in _VIEWS.items():
            frames = _indexed_frames(trial / folder_name, prefix)
            if len(frames) < 2:
                raise RuntimeError(f"{episode_id} is missing {folder_name}")
            sequence = work / feature.replace(".", "_")
            sequence.mkdir()
            for index, source in enumerate(frames):
                destination = sequence / f"{index:06d}.jpg"
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copyfile(source, destination)
            video = work / f"{feature.replace('.', '_')}.mp4"
            _jpg_to_mp4(sequence, video, fps=10)
            videos[feature] = video
            counts.append(len(frames))
        length = min(counts)
        states = _states(trial / "robot_data", length)
        joints = states[:, :7].astype(np.float32)
        gripper = np.zeros((length, 1), dtype=np.float32)
        episode = AlignedEpisode(
            key=episode_id,
            task=trial.name.replace("_", " "),
            fps=10.0,
            videos=videos,
            columns={
                "observation.joint_position": joints,
                "observation.gripper_position": gripper,
                "action.joint_position": np.vstack([joints[1:], joints[-1:]]),
                "action.gripper_position": np.vstack([gripper[1:], gripper[-1:]]),
            },
        )
        return episode


def _indexed_frames(folder: Path, prefix: str) -> list[Path]:
    frames = []
    for path in folder.glob(f"{prefix}*.jpg"):
        suffix = path.stem[len(prefix) :]
        if suffix.isdigit():
            frames.append((int(suffix), path))
    frames.sort()
    return [path for _, path in frames]


def _states(folder: Path, length: int) -> np.ndarray:
    rows = []
    for index in range(length):
        path = folder / f"combined_data_{index}.npy"
        if not path.is_file():
            rows.append(np.zeros(7, dtype=np.float64))
            continue
        values = np.load(path, allow_pickle=True).reshape(-1)
        padded = np.zeros(7, dtype=np.float64)
        padded[: min(7, values.size)] = values[:7]
        rows.append(padded)
    return np.stack(rows)


def _jpg_to_mp4(folder: Path, output: Path, fps: int) -> None:
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(folder / "%06d.jpg"),
            "-frames:v",
            str(len(list(folder.glob("*.jpg")))),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
