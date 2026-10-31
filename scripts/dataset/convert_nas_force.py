#!/usr/bin/env python3
"""Convert NAS tactile datasets into LeRobot v3 MP4 folders, then delete the raw files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data_analysis.data_clean import ALIGNERS  # noqa: E402

NAS = Path("/mnt/nas")
LOCAL_VITAMIN = Path("/home/mirnagroup/tactile-stage/files/ViTaMIn/ViTaMIn.tar")
JOBS = (
    {
        "name": "tactile_video_pretrain_nc",
        "source": NAS / "tactile-video-pretrain-nc",
        "output": NAS / "tactile-video-pretrain-nc" / "lerobot_v3",
        "keep": {"lerobot_v3"},
    },
    {
        "name": "freetacman",
        "source": NAS / "FreeTacMan",
        "output": NAS / "FreeTacMan" / "lerobot_v3",
        "keep": {"lerobot_v3"},
    },
    {
        "name": "haptile",
        "source": NAS / "HapTile",
        "output": NAS / "HapTile" / "lerobot_v3",
        "keep": {"lerobot_v3"},
    },
    {
        "name": "feelsight",
        "source": NAS / "FeelSight",
        "output": NAS / "FeelSight" / "lerobot_v3",
        "keep": {"lerobot_v3"},
    },
    {
        "name": "vitamin",
        "source": LOCAL_VITAMIN if LOCAL_VITAMIN.is_file() else NAS / "ViTaMIn",
        "output": NAS / "ViTaMIn" / "lerobot_v3",
        "delete_paths": [
            NAS / "ViTaMIn" / "ViTaMIn.tar",
            NAS / "ViTaMIn" / "ViTaMIn.tar.index.json",
            LOCAL_VITAMIN,
            LOCAL_VITAMIN.with_name(LOCAL_VITAMIN.name + ".index.json"),
        ],
    },
    {
        "name": "wirefishing",
        "source": NAS / "WireFishing-M",
        "output": NAS / "WireFishing-M" / "lerobot_v3",
        "keep": {"lerobot_v3"},
    },
)


def log(message: str) -> None:
    print(message, flush=True)


def verify(output: Path) -> None:
    info = json.loads((output / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError("info.json is not v3.0")
    episodes_path = output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not episodes_path.is_file():
        raise RuntimeError("missing episode metadata")
    import pyarrow.parquet as pq

    table = pq.read_table(episodes_path)
    rows = table.to_pylist()
    if len(rows) != info["total_episodes"] or not rows:
        raise RuntimeError(f"episode count {len(rows)} != {info['total_episodes']}")
    if rows[-1]["dataset_to_index"] != info["total_frames"]:
        raise RuntimeError("frame count does not match the last episode")
    video_keys = [key for key, spec in info["features"].items() if spec.get("dtype") == "video"]
    if not any(key.startswith("observation.gripper_image") for key in video_keys):
        raise RuntimeError(f"no gripper video in {video_keys}")
    sample_indexes = sorted({0, len(rows) // 2, len(rows) - 1})
    for index in sample_indexes:
        row = rows[index]
        data = output / info["data_path"].format(
            chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
        )
        frame_table = pq.read_table(data)
        if frame_table.num_rows != row["length"]:
            raise RuntimeError(f"episode {index} parquet length {frame_table.num_rows} != {row['length']}")
        for key in video_keys:
            video = output / info["video_path"].format(
                video_key=key,
                chunk_index=row[f"videos/{key}/chunk_index"],
                file_index=row[f"videos/{key}/file_index"],
            )
            if not video.is_file() or video.stat().st_size < 32:
                raise RuntimeError(f"missing video {video}")
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode != 0 or not (probe.stdout or "").strip():
                raise RuntimeError(f"ffprobe failed for {video}: {probe.stderr[-200:]}")
    (output / "verified.json").write_text(json.dumps({"episodes": len(rows), "frames": info["total_frames"]}) + "\n")
    log(f"[verify] {output} episodes={len(rows)} frames={info['total_frames']}")


def delete_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    log(f"[delete] {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def delete_raw(job: dict) -> None:
    for path in job.get("delete_paths", []):
        delete_path(Path(path))
    source = job["source"]
    keep = set(job.get("keep", set()))
    if source.is_dir():
        for child in source.iterdir():
            if child.name in keep or child.name.startswith("."):
                continue
            delete_path(child)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[job["name"] for job in JOBS])
    args = parser.parse_args()
    selected = [job for job in JOBS if args.dataset in {None, job["name"]}]
    failed = []
    for job in selected:
        name = job["name"]
        source = job["source"]
        output = job["output"]
        log(f"[{name}] {source} -> {output}")
        if (output / "verified.json").is_file():
            log(f"[{name}] already verified")
            delete_raw(job)
            continue
        if not source.exists():
            failed.append(name)
            log(f"[{name}] source missing")
            continue
        try:
            ALIGNERS[name]().convert_all(source, output)
            verify(output)
            delete_raw(job)
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            log(f"[{name}] failed: {exc}")
            traceback.print_exc()
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
