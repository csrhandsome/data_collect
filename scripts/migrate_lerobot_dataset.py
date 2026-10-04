"""Convert a local v2.1 dataset to a new v3.0 directory, preserving sidecars."""

import argparse
import shutil
import tempfile
from pathlib import Path


def migrate_dataset(source: Path, output: Path) -> Path:
    from lerobot.datasets.utils import (
        DEFAULT_DATA_FILE_SIZE_IN_MB,
        DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    )
    from lerobot.scripts.convert_dataset_v21_to_v30 import (
        convert_data,
        convert_episodes_metadata,
        convert_info,
        convert_tasks,
        convert_videos,
        validate_local_dataset_version,
    )

    from control.collection._episode_deletion import _patch_audio_json, _patch_sync_json
    from replay.scripts._common import _load_metadata

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Choose a new output directory outside the source dataset")
    validate_local_dataset_version(source)
    _, episodes = _load_metadata(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lerobot-migrate-", dir=output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        convert_info(source, staged, DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        convert_tasks(source, staged)
        data = convert_data(source, staged, DEFAULT_DATA_FILE_SIZE_IN_MB)
        videos = convert_videos(source, staged, DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        convert_episodes_metadata(source, staged, data, videos)
        for path in source.iterdir():
            if path.name in {"meta", "data", "videos", "images", ".cache", ".replay-cache"}:
                continue
            if path.is_dir():
                shutil.copytree(path, staged / path.name)
            else:
                shutil.copy2(path, staged / path.name)
        for index in episodes:
            _patch_audio_json(staged / "audio" / f"episode_{index:06d}.audio.json", output, index)
            for folder in ("audio", ""):
                _patch_sync_json(
                    staged / folder / f"episode_{index:06d}.sync.json", index, index, output
                )
        _load_metadata(staged)
        staged.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="将旧 v2.1 数据复制转换成 v3.0，保留音频和同步记录"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(migrate_dataset(args.input, args.output))


if __name__ == "__main__":
    main()
