"""Use upstream v3 merging while retaining collection audio and traces."""

import shutil
import tempfile
from pathlib import Path


def merge_v3(sources: list[Path], output: Path, overwrite: bool) -> Path:
    from control.util.hub_compat import allow_hub1_for_transformers

    allow_hub1_for_transformers()
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import merge_datasets

    from control.collection._episode_deletion import (
        _episode_audio_path_map,
        _exchange_directories,
        _patch_audio_json,
        _patch_sync_json,
        _source_manifest,
    )
    from data_analysis.dataset_io import episode_rows
    from replay.scripts._common import _load_metadata

    output = output.resolve()
    if any(
        output == source.resolve() or output.is_relative_to(source.resolve()) for source in sources
    ):
        raise ValueError("Output must be outside all source datasets")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    for source in sources:
        _source_manifest(source)
        _load_metadata(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    datasets = [LeRobotDataset(path.name, root=path, video_backend="pyav") for path in sources]
    with tempfile.TemporaryDirectory(prefix=".lerobot-merge-", dir=output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        merged = merge_datasets(datasets, output.name, output_dir=staged)
        merged.finalize()
        # Upstream aggregation retains source metadata shard indices in copied rows.
        # Rebind these references to the actual destination shard before local validation.
        import pyarrow as pa
        import pyarrow.parquet as pq

        for path in (staged / "meta/episodes").glob("chunk-*/file-*.parquet"):
            table = pq.ParquetFile(path).read()
            for key, value in {
                "meta/episodes/chunk_index": int(path.parent.name.split("-")[-1]),
                "meta/episodes/file_index": int(path.stem.split("-")[-1]),
            }.items():
                field = table.schema.field(key)
                table = table.set_column(
                    table.schema.get_field_index(key),
                    field,
                    pa.array([value] * table.num_rows, type=field.type),
                )
            pq.write_table(table, path)
        new_index = 0
        for source in sources:
            for row in episode_rows(source):
                old = row["episode_index"]
                for src, target in _episode_audio_path_map(source, old, new_index).items():
                    if not src.exists():
                        continue
                    target = staged / target.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if src.is_dir():
                        shutil.copytree(src, target)
                    else:
                        shutil.copy2(src, target)
                _patch_audio_json(
                    staged / "audio" / f"episode_{new_index:06d}.audio.json", output, new_index
                )
                for folder in ("audio", ""):
                    _patch_sync_json(
                        staged / folder / f"episode_{new_index:06d}.sync.json",
                        new_index,
                        old,
                        output,
                    )
                new_index += 1
        _load_metadata(staged)
        if output.exists():
            _exchange_directories(output, staged)
        else:
            staged.rename(output)
    return output
