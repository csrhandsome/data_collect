"""Local v2/v3 metadata and per-episode tables, including shared v3 shards."""

from pathlib import Path

import pyarrow.parquet as pq

from replay.scripts._common import _episode_data_path, _load_metadata


def episode_rows(root: Path) -> list[dict]:
    return list(_load_metadata(root)[1].values())


def episode_paths(root: Path) -> dict[int, Path]:
    info, episodes = _load_metadata(root)
    return {index: _episode_data_path(root, info, row) for index, row in episodes.items()}


def episode_dataframe(path: Path, index: int, columns: list[str] | None = None):
    table = pq.read_table(path, columns=columns, filters=[("episode_index", "=", int(index))])
    return table.to_pandas()


def rewrite_v3_tasks(root: Path, prompts: dict[int, str]) -> dict:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from lerobot.datasets.io_utils import write_tasks

    task_names = list(dict.fromkeys(prompts.values()))
    indices = {name: index for index, name in enumerate(task_names)}
    mapping = {episode: indices[prompt] for episode, prompt in prompts.items()}
    counts = []
    for path in sorted(set(episode_paths(root).values())):
        table = pq.ParquetFile(path).read()
        values = [mapping[index] for index in table["episode_index"].to_pylist()]
        counts.extend(values)
        field = table.schema.field("task_index")
        table = table.set_column(
            table.schema.get_field_index("task_index"), field, pa.array(values, type=field.type)
        )
        pq.write_table(table, path)
    for path in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        table = pq.ParquetFile(path).read()
        rows = table.to_pylist()
        for row in rows:
            index = row["episode_index"]
            row["tasks"] = [prompts[index]]
            for stat, value in {
                "min": mapping[index],
                "max": mapping[index],
                "mean": float(mapping[index]),
                "std": 0.0,
                "count": row["length"],
            }.items():
                key = f"stats/task_index/{stat}"
                if key in row:
                    row[key] = [value]
            for quantile in ("q01", "q10", "q50", "q90", "q99"):
                key = f"stats/task_index/{quantile}"
                if key in row:
                    row[key] = [float(mapping[index])]
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    write_tasks(pd.DataFrame({"task_index": range(len(task_names))}, index=task_names), root)
    import json

    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = len(task_names)
    info_path.write_text(json.dumps(info, indent=2))
    stats_path = root / "meta/stats.json"
    if counts and stats_path.exists():
        stats = json.loads(stats_path.read_text())
        values = np.array(counts)
        stats["task_index"] = {
            "min": [int(values.min())],
            "max": [int(values.max())],
            "mean": [float(values.mean())],
            "std": [float(values.std())],
            "count": [len(values)],
        }
        for quantile, fraction in {
            "q01": 0.01,
            "q10": 0.1,
            "q50": 0.5,
            "q90": 0.9,
            "q99": 0.99,
        }.items():
            stats["task_index"][quantile] = [float(np.quantile(values, fraction))]
        stats_path.write_text(json.dumps(stats, indent=2))
    return {
        "num_tasks": len(task_names),
        "episode_task_indices": mapping,
        "tasks": [{"task_index": index, "task": name} for index, name in enumerate(task_names)],
    }
