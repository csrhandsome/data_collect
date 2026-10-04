"""Keep metadata shards readable after resuming converted historical datasets."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def normalize_episode_metadata(root: Path) -> None:
    paths = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if len(paths) < 2:
        return
    schemas = [pq.read_schema(path) for path in paths]
    if all(schema.equals(schemas[0], check_metadata=False) for schema in schemas[1:]):
        return
    # Legacy v2 stats have no quantiles. New writers add them, but HF's parquet
    # loader requires identical columns and types in every metadata shard.
    common = set.intersection(*(set(schema.names) for schema in schemas))
    missing = set.union(*(set(schema.names) for schema in schemas)) - common
    if any(not key.startswith("stats/") for key in missing):
        raise ValueError("Inconsistent episode metadata columns; restore metadata before recording")
    schemas = [pa.schema([field for field in schema if field.name in common]) for schema in schemas]
    unified = pa.unify_schemas(schemas, promote_options="permissive")
    for path in paths:
        table = pq.ParquetFile(path).read(columns=unified.names).cast(unified)
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary)
        temporary.replace(path)
