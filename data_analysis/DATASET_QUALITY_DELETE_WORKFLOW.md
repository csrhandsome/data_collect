# Dataset Quality Check and Episode Deletion Workflow

This workflow is for reviewing collected LeRobot datasets and safely deleting bad episodes by `episode_index`.

## 1. Run Dataset Quality Check

From the project root:

```bash
uv run -m data_analysis.check_dataset_quality \
  --dataset-path data/dataset/<dataset_name> \
  --output-dir data_analysis/quality_reports/<dataset_name>
```

The report is saved to:

```text
data_analysis/quality_reports/<dataset_name>/quality_report.json
```

## 2. Review the Report

Ask AI or manually inspect `quality_report.json` to summarize suspicious episodes.

Common signals to check:

- missing or corrupted parquet files
- missing audio, audio metadata, or sync metadata
- audio duration or effective sample-rate mismatch
- input overflow warnings
- low voice activity ratio
- timestamp gaps or FPS issues

Do not delete only because one warning appears. Put episodes into three groups:

- delete: clearly broken data
- review: suspicious but needs manual confirmation
- keep: warning is expected or harmless

## 3. Preview Deletion

Run the deletion command without `--yes` first:

```bash
uv run episode-delete \
  --dataset data/dataset/<dataset_name> \
  --episode-index <episode_index>
```

Check that the preview shows the correct `Delete episode_index`, deleted frame count, and renumbering range.

## 4. Execute Deletion

Only after confirming the dry-run output:

```bash
uv run episode-delete \
  --dataset data/dataset/<dataset_name> \
  --episode-index <episode_index> \
  --yes
```

The command deletes the target episode and renumbers later episodes, including parquet files, audio sidecars, sync metadata, and LeRobot metadata files.

Deletion is prepared in a full sibling copy, including updates to VAD clip paths and segment IDs. The deletion engine checks the resulting Parquet indices and sidecar references before committing with Linux `renameat2(RENAME_EXCHANGE)`. Failures before commit leave the original dataset unchanged. The filesystem must support directory exchange and have space for a complete copy plus rewritten Parquet files. Normal completion removes the temporary tree; a forcibly killed process can leave `.<dataset_name>.delete-*` staging directories, which can be removed after confirming no deletion is running. Pause collection and other dataset writers while deleting; the deletion lock only coordinates deletion transactions.

## Python API

Use `EpisodeDeleter` for the same transactional deletion from Python:

```python
from control.collection import EpisodeDeleter

deleter = EpisodeDeleter("data/dataset/<dataset_name>")
plan = deleter.preview_episode(12)  # Returns a plan; changes no files.
print(plan.deleted_frames, plan.total_episodes_after)
result = deleter.delete_episode(12)  # Deletes and renumbers subsequent episodes.
```

`delete_episode(index, dry_run=True)` also previews deletion. Both methods return
an immutable `EpisodeDeletionResult` containing the dataset path, episode index,
deleted frame count, episode/frame totals before and after the operation, and
`dry_run`. Invalid indices (including an empty dataset) raise `ValueError`; a
missing dataset or Parquet file raises `FileNotFoundError`. Transaction errors
and interruptions propagate to the caller. The CLI alone translates exceptions
to exit codes (`0` success, `2` failure, `130` interruption).

`EpisodeDeleter` is the single business API. The `episode-delete` command and HTTP
backend use it. The former deletion scripts and standalone deletion functions
have been removed. Transaction implementation lives in the private
`control.collection._episode_deletion` module and preserves the deletion lock,
source change detection, VAD references and ASR annotations.

Close recording resources before deletion and reopen the dataset afterwards:
atomic directory replacement invalidates previously opened dataset handles and
cached episode metadata. The same `EpisodeDeleter` instance can be reused because
each call reads the dataset at its current path.

## Notes

- `episode_index` comes from `meta/episodes.jsonl`.
- Episode numbering starts from `0`.
- If `episode_index=12`, it usually corresponds to `episode_000012.parquet` and `audio/episode_000012.*`.
- Prefer deleting one confirmed bad episode at a time, then rerun the quality check.
