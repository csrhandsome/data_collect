# Dataset Quality Check and Episode Deletion Workflow

This workflow is for reviewing collected LeRobot datasets and safely deleting bad episodes by `episode_index`.

## 1. Run Dataset Quality Check

From the project root:

```bash
uv run -m data_analysis.check_dataset_quality \
  --dataset-path data/openpi/<dataset_name> \
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

Always run the deletion script without `--yes` first:

```bash
uv run data_analysis/delete_lastest_episode_by_id.py \
  --dataset data/openpi/<dataset_name> \
  --episode-index <episode_index>
```

Check that the preview shows the correct `Delete episode_index`, deleted frame count, and renumbering range.

## 4. Execute Deletion

Only after confirming the dry-run output:

```bash
uv run data_analysis/delete_lastest_episode_by_id.py \
  --dataset data/openpi/<dataset_name> \
  --episode-index <episode_index> \
  --yes
```

The script deletes the target episode and renumbers later episodes, including parquet files, audio sidecars, sync metadata, and LeRobot metadata files.

## Notes

- `episode_index` comes from `meta/episodes.jsonl`.
- Episode numbering starts from `0`.
- If `episode_index=12`, it usually corresponds to `episode_000012.parquet` and `audio/episode_000012.*`.
- Prefer deleting one confirmed bad episode at a time, then rerun the quality check.
