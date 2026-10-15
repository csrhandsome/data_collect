"""File-level tests run before API integration, including genuine packed v3 fixtures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from replay.scripts._common import MissingDatasetFile
from replay.scripts.generate_demo import generate_demo
from replay.scripts.inspect_dataset import inspect_dataset
from replay.scripts.read_dataset import read_dataset
from replay.scripts.read_ee import read_ee
from replay.scripts.read_episode import read_episode
from replay.scripts.resolve_video import resolve_video


def _edit_info(root: Path, **changes: object) -> None:
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info.update(changes)
    path.write_text(json.dumps(info))


def _edit_episodes(root: Path, function) -> None:
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        function(row)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _edit_data(root: Path, function) -> None:
    path = root / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        function(row)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.mark.parametrize("version_index", [0, 1])
def test_complete_catalog_and_episode_blocks(demo_roots: list[Path], version_index: int) -> None:
    root = demo_roots[version_index]
    dataset = read_dataset(root)
    assert dataset["version"] == ["v2.1", "v3.0"][version_index]
    assert (dataset["total_episodes"], dataset["total_frames"], dataset["fps"]) == (3, 540, 30)
    assert [episode["episode_index"] for episode in dataset["episodes"]] == [0, 1, 2]
    features = {block["key"]: block for block in dataset["features"]}
    assert features["ee_pose"]["kind"] == "ee"
    assert features["exterior_image_1_left"]["kind"] == "video"
    assert features["observation.state"]["kind"] == "series"
    assert set(features) == set(json.loads((root / "meta/info.json").read_text())["features"])
    episode = read_episode(root, 1)
    assert episode["blocks"] == dataset["features"]
    assert episode["length"] == 180 and episode["duration_s"] == 6
    assert episode["tasks"] == ["Trace a placement arc"]


@pytest.mark.parametrize("version_index", [0, 1])
def test_collector_fields_match_standard_aliases(
    demo_roots: list[Path], version_index: int
) -> None:
    root = demo_roots[version_index]
    blocks = {block["key"]: block for block in read_dataset(root)["features"]}
    expected_shapes = {"joint_position": [7], "gripper_position": [1], "actions": [7]}
    for key, shape in expected_shapes.items():
        assert blocks[key]["shape"] == shape
        assert blocks[key]["dtype"] == "float32"
        assert blocks[key]["kind"] == "series"
    if version_index == 0:
        tables = [pq.read_table(path) for path in sorted((root / "data").glob("*/*.parquet"))]
        table = pa.concat_tables(tables)
    else:
        table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    rows = table.to_pylist()
    assert len(rows) == 540
    for row in rows:
        assert row["joint_position"] == row["observation.state"][:7]
        assert row["gripper_position"] == row["gripper"]
        assert row["actions"] == row["action"]
    stats = json.loads((root / "meta/stats.json").read_text())
    for key in expected_shapes:
        assert stats[key]["count"] == [540]
        values = np.asarray([row[key] for row in rows])
        np.testing.assert_allclose(stats[key]["mean"], np.atleast_1d(values.mean(axis=0)))


@pytest.mark.parametrize("episode_index", [0, 1, 2])
def test_both_formats_produce_same_isolated_episode(
    demo_roots: list[Path],
    episode_index: int,
) -> None:
    old = read_ee(demo_roots[0], episode_index)
    packed = read_ee(demo_roots[1], episode_index)
    assert old == packed
    assert packed["point_count"] == packed["total_points"] == 180
    assert packed["timestamps"][0] == 0
    assert packed["timestamps"][-1] == pytest.approx(179 / 30)
    assert packed["units"] == ["m", "m", "m", "rad", "rad", "rad"]
    raw = pq.read_table(demo_roots[1] / "data/chunk-000/file-000.parquet").to_pylist()
    expected = [row["ee_pose"] for row in raw if row["episode_index"] == episode_index]
    np.testing.assert_allclose(packed["values"], expected)
    assert packed["bounds"]["min"] == np.min(expected, axis=0).tolist()
    assert packed["bounds"]["max"] == np.max(expected, axis=0).tolist()


@pytest.mark.parametrize("max_points", [2, 3, 17, 180, 1000])
def test_downsampling_keeps_endpoints_and_complete_bounds(
    demo_roots: list[Path],
    max_points: int,
) -> None:
    full = read_ee(demo_roots[1], 1)
    sampled = read_ee(demo_roots[1], 1, max_points=max_points)
    assert sampled["point_count"] == min(max_points, 180)
    assert sampled["total_points"] == 180
    for key in ("values", "timestamps"):
        assert sampled[key][0] == full[key][0]
        assert sampled[key][-1] == full[key][-1]
    assert sampled["bounds"] == full["bounds"]
    assert np.all(np.diff(sampled["timestamps"]) > 0)


@pytest.mark.parametrize("camera", ["exterior_image_1_left", "wrist_image_left"])
def test_packed_video_offsets_and_actual_h264(demo_roots: list[Path], camera: str) -> None:
    videos = [resolve_video(demo_roots[1], episode, camera) for episode in range(3)]
    assert len({video["path"] for video in videos}) == 1
    assert [video["start_time_s"] for video in videos] == [0, 6, 12]
    assert [video["end_time_s"] for video in videos] == [6, 12, 18]
    assert all(video["duration_s"] == 6 and video["fps"] == 30 for video in videos)
    reader = imageio_ffmpeg.read_frames(str(videos[0]["path"]), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        assert metadata["codec"] == "h264"
        assert metadata["size"] == (640, 360)
        assert metadata["fps"] == 30
        assert metadata["duration"] == pytest.approx(18, abs=0.04)
        frames = []
        for frame_index, frame in enumerate(reader):
            if frame_index in (0, 180, 360):
                frames.append(frame)
        assert frame_index + 1 == 540
        assert len(frames) == 3 and len(set(frames)) == 3
    finally:
        reader.close()
    old = resolve_video(demo_roots[0], 2, camera)
    assert old["start_time_s"] == 0 and old["end_time_s"] == 6
    assert old["path"].name == "episode_000002.mp4"


def test_video_fps_is_separate_from_dataset_fps(editable_v3: Path) -> None:
    _edit_info(editable_v3, fps=100)
    video = resolve_video(editable_v3, 1, "wrist_image_left")
    assert video["fps"] == 30 and video["duration_s"] == 6


def test_custom_paths_and_nonzero_shard_indices(editable_v3: Path) -> None:
    root = editable_v3
    _edit_info(
        root,
        data_path="records/c{chunk_index}/p{file_index}.parquet",
        video_path="cameras/{video_key}/c{chunk_index}/v{file_index}.mp4",
    )
    _edit_episodes(
        root,
        lambda row: row.update(
            {
                "data/chunk_index": 4,
                "data/file_index": 9,
                "videos/wrist_image_left/chunk_index": 5,
                "videos/wrist_image_left/file_index": 11,
            }
        ),
    )
    data = root / "records/c4/p9.parquet"
    data.parent.mkdir(parents=True)
    (root / "data/chunk-000/file-000.parquet").rename(data)
    video = root / "cameras/wrist_image_left/c5/v11.mp4"
    video.parent.mkdir(parents=True)
    (root / "videos/wrist_image_left/chunk-000/file-000.mp4").rename(video)
    assert read_ee(root, 2)["total_points"] == 180
    assert resolve_video(root, 2, "wrist_image_left")["path"] == video.resolve()


def test_custom_metadata_path(editable_v3: Path) -> None:
    root = editable_v3
    _edit_info(root, episodes_path="catalog/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    destination = root / "catalog/chunk-000/file-000.parquet"
    destination.parent.mkdir(parents=True)
    (root / "meta/episodes/chunk-000/file-000.parquet").rename(destination)
    assert read_dataset(root)["total_episodes"] == 3


def test_timestamp_normalization_and_frame_order(editable_v3: Path) -> None:
    _edit_data(editable_v3, lambda row: row.update(timestamp=row["timestamp"] + 100))
    path = editable_v3 / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()[::-1]
    pq.write_table(pa.Table.from_pylist(rows), path)
    ee = read_ee(editable_v3, 2)
    assert ee["timestamps"][0] == 0
    assert ee["timestamps"][-1] == pytest.approx(179 / 30)


@pytest.mark.parametrize("bad_points", [0, 1, -5, 2.5, True])
def test_invalid_max_points(demo_roots: list[Path], bad_points) -> None:
    with pytest.raises(ValueError, match="max_points"):
        read_ee(demo_roots[1], 0, max_points=bad_points)


def test_unknown_episode_feature_and_unsupported_field(demo_roots: list[Path]) -> None:
    root = demo_roots[1]
    with pytest.raises(KeyError):
        read_episode(root, 99)
    with pytest.raises(KeyError):
        read_ee(root, 0, "unknown")
    with pytest.raises(ValueError, match="support EE"):
        read_ee(root, 0, "observation.state")
    with pytest.raises(ValueError, match="not a video"):
        resolve_video(root, 0, "ee_pose")


@pytest.mark.parametrize(
    "path_value",
    ["../outside.parquet", "/tmp/outside.parquet", "data/{chunk_index.__class__}/file.parquet"],
)
def test_rejects_unsafe_data_templates(editable_v3: Path, path_value: str) -> None:
    _edit_info(editable_v3, data_path=path_value)
    with pytest.raises(ValueError):
        read_ee(editable_v3, 0)


def test_rejects_unsafe_video_and_metadata_templates(editable_v3: Path) -> None:
    _edit_info(editable_v3, video_path="../escape/{video_key}.mp4")
    with pytest.raises(ValueError, match="unsafe"):
        resolve_video(editable_v3, 0, "wrist_image_left")
    _edit_info(editable_v3, episodes_path="../metadata/*.parquet")
    with pytest.raises(ValueError, match="unsafe"):
        read_dataset(editable_v3)


@pytest.mark.parametrize(
    "relative,entry",
    [
        ("meta/info.json", "dataset"),
        ("meta/episodes/chunk-000/file-000.parquet", "dataset"),
        ("data/chunk-000/file-000.parquet", "ee"),
        ("videos/wrist_image_left/chunk-000/file-000.mp4", "video"),
    ],
)
def test_rejects_symlink_escapes(editable_v3: Path, relative: str, entry: str) -> None:
    path = editable_v3 / relative
    external = editable_v3.parent / ("external" + path.suffix)
    path.rename(external)
    path.symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        if entry == "ee":
            read_ee(editable_v3, 0)
        elif entry == "video":
            resolve_video(editable_v3, 0, "wrist_image_left")
        else:
            read_dataset(editable_v3)


def test_missing_data_file_has_distinct_exception(editable_v3: Path) -> None:
    (editable_v3 / "data/chunk-000/file-000.parquet").unlink()
    with pytest.raises(MissingDatasetFile, match="not found"):
        read_ee(editable_v3, 0)


@pytest.mark.parametrize("mutation", ["length", "duplicate", "chunk", "chunk_path", "video_offset"])
def test_rejects_bad_episode_metadata(editable_v3: Path, mutation: str) -> None:
    if mutation == "length":
        _edit_episodes(editable_v3, lambda row: row.update(length=row["length"] - 1))
    elif mutation == "duplicate":
        _edit_episodes(editable_v3, lambda row: row.update(episode_index=0))
    elif mutation == "chunk":
        _edit_episodes(editable_v3, lambda row: row.update({"meta/episodes/chunk_index": -1}))
    elif mutation == "chunk_path":
        _edit_episodes(editable_v3, lambda row: row.update({"meta/episodes/chunk_index": 7}))
    else:
        _edit_episodes(
            editable_v3,
            lambda row: row.update(
                {
                    "videos/wrist_image_left/from_timestamp": 20,
                }
            ),
        )
    with pytest.raises(ValueError):
        resolve_video(editable_v3, 0, "wrist_image_left")


@pytest.mark.parametrize("mutation", ["nan", "shape", "timestamp", "frame_index", "index"])
def test_rejects_malformed_episode_rows(editable_v3: Path, mutation: str) -> None:
    if mutation == "nan":
        _edit_data(editable_v3, lambda row: row.update(ee_pose=[float("nan")] * 6))
    elif mutation == "shape":
        _edit_data(editable_v3, lambda row: row.update(ee_pose=[0.0] * 5))
    elif mutation == "timestamp":
        _edit_data(editable_v3, lambda row: row.update(timestamp=0.0))
    elif mutation == "frame_index":
        _edit_data(editable_v3, lambda row: row.update(frame_index=0))
    else:
        _edit_data(editable_v3, lambda row: row.update(index=0))
    with pytest.raises(ValueError):
        read_ee(editable_v3, 0)


def test_stats_are_coherent_and_inspector_exposes_offsets(demo_roots: list[Path]) -> None:
    for root in demo_roots:
        stats = json.loads((root / "meta/stats.json").read_text())
        values = np.concatenate([read_ee(root, episode)["values"] for episode in range(3)])
        np.testing.assert_allclose(stats["ee_pose"]["mean"], values.mean(axis=0))
        np.testing.assert_allclose(stats["ee_pose"]["std"], values.std(axis=0))
        assert stats["ee_pose"]["count"] == [540]
        assert stats["wrist_image_left"]["count"] == [540]
    inspected = inspect_dataset(demo_roots[1], 2)
    assert inspected["previews"]["wrist_image_left"]["start_time_s"] == 12
    assert len(inspected["previews"]["ee_pose"]["values"]) == 8


def test_v3_task_table_preserves_upstream_string_index(tmp_path: Path) -> None:
    from replay.scripts._demo import _tasks_table, _write_table

    path = tmp_path / "tasks.parquet"
    _write_table(path, _tasks_table())
    table = pq.read_table(path)
    metadata = json.loads(table.schema.metadata[b"pandas"])
    assert metadata["index_columns"] == ["__index_level_0__"]
    assert table.schema.field("__index_level_0__").type == pa.string()
    assert table.schema.field("task_index").type == pa.int64()
    assert table["task_index"].to_pylist() == [0, 1, 2]
    assert table["__index_level_0__"].to_pylist() == [
        "Approach the orange block",
        "Trace a placement arc",
        "Return above the workbench",
    ]


def test_v20_metadata_is_supported(copy_dataset) -> None:
    root = copy_dataset()
    _edit_info(root, codebase_version="v2.0")
    assert read_dataset(root)["version"] == "v2.0"
    assert read_ee(root, 2)["total_points"] == 180


def test_generator_refuses_unmarked_or_symlink_targets(tmp_path: Path) -> None:
    target = tmp_path / "demo_v21"
    target.mkdir()
    important = target / "important.txt"
    important.write_text("keep")
    with pytest.raises(ValueError, match="unmarked"):
        generate_demo(tmp_path)
    assert important.read_text() == "keep"
    shutil.rmtree(target)
    external = tmp_path / "other"
    external.mkdir()
    target.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        generate_demo(tmp_path)
    assert external.is_dir()
