"""Grouped locks still coordinate readers, deletion and sidecar writers across processes."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from replay.scripts.dataset_lock import dataset_access, dataset_lock_path
from replay.scripts.read_sidecars import read_sync


def run_python(code, *args):
    return subprocess.run(
        [sys.executable, "-c", code, *map(str, args)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )


@pytest.mark.parametrize("source", ["", "collection", "inference"])
def test_reader_blocks_deletion_in_another_process(tmp_path, source):
    root = tmp_path / source / "demo"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}")
    with dataset_access(root):
        result = run_python(
            "import sys\n"
            "from control.collection.deletion import EpisodeDeleter\n"
            "try:\n"
            "    EpisodeDeleter(sys.argv[1]).delete_episode(0)\n"
            "except RuntimeError as exc:\n"
            "    assert '删除任务' in str(exc), str(exc)\n"
            "else:\n"
            "    raise AssertionError('Deletion bypassed the reader lock')\n",
            root,
        )
        assert result.returncode == 0
    # Closing the reader releases the lock, while the file remains reusable.
    run_python(
        "import sys\nfrom pathlib import Path\n"
        "from replay.scripts.dataset_lock import dataset_access\n"
        "with dataset_access(Path(sys.argv[1]), exclusive=True):\n    pass\n",
        root,
    )
    expected = tmp_path / "lock" / (source or "collection") / "demo/access.lock"
    assert expected.is_file()
    assert not list(root.rglob("*.lock"))
    assert not list(root.parent.glob(".*.delete.lock"))


@pytest.mark.parametrize("source", ["", "collection", "inference"])
def test_concurrent_sidecar_updates_use_grouped_locks(tmp_path, source):
    root = tmp_path / source / "demo"
    root.mkdir(parents=True)

    def update(index):
        run_python(
            "import sys\nfrom replay.scripts.read_sidecars import update_sync\n"
            "for i in range(20):\n"
            "    update_sync(sys.argv[1], 0, {sys.argv[2]: i})\n",
            root,
            f"writer_{index}",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(update, range(4)))
    assert read_sync(root, 0) == {"episode_index": 0, **{f"writer_{i}": 19 for i in range(4)}}
    directory = tmp_path / "lock" / (source or "collection") / "demo"
    assert {path.name for path in directory.iterdir()} == {"access.lock", "sync.lock"}
    assert not list(root.rglob("*.lock"))


def test_same_named_collection_and_inference_have_independent_locks(tmp_path):
    collection = tmp_path / "collection/demo"
    inference = tmp_path / "inference/demo"
    with dataset_access(collection, exclusive=True):
        run_python(
            "import sys\nfrom pathlib import Path\n"
            "from replay.scripts.dataset_lock import dataset_access\n"
            "with dataset_access(Path(sys.argv[1]), exclusive=True):\n    pass\n",
            inference,
        )


def test_lock_directory_cannot_escape_storage_root(tmp_path):
    data_root, outside = tmp_path / "datasets", tmp_path / "outside"
    data_root.mkdir()
    outside.mkdir()
    (data_root / "lock").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Lock directory escapes"):
        dataset_lock_path(data_root / "collection/demo")
    assert not list(outside.iterdir())
