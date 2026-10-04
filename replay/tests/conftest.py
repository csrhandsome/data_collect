"""Shared temporary datasets for reader, API, and operation tests."""

import shutil
from pathlib import Path

import pytest

from replay.scripts.generate_demo import generate_demo


@pytest.fixture(scope="session")
def demo_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Encode the matching v2.1/v3.0 datasets once per test run."""
    root = tmp_path_factory.mktemp("replay-demo-source")
    generate_demo(root)
    return root


@pytest.fixture(scope="module")
def demo_root(demo_source: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Give each module its own copy, including any generated caches."""
    root = tmp_path_factory.mktemp("replay-demo")
    shutil.copytree(demo_source, root, dirs_exist_ok=True)
    return root


@pytest.fixture(scope="module")
def demo_roots(demo_root: Path) -> list[Path]:
    return [demo_root / "demo_v21", demo_root / "demo_v30"]


@pytest.fixture
def copy_dataset(demo_root: Path, tmp_path: Path):
    """Copy a demo before a test deletes or changes its contents."""

    def copy(name: str = "demo_v21", destination: Path | None = None) -> Path:
        target = (tmp_path if destination is None else destination) / name
        shutil.copytree(demo_root / name, target)
        return target

    return copy


@pytest.fixture
def editable_v3(copy_dataset) -> Path:
    return copy_dataset("demo_v30")
