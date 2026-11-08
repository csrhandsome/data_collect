"""Keep readers and sidecar writes consistent across a dataset directory exchange."""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Literal


def dataset_lock_path(root: Path, *, kind: Literal["access", "sync"] = "access") -> Path:
    """Group persistent locks outside datasets, including historical flat collections."""
    root = Path(root).resolve()
    source = root.parent.name
    if source in {"collection", "inference"}:
        data_root = root.parent.parent
    else:
        data_root, source = root.parent, "collection"
    directory = data_root / "lock" / source / root.name
    if not directory.resolve().is_relative_to(data_root):
        raise ValueError("Lock directory escapes dataset storage root")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{kind}.lock"


@contextmanager
def dataset_access(root: Path, *, exclusive: bool = False):
    # Readers, deletion and compaction must share one persistent lock inode.
    path = dataset_lock_path(root)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        os.close(descriptor)
