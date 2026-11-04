"""Keep readers and sidecar writes consistent across a dataset directory exchange."""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def dataset_access(root: Path, *, exclusive: bool = False):
    root = Path(root).resolve()
    # Share the persistent sibling lock used by transactional episode deletion.
    path = root.parent / f".{root.name}.delete.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        os.close(descriptor)
