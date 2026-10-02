"""Resolve dataset IDs only within the configured local data root."""

from pathlib import Path

from replay.backend.errors import ReplayError


class DatasetRegistry:
    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()

    def entries(self) -> dict[str, Path]:
        root = self.data_root
        try:
            if (root / "meta" / "info.json").is_file():
                return {root.name: root}
            if not root.is_dir():
                return {}
            candidates = sorted(root.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise ReplayError(422, "Dataset directory could not be read.") from exc
        entries = {}
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except (OSError, RuntimeError):
                continue
            if not resolved.is_relative_to(root) or not resolved.is_dir():
                continue
            if (resolved / "meta" / "info.json").is_file():
                entries[candidate.name] = resolved
        return entries

    def get(self, dataset_id: str) -> Path:
        root = self.entries().get(dataset_id)
        if root is None:
            raise ReplayError(404, "Dataset was not found.")
        return root
