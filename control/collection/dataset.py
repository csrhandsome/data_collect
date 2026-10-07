"""One schema builder, with recording space independent of EE control."""

import json
from pathlib import Path

from control.collection.schema import features
from control.util.lerobot_util import _resume_existing_dataset_for_recording


def open_dataset(config):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = config["dataset"]
    repo = f"{cfg['repo_id']}_{cfg['date']}"
    # Keep the Hub repo ID independent of the local, flat dataset directory.
    root = Path(cfg.get("root", "data/dataset")) / repo.rsplit("/", 1)[-1]
    fps = float(config["camera"].get("fps", 30))
    schema = features(
        int(config["camera"].get("image_hw", 224)),
        cfg["action_space"],
        config.get("tactile", {}).get("enabled", False),
        config.get("observation"),
        config.get("action"),
    )
    if root.exists():
        from lerobot.utils.constants import DEFAULT_FEATURES

        info = json.loads((root / "meta/info.json").read_text())
        expected = {key: (value["dtype"], list(value["shape"])) for key, value in schema.items()}
        actual = {
            key: (value["dtype"], value["shape"])
            for key, value in info["features"].items()
            if key not in DEFAULT_FEATURES
        }
        if info["fps"] != fps or expected != actual:
            raise ValueError(
                "Dataset schema/fps mismatch; select a new dataset.date before recording"
            )
        dataset = _resume_existing_dataset_for_recording(repo, root)
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo,
            root=root,
            fps=fps,
            robot_type="panda",
            features=schema,
            image_writer_threads=4,
            image_writer_processes=0,
            metadata_buffer_size=1,
        )
    return dataset, root
