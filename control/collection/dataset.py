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
    if not fps.is_integer():
        raise ValueError("Video recording requires an integer camera.fps with this LeRobot version")
    fps = int(fps)
    schema = features(
        int(config["camera"].get("image_hw", 224)),
        cfg["action_space"],
        config.get("tactile", {}).get("enabled", False),
        config.get("observation"),
        config.get("action"),
    )
    from lerobot.configs import RGBEncoderConfig

    encoder = RGBEncoderConfig(
        **cfg.get(
            "video_encoder",
            {
                "vcodec": "h264",
                "crf": 18,
                "g": 2,
                "preset": "fast",
            },
        )
    )
    if root.exists():
        from lerobot.utils.constants import DEFAULT_FEATURES

        from control.util.lerobot_util import _ensure_local_recording_metadata

        info = json.loads((root / "meta/info.json").read_text())
        if info.get("codebase_version") != "v3.0":
            _ensure_local_recording_metadata(root)
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
        dataset = _resume_existing_dataset_for_recording(repo, root, rgb_encoder=encoder)
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
            use_videos=any(ft["dtype"] == "video" for ft in schema.values()),
            rgb_encoder=encoder,
            encoder_threads=2,
        )
    return dataset, root
