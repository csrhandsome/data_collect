"""One schema builder, with recording space independent of EE control."""

import json
from pathlib import Path

import numpy as np

from control.util.lerobot_util import _resume_existing_dataset_for_recording


def features(image_hw, action_space, tactile):
    image = {
        "dtype": "image",
        "shape": (image_hw, image_hw, 3),
        "names": ["height", "width", "channel"],
    }
    result = {
        key: image.copy()
        for key in ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]
    }
    if tactile:
        result.update({key: image.copy() for key in ["gripper_image_left", "gripper_image_right"]})
    for key, names in {
        "joint_position": [f"joint_{i}" for i in range(7)],
        "ee_pose": ["x", "y", "z", "roll", "pitch", "yaw"],
        "ee_position": ["x", "y", "z"],
        "gripper_position": ["commanded_open_ratio"],
        "actions": (
            [f"joint_{i}" for i in range(7)]
            if action_space == "joint"
            else ["x", "y", "z", "roll", "pitch", "yaw"]
        )
        + ["commanded_open_ratio"],
    }.items():
        result[key] = {"dtype": "float32", "shape": (len(names),), "names": names}
    return result


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
    )
    if root.exists():
        info = json.loads((root / "meta/info.json").read_text())
        expected = {key: (value["dtype"], list(value["shape"])) for key, value in schema.items()}
        actual = {
            key: (value["dtype"], value["shape"])
            for key, value in info["features"].items()
            if key in schema
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


def recorded_action(state, action_space):
    values = state.joint_positions if action_space == "joint" else state.ee_pose.vector
    return np.concatenate([values, [state.gripper.commanded_open_ratio]]).astype(np.float32)
