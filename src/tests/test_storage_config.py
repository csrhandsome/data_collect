import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from control.collection.dataset import open_dataset
from control.collection.pipeline import run_collection
from control.collection.schema import FIELD_NAMES, features
from control.config import load_config, validate
from data_analysis.dataset_io import episode_dataframe, episode_paths


def test_storage_inheritance_and_partial_override(tmp_path, monkeypatch):
    panda_path = Path(__file__).resolve().parents[2] / "config/train/panda.yaml"
    custom_path = tmp_path / "custom.yaml"
    custom_path.write_text(
        yaml.safe_dump(
            {
                "extends": str(panda_path),
                "observation": {"exterior_image": False},
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    config = load_config(custom_path)
    assert config["observation"]["exterior_image"] is False
    assert config["observation"]["wrist_image_left"] is True
    assert {f"{group}.{key}" for group in ("observation", "action") for key in config[group]} == {
        "observation.exterior_image",
        "observation.wrist_image_left",
        "observation.gripper_image_left",
        "observation.gripper_image_right",
        "observation.joint_position",
        "observation.ee_pose",
        "observation.ee_position",
        "observation.gripper_position",
        "action.joint_position",
        "action.ee_pose",
        "action.gripper_position",
    }


@pytest.mark.parametrize(
    "group,fields",
    [
        ("observation", ["ee_pose"]),
        ("observation", {"unknown": True}),
        ("action", {"ee_pose": "false"}),
        ("action", {"actions": True}),
    ],
)
def test_invalid_storage_selection_is_rejected(group, fields):
    config = load_config()
    config[group] = fields
    with pytest.raises(ValueError, match=group):
        validate(config)


def test_unavailable_storage_selection_is_rejected():
    config = load_config()
    for group in ("observation", "action"):
        config[group] = {key: key.startswith("gripper_image") for key in config[group]}
    with pytest.raises(ValueError, match="available field"):
        validate(config)
    config["tactile"]["enabled"] = True
    config["gripper"]["type"] = "dh5"
    validate(config)


def test_legacy_field_selection_is_rejected():
    config = load_config()
    config["dataset"]["fields"] = {"actions": True}
    with pytest.raises(ValueError, match="dataset.fields"):
        validate(config)


def test_parent_field_selection_uses_child_tactile_settings(tmp_path):
    config = load_config()
    parent = tmp_path / "dataset.yaml"
    for group in ("observation", "action"):
        config[group] = {key: key == "gripper_image_left" for key in config[group]}
    parent.write_text(yaml.safe_dump(config))
    child = tmp_path / "child.yaml"
    child.write_text(
        yaml.safe_dump(
            {"extends": "dataset.yaml", "gripper": {"type": "dh5"}, "tactile": {"enabled": True}}
        )
    )
    config = load_config(child)
    assert set(features(32, "ee", True, config["observation"], config["action"])) == {
        "observation.gripper_image_left"
    }


@pytest.mark.parametrize("space,save_actions", [("ee", True), ("joint", True), ("ee", False)])
def test_selected_fields_are_saved_and_schema_changes_prevent_resume(tmp_path, space, save_actions):
    config = copy.deepcopy(load_config())
    config["dataset"].update(root=str(tmp_path), date="selected", action_space=space)
    config["observation"] = {key: key == "ee_position" for key in config["observation"]}
    config["action"] = {key: save_actions for key in config["action"]}
    config["audio"].update(enabled=False, vad_enabled=False)
    run_collection(config, dry_run=True, max_steps=30)
    root = tmp_path / "franka_lerobot_selected"
    info = json.loads((root / "meta/info.json").read_text())
    selected = {"observation.ee_position"} | (
        {"action.joint_position", "action.ee_pose", "action.gripper_position"}
        if save_actions
        else set()
    )
    assert set(info["features"]) & FIELD_NAMES == selected
    rows = episode_dataframe(episode_paths(root)[0], 0)
    assert set(rows.columns) & FIELD_NAMES == selected
    sync = json.loads((root / "episode_000000.sync.json").read_text())
    assert (root / sync["action_trace"]).is_file()
    if save_actions:
        for key in ("joint_position", "ee_pose", "gripper_position"):
            expected = [record["action_" + key] for record in sync["frame_records"]]
            if key == "gripper_position":
                expected = np.asarray(expected)[:, None]
            np.testing.assert_allclose(
                np.stack(rows["action." + key]).reshape(np.asarray(expected).shape),
                expected,
                rtol=1e-6,
                atol=1e-7,
            )
    dataset, _ = open_dataset(config)
    dataset.finalize()
    config["observation"]["joint_position"] = True
    with pytest.raises(ValueError, match="schema/fps mismatch"):
        open_dataset(config)
    config["observation"]["joint_position"] = False
    if save_actions:
        config["action"]["ee_pose"] = False
        with pytest.raises(ValueError, match="schema/fps mismatch"):
            open_dataset(config)
