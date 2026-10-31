"""Driver configuration must work without collection or dataset packages."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from control.config import load_config
from control.yaml_config import load_yaml_config


def test_lightweight_reader_matches_collection_config_control_values():
    assert load_yaml_config()["control"] == load_config()["control"]


def test_nested_inheritance_keeps_parent_values_and_child_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("control:\n  ee_filter_coeff: 0.35\n  ee_nullspace_stiffness: 0.5\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\ncontrol:\n  ee_filter_coeff: 0.25\n")
    assert load_yaml_config(child)["control"] == {
        "ee_filter_coeff": 0.25, "ee_nullspace_stiffness": 0.5,
    }
    child.write_text("extends: child.yaml\n")
    with pytest.raises(ValueError, match="Cyclic"):
        load_yaml_config(child)
    child.write_text("- invalid\n")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml_config(child)


def test_driver_parameter_generation_does_not_import_collection_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "controllers.yaml"
    script = '''
import importlib.abc
import sys
from pathlib import Path
class RejectCollectionImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"lerobot", "torch", "numpy", "panda_py", "teleop_xr"}:
            raise ImportError("Driver imported collection dependency: " + fullname)
sys.meta_path.insert(0, RejectCollectionImports())
from control.yaml_config import load_yaml_config
from control.ros2.controller_config import write_controller_config
write_controller_config(
    Path(sys.argv[1]) / "src/env/panda_cartesian_controller/config/controllers.yaml",
    load_yaml_config(Path(sys.argv[1]) / "config/train/panda.yaml"), sys.argv[2],
)
assert "control.config" not in sys.modules
'''
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run([sys.executable, "-c", script, str(root), str(output)], env=env, check=True)
    parameters = yaml.safe_load(output.read_text())["cartesian_pose_impedance_controller"]["ros__parameters"]
    assert parameters["filter_coefficient"] == load_config()["control"]["ee_filter_coeff"]
