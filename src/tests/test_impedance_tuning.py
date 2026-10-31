"""Both transports must consume the same tuning before motion can start."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from control.common.impedance import impedance_parameters
from control.config import load_config, validate
from control.direct.fake import FakeBackend
from control.direct.franka_runner_stable import PandaBackend
from control.ros2.controller_config import write_controller_config

ROOT = Path(__file__).resolve().parents[2]


def test_sdk_and_ros_launch_consume_identical_custom_tuning(tmp_path, monkeypatch):
    config = load_config()
    config["control"].update(
        ee_filter_coeff=0.25, ee_translational_stiffness=450,
        ee_rotational_stiffness=25, ee_nullspace_stiffness=0.8,
    )
    parameters = impedance_parameters(config["control"])
    output = write_controller_config(
        ROOT / "src/env/panda_cartesian_controller/config/controllers.yaml", config,
        tmp_path / "controllers.yaml",
    )
    written = yaml.safe_load(Path(output).read_text())
    assert written["cartesian_pose_impedance_controller"]["ros__parameters"] == {
        "arm_id": "panda", **parameters,
    }
    assert written["controller_manager"]["ros__parameters"]["update_rate"] == 1000
    assert "password" not in Path(output).read_text()

    constructed = {}

    class Controller:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

        def set_control(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "panda_py", SimpleNamespace(
        controllers=SimpleNamespace(CartesianImpedance=Controller)
    ))
    backend = PandaBackend(config)
    backend.panda = SimpleNamespace(start_controller=lambda controller: None)
    backend.start(FakeBackend().snapshot())
    assert constructed["filter_coeff"] == parameters["filter_coefficient"]
    assert constructed["nullspace_stiffness"] == parameters["nullspace_stiffness"]
    np.testing.assert_array_equal(
        np.diag(constructed["impedance"]),
        [parameters["translational_stiffness"]] * 3 + [parameters["rotational_stiffness"]] * 3,
    )


@pytest.mark.parametrize("key,value", [
    ("ee_filter_coeff", 0), ("ee_filter_coeff", 1.1), ("ee_filter_coeff", float("nan")),
    ("ee_translational_stiffness", -1), ("ee_rotational_stiffness", float("inf")),
    ("ee_nullspace_stiffness", -0.1),
])
def test_invalid_tuning_fails_config_validation_before_connecting(key, value):
    config = load_config()
    config["control"][key] = value
    with pytest.raises(ValueError, match=key):
        validate(config)


def test_preflight_rejects_an_old_installed_controller(tmp_path):
    spec = importlib.util.spec_from_file_location("preflight", ROOT / "src/env/check_controller.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    library = tmp_path / "old.so"
    library.write_bytes(b"old controller without configurable smoothing")
    with pytest.raises(RuntimeError, match="recreate this container"):
        module.check_installed(library)
