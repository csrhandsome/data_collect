"""Exercise the actual C++ joint torque law against coupled Panda inertia."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def servo_simulation(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None or not Path("/usr/include/eigen3/Eigen/Core").is_file():
        pytest.skip("C++ joint dynamics regression requires g++ and Eigen")
    binary = tmp_path_factory.mktemp("joint-servo") / "simulate"
    subprocess.run([
        compiler, "-std=c++20", "-O2", "-I/usr/include/eigen3",
        "-I" + str(ROOT / "src/env/panda_cartesian_controller/include"),
        str(Path(__file__).with_name("joint_reset_dynamics.cpp")), "-o", str(binary),
    ], check=True, capture_output=True, text=True)

    def run(mode="fixed", scale=1, period=0.001):
        output = subprocess.check_output([str(binary), mode, str(scale), str(period)], text=True)
        return json.loads(output)

    return run


def test_old_damping_reproduces_endpoint_oscillation_and_timeout(servo_simulation):
    result = servo_simulation("old")
    assert not result["arrived"]
    assert result["end_position_error"] < 0.02
    assert result["end_velocity"] > 0.005
    assert result["endpoint_peak_to_peak"] > 0.001


@pytest.mark.parametrize("scale,period", [(0.05, 0.001), (1, 0.001), (1, 0.002)])
def test_paired_gains_settle_without_relaxing_arrival_or_torque_limits(
    servo_simulation, scale, period
):
    result = servo_simulation(scale=scale, period=period)
    assert result["arrived"]
    assert result["end_position_error"] < 0.02
    assert result["end_velocity"] < 0.005
    assert result["peak_torque_change"] <= 1.00001
