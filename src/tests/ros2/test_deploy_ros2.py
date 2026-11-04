"""Check host/container routing without Docker, ROS or hardware."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/deploy_ros2.sh"


@pytest.fixture
def deployment(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Redirect only the OS marker paths so tests work on both hosts and containers.
    source = SCRIPT.read_text()
    for marker in ("/.dockerenv", "/run/.containerenv", "/proc/1/cgroup"):
        source = source.replace(marker, str(tmp_path / Path(marker).name))
    script = scripts / SCRIPT.name
    script.write_text(source)
    shutil.copy2(SCRIPT.with_name("deploy_docker.sh"), scripts)
    for command in ("bash", "dirname", "grep"):
        (bin_dir / command).symlink_to(shutil.which(command))
    env = {**os.environ, "PATH": str(bin_dir)}

    def run(*args):
        return subprocess.run(
            ["/bin/bash", str(script), *args], env=env,
            capture_output=True, text=True, timeout=5,
        )

    return tmp_path, bin_dir, run


def executable(path, body):
    path.write_text(f"#!{sys.executable}\n{body}\n")
    path.chmod(0o755)


@pytest.mark.parametrize("host_ros", [False, True])
def test_host_enters_docker_and_preserves_arguments_and_exit_code(deployment, host_ros):
    root, bin_dir, run = deployment
    if host_ros:
        executable(bin_dir / "ros2", "raise SystemExit(0)")
    executable(bin_dir / "docker", f"""
import json, sys
from pathlib import Path
if sys.argv[1:3] == ['image', 'inspect']:
    raise SystemExit(0)
Path({str(root / 'docker-args')!r}).write_text(json.dumps(sys.argv[1:]))
raise SystemExit(17)
""")
    result = run("robot_ip:=192.0.2.1", "label:=two words")
    assert result.returncode == 17, result.stderr
    args = json.loads((root / "docker-args").read_text())
    assert args[:3] == ["run", "--rm", "-it"]
    assert args[-6:] == [
        "data-collect/franka-ros2-jazzy:dev", "bash", "scripts/deploy_ros2.sh",
        "--inside", "robot_ip:=192.0.2.1", "label:=two words",
    ]
    assert "Host environment detected" in result.stdout


@pytest.mark.parametrize("marker", [".dockerenv", ".containerenv", "cgroup", "--inside"])
def test_container_without_ros_fails_without_trying_nested_docker(deployment, marker):
    root, bin_dir, run = deployment
    if marker != "--inside":
        (root / marker).write_text("0::/system.slice/docker-abc.scope\n")
    executable(bin_dir / "docker", "raise RuntimeError('nested Docker must not run')")
    result = run(*(["--inside"] if marker == "--inside" else []))
    assert result.returncode == 1
    assert "ROS 2 environment missing inside the container" in result.stderr
    assert "nested Docker" not in result.stderr


def test_container_starts_directly_and_preserves_bringup_failure(deployment):
    root, bin_dir, run = deployment
    (root / ".dockerenv").touch()
    executable(bin_dir / "ros2", "raise SystemExit(0)")
    executable(bin_dir / "docker", "raise RuntimeError('nested Docker must not run')")
    executable(bin_dir / "setsid", """
import json, sys
print(json.dumps(sys.argv[1:]), flush=True)
raise SystemExit(7 if sys.argv[2].endswith('/bringup.sh') else 0)
""")
    result = run("label:=two words")
    assert result.returncode == 7, result.stderr
    assert "Container environment detected" in result.stdout
    calls = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("[")]
    assert len(calls) == 4
    assert ["bash", str(root / "src/env/bringup.sh"), "label:=two words"] in calls
    assert "nested Docker" not in result.stderr


def test_host_without_docker_fails_before_starting_services(deployment):
    _, _, run = deployment
    result = run()
    assert result.returncode == 1
    assert "Docker is missing" in result.stderr


def test_ur_host_preserves_opt_in_flag(deployment):
    root, bin_dir, run = deployment
    executable(bin_dir / "docker", f"""
import json, sys
from pathlib import Path
if sys.argv[1:3] == ['image', 'inspect']:
    raise SystemExit(0)
Path({str(root / 'docker-args')!r}).write_text(json.dumps(sys.argv[1:]))
""")
    result = run("--ur", "use_mock_hardware:=true")
    assert result.returncode == 0, result.stderr
    args = json.loads((root / "docker-args").read_text())
    assert args[-6:] == ["data-collect/franka-ros2-jazzy:dev", "bash", "scripts/deploy_ros2.sh",
                         "--inside", "--ur", "use_mock_hardware:=true"]


def test_ur_container_never_starts_franka_bringup(deployment):
    root, bin_dir, run = deployment
    (root / ".dockerenv").touch()
    executable(bin_dir / "ros2", "raise SystemExit(0)")
    executable(bin_dir / "setsid", """
import json, os, sys
print(json.dumps({'args': sys.argv[1:], 'config': os.environ['DATA_COLLECT_CONFIG']}), flush=True)
""")
    result = run("use_mock_hardware:=true", "--ur")
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(calls) == 4
    assert all(item['config'].endswith('/config/train/ur.yaml') for item in calls)
    assert any(item['args'] == ['bash', str(root / 'src/env/bringup_ur.sh'), 'use_mock_hardware:=true'] for item in calls)
    assert not any(str(root / 'src/env/bringup.sh') in item['args'] for item in calls)
