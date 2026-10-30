"""Desk/ROS startup ordering and shutdown, without contacting a robot."""

import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "env/desk_bringup.py"
spec = importlib.util.spec_from_file_location("desk_bringup", SOURCE)
bringup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bringup)


@pytest.fixture
def desk(monkeypatch):
    events = []

    class Desk:
        token = {}

        def __init__(self, *args, **kwargs):
            pass

        def login(self):
            events.append("login")

        def take_control(self):
            return True

        def unlock(self):
            events.append("unlock")

        def activate_fci(self):
            events.append("activate")

        def deactivate_fci(self):
            events.append("deactivate")

        def release_control(self):
            events.append("release")

        def logout(self):
            events.append("logout")

    monkeypatch.delenv("ROS2_DESK_AUTO_UNLOCK", raising=False)
    monkeypatch.setattr(bringup, "Desk", Desk)
    monkeypatch.setattr(bringup, "robot_credentials", lambda _: ("robot", "user", "password"))
    return Desk, events


def test_desk_wraps_ros_lifetime_and_preserves_failure(desk, monkeypatch):
    _, events = desk

    def run(command):
        assert command[-1] == "robot_config_file:=custom.yaml"
        events.extend(["ros_start", "ros_exit"])
        return 6

    monkeypatch.setattr(bringup, "run_launch", run)
    assert bringup.main(["robot_config_file:=custom.yaml"]) == 6
    assert events == ["login", "unlock", "activate", "ros_start", "ros_exit", "deactivate", "release", "logout"]


@pytest.mark.parametrize("step", ["unlock", "activate_fci"])
def test_desk_failure_blocks_ros_and_releases_session(desk, monkeypatch, step):
    cls, events = desk

    def fail(self):
        raise ConnectionError("sensitive HTTP response")

    monkeypatch.setattr(cls, step, fail)
    monkeypatch.setattr(bringup, "run_launch", lambda _: pytest.fail("ROS must not start"))
    assert bringup.main([]) == 1
    assert events[-2:] == ["release", "logout"]


def test_another_owner_is_not_released(desk, monkeypatch):
    cls, events = desk
    monkeypatch.setattr(cls, "take_control", lambda _: False)
    monkeypatch.setattr(bringup, "run_launch", lambda _: pytest.fail("ROS must not start"))
    assert bringup.main([]) == 1
    assert events == ["login", "logout"]


def test_manual_mode_does_not_touch_desk(desk, monkeypatch):
    _, events = desk
    monkeypatch.setenv("ROS2_DESK_AUTO_UNLOCK", "0")
    monkeypatch.setattr(bringup, "run_launch", lambda _: 0)
    assert bringup.main([]) == 0
    assert events == []


def test_cleanup_continues_after_fci_deactivation_error(desk, monkeypatch):
    cls, events = desk

    def fail(self):
        raise ConnectionError("failure")

    monkeypatch.setattr(cls, "deactivate_fci", fail)
    with bringup.desk_session([]):
        pass
    assert events[-2:] == ["release", "logout"]


def test_credentials_follow_driver_override_and_config_inheritance(tmp_path, monkeypatch):
    default = tmp_path / "default.yaml"
    default.write_text('ROBOT1:\n  robot_ip: old-host\n')
    driver = tmp_path / "driver.yaml"
    driver.write_text('ROBOT1:\n  robot_ip: driver-host\n')
    parent = tmp_path / "parent.yaml"
    parent.write_text('robot:\n  hostname: ignored-host\n  username: parent-user\n  password: parent-pass\n')
    config = tmp_path / "config.yaml"
    config.write_text('extends: parent.yaml\nrobot:\n  username: child-user\n')
    monkeypatch.setenv("DATA_COLLECT_CONFIG", str(config))
    monkeypatch.delenv("FRANKA_USERNAME", raising=False)
    monkeypatch.delenv("FRANKA_PASSWORD", raising=False)
    args = [f"robot_config_file:={default}", f"robot_config_file:={driver}"]
    assert bringup.robot_credentials(args) == ("driver-host", "child-user", "parent-pass")
    monkeypatch.setenv("FRANKA_USERNAME", "env-user")
    monkeypatch.setenv("FRANKA_PASSWORD", "env-pass")
    assert bringup.robot_credentials(args) == ("driver-host", "env-user", "env-pass")


def test_missing_credentials_blocks_setup(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text('robot: {}\n')
    monkeypatch.setenv("DATA_COLLECT_CONFIG", str(config))
    monkeypatch.delenv("FRANKA_USERNAME", raising=False)
    monkeypatch.delenv("FRANKA_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="FRANKA_USERNAME"):
        bringup.robot_credentials([])


def test_sigterm_waits_for_launch_shutdown(tmp_path):
    ready = tmp_path / "ready"
    stopped = tmp_path / "stopped"
    child_code = (
        "import signal, time; from pathlib import Path; "
        f"signal.signal(signal.SIGINT, lambda *_: (Path({str(stopped)!r}).touch(), exit(0))); "
        f"Path({str(ready)!r}).touch(); time.sleep(30)"
    )
    supervisor_code = (
        "import importlib.util, sys; "
        f"s=importlib.util.spec_from_file_location('bringup', {str(SOURCE)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"sys.exit(m.run_launch([sys.executable, '-c', {child_code!r}]))"
    )
    process = subprocess.Popen([sys.executable, "-c", supervisor_code], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.02)
        assert ready.exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 143
        assert stopped.exists()
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=20)


def test_supervisor_preserves_child_exit_code():
    assert bringup.run_launch([sys.executable, "-c", "raise SystemExit(7)"]) == 7


def test_host_deploy_fails_before_starting_background_processes(tmp_path):
    # Invoke bash by absolute path; an empty PATH simulates a host without ROS.
    result = subprocess.run(
        ["/bin/bash", str(SOURCE.parents[2] / "scripts/deploy_ros2.sh")],
        env={**os.environ, "PATH": str(tmp_path)}, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "deploy_docker.sh" in result.stderr
    assert "ros2: command not found" not in result.stderr


@pytest.fixture
def http_desk(tmp_path):
    client = bringup.Desk("robot", "testuser", "testpass", tmp_path / "tokens.json")
    requests = []
    responses = []

    def open_request(request, timeout):
        requests.append((request, timeout))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, (dict, list)):
            response = json.dumps(response)
        return io.BytesIO(response.encode())

    client.opener.open = open_request
    return client, requests, responses


def test_independent_http_protocol_and_cleanup(http_desk, monkeypatch):
    client, requests, responses = http_desk
    # Any accidental panda-py import would fail, including native SDK loading.
    monkeypatch.setitem(sys.modules, "panda_py", None)
    responses.extend([
        "auth-cookie", {"activeToken": None}, {"id": 12, "token": "control-token"},
        "", "", "", "", "",
    ])
    client.login()
    assert client.take_control()
    assert client.token_path.stat().st_mode & 0o777 == 0o600
    assert json.loads(client.token_path.read_text())["robot"]["token"] == "control-token"
    client.unlock()
    client.activate_fci()
    client.deactivate_fci()
    client.release_control()
    client.logout()
    assert json.loads(client.token_path.read_text()) == {}
    assert [(r.method, urlsplit(r.full_url).path) for r, _ in requests] == [
        ("POST", "/admin/api/login"), ("GET", "/admin/api/control-token"),
        ("POST", "/admin/api/control-token/request"), ("POST", "/desk/api/robot/open-brakes"),
        ("POST", "/admin/api/control-token/fci"), ("DELETE", "/admin/api/control-token/fci"),
        ("DELETE", "/admin/api/control-token"), ("POST", "/admin/api/logout"),
    ]
    assert json.loads(requests[0][0].data)["login"] == "testuser"
    unlock, timeout = requests[3]
    assert unlock.get_header("Cookie") == "authorization=auth-cookie"
    assert unlock.get_header("X-control-token") == "control-token"
    assert 'name="force"' in unlock.data.decode()
    assert "\r\nTrue\r\n" in unlock.data.decode()
    assert timeout == 60
    assert json.loads(requests[4][0].data) == {"token": "control-token"}


def test_password_encoding_matches_reference_protocol():
    # Reference output from Desk.encode_password for dummy credentials.
    assert bringup.Desk.encode_password("testuser", "testpass") == (
        "MTAsMTU0LDYsMTI4LDIzNSwyNTEsMzcsMTUsMjMzLDkzLDE4MCwxNSwxMDYsMTA2LDMzLDEzNiwx\n"
        "MjQsNzksMiwxMDYsMTM2LDMyLDg3LDE3NCwxMDUsMTk5LDE5NSwzNCw4OCwyMjYsMTIxLDcw\n"
    )


def test_existing_token_is_reused_after_container_restart(http_desk):
    client, requests, responses = http_desk
    client._save_token({"id": 12, "token": "saved-token"})
    responses.append({"activeToken": {"id": "12", "ownedBy": "testuser"}})
    assert client.take_control()
    assert client.token["token"] == "saved-token"
    assert len(requests) == 1


def test_foreign_control_token_is_not_claimed(http_desk):
    client, requests, responses = http_desk
    client._save_token({"id": 12, "token": "saved-token"})
    responses.append({"activeToken": {"id": 99, "ownedBy": "other-user"}})
    assert not client.take_control()
    assert not client.token
    assert len(requests) == 1


def test_old_desk_does_not_require_control_token_or_fci_endpoint(http_desk):
    client, requests, responses = http_desk
    responses.extend([HTTPError("https://robot", 404, "not found", {}, io.BytesIO(b"")), ""])
    assert client.take_control()
    client.unlock()
    client.activate_fci()
    client.deactivate_fci()
    client.release_control()
    assert len(requests) == 2
    assert client.legacy


@pytest.mark.parametrize("status", [401, 403, 500])
def test_http_errors_are_not_mistaken_for_old_firmware_or_leaked(http_desk, status):
    client, _, responses = http_desk
    responses.append(HTTPError("https://robot", status, "password-secret", {}, io.BytesIO(b"token-secret")))
    with pytest.raises(bringup.DeskError) as error:
        client.take_control()
    assert error.value.status == status
    assert "secret" not in str(error.value)
    assert not client.legacy


def test_logout_of_invalidated_session_is_idempotent(http_desk):
    client, requests, responses = http_desk
    client.logged_in = True
    client.authorization = "expired-cookie"
    responses.append(HTTPError("https://robot", 401, "unauthorized", {}, io.BytesIO(b"")))
    client.logout()
    client.logout()
    assert not client.logged_in
    assert not client.authorization
    assert len(requests) == 1


def test_logout_does_not_hide_server_failure(http_desk):
    client, _, responses = http_desk
    client.logged_in = True
    responses.append(HTTPError("https://robot", 500, "error", {}, io.BytesIO(b"")))
    with pytest.raises(bringup.DeskError):
        client.logout()
    assert client.logged_in
