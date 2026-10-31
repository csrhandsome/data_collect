"""Standalone Panda Desk HTTP client and ROS bringup, without panda-py.

Protocol reference: https://www.hyperturing.de/panda-py/_modules/panda_py.html
Only the Desk web protocol is used; ROS remains the sole FCI motion client.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import ssl
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

import yaml

ROOT = Path(__file__).resolve().parents[2]


class DeskError(RuntimeError):
    """Safe diagnostic that never includes server bodies or authentication data."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Desk:
    """Minimal HTTPS client for Panda/FER Desk, including pre-SPoC firmware."""

    def __init__(self, hostname, username, password, token_path=None):
        address = urlsplit(f"https://{hostname}")
        if not address.hostname or address.username or address.password or address.path or address.query or address.fragment:
            raise DeskError("Desk hostname must be a host or IP address, optionally with a port")
        self.host = hostname
        self.username = username
        self.password = password
        self.authorization = ""
        self.token = {}
        self.legacy = False
        self.logged_in = False
        self.token_path = Path(token_path or ROOT / "data/runtime/desk_tokens.json")
        # Desk serves a self-signed certificate. Keep this exception local to this
        # direct robot connection; do not alter global TLS or proxy settings.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        self.opener = build_opener(ProxyHandler({}), HTTPSHandler(context=context), NoRedirect())

    def _request(self, method, path, payload=None, *, body=None, headers=None, timeout=10):
        request_headers = dict(headers or {})
        if self.authorization:
            request_headers["Cookie"] = f"authorization={self.authorization}"
        if payload is not None:
            body = json.dumps(payload).encode()
            request_headers["Content-Type"] = "application/json"
        request = Request(f"https://{self.host}{path}", data=body, headers=request_headers, method=method)
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            status = exc.code
            exc.close()
            raise DeskError(f"Desk {method} {path}: HTTP {status}", status) from None
        except (URLError, TimeoutError, OSError):
            raise DeskError(f"Desk {method} {path}: connection failed or timed out") from None

    @staticmethod
    def encode_password(username, password):
        digest = hashlib.sha256(f"{password}#{username}@franka".encode()).digest()
        # Desk encodes decimal digest bytes, not the raw hash or its hex string.
        return base64.encodebytes(",".join(str(value) for value in digest).encode()).decode()

    def login(self):
        self.authorization = self._request("POST", "/admin/api/login", {
            "login": self.username,
            "password": self.encode_password(self.username, self.password),
        }).strip()
        self.logged_in = True

    def _cached_tokens(self):
        if not self.token_path.exists():
            return {}
        return json.loads(self.token_path.read_text())

    def _save_token(self, token):
        tokens = self._cached_tokens()
        if token:
            tokens[self.host] = token
        else:
            tokens.pop(self.host, None)
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic, mode 0600; survives --rm Docker runs through the data mount.
        with tempfile.NamedTemporaryFile(mode="w", dir=self.token_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(tokens, stream)
                stream.close()
                temporary.replace(self.token_path)
            finally:
                temporary.unlink(missing_ok=True)

    def take_control(self):
        try:
            active = json.loads(self._request("GET", "/admin/api/control-token"))["activeToken"]
        except DeskError as exc:
            if exc.status != 404:
                raise
            # Old Desk has neither control tokens nor a separate FCI switch.
            self.legacy = True
            return True
        cached = self._cached_tokens().get(self.host, {})
        if active is not None:
            if cached.get("token") and str(active["id"]) == str(cached.get("id")):
                self.token = cached
                return True
            return False
        self.token = json.loads(self._request(
            "POST", "/admin/api/control-token/request", {"requestedBy": self.username},
        ))
        if not self.token.get("token") or "id" not in self.token:
            raise DeskError("Desk returned an invalid control token")
        self._save_token(self.token)
        return True

    def unlock(self):
        boundary = "data-collect-desk-brakes"
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="force"; filename="force"\r\n'
            f'\r\nTrue\r\n--{boundary}--\r\n'
        ).encode()
        self._request("POST", "/desk/api/robot/open-brakes", body=body, headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-Control-Token": self.token.get("token", ""),
        }, timeout=60)

    def activate_fci(self):
        if not self.legacy:
            self._request("POST", "/admin/api/control-token/fci", {"token": self.token["token"]})

    def deactivate_fci(self):
        if not self.legacy:
            self._request("DELETE", "/admin/api/control-token/fci", {"token": self.token["token"]})

    def release_control(self):
        if not self.legacy and self.token:
            self._request("DELETE", "/admin/api/control-token", {"token": self.token["token"]})
            self._save_token({})
            self.token = {}

    def logout(self):
        if self.logged_in:
            try:
                self._request("POST", "/admin/api/logout")
            except DeskError as exc:
                # An expired/invalidated session is already logged out.
                if exc.status != 401:
                    raise
            self.authorization = ""
            self.logged_in = False


def robot_credentials(launch_args):
    # Match ROS launch's last assignment, including an overridden driver config.
    path = ROOT / "src/env/franka.config.yaml"
    for arg in launch_args:
        if arg.startswith("robot_config_file:="):
            path = Path(arg.split(":=", 1)[1])
    robots = yaml.safe_load(path.read_text())
    if not isinstance(robots, dict) or len(robots) != 1:
        raise ValueError("Desk bringup requires one robot in robot_config_file")
    host = next(iter(robots.values()))["robot_ip"]

    # Read inherited credentials without importing the collection/hardware stack.
    def read_robot(path, seen):
        path = path.resolve()
        if path in seen:
            raise ValueError("Cyclic YAML inheritance in Desk credential config")
        seen = seen | {path}
        payload = yaml.safe_load(path.read_text())
        parent = payload.get("extends")
        robot = read_robot(path.parent / parent, seen) if parent else {}
        robot.update(payload.get("robot", {}))
        return robot

    config = Path(os.environ.get("DATA_COLLECT_CONFIG") or ROOT / "config/train/panda.yaml")
    robot = read_robot(config, set())
    username = os.environ.get("FRANKA_USERNAME") or robot.get("username")
    password = os.environ.get("FRANKA_PASSWORD") or robot.get("password")
    if not username or not password:
        raise ValueError("Set FRANKA_USERNAME / FRANKA_PASSWORD or robot.username / robot.password")
    return host, username, password


@contextmanager
def desk_session(launch_args):
    if os.environ.get("ROS2_DESK_AUTO_UNLOCK", "1") == "0":
        print("Desk automation disabled; unlock brakes and activate FCI manually.", flush=True)
        yield
        return

    host, username, password = robot_credentials(launch_args)
    print(f"Desk: connecting to {host} to unlock brakes and activate FCI...", flush=True)
    desk = Desk(host, username, password)
    owned = False
    activation_attempted = False
    try:
        desk.login()
        owned = desk.take_control()
        if not owned:
            print("Desk is owned by another session; release control in Desk first.", file=sys.stderr)
            raise RuntimeError("Desk is owned by another session; release control in Desk first")
        desk.unlock()
        activation_attempted = True
        desk.activate_fci()
        print("Desk: brakes unlocked, FCI activated; starting ROS controllers.", flush=True)
        yield
    finally:
        # Stop ROS first (the caller exits run_launch before reaching this block).
        callbacks = []
        if activation_attempted:
            callbacks.append(desk.deactivate_fci)
        if owned or desk.token:
            callbacks.append(desk.release_control)
        callbacks.append(desk.logout)
        for callback in callbacks:
            try:
                callback()
            except Exception as exc:
                print(f"Desk cleanup {callback.__name__} failed ({type(exc).__name__}); check Desk.", file=sys.stderr)


def run_launch(command):
    """Forward shutdown to ROS and wait before relinquishing Desk ownership."""
    child = None
    interrupted = 0
    deadline = None

    def stop(signum, _frame):
        nonlocal interrupted, deadline
        if not interrupted:
            interrupted = signum
            deadline = time.monotonic() + 15
        if child is not None and child.poll() is None:
            try:
                # ROS launch handles SIGINT with orderly shutdown of all nodes.
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        child = subprocess.Popen(command, start_new_session=True)
        if interrupted:
            stop(interrupted, None)
        while True:
            try:
                status = child.wait(timeout=0.2)
                return 128 + interrupted if interrupted else (128 - status if status < 0 else status)
            except subprocess.TimeoutExpired:
                if deadline is not None and time.monotonic() >= deadline:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
                    return 128 + interrupted
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(launch_args):
    command = ["ros2", "launch", "panda_cartesian_controller", "panda_bringup.launch.py", *launch_args]

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    # Also unwind Desk setup if deployment is stopped during unlocking.
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        with desk_session(launch_args):
            return run_launch(command)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        # Only our sanitized Desk errors are safe to print.
        detail = f" {exc}." if isinstance(exc, DeskError) else ""
        print(
            f"ROS bringup failed ({type(exc).__name__}).{detail} Check Desk credentials, control ownership, "
            "brakes and FCI. Emergency/user stop must be released physically. "
            "Use ROS2_DESK_AUTO_UNLOCK=0 for manual Desk preparation.",
            file=sys.stderr,
        )
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
