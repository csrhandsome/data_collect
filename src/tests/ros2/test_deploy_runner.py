"""Ensure runner/page supervisors leave an independently started ROS deployment alone."""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/deploy_runner.sh"


@pytest.fixture
def launch(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "bin").mkdir()
    script = tmp_path / "scripts/deploy_runner.sh"
    shutil.copy2(SCRIPT, script)
    events = tmp_path / "events"
    program = tmp_path / "mock.py"
    program.write_text('''
import os, signal, sys, time
from pathlib import Path
root = Path(os.environ['MOCK_ROOT'])
def event(value):
    with (root / 'events').open('a') as stream:
        stream.write(value + '\\n')
if sys.argv[1] == 'driver':
    def stop(*_):
        event('driver_stop')
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    event('driver_start')
    time.sleep(0.1)
    (root / 'ready').touch()
    while True:
        time.sleep(0.02)
elif sys.argv[1] == 'web':
    if 'install' in sys.argv:
        raise SystemExit(0)
    def stop(*_):
        event('web_stop')
        raise SystemExit(0)
    signal.signal(signal.SIGINT, stop)
    event('web_start')
    while True:
        time.sleep(0.02)
elif sys.argv[1] == '-':
    sys.stdin.read()
    event('preflight')
    if os.environ.get('MOCK_PREFLIGHT_FAIL'):
        raise SystemExit(9)
elif sys.argv[1:3] in (['-m', 'vr_collect'], ['-m', 'replay.backend.cli']):
    def stop(*_):
        event('collector_saved')
        time.sleep(0.1)
        raise SystemExit(0)
    signal.signal(signal.SIGINT, stop)
    if sys.argv[2] == 'replay.backend.cli':
        event('api_start')
    event('collector_start')
    if os.environ.get('MOCK_COLLECT_EXIT'):
        raise SystemExit(int(os.environ['MOCK_COLLECT_EXIT']))
    while True:
        time.sleep(0.02)
''')
    python = tmp_path / "bin/python"
    python.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{program}" "$@"\n')
    ros2 = tmp_path / "bin/ros2"
    ros2.write_text('#!/bin/bash\nexit 0\n')
    for executable in (python, ros2):
        executable.chmod(0o755)
    (tmp_path / "scripts/deploy_ros2.sh").write_text(
        f'#!/bin/bash\nexec "{sys.executable}" "{program}" driver\n'
    )

    def start(with_api=False, **extra):
        return subprocess.Popen(
            ["/bin/bash", str(script), "--inside", *(["--with-api"] if with_api else []), "--max-steps", "5"],
            env={**os.environ, "PATH": f'{tmp_path / "bin"}:/usr/bin:/bin',
                 "MOCK_ROOT": str(tmp_path), **extra},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

    driver = subprocess.Popen(
        [sys.executable, str(program), "driver"],
        env={**os.environ, "MOCK_ROOT": str(tmp_path)}, start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while not (tmp_path / "ready").exists():
        assert driver.poll() is None and time.monotonic() < deadline
        time.sleep(0.02)
    start.driver = driver
    try:
        yield start, events
    finally:
        driver.terminate()
        driver.wait(timeout=5)


def test_runner_does_not_start_ros_and_preserves_failure(launch):
    start, events = launch
    process = start(MOCK_COLLECT_EXIT="7")
    output, _ = process.communicate(timeout=5)
    assert process.returncode == 7, output
    assert events.read_text().splitlines() == [
        "driver_start", "preflight", "collector_start",
    ]


def test_page_mode_starts_api_without_starting_ros(launch):
    start, events = launch
    process = start(with_api=True, MOCK_COLLECT_EXIT="7")
    output, _ = process.communicate(timeout=5)
    assert process.returncode == 7, output
    assert events.read_text().splitlines() == [
        "driver_start", "preflight", "api_start", "collector_start",
    ]


def test_config_failure_never_starts_collection(launch):
    start, events = launch
    process = start(MOCK_PREFLIGHT_FAIL="1")
    output, _ = process.communicate(timeout=5)
    assert process.returncode == 9, output
    assert "collector_start" not in events.read_text()


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_shutdown_saves_collection_and_leaves_driver_running(launch, sig):
    start, events = launch
    process = start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if events.exists() and "collector_start" in events.read_text():
                break
            assert process.poll() is None
            time.sleep(0.02)
        else:
            pytest.fail("Collector failed to start")
        process.send_signal(sig)
        output, _ = process.communicate(timeout=5)
        assert process.returncode == 128 + sig, output
        assert events.read_text().splitlines()[-1] == "collector_saved"
        assert start.driver.poll() is None
        assert events.read_text().splitlines().count("driver_start") == 1
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=5)


def test_help_and_invalid_options_do_not_start_docker():
    help_result = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "deploy_page.sh" in help_result.stdout
    invalid = subprocess.run(["bash", str(SCRIPT), "--max-steps", "0"], capture_output=True, text=True)
    assert invalid.returncode == 2


def test_page_entrypoint_starts_runner_and_web_and_cleans_up(launch):
    start, events = launch
    root = events.parent
    script = root / "scripts/deploy_page.sh"
    shutil.copy2(SCRIPT.with_name("deploy_page.sh"), script)
    pnpm = root / "bin/pnpm"
    pnpm.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{root / "mock.py"}" web "$@"\n')
    docker = root / "bin/docker"
    docker.write_text(
        '#!/bin/bash\nif [[ "$1" == image || "$*" == "compose version" ]]; then exit 0; fi\n'
        f'exec bash "{root / "scripts/deploy_runner.sh"}" --inside --with-api\n'
    )
    pnpm.chmod(0o755)
    docker.chmod(0o755)
    process = subprocess.Popen(
        ["/bin/bash", str(script)],
        env={**os.environ, "PATH": f'{root / "bin"}:/usr/bin:/bin', "MOCK_ROOT": str(root)},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if events.exists() and "collector_start" in events.read_text():
                break
            assert process.poll() is None
            time.sleep(0.02)
        else:
            pytest.fail("Page did not start its managed runner")
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=5)
        assert process.returncode == 143, output
        values = events.read_text().splitlines()
        assert "web_start" in values and "api_start" in values
        assert values.index("collector_saved") < values.index("web_stop")
        assert "driver_stop" not in values and values.count("driver_start") == 1
        assert start.driver.poll() is None
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=5)
