"""Desktop-specific routing and shutdown contracts, without hardware."""

import sys
import time

from fastapi.testclient import TestClient

from replay.backend.main import create_app
from replay.backend.operations import OperationManager


def test_packaged_commands_are_fixed_and_use_the_user_config(tmp_path):
    manager = OperationManager(
        tmp_path, runtime_command=["/installed/bin/data-collect-runtime"],
        config_path=tmp_path / "panda.yaml", simulate=True, max_steps=25,
    )
    collect = manager.command("collect")
    assert collect == [
        "/installed/bin/data-collect-runtime", "collect", "--config",
        str(tmp_path / "panda.yaml"), "--dry-run", "--max-steps", "25",
    ]
    replay = manager.command("replay", ["--dataset", "/data/demo"])
    assert "uv" not in replay and "--dry-run" in replay
    deletion = manager.command("delete", ["--dataset", "/data/demo", "--yes"])
    assert deletion == [
        "/installed/bin/data-collect-runtime", "delete", "--dataset", "/data/demo", "--yes",
    ]


def test_frontend_and_api_share_an_origin(tmp_path):
    frontend = tmp_path / "frontend"
    (frontend / "assets").mkdir(parents=True)
    (frontend / "index.html").write_text("<html>desktop</html>")
    (frontend / "assets/main.js").write_text("console.log('desktop')")
    with TestClient(create_app(tmp_path / "datasets", frontend_root=frontend)) as client:
        assert client.get("/").text == "<html>desktop</html>"
        assert client.get("/replay").text == "<html>desktop</html>"
        assert client.get("/assets/main.js").status_code == 200
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/does-not-exist").status_code == 404
        assert client.get("/assets/missing.js").status_code == 404
        assert client.post(
            "/api/operations", json={"kind": "collect"},
            headers={"Origin": "https://other.example"},
        ).status_code == 403


def test_desktop_shutdown_waits_for_deletion_transaction(tmp_path):
    manager = OperationManager(tmp_path)
    marker = tmp_path / "committed"
    manager.start("delete", [
        sys.executable, "-c",
        f"import time; from pathlib import Path; print('READY', flush=True); "
        f"time.sleep(0.5); Path({str(marker)!r}).write_text('done')",
    ])
    deadline = time.monotonic() + 5
    while "READY" not in manager.snapshot().logs:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    manager.close(timeout=None)
    assert marker.read_text() == "done"
    assert manager.snapshot().state == "succeeded"


def test_desktop_shutdown_interrupts_collection_and_waits_for_save(tmp_path):
    manager = OperationManager(tmp_path)
    marker = tmp_path / "saved"
    manager.start("collect", [
        sys.executable, "-c",
        "import signal, time, sys; from pathlib import Path; "
        f"signal.signal(signal.SIGINT, lambda *_: (time.sleep(0.2), "
        f"Path({str(marker)!r}).write_text('saved'), sys.exit(0))); "
        "print('READY', flush=True); time.sleep(20)",
    ])
    deadline = time.monotonic() + 5
    while "READY" not in manager.snapshot().logs:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    manager.close(timeout=None)
    assert marker.read_text() == "saved"
    assert manager.snapshot().state == "cancelled"
