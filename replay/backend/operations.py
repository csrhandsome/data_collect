"""One supervised local task at a time, with bounded logs and cooperative stopping."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from replay.backend.errors import ReplayError

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_STATES = {"running", "stopping"}


class Operation(BaseModel):
    id: str
    kind: Literal["collect", "replay", "delete"]
    state: Literal["running", "stopping", "succeeded", "failed", "cancelled"]
    dataset_id: str | None = None
    episode_index: int | None = None
    started_at: str
    finished_at: str | None = None
    return_code: int | None = None
    logs: list[str] = Field(default_factory=list)


class OperationManager:
    def __init__(
        self,
        repo_root: Path = REPO_ROOT,
        *,
        runtime_command: list[str] | None = None,
        config_path: Path | None = None,
        simulate: bool = False,
        max_steps: int = 60,
    ):
        self.repo_root = repo_root
        self.runtime_command = runtime_command
        self.config_path = config_path
        self.simulate = simulate
        self.max_steps = max_steps
        self._lock = threading.RLock()
        self._operation: Operation | None = None
        self._process: subprocess.Popen | None = None
        self._logs: deque[str] = deque(maxlen=80)
        self._thread: threading.Thread | None = None
        self._closed = False

    def command(self, kind: str, args: list[str] | None = None) -> list[str]:
        if self.runtime_command:
            command = [*self.runtime_command, kind]
            if kind != "delete" and self.config_path:
                command.extend(["--config", str(self.config_path)])
        else:
            command = {
                "collect": ["uv", "run", "vr_collect"],
                "replay": ["uv", "run", "python", "-m", "replay.scripts.replay_robot"],
                "delete": ["uv", "run", "episode-delete"],
            }[kind].copy()
        command.extend(args or [])
        if self.simulate and kind in ("collect", "replay"):
            command.append("--dry-run")
            if kind == "collect":
                command.extend(["--max-steps", str(self.max_steps)])
        return command

    def snapshot(self) -> Operation | None:
        with self._lock:
            return (
                self._operation.model_copy(update={"logs": list(self._logs)})
                if self._operation
                else None
            )

    def start(self, kind, command, *, dataset_id=None, episode_index=None) -> Operation:
        with self._lock:
            if self._closed:
                raise ReplayError(503, "服务正在关闭，无法启动任务。")
            if self._operation and self._operation.state in ACTIVE_STATES:
                raise ReplayError(409, "已有任务在运行，请等待完成或先停止。")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            env.pop("PYTHONPATH", None)
            try:
                self._process = subprocess.Popen(
                    command,
                    cwd=self.repo_root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    start_new_session=True,
                )
            except OSError as exc:
                raise ReplayError(503, "无法启动任务，请检查程序安装及运行日志。") from exc
            self._logs.clear()
            self._operation = Operation(
                id=uuid.uuid4().hex,
                kind=kind,
                state="running",
                dataset_id=dataset_id,
                episode_index=episode_index,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            self._thread = threading.Thread(target=self._watch, daemon=True, name="workspace-task")
            self._thread.start()
            return self.snapshot()

    def _watch(self):
        process = self._process
        assert process is not None and process.stdout is not None
        try:
            for line in process.stdout:
                with self._lock:
                    self._logs.append(line.rstrip()[-1500:])
            code = process.wait()
            with self._lock:
                operation = self._operation
                assert operation is not None
                stopping = operation.state == "stopping"
                operation.state = (
                    "cancelled"
                    if stopping and code in (0, -signal.SIGINT, 130)
                    else "succeeded"
                    if code == 0
                    else "failed"
                )
                operation.return_code = code
                operation.finished_at = datetime.now(timezone.utc).isoformat()
        finally:
            process.stdout.close()

    def stop(self, operation_id: str) -> Operation:
        with self._lock:
            operation = self._operation
            if operation is None or operation.id != operation_id:
                raise ReplayError(404, "任务不存在或已被更新。")
            if operation.kind == "delete" and operation.state in ACTIVE_STATES:
                raise ReplayError(409, "删除正在更新数据和编号，请等待完成。")
            if operation.state == "running":
                operation.state = "stopping"
                self._logs.append("正在请求停止；等待设备关闭及文件保存…")
                try:
                    os.killpg(self._process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            return self.snapshot()

    def close(self, timeout: float | None = 30):
        with self._lock:
            self._closed = True
            if self._operation and self._operation.kind != "delete":
                self.stop(self._operation.id)
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
