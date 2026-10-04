"""Fixed local commands; HTTP callers cannot provide shell commands or paths."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from replay.backend.dependencies import get_registry
from replay.backend.errors import ReplayError
from replay.backend.operations import ACTIVE_STATES, Operation, OperationManager
from replay.backend.registry import DatasetRegistry
from replay.backend.services.get_dataset import get_dataset

router = APIRouter(prefix="/operations", tags=["operations"])


def get_manager(request: Request) -> OperationManager:
    return request.app.state.operation_manager


Manager = Annotated[OperationManager, Depends(get_manager)]
Registry = Annotated[DatasetRegistry, Depends(get_registry)]


class StartOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["collect", "replay", "delete"]
    dataset_id: str | None = None
    episode_index: int | None = Field(default=None, ge=0, strict=True)


class OperationStatus(BaseModel):
    operation: Operation | None


@router.get("", response_model=OperationStatus)
def status(manager: Manager):
    return OperationStatus(operation=manager.snapshot())


@router.post("", response_model=Operation, status_code=202)
def start(payload: StartOperation, registry: Registry, manager: Manager):
    current = manager.snapshot()
    if current and current.state in ACTIVE_STATES:
        raise ReplayError(409, "已有任务在运行，请等待完成或先停止。")
    if payload.kind == "collect":
        if payload.dataset_id is not None or payload.episode_index is not None:
            raise ReplayError(422, "采集使用 panda.yaml 配置，无需指定回放数据集。")
        return manager.start("collect", ["uv", "run", "vr_collect"])
    if not payload.dataset_id:
        raise ReplayError(422, "请选择数据集。")
    root = registry.get(payload.dataset_id)
    detail = get_dataset(registry, payload.dataset_id)
    indices = [episode.episode_index for episode in detail.episodes]
    if payload.kind == "replay":
        if not indices:
            raise ReplayError(422, "数据集没有已保存的 episode。")
        index = max(indices)
        if payload.episode_index is not None and payload.episode_index != index:
            raise ReplayError(409, "最后保存的 episode 已变化，请刷新数据后重新确认。")
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "replay.scripts.replay_robot",
            "--dataset",
            str(root),
            "--episode-index",
            str(index),
        ]
    else:
        index = payload.episode_index
        if index is None or index not in indices:
            raise ReplayError(404, "待删除的 episode 不存在。")
        if detail.version not in ("v2.0", "v2.1", "v3.0"):
            raise ReplayError(
                422, "删除仅支持 LeRobot v2/v3 数据集。"
            )
        command = [
            "uv",
            "run",
            "episode-delete",
            "--dataset",
            str(root),
            "--episode-index",
            str(index),
            "--yes",
        ]
    return manager.start(payload.kind, command, dataset_id=payload.dataset_id, episode_index=index)


@router.post("/{operation_id}/stop", response_model=Operation)
def stop(operation_id: str, manager: Manager):
    return manager.stop(operation_id)
