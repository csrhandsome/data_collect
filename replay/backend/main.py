"""Run from repository root: uv run uvicorn replay.backend.main:app."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from replay.backend.api import datasets, ee, episodes, health, operations, streams, video
from replay.backend.config import configured_data_root
from replay.backend.errors import ReplayError
from replay.backend.operations import OperationManager
from replay.backend.registry import DatasetRegistry


def create_app(
    data_root: Path | None = None, operation_manager: OperationManager | None = None
) -> FastAPI:
    manager = operation_manager or OperationManager()

    @asynccontextmanager
    async def lifespan(_application):
        yield
        await asyncio.to_thread(manager.close)

    application = FastAPI(title="Franka Replay", version="0.1.0", lifespan=lifespan)
    application.state.dataset_registry = DatasetRegistry(configured_data_root(data_root))
    application.state.operation_manager = manager

    @application.middleware("http")
    async def same_origin_writes(request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse(
                    status_code=403, content={"detail": "操作请求必须来自本工作台页面。"}
                )
        return await call_next(request)

    @application.exception_handler(ReplayError)
    async def replay_error(_request: Request, exc: ReplayError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    for router in (
        health.router,
        datasets.router,
        episodes.router,
        ee.router,
        video.router,
        streams.router,
        operations.router,
    ):
        application.include_router(router, prefix="/api")
    return application


app = create_app()
