"""Run from repository root: uv run uvicorn replay.backend.main:app."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from replay.backend.api import datasets, ee, episodes, health, streams, video
from replay.backend.config import configured_data_root
from replay.backend.errors import ReplayError
from replay.backend.registry import DatasetRegistry


def create_app(data_root: Path | None = None) -> FastAPI:
    application = FastAPI(title="Franka Replay", version="0.1.0")
    application.state.dataset_registry = DatasetRegistry(configured_data_root(data_root))

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
    ):
        application.include_router(router, prefix="/api")
    return application


app = create_app()
