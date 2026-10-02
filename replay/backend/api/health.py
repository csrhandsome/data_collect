from typing import Literal

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, Literal["ok"]]:
    return {"status": "ok"}
