from fastapi import Request

from replay.backend.registry import DatasetRegistry


def get_registry(request: Request) -> DatasetRegistry:
    return request.app.state.dataset_registry
