"""Read the ROS monitor's shared snapshot without importing ROS in the web server."""

from fastapi import APIRouter, Response

from control.ros2.health_store import read_health

router = APIRouter(prefix="/ros2", tags=["ros2"])


@router.get("/health")
def ros2_health(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return read_health()
