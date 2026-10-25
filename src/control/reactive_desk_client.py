"""Optional Reactive Desk transport, shared by collection and inference."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import websockets
import websockets.sync.client
from websockets.exceptions import InvalidHandshake


class ReactiveDeskVlaClient:
    """Minimal Reactive Desk websocket client compatible with the openpi sender."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        path: str = "/ws/VlaIngest",
        *,
        enabled: bool = True,
    ) -> None:
        self._uri = self._build_ws_uri(host, port, path)
        self._enabled = enabled
        self._ws: websockets.sync.client.ClientConnection | None = None

    def connect(self) -> None:
        if not self._enabled or self._ws is not None:
            return

        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            open_timeout=1,
        )
        self._ws.recv(timeout=1)

    def send_predictions(
        self,
        xyz: np.ndarray,
        probabilities: np.ndarray,
        *,
        prompt: str = "",
        is_executing: bool = True,
    ) -> bool:
        if not self._enabled:
            return False

        xyz = np.asarray(xyz, dtype=np.float32)
        probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)

        predictions = [
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]) if point.shape[0] > 2 else 0.0,
                "probability": float(probabilities[rank]) if rank < probabilities.shape[0] else 1.0,
                "rank": int(rank),
            }
            for rank, point in enumerate(xyz)
        ]
        payload = {
            "type": "vla_predictions",
            "predictions": predictions,
            "is_executing": is_executing,
            "current_prompt": prompt,
        }

        try:
            self.connect()
            if self._ws is None:
                return False
            self._ws.send(json.dumps(payload))
            self._ws.recv(timeout=1)
            return True
        except (websockets.ConnectionClosed, InvalidHandshake, OSError, TimeoutError):
            self.close()
            return False

    def close(self) -> None:
        if self._ws is None:
            return
        self._ws.close()
        self._ws = None

    @staticmethod
    def _build_ws_uri(host: str, port: int, path: str) -> str:
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        if path:
            path = path if path.startswith("/") else f"/{path}"
            if not uri.endswith(path):
                uri = f"{uri.rstrip('/')}{path}"
        return uri


class ScenePublisher:
    """A bounded worker keeps optional scene transport off the servo loop."""

    def __init__(self, config, dry_run=False):
        cfg = config.get("reactive_desk", {})
        self.client = ReactiveDeskVlaClient(
            host=cfg.get("reactive_desk_host", "127.0.0.1"),
            port=int(cfg.get("reactive_desk_port", 8000)),
            path=cfg.get("reactive_desk_path", "/ws/VlaIngest"),
            enabled=cfg.get("reactive_desk_enabled", False) and not dry_run,
        )
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene")
        self.future = None
        self.last_ns = 0
        self.period_ns = round(1e9 / float(cfg.get("publish_hz", 10)))

    def publish(self, position, now_ns, prompt):
        if not self.client._enabled or now_ns - self.last_ns < self.period_ns:
            return
        if self.future is not None and not self.future.done():
            return
        if self.future is not None:
            self.future.result()
        self.last_ns = now_ns
        self.future = self.pool.submit(
            self.client.send_predictions, np.asarray(position).copy(), np.ones(1), prompt=prompt
        )

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.client.close()
