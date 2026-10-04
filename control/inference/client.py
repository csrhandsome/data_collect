"""Generic obs→action websocket client. The neural net stays on the GPU server."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

import websockets.exceptions
import websockets.sync.client

from control.util import msgpack_numpy

logger = logging.getLogger(__name__)


class PolicyClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8001,
        *,
        connect_timeout_s=10.0,
        request_timeout_s=5.0,
    ) -> None:
        if host.startswith(("ws://", "wss://")):
            self._uri = host
            if port is not None and ":" not in host.rstrip("/").split("://", 1)[-1]:
                self._uri = f"{host}:{port}"
        else:
            self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._request_timeout = request_timeout_s
        self._connect_timeout = connect_timeout_s
        self._ws, self.server_metadata = self._wait_for_server(timeout=connect_timeout_s)

    def _connect(
        self,
    ) -> tuple[websockets.sync.client.ClientConnection, dict[str, Any]]:
        try:
            conn = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                open_timeout=10,
                ping_timeout=300,
            )
        except TypeError:
            conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
        try:
            metadata = msgpack_numpy.unpackb(conn.recv(timeout=self._request_timeout))
            if not isinstance(metadata, dict):
                raise ValueError("Policy metadata must be a dictionary")
        except BaseException:
            conn.close()
            raise
        return conn, metadata

    def _wait_for_server(
        self, timeout: float | None = None
    ) -> tuple[websockets.sync.client.ClientConnection, dict[str, Any]]:
        logger.info("Waiting for policy server at %s", self._uri)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                conn, metadata = self._connect()
                logger.info("Connected to policy server %s", self._uri)
                return conn, metadata
            except (ConnectionRefusedError, OSError, TimeoutError):
                if deadline is not None and time.monotonic() >= deadline:
                    raise ConnectionRefusedError(
                        f"Timed out waiting for policy server at {self._uri}"
                    ) from None
                logger.info("Still waiting for %s", self._uri)
                time.sleep(1.0)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        data = self._packer.pack(obs)
        try:
            self._ws.send(data)
            response = self._ws.recv(timeout=self._request_timeout)
        except websockets.exceptions.ConnectionClosed:
            self._ws, metadata = self._wait_for_server(timeout=self._connect_timeout)
            if metadata != self.server_metadata:
                self.close()
                raise ValueError("Policy metadata changed after reconnect")
            self._ws.send(data)
            response = self._ws.recv(timeout=self._request_timeout)
        if isinstance(response, str):
            # A string response is an error message from the server protocol.
            raise RuntimeError(f"Policy server error:\n{response}")  # noqa: TRY004
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._ws.close()
