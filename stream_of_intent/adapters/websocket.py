from __future__ import annotations

import asyncio
import threading
from typing import Iterator, Set

import websockets
import websockets.server

from stream_of_intent.adapters.base import BaseAdapter, logger
from stream_of_intent.config import StreamConfig
from stream_of_intent.types import Event


class WebSocketAdapter(BaseAdapter):
    """Runs an async WebSocket server and fans out events to every client.

    The server is started in a background thread so :meth:`stream` can
    block on the (synchronous) event iterator in the main thread.
    """

    def __init__(self, config: StreamConfig) -> None:
        super().__init__(config)
        self._clients: Set[websockets.server.ServerConnection] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server_thread: threading.Thread | None = None
        self._server: websockets.server.Server | None = None

    def connect(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._server_thread = threading.Thread(
            target=self._run_server, daemon=True,
        )
        self._server_thread.start()
        logger.info(
            "WebSocketAdapter: server listening on ws://%s:%d",
            self.config.host, self.config.port,
        )

    def send(self, event: Event) -> None:
        payload = self._serialize_event(event)
        if self._loop is None or self._loop.is_closed():
            return
        future = asyncio.run_coroutine_threadsafe(
            self._broadcast(payload), self._loop,
        )
        future.result(timeout=5)

    def disconnect(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
        logger.info("WebSocketAdapter: server stopped.")

    def _run_server(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())
        self._loop.run_forever()

    async def _start_server(self) -> None:
        self._server = await websockets.server.serve(
            self._handler, self.config.host, self.config.port,
        )

    async def _handler(
        self, ws: websockets.server.ServerConnection,
    ) -> None:
        self._clients.add(ws)
        logger.debug("WebSocket client connected (%d total).", len(self._clients))
        try:
            async for _ in ws:
                pass  # keep connection alive; we only send
        finally:
            self._clients.discard(ws)
            logger.debug("WebSocket client disconnected (%d total).", len(self._clients))

    async def _broadcast(self, payload: str) -> None:
        if not self._clients:
            return
        websockets.broadcast(self._clients, payload)
