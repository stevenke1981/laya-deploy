"""Ways to ask laya a question, all returning the Decisions-wire response dict.

- `HttpAsker`: POST to a running laya-deploy server (or any Decisions-compatible URL).
- `EngineAsker`: call an in-process engine directly (no server, no HTTP).

Both are async callables: `await ask(state, questions) -> dict` with an `answers` key.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Protocol

import anyio
import httpx

from .engine import Engine

DEFAULT_URL = "http://127.0.0.1:8100/api/alpha/decisions"


class AskError(RuntimeError):
    """The decision call failed (transport, HTTP status, or engine error)."""


class Asker(Protocol):
    async def __call__(self, state: Any, questions: Mapping[str, Any]) -> dict[str, Any]: ...


class HttpAsker:
    def __init__(self, url: str = DEFAULT_URL, model: str = "multilingual", api_key: str = "",
                 timeout: float = 30.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.url, self.model = url, model
        self._http = httpx.AsyncClient(timeout=timeout, headers=headers, transport=transport)

    async def __call__(self, state: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            resp = await self._http.post(self.url, json={"model": self.model, "state": state,
                                                         "questions": dict(questions)})
        except httpx.TransportError as exc:
            raise AskError(f"cannot reach {self.url}: {exc!r}") from exc
        if resp.status_code >= 400:
            raise AskError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise AskError("server returned invalid JSON") from exc
        payload.setdefault("latency_ms", round((time.perf_counter() - t0) * 1000, 1))
        return payload

    async def aclose(self) -> None:
        await self._http.aclose()


class EngineAsker:
    def __init__(self, engine: Engine, model: str = "multilingual") -> None:
        self.engine, self.model = engine, model

    async def __call__(self, state: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            result = await anyio.to_thread.run_sync(self.engine.predict, state, dict(questions), self.model)
        except Exception as exc:
            raise AskError(f"{type(exc).__name__}: {exc}") from exc
        return {**result, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    async def aclose(self) -> None:
        return None
