"""Inference engines behind the HTTP server.

`LayaEngine` wraps `laya.Router` (torch + transformers, imported lazily so the
server, CLI and tests load without the model stack). `FakeEngine` answers with
deterministic probabilities for tests and offline wiring checks.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from typing import Any, Protocol

from .config import AUTO, Settings


class EngineError(RuntimeError):
    """Inference failed in a way the caller should see as HTTP 500."""


class Engine(Protocol):
    def predict(self, state: Any, questions: Mapping[str, Any], model: str, max_len: int = 0) -> dict[str, Any]: ...

    def info(self) -> dict[str, Any]: ...


class LayaEngine:
    """One Router per process. GPU forward passes are serialized by a lock."""

    def __init__(self, settings: Settings) -> None:
        os.environ.setdefault("USE_TF", "0")  # TF's runtime can deadlock model construction
        try:
            from laya import Router
        except ImportError as exc:  # pragma: no cover - depends on the optional model stack
            raise EngineError("laya is not installed: run `uv sync --extra model`") from exc
        import torch

        self.device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu = torch.cuda.get_device_name(0) if self.device.startswith("cuda") else ""
        self._router = Router(
            device=self.device,
            default="multilingual" if settings.default_model == AUTO else settings.default_model,
            max_loaded=settings.max_loaded,
        )
        self._lock = threading.Lock()
        self.load_ms: dict[str, int] = {}
        for name in settings.preload:
            t0 = time.perf_counter()
            self._router.load(name)
            self.load_ms[name] = round((time.perf_counter() - t0) * 1000)

    def predict(self, state: Any, questions: Mapping[str, Any], model: str, max_len: int = 0) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if model != AUTO:
            kwargs["model"] = model
        if max_len:
            kwargs["max_len"] = max_len
        with self._lock:
            try:
                return self._router.predict(state, dict(questions), **kwargs)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(str(exc)) from exc
            except Exception as exc:  # torch / CUDA failures
                raise EngineError(f"{type(exc).__name__}: {exc}") from exc

    def info(self) -> dict[str, Any]:
        return {
            "engine": "laya",
            "device": self.device,
            "gpu": self.gpu,
            "loaded": list(self._router.loaded),
            "load_ms": dict(self.load_ms),
        }


class FakeEngine:
    """Deterministic stand-in: first option wins, noul = 0.5. No model, no GPU."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, state: Any, questions: Mapping[str, Any], model: str, max_len: int = 0) -> dict[str, Any]:
        self.calls.append({"state": state, "questions": dict(questions), "model": model, "max_len": max_len})
        answers: dict[str, Any] = {}
        for key, q in questions.items():
            kind = q.get("type")
            criteria = q.get("criteria")
            if kind == "choice":
                labels = list(criteria)
                rest = 0.3 / max(1, len(labels) - 1)
                probs = {lab: (0.7 if i == 0 else rest) for i, lab in enumerate(labels)}
                answers[key] = {"type": "choice", "choice": labels[0], "probabilities": probs, "confidence": 0.7}
            elif kind == "score":
                n = len(criteria)
                probs = {str(i): 1.0 / n for i in range(n)}
                answers[key] = {"type": "score", "score": (n - 1) / 2, "probabilities": probs, "confidence": 1.0 / n}
            else:
                answers[key] = {"type": "noul", "noul": 0.5, "confidence": 0.5}
        return {
            "model": "laya-fake",
            "answers": answers,
            "usage": {"input_tokens": len(str(state)) // 4, "output_tokens": 0},
            "routing": {"model": "multilingual" if model == AUTO else model, "reason": "fake engine"},
        }

    def info(self) -> dict[str, Any]:
        return {"engine": "fake", "device": "none", "gpu": "", "loaded": [], "load_ms": {}}
