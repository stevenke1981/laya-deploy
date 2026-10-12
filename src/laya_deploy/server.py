"""FastAPI app: an OpenRouter-Decisions-compatible front for a local laya.

    POST /api/alpha/decisions   same wire format as https://openrouter.ai/api/alpha/decisions
    POST /v1/decisions          alias
    GET  /health                liveness + device + loaded checkpoints

Point decision-cua at it with
    DCUA_DECISION_URL=http://127.0.0.1:8100/api/alpha/decisions
"""
from __future__ import annotations

import hmac
import time
import uuid
from collections.abc import Mapping
from typing import Any

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Settings, resolve_checkpoint
from .engine import Engine, EngineError

QUESTION_TYPES = {"choice", "noul", "score"}
MAX_QUESTIONS = 64
MAX_BODY_BYTES = 1_000_000


def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "code": code}}, status_code=status)


def validate_request(body: Any) -> str | None:
    """Return an error message for a malformed request, or None if it is usable."""
    if not isinstance(body, dict):
        return "body must be a JSON object"
    if body.get("state") is None:
        return "state is required"
    if not isinstance(body["state"], (str, dict, list)):
        return "state must be a string, object or array"
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        return "questions must be a non-empty object"
    if len(questions) > MAX_QUESTIONS:
        return f"at most {MAX_QUESTIONS} questions per request"
    for key, q in questions.items():
        err = _validate_question(q)
        if err:
            return f"questions.{key}: {err}"
    return None


def _validate_question(q: Any) -> str | None:
    if not isinstance(q, dict):
        return "must be an object"
    kind = q.get("type")
    if kind not in QUESTION_TYPES:
        return f"type must be one of {sorted(QUESTION_TYPES)}"
    if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
        return "instructions must be a non-empty string"
    criteria = q.get("criteria")
    if kind == "choice" and (not isinstance(criteria, dict) or len(criteria) < 2):
        return "choice criteria must be an object with at least two options"
    if kind == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
        return "score criteria must be a list with at least two levels"
    if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
        return "noul criteria, if given, must be an object"
    return None


def to_decisions_response(result: Mapping[str, Any], checkpoint: str, label: str, latency_ms: float) -> dict[str, Any]:
    """Shape a laya result like an OpenRouter Decisions response (cost is always 0 locally)."""
    usage = dict(result.get("usage") or {})
    routing = result.get("routing") or {}
    served = routing.get("model") or checkpoint
    return {
        "id": f"laya-{uuid.uuid4().hex[:16]}",
        "model": f"{label}/{served}",
        "provider": "local",
        "answers": result.get("answers", {}),
        "usage": {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": 0,
            "cost": 0.0,
            "truncated": bool(usage.get("truncated", False)),
        },
        "routing": routing,
        "latency_ms": round(latency_ms, 1),
    }


def create_app(settings: Settings, engine: Engine) -> FastAPI:
    app = FastAPI(title="laya-deploy", version="0.1.0", docs_url=None, redoc_url=None)

    def authorized(request: Request) -> bool:
        if not settings.api_key:
            return True
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else ""
        return hmac.compare_digest(token.encode(), settings.api_key.encode())

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "default_model": settings.default_model, **engine.info()}

    async def decide(request: Request) -> JSONResponse:
        if not authorized(request):
            return _error(401, "missing or invalid bearer token", "unauthorized")
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return _error(413, f"body larger than {MAX_BODY_BYTES} bytes", "payload_too_large")
        try:
            body = await request.json() if raw else None
        except ValueError:
            return _error(400, "body is not valid JSON", "invalid_json")
        err = validate_request(body)
        if err:
            return _error(400, err, "invalid_request")
        checkpoint = resolve_checkpoint(body.get("model"), settings.default_model)
        t0 = time.perf_counter()
        try:
            result = await anyio.to_thread.run_sync(
                engine.predict, body["state"], body["questions"], checkpoint, settings.max_len
            )
        except ValueError as exc:
            return _error(400, str(exc), "invalid_request")
        except EngineError as exc:
            return _error(500, str(exc), "inference_failed")
        latency = (time.perf_counter() - t0) * 1000
        return JSONResponse(to_decisions_response(result, checkpoint, settings.model_label, latency))

    app.add_api_route("/api/alpha/decisions", decide, methods=["POST"])
    app.add_api_route("/v1/decisions", decide, methods=["POST"])
    return app
