"""Server tests with the fake engine: no model download, no GPU."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from laya_deploy.config import Settings, resolve_checkpoint
from laya_deploy.engine import EngineError, FakeEngine
from laya_deploy.server import create_app

QUESTIONS = {
    "pick": {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": "second"}},
    "yes": {"type": "noul", "instructions": "Is it?"},
    "lvl": {"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high"]},
}


def client(**kw) -> tuple[TestClient, FakeEngine]:
    engine = FakeEngine()
    return TestClient(create_app(Settings(**kw), engine)), engine


def test_decisions_endpoint_returns_openrouter_shape():
    c, engine = client()
    r = c.post("/api/alpha/decisions", json={"model": "microsoft/microsoft-decision-1",
                                             "state": "hello", "questions": QUESTIONS})
    assert r.status_code == 200
    body = r.json()
    assert body["answers"]["pick"]["choice"] == "a"
    assert body["answers"]["yes"]["noul"] == 0.5
    assert set(body["answers"]["lvl"]["probabilities"]) == {"0", "1", "2"}
    assert body["usage"]["cost"] == 0.0 and body["id"].startswith("laya-")
    # a non-laya model id falls back to the default checkpoint
    assert engine.calls[0]["model"] == "multilingual"


def test_v1_alias_and_explicit_checkpoint():
    c, engine = client()
    r = c.post("/v1/decisions", json={"model": "laya/english", "state": {"x": 1}, "questions": QUESTIONS})
    assert r.status_code == 200
    assert engine.calls[0]["model"] == "english"
    assert r.json()["model"] == "laya/english"


@pytest.mark.parametrize("body,fragment", [
    ({"questions": QUESTIONS}, "state is required"),
    ({"state": "s", "questions": {}}, "non-empty"),
    ({"state": "s", "questions": {"q": {"type": "maybe", "instructions": "x"}}}, "type must be"),
    ({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": ""}}}},
     "at least two options"),
    ({"state": "s", "questions": {"q": {"type": "score", "instructions": "x", "criteria": "abc"}}},
     "at least two levels"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": " "}}}, "instructions"),
    ({"state": 5, "questions": QUESTIONS}, "state must be"),
])
def test_invalid_requests_are_400(body, fragment):
    c, engine = client()
    r = c.post("/api/alpha/decisions", json=body)
    assert r.status_code == 400
    assert fragment in r.json()["error"]["message"]
    assert engine.calls == []


def test_invalid_json_is_400():
    c, _ = client()
    r = c.post("/api/alpha/decisions", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_json"


def test_bearer_auth_when_key_set():
    c, _ = client(api_key="s3cret")
    body = {"state": "s", "questions": QUESTIONS}
    assert c.post("/api/alpha/decisions", json=body).status_code == 401
    bad = c.post("/api/alpha/decisions", json=body, headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401
    ok = c.post("/api/alpha/decisions", json=body, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_engine_failure_is_500():
    class Broken(FakeEngine):
        def predict(self, *a, **k):
            raise EngineError("CUDA out of memory")

    c = TestClient(create_app(Settings(), Broken()))
    r = c.post("/api/alpha/decisions", json={"state": "s", "questions": QUESTIONS})
    assert r.status_code == 500 and "out of memory" in r.json()["error"]["message"]


def test_health():
    c, _ = client()
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["engine"] == "fake"


@pytest.mark.parametrize("requested,expected", [
    (None, "multilingual"),
    ("microsoft/microsoft-decision-1", "multilingual"),
    ("english", "english"),
    ("laya/typed-decisions", "typed-decisions"),
    ("convaiinnovations/laya", "english"),
    ("convaiinnovations/laya:multilingual", "multilingual"),
    ("AUTO", "auto"),
])
def test_resolve_checkpoint(requested, expected):
    assert resolve_checkpoint(requested, "multilingual") == expected
