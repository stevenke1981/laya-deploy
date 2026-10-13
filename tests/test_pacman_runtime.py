"""A model test must stop, rather than keep playing with a substitute policy."""
import asyncio
from dataclasses import replace

import pytest

from laya_deploy.client import AskError
from laya_deploy.games.pacman import PacmanGame
from laya_deploy.games.pacman_agent import MoveDecision
from laya_deploy.games.pacman_runtime import RequiredModelPolicy


class Policy:
    name = "LAYA TEST"

    def __init__(self, source="laya"):
        self.decision = MoveDecision("left", {"left": 0.7, "right": 0.3}, source, error="offline")
        self.calls = 0

    async def decide(self, game):
        self.calls += 1
        return self.decision


class Asker:
    def __init__(self, response=None):
        self.response = response
        self.calls = 0
        self.loop = None

    async def __call__(self, state, questions):
        self.calls += 1
        self.loop = asyncio.get_running_loop()
        if isinstance(self.response, Exception):
            raise self.response
        if self.response is not None:
            return self.response
        return {"answers": {"ready": {"choice": "safe", "probabilities": {"safe": 0.9, "dangerous": 0.1}}}}


def test_readiness_and_decisions_share_loop_and_probe_only_once():
    asker, inner = Asker(), Policy()
    policy = RequiredModelPolicy(inner, asker)

    async def go():
        for _ in range(2):
            assert (await policy.decide(PacmanGame())).source == "laya"
            assert asker.loop is asyncio.get_running_loop()

    asyncio.run(go())
    assert asker.calls == 1 and inner.calls == 2 and policy.name == inner.name


@pytest.mark.parametrize("response", [
    AskError("connection refused"), {}, [],
    {"routing": {"reason": "fake engine"}},
    {"answers": {"ready": {"choice": "safe", "probabilities": {"safe": 1}}}},
    {"answers": {"ready": {"choice": "other", "probabilities": {"safe": 0.9, "dangerous": 0.1}}}},
    {"answers": {"ready": {"choice": "safe", "probabilities": {"safe": float("nan"), "dangerous": 0.1}}}},
    {"answers": {"ready": {"choice": "safe", "probabilities": {"safe": -0.1, "dangerous": 1.1}}}},
    {"answers": {"ready": {"choice": "safe", "probabilities": {"safe": 0, "dangerous": 0}}}},
])
def test_bad_readiness_never_runs_game_policy(response):
    inner = Policy()
    with pytest.raises(AskError):
        asyncio.run(RequiredModelPolicy(inner, Asker(response)).decide(PacmanGame()))
    assert inner.calls == 0


def test_fallback_stops_and_retry_checks_model_again():
    inner, asker = Policy("fallback"), Asker()
    policy = RequiredModelPolicy(inner, asker)

    async def go():
        with pytest.raises(AskError, match="offline"):
            await policy.decide(PacmanGame())
        inner.decision = replace(inner.decision, source="laya")
        await policy.decide(PacmanGame())

    asyncio.run(go())
    assert asker.calls == 2


@pytest.mark.parametrize("source", ["cache", "forced"])
def test_legitimate_non_inference_moves_remain_available(source):
    assert asyncio.run(RequiredModelPolicy(Policy(source), Asker()).decide(PacmanGame())).source == source


@pytest.mark.parametrize("decision", [
    MoveDecision("up", {"up": 1}, "laya"),
    MoveDecision("left", {"left": float("nan")}, "laya"),
    MoveDecision("left", {"left": 1}, "heuristic"),
])
def test_invalid_runtime_decision_stops(decision):
    inner = Policy()
    inner.decision = decision
    with pytest.raises(AskError):
        asyncio.run(RequiredModelPolicy(inner, Asker()).decide(PacmanGame()))
