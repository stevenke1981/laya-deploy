"""Strict model readiness and move validation for the Pac-Man test commands."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..client import Asker, AskError
from .pacman import PacmanGame
from .pacman_agent import MoveDecision, Policy

READY_QUESTION = {
    "type": "choice",
    "instructions": "Is this Pac-Man move safe or dangerous?",
    "criteria": {"safe": "safe, no ghost nearby", "dangerous": "deadly, runs into a ghost"},
}


def _validate_answers(payload: Any, questions: Mapping[str, Any]) -> None:
    try:
        if (payload.get("routing") or {}).get("reason") == "fake engine":
            raise AskError("The server is using a fake engine. Start `laya-deploy serve` without --fake.")
        for key, question in questions.items():
            answer = payload["answers"][key]
            probabilities = answer["probabilities"]
            if set(probabilities) != set(question["criteria"]) or answer["choice"] not in probabilities:
                raise ValueError(f"{key}: answer has unexpected choices")
            values = tuple(float(p) for p in probabilities.values())
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in values) or not math.isclose(sum(values), 1, abs_tol=0.01):
                raise ValueError(f"{key}: probabilities are invalid")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise AskError(f"Invalid Laya answer: {exc}") from exc


class ValidatedAsker:
    """Validate raw choice answers before policies can transform or cache them."""

    def __init__(self, ask: Asker) -> None:
        self.ask = ask

    async def __call__(self, state: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
        payload = await self.ask(state, questions)
        _validate_answers(payload, questions)
        return payload


async def check_model(ask: Asker) -> None:
    """Exercise inference, not merely HTTP liveness; run on the gameplay event loop."""
    await ValidatedAsker(ask)({"subject": "safe, no ghost nearby"}, {"ready": READY_QUESTION})


class RequiredModelPolicy:
    """Keep library fallback behavior, but never count it as a successful model test."""

    def __init__(self, policy: Policy, ask: Asker) -> None:
        self.policy, self.ask = policy, ask
        self.name = policy.name
        self._ready = False

    async def decide(self, game: PacmanGame) -> MoveDecision:
        try:
            if not self._ready:
                await check_model(self.ask)
                self._ready = True
            decision = await self.policy.decide(game)
            if decision.source not in {"laya", "cache", "forced"}:
                raise AskError(decision.error or "Laya returned a substitute policy decision")
            if decision.move not in game.legal_moves():
                raise AskError(f"Laya returned an illegal move: {decision.move!r}")
            values = tuple(float(p) for p in decision.probabilities.values())
            if not values or not all(math.isfinite(p) and 0 <= p <= 1 for p in values):
                raise AskError("Laya returned invalid move probabilities")
            return decision
        except (AskError, TypeError, ValueError, KeyError) as exc:
            self._ready = False
            raise AskError(f"Laya model test stopped: {exc}") from exc
