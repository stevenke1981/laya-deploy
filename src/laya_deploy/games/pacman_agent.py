"""Policies that pick Pac-Man's next move, plus a headless game loop.

LayaPolicy asks laya one `choice` question per tick: the options are the legal directions,
each described by what happens if Pac-Man goes that way, and the answer carries a probability
per direction, which the UI draws as arrows. HeuristicPolicy produces the same shape of answer
offline and is the baseline laya is measured against.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from ..client import Asker, AskError
from .pacman import INF, OPPOSITE, MoveInfo, PacmanGame

log = logging.getLogger("laya_deploy.games.pacman")

MODEL_SOURCE = "laya"
INSTRUCTIONS = (
    "You are playing Pac-Man. Pick Pac-Man's next move. Eat pellets quickly, never move onto "
    "or next to a dangerous ghost, prefer moves that keep dangerous ghosts far away, and chase "
    "frightened (edible) ghosts while they are close."
)
HEURISTIC_TEMPERATURE = 2.0


@dataclass
class MoveDecision:
    move: str
    probabilities: dict[str, float]
    source: str               # "laya" | "heuristic" | "forced" | "fallback"
    latency_ms: float = 0.0
    cost: float | None = None
    error: str = ""

    @property
    def p(self) -> float:
        return self.probabilities.get(self.move, 0.0)


class Policy(Protocol):
    name: str

    async def decide(self, game: PacmanGame) -> MoveDecision: ...


def describe_move(info: MoveInfo) -> str:
    """One option's criteria text: what happens if Pac-Man moves this way."""
    x, y = info.target
    parts = [f"move {info.direction} to x={x}, y={y}"]
    if info.power:
        parts.append("eats a power pellet")
    elif info.pellet:
        parts.append("eats a pellet")
    elif info.pellet_dist < INF:
        parts.append(f"nearest pellet {info.pellet_dist} steps away")
    if info.ghost_dist == 0:
        parts.append("runs INTO a dangerous ghost")
    elif info.ghost_dist < INF:
        parts.append(f"nearest dangerous ghost {info.ghost_dist} steps away")
    else:
        parts.append("no dangerous ghost can reach")
    if info.edible_dist < INF:
        parts.append(f"edible ghost {info.edible_dist} steps away")
    return ", ".join(parts)


def _forced(game: PacmanGame) -> MoveDecision | None:
    legal = game.legal_moves()
    if len(legal) > 1:
        return None
    move = legal[0] if legal else game.pac_dir
    return MoveDecision(move, {move: 1.0} if legal else {}, "forced")


def heuristic_scores(game: PacmanGame) -> dict[str, float]:
    scores = {}
    for d in game.legal_moves():
        info = game.move_info(d)
        s = 0.0
        if info.ghost_dist == 0:
            s -= 100
        elif info.ghost_dist <= 2:
            s -= 40 / info.ghost_dist
        elif info.ghost_dist <= 4:
            s -= 5
        s += 6 if info.power else 3 if info.pellet else 0
        s -= 0.3 * min(info.pellet_dist, 30)
        if info.edible_dist < INF:
            s += max(0, 8 - info.edible_dist)
        if d == OPPOSITE[game.pac_dir]:
            s -= 0.5  # mild preference for keeping momentum
        scores[d] = s
    return scores


def softmax(scores: dict[str, float], temperature: float = HEURISTIC_TEMPERATURE) -> dict[str, float]:
    if not scores:
        return {}
    top = max(scores.values())
    exp = {k: math.exp((v - top) / temperature) for k, v in scores.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


class HeuristicPolicy:
    name = "HEURISTIC"

    async def decide(self, game: PacmanGame) -> MoveDecision:
        forced = _forced(game)
        if forced:
            return forced
        probs = softmax(heuristic_scores(game))
        return MoveDecision(max(probs, key=probs.__getitem__), probs, "heuristic")


class LayaPolicy:
    """laya picks the move; if the call fails, the heuristic covers that tick."""

    def __init__(self, ask: Asker, label: str = "multilingual", fallback: Policy | None = None) -> None:
        self.ask = ask
        self.name = f"LAYA {label.upper()}"
        self.fallback = fallback if fallback is not None else HeuristicPolicy()

    async def decide(self, game: PacmanGame) -> MoveDecision:
        forced = _forced(game)
        if forced:
            return forced
        legal = game.legal_moves()
        question = {"type": "choice", "instructions": INSTRUCTIONS,
                    "criteria": {d: describe_move(game.move_info(d)) for d in legal}}
        try:
            payload = await self.ask(game.to_text(), {"move": question})
            answer = payload["answers"]["move"]
            choice = answer["choice"]
            if choice not in legal:
                raise AskError(f"laya chose unsupported option {choice!r}")
        except (AskError, KeyError, TypeError) as exc:
            log.warning("laya failed, using fallback: %s", exc)
            d = await self.fallback.decide(game)
            d.source, d.error = "fallback", str(exc)
            return d
        raw = answer.get("probabilities") or {}
        probs = {d: float(raw.get(d, 0.0)) for d in legal}
        cost = (payload.get("usage") or {}).get("cost", 0.0)
        return MoveDecision(choice, probs, MODEL_SOURCE, float(payload.get("latency_ms", 0.0)), cost)


@dataclass
class GameSummary:
    score: int
    steps: int
    won: bool
    lives: int
    model_calls: int = 0
    cost: float = 0.0
    latencies: list = field(default_factory=list)
    fallbacks: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0


StepHook = Callable[[int, MoveDecision, PacmanGame], Awaitable[None] | None]


async def play(game: PacmanGame, policy: Policy, max_steps: int = 600,
               on_step: StepHook | None = None) -> GameSummary:
    """Run the game without a window until it ends or `max_steps` ticks pass."""
    summary = GameSummary(score=0, steps=0, won=False, lives=game.lives)
    while not game.over and summary.steps < max_steps:
        decision = await policy.decide(game)
        if decision.source == MODEL_SOURCE:
            summary.model_calls += 1
            summary.cost += decision.cost or 0.0
            summary.latencies.append(decision.latency_ms)
        elif decision.source == "fallback":
            summary.fallbacks += 1
        game.step(decision.move)
        summary.steps += 1
        if on_step is not None:
            maybe = on_step(summary.steps, decision, game)
            if maybe is not None:
                await maybe
    summary.score, summary.won, summary.lives = game.score, game.won, game.lives
    return summary
