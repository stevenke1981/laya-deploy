"""Pac-Man played in laya decision mode (see `laya_deploy.mode`).

Each move is reduced to two short phrases, a danger phrase and a food phrase. laya judges
each danger phrase alone (safe vs dangerous) and matches the food phrases against the wanted
move; the code keeps the safest moves, then the best food, then breaks ties by keeping the
heading. Phrases repeat, so after a few dozen calls almost every judgment comes from the cache.
"""
from __future__ import annotations

import logging

from ..client import AskError
from ..mode import Judge
from .pacman import OPPOSITE, MoveInfo, PacmanGame
from .pacman_agent import MODEL_SOURCE, HeuristicPolicy, MoveDecision, _forced

log = logging.getLogger("laya_deploy.games.pacman")

SAFE = ("safe", "safe, no ghost nearby")
DANGEROUS = ("dangerous", "deadly or dangerous, a ghost is very close")
WANTED = "eats a pellet or chases an edible ghost"


def danger_phrase(info: MoveInfo) -> str:
    g = info.ghost_dist
    if g == 0:
        return "deadly, runs into a ghost"
    if g <= 2:
        return "dangerous, ghost very close"
    if g <= 4:
        return "risky, ghost nearby"
    return "safe, no ghost nearby"


def food_phrase(info: MoveInfo) -> str:
    if info.edible_dist <= 6:
        return "chases an edible ghost"
    if info.power:
        return "eats a power pellet"
    if info.pellet:
        return "eats a pellet"
    if info.pellet_dist <= 3:
        return "pellet close ahead"
    if info.pellet_dist <= 8:
        return "pellet a few steps away"
    return "no pellet nearby"


class NativeLayaPolicy:
    """laya judges short phrases; the code owns ordering and tie-breaks."""

    def __init__(self, judge: Judge, label: str = "english", tolerance: float = 0.0) -> None:
        self.judge = judge
        self.tolerance = tolerance
        self.name = f"LAYA {label.upper()} NATIVE"

    async def decide(self, game: PacmanGame) -> MoveDecision:
        forced = _forced(game)
        if forced:
            return forced
        try:
            return await self._decide(game)
        except (AskError, KeyError, TypeError) as exc:
            log.warning("laya failed, using fallback: %s", exc)
            d = await HeuristicPolicy().decide(game)
            d.source, d.error = "fallback", str(exc)
            return d

    async def _decide(self, game: PacmanGame) -> MoveDecision:
        calls_before, latency_before = self.judge.calls, self.judge.latency_ms
        legal = game.legal_moves()
        info = {d: game.move_info(d) for d in legal}

        keep, p_safe = await self.judge.filter(
            {d: danger_phrase(info[d]) for d in legal}, SAFE, DANGEROUS,
            question="Is this Pac-Man move safe or dangerous?", tolerance=self.tolerance)

        foods = {d: food_phrase(info[d]) for d in keep}
        distinct = sorted(set(foods.values()))
        if len(distinct) > 1:
            options = {f"o{i}": text for i, text in enumerate(distinct)}
            verdict = await self.judge.match(WANTED, options, min_p=0.0, field="wanted_move",
                                             question="Which move matches the wanted move?")
            pref = {text: verdict.probabilities.get(f"o{i}", 0.0) for i, text in enumerate(distinct)}
            best = max(pref[foods[d]] for d in keep)
            keep = [d for d in keep if pref[foods[d]] >= best]

        def tie_break(d: str) -> tuple:
            return (d == game.pac_dir, d != OPPOSITE[game.pac_dir], -info[d].pellet_dist)

        move = max(keep, key=tie_break)
        total = sum(p_safe.values()) or 1.0
        probs = {d: p / total for d, p in p_safe.items()}   # arrows show each move's share of safety
        source = MODEL_SOURCE if self.judge.calls > calls_before else "cache"
        return MoveDecision(move, probs, source, self.judge.latency_ms - latency_before, 0.0)
