"""laya decision mode: question shapes that fit how laya actually reads.

Measured on this repo's benchmark (see `laya-deploy eval`), laya does well when

1. `state` holds only the thing being judged (an intent, an action, a short phrase),
2. the options carry the candidates, each described by what sets it apart,
3. there is no catch-all option ("none", "other"): abstain in code with a threshold,
4. a yes/no judgment is a two-option `choice` with contrasting descriptions, not a `noul`,
5. each call asks about one subject (questions in one call share one state),

and the calling code owns ordering, tie-breaks and thresholds. Long prompts that pack the
facts into `instructions`, catch-all options, and per-item questions sharing
one state all scored far worse.

`Judge` wraps any `Asker` (HTTP server or in-process engine) and caches identical calls:
when the subject text repeats, laya is asked once.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .client import Asker

DEFAULT_CACHE = 4096


@dataclass(frozen=True)
class Verdict:
    """The answer to one judgment: winning label, its probability, all probabilities."""

    label: str | None          # None = abstained (top probability below the threshold)
    p: float
    probabilities: dict[str, float]

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)


class Judge:
    """laya-native decisions over an `Asker`, with an LRU cache of identical calls."""

    def __init__(self, ask: Asker, cache_size: int = DEFAULT_CACHE) -> None:
        self.ask = ask
        self.cache_size = cache_size
        self._cache: OrderedDict[str, dict[str, float]] = OrderedDict()
        self.calls = 0       # requests actually sent to laya
        self.hits = 0        # answers served from the cache
        self.latency_ms = 0.0  # total time spent in laya calls

    async def _probabilities(self, state: Any, question: Mapping[str, Any]) -> dict[str, float]:
        key = json.dumps([state, question], sort_keys=True, ensure_ascii=False)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        payload = await self.ask(state, {"q": dict(question)})
        probs = {str(k): float(v) for k, v in payload["answers"]["q"]["probabilities"].items()}
        self.calls += 1
        self.latency_ms += float(payload.get("latency_ms", 0.0))
        self._cache[key] = probs
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return probs

    async def classify(self, subject: str, labels: Mapping[str, str], *, question: str = "Which label fits?",
                       field: str = "subject", min_p: float = 0.0) -> Verdict:
        """Put `subject` in its own label. Two labels = a yes/no judgment with contrasting descriptions."""
        if len(labels) < 2:
            raise ValueError("classify needs at least two labels")
        q = {"type": "choice", "instructions": question, "criteria": dict(labels)}
        probs = await self._probabilities({field: subject}, q)
        return _verdict(probs, min_p)

    async def match(self, target: str, candidates: Mapping[str, str], *, min_p: float = 0.5,
                    question: str = "Which candidate is the target?", field: str = "target") -> Verdict:
        """Which candidate does `target` describe? Abstains (label None) below `min_p`; never offers 'none'."""
        if not candidates:
            return Verdict(None, 0.0, {})
        if len(candidates) == 1:  # laya needs two options; score the single one against a foil
            (cid, desc), = candidates.items()
            v = await self.classify(target, {cid: desc, "_other": "something else"}, question=question,
                                    field=field, min_p=min_p)
            return Verdict(cid if v.label == cid else None, v.probabilities.get(cid, 0.0),
                           {cid: v.probabilities.get(cid, 0.0)})
        q = {"type": "choice", "instructions": question, "criteria": dict(candidates)}
        probs = await self._probabilities({field: target}, q)
        return _verdict(probs, min_p)

    async def filter(self, items: Mapping[str, str], good: tuple[str, str], bad: tuple[str, str], *,
                     question: str, tolerance: float = 0.0) -> tuple[list[str], dict[str, float]]:
        """Judge each item alone (`good` vs `bad` label) and keep those within `tolerance` of the best.

        `good` / `bad` are (label, description). Returns (kept ids in input order, P(good) per id).
        """
        p_good: dict[str, float] = {}
        for item_id, text in items.items():
            v = await self.classify(text, dict((good, bad)), question=question)
            p_good[item_id] = v.probabilities.get(good[0], 0.0)
        if not p_good:
            return [], {}
        top = max(p_good.values())
        return [i for i in items if p_good[i] >= top - tolerance], p_good

    @property
    def stats(self) -> dict[str, int]:
        return {"calls": self.calls, "cache_hits": self.hits, "cached": len(self._cache),
                "latency_ms": round(self.latency_ms, 1)}


def _verdict(probs: dict[str, float], min_p: float) -> Verdict:
    if not probs:
        return Verdict(None, 0.0, {})
    best = max(probs, key=probs.__getitem__)
    p = probs[best]
    return Verdict(best if p >= min_p else None, p, probs)


# --------------------------------------------------------------------------- ready-made judgments
RISK_LABELS = {
    "destructive": "deletes, removes, discards, sends, posts, pays, resets, uninstalls, overwrites, replaces",
    "harmless": "opens, views, navigates, types, scrolls, saves, cancels, adjusts a setting",
}


async def risk(judge: Judge, action: str) -> float:
    """P(destructive) for a one-line action description, e.g. 'click Button "Delete all" in Notepad'."""
    v = await judge.classify(action, RISK_LABELS, question="Is this action destructive or harmless?", field="action")
    return v.probabilities.get("destructive", 0.0)


async def ground(judge: Judge, intent: str, elements: Mapping[str, str], min_p: float = 0.5) -> Verdict:
    """Which UI element does `intent` mean? `elements` maps id -> one-line description."""
    return await judge.match(intent, elements, min_p=min_p, field="step_intent",
                             question="Which on-screen element should the agent act on to carry out the step intent exactly?")
