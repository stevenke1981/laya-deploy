"""laya decision mode: question shapes, abstention, caching, and the native Pac-Man policy."""
import asyncio

import pytest

from laya_deploy.client import AskError
from laya_deploy.games.pacman import PacmanGame
from laya_deploy.games.pacman_agent import play
from laya_deploy.games.pacman_native import NativeLayaPolicy, danger_phrase, food_phrase
from laya_deploy.mode import Judge, ground, risk


class ScriptedAsker:
    """Records requests; answers with `pick(state, question) -> {label: p}`."""

    def __init__(self, pick):
        self.pick, self.requests = pick, []

    async def __call__(self, state, questions):
        self.requests.append((state, questions))
        (key, q), = questions.items()
        probs = self.pick(state, q)
        best = max(probs, key=probs.get)
        return {"answers": {key: {"type": "choice", "choice": best, "probabilities": probs}}, "latency_ms": 5.0}


def first_wins(state, q):
    labels = list(q["criteria"])
    return {lab: (0.8 if i == 0 else 0.2 / (len(labels) - 1)) for i, lab in enumerate(labels)}


def test_match_puts_only_the_target_in_state_and_never_offers_none():
    ask = ScriptedAsker(first_wins)
    v = asyncio.run(ground(Judge(ask), "the Save button", {"e1": 'Button "Save"', "e2": 'Button "Cancel"'}))
    state, questions = ask.requests[0]
    assert state == {"step_intent": "the Save button"}
    assert set(questions["q"]["criteria"]) == {"e1", "e2"} and questions["q"]["type"] == "choice"
    assert v.label == "e1" and v.p == 0.8


def test_match_abstains_below_min_p():
    ask = ScriptedAsker(lambda s, q: {k: 1 / len(q["criteria"]) for k in q["criteria"]})
    v = asyncio.run(Judge(ask).match("x", {"a": "A", "b": "B", "c": "C"}, min_p=0.5))
    assert v.label is None and v.p == pytest.approx(1 / 3)


def test_match_with_one_candidate_uses_a_foil():
    ask = ScriptedAsker(first_wins)
    v = asyncio.run(Judge(ask).match("x", {"only": "the only one"}))
    assert set(ask.requests[0][1]["q"]["criteria"]) == {"only", "_other"}
    assert v.label == "only" and set(v.probabilities) == {"only"}


def test_match_without_candidates_abstains_without_calling():
    ask = ScriptedAsker(first_wins)
    assert asyncio.run(Judge(ask).match("x", {})).label is None and ask.requests == []


def test_classify_needs_two_labels():
    with pytest.raises(ValueError):
        asyncio.run(Judge(ScriptedAsker(first_wins)).classify("x", {"only": "one"}))


def test_identical_calls_are_cached():
    ask = ScriptedAsker(first_wins)
    judge = Judge(ask)

    async def go():
        for _ in range(3):
            await judge.classify("same", {"a": "A", "b": "B"})
        await judge.classify("other", {"a": "A", "b": "B"})

    asyncio.run(go())
    assert len(ask.requests) == 2 and judge.stats["calls"] == 2 and judge.stats["cache_hits"] == 2
    assert judge.stats["latency_ms"] == 10.0


def test_cache_is_bounded():
    judge = Judge(ScriptedAsker(first_wins), cache_size=2)

    async def go():
        for s in ("a", "b", "c", "a"):
            await judge.classify(s, {"x": "X", "y": "Y"})

    asyncio.run(go())
    assert judge.stats["cached"] == 2 and judge.calls == 4   # "a" was evicted, so asked again


def test_filter_keeps_items_within_tolerance():
    p = {"safe one": 0.9, "almost": 0.86, "bad": 0.1}
    ask = ScriptedAsker(lambda s, q: {"good": p[s["subject"]], "bad": 1 - p[s["subject"]]})
    judge = Judge(ask)
    kept, scores = asyncio.run(judge.filter({"a": "safe one", "b": "almost", "c": "bad"},
                                            ("good", "fine"), ("bad", "not fine"), question="ok?", tolerance=0.05))
    assert kept == ["a", "b"] and scores["c"] == 0.1
    kept, _ = asyncio.run(judge.filter({"a": "safe one", "b": "almost"}, ("good", "fine"), ("bad", "not fine"),
                                       question="ok?"))
    assert kept == ["a"]


def test_risk_is_a_two_label_choice_on_the_action():
    ask = ScriptedAsker(lambda s, q: {"destructive": 0.9, "harmless": 0.1})
    assert asyncio.run(risk(Judge(ask), 'click Button "Delete all"')) == 0.9
    state, questions = ask.requests[0]
    assert state == {"action": 'click Button "Delete all"'}
    assert set(questions["q"]["criteria"]) == {"destructive", "harmless"}


def test_phrases_are_coarse_and_ordered():
    game = PacmanGame()
    info = game.move_info("left")
    assert food_phrase(info) == "eats a pellet"
    assert danger_phrase(info).startswith("safe")


def oracle(state, q):
    """Answers like laya does on these phrases: 'safe' text is safe, pellets beat no pellets."""
    crit = q["criteria"]
    if "safe" in crit:
        p = 0.98 if state["subject"].startswith("safe") else 0.02 if state["subject"].startswith("deadly") else 0.2
        return {"safe": p, "dangerous": 1 - p}
    order = ["chases an edible ghost", "eats a power pellet", "eats a pellet", "pellet close ahead",
             "pellet a few steps away", "no pellet nearby"]
    raw = {k: float(len(order) - order.index(v)) for k, v in crit.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def test_native_policy_avoids_a_ghost_and_caches_repeated_phrases():
    ask = ScriptedAsker(oracle)
    policy = NativeLayaPolicy(Judge(ask))
    game = PacmanGame()
    next(g for g in game.ghosts if g.name == "blinky").pos = (8, 15)   # ghost right next to the left move
    d = asyncio.run(policy.decide(game))
    assert d.move == "right" and d.source == "laya" and d.p > 0.5
    summary = asyncio.run(play(PacmanGame(seed=1), NativeLayaPolicy(Judge(ScriptedAsker(oracle))), max_steps=120))
    assert summary.cache_hits > summary.model_calls > 0 and summary.fallbacks == 0


def test_native_policy_falls_back_when_laya_fails():
    async def broken(state, questions):
        raise AskError("down")

    d = asyncio.run(NativeLayaPolicy(Judge(broken)).decide(PacmanGame()))
    assert d.source == "fallback" and d.error == "down"


def test_cli_has_style_and_eval():
    from laya_deploy.cli import build_parser

    p = build_parser()
    assert p.parse_args(["pacman"]).style == "native"
    assert p.parse_args(["pacman", "--style", "prompt"]).style == "prompt"
    assert p.parse_args(["eval", "--models", "english"]).models == ["english"]
