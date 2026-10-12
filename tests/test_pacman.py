"""Pac-Man demo: game rules (ported from decision-cua) and the laya policy.

Policy tests run through the real FastAPI app with the fake engine or a mock transport,
so no model download or GPU is needed.
"""
import asyncio

import httpx
import pytest

from laya_deploy.client import EngineAsker, HttpAsker
from laya_deploy.config import Settings
from laya_deploy.engine import FakeEngine
from laya_deploy.games.pacman import GHOST_POINTS, PELLET_POINTS, PacmanGame
from laya_deploy.games.pacman_agent import (
    MODEL_SOURCE,
    HeuristicPolicy,
    LayaPolicy,
    play,
)
from laya_deploy.server import create_app

URL = "http://laya.test/api/alpha/decisions"


def ghost(game, name):
    return next(g for g in game.ghosts if g.name == name)


def test_start_position_and_legal_moves():
    game = PacmanGame()
    assert game.pac == (9, 15)
    assert game.legal_moves() == ["left", "right"]
    assert len(game.ghosts) == 4 and ghost(game, "blinky").dangerous
    assert all(g.in_house for g in game.ghosts if g.name != "blinky")


def test_eating_a_pellet_scores():
    game = PacmanGame()
    game.step("left")
    assert game.pac == (8, 15) and game.score == PELLET_POINTS and "pellet" in game.events


def test_tunnel_wraps_around():
    game = PacmanGame()
    game.pac, game.pac_dir = (0, 9), "left"
    game.step("left")
    assert game.pac == (18, 9)


def test_illegal_direction_keeps_heading():
    game = PacmanGame()
    game.step("up")  # wall above the start cell
    assert game.pac == (8, 15)  # kept the initial heading (left)


def test_dangerous_ghost_costs_a_life_and_resets():
    game = PacmanGame()
    ghost(game, "blinky").pos = (8, 15)
    game.step("left")
    assert "death" in game.events and game.lives == 2
    assert game.pac == (9, 15) and ghost(game, "blinky").pos == (9, 7)


def test_power_pellet_makes_ghosts_edible():
    game = PacmanGame()
    game.pac, game.pac_dir = (2, 15), "left"
    game.step("left")
    blinky = ghost(game, "blinky")
    assert "power" in game.events and blinky.edible
    blinky.pos = (1, 14)
    score = game.score
    game.step("up")
    assert "eat-ghost:blinky" in game.events and blinky.in_house
    assert game.score >= score + GHOST_POINTS


def test_text_state_lists_the_board_and_legal_moves():
    text = PacmanGame().to_text()
    assert "P" in text and "Legal moves: left, right." in text
    assert "blinky at x=9, y=7: dangerous" in text


def test_move_info_measures_ghost_distance():
    game = PacmanGame()
    ghost(game, "blinky").pos = (6, 15)
    left, right = game.move_info("left"), game.move_info("right")
    assert left.ghost_dist == 2 and right.ghost_dist > left.ghost_dist


def test_heuristic_avoids_an_adjacent_ghost():
    game = PacmanGame()
    ghost(game, "blinky").pos = (8, 15)
    d = asyncio.run(HeuristicPolicy().decide(game))
    assert d.move == "right" and abs(sum(d.probabilities.values()) - 1) < 1e-9
    assert d.probabilities["left"] < 0.01


def mock_asker(handler):
    return HttpAsker(URL, "english", transport=httpx.MockTransport(handler))


def app_asker(engine):
    """HttpAsker wired to the real FastAPI app (in memory) backed by `engine`."""
    app = create_app(Settings(), engine)
    return HttpAsker(URL, "english", transport=httpx.ASGITransport(app=app))


def run(coro_factory, asker):
    async def go():
        try:
            return await coro_factory()
        finally:
            await asker.aclose()
    return asyncio.run(go())


def test_laya_policy_goes_through_the_server_and_offers_only_legal_moves():
    engine = FakeEngine()
    asker = app_asker(engine)
    d = run(lambda: LayaPolicy(asker, "english").decide(PacmanGame()), asker)
    sent = engine.calls[0]
    assert sent["model"] == "english"
    assert set(sent["questions"]["move"]["criteria"]) == {"left", "right"}
    assert "eats a pellet" in sent["questions"]["move"]["criteria"]["left"]
    assert "Legal moves" in sent["state"]
    # the fake engine picks the first option
    assert d.move == "left" and d.source == MODEL_SOURCE and d.p == 0.7 and d.cost == 0.0
    assert d.latency_ms >= 0


def test_single_exit_is_forced_without_calling_laya():
    def handler(request):
        raise AssertionError("laya must not be called for a forced move")

    asker = mock_asker(handler)
    d = run(lambda: LayaPolicy(asker).decide(PacmanGame(maze=("#####", "#P.##", "#####"))), asker)
    assert d.move == "right" and d.source == "forced"


@pytest.mark.parametrize("response", [
    httpx.Response(500, json={"error": {"message": "CUDA out of memory"}}),
    httpx.Response(200, json={"answers": {"move": {"choice": "up"}}}),   # not a legal move
    httpx.Response(200, json={"answers": {}}),                           # missing answer
    httpx.Response(200, text="not json"),
])
def test_bad_laya_answers_fall_back_to_heuristic(response):
    asker = mock_asker(lambda request: response)
    d = run(lambda: LayaPolicy(asker).decide(PacmanGame()), asker)
    assert d.source == "fallback" and d.error and d.move in {"left", "right"}


def test_unreachable_server_falls_back():
    def handler(request):
        raise httpx.ConnectError("refused")

    asker = mock_asker(handler)
    d = run(lambda: LayaPolicy(asker).decide(PacmanGame()), asker)
    assert d.source == "fallback" and "cannot reach" in d.error


def test_headless_game_through_the_server_counts_calls():
    engine = FakeEngine()
    asker = app_asker(engine)
    summary = run(lambda: play(PacmanGame(seed=1), LayaPolicy(asker), max_steps=40), asker)
    assert summary.steps <= 40 and summary.model_calls == len(engine.calls) > 0
    assert summary.fallbacks == 0 and summary.cost == 0.0


def test_engine_asker_skips_http():
    engine = FakeEngine()
    asker = EngineAsker(engine, "multilingual")
    summary = run(lambda: play(PacmanGame(seed=2), LayaPolicy(asker), max_steps=20), asker)
    assert summary.model_calls == len(engine.calls) > 0
    assert all(c["model"] == "multilingual" for c in engine.calls)


def test_heuristic_game_is_deterministic_for_a_seed():
    a = asyncio.run(play(PacmanGame(seed=3), HeuristicPolicy(), max_steps=150))
    b = asyncio.run(play(PacmanGame(seed=3), HeuristicPolicy(), max_steps=150))
    assert (a.score, a.steps, a.lives) == (b.score, b.steps, b.lives) and a.score > 0


def test_pacman_cli_parses():
    from laya_deploy.cli import build_parser

    args = build_parser().parse_args(["pacman", "--headless", "--games", "3", "--model", "english"])
    assert args.games == 3 and args.model == "english" and args.headless
