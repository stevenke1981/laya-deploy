"""`laya-deploy pacman`: watch laya (or the offline heuristic) play Pac-Man, one decision per tick."""
from __future__ import annotations

import argparse
import asyncio
import statistics
from typing import Any

from .client import DEFAULT_URL, EngineAsker, HttpAsker
from .config import AUTO, CHECKPOINTS, Settings


def add_parser(sub: Any) -> None:
    sp = sub.add_parser("pacman", help="Pac-Man demo: laya picks every move")
    sp.add_argument("--model", default="english", choices=CHECKPOINTS + (AUTO,))
    sp.add_argument("--style", default="native", choices=("native", "prompt"),
                    help="native = laya decision mode (short phrases, cached); prompt = one long-prompt question per move")
    sp.add_argument("--url", default=DEFAULT_URL, help="laya-deploy server to ask (default: local)")
    sp.add_argument("--direct", action="store_true",
                    help="load laya in this process instead of calling a server")
    sp.add_argument("--fake", action="store_true", help="offline heuristic policy, no model")
    sp.add_argument("--headless", action="store_true", help="no window; print every step")
    sp.add_argument("--games", type=int, default=1,
                    help="headless: play this many seeds and compare with the heuristic")
    sp.add_argument("--steps", type=int, default=600)
    sp.add_argument("--delay", type=int, default=400, help="window: ms the arrows stay before moving")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--quiet", action="store_true", help="headless: only print summaries")
    sp.set_defaults(func=cmd_pacman)


def _make_policy(args: argparse.Namespace):
    from .games.pacman_agent import HeuristicPolicy, LayaPolicy
    from .games.pacman_native import NativeLayaPolicy
    from .mode import Judge

    if args.fake:
        return HeuristicPolicy(), None
    if args.direct:
        from .engine import LayaEngine

        s = Settings.from_env()
        engine = LayaEngine(Settings(**{**s.__dict__, "preload": ()}))
        asker = EngineAsker(engine, args.model)
    else:
        asker = HttpAsker(args.url, args.model, Settings.from_env().api_key)
    if args.style == "native":
        return NativeLayaPolicy(Judge(asker), args.model), asker
    return LayaPolicy(asker, args.model), asker


def _print_step(i: int, d: Any, game: Any) -> None:
    probs = " ".join(f"{k}={v:.2f}" for k, v in sorted(d.probabilities.items(), key=lambda kv: -kv[1]))
    events = f"  [{', '.join(game.events)}]" if game.events else ""
    print(f"{i:04d} {d.move:5s} p={d.p:.2f} {d.source:9s} score={game.score:5d} lives={game.lives}  "
          f"{probs}{events}")


def _describe(summary: Any) -> str:
    result = "WIN" if summary.won else ("GAME OVER" if summary.lives <= 0 else "STEP LIMIT")
    return (f"{result:10s} score={summary.score:5d} steps={summary.steps:4d} lives={summary.lives} "
            f"calls={summary.model_calls:4d} cached={summary.cache_hits:4d} fallbacks={summary.fallbacks} "
            f"avg={summary.avg_latency_ms:5.1f} ms")


async def _run_headless(args: argparse.Namespace, policy: Any) -> list:
    from .games.pacman import PacmanGame
    from .games.pacman_agent import play

    results = []
    for seed in range(args.seed, args.seed + args.games):
        hook = None if args.quiet or args.games > 1 else _print_step
        summary = await play(PacmanGame(seed=seed), policy, max_steps=args.steps, on_step=hook)
        print(f"[{policy.name}] seed={seed} {_describe(summary)}")
        results.append(summary)
    return results


def _table(name: str, results: list) -> str:
    scores = [r.score for r in results]
    wins = sum(r.won for r in results)
    return (f"{name:22s} games={len(results)} wins={wins} mean_score={statistics.mean(scores):7.1f} "
            f"median={statistics.median(scores):6.0f} mean_steps={statistics.mean(r.steps for r in results):6.1f}")


def cmd_pacman(args: argparse.Namespace) -> int:
    from .client import AskError

    policy, asker = _make_policy(args)
    if not args.headless:
        from .games.pacman_tk import run_app

        run_app(policy, seed=args.seed, delay_ms=args.delay, max_steps=args.steps,
                closer=asker.aclose if asker else None)
        return 0

    async def go() -> int:
        try:
            if asker is not None:  # fail fast if the server is not up
                try:
                    await asker("ping", {"ok": {"type": "noul", "instructions": "Is this a test?"}})
                except AskError as exc:
                    print(f"laya is not reachable: {exc}\nStart it with `laya-deploy serve`, "
                          "or pass --direct / --fake.")
                    return 1
            results = await _run_headless(args, policy)
            if args.games > 1 and not args.fake:
                from .games.pacman_agent import HeuristicPolicy

                baseline = await _run_headless(args, HeuristicPolicy())
                print()
                print(_table(policy.name, results))
                print(_table("HEURISTIC (baseline)", baseline))
            return 0
        finally:
            if asker is not None:
                await asker.aclose()

    return asyncio.run(go())
