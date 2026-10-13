"""`laya-deploy pacman`: watch laya (or the offline heuristic) play Pac-Man, one decision per tick."""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from dataclasses import replace
from typing import Any

from .client import DEFAULT_URL, EngineAsker, HttpAsker
from .config import AUTO, CHECKPOINTS, Settings


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _delay(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("delay must be zero or greater")
    return number


def add_parser(sub: Any) -> None:
    sp = sub.add_parser("pacman", help="Pac-Man demo: laya picks every move")
    sp.add_argument("--model", default="english", choices=CHECKPOINTS + (AUTO,))
    sp.add_argument("--style", default="native", choices=("native", "prompt"),
                    help="native = laya decision mode (short phrases, cached); prompt = one long-prompt question per move")
    sp.add_argument("--url", default=DEFAULT_URL, help="laya-deploy server to ask (default: local)")
    backend = sp.add_mutually_exclusive_group()
    backend.add_argument("--direct", action="store_true",
                    help="load laya in this process instead of calling a server")
    backend.add_argument("--fake", action="store_true", help="offline heuristic policy, no model")
    sp.add_argument("--headless", action="store_true", help="no window; print every step")
    sp.add_argument("--games", type=_positive_int, default=1,
                    help="headless: play this many seeds and compare with the heuristic")
    sp.add_argument("--steps", type=_positive_int, default=600)
    sp.add_argument("--delay", type=_delay, default=400, help="window: ms the arrows stay before moving")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--quiet", action="store_true", help="headless: only print summaries")
    sp.set_defaults(func=cmd_pacman)


def _make_policy(args: argparse.Namespace):
    from .games.pacman_agent import HeuristicPolicy, LayaPolicy
    from .games.pacman_native import NativeLayaPolicy
    from .games.pacman_runtime import RequiredModelPolicy, ValidatedAsker
    from .mode import Judge

    if args.fake:
        return HeuristicPolicy(), None
    if args.direct:
        from .engine import LayaEngine

        s = Settings.from_env()
        print("Loading local Laya model (first launch can take a moment)...", flush=True)
        engine = LayaEngine(replace(s, preload=()))
        print(f"Laya device: {engine.device}" + (f" ({engine.gpu})" if engine.gpu else ""), flush=True)
        asker = EngineAsker(engine, args.model)
    else:
        asker = HttpAsker(args.url, args.model, Settings.from_env().api_key)
    validated = ValidatedAsker(asker)
    if args.style == "native":
        policy = NativeLayaPolicy(Judge(validated), args.model)
    else:
        policy = LayaPolicy(validated, args.model)
    return RequiredModelPolicy(policy, asker), asker


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

    try:
        policy, asker = _make_policy(args)
    except Exception as exc:
        print(f"Cannot start Laya: {exc}\nCheck model installation: uv sync --extra model", file=sys.stderr)
        return 1
    if not args.headless:
        from .games.pacman_tk import run_app

        try:
            run_app(policy, seed=args.seed, delay_ms=args.delay, max_steps=args.steps,
                    closer=asker.aclose if asker else None)
        except Exception as exc:
            print(f"Cannot open Pac-Man window: {exc}", file=sys.stderr)
            return 1
        return 0

    async def go() -> int:
        try:
            results = await _run_headless(args, policy)
            if args.games > 1 and not args.fake:
                from .games.pacman_agent import HeuristicPolicy

                baseline = await _run_headless(args, HeuristicPolicy())
                print()
                print(_table(policy.name, results))
                print(_table("HEURISTIC (baseline)", baseline))
            return 0
        except AskError as exc:
            print(f"{exc}\nStart `laya-deploy serve` or use --direct for the local model.", file=sys.stderr)
            return 1
        finally:
            if asker is not None:
                await asker.aclose()

    return asyncio.run(go())
