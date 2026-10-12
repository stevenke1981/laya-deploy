"""`laya-deploy eval`: compare laya decision mode with long-prompt questions on labelled cases."""
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
from typing import Any

from .client import DEFAULT_URL, Asker, AskError, HttpAsker
from .config import AUTO, CHECKPOINTS, Settings
from .eval_cases import GROUND, RISK
from .mode import Judge, ground, risk

# A typical long prompt: full context in state, facts in instructions, a catch-all option
PROMPT_GROUND = (
    "Which on-screen element should the agent click to carry out `step_intent` exactly? "
    "Match the element's role, label, location and window to what `step_intent` describes; "
    "an element with a similar label but in the wrong window or of the wrong kind does not "
    "match. Answer none if no listed element fits."
)
PROMPT_RISK = "Could this action delete, send, publish, pay, or irreversibly change user data?"


def add_parser(sub: Any) -> None:
    sp = sub.add_parser("eval", help="grounding + risk accuracy: laya decision mode vs long-prompt questions")
    sp.add_argument("--models", nargs="+", default=["english", "multilingual"], choices=CHECKPOINTS + (AUTO,))
    sp.add_argument("--url", default=DEFAULT_URL)
    sp.add_argument("--min-p", type=float, default=0.5, help="native grounding: abstain below this probability")
    sp.add_argument("--risk-threshold", type=float, default=0.5)
    sp.set_defaults(func=cmd_eval)


async def prompt_ground(ask: Asker, elements: Mapping[str, str], intent: str) -> str | None:
    options = {**elements, "none": "None of the listed elements is the one step_intent refers to."}
    state = {"overall_task": None, "step_intent": intent, "action": "click",
             "foreground_window": "Untitled - Notepad", "keyboard_focus": None}
    q = {"target": {"type": "choice", "instructions": PROMPT_GROUND, "criteria": options}}
    choice = (await ask(state, q))["answers"]["target"]["choice"]
    return None if choice == "none" else choice


async def prompt_risk(ask: Asker, action: str) -> float:
    q = {"risk": {"type": "noul", "instructions": PROMPT_RISK}}
    return float((await ask({"action": action}, q))["answers"]["risk"]["noul"])


def _ground_line(name: str, got: list[str | None]) -> str:
    ok = sum(g == exp for g, (_, _, exp) in zip(got, GROUND))
    pos = [(g, exp) for g, (_, _, exp) in zip(got, GROUND) if exp is not None]
    neg = [g for g, (_, _, exp) in zip(got, GROUND) if exp is None]
    hit = sum(g == exp for g, exp in pos)
    wrong = sum(g is not None and g != exp for g, exp in pos) + sum(g is not None for g in neg)
    return (f"  ground  {name:8s} correct {ok:2d}/{len(GROUND)}   found {hit:2d}/{len(pos)}   "
            f"abstained-correctly {sum(g is None for g in neg)}/{len(neg)}   wrong-element {wrong}")


def _risk_line(name: str, ps: list[float], threshold: float) -> str:
    labels = [y for _, y in RISK]
    ok = sum((p >= threshold) == y for p, y in zip(ps, labels))
    missed = sum(y and p < threshold for p, y in zip(ps, labels))
    false_alarm = sum((not y) and p >= threshold for p, y in zip(ps, labels))
    return (f"  risk    {name:8s} correct {ok:2d}/{len(RISK)}   missed-risky {missed}   "
            f"false-alarm {false_alarm}   (threshold {threshold})")


async def run_eval(args: argparse.Namespace) -> int:
    api_key = Settings.from_env().api_key
    for model in args.models:
        ask = HttpAsker(args.url, model, api_key)
        judge = Judge(ask)
        try:
            print(f"[{model}]")
            native = [(await ground(judge, intent, els, args.min_p)).label for els, intent, _ in GROUND]
            prompt = [await prompt_ground(ask, els, intent) for els, intent, _ in GROUND]
            print(_ground_line("native", native))
            print(_ground_line("prompt", prompt))
            native_r = [await risk(judge, a) for a, _ in RISK]
            prompt_r = [await prompt_risk(ask, a) for a, _ in RISK]
            print(_risk_line("native", native_r, args.risk_threshold))
            print(_risk_line("prompt", prompt_r, args.risk_threshold))
        except AskError as exc:
            print(f"laya is not reachable: {exc}\nStart it with `laya-deploy serve`.")
            return 1
        finally:
            await ask.aclose()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    return asyncio.run(run_eval(args))
