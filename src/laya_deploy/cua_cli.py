"""`laya-deploy mcp | cua-observe | cua-ground`: laya computer use from the command line."""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from .cua.settings import CuaSettings


def add_parser(sub: Any) -> None:
    sp = sub.add_parser("mcp", help="run the computer-use MCP server (stdio)")
    sp.set_defaults(func=cmd_mcp)

    sp = sub.add_parser("cua-observe", help="print what laya computer use sees on screen")
    sp.add_argument("--json", action="store_true", help="machine-readable output with revision")
    sp.add_argument("--full", action="store_true", help="list every element, not just the top ones")
    sp.add_argument("--all", action="store_true", help="include every window, not just the foreground")
    sp.set_defaults(func=cmd_observe)

    sp = sub.add_parser("cua-ground", help="dry run: which element a description resolves to (no input)")
    sp.add_argument("target", help="description, element id (e12) or point (x,y)")
    sp.add_argument("--action", default="click", choices=["click", "type", "scroll"])
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_ground)


def cmd_mcp(args: argparse.Namespace) -> int:
    from .cua.mcp_server import run

    run()
    return 0


async def _with_computer(settings: CuaSettings, fn: Any) -> Any:
    from .cua.controller import LayaComputer

    computer = LayaComputer(settings)
    try:
        return await fn(computer)
    finally:
        await computer.aclose()


def cmd_observe(args: argparse.Namespace) -> int:
    fmt = "json" if args.json else "text"
    out = asyncio.run(_with_computer(
        CuaSettings.from_env(), lambda c: c.observe(full=args.full, all_windows=args.all, fmt=fmt)))
    print(out)
    return 0


def format_ground(res: dict[str, Any]) -> str:
    lines = [f"status: {res['status']}  ({res.get('message', '')})", f"p: {res.get('p')}"]
    el = res.get("element")
    if el:
        lines.append(f"element: [{el['id']}] {el['role']} \"{el['name']}\" in \"{el['window']}\"")
    for c in res.get("candidates", []):
        lines.append(f"  candidate [{c['id']}] p={c['p']:.2f} {c['describe']}")
    if res.get("risk_p") is not None:
        lines.append(f"risk p={res['risk_p']}  needs_confirmation={res['needs_confirmation']}")
    lines.append(f"revision: {res.get('revision')}  timing: {res.get('timing_ms')}")
    return "\n".join(lines)


def cmd_ground(args: argparse.Namespace) -> int:
    res = asyncio.run(_with_computer(
        CuaSettings.from_env(), lambda c: c.ground(args.target, args.action)))
    print(json.dumps(res, ensure_ascii=False, indent=2) if args.json else format_ground(res))
    return 0 if res["status"] == "ok" else 1
