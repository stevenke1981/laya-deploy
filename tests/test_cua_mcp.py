"""MCP server and `laya-deploy` computer-use subcommands, on the fake desktop (no laya server needed)."""
import argparse
import asyncio
import json

import pytest

pytest.importorskip("mcp")

from laya_deploy import cua_cli
from laya_deploy.client import AskError
from laya_deploy.cua import controller, mcp_server
from laya_deploy.cua.mcp_server import INSTRUCTIONS, SERVER_NAME, build_server
from laya_deploy.cua.settings import CuaSettings

TOOLS = {"observe", "ground", "list_windows", "focus_window", "launch", "click", "type_text",
         "hotkey", "scroll", "wait", "screenshot"}
DEAD_URL = "http://laya.invalid/api/alpha/decisions"
VAGUE = "the control that stores the document"   # no exact name match, so laya must be asked


class DeadAsker:
    """Stands in for HttpAsker when `laya-deploy serve` is not running."""

    def __init__(self, *args, **kwargs):
        pass

    async def __call__(self, state, questions):
        raise AskError(f"cannot reach {DEAD_URL}")

    async def aclose(self):
        pass


@pytest.fixture(autouse=True)
def laya_is_down(monkeypatch):
    monkeypatch.setattr(controller, "HttpAsker", DeadAsker)


def settings(**kw):
    return CuaSettings(backend="fake", settle_s=0.0, laya_url=DEAD_URL, **kw)


def text_of(result):
    content = getattr(result, "content", result)
    if isinstance(content, tuple):  # SDK 1.x returns (content, structured)
        content = content[0]
    return "".join(getattr(c, "text", "") for c in content)


def call(server, name, **args):
    return text_of(asyncio.run(server.call_tool(name, args)))


def test_server_lists_the_expected_tools():
    server = build_server(settings())
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == TOOLS


def test_server_name_and_instructions():
    server = build_server(settings())
    assert server.name == SERVER_NAME == "laya-computer-use"
    for word in ("observe", "ambiguous", "needs_confirmation", "confirm=true", "stale", "untrusted"):
        assert word in INSTRUCTIONS
    assert "cannot plan" in INSTRUCTIONS


def test_observe_tool_returns_outline_with_revision():
    server = build_server(settings())
    out = call(server, "observe")
    assert '"Save"' in out and "revision: " in out
    data = json.loads(call(server, "observe", format="json"))
    assert data["revision"] and data["elements"]


def test_click_by_point_fails_closed_when_laya_is_down():
    server = build_server(settings())
    out = call(server, "click", target="230,90")
    assert out.startswith("[needs_confirmation]") and "confirm=true" in out
    done = call(server, "click", target="230,90", confirm=True)
    assert done.startswith("[done]")


def test_description_target_reports_laya_unreachable():
    server = build_server(settings())
    out = call(server, "click", target=VAGUE)
    assert out.startswith("[error]") and "laya-deploy serve" in out


def test_type_without_enter_needs_no_laya():
    server = build_server(settings())
    call(server, "observe")
    out = call(server, "type_text", text="hello", target="e5")
    assert out.startswith("[done]")


def test_ground_tool_returns_json_error_status_without_laya():
    server = build_server(settings())
    out = json.loads(call(server, "ground", target=VAGUE))
    assert out["status"] == "error" and out["would_act"] is False


def test_cli_registers_subcommands():
    p = argparse.ArgumentParser()
    cua_cli.add_parser(p.add_subparsers(dest="cmd", required=True))
    a = p.parse_args(["mcp"])
    assert a.cmd == "mcp" and a.func is cua_cli.cmd_mcp
    a = p.parse_args(["cua-observe", "--json", "--full", "--all"])
    assert (a.json, a.full, a.all) == (True, True, True) and a.func is cua_cli.cmd_observe
    a = p.parse_args(["cua-ground", "the Save button"])
    assert a.target == "the Save button" and a.action == "click" and a.json is False
    a = p.parse_args(["cua-ground", "e3", "--action", "type", "--json"])
    assert a.action == "type" and a.json is True and a.func is cua_cli.cmd_ground
    with pytest.raises(SystemExit):
        p.parse_args(["cua-ground", "x", "--action", "drag"])


def test_cli_observe_and_ground_run_on_the_fake_backend(monkeypatch, capsys):
    monkeypatch.setenv("LAYA_CUA_BACKEND", "fake")
    monkeypatch.setenv("LAYA_CUA_URL", DEAD_URL)
    monkeypatch.setenv("LAYA_CUA_SETTLE_S", "0")
    p = argparse.ArgumentParser()
    cua_cli.add_parser(p.add_subparsers(dest="cmd", required=True))

    args = p.parse_args(["cua-observe", "--json"])
    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["revision"]

    args = p.parse_args(["cua-observe"])
    assert args.func(args) == 0
    assert "Foreground window" in capsys.readouterr().out

    args = p.parse_args(["cua-ground", "230,90", "--json"])
    assert args.func(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok" and out["needs_confirmation"] is True  # laya down: fails closed

    args = p.parse_args(["cua-ground", VAGUE])
    assert args.func(args) == 1
    assert "status: error" in capsys.readouterr().out


def test_lifespan_closes_the_controller_on_shutdown(monkeypatch):
    captured = []
    closed = []

    class Recording(mcp_server._Server):
        def __init__(self, *args, **kwargs):
            captured.append(kwargs.get("lifespan"))
            super().__init__(*args, **kwargs)

    async def fake_aclose(self):
        closed.append(self)

    monkeypatch.setattr(mcp_server, "_Server", Recording)
    monkeypatch.setattr(controller.LayaComputer, "aclose", fake_aclose)
    server = build_server(settings())
    assert captured and captured[0] is not None   # the SDK accepted a lifespan
    call(server, "observe")   # creates the controller

    async def shutdown():
        async with captured[0](server):
            pass

    asyncio.run(shutdown())
    assert len(closed) == 1
    asyncio.run(shutdown())   # idempotent: nothing left to close
    assert len(closed) == 1


def test_atexit_hook_closes_a_leftover_controller(monkeypatch):
    closed = []

    class Fake:
        async def aclose(self):
            closed.append(1)

    holder = {"c": Fake()}
    mcp_server._close_holder_sync(holder)
    assert closed == [1] and holder == {}
    mcp_server._close_holder_sync(holder)   # empty: no-op
    assert closed == [1]
