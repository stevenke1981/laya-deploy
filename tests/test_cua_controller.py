"""LayaComputer on the fake desktop with stub grounder/guard: no GPU, laya server or real desktop."""
import asyncio
import json
import threading
from dataclasses import dataclass, replace

import pytest

from laya_deploy.client import AskError
from laya_deploy.cua import controller as ctl
from laya_deploy.cua.backends.fake import FakeBackend
from laya_deploy.cua.controller import ActionResult, LayaComputer, Timing
from laya_deploy.cua.keys import parse_chord
from laya_deploy.cua.settings import CuaSettings


@dataclass
class StubGround:
    element: object
    p: float
    status: str
    candidates: list
    calls: int = 0


@dataclass
class StubRisk:
    p: float
    laya_p: float = 0.0
    keyword: str | None = None
    needs_confirmation: bool = False
    error: str = ""


class StubGrounder:
    """Resolves a description by name substring; `override` forces a result."""

    def __init__(self, override=None, error=None):
        self.override, self.error, self.requests = override, error, []

    async def ground(self, target, snap, action="click"):
        self.requests.append((target, action))
        if self.error:
            raise self.error
        if self.override:
            return self.override(snap)
        words = [w for w in target.lower().split() if len(w) > 3]
        hits = [e for e in snap.elements if any(w in e.name.lower() for w in words)]
        if not hits:
            return StubGround(None, 0.0, "no_match", [])
        return StubGround(hits[0], 0.9, "ok", [(hits[0], 0.9)])


class StubGuard:
    """Risky when the action text mentions delete; counts calls."""

    def __init__(self, error=None):
        self.texts, self.error = [], error

    async def risk(self, action_text):
        self.texts.append(action_text)
        if self.error:
            raise self.error
        risky = "delete" in action_text.lower()
        return StubRisk(0.95 if risky else 0.05, needs_confirmation=risky)


@pytest.fixture(autouse=True)
def fast_launch(monkeypatch):
    monkeypatch.setattr(ctl, "LAUNCH_WAIT_S", 0.3)
    monkeypatch.setattr(ctl, "LAUNCH_GRACE_S", 0.1)


def make(grounder=None, guard=None, **settings):
    backend = FakeBackend()
    s = CuaSettings(backend="fake", settle_s=0.0, **settings)
    c = LayaComputer(s, backend_factory=lambda: backend, grounder=grounder or StubGrounder(),
                     guard=guard or StubGuard())
    return c, backend


def run(coro):
    return asyncio.run(coro)


def eid(c, name, role=None):
    el = next(e for e in c.last.elements if e.name == name and (role is None or e.role == role))
    return el.id


async def observed(c):
    await c.observe()
    return c


def test_observe_text_lists_ids_and_revision():
    c, _ = make()
    text = run(c.observe())
    assert "Foreground window" in text and '"Save"' in text
    assert "[e1]" in text and "revision: " in text


def test_observe_json_has_revision_and_elements():
    c, _ = make()
    data = json.loads(run(c.observe(fmt="json")))
    assert len(data["revision"]) == 16
    assert {e["name"] for e in data["elements"]} >= {"Save", "Delete all"}
    assert run(c.observe(fmt="xml")).startswith("[error]")


def test_click_by_id_skips_grounding():
    async def scenario():
        c, backend = make()
        await c.observe()
        res = await c.click(eid(c, "Save", "Button"))
        return c, backend, res

    c, backend, res = run(scenario())
    assert res.status == "done" and res.ok and res.revision
    assert c.grounder.requests == []
    assert backend.fg == 3  # the Save As dialog opened
    assert {"observe", "decide", "act", "total"} <= set(res.timing_ms)


def test_click_by_point_skips_grounding():
    c, backend = make()
    res = run(c.click("230,90"))
    assert res.status == "done" and c.grounder.requests == []
    assert backend.fg == 3 and "Point" in res.target


def test_click_point_outside_screen_is_not_found():
    c, backend = make()
    res = run(c.click("99999,5"))
    assert res.status == "not_found" and backend.log == []


def test_click_by_description_uses_grounder():
    c, backend = make()
    res = run(c.click("the Save button"))
    assert res.status == "done" and res.grounding_p == 0.9
    assert c.grounder.requests == [("the Save button", "click")]
    assert backend.fg == 3


def test_ambiguous_lists_candidates_and_sends_no_input():
    def low(snap):
        els = [e for e in snap.elements if e.name in ("Save", "Edit")]
        return StubGround(els[0], 0.32, "low_confidence", [(els[0], 0.32), (els[1], 0.30)])

    c, backend = make(grounder=StubGrounder(override=low))
    res = run(c.click("save something"))
    assert res.status == "ambiguous" and not res.ok
    assert "e" in res.message and "p=0.32" in res.message and "Candidates" in res.message
    assert [x["id"] for x in res.extra["candidates"]] and backend.log == []


def test_no_match_is_not_found():
    c, backend = make()
    res = run(c.click("the zebra control"))
    assert res.status == "not_found" and backend.log == []


def test_stale_revision_sends_no_input():
    c, backend = make()
    res = run(c.click("230,90", if_revision="deadbeefdeadbeef"))
    assert res.status == "stale" and res.revision and backend.log == []


def test_matching_revision_acts():
    async def scenario():
        c, backend = make()
        rev = json.loads(await c.observe(fmt="json"))["revision"]
        return await c.click("230,90", if_revision=rev), backend

    res, backend = run(scenario())
    assert res.status == "done" and backend.fg == 3


def test_risky_click_needs_confirmation_then_confirm_acts():
    async def scenario():
        c, backend = make()
        await c.observe()
        backend.focus.value = "precious"
        target = eid(c, "Delete all")
        first = await c.click(target)
        text_after_first = backend.editor_text
        second = await c.click(target, confirm=True)
        return first, text_after_first, second, backend

    first, text_after_first, second, backend = run(scenario())
    assert first.status == "needs_confirmation" and not first.ok
    assert first.risk_p == pytest.approx(0.95) and "confirm=true" in first.message
    assert text_after_first == "precious"
    assert second.status == "done" and backend.editor_text == ""


def test_auto_confirm_risky_setting_skips_confirmation():
    c, backend = make(auto_confirm_risky=True)
    backend.focus.value = "x"
    res = run(c.click("the delete all control"))
    assert res.status == "done" and backend.editor_text == ""


def test_guard_text_names_the_element_and_window():
    c, _ = make()
    run(c.click("the Save button"))
    assert 'click Button "Save"' in c.guard.texts[0]
    assert 'window "Untitled - Notepad"' in c.guard.texts[0]


def test_type_without_enter_skips_the_gate():
    async def scenario():
        c, backend = make()
        await c.observe()
        return c, backend, await c.type_text("hello", target=eid(c, "Text editor"))

    c, backend, res = run(scenario())
    assert res.status == "done" and backend.editor_text == "hello"
    assert c.guard.texts == []


def test_type_with_enter_is_gated():
    async def scenario():
        c, backend = make(guard=StubGuard())
        await c.observe()
        return c, await c.type_text("delete everything", enter=True)

    c, res = run(scenario())
    assert res.status == "needs_confirmation"
    assert len(c.guard.texts) == 1 and "press Enter" in c.guard.texts[0]


def test_type_with_enter_acts_when_safe_and_clears():
    c, backend = make()
    backend.focus.value = "old"
    res = run(c.type_text("new", clear=True, enter=True))
    assert res.status == "done" and backend.editor_text == "new"
    assert "press enter" in backend.log


def test_grounder_ask_error_reports_error_status():
    c, backend = make(grounder=StubGrounder(error=AskError("cannot reach http://127.0.0.1:8100")))
    res = run(c.click("the Save button"))
    assert res.status == "error" and "laya-deploy serve" in res.message and backend.log == []


def test_ids_and_points_still_work_when_laya_is_down():
    c, backend = make(grounder=StubGrounder(error=AskError("down")), guard=StubGuard(error=AskError("down")))
    res = run(c.click("230,90"))  # risk check fails closed
    assert res.status == "needs_confirmation" and "could not be reached" in res.message
    assert backend.log == []
    res = run(c.click("230,90", confirm=True))
    assert res.status == "done" and backend.fg == 3


def test_risk_result_error_field_fails_closed():
    class ErrGuard(StubGuard):
        async def risk(self, action_text):
            return StubRisk(1.0, error="HTTP 500")

    c, _ = make(guard=ErrGuard())
    res = run(c.click("230,90"))
    assert res.status == "needs_confirmation" and "HTTP 500" in res.message


def test_hotkey_routine_chord_acts_without_asking_laya():
    c, backend = make()
    res = run(c.hotkey("Ctrl+S"))
    assert res.status == "done" and backend.fg == 3 and "press ctrl+s" in backend.log
    assert c.guard.texts == []


@pytest.mark.parametrize("chord", ["esc", "ctrl+a", "tab", "shift+tab", "down", "ctrl+shift+s", "alt",
                                   "ctrl++", "ctrl+-", "ctrl+0", "f5", "ctrl+z"])
def test_routine_chords_skip_the_gate(chord):
    assert ctl.is_routine_chord(parse_chord(chord))
    c, backend = make(guard=StubGuard())
    res = run(c.hotkey(chord))
    assert res.ok and c.guard.texts == []


@pytest.mark.parametrize("chord", ["enter", "delete", "ctrl+w", "ctrl+enter", "ctrl+x", "ctrl+v", "alt+f4"])
def test_other_chords_are_not_routine(chord):
    assert not ctl.is_routine_chord(parse_chord(chord))


def test_hotkey_risk_text_names_the_window_not_the_focused_element():
    c, _ = make()
    run(c.hotkey("enter"))
    assert c.guard.texts == ['press enter, window "Untitled - Notepad"']


@pytest.mark.parametrize("kwargs", [{"clear": True}, {"text": "a\nb"}, {"text": "a\rb"}, {"text": "a\tb"}])
def test_type_text_gate_covers_clear_and_control_characters(kwargs):
    c, backend = make(guard=StubGuard())
    res = run(c.type_text(**{"text": "x", **kwargs}))
    assert res.status == "done" and len(c.guard.texts) == 1   # gated, judged harmless, typed


def test_type_text_with_newline_can_need_confirmation():
    c, backend = make()
    res = run(c.type_text("delete\nall"))
    assert res.status == "needs_confirmation" and backend.editor_text == ""


def test_typing_into_a_non_typeable_element_gates_the_preliminary_click():
    async def scenario():
        c, backend = make()
        await c.observe()
        safe = await c.type_text("hi", target=eid(c, "File"))
        risky = await c.type_text("hi", target=eid(c, "Delete all"))
        return c, backend, safe, risky

    c, backend, safe, risky = run(scenario())
    assert safe.status == "done"
    assert c.guard.texts[0].startswith('click MenuItem "File"')
    assert risky.status == "needs_confirmation"
    assert not any("click left (310" in line for line in backend.log)   # Delete all was not clicked


def test_typing_into_a_focused_typeable_element_has_no_click_gate():
    async def scenario():
        c, backend = make()
        await c.observe()
        return c, await c.type_text("hi", target=eid(c, "Text editor"))

    c, res = run(scenario())
    assert res.status == "done" and c.guard.texts == []


def test_hotkey_risky_chord_needs_confirmation_without_asking_laya():
    c, backend = make()
    res = run(c.hotkey("alt+f4"))
    assert res.status == "needs_confirmation" and backend.log == [] and c.guard.texts == []
    assert run(c.hotkey("alt+f4", confirm=True)).ok


def test_hotkey_bad_chord_is_error():
    c, _ = make()
    assert run(c.hotkey("ctrl+nonsense")).status == "error"


def test_unchanged_screen_is_unverified():
    c, _ = make()
    res = run(c.click("1700,1000"))
    assert res.status == "unverified" and res.ok and "NOT confirmed" in res.message


def test_expect_text_confirms_an_unchanged_screen():
    c, _ = make()
    assert run(c.click("1700,1000", expect="Untitled - Notepad")).status == "done"


def test_foreground_is_restored_before_input():
    async def scenario():
        c, backend = make()
        await c.observe()
        save = eid(c, "Save", "Button")
        backend.fg = 2  # another app stole the foreground after observe()
        res = await c.click(save)
        return res, backend

    res, backend = run(scenario())
    assert res.status == "done" and "focus Untitled - Notepad" in backend.log
    assert backend.fg == 3


def test_vanished_id_is_not_found():
    c, backend = make()
    res = run(c.click("e999"))
    assert res.status == "not_found" and "observe()" in res.message and backend.log == []


def test_launch_is_gated_then_launches():
    c, backend = make()
    res = run(c.launch("notepad"))
    assert res.status == "done" and "launch notepad" in backend.log
    assert "launch" in c.guard.texts[0]
    c2, backend2 = make(guard=StubGuard())
    assert run(c2.launch("delete-everything.exe")).status == "needs_confirmation"
    assert backend2.log == []
    assert run(c2.launch("delete-everything.exe", confirm=True)).status == "done"


def test_launch_focuses_the_new_window():
    c, backend = make()
    original = backend.launch

    def launch(target):
        from laya_deploy.cua.backends.base import Rect
        from laya_deploy.cua.backends.fake import FakeWindow
        backend.windows.append(FakeWindow(9, "Calc", "calc.exe", Rect(0, 0, 100, 100)))
        return original(target)

    backend.launch = launch
    res = run(c.launch("calc"))
    assert res.status == "done" and res.target == "Calc" and backend.fg == 9


def test_focus_window_by_title_and_ambiguity():
    c, backend = make()
    assert run(c.focus_window("explorer")).status == "done" and backend.fg == 2
    assert run(c.focus_window("n")).status == "ambiguous"
    assert run(c.focus_window("calculator")).status == "not_found"


def test_scroll_wait_list_and_screenshot():
    c, backend = make()
    assert run(c.scroll("down", 3)).status == "done"
    assert any(line.startswith("scroll -3") for line in backend.log)
    assert run(c.wait(0)).ok
    assert "Notepad" in run(c.list_windows())
    assert run(c.screenshot()) is None


def test_ground_dry_run_sends_no_input():
    c, backend = make()
    out = run(c.ground("the Save button"))
    assert out["status"] == "ok" and out["element"]["name"] == "Save" and out["would_act"]
    assert out["risk_p"] == pytest.approx(0.05) and out["needs_confirmation"] is False
    assert out["revision"] and backend.log == []
    bad = run(c.ground("x", action="drag"))
    assert bad["status"] == "error"


def test_ground_dry_run_when_laya_down():
    c, _ = make(grounder=StubGrounder(error=AskError("down")))
    out = run(c.ground("the Save button"))
    assert out["status"] == "error" and not out["would_act"] and out["element"] is None


def test_default_construction_shares_one_judge_and_uses_the_asker():
    pytest.importorskip("laya_deploy.cua.guard")
    pytest.importorskip("laya_deploy.cua.grounding")
    calls = []

    async def asker(state, questions):
        calls.append(state)
        raise AskError("laya is down")

    backend = FakeBackend()
    c = LayaComputer(CuaSettings(backend="fake", settle_s=0.0), asker=asker,
                     backend_factory=lambda: backend)
    assert c.grounder.judge is c.guard.judge
    res = run(c.click("the thing that saves"))
    assert res.status == "error" and "laya-deploy serve" in res.message
    run(c.aclose())


def test_action_result_to_text_and_timing():
    r = ActionResult(True, "done", "click Save", target="[e3] Button", grounding_p=0.91, risk_p=0.1,
                     changes="window opened", revision="abc", timing_ms={"act": 5})
    text = r.to_text()
    assert text.startswith("[done] click Save") and "grounding p=0.91" in text
    assert "revision: abc" in text and "act 5ms" in text
    t = Timing()
    with t("observe"):
        pass
    assert set(t.done()) == {"observe", "total"}


# ---------------------------------------------------------------- confirmation pins what was judged
def test_needs_confirmation_pins_element_and_revision():
    async def scenario():
        c, _ = make()
        await c.observe()
        return await c.click(eid(c, "Delete all"))

    res = run(scenario())
    assert res.status == "needs_confirmation" and res.revision
    assert res.target.startswith("[e4] ")
    assert f'confirm=true, target="e4" and if_revision="{res.revision}"' in res.message


def test_hotkey_confirmation_pins_the_revision_only():
    c, _ = make()
    res = run(c.hotkey("alt+f4"))
    assert res.status == "needs_confirmation" and res.revision
    assert f'confirm=true and if_revision="{res.revision}"' in res.message and "target=" not in res.message


def test_confirmed_description_reuses_the_approved_element():
    picks = iter(["Delete all", "Save"])   # a second grounding run would choose differently

    def pick(snap):
        name = next(picks)
        el = next(e for e in snap.elements if e.name == name)
        return StubGround(el, 0.9, "ok", [(el, 0.9)])

    c, backend = make(grounder=StubGrounder(override=pick))
    backend.focus.value = "precious"
    first = run(c.click("whatever it is"))
    assert first.status == "needs_confirmation"
    second = run(c.click("whatever it is", confirm=True, if_revision=first.revision))
    assert second.status == "done" and backend.editor_text == "" and backend.fg == 1
    assert len(c.grounder.requests) == 1   # the pinned element, not a new grounding


def test_confirm_without_if_revision_grounds_again():
    c, backend = make()
    backend.focus.value = "text"
    first = run(c.click("the delete all control"))
    again = run(c.click("the delete all control", confirm=True))
    assert first.status == "needs_confirmation" and again.status == "done"
    assert len(c.grounder.requests) == 2


# ---------------------------------------------------------------- concurrency, foreground, laya errors
def test_concurrent_calls_do_not_interleave():
    class SlowGuard(StubGuard):
        active = peak = 0

        async def risk(self, action_text):
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
            await asyncio.sleep(0.03)
            type(self).active -= 1
            return StubRisk(0.05)

    async def scenario():
        c, _ = make(guard=SlowGuard())
        await asyncio.gather(c.click("1700,1000"), c.click("1700,1001"), c.observe(),
                             c.ground("1700,1002"), c.hotkey("enter"))

    run(scenario())
    assert SlowGuard.peak == 1


def unfocus_all(backend):
    real = backend.list_windows
    backend.list_windows = lambda: [replace(w, focused=False) for w in real()]


def test_no_focused_window_triggers_a_focus_attempt():
    async def scenario():
        c, backend = make()
        await c.observe()
        unfocus_all(backend)
        return await c.click("230,90"), backend

    res, backend = run(scenario())
    assert res.status == "done" and "focus Untitled - Notepad" in backend.log


def test_failed_refocus_sends_no_input():
    async def scenario():
        c, backend = make()
        await c.observe()
        unfocus_all(backend)
        backend.focus_window = lambda w: False
        return await c.click("230,90"), backend

    res, backend = run(scenario())
    assert res.status == "error" and "brought to the front" in res.message
    assert not any(line.startswith("click") for line in backend.log)


@pytest.mark.parametrize("exc", [KeyError("answers"), TypeError("bad"), ValueError("bad"), AttributeError("bad")])
def test_malformed_laya_answers_fail_closed(exc):
    c, backend = make(guard=StubGuard(error=exc))
    res = run(c.click("230,90"))
    assert res.status == "needs_confirmation" and type(exc).__name__ in res.message and backend.log == []
    c2, _ = make(grounder=StubGrounder(error=exc))
    res = run(c2.click("the Save button"))
    assert res.status == "error" and "laya-deploy serve" in res.message


def test_launch_backend_error_during_wait_is_an_error_status():
    c, backend = make()
    calls = {"n": 0}
    real = backend.list_windows

    def flaky():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("enumeration failed")
        return real()

    backend.list_windows = flaky
    res = run(c.launch("notepad"))
    assert res.status == "error" and "enumeration failed" in res.message


@pytest.mark.parametrize("app", ["notepad", "Calc", "code", "ms-settings:display", "ms-settings:",
                                 "https://example.com/a?b=1", "http://localhost:8080"])
def test_safe_launch_targets(app):
    assert ctl.is_safe_launch(app)


@pytest.mark.parametrize("app", ["cmd", "powershell", "notepad.exe", "notepad C:\\secret.txt",
                                 "C:\\tools\\run.exe", "setup.bat", "run.ps1", "x.vbs", "x.cmd",
                                 "powershell -enc AAAA", "https://a.com/x y", "file:///C:/x", "unknownapp",
                                 "https://a.com/?q=1&calc", "notepad & calc"])
def test_unsafe_launch_targets(app):
    assert not ctl.is_safe_launch(app)


def test_unsafe_launch_needs_confirmation_even_when_laya_says_safe():
    c, backend = make()
    res = run(c.launch("powershell -enc AAAA"))
    assert res.status == "needs_confirmation" and "allow-listed" in res.message and backend.log == []
    assert res.risk_p >= 0.5
    assert run(c.launch("powershell -enc AAAA", confirm=True)).status == "done"
    assert run(c.launch("ms-settings:display")).status == "done"


def test_aclose_does_not_block_the_event_loop_thread():
    async def scenario():
        c, backend = make()
        await c.observe()
        seen = []
        real = c.backend.shutdown
        c.backend.shutdown = lambda: (seen.append(threading.current_thread()), real())
        await c.aclose()
        return seen

    seen = run(scenario())
    assert len(seen) == 1 and seen[0] is not threading.main_thread()
