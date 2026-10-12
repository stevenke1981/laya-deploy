"""Perception and input layer of laya computer use (ported from decision-cua): ids, outline, diff, keys."""
import sys

import pytest

from laya_deploy.cua.backends import create_backend
from laya_deploy.cua.backends.fake import FakeBackend
from laya_deploy.cua.keys import is_risky_chord, parse_chord
from laya_deploy.cua.observe import (
    IdRegistry,
    diff,
    parse_point,
    render_outline,
    snapshot_json,
)
from laya_deploy.cua.settings import CuaSettings


def snap_of(backend, reg=None):
    reg = reg or IdRegistry()
    return reg.assign(backend.snapshot())


def test_ids_are_stable_across_snapshots():
    b = FakeBackend()
    reg = IdRegistry()
    s1 = snap_of(b, reg)
    save_id = next(e.id for e in s1.elements if e.name == "Save")
    b._fire("open_save")
    b._fire("cancel")
    s2 = snap_of(b, reg)
    assert next(e.id for e in s2.elements if e.name == "Save") == save_id


def test_outline_json_and_diff():
    b = FakeBackend()
    reg = IdRegistry()
    before = snap_of(b, reg)
    text = render_outline(before)
    assert 'Foreground window: "Untitled - Notepad"' in text and "[e" in text
    data = snapshot_json(before)
    assert data["revision"] and all(e["id"].startswith("e") for e in data["elements"])
    b._fire("open_save")
    after = snap_of(b, reg)
    d = diff(before, after)
    assert d["foreground_after"] == "Save As" and not d["nothing_changed"]


def test_key_chords():
    assert parse_chord("Ctrl+Shift+S") == ["ctrl", "shift", "s"]
    assert parse_chord("ctrl++") == ["ctrl", "+"]
    assert is_risky_chord(parse_chord("alt+F4"))
    with pytest.raises(ValueError):
        parse_chord("ctrl+hyperspace")


def test_parse_point():
    assert parse_point("640,360") == (640, 360)
    assert parse_point(" (12, -4) ") == (12, -4)
    assert parse_point("e12") is None and parse_point("the Save button") is None


def test_backend_factory():
    assert isinstance(create_backend("fake"), FakeBackend)
    with pytest.raises(ValueError):
        create_backend("linux")


def test_settings_defaults_and_validation(monkeypatch):
    for k in [k for k in __import__("os").environ if k.startswith("LAYA_CUA_")]:
        monkeypatch.delenv(k)
    s = CuaSettings.from_env()
    assert s.laya_model == "english" and s.ground_min_p == 0.5 and not s.auto_confirm_risky
    monkeypatch.setenv("LAYA_CUA_BACKEND", "mac")
    with pytest.raises(SystemExit):
        CuaSettings.from_env()
    monkeypatch.setenv("LAYA_CUA_BACKEND", "fake")
    monkeypatch.setenv("LAYA_CUA_RISK_THRESHOLD", "2")
    with pytest.raises(SystemExit):
        CuaSettings.from_env()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows SendInput backend")
def test_press_sends_chord_inside_ime_closed_block(monkeypatch):
    """press() must close the focused IME while the chord is sent (an open CJK IME swallows keys)."""
    from laya_deploy.cua.backends import windows as w

    events: list[str] = []

    class FakeIME:
        def __enter__(self):
            events.append("ime-closed")

        def __exit__(self, *exc):
            events.append("ime-restored")

    monkeypatch.setattr(w, "_ime_closed", lambda: FakeIME())
    monkeypatch.setattr(w, "_send", lambda *inputs: events.append(f"send:{len(inputs)}"))
    monkeypatch.setattr(w.time, "sleep", lambda _s: None)
    w.WindowsBackend().press(["ctrl", "l"])
    assert events == ["ime-closed", "send:2", "send:2", "ime-restored"]
