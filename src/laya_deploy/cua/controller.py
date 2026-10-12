"""LayaComputer: the high-level actions behind the MCP tools.

The client's model plans ("click the Save button"). This controller perceives the screen,
lets laya pick the element and judge risk (laya decision mode), sends the input, and checks
that the screen changed:

    snapshot -> resolve target (id | x,y | laya grounding) -> risk gate -> act -> settle -> verify

Statuses: done | unverified | needs_confirmation | ambiguous | not_found | stale | error.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING, Any

from ..client import Asker, AskError, HttpAsker
from .backends import Backend, create_backend
from .backends.base import Element, Snapshot, WindowInfo
from .keys import is_risky_chord, parse_chord
from .observe import (
    TYPEABLE,
    IdRegistry,
    diff,
    element_dict,
    parse_point,
    point_element,
    render_outline,
    snapshot_json,
    snapshot_revision,
    summarize_diff,
)
from .settings import CuaSettings

if TYPE_CHECKING:
    from .grounding import LayaGrounder
    from .guard import LayaGuard

log = logging.getLogger("laya_deploy.cua")

LAUNCH_WAIT_S = 6.0     # how long launch() waits for a new window
LAUNCH_GRACE_S = 1.5    # after this, any new window counts even if its process name differs
ELEMENT_ID = re.compile(r"^e\d+$", re.IGNORECASE)
MAX_CANDIDATES = 6
MAX_PINS = 64           # approved descriptions remembered between a confirmation request and its repeat
# Anything laya's judge may trip over besides AskError (malformed replies from a server or judge).
LAYA_ERRORS = (AskError, KeyError, TypeError, ValueError, AttributeError)
CONFIRM_KEYS = ("\r", "\n", "\t")   # the backend types these as real Enter / Tab presses

# Chords that never do anything consequential by themselves: they skip the risk gate.
ROUTINE_CHORDS = frozenset({
    ("esc",), ("tab",), ("shift", "tab"), ("up",), ("down",), ("left",), ("right",), ("home",),
    ("end",), ("pageup",), ("pagedown",), ("f1",), ("f3",), ("f5",), ("alt",),
    ("ctrl", "a"), ("ctrl", "c"), ("ctrl", "f"), ("ctrl", "z"), ("ctrl", "y"), ("ctrl", "s"),
    ("ctrl", "shift", "s"), ("ctrl", "o"), ("ctrl", "n"), ("ctrl", "t"), ("ctrl", "l"),
    ("ctrl", "home"), ("ctrl", "end"), ("ctrl", "+"), ("ctrl", "-"), ("ctrl", "="), ("ctrl", "0"),
})
# launch() needs no confirmation only for these bare program names, settings URIs and web URLs.
SAFE_PROGRAMS = frozenset({"notepad", "calc", "mspaint", "explorer", "write", "wordpad", "charmap",
                           "snippingtool", "msedge", "chrome", "firefox", "code"})
_SETTINGS_URI = re.compile(r"ms-settings:[\w\-:/.?=&%]*", re.IGNORECASE)
_WEB_URL = re.compile(r"https?://[^\s\"'<>^|&]+", re.IGNORECASE)




class BackendThread:
    """Runs every backend call on one dedicated thread (UI Automation is COM, thread-affine)."""

    def __init__(self, factory: Callable[[], Backend]) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-backend")
        self._local = threading.local()
        self._factory = factory

    def _backend(self) -> Backend:
        b = getattr(self._local, "backend", None)
        if b is None:
            b = self._factory()
            self._local.backend = b
        return b

    async def call(self, fn_name: str, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool, lambda: getattr(self._backend(), fn_name)(*args, **kwargs))

    def shutdown(self) -> None:
        def _close() -> None:
            b = getattr(self._local, "backend", None)
            if b is not None:
                b.close()
        try:
            self._pool.submit(_close).result(timeout=5)
        except Exception:
            log.debug("backend close failed", exc_info=True)
        self._pool.shutdown(wait=False)


@dataclass
class ActionResult:
    ok: bool
    status: str
    message: str
    target: str | None = None
    grounding_p: float | None = None
    risk_p: float | None = None
    changes: str = ""
    revision: str | None = None
    timing_ms: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        parts = [f"[{self.status}] {self.message}"]
        if self.target:
            parts.append(f"target: {self.target}")
        nums = []
        if self.grounding_p is not None:
            nums.append(f"grounding p={self.grounding_p:.2f}")
        if self.risk_p is not None:
            nums.append(f"risk p={self.risk_p:.2f}")
        if nums:
            parts.append(", ".join(nums))
        if self.changes:
            parts.append(f"changes: {self.changes}")
        if self.revision:
            parts.append(f"revision: {self.revision}")
        if self.timing_ms:
            parts.append("timing: " + ", ".join(f"{k} {v}ms" for k, v in self.timing_ms.items()))
        return "\n".join(parts)


class Timing:
    """Wall-clock milliseconds per stage of one action (observe, decide, act, settle, verify)."""

    def __init__(self) -> None:
        self.ms: dict[str, int] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.ms[stage] = self.ms.get(stage, 0) + round((time.perf_counter() - t0) * 1000)

    def done(self) -> dict[str, int]:
        return {**self.ms, "total": round((time.perf_counter() - self._t0) * 1000)}


@dataclass
class _Resolved:
    element: Element | None
    p: float = 0.0
    failure: ActionResult | None = None
    candidates: list[tuple[Element, float]] = field(default_factory=list)


@dataclass
class _Risk:
    p: float
    needs: bool
    keyword: str | None = None
    error: str | None = None
    note: str | None = None


def combine_risks(risks: list[_Risk]) -> _Risk:
    """The strictest of several checks: highest p, confirmation if any check asks for it."""
    notes = [r.note for r in risks if r.note]
    return _Risk(max(r.p for r in risks), any(r.needs for r in risks),
                 next((r.keyword for r in risks if r.keyword), None),
                 next((r.error for r in risks if r.error), None), "; ".join(notes) or None)


def is_routine_chord(keys: list[str]) -> bool:
    return tuple(keys) in ROUTINE_CHORDS


def is_safe_launch(app: str) -> bool:
    a = app.strip()
    return (a.lower() in SAFE_PROGRAMS or _SETTINGS_URI.fullmatch(a) is not None
            or _WEB_URL.fullmatch(a) is not None)


def _locked(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Run an acting/observing method under the computer's lock: concurrent calls queue up."""
    @functools.wraps(fn)
    async def wrapper(self: LayaComputer, *args: Any, **kwargs: Any) -> Any:
        async with self._lock:
            return await fn(self, *args, **kwargs)
    return wrapper


def _why(exc: Exception) -> str:
    return str(exc) if isinstance(exc, AskError) else f"{type(exc).__name__}: {exc}"


def _fallback_verify(before: Snapshot, after: Snapshot) -> tuple[bool, str]:
    d = diff(before, after)
    return (not d["nothing_changed"], summarize_diff(d))


def _expect_seen(snap: Snapshot, expect: str) -> bool:
    needle = expect.strip().lower()
    if not needle:
        return False
    fg = snap.foreground.title.lower() if snap.foreground else ""
    return needle in fg or any(needle in f"{e.name} {e.value or ''}".lower() for e in snap.elements)


def _candidate_lines(cands: list[tuple[Element, float]]) -> str:
    return "\n".join(f"- [{e.id}] {e.describe(window=False)} (p={p:.2f})"
                     for e, p in cands[:MAX_CANDIDATES])


def _describe(el: Element | None) -> str:
    if el is None:
        return "(keyboard focus)"
    return f"[{el.id}] {el.describe(window=False)}" if el.id else el.describe(window=False)


class LayaComputer:
    def __init__(self, settings: CuaSettings, *, asker: Asker | None = None,
                 backend_factory: Callable[[], Backend] | None = None,
                 grounder: LayaGrounder | None = None, guard: LayaGuard | None = None) -> None:
        self.s = settings
        self.asker = asker
        self._own_asker = False
        if grounder is None or guard is None:
            grounder, guard = self._default_decisions(grounder, guard)
        self.grounder, self.guard = grounder, guard
        try:
            from .guard import verify_change
            self._verify: Callable[[Snapshot, Snapshot], tuple[bool, str]] = verify_change
        except ImportError:
            self._verify = _fallback_verify
        self.backend = BackendThread(
            backend_factory or (lambda: create_backend(settings.backend, settings.click_mode)))
        self.registry = IdRegistry()
        self.last: Snapshot | None = None
        self._lock = asyncio.Lock()   # one acting/observing call at a time
        self._pinned: dict[tuple[str, str, str], str] = {}   # (kind, description, revision) -> id

    def _default_decisions(self, grounder: Any, guard: Any) -> tuple[Any, Any]:
        """Build the missing grounder/guard over one Judge, so they share its cache."""
        from ..mode import Judge
        from .grounding import LayaGrounder
        from .guard import LayaGuard

        s = self.s
        if self.asker is None:
            self.asker, self._own_asker = HttpAsker(s.laya_url, s.laya_model, s.api_key), True
        judge = Judge(self.asker)
        return (grounder or LayaGrounder(judge, min_p=s.ground_min_p),
                guard or LayaGuard(judge, threshold=s.risk_threshold))

    async def aclose(self) -> None:
        await asyncio.to_thread(self.backend.shutdown)   # joins the backend thread: keep the loop free
        closer = getattr(self.asker, "aclose", None)
        if self._own_asker and closer is not None:
            await closer()

    # ------------------------------------------------------------------ perception
    async def _snapshot(self, scope: str = "foreground") -> Snapshot:
        snap = await self.backend.call("snapshot", scope, 600, 4.0)
        self.registry.assign(snap)
        self.last = snap
        return snap

    @_locked
    async def observe(self, full: bool = False, all_windows: bool = False, fmt: str = "text") -> str:
        """fmt='text': outline for the planner. fmt='json': ids, rects, values, revision."""
        if fmt not in ("text", "json"):
            return "[error] format must be 'text' or 'json'"
        snap = await self._snapshot("all" if all_windows else "foreground")
        if fmt == "json":
            return json.dumps(snapshot_json(snap, limit=self.s.outline_limit, full=full),
                              ensure_ascii=False)
        text = render_outline(snap, limit=self.s.outline_limit, full=full)
        return f"{text}\nrevision: {snapshot_revision(snap)}"

    @_locked
    async def ground(self, target: str, action: str = "click") -> dict[str, Any]:
        """Dry run: which element `target` resolves to and how risky acting on it looks. No input."""
        if action not in ("click", "type", "scroll"):
            return {"status": "error", "message": "action must be click, type or scroll"}
        timing = Timing()
        prev = self.last
        with timing("observe"):
            snap = await self._snapshot()
        with timing("decide"):
            res = await self._resolve(target, snap, "type" if action == "type" else "click", prev)
            risk = await self._risk(self._action_text(action, res.element)) if res.element else None
        out: dict[str, Any] = {
            "status": res.failure.status if res.failure else "ok",
            "message": res.failure.message if res.failure else "resolved",
            "p": round(res.p, 4),
            "element": element_dict(res.element) if res.element else None,
            "candidates": [{"id": e.id, "p": round(p, 4), "describe": e.describe(window=False)}
                           for e, p in res.candidates[:8]],
            "risk_p": round(risk.p, 4) if risk else None,
            "needs_confirmation": risk.needs and not self.s.auto_confirm_risky if risk else None,
            "would_act": res.failure is None,
            "revision": snapshot_revision(snap),
            "timing_ms": timing.done(),
        }
        if risk and risk.error:
            out["risk_error"] = risk.error
        return out

    async def list_windows(self) -> str:
        wins: list[WindowInfo] = await self.backend.call("list_windows")
        return "\n".join(f"- {w.describe()}" for w in wins) or "(no windows)"

    @_locked
    async def screenshot(self, annotate: bool = True) -> bytes | None:
        shot = await self.backend.call("screenshot")
        if shot is None:
            return None
        png, origin = shot
        if not annotate or self.last is None:
            return png
        try:
            from .vision import annotate as draw
            return draw(png, origin, self.last.elements)
        except ImportError:
            return png

    # ------------------------------------------------------------------ windows
    @_locked
    async def focus_window(self, query: str) -> ActionResult:
        wins: list[WindowInfo] = await self.backend.call("list_windows")
        q = query.strip().lower()
        matches = [w for w in wins if q and q in w.title.lower()]
        if len(matches) > 1:
            matches = [w for w in matches if w.title.lower() == q] or matches
        if len(matches) != 1:
            if not wins:
                return ActionResult(False, "not_found", "No windows are open.")
            listing = "\n".join(f"- {w.describe()}" for w in (matches or wins)[:15])
            return ActionResult(False, "ambiguous" if matches else "not_found",
                                f"{len(matches)} windows match {query!r}. "
                                f"Use a more specific title. Windows:\n{listing}")
        win = matches[0]
        ok = await self.backend.call("focus_window", win)
        await asyncio.sleep(self.s.settle_s / 2)
        return ActionResult(ok, "done" if ok else "error",
                            f'Focused "{win.title}"' if ok else f'Could not bring "{win.title}" to front',
                            target=win.title)

    @_locked
    async def launch(self, app: str, confirm: bool = False) -> ActionResult:
        risk = await self._risk(f"launch program or open {app!r}")
        if not is_safe_launch(app):   # arguments, paths, scripts, unlisted programs: always ask
            unlisted = _Risk(self.s.risk_threshold, True,
                             note="not a plain allow-listed program, settings URI or web URL")
            risk = combine_risks([risk, unlisted])
        gate = self._gate(risk, f"launch {app!r}", confirm)
        if gate:
            return gate
        try:
            known = {w.handle for w in await self.backend.call("list_windows")}
            if not await self.backend.call("launch", app):
                return ActionResult(False, "error", f"Could not launch {app!r}", risk_p=risk.p)
            new = await self._wait_new_window(known, app)
            await asyncio.sleep(self.s.settle_s)
            brought = new is not None and await self._bring_up(new)
        except Exception as exc:  # noqa: BLE001 - OS-level launch failure
            return ActionResult(False, "error", f"launch failed: {exc}", risk_p=risk.p)
        if new is None:
            msg = (f"Launched {app!r}, but no new window appeared within {LAUNCH_WAIT_S:g}s "
                   "(it may have reused an existing window). Check the foreground before typing.")
        elif brought:
            msg = f'Launched {app!r}; "{new.title}" is in the foreground'
        else:
            msg = (f'Launched {app!r}, but "{new.title}" could not be brought to the foreground; '
                   "call focus_window before acting")
        return ActionResult(True, "done", msg, target=new.title if new else None, risk_p=risk.p)

    async def _wait_new_window(self, known: set[Any], app: str) -> WindowInfo | None:
        """A new window, preferring one whose process matches the launched app."""
        stem = PureWindowsPath(app.strip('"')).stem.lower()
        start = time.monotonic()
        while True:
            await asyncio.sleep(0.25)
            wins: list[WindowInfo] = await self.backend.call("list_windows")
            new = [w for w in wins if w.handle not in known]
            own = [w for w in new if stem and PureWindowsPath(w.process).stem.lower() == stem]
            waited = time.monotonic() - start
            if own or (new and waited >= LAUNCH_GRACE_S) or waited >= LAUNCH_WAIT_S:
                pool = own or new
                return next((w for w in pool if w.focused), pool[0] if pool else None)

    async def _bring_up(self, win: WindowInfo) -> bool:
        """Focus `win` if needed, then confirm from a fresh window list that it is in front."""
        wins: list[WindowInfo] = await self.backend.call("list_windows")
        current = next((w for w in wins if w.handle == win.handle), None)
        if current is None:
            return False
        if not current.focused and not await self.backend.call("focus_window", current):
            return False
        wins = await self.backend.call("list_windows")
        return any(w.handle == win.handle and w.focused for w in wins)

    # ------------------------------------------------------------------ actions
    async def click(self, target: str, button: str = "left", double: bool = False,
                    expect: str | None = None, confirm: bool = False,
                    if_revision: str | None = None) -> ActionResult:
        if button not in ("left", "right", "middle"):
            return ActionResult(False, "error", "button must be left, right or middle")
        verb = "double-click" if double else "right-click" if button == "right" else "click"

        async def act(el: Element | None) -> None:
            assert el is not None and el.rect is not None
            if (self.s.click_mode in ("invoke", "auto") and button == "left" and not double
                    and el.role != "Point" and await self.backend.call("invoke", el)):
                return
            rect = await self.backend.call("current_rect", el) or el.rect
            await self.backend.call("mouse_click", *rect.center, button, double)

        return await self._perform("click", verb, target, act, expect=expect, confirm=confirm,
                                   if_revision=if_revision, gated=True)

    async def type_text(self, text: str, target: str | None = None, clear: bool = False,
                        enter: bool = False, confirm: bool = False,
                        if_revision: str | None = None) -> ActionResult:
        shown = text if len(text) <= 80 else text[:77] + "..."
        verb = ("clear the field, then " if clear else "") + f"type {shown!r}"
        verb += " and press Enter" if enter else ""

        async def act(el: Element | None) -> None:
            if el is not None and el.rect is not None and not el.focused:
                await self.backend.call("mouse_click", *el.rect.center, "left", False)
                await asyncio.sleep(0.1)
            if clear and (el is None or not await self.backend.call("set_value", el, "")):
                await self.backend.call("press", ["ctrl", "a"])
                await self.backend.call("press", ["delete"])
            await self.backend.call("type_text", text)
            if enter:
                await asyncio.sleep(0.05)
                await self.backend.call("press", ["enter"])

        # Plain typing is low risk. Enter/Tab characters, clearing and submitting are not.
        gated = enter or clear or any(c in text for c in CONFIRM_KEYS)
        return await self._perform("type", verb, target, act, expect=None, confirm=confirm,
                                   if_revision=if_revision, gated=gated)

    async def hotkey(self, keys: str, confirm: bool = False,
                     if_revision: str | None = None) -> ActionResult:
        try:
            chord = parse_chord(keys)
        except ValueError as exc:
            return ActionResult(False, "error", str(exc))

        async def act(_: Element | None) -> None:
            await self.backend.call("press", chord)

        risky = is_risky_chord(chord)
        return await self._perform("keys", f"press {'+'.join(chord)}", None, act, expect=None,
                                   confirm=confirm, if_revision=if_revision,
                                   gated=risky or not is_routine_chord(chord), always_risky=risky)

    @_locked
    async def scroll(self, direction: str = "down", amount: int = 5,
                     target: str | None = None) -> ActionResult:
        clicks = abs(int(amount)) * (1 if direction.lower() in ("up", "u") else -1)
        prev = self.last
        snap = await self._snapshot()
        if target:
            res = await self._resolve(target, snap, "click", prev)
            if res.failure or res.element is None or res.element.rect is None:
                return res.failure or ActionResult(False, "not_found", f"No element matches {target!r}")
            rect = res.element.rect
        else:
            fg = snap.foreground
            rect = fg.rect if fg and fg.rect else snap.screen
        await self.backend.call("scroll", *rect.center, clicks)
        await asyncio.sleep(self.s.settle_s / 2)
        after = await self._snapshot()
        return ActionResult(True, "done", f"Scrolled {direction} {abs(clicks)}",
                            changes=self._verify(snap, after)[1], revision=snapshot_revision(after))

    async def wait(self, seconds: float) -> ActionResult:
        await asyncio.sleep(max(0.0, min(float(seconds), 30.0)))
        return ActionResult(True, "done", f"Waited {seconds}s")

    # ------------------------------------------------------------------ internals
    @_locked
    async def _perform(self, kind: str, verb: str, target: str | None,
                       act: Callable[[Element | None], Awaitable[None]], *, expect: str | None,
                       confirm: bool, if_revision: str | None, gated: bool,
                       always_risky: bool = False) -> ActionResult:
        timing = Timing()
        prev = self.last
        with timing("observe"):
            before = await self._snapshot()
        stale = self._stale(before, if_revision, timing)
        if stale:
            return stale
        revision = snapshot_revision(before)
        spec = target.strip() if target else None
        is_description = bool(spec) and not ELEMENT_ID.match(spec) and parse_point(spec) is None
        pin_key = (kind, spec or "", (if_revision or "").strip())
        if confirm and is_description and pin_key in self._pinned:
            target = spec = self._pinned[pin_key]   # the element the user was asked about
            is_description = False
        with timing("decide"):
            if target is None:
                res = _Resolved(before.focused, 1.0)
            else:
                res = await self._resolve(target, before, "type" if kind == "type" else "click", prev)
            if res.failure:
                res.failure.timing_ms = timing.done()
                return res.failure
            el = res.element
            risk = await self._assess(kind, verb, el, before, gated, always_risky)
        desc = _describe(el)
        if risk is not None:
            text = self._action_text(verb, None if kind == "keys" else el, before, kind)
            pin = (el.id or spec) if target is not None and el is not None else None
            gate = self._gate(risk, text, confirm, target=desc, grounding_p=res.p,
                              revision=revision, pin=pin)
            if gate:
                if is_description and el is not None and el.id:
                    self._pin(kind, spec or "", revision, el.id)
                gate.timing_ms = timing.done()
                return gate
        stale = await self._recheck(if_revision, timing)
        if stale:
            return stale
        with timing("act"):
            try:
                await self._ensure_foreground(el, before)
                await act(el)
            except Exception as exc:  # noqa: BLE001 - OS-level failure (UIPI, element vanished, ...)
                return ActionResult(False, "error", f"{verb} failed: {exc}", target=desc,
                                    timing_ms=timing.done())
        return await self._finish(verb, desc, expect, before, res.p if target else None,
                                  risk.p if risk else None, timing, typing=kind == "type")

    def _pin(self, kind: str, spec: str, revision: str, eid: str) -> None:
        """Remember which element a description meant when confirmation was asked."""
        if len(self._pinned) >= MAX_PINS:
            self._pinned.pop(next(iter(self._pinned)))
        self._pinned[(kind, spec, revision)] = eid

    async def _assess(self, kind: str, verb: str, el: Element | None, snap: Snapshot,
                      gated: bool, always_risky: bool) -> _Risk | None:
        """Risk of the action, or None when nothing needs a check."""
        if always_risky:
            return _Risk(1.0, True, note="risky key chord")
        texts = []
        if gated:
            texts.append(self._action_text(verb, None if kind == "keys" else el, snap, kind))
        if kind == "type" and el is not None and not el.focused and el.role not in TYPEABLE:
            texts.append(self._action_text("click", el))   # typing here starts with a click
        if not texts:
            return None
        return combine_risks([await self._risk(t) for t in texts])

    @staticmethod
    def _action_text(verb: str, el: Element | None, snap: Snapshot | None = None,
                     kind: str = "click") -> str:
        """Plain description of the action for the risk judge, e.g. 'click Button "Delete all", window "X"'."""
        if el is None:
            fg = snap.foreground.title if snap and snap.foreground else "unknown"
            return f'{verb}, window "{fg}"'
        window = f', window "{el.window}"' if el.window else ""
        joiner = " into " if kind == "type" else " "
        return f"{verb}{joiner}{el.describe(position=False, window=False)}{window}"

    async def _finish(self, verb: str, desc: str, expect: str | None, before: Snapshot,
                      grounding_p: float | None, risk_p: float | None, timing: Timing,
                      typing: bool) -> ActionResult:
        with timing("settle"):
            await asyncio.sleep(self.s.settle_s)
        with timing("observe"):
            after = await self._snapshot()
        with timing("verify"):
            changed, summary = self._verify(before, after)
        confirmed = changed or (expect is not None and _expect_seen(after, expect)) or (
            typing and expect is None)
        msg = verb if confirmed else f"{verb} - outcome NOT confirmed ({summary}); re-observe before continuing"
        return ActionResult(True, "done" if confirmed else "unverified", msg, target=desc,
                            grounding_p=grounding_p, risk_p=risk_p, changes=summary,
                            revision=snapshot_revision(after), timing_ms=timing.done())

    def _stale(self, snap: Snapshot, if_revision: str | None, timing: Timing) -> ActionResult | None:
        """Refuse to act when the caller decided on a screen state that is no longer current."""
        if not if_revision:
            return None
        current = snapshot_revision(snap)
        if current == if_revision.strip():
            return None
        return ActionResult(False, "stale",
                            f"The screen changed since revision {if_revision} (now {current}). Nothing was "
                            "done. Observe again and re-decide.", revision=current, timing_ms=timing.done())

    async def _recheck(self, if_revision: str | None, timing: Timing) -> ActionResult | None:
        """Deciding takes time: confirm the state again right before input."""
        if not if_revision:
            return None
        with timing("observe"):
            now = await self._snapshot()
        return self._stale(now, if_revision, timing)

    async def _risk(self, text: str) -> _Risk:
        """Risk of an action. Fails closed: any laya failure requires confirmation."""
        try:
            r = await self.guard.risk(text)
            error = getattr(r, "error", None)
            return _Risk(r.p, bool(r.needs_confirmation) or bool(error), r.keyword, error)
        except LAYA_ERRORS as exc:
            return _Risk(1.0, True, error=_why(exc))

    def _gate(self, risk: _Risk, detail: str, confirm: bool, target: str | None = None,
              grounding_p: float | None = None, revision: str | None = None,
              pin: str | None = None) -> ActionResult | None:
        if not risk.needs or confirm or self.s.auto_confirm_risky:
            return None
        why = (f"laya could not be reached to judge the risk ({risk.error})" if risk.error
               else f"risk p={risk.p:.2f}" + (f", matched {risk.keyword!r}" if risk.keyword else ""))
        if risk.note:
            why = f"{risk.note}; {why}"
        args = ["confirm=true"]
        if pin:
            args.append(f'target="{pin}"')
        if revision:
            args.append(f'if_revision="{revision}"')
        repeat = args[0] if len(args) == 1 else ", ".join(args[:-1]) + " and " + args[-1]
        return ActionResult(False, "needs_confirmation",
                            f"This looks consequential: {detail} ({why}). Ask the user, then repeat "
                            f"the same call with {repeat} if they agree.",
                            target=target, risk_p=risk.p, grounding_p=grounding_p, revision=revision)

    async def _resolve(self, target: str, snap: Snapshot, action: str,
                       prev: Snapshot | None) -> _Resolved:
        """Element id / 'x,y' need no laya; anything else is grounded by laya."""
        spec = target.strip()
        if ELEMENT_ID.match(spec):
            return self._resolve_id(spec.lower(), snap, prev)
        point = parse_point(spec)
        if point is not None:
            if not snap.screen.contains(*point):
                return _Resolved(None, failure=ActionResult(
                    False, "not_found", f"Point {spec} is outside the screen {snap.screen}."))
            return _Resolved(point_element(snap, *point), 1.0)
        try:
            g = await self.grounder.ground(spec, snap, action)
            cands = list(g.candidates)
            found = g.status == "ok" and g.element is not None
        except LAYA_ERRORS as exc:
            return _Resolved(None, failure=ActionResult(
                False, "error", f"Cannot reach laya to ground {spec!r} ({_why(exc)}). Start "
                "`laya-deploy serve`, or target an element id like 'e12' or a point 'x,y'."))
        if found:
            return _Resolved(g.element, g.p, candidates=cands)
        listing = f"\nCandidates:\n{_candidate_lines(cands)}" if cands else ""
        if g.status == "low_confidence":
            fail = ActionResult(False, "ambiguous", f"Not confident which element {spec!r} means "
                                f"(best p={g.p:.2f}).{listing}\nRetry with an element id like 'e12' "
                                "or a more specific description.", grounding_p=g.p)
        else:
            fail = ActionResult(False, "not_found", f"No on-screen element matches {spec!r}.{listing}\n"
                                "Observe again, scroll, switch windows, or use a keyboard shortcut.")
        fail.extra["candidates"] = [{"id": e.id, "p": round(p, 4), "describe": e.describe(window=False)}
                                    for e, p in cands[:MAX_CANDIDATES]]
        return _Resolved(None, g.p, fail, cands)

    @staticmethod
    def _resolve_id(eid: str, snap: Snapshot, prev: Snapshot | None) -> _Resolved:
        el = snap.by_id(eid)
        if el is None and prev is not None:  # observed in another window that is still open
            old = prev.by_id(eid)
            if old is not None and any(w.title == old.window for w in snap.windows):
                el = old
        if el is None:
            return _Resolved(None, failure=ActionResult(
                False, "not_found", f"Element {eid} is no longer on screen. Call observe() again."))
        return _Resolved(el, 1.0)

    async def _ensure_foreground(self, el: Element | None, snap: Snapshot) -> None:
        """Input lands in the foreground window; bring back the window the target was observed in.

        Raises (so no input is sent) when the target window cannot be brought to the front.
        """
        fg = snap.foreground
        target = None
        if el is not None and el.window and fg is not None and el.window not in (fg.title, f"{fg.title} (popup)"):
            target = next((w for w in snap.windows if w.title == el.window), None)
        target = target or fg
        if target is None:
            return
        windows: list[WindowInfo] = await self.backend.call("list_windows")
        now = next((w for w in windows if w.focused), None)
        if now is not None and now.handle == target.handle:
            return
        if not await self.backend.call("focus_window", target):
            holder = f'"{now.title}"' if now else "no known window"
            raise RuntimeError(f'{holder} has the foreground and "{target.title}" could not be '
                               "brought to the front")
        await asyncio.sleep(self.s.settle_s / 2)
