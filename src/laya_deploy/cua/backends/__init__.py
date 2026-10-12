"""OS backends for laya computer use (ported from decision-cua). Windows UI Automation + a fake desktop."""
from __future__ import annotations

import sys

from .base import Backend, Element, Rect, Snapshot, WindowInfo

__all__ = ["Backend", "Element", "Rect", "Snapshot", "WindowInfo", "create_backend"]


def create_backend(kind: str = "auto", click_mode: str = "mouse") -> Backend:
    """Create a backend. Must be called on the thread that will use it (COM on Windows)."""
    if kind == "auto":
        kind = "windows" if sys.platform == "win32" else "fake"
    if kind == "windows":
        from .windows import WindowsBackend

        return WindowsBackend(click_mode=click_mode)
    if kind == "fake":
        from .fake import FakeBackend

        return FakeBackend()
    raise ValueError(f"unknown backend {kind!r} (choose auto, windows or fake)")
