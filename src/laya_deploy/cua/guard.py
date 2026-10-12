"""Safety gate before input, and a cheap "did the screen change?" check after it.

Risk = laya's destructive/harmless judgment (`mode.risk`) combined with a keyword floor, because
laya missed some risky actions in our eval. A laya failure fails CLOSED: the action needs confirmation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..client import AskError
from ..mode import Judge, risk
from .backends.base import Snapshot
from .glossary import ZH_EN
from .observe import diff, snapshot_revision, summarize_diff

KEYWORD_FLOOR = 0.9
SUMMARY_LIMIT = 300

# English patterns match whole words (with common inflections). Deliberately absent: Save, Cancel,
# Open, Edit. Bare "Format" is a menu in editors, so it only counts next to a drive-like noun, and
# "Replace..." (an ellipsis opens a dialog) is a menu item, while "Replace" / "Replace All" is not.
_EN = [
    r"delet(?:e|es|ed|ing|ion)", r"remov(?:e|es|ed|ing|al)", r"discard(?:s|ed|ing)?",
    r"send(?:s|ing)?(?!\s+to\b)", r"post(?:s|ed|ing)?", r"publish(?:es|ed|ing)?",
    r"pay(?:s|ing|ment|ments)?", r"purchas(?:e|es|ed|ing)", r"buy(?:s|ing)?", r"check\s?out",
    r"submit(?:s|ted|ting)?", r"uninstall(?:s|ed|ing)?", r"reset(?:s|ting)?",
    r"format\s+(?:the\s+)?(?:drive|disk|disc|volume|partition|usb|sd\s*card|[a-z]:)",
    r"empty\s+(?:the\s+)?(?:recycle\s+bin|trash)", r"permanent(?:ly)?",
    r"overwrit(?:e|es|ing|ten)", r"replace(?:s|d|ment)?(?!\s*(?:\.\.\.|…))",
    r"sign\s?(?:out|off)", r"log\s?(?:out|off)", r"erase(?:s|d)?", r"wipe(?:s|d)?",
    r"shut\s?down", r"restart", r"don'?t\s+save", r"place\s+(?:your\s+|the\s+)?order",
]
_ZH = [
    "刪除", "移除", "送出", "傳送", "發布", "付款", "購買", "解除安裝", "重設", "永久", "取代", "覆寫", "登出",
    "清空", "結帳", "提交", "發送", "格式化", "關機", "重新啟動", "不儲存", "捨棄", "丟棄", "下單", "轉帳",
    "删除", "发送", "发布", "购买", "卸载", "重置", "替换", "覆盖", "退出登录",
]
_KEYWORD_RE = re.compile(r"(?<![a-z0-9])(?:" + "|".join(_EN) + r")(?![a-z0-9])|" + "|".join(map(re.escape, _ZH)),
                         re.IGNORECASE)


def keyword_match(action_text: str) -> str | None:
    """The first risky keyword found in the action text (lowercased), or None."""
    m = _KEYWORD_RE.search(action_text or "")
    return m.group(0).lower() if m else None


@dataclass
class RiskResult:
    p: float                  # final risk used for gating = max(laya_p, keyword floor)
    laya_p: float             # laya P(destructive)
    keyword: str | None       # matched hard-coded risky keyword, if any
    needs_confirmation: bool
    error: str = ""           # laya failure text; then p = 1.0 (fail closed)


class LayaGuard:
    def __init__(self, judge: Judge, threshold: float = 0.5) -> None:
        self.judge = judge
        self.threshold = threshold

    async def risk(self, action_text: str) -> RiskResult:
        keyword = keyword_match(action_text)
        try:
            laya_p = await risk(self.judge, laya_subject(action_text))
        except AskError as exc:
            return RiskResult(1.0, 0.0, keyword, True, error=str(exc))
        p = max(laya_p, KEYWORD_FLOOR) if keyword else laya_p
        return RiskResult(p, laya_p, keyword, p >= self.threshold)


_ELEMENT_RE = re.compile(r'^(?P<pre>.*?)(?P<role>[A-Z][A-Za-z]*) "(?P<label>[^"]*)"')
_ACCEL_RE = re.compile(r"\s*\(&?[A-Za-z0-9]\)\s*$|&|\.\.\.$|…$")


def laya_subject(action_text: str) -> str:
    """The part of an action laya judges best: verb, role and an English label, nothing else.

    Measured (english checkpoint): 'click Button "Cancel", window "另存新檔"' scored 0.58 and the
    Chinese label alone up to 0.72, while 'click Button "Cancel"' scored 0.25; risky actions kept
    0.64-0.97. The keyword floor still sees the full text, so Chinese labels and windows are not lost.
    """
    m = _ELEMENT_RE.match(action_text)
    if not m:   # keys / launch: drop the trailing window clause
        return action_text.split(', window "')[0]
    label = _ACCEL_RE.sub("", m.group("label")).strip()
    english = ZH_EN.get(label)
    return f'{m.group("pre")}{m.group("role")} "{english.capitalize() if english else label}"'


def verify_change(before: Snapshot, after: Snapshot) -> tuple[bool, str]:
    """(screen changed?, short human summary) — pure code, no laya."""
    d = diff(before, after)
    if not d["nothing_changed"]:
        return True, summarize_diff(d)[:SUMMARY_LIMIT]
    if snapshot_revision(before) != snapshot_revision(after):
        return True, "Minor change (focus, state or position of existing elements)."
    return False, summarize_diff(d)
