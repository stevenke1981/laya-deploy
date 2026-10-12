"""Map a described UI target ("the Save button in the dialog") onto one element with laya decision mode.

Pipeline: code narrows the snapshot to plausible elements (feasibility filter, then a lexical
shortlist), laya scores the shortlist as `Judge.match` options, and code owns the abstention
threshold. There is never a "none" option: weak winners come back as `low_confidence` / `no_match`.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from ..mode import Judge
from .backends.base import Element, Snapshot
from .glossary import translations
from .observe import candidates

GROUND_FIELD = "step_intent"
GROUND_QUESTION = "Which on-screen element should the agent act on to carry out the step intent exactly?"
MAX_ALTERNATIVES = 8
# The final round compares a few winners, so its top probability is relative: with two finalists it is
# always >= 0.5 even when the target is not on screen. The winner is therefore re-scored alone against a
# foil. Measured (english): present targets 0.76-0.91, absent ones 0.25-0.71, so the foil score alone
# cannot decide; "ok" also needs text overlap with the target, or a foil score of at least STRONG_P.
FOIL = "a different on-screen element"
STRONG_P = 0.85

_CJK = "㐀-鿿぀-ヿ가-힯"
_TOKEN_RE = re.compile(rf"[{_CJK}]|[^\W_{_CJK}]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_QUOTED_RE = re.compile(r"\"([^\"]+)\"|(?<!\w)'([^']+)'(?!\w)|“([^”]+)”|「([^」]+)」")

STOPWORDS = frozenset(
    ["the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "and", "or", "that", "this", "it", "its", "click", "press", "tap", "double", "right", "left", "dialog", "window", "popup", "pane", "panel", "screen", "app", "application", "please"]
    + list("的在按點擊鍵")
)

_BUTTON = ("Button", "SplitButton")
_MENU = ("MenuItem", "Menu", "MenuBar")
_INPUT = ("Edit", "ComboBox", "Document", "Spinner")
ROLE_WORDS: dict[str, tuple[str, ...]] = {
    "button": _BUTTON, "btn": _BUTTON,
    "menu": _MENU, "menuitem": _MENU, "item": ("MenuItem", "ListItem", "TreeItem", "DataItem"),
    "box": _INPUT, "field": _INPUT, "input": _INPUT, "textbox": _INPUT, "textfield": _INPUT,
    "tab": ("TabItem",), "link": ("Hyperlink",), "hyperlink": ("Hyperlink",),
    "checkbox": ("CheckBox",), "radio": ("RadioButton",), "dropdown": ("ComboBox",), "combobox": ("ComboBox",),
    "list": ("ListItem", "List"), "listitem": ("ListItem",), "slider": ("Slider",),
}
# zh-TW / zh-CN role phrases are matched as substrings, then removed before tokenizing.
ZH_ROLE_PHRASES: dict[str, tuple[str, ...]] = {
    "按鈕": _BUTTON, "按钮": _BUTTON,
    "選單": _MENU, "菜单": _MENU, "功能表": _MENU,
    "輸入框": _INPUT, "文字方塊": _INPUT, "欄位": _INPUT,
    "分頁": ("TabItem",), "連結": ("Hyperlink",),
}

# lexical weights per element field
W_NAME, W_AUTOMATION, W_VALUE, W_PATH, W_WINDOW, W_PHRASE, W_ROLE = 3.0, 1.5, 1.0, 0.5, 0.5, 2.0, 1.0


def tokens(text: str) -> list[str]:
    """Lowercase words and single CJK characters (so Chinese matches by character overlap)."""
    return _TOKEN_RE.findall(_CAMEL_RE.sub(" ", text or "").casefold())


@dataclass(frozen=True)
class TargetTerms:
    """What a target text asks for, split into content tokens and role hints."""

    text: str                     # target with zh role phrases removed
    content: tuple[str, ...]      # tokens left after stopwords and role words
    roles: frozenset[str]         # UIA roles named by role words ("button" -> Button, SplitButton)
    quoted: tuple[str, ...]       # quoted spans, e.g. "Save"
    alt: tuple[str, ...] = ()     # the label in the other language, e.g. "File" -> ("檔案", "文件")


def parse_target(target: str) -> TargetTerms:
    roles: set[str] = set()
    text = target
    for phrase, rs in ZH_ROLE_PHRASES.items():
        if phrase in text:
            roles.update(rs)
            text = text.replace(phrase, " ")
    quoted = tuple(next(g for g in m.groups() if g).strip() for m in _QUOTED_RE.finditer(text))
    content: list[str] = []
    for tok in tokens(text):
        if tok in ROLE_WORDS:
            roles.update(ROLE_WORDS[tok])
        elif tok not in STOPWORDS:
            content.append(tok)
    alt = tuple(translations(text, content))
    return TargetTerms(text.strip(), tuple(content), frozenset(roles), quoted, alt)


def _overlap(terms: set[str], text: str) -> float:
    if not terms or not text:
        return 0.0
    return len(terms & set(tokens(text))) / len(terms)


def text_score(el: Element, terms: TargetTerms) -> float:
    """Token overlap between the target and the element's name / id / value / path / window.

    A translated label (glossary) scores like the name it stands for, so 檔案 counts for "File".
    """
    score = _score(el, terms)
    for label in terms.alt:
        score = max(score, W_NAME * _overlap(set(tokens(label)), el.name))
    return score


def _score(el: Element, terms: TargetTerms) -> float:
    want = set(terms.content)
    if not want:
        return 0.0
    score = (W_NAME * _overlap(want, el.name) + W_AUTOMATION * _overlap(want, el.automation_id)
             + W_VALUE * _overlap(want, el.value or "") + W_PATH * _overlap(want, el.path)
             + W_WINDOW * _overlap(want, el.window))
    phrase = " ".join(terms.content)
    if phrase and len(terms.content) > 1 and phrase in " ".join(tokens(el.name)):
        score += W_PHRASE
    return score


def shortlist_elements(pool: list[Element], terms: TargetTerms, limit: int) -> tuple[list[Element], bool]:
    """Top `limit` elements by lexical score (stable on ties). Second value: any text overlap at all?"""
    scored = []
    for i, el in enumerate(pool):
        txt = text_score(el, terms)
        role = W_ROLE if el.role in terms.roles else 0.0
        scored.append((txt + role, txt, i, el))
    scored.sort(key=lambda s: (-s[0], s[2]))
    return [s[3] for s in scored[:limit]], any(s[1] > 0 for s in scored)


def exact_match(pool: list[Element], terms: TargetTerms) -> Element | None:
    """The one element whose name equals the quoted/plain target (case-insensitive) and whose role agrees.

    Words left in the target that do not belong to the name (e.g. "in the Save As dialog") must show
    up in the element's window / path / role, otherwise we leave the choice to laya.
    """
    if terms.quoted:
        core = tokens(terms.quoted[0])
        rest = [t for t in tokens(_QUOTED_RE.sub(" ", terms.text)) if t not in STOPWORDS and t not in ROLE_WORDS]
    else:
        core, rest = list(terms.content), []
    if not core:
        return None
    names = [core] + ([tokens(a) for a in terms.alt] if not terms.quoted and len(core) == 1 else [])
    hits = [el for el in pool if tokens(el.name) in names and (not terms.roles or el.role in terms.roles)]
    if len(hits) != 1:
        return None
    el = hits[0]
    context = set(tokens(f"{el.window} {el.path} {el.role} {el.automation_id}"))
    return el if all(t in context for t in rest) else None


@dataclass
class GroundingResult:
    element: Element | None
    p: float
    status: str            # "ok" | "low_confidence" | "no_match" | "no_candidates"
    candidates: list[tuple[Element, float]] = field(default_factory=list)   # best first, may be empty
    calls: int = 0         # laya requests sent (cache hits not counted)


class LayaGrounder:
    """Ground a target description with a lexical shortlist, then laya `match` (tournament if long).

    Raises `client.AskError` when laya is unreachable; the caller decides how to report that.
    """

    def __init__(self, judge: Judge, min_p: float = 0.5, shortlist: int = 12, group: int = 8) -> None:
        self.judge = judge
        self.min_p = min_p
        self.shortlist = max(2, shortlist)
        self.group = max(2, group)

    async def ground(self, target: str, snap: Snapshot, action: str = "click") -> GroundingResult:
        pool = candidates(snap, action)
        if not pool:
            return GroundingResult(None, 0.0, "no_candidates")
        terms = parse_target(target)
        sure = exact_match(pool, terms)
        if sure is not None:   # unique exact name + agreeing role: certain without asking laya
            return GroundingResult(sure, 1.0, "ok", [(sure, 1.0)], calls=0)
        short, has_text = shortlist_elements(pool, terms, self.shortlist)
        calls_before = self.judge.calls
        hint = f"{target} ({' / '.join(terms.alt)})" if terms.alt else target
        ranking, losers = await self._tournament(hint, short)
        best, p = ranking[0]
        p_alone = await self._alone(hint, best)
        calls = self.judge.calls - calls_before
        extra = sorted(losers, key=lambda t: -t[1])
        alternatives = (ranking + extra)[:MAX_ALTERNATIVES]
        supported = text_score(best, terms) > 0 or p_alone >= STRONG_P
        p_final = min(p, p_alone)
        if p_final >= self.min_p and supported:
            return GroundingResult(best, p_final, "ok", alternatives, calls)
        status = "low_confidence" if has_text else "no_match"
        return GroundingResult(best, p_final, status, alternatives, calls)

    async def _alone(self, target: str, el: Element) -> float:
        """Absolute score of one element: does `target` mean this element, or something else?"""
        verdict = await self.judge.classify(target, {"it": el.describe(position=False, window=False), "_other": FOIL},
                                            field=GROUND_FIELD, question=GROUND_QUESTION)
        return verdict.probabilities.get("it", 0.0)

    async def _tournament(self, target: str, short: list[Element]
                          ) -> tuple[list[tuple[Element, float]], list[tuple[Element, float]]]:
        """Return (final-round ranking best first, eliminated elements with their group-round p)."""
        losers: list[tuple[Element, float]] = []
        round_pool = short
        while len(round_pool) > self.group:
            groups = [round_pool[i:i + self.group] for i in range(0, len(round_pool), self.group)]
            ranked = await asyncio.gather(*(self._match(target, g) for g in groups))
            round_pool = [r[0][0] for r in ranked]
            for r in ranked:
                losers.extend(r[1:])
        return await self._match(target, round_pool), losers

    async def _match(self, target: str, group: list[Element]) -> list[tuple[Element, float]]:
        keys: dict[str, Element] = {}
        for i, el in enumerate(group):
            key = el.id or f"x{i}"
            keys[key if key not in keys else f"{key}#{i}"] = el
        options = {k: el.describe(position=False, window=False) for k, el in keys.items()}
        verdict = await self.judge.match(target, options, min_p=0.0, field=GROUND_FIELD, question=GROUND_QUESTION)
        ranked = [(keys[k], p) for k, p in verdict.ranked() if k in keys]
        return ranked or [(group[0], 0.0)]
