"""LayaGuard: laya risk + keyword floor, fail-closed on AskError; verify_change on fake snapshots."""
import asyncio

import pytest

from laya_deploy.client import AskError
from laya_deploy.cua.backends import create_backend
from laya_deploy.cua.guard import KEYWORD_FLOOR, LayaGuard, keyword_match, verify_change
from laya_deploy.cua.observe import IdRegistry
from laya_deploy.mode import Judge


class ScriptedAsker:
    def __init__(self, p_destructive):
        self.p, self.requests = p_destructive, []

    async def __call__(self, state, questions):
        self.requests.append((state, questions))
        (key, q), = questions.items()
        probs = {"destructive": self.p, "harmless": 1 - self.p}
        best = max(probs, key=probs.get)
        return {"answers": {key: {"type": "choice", "choice": best, "probabilities": probs}}, "latency_ms": 1.0}


def check(p_laya, text, threshold=0.5):
    ask = ScriptedAsker(p_laya)
    return asyncio.run(LayaGuard(Judge(ask), threshold).risk(text)), ask


def test_asks_laya_with_the_action_in_state():
    res, ask = check(0.2, 'click Button "Save" in Notepad')
    assert ask.requests[0][0] == {"action": 'click Button "Save"'}      # trailing context stripped (laya_subject)
    assert set(ask.requests[0][1]["q"]["criteria"]) == {"destructive", "harmless"}
    assert (res.p, res.laya_p, res.keyword, res.needs_confirmation, res.error) == (0.2, 0.2, None, False, "")


def test_laya_alone_can_require_confirmation():
    res, _ = check(0.7, "click Button 'Do it'")
    assert res.needs_confirmation and res.p == 0.7 and res.keyword is None


def test_threshold_is_configurable():
    assert not check(0.7, "click Button 'Do it'", threshold=0.8)[0].needs_confirmation
    assert check(0.3, "click Button 'Do it'", threshold=0.2)[0].needs_confirmation


@pytest.mark.parametrize("text", [
    'click Button "Delete all"', "click Button 'Remove'", "click Button Discard changes", "click Send",
    "click Post", "Publish now", "click Pay", "Complete purchase", "Buy now", "Proceed to Checkout",
    "click Submit", "Uninstall app", "Reset settings", "Format the drive D:", "Empty Recycle Bin",
    "delete permanently", "Overwrite file", "Replace", "Replace All", "Sign out", "click Don't Save",
    "刪除檔案", "移除", "送出", "傳送訊息", "發布", "付款", "購買", "解除安裝", "重設", "永久刪除", "取代", "覆寫", "登出",
])
def test_keyword_floor_catches_risky_actions_laya_misses(text):
    res, _ = check(0.05, text)
    assert res.keyword and res.p == KEYWORD_FLOOR and res.laya_p == 0.05 and res.needs_confirmation


@pytest.mark.parametrize("text", [
    'click Button "Save"', 'click Button "Cancel"', 'click MenuItem "Open"', 'click MenuItem "Edit"',
    'click MenuItem "Format"', 'click MenuItem "Replace..."', 'click MenuItem "Send to"',
    "click Edit \"Text editor\"", "press ctrl+s", "click Hyperlink \"Postcode lookup\"",
])
def test_no_false_positive_keywords(text):
    assert keyword_match(text) is None
    res, _ = check(0.05, text)
    assert not res.needs_confirmation and res.p == 0.05


def test_keyword_never_lowers_a_higher_laya_score():
    res, _ = check(0.97, "click Delete")
    assert res.p == 0.97 and res.keyword == "delete"


def test_ask_error_fails_closed_and_reports_the_error():
    async def broken(state, questions):
        raise AskError("cannot reach http://127.0.0.1:8100")

    res = asyncio.run(LayaGuard(Judge(broken)).risk("click Delete"))
    assert (res.p, res.needs_confirmation, res.keyword) == (1.0, True, "delete")
    assert "cannot reach" in res.error
    res = asyncio.run(LayaGuard(Judge(broken)).risk('click Button "Save"'))
    assert res.needs_confirmation and res.p == 1.0 and res.keyword is None and res.error


# ------------------------------------------------------------------------------ verify_change
def snap(backend):
    return IdRegistry().assign(backend.snapshot("all"))


def test_verify_change_reports_an_opened_dialog():
    backend = create_backend("fake")
    before = snap(backend)
    backend.press(["ctrl", "s"])
    changed, summary = verify_change(before, snap(backend))
    assert changed and "Save As" in summary and len(summary) < 300


def test_verify_change_reports_typed_text():
    backend = create_backend("fake")
    before = snap(backend)
    backend.type_text("hello")
    changed, summary = verify_change(before, snap(backend))
    assert changed and "hello" in summary


def test_verify_change_sees_nothing_when_nothing_happened():
    backend = create_backend("fake")
    changed, summary = verify_change(snap(backend), snap(backend))
    assert not changed and "No visible change" in summary


def test_verify_change_sees_a_focus_only_change():
    backend = create_backend("fake")
    before = snap(backend)
    backend.focus = backend.windows[0].widgets[2]       # focus moves to the Save button
    changed, summary = verify_change(before, snap(backend))
    assert changed and summary


@pytest.mark.parametrize("text,subject", [
    ('click MenuItem "檔案(F)", in MenuBar \'應用程式\', window "未命名 - 記事本"', 'click MenuItem "File"'),
    ('click Button "取消", window "另存新檔"', 'click Button "Cancel"'),
    ('click Button "Save As...", window "Notepad"', 'click Button "Save As"'),
    ('click Button "新增資料夾"', 'click Button "新增資料夾"'),          # not in the glossary: kept as is
    ('press ctrl+w, window "Mail"', 'press ctrl+w'),
    ("launch program or open 'notepad'", "launch program or open 'notepad'"),
])
def test_laya_sees_only_verb_role_and_english_label(text, subject):
    from laya_deploy.cua.guard import laya_subject

    assert laya_subject(text) == subject


def test_keyword_floor_still_sees_the_full_chinese_text():
    seen = []

    async def ask(state, questions):
        seen.append(state["action"])
        return {"answers": {"q": {"choice": "harmless", "probabilities": {"destructive": 0.1, "harmless": 0.9}}}}

    r = asyncio.run(LayaGuard(Judge(ask)).risk('click Button "刪除", window "檔案總管"'))
    assert seen == ['click Button "Delete"'] and r.keyword and r.needs_confirmation
