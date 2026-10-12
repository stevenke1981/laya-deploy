"""LayaGrounder: lexical shortlist, tournament, abstention statuses, exact-name short-circuit."""
import asyncio

import pytest

from laya_deploy.client import AskError
from laya_deploy.cua.backends.fake import FakeBackend
from laya_deploy.cua.grounding import (
    GROUND_FIELD,
    LayaGrounder,
    parse_target,
    shortlist_elements,
    text_score,
    tokens,
)
from laya_deploy.cua.observe import IdRegistry
from laya_deploy.mode import Judge


class ScriptedAsker:
    """Records requests; answers with `pick(state, question) -> {label: p}`."""

    def __init__(self, pick):
        self.pick, self.requests = pick, []

    async def __call__(self, state, questions):
        self.requests.append((state, questions))
        (key, q), = questions.items()
        probs = self.pick(state, q)
        best = max(probs, key=probs.get)
        return {"answers": {key: {"type": "choice", "choice": best, "probabilities": probs}}, "latency_ms": 1.0}


def favour(*needles, hi=0.8):
    """laya stand-in: the option whose description contains a needle gets `hi`, the rest share the remainder."""
    def pick(state, q):
        crit = q["criteria"]
        winner = next((k for k, d in crit.items() if any(n in d for n in needles)), next(iter(crit)))
        rest = (1 - hi) / max(1, len(crit) - 1)
        return {k: (hi if k == winner else rest) for k in crit}
    return pick


def uniform(state, q):
    return {k: 1 / len(q["criteria"]) for k in q["criteria"]}


def snapshot(backend=None, scope="foreground"):
    return IdRegistry().assign((backend or FakeBackend()).snapshot(scope))


def save_as_backend():
    backend = FakeBackend()
    backend.mouse_click(230, 90)      # the Save button opens the Save As dialog
    return backend


def ground(grounder, target, snap, action="click"):
    return asyncio.run(grounder.ground(target, snap, action))


def test_tokens_split_camel_case_and_cjk_characters():
    assert tokens("SaveButton") == ["save", "button"]
    assert tokens("儲存檔案 (Save)") == ["儲", "存", "檔", "案", "save"]


def test_parse_target_separates_role_words_and_stopwords():
    t = parse_target("click the Save button")
    assert t.content == ("save",) and t.roles == {"Button", "SplitButton"}
    assert parse_target('the "Delete all" menu').quoted == ("Delete all",)
    assert parse_target("儲存按鈕").roles == {"Button", "SplitButton"}


def test_shortlist_ranks_name_overlap_and_role_first():
    pool = snapshot(FakeBackend(extra_buttons=20)).elements
    short, has_text = shortlist_elements(pool, parse_target("Delete all button"), 5)
    assert has_text and short[0].name == "Delete all" and len(short) == 5
    short, has_text = shortlist_elements(pool, parse_target("the zebra"), 5)
    assert not has_text and len(short) == 5


def test_text_score_uses_automation_id_and_cjk_overlap():
    snap = snapshot()
    save = next(e for e in snap.elements if e.name == "Save")
    assert text_score(save, parse_target("SaveButton")) > 0
    assert text_score(save, parse_target("Cancel")) == 0
    save.name = "儲存檔案"
    assert text_score(save, parse_target("儲存")) > text_score(save, parse_target("刪除"))


def test_exact_unique_name_short_circuits_without_calling_laya():
    ask = ScriptedAsker(uniform)
    g = LayaGrounder(Judge(ask))
    for target in ('the "Save" button', "Save button", "click Save", '"save"'):
        r = ground(g, target, snapshot())
        assert (r.status, r.p, r.calls) == ("ok", 1.0, 0) and r.element.name == "Save"
    assert ask.requests == []


def test_short_circuit_requires_agreeing_role():
    ask = ScriptedAsker(favour('"Save"'))
    r = ground(LayaGrounder(Judge(ask)), "Save menu", snapshot())   # "Save" is a Button, not a menu item
    assert len(ask.requests) == 2 and r.calls == 2          # match round + absolute re-score of the winner


def test_short_circuit_requires_a_unique_name():
    snap = snapshot(save_as_backend(), scope="all")      # two "Save" buttons now
    assert sum(e.name == "Save" for e in snap.elements) == 2
    ask = ScriptedAsker(favour('"Save"'))
    r = ground(LayaGrounder(Judge(ask)), "Save button", snap)
    assert r.calls >= 1 and ask.requests and r.status == "ok"


def test_context_words_must_match_before_short_circuit():
    snap = snapshot(save_as_backend(), scope="all")
    g = LayaGrounder(Judge(ScriptedAsker(favour('"Cancel"'))))
    r = ground(g, 'the "Cancel" button in the Save As dialog', snap)
    assert r.calls == 0 and r.element.name == "Cancel"
    r = ground(g, 'the "Cancel" button in the Downloads folder', snap)   # "downloads" is nowhere near Cancel
    assert r.calls >= 1


def test_laya_calls_use_decision_mode_shape():
    ask = ScriptedAsker(favour("Delete all", hi=0.9))
    r = ground(LayaGrounder(Judge(ask)), "wipe the whole document", snapshot())
    state, q = ask.requests[0]
    assert state == {GROUND_FIELD: "wipe the whole document"}
    assert q["q"]["type"] == "choice" and "none" not in q["q"]["criteria"]
    crit = q["q"]["criteria"]
    assert all(" at (" not in d and "window" not in d for d in crit.values())   # describe(position=False, window=False)
    foil_state, foil_q = ask.requests[1]                    # the winner re-scored alone against a foil
    assert foil_state == state and set(foil_q["q"]["criteria"]) == {"it", "_other"}
    assert r.element.name == "Delete all" and r.status == "ok" and r.calls == 2
    assert r.candidates[0][0] is r.element and r.candidates[0][1] == r.p


def test_tournament_when_shortlist_exceeds_group():
    snap = snapshot(FakeBackend(extra_buttons=15))
    ask = ScriptedAsker(favour('"Tool 12"'))
    g = LayaGrounder(Judge(ask), shortlist=12, group=4)
    r = ground(g, "tool number 12", snap)
    sizes = [len(q["q"]["criteria"]) for _, q in ask.requests]
    assert sizes == [4, 4, 4, 3, 2]     # three groups, a final of the 3 winners, then the winner vs a foil
    assert r.element.name == "Tool 12" and r.calls == 5 and r.status == "ok"
    assert len(r.candidates) > 3                            # final ranking plus group-round losers


def test_low_confidence_when_text_overlaps_but_laya_is_unsure():
    snap = snapshot(FakeBackend(extra_buttons=3))
    r = ground(LayaGrounder(Judge(ScriptedAsker(uniform)), min_p=0.5), "Tool", snap)
    assert r.status == "low_confidence" and r.p < 0.5 and r.element is not None and len(r.candidates) >= 2


def test_no_match_when_nothing_overlaps_and_laya_is_unsure():
    r = ground(LayaGrounder(Judge(ScriptedAsker(uniform))), "zebra crossing", snapshot())
    assert r.status == "no_match" and r.element is not None and r.candidates and r.calls == 2


def test_confident_laya_answer_is_ok_even_without_lexical_overlap():
    r = ground(LayaGrounder(Judge(ScriptedAsker(favour("Delete all", hi=0.9)))), "wipe the whole document", snapshot())
    assert r.status == "ok" and r.element.name == "Delete all"


def test_no_overlap_and_a_middling_alone_score_abstains():
    """The tournament's relative winner is not enough: an absent target must not come back ok."""
    r = ground(LayaGrounder(Judge(ScriptedAsker(favour("Delete all", hi=0.8)))), "wipe the whole document", snapshot())
    assert r.status == "no_match" and r.element.name == "Delete all" and r.p == pytest.approx(0.8)


def test_final_of_two_cannot_force_ok_when_the_winner_alone_is_weak():
    def pick(state, q):
        crit = q["criteria"]
        if set(crit) == {"it", "_other"}:
            return {"it": 0.3, "_other": 0.7}                 # alone, the winner does not fit
        first = next(iter(crit))
        return {k: (0.9 if k == first else 0.1 / (len(crit) - 1)) for k in crit}

    r = ground(LayaGrounder(Judge(ScriptedAsker(pick))), "save the document now", snapshot())
    assert r.status == "low_confidence" and r.p == pytest.approx(0.3)


def test_glossary_finds_chinese_labels_for_english_targets():
    from laya_deploy.cua.backends.base import Element, Rect

    snap = snapshot()
    snap.elements.append(Element(id="e99", role="MenuItem", name="說明", window="Untitled - Notepad",
                                 rect=Rect(10, 10, 40, 30)))
    ask = ScriptedAsker(uniform)
    r = ground(LayaGrounder(Judge(ask)), "Help menu", snap)
    assert r.element.name == "說明" and r.calls == 0 and r.status == "ok"    # translated exact name
    assert parse_target("儲存按鈕").alt == ("save",)


def test_no_candidates_for_an_empty_snapshot_or_nothing_enabled():
    ask = ScriptedAsker(uniform)
    snap = snapshot()
    snap.elements = []
    r = ground(LayaGrounder(Judge(ask)), "Save", snap)
    assert (r.status, r.element, r.calls) == ("no_candidates", None, 0) and ask.requests == []
    snap = snapshot()
    for e in snap.elements:
        e.enabled = False
    assert ground(LayaGrounder(Judge(ask)), "Save", snap).status == "no_candidates"


def test_type_action_only_considers_typeable_elements():
    ask = ScriptedAsker(favour("Document"))
    r = ground(LayaGrounder(Judge(ask)), "the text area", snapshot(), action="type")
    assert r.element.role == "Document"
    assert all("Button" not in d for _, q in ask.requests for d in q["q"]["criteria"].values())


def test_cache_hits_are_not_counted_as_calls():
    g = LayaGrounder(Judge(ScriptedAsker(favour("Delete all", hi=0.9))))
    first = ground(g, "wipe the whole document", snapshot())
    second = ground(g, "wipe the whole document", snapshot())
    assert first.calls == 2 and second.calls == 0 and second.element.name == "Delete all"


def test_ask_error_propagates():
    async def broken(state, questions):
        raise AskError("down")

    with pytest.raises(AskError):
        ground(LayaGrounder(Judge(broken)), "wipe the whole document", snapshot())
