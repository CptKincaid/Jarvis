"""FOUND 2026-09-02 23:26 (Hunter, by voice): "Clear the transcript".

    23:26:01.211 transcriber  Transcribed: 'Clear the transcript'
    23:26:01.531 commander    handle 'Clear the transcript' source=voice
    23:26:01.532 commander    Ignored (background chat, conf=0.80)

Three words, no rung, so the intent gate called it background chat and
dropped it in silence.  This file covers the rung that answers it.

WHAT IS CLEARED, AND WHAT IS NOT.  The pane he is looking at, and nothing
else.  The conversation the model sees lives in jarvis/memory.py and the
context engine and is NOT touched -- wiping glass is cosmetic, wiping
context changes what Jarvis knows mid-sentence, and he asked for "the
transcript", which is the thing on screen.  The spoken line says which he
got so the difference is never left to be guessed at.

THE COLLISION.  "clear" is already his verb for his LISTS
(_LIST_CLEAR_RX: "clear the shopping list", logged twice on 08-31 and
09-01).  This repo has shipped this exact bug before -- a widened undo
grammar quietly ate "cancel that one" -- so the negative table below is
built from his REAL utterances, grepped out of /tmp/vss_voice/jarvis.log
and jarvis.log.1, and every one of them is asserted against the SHIPPING
dispatch order rather than against the new regex alone.

Display-free, the way tests/test_holo_geometry.py and
tests/test_found_standby_drag_snapback.py are: TranscriptView.clear_all is
taken UNBOUND and driven against a fake canvas that records its deletes.
No Tk root is created -- a real toplevel on the live display is the window
churn behind the 2026-08-26 desktop freeze.
"""
import types
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from jarvis.commander import (ASSISTANT_TIER1, REGISTRY, Commander,
                              IntentClassifier, TRANSCRIPT_CLEAR_LINE,
                              _TRANSCRIPT_CLEAR_RX, transcript_clear_line,
                              transcript_clear_status)
from jarvis.config import CONFIG
from jarvis.events import ClearTranscript, bus
from jarvis.router import Router
from jarvis.tools.notes import NotesStore
from jarvis.ui.views import TranscriptView


# ------------------------------------------------------------ the corpus
# Positives.  His own words first; the rest are the phrasings the same
# desk-side verb reaches for.
SAYS_CLEAR_IT = (
    "Clear the transcript",              # 2026-09-02 23:26, verbatim
    "clear the transcript",
    "clear transcript",
    "clear my transcript",
    "clear the transcript, please",
    "jarvis, clear the transcript",
    "wipe the transcript",
    "erase the transcript",
    "empty the transcript",
    "clear out the transcript",
    "reset the transcript",
    "clear the screen",
    "clear the console",
    "clear the display",
    "clear the chat",
    "clear the chat history",
    "clear the conversation",
    "wipe the screen clean",
    "clear the transcript window",
    "clear the screen for me",
    # The LEFT edge (review, 2026-09-03).  Every one of these reached NO
    # rung before the courtesy prefix went in: the first eight ended at
    # "Thinking...", handed to a model with no tool to clear anything, and
    # the rest at "Was that for me?".  "jarvis please ..." is on the list
    # because being explicitly ADDRESSED did not save it either.
    "please clear the transcript",
    "jarvis please clear the transcript",
    "jarvis, please clear the transcript",
    "please, jarvis, clear the transcript",
    "can you clear the transcript",
    "can you please clear the screen",
    "could you clear the transcript",
    "would you clear the transcript",
    "go ahead and clear the transcript",
    "let's clear the transcript",
    "just clear the transcript",
    "clear the transcript now",
    "clear the whole transcript",
    "wipe that display right now",
    # "that" was missing while "this" and "your" were in.
    "clear that transcript",
    "clear that screen",
    "clear this transcript",
    "clear our chat",
    # The TRAILING edge (review, 2026-09-03).  "thanks" was accepted and
    # "thank you" was not, so the comma he did or did not say decided
    # whether the sentence reached a rung at all: with one the compound
    # splitter rescued it, without one it went to the classifier.
    "clear the transcript thank you",
    "clear the transcript, thank you",
    "clear the screen thank you very much",
    "jarvis clear the transcript thank you",
)

# Negatives.  The first four are HIS, off the log; the rest are ordinary
# speech and neighbouring commands that happen to carry the word.  None of
# them may reach this rung.
MEANS_SOMETHING_ELSE = (
    "clear the shopping list",           # 2026-09-01 12:12, cli
    "Clear the shopping list",           # 2026-09-01 20:58, voice
    "Get rid of that shopping list.",
    "cross the second one off the list",
    "clear my shopping list",
    "clear out my packing list",
    "clear the list",
    "clear my calendar",
    "clear my schedule",
    "clear my inbox",
    "clear the notifications",
    "clear your memory",
    "forget what I just said",
    "is that clear",
    "all clear",
    "the sky is clear",
    "clear as day",
    "clear out the garage",
    "clear my head",
    "clear the air",
    "clear the log",
    "clear the history",
    "clear the screen list",
    # The same courtesies on the other side of the collision: widening the
    # LEFT edge must not have bought his lists a way in.  It cannot --
    # the language is still end-anchored on the pane nouns, and "list" is
    # in neither that set nor the tail -- and these say so out loud.
    "please clear the shopping list",
    "jarvis please clear the shopping list",
    "can you clear my grocery list",
    "could you clear the to do list",
    "go ahead and clear the shopping list",
    "just clear my shopping list",
    "clear that shopping list",
    "clear the whole shopping list",
    "clear that list",
    "clear our list",
    "clear the shopping list now",
    # ...and ordinary speech that now opens with an accepted courtesy.
    "let's clear the air",
    "just clear my head",
    "can you clear that up",
    "please clear my calendar",
    "go ahead and clear the garage",
    # ...and the trailing courtesy is a TAIL, not a way in: the pane noun
    # is still what the language is anchored on.
    "clear the shopping list thank you",
    "clear my calendar thank you",
    "thank you",
)


def _accepts(cmd, text: str) -> bool:
    try:
        return bool(cmd.matcher(text))
    except Exception:                           # noqa: BLE001 - probe only
        return False


# Rungs whose matcher accepts ANY string are fall-throughs, not claims:
# "workflow" is `lambda t: True` and leans entirely on its handler
# returning None (_try_registry carries on past a None).  Discovered by
# probing with nonsense rather than hard-coded, so a second one cannot
# quietly make these order assertions vacuous.
_FALL_THROUGH = {c.name for c in REGISTRY
                 if _accepts(c, "qwertt zzz plugh mimsy borogove")}


def _first_registry_match(text: str):
    """The name of the FIRST registry rung that CLAIMS `text` -- i.e. the
    rung that would run it.  Order is the whole point of these assertions,
    so this walks REGISTRY in the same order _try_registry does rather than
    asking one matcher in isolation."""
    for cmd in REGISTRY:
        if cmd.name in _FALL_THROUGH:
            continue
        if _accepts(cmd, text):
            return cmd.name
    return None


def _tier1_match(text: str):
    """The unprefixed pass, through the shipping probe."""
    return Commander._match_assistant(None, text)


def _spoken(text: str) -> str:
    """What the voice path hands the registry: prefix stripped, lowered."""
    from jarvis.commander import strip_jarvis_prefix
    return strip_jarvis_prefix(text) or text.strip().lower().rstrip(".!?")


# ======================================================== the grammar
def test_his_words_reach_the_new_rung_and_no_other():
    for said in SAYS_CLEAR_IT:
        assert _TRANSCRIPT_CLEAR_RX.match(_spoken(said)), said
        assert _first_registry_match(_spoken(said)) == "clear transcript", said


def test_the_named_list_family_keeps_every_clear_he_has_ever_said_to_it():
    """His logged "clear the shopping list" must still be the LIST rung's.

    Both directions: the new regex refuses it, AND the registry order puts
    "list clear" first, so even a future widening of one cannot quietly
    take the other's words."""
    for said in ("clear the shopping list", "Clear the shopping list",
                 "clear my shopping list", "clear out my packing list"):
        text = _spoken(said)
        assert not _TRANSCRIPT_CLEAR_RX.match(text), said
        assert _first_registry_match(text) == "list clear", said
        assert _tier1_match(text) == "list clear", said


def test_ordinary_speech_carrying_the_word_reaches_neither():
    for said in MEANS_SOMETHING_ELSE:
        text = _spoken(said)
        assert not _TRANSCRIPT_CLEAR_RX.match(text), said
        assert _first_registry_match(text) != "clear transcript", said
        assert _tier1_match(text) != "clear transcript", said


def test_the_rung_sits_below_the_named_lists_in_the_table():
    """Belt and braces for the assertions above: if anyone ever loosens
    _TRANSCRIPT_CLEAR_RX, the ORDER still hands "... list" to the lists."""
    names = [c.name for c in REGISTRY]
    assert names.index("clear transcript") > names.index("list clear")


def test_no_earlier_stage_takes_the_words_before_the_registry_sees_them():
    """Two stages run AHEAD of the registry on the live box and are stubbed
    out in the fixture below, so they are checked against the shipping
    functions here: the desktop chain (which owns every "switch to X" and
    would happily target a window called "the transcript") and the
    read-aloud transport, which owns "skip" / "back" / "pause" whenever
    something is being read."""
    from jarvis.commander import read_control_kind
    from jarvis.desktop import parse_desktop_action

    for said in SAYS_CLEAR_IT:
        text = _spoken(said)
        assert parse_desktop_action(text) is None, said
        assert not read_control_kind(text), said


def test_the_bare_utterance_is_tier_one_or_the_gate_eats_it_again():
    """23:26:01.532 "Ignored (background chat, conf=0.80)".  The hotword
    eats the wake word, so the words arrive bare; without a Tier-1 name the
    intent classifier drops three words in silence, exactly as it did."""
    assert "clear transcript" in [c.name for c in ASSISTANT_TIER1]
    assert _tier1_match("clear the transcript") == "clear transcript"


# ======================================================== the handler
class Cfg:
    def __init__(self):
        self.data = {"claude.big_model": "fable", "briefing.enabled": False}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setup_line(self, section):
        return f"I'll need {section} set up, sir."


@pytest.fixture
def commander(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    for key, val in (("voice_cmds", True), ("jarvis_mode", True),
                     ("auto_type", True), ("talkback", False)):
        monkeypatch.setattr(CONFIG, key, val)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=MagicMock(), assistant=Cfg(),
        timekeeper=MagicMock(), approvals=MagicMock(), claude=MagicMock(),
        notes=NotesStore(tmp_path / "notes.db"))
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.interpret_intent.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.approvals.pending.return_value = []
    svc.claude.active_project = "jarvis"
    svc.brain.local_line.return_value = "Right away, sir."
    tk = svc.timekeeper
    tk.ringing = None
    tk.parse_when.return_value = datetime.now() + timedelta(hours=9)
    svc.router = Router(svc.assistant, classify=lambda t: ("local", 0.2))
    return Commander(svc)


@pytest.fixture
def cleared():
    """Every ClearTranscript published while the test runs."""
    seen = []
    bus.subscribe(ClearTranscript, seen.append)
    yield seen
    bus.unsubscribe(ClearTranscript, seen.append)


def test_the_command_asks_the_window_to_empty_the_pane_exactly_once(
        commander, cleared):
    res = commander.handle("Jarvis, clear the transcript", source="voice")
    assert res.handled
    assert len(cleared) == 1


def test_the_reply_says_screen_only_and_that_it_does_not_come_back(
        commander, cleared):
    """Decision 1 and 2, spoken rather than left to be guessed at: he is
    told the memory is intact (so "clear the transcript" is never mistaken
    for "forget everything") and that there is no putting the cards back."""
    res = commander.handle("clear the transcript", source="voice")
    assert res.reply == TRANSCRIPT_CLEAR_LINE
    assert res.speak
    low = TRANSCRIPT_CLEAR_LINE.lower()
    assert "forgotten" in low or "forget" in low
    assert "back" in low
    assert "sir" in low


def test_a_screen_wipe_is_not_read_back_and_is_not_undoable(commander, cleared):
    """Decisions 2 and 3.  _h_list_clear stashes a destructive read-back
    because it destroys DATA; nothing is lost here, so a "are you sure"
    on a screen wipe would only be in the way.  And `undo=None` leaves
    "scratch that" its old meaning rather than offering a restore this
    rung cannot honour."""
    res = commander.handle("clear the transcript", source="voice")
    assert commander._pending_destructive is None
    assert res.undo is None
    assert len(cleared) == 1                    # ran now, not on a "yes"


def _standing(commander, n: int):
    """Make the approvals service report n questions waiting on him."""
    commander._svc("approvals").pending.return_value = [object()] * n


def test_a_question_left_standing_changes_the_words_he_hears(
        commander, cleared):
    """clear_all KEEPS an unanswered approval card on purpose -- it holds
    the only hand-answerable ALLOW / DENY for a blocked Claude run -- so
    in that one case the pane is not clear when the wipe lands.  He used
    to hear "Screen's clear, sir" and see a card, with an 1800 ms toast as
    the only correction."""
    _standing(commander, 1)
    res = commander.handle("clear the transcript", source="voice")
    assert len(cleared) == 1                    # the wipe still happens
    assert res.reply != TRANSCRIPT_CLEAR_LINE
    low = res.reply.lower()
    assert "one question" in low and "still waiting" in low
    # ...and the two facts the plain line carries are still both in it.
    assert "forgotten" in low and "bring back" in low


def test_the_status_strip_says_it_too_because_the_toast_does_not_last(
        commander, cleared):
    """The pane's own notice is a Toast, gone in 1800 ms (widgets.py).
    The status strip is where a correction can still be read afterwards,
    so the flat "Transcript cleared" is not what goes there."""
    _standing(commander, 1)
    res = commander.handle("clear the transcript", source="voice")
    assert res.status != "Transcript cleared"
    assert "1 question" in res.status and "standing" in res.status
    _standing(commander, 3)
    res = commander.handle("clear the transcript", source="voice")
    assert "3 questions" in res.status


def test_two_questions_left_standing_are_counted_not_pluralised_wrongly(
        commander, cleared):
    _standing(commander, 2)
    res = commander.handle("clear the transcript", source="voice")
    low = res.reply.lower()
    assert "2 questions" in low and "those cards" in low
    assert "one question" not in low


def test_nothing_waiting_still_gets_the_plain_line(commander, cleared):
    _standing(commander, 0)
    res = commander.handle("clear the transcript", source="voice")
    assert res.reply == TRANSCRIPT_CLEAR_LINE
    assert res.status == "Transcript cleared"


def test_a_broken_or_absent_approvals_service_never_holds_up_the_wipe(
        commander, cleared):
    """The count is a courtesy on top of the wipe, never a precondition
    for it: a service that throws, or is not there at all, must still
    empty the pane and say the plain line."""
    commander._svc("approvals").pending.side_effect = RuntimeError("down")
    res = commander.handle("clear the transcript", source="voice")
    assert res.reply == TRANSCRIPT_CLEAR_LINE
    assert len(cleared) == 1
    commander.services.approvals = None
    res = commander.handle("clear the transcript", source="voice")
    assert res.reply == TRANSCRIPT_CLEAR_LINE
    assert len(cleared) == 2


def test_an_unanswered_was_that_for_me_card_is_counted_too(commander, cleared):
    """FOUND in review 2026-09-03.  The pane has TWO producers of question
    cards and this counted one.  main_window._ev_uncertain puts "Was that
    for me?" into the SAME TranscriptView._approvals dict via add_approval,
    clear_all keeps it while it is unanswered -- and ApprovalService
    .pending() has never heard of it, so he was told "Screen's clear, sir"
    over a card still on the glass, and every later wipe kept it too."""
    _standing(commander, 0)
    commander.uncertain_open = lambda: 1
    res = commander.handle("clear the transcript", source="voice")
    assert len(cleared) == 1                    # the wipe still happens
    assert res.reply != TRANSCRIPT_CLEAR_LINE
    low = res.reply.lower()
    assert "one question" in low and "still waiting" in low
    assert "1 question" in res.status and "standing" in res.status


def test_the_two_kinds_of_card_are_added_up_not_chosen_between(
        commander, cleared):
    """A blocked Claude approval AND an unanswered prompt can be up at the
    same time -- they are different services and neither knows the other."""
    _standing(commander, 1)
    commander.uncertain_open = lambda: 1
    res = commander.handle("clear the transcript", source="voice")
    assert "2 questions" in res.reply.lower()
    assert "2 questions" in res.status


def test_a_hook_that_throws_never_holds_up_the_wipe(commander, cleared):
    """Same rule the approvals leg has kept since it went in: the count is
    a courtesy on top of the wipe, never a precondition for it."""
    def _boom():
        raise RuntimeError("app is going down")

    commander.uncertain_open = _boom
    res = commander.handle("clear the transcript", source="voice")
    assert res.reply == TRANSCRIPT_CLEAR_LINE
    assert len(cleared) == 1


def test_a_commander_with_no_services_at_all_still_wipes_the_pane():
    """FOUND in review 2026-09-03.  The count read `c._svc("approvals")`,
    and _svc is `getattr(self.services, ...)` -- so on the 13 test-shaped
    commanders built with object.__new__ it raised AttributeError BEFORE
    the try block, _try_registry answered "Command failed: clear
    transcript", and because the publish comes after the count the pane
    was never wiped at all.  The docstring said "Never raises and never
    blocks the wipe"."""
    from jarvis.commander import _h_transcript_clear, _standing_questions
    c = object.__new__(Commander)
    assert not hasattr(c, "services")
    assert _standing_questions(c) == 0
    seen = []
    bus.subscribe(ClearTranscript, seen.append)
    try:
        res = _h_transcript_clear(c, "clear the transcript", None)
    finally:
        bus.unsubscribe(ClearTranscript, seen.append)
    assert res.reply == TRANSCRIPT_CLEAR_LINE
    assert len(seen) == 1


def test_the_app_hook_reports_the_prompts_the_pane_is_still_holding():
    """The other end of the same fact: App._pending_uncertain is what
    _on_uncertain fills and uncertain_answer / _claim_uncertain empty, and
    app._ask_uncertain returns WITHOUT publishing UncertainResolved when
    the 5 s window hears nothing -- which is how the card gets stranded in
    the first place.  Taken unbound against a stand-in self, the way
    clear_all is below: no App is built and no window is opened."""
    import threading
    from jarvis.app import JarvisApp
    app = types.SimpleNamespace(_uncertain_lock=threading.Lock(),
                                _pending_uncertain={})
    assert JarvisApp._uncertain_open(app) == 0
    app._pending_uncertain["r1"] = "turn the kettle on"
    assert JarvisApp._uncertain_open(app) == 1
    # ...and before __init__ has made the dict at all.
    bare = types.SimpleNamespace(_uncertain_lock=threading.Lock())
    assert JarvisApp._uncertain_open(bare) == 0


def test_the_app_wires_that_hook_to_the_commander():
    """A count nothing calls is not a fix.  Asserted on the source beside
    the two hooks it belongs with, because building a real App opens the
    microphone, the model and a Tk window."""
    import inspect
    from jarvis.app import JarvisApp
    src = inspect.getsource(JarvisApp.__init__)
    assert "self.commander.uncertain_open = self._uncertain_open" in src
    assert "self.commander.claim_uncertain" in src


def test_the_spoken_count_is_the_count_the_pane_actually_keeps():
    """The two ends of the same fact, tied together: ApprovalService
    .pending() is what the commander counts, and `not answered` is what
    TranscriptView.clear_all keeps.  One unanswered request, one card left
    on the glass, one question in the line he hears."""
    pane = _Pane(cards=2)
    pane.add_approval_stub("req-1")             # unanswered == still pending
    pane.clear_all()
    assert len(pane._cards) == 1
    assert "one question" in transcript_clear_line(len(pane._cards)).lower()
    assert "1 question" in transcript_clear_status(len(pane._cards))


def test_the_prompt_card_is_kept_by_the_same_rule_the_approval_is():
    """Why the count has two halves.  main_window._ev_uncertain calls the
    SAME TranscriptView.add_approval, so an unanswered "Was that for me?"
    lands in _approvals and clear_all keeps it on exactly the same test --
    but ApprovalService.pending() would have reported nothing."""
    pane = _Pane(cards=2)
    pane.add_approval_stub("uncertain-1")       # add_approval, YES / NO
    pane.clear_all()
    assert len(pane._cards) == 1
    assert pane.toast.shown == [("Question left standing", "info")]


def test_his_shopping_list_still_gets_its_read_back_not_a_screen_wipe(
        commander, cleared):
    from jarvis.tools.notes import list_kind
    store = commander._svc("notes")
    kind = list_kind(store.make_list("shopping"))
    store.add_items(kind, ["milk", "eggs"])
    res = commander.handle("Jarvis, clear the shopping list", source="voice")
    assert "shopping" in (res.reply or "").lower()
    assert not cleared                          # the pane was never touched
    assert store.count(kind) == 2               # ...and neither was the list


# ======================================================== the pane
class _Widget:
    def __init__(self):
        self.destroyed = 0

    def destroy(self):
        self.destroyed += 1


class _Canvas:
    """Just the calls clear_all makes on the canvas."""

    def __init__(self, raises=False):
        self.deleted = []
        self.raises = raises

    def delete(self, win):
        if self.raises:
            import tkinter as tk
            raise tk.TclError("invalid command name")
        self.deleted.append(win)


class _Toast:
    def __init__(self):
        self.shown = []

    def show(self, text, kind="info"):
        self.shown.append((text, kind))


class _Pane:
    """A TranscriptView's clear-path state, with the two shipping methods
    bound to it.  Everything the wipe touches and nothing it does not."""

    clear_all = TranscriptView.clear_all
    clear_partial = TranscriptView.clear_partial

    def __init__(self, cards=0, partial=False, progress=False, raises=False):
        self.canvas = _Canvas(raises=raises)
        self.toast = _Toast()
        self._pinned = False                    # he had scrolled up
        self._approvals: dict = {}
        self._cards = [self._entry(i) for i in range(cards)]
        self._partial = self._entry("ghost") if partial else None
        self._progress = (self._cards[-1], ["step"]) if progress else None
        self.layouts = 0

    def _entry(self, win):
        return [_Widget(), _Widget(), "jarvis", win, 0]

    def _schedule_layout(self):
        self.layouts += 1

    def add_approval_stub(self, request_id, answered=False):
        entry = self._entry(f"ap-{request_id}")
        self._cards.append(entry)
        self._approvals[request_id] = {"stamp": _Widget(), "stamp_text": "12:00",
                                       "buttons": (), "done": False,
                                       "answered": answered, "card": entry[0]}
        return entry


def test_the_pane_empties_and_the_canvas_records_every_delete():
    pane = _Pane(cards=3)
    wins = [e[3] for e in pane._cards]
    cards = [e[0] for e in pane._cards]
    assert pane.clear_all() == 3
    assert pane._cards == []
    assert pane.canvas.deleted == wins          # every window item, in order
    assert all(c.destroyed == 1 for c in cards)
    assert pane.layouts >= 1                    # the empty stack is re-laid


def test_the_ghost_preview_and_the_open_progress_run_go_with_them():
    pane = _Pane(cards=2, partial=True, progress=True)
    ghost = pane._partial[3]
    pane.clear_all()
    assert pane._partial is None
    assert pane._progress is None               # or the next line re-opens it
    assert ghost in pane.canvas.deleted


def test_clearing_re_pins_the_view_to_the_bottom():
    """He may have scrolled up to read something before asking for the
    wipe; an empty column that is not pinned would leave the NEXT card
    landing off-screen."""
    pane = _Pane(cards=2)
    pane.clear_all()
    assert pane._pinned is True


def test_clearing_an_empty_pane_is_harmless():
    pane = _Pane(cards=0)
    assert pane.clear_all() == 0
    assert pane.canvas.deleted == []
    assert pane._cards == []
    assert pane._partial is None
    assert pane.toast.shown == []


def test_clearing_twice_running_is_harmless():
    pane = _Pane(cards=2)
    pane.clear_all()
    assert pane.clear_all() == 0


def test_a_dead_canvas_does_not_take_the_wipe_down_with_it():
    """Standby tears nothing down, but a wipe racing a window close must
    not raise inside a bus subscriber."""
    pane = _Pane(cards=2, raises=True)
    assert pane.clear_all() == 2
    assert pane._cards == []


def test_a_question_still_waiting_on_him_survives_the_wipe():
    """An unanswered approval card carries the ONLY hand-answerable ALLOW
    / DENY for a Claude run that is blocked on it.  Destroying it would
    strand that run until the approval timeout, so the wipe leaves it --
    and says so, quietly, in a toast rather than in the spoken line."""
    pane = _Pane(cards=2)
    held = pane.add_approval_stub("req-1")
    removed = pane.clear_all()
    assert removed == 2
    assert pane._cards == [held]
    assert held[0].destroyed == 0
    assert held[3] not in pane.canvas.deleted
    assert pane.toast.shown and "question" in pane.toast.shown[0][0].lower()


def test_an_answered_approval_card_goes_like_any_other():
    pane = _Pane(cards=1)
    done = pane.add_approval_stub("req-1", answered=True)
    assert pane.clear_all() == 2
    assert pane._cards == []
    assert done[0].destroyed == 1
    # the id stays registered (add_approval refuses a repeat) but the dead
    # widget reference does not
    assert "req-1" in pane._approvals
    assert pane._approvals["req-1"]["card"] is None
    assert pane.toast.shown == []


def test_the_approval_bookkeeping_marks_an_answered_question_answered():
    """clear_all reads `answered`, and resolve_approval is the only writer.
    It is set for BOTH resolution paths -- the buttons' own answer() passes
    mark=False, so `done` alone would have called a question he had just
    clicked "still waiting"."""
    pane = _Pane()
    entry = pane.add_approval_stub("req-1")
    info = pane._approvals["req-1"]
    info["buttons"] = (_Button(), _Button())
    TranscriptView.resolve_approval(pane, "req-1", True, mark=False)
    assert info["answered"] is True
    assert info["done"] is False                # unmarked, but answered
    assert pane.clear_all() == 1
    assert entry[0].destroyed == 1


class _Button:
    def __init__(self):
        self.enabled = True

    def set_enabled(self, on):
        self.enabled = on


# ======================================================== the window
def test_the_window_wipes_the_pane_when_the_event_lands():
    from jarvis.ui.main_window import MainWindow

    class _Fake:
        def __init__(self):
            self.transcript = _Pane(cards=2)

    win = _Fake()
    MainWindow._ev_transcript_clear(win, ClearTranscript())
    assert win.transcript._cards == []


def test_clearing_in_standby_does_not_crash_and_does_not_wake_the_console():
    """The slab hides the transcript in ambient AND standby (main_window
    1526-1529 pauses the atmosphere loop there).  The wipe is canvas work
    underneath it: it must neither touch the mode machine nor need the
    pane to be visible -- he may be clearing the desk on his way out."""
    from jarvis.ui.main_window import MainWindow

    class _Fake:
        def __init__(self):
            self.transcript = _Pane(cards=3)
            self.transcript._atmo_paused = True     # standby / ambient
            self.modes = MagicMock()
            self.room = MagicMock()

    win = _Fake()
    MainWindow._ev_transcript_clear(win, ClearTranscript())
    assert win.transcript._cards == []
    assert win.transcript._atmo_paused is True
    win.modes.assert_not_called()
    assert not win.room.method_calls
