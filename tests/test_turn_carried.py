"""The turn's addressee is CARRIED as a value, never looked up after a wait.

Round-3 review (09-05) rejected the lane on five MEASURED holes. Four of
them are the same root cause wearing four coats: ``jarvis/scope.py`` was
process-global mutable state, WRITTEN outside ``Commander._turn_lock``
(``app._gate_admits`` / ``app._the_turn_is_his``) and READ inside it
(``Commander.handle``, after the lock). Between the write and the read
sits an UNBOUNDED wait -- the whole time another turn holds the lock --
and the reviewer measured, with a control arm, that the window is not
theoretical:

  * lock contended, guest's sentence queued, his keyboard then clears the
    scope: 200/200 trials answered the GUEST from his notes;
  * the same race the other way: 198/200 refused HIM his own notes;
  * contention with no scope flip (the control): 0/200 either way.

So the fix is not a condition patched at the leak. The reading is taken
WHERE THE TURN IS ATTRIBUTED and travels as an argument -- the shape the
lane already proved it knew, since ``brain._chat_sync`` takes
``addressee=`` for exactly this reason. Three further doors sat above or
beside the one scope read (the debrief, the gesture cast, the repeat
rung); each now asks the same question of the same carried value.

Every test here FAILED before the fix and the failure is the point: they
are the review's rows, not a description of them.

Fakes only -- MagicMock services, a tmp_path Registry, invented people.
No mic, no camera, no model, no live app, no config, no display.
"""
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.app as app_mod
from jarvis import debrief as debrief_mod
from jarvis import scope as scope_mod
from jarvis.commander import Commander, CommandResult, IntentClassifier
from jarvis.config import CONFIG
from jarvis.honorific import ADDRESSEE_TTL
from tests.test_known_tier1 import MARA_LINE, _commander, _services
from tests.test_owner_gate_wiring import _stand_in, _watching

MARA = ("Mara", "ma'am")
HIS_NOTES = "1. buy milk. 2. call the bank about the mortgage."
# A line of his the review used to prove the repeat rung re-discloses.
SENSITIVE = "Your bank balance is 412 dollars and the code is 88213, sir."


@pytest.fixture(autouse=True)
def _config(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", False)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    scope_mod.clear_addressee()
    yield
    scope_mod.clear_addressee()


# --------------------------------------------------------------- the race
def _turn_in_a_thread(c, text, source, addressee=None):
    """Run one ``handle`` on its own thread and hand back (thread, box).

    The lock is held by the CALLER, deterministically -- never a sleep
    waiting on a condition the code might not reach. The 50 ms is a fixed
    pause to let the worker arrive at the lock, and it is bounded.
    """
    box = {}
    kw = {} if addressee is None else {"addressee": addressee}

    def run():
        box["res"] = c.handle(text, source=source, **kw)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.05)
    return t, box


def test_a_queued_guest_sentence_cannot_read_his_notes_when_his_turn_clears_the_scope():
    """BLOCKER 1, the reviewer's 200/200 row, made deterministic.

    Mara is attributed; her sentence queues on a held lock; his keyboard
    turn clears the global scope; her sentence wakes. Before the fix she
    was answered "1. buy milk. 2. call the bank about the mortgage."
    """
    c, svc = _commander()
    scope_mod.set_addressee(*MARA)
    c._turn_lock.acquire()
    try:
        t, box = _turn_in_a_thread(c, "what's on my to-do list", "voice")
        scope_mod.clear_addressee()      # his keyboard, on another thread
    finally:
        c._turn_lock.release()
    t.join(timeout=10)
    assert not t.is_alive()
    assert box["res"].reply == MARA_LINE, box["res"].reply
    assert svc.notes.list_text.called is False


def test_his_queued_sentence_is_not_refused_when_a_guest_is_admitted_in_the_window():
    """BLOCKER 2, the owner lockout -- the same race the other way.

    His typed turn queues on a held lock; a guest voice turn is admitted
    in the window; his sentence wakes. Before the fix he was told "That
    one's Hunter's, Mara." about his own to-do list.
    """
    c, svc = _commander()
    scope_mod.clear_addressee()
    c._turn_lock.acquire()
    try:
        t, box = _turn_in_a_thread(c, "what's on my to-do list", "typed")
        scope_mod.set_addressee(*MARA)   # a guest admitted, on another thread
    finally:
        c._turn_lock.release()
    t.join(timeout=10)
    assert not t.is_alive()
    assert svc.notes.list_text.called is True, box["res"].reply
    assert box["res"].reply != MARA_LINE


def test_handle_takes_the_turns_addressee_as_an_argument():
    """The design item: the reading is PASSED, not looked up. A caller
    that knows whose turn this is says so, and nothing the module state
    does afterwards can change the answer."""
    c, svc = _commander()
    scope_mod.clear_addressee()          # the module says: the owner
    res = c.handle("what's on my to-do list", source="voice", addressee=MARA)
    assert res.reply == MARA_LINE
    assert svc.notes.list_text.called is False

    c, svc = _commander()
    scope_mod.set_addressee(*MARA)       # the module says: the guest
    c.handle("what's on my to-do list", source="voice",
             addressee=scope_mod.OWNER)
    assert svc.notes.list_text.called is True


def test_the_reading_is_taken_before_the_lock_not_after_it():
    """Source pin for the root cause: in ``handle``'s body the addressee
    reading must come BEFORE ``with self._turn_lock``, so no unbounded
    wait can sit between the reading and its use."""
    import inspect
    body = inspect.getsource(Commander.handle)
    read = body.index("addressee")
    lock = body.index("with self._turn_lock")
    assert read < lock, "the scope is read after the lock wait again"


# ------------------------------------------------------------- the debrief
def _debriefing(monkeypatch):
    """The real ``_debrief_reply`` and the real ``_dispatch``, with the
    open question armed. The lane's own harness stubbed this method out,
    which is why its evidence could not see the door."""
    a = SimpleNamespace()
    a.memory = MagicMock(name="memory")
    a.context = MagicMock(name="context")
    a.commander = SimpleNamespace(
        handle=lambda text, source, **kw: CommandResult(
            handled=True, reply="", speak=False, status=""))
    a.timekeeper = None
    a._pending_uncertain = None
    a.DEBRIEF_TTL_S = app_mod.JarvisApp.DEBRIEF_TTL_S
    a.DEBRIEF_FILED_LINE = app_mod.JarvisApp.DEBRIEF_FILED_LINE
    a.DEBRIEF_DECLINED_LINE = app_mod.JarvisApp.DEBRIEF_DECLINED_LINE
    cand = debrief_mod.Candidate("k1", "BIOSENSORS MIDTERM", "midterm",
                                 debrief_mod.datetime.now())
    a._pending_debrief = {"key": cand.key, "cand": cand, "at": time.monotonic()}
    monkeypatch.setattr(debrief_mod, "floor_holder", lambda app: "")
    a._debrief_reply = app_mod.JarvisApp._debrief_reply.__get__(a)
    return a, cand


def test_a_guest_cannot_answer_his_debrief_and_the_question_survives(monkeypatch):
    """BLOCKER 3. Her sentence was written to BOTH his private sinks and
    it SPENT his once-ever question -- ``mark_asked`` is already on the
    ledger, so he is never asked again."""
    a, cand = _debriefing(monkeypatch)
    filed = a._debrief_reply("honestly it was a disaster, he ran out of time",
                             "voice", MARA)
    assert filed is None
    assert a.memory.remember.called is False
    assert a.context.journal_debrief.called is False
    # The question is NOT consumed: it is his, and it is still open.
    assert a._pending_debrief is not None


def test_his_own_answer_still_files_the_debrief(monkeypatch):
    """The control -- the feature is not broken, only scoped."""
    a, cand = _debriefing(monkeypatch)
    filed = a._debrief_reply("it went fine, I finished early", "voice",
                             scope_mod.OWNER)
    assert filed is not None and filed.status.startswith("Debrief filed")
    assert a.memory.remember.called is True
    assert a.context.journal_debrief.called is True
    assert a._pending_debrief is None


# -------------------------------------------------------- the repeat rung
def test_a_guest_asking_for_a_repeat_does_not_hear_his_last_answer():
    """BLOCKER 4. ``tts.last_text`` has no owner and no age, and
    ``_handle_known`` handed it back verbatim on five phrasings."""
    for phrase in ("say that again", "what was that", "repeat that",
                   "come again", "pardon"):
        c, svc = _commander()
        svc.tts.last_text = SENSITIVE
        res = c.handle(phrase, source="voice", addressee=MARA)
        assert SENSITIVE not in (res.reply or ""), (phrase, res.reply)


def test_a_guest_may_hear_back_the_line_that_was_said_to_her():
    """The control: what she is replaying is a line the commander wrote
    FOR HER on the turn before, never ``tts.last_text``."""
    c, svc = _commander()
    svc.tts.last_text = SENSITIVE
    first = c.handle("what's on my to-do list", source="voice", addressee=MARA)
    assert first.reply == MARA_LINE
    again = c.handle("say that again", source="voice", addressee=MARA)
    assert again.reply == MARA_LINE
    assert SENSITIVE not in (again.reply or "")


def test_a_guests_repeat_is_bounded_by_the_addressee_ttl(monkeypatch):
    """Nothing tied ``last_text`` to her having been in the room when it
    was first said. Now the line she may hear back expires with her
    attribution -- one number, ``honorific.ADDRESSEE_TTL``."""
    c, svc = _commander()
    svc.tts.last_text = SENSITIVE
    c.handle("what's on my to-do list", source="voice", addressee=MARA)
    clock = [time.monotonic() + float(ADDRESSEE_TTL) + 1.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    stale = c.handle("say that again", source="voice", addressee=MARA)
    assert MARA_LINE not in (stale.reply or "")
    assert SENSITIVE not in (stale.reply or "")


def test_his_own_repeat_still_replays_his_last_line():
    """The control: the owner's repeat is untouched."""
    c, svc = _commander()
    svc.tts.last_text = SENSITIVE
    res = c.handle("say that again", source="voice",
                   addressee=scope_mod.OWNER)
    assert res.reply == SENSITIVE


# ------------------------------------------------------- the gesture cast
def test_a_guests_sentence_does_not_put_his_gesture_carry_down():
    """BLOCKER 5, the merge author's own finding: ``_cast_spoken_over``
    ran ABOVE the lock and above the scope read, so her sentence wrote to
    his live carry."""
    svc = _services()
    svc.gesture = MagicMock(name="gesture")
    c, _ = _commander(svc)
    res = c.handle("what's on my to-do list", source="voice", addressee=MARA)
    assert res.reply == MARA_LINE
    assert svc.gesture.spoken_over.called is False


def test_his_own_sentence_still_puts_a_live_carry_down():
    """The control: the feature survives for him."""
    svc = _services()
    svc.gesture = MagicMock(name="gesture")
    c, _ = _commander(svc)
    c.handle("what's on my to-do list", source="voice",
             addressee=scope_mod.OWNER)
    assert svc.gesture.spoken_over.called is True


# ----------------------------------------------------- the refused turn
def test_a_refused_voice_turn_does_not_own_the_scope(tmp_path):
    """NAMED LOWER, reviewer + verdict: ``_gate_admits`` wrote the
    attribution BEFORE it checked ``d.admit``, so a turn the gate REFUSED
    still left that person owning the process-wide scope for 120 s."""
    a = _watching(_stand_in(tmp_path, known=True), "heather")
    scope_mod.clear_addressee()
    assert a._gate_admits("read me my mail", {}) is False
    assert scope_mod.addressee() == ("", "sir")
    assert a._gate_who == ""


def test_an_admitted_turn_still_attributes(tmp_path):
    """The control: the voice half of "per turn" is unchanged."""
    a = _watching(_stand_in(tmp_path, known=True), "heather")
    assert a._gate_admits("what time is it", {}) is True
    assert scope_mod.addressee() == ("Heather", "ma'am")
    assert a._gate_addressee == ("Heather", "ma'am")


# -------------------------------------------- his conversational record
def _after(result, addressee):
    a = SimpleNamespace()
    a.context = MagicMock(name="context")
    a._reopen_mic = False
    a._followup_after_speech = False
    a._briefing_due = lambda: False
    a._briefing_pending = False
    a._consider_aside = lambda text, res: None
    a._mark_briefing_delivered = lambda: None
    a._after_dispatch = app_mod.JarvisApp._after_dispatch.__get__(a)
    a._after_dispatch("what's on my to-do list", "voice", result, addressee)
    return a


def test_a_guests_turn_is_not_written_into_his_conversation_memory():
    """NAMED LOWER: ``_after_dispatch`` filed her sentence and Jarvis's
    refusal of it into HIS record. A turn that was not his leaves no
    trace in it."""
    res = CommandResult(handled=True, reply=MARA_LINE, speak=True,
                        status="Not Mara's")
    a = _after(res, MARA)
    assert a.context.add_exchange.called is False


def test_his_own_turn_is_still_written_into_his_conversation_memory():
    res = CommandResult(handled=True, reply=HIS_NOTES, speak=True, status="Notes")
    a = _after(res, scope_mod.OWNER)
    assert a.context.add_exchange.called is True


# -------------------------------------------------- the third scope door
def test_resolve_uncertain_refuses_a_turn_that_is_not_his():
    """NAMED LOWER: ``resolve_uncertain`` takes the turn lock but took no
    scope reading, so it ran his full router whatever the scope said --
    a third path around the one read."""
    c, svc = _commander()
    scope_mod.set_addressee(*MARA)
    res = c.resolve_uncertain("what's on my to-do list", True)
    assert res.reply == MARA_LINE
    assert svc.notes.list_text.called is False


def test_resolve_uncertain_still_routes_his_own_answer():
    c, svc = _commander()
    scope_mod.clear_addressee()
    c.resolve_uncertain("what's on my to-do list", True)
    assert svc.notes.list_text.called is True


# -------------------------------------------------------- the dispatch
def test_dispatch_carries_one_reading_into_every_door(tmp_path):
    """The whole shape in one row: ``_dispatch`` takes the reading where
    the turn is attributed and hands the SAME value to the debrief, the
    commander and the bookkeeping. Nothing downstream looks it up."""
    a = _stand_in(tmp_path, known=True)
    seen = {}
    a.commander = SimpleNamespace(
        handle=lambda text, source, **kw: (
            seen.__setitem__("handle", kw.get("addressee")),
            CommandResult(handled=True, reply="", speak=False, status=""))[1])
    a._debrief_reply = lambda text, source, addressee: seen.__setitem__(
        "debrief", addressee)
    a._after_dispatch = lambda text, source, result, addressee: \
        seen.__setitem__("after", addressee)
    a._emit_result = lambda r: r
    a._the_turn_is_his = app_mod.JarvisApp._the_turn_is_his.__get__(a)
    a._dispatch = app_mod.JarvisApp._dispatch.__get__(a)
    a._last_user_text = a._last_source = ""
    a._stream_muted = False
    a._dispatch_gen = 0
    a._active_turn_id = ""
    a._turn_start = lambda: None
    a._turn_finished = lambda: None
    a._turn_after_result = lambda result: None
    a._async_turn = None
    a.turns = SimpleNamespace(mark=lambda *x: None, abandon=lambda *x: None)

    _watching(a, "heather")
    assert a._gate_admits("what time is it", {}) is True
    a._dispatch("what's on my to-do list", "voice", addressee=a._gate_addressee)
    assert seen["handle"] == ("Heather", "ma'am")
    assert seen["debrief"] == ("Heather", "ma'am")
    assert seen["after"] == ("Heather", "ma'am")

    seen.clear()
    a._dispatch("what's on my to-do list", "typed")
    assert seen["handle"] == ("", "sir")
    assert seen["debrief"] == ("", "sir")
    assert seen["after"] == ("", "sir")
