"""Destructive read-back (2026-08-30): "cancel all alarms" / "clear my list"
with more than one item is read back ("Cancel all three alarms, sir?") and
waits for a yes; a shaky transcript (Whisper avg_logprob under
confirm.shaky_logprob) gets one even for a single whole-list cancel. A
single named / last cancel is never read back. The yes arrives through the
follow-up window the app already opens after any spoken reply, so the
confirmation is VAD-timed for free.

Commander: real Router, mocked timekeeper. Notes: a real NotesStore at
tmp_path through the real tool. App: the confidence plumbing from
_process_audio to commander.handle.
"""
import threading
from datetime import datetime
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import jarvis.app as app_mod
from jarvis.commander import Commander, CommandResult, IntentClassifier
from jarvis.config import CONFIG
from jarvis.router import Router
from jarvis.tools.notes import NotesStore, make_tools


class Cfg:
    def __init__(self, **over):
        self.data = {"claude.big_model": "fable", "briefing.enabled": False}
        self.data.update(over)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setup_line(self, section):
        return f"I'll need {section} set up, sir."


@pytest.fixture
def rich(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    # raising=False: FEEDBACK_LOG arrives with the voice-feedback feature,
    # and this file must run on the commits before it too.
    monkeypatch.setattr(Commander, "FEEDBACK_LOG", tmp_path / "feedback.jsonl",
                        raising=False)
    for key, val in (("voice_cmds", True), ("jarvis_mode", True),
                     ("auto_type", True), ("talkback", False)):
        monkeypatch.setattr(CONFIG, key, val)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=MagicMock(), assistant=Cfg(),
        timekeeper=MagicMock(), notes=MagicMock(), approvals=MagicMock(),
        claude=MagicMock())
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.timekeeper.ringing = None
    svc.timekeeper.list.return_value = [object(), object(), object()]
    svc.timekeeper.cancel.return_value = 3
    svc.approvals.pending.return_value = []
    svc.claude.active_project = "jarvis"
    svc.brain.local_line.return_value = "Right away, sir."
    svc.router = Router(svc.assistant, classify=lambda t: ("local", 0.2))
    return Commander(svc), svc


# ------------------------------------------------------- timekeeper
def test_a_bulk_cancel_is_read_back_and_a_yes_runs_it(rich):
    c, svc = rich
    res = c.handle("cancel all my alarms", source="typed")
    assert res.reply == "Cancel all three alarms, sir?" and res.speak
    assert res.status == "Confirm?" and res.done
    svc.timekeeper.cancel.assert_not_called()

    res = c.handle("yes", source="typed")
    svc.timekeeper.cancel.assert_called_once_with("all", "alarm")
    assert res.reply == "Cancelled 3, sir."


def test_a_no_drops_the_offer_for_good(rich):
    c, svc = rich
    c.handle("cancel all alarms", source="typed")
    res = c.handle("no", source="typed")
    assert res.reply == "Very good, sir." and res.status == "Dropped"
    c.handle("yes", source="typed")               # nothing pending any more
    svc.timekeeper.cancel.assert_not_called()


def test_changing_the_subject_drops_the_offer(rich):
    """Changing the subject is not consent, and a lingering offer would
    attach the next stray "yes" to a stale cancel."""
    c, svc = rich
    c.handle("cancel all my timers", source="typed")
    c.handle("what's the weather tomorrow", source="typed")
    assert svc.brain.chat.call_args.args == ("what's the weather tomorrow",)
    c.handle("yes", source="typed")
    svc.timekeeper.cancel.assert_not_called()


def test_a_single_item_is_cancelled_outright(rich):
    c, svc = rich
    svc.timekeeper.list.return_value = [object()]
    svc.timekeeper.cancel.return_value = 1
    res = c.handle("cancel all my alarms", source="typed")
    assert res.reply == "Cancelled, sir."
    svc.timekeeper.cancel.assert_called_once_with("all", "alarm")


@pytest.mark.parametrize("said", ["cancel the timer", "stop the timer", "cancel that timer"])
def test_a_named_or_last_cancel_is_never_read_back(rich, said):
    c, svc = rich
    svc.timekeeper.cancel.return_value = 1
    res = c.handle(said, source="typed")
    assert res.reply == "Cancelled, sir."
    svc.timekeeper.cancel.assert_called_once_with("last", "timer")


def test_a_shaky_transcript_reads_back_even_one_item(rich, monkeypatch):
    """-0.9 is below confirm.shaky_logprob (-0.7): the words may be wrong,
    so the one-alarm wipe is asked first. At -0.3 it just happens."""
    c, svc = rich
    monkeypatch.setattr(c.intent, "classify", lambda t: (IntentClassifier.YES, 0.9))
    svc.timekeeper.list.return_value = [object()]
    svc.timekeeper.cancel.return_value = 1
    res = c.handle("cancel all my alarms", source="voice", confidence=-0.9)
    assert res.reply == "Cancel the alarm, sir?"
    svc.timekeeper.cancel.assert_not_called()
    res = c.handle("yes", source="voice", confidence=-0.2)
    assert res.reply == "Cancelled, sir."
    svc.timekeeper.cancel.reset_mock()
    res = c.handle("cancel all my alarms", source="voice", confidence=-0.3)
    assert res.reply == "Cancelled, sir."


def test_the_shaky_threshold_is_configurable(rich, monkeypatch):
    c, svc = rich
    monkeypatch.setattr(c.intent, "classify", lambda t: (IntentClassifier.YES, 0.9))
    svc.assistant.data["confirm.shaky_logprob"] = -0.5
    svc.timekeeper.list.return_value = [object()]
    res = c.handle("cancel all my alarms", source="voice", confidence=-0.6)
    assert res.status == "Confirm?"


def test_read_back_can_be_switched_off(rich):
    c, svc = rich
    svc.assistant.data["confirm.read_back"] = False
    res = c.handle("cancel all my alarms", source="typed")
    assert res.reply == "Cancelled 3, sir."


def test_an_old_offer_does_not_take_a_late_yes(rich):
    c, svc = rich
    c.handle("cancel all my alarms", source="typed")
    run, line, stamp = c._pending_destructive
    c._pending_destructive = (run, line, stamp - 120)
    c.handle("yes", source="typed")
    svc.timekeeper.cancel.assert_not_called()


# ------------------------------------------------------------ notes
def _store(tmp_path, *todos):
    store = NotesStore(tmp_path / "notes.db")
    for t in todos:
        store.add("todo", t)
    return store


def test_a_whole_list_clear_is_read_back_by_the_tool_and_run_by_the_commander(rich, tmp_path):
    c, svc = rich
    store = _store(tmp_path, "milk", "eggs", "bread")
    svc.notes = store
    spec, = make_tools(None, SimpleNamespace(notes=store))

    res = spec.handler(action="remove", kind="todo", which="all")
    assert res.speak == "Clear all three to-dos, sir?"
    assert store.count("todo") == 3 and isinstance(store.pending_clear, dict)

    out = c.handle("yes", source="typed")
    assert out.reply == "All cleared, sir." and out.status == "Cleared 3 todos"
    assert store.count("todo") == 0 and store.pending_clear is None


def test_a_one_item_list_is_cleared_outright(tmp_path):
    store = _store(tmp_path, "milk")
    spec, = make_tools(None, SimpleNamespace(notes=store))
    res = spec.handler(action="remove", kind="todo", which="all")
    assert store.count("todo") == 0 and store.pending_clear is None
    assert "clear" in res.speak.lower()


def test_a_new_subject_drops_the_list_offer(rich, tmp_path):
    c, svc = rich
    store = _store(tmp_path, "milk", "eggs")
    svc.notes = store
    spec, = make_tools(None, SimpleNamespace(notes=store))
    spec.handler(action="remove", kind="todo", which="all")
    c.handle("what's the weather tomorrow", source="typed")
    assert store.pending_clear is None and store.count("todo") == 2
    c.handle("yes", source="typed")
    assert store.count("todo") == 2


def test_the_list_read_back_obeys_the_switch(tmp_path):
    store = _store(tmp_path, "milk", "eggs")
    spec, = make_tools(Cfg(**{"confirm.read_back": False}),
                       SimpleNamespace(notes=store))
    res = spec.handler(action="remove", kind="todo", which="all")
    assert store.count("todo") == 0 and res.speak == "All cleared, sir."


# ------------------------------------------------------------- app
def test_the_transcript_confidence_reaches_the_commander(monkeypatch):
    """Transcribed.confidence never reached the commander: _process_audio
    hands it to _dispatch, which passes it to handle() only when it has one
    (typed text has none, and a stand-in commander need not take the
    keyword)."""
    a = object.__new__(app_mod.JarvisApp)
    a._audio_busy = threading.Event()
    a._say_again_count = 0
    a.turns = SimpleNamespace(mark=lambda *x, **k: None, abandon=lambda r: None)
    a.speaker = SimpleNamespace(enrolled=False)
    a.transcriber = SimpleNamespace(transcribe=lambda audio: SimpleNamespace(
        text="cancel all my alarms", confidence=-0.8, accepted=True))
    a._maybe_learn_voice = lambda *x: None
    monkeypatch.setattr(CONFIG, "speaker_verify", False)
    seen = []
    a._dispatch = lambda text, source, **kw: seen.append((text, source, kw))

    a._process_audio(np.zeros(16000, dtype=np.float32))

    assert seen == [("cancel all my alarms", "voice", {"confidence": -0.8})]
    assert not a._audio_busy.is_set()


def _dispatch_app(handle):
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(get=lambda k, d=None: d, user_name="Hunter")
    a._init_assistant_state()
    a.said = []
    a._say = a.said.append
    a.exchanges = []
    a.context = SimpleNamespace(add_exchange=lambda u, j: a.exchanges.append((u, j)))
    a.commander = SimpleNamespace(handle=handle)
    a.turns = SimpleNamespace(mark=lambda *x, **k: None, abandon=lambda r: None)
    a._turn_start = lambda: None
    a._turn_after_result = lambda r: None
    a._turn_finished = lambda: None
    return a


def test_dispatch_forwards_confidence_only_when_it_has_one():
    seen = []

    def handle(text, source, **kw):
        seen.append(kw)
        return CommandResult(handled=True, status="ok")
    a = _dispatch_app(handle)
    a._dispatch("hello", "voice", confidence=-0.4)
    a._dispatch("hello", "typed")
    assert seen == [{"confidence": -0.4}, {}]


def test_dispatch_records_the_exchange_under_the_corrected_text():
    """"No, I said X" / "that was for you" answered X, not the words said."""
    def handle(text, source, **kw):
        return CommandResult(handled=True, reply="Nothing on tomorrow, sir.", speak=True,
                             status="ok", corrected="what's on my calendar tomorrow")
    a = _dispatch_app(handle)
    a._dispatch("no, I said what's on my calendar tomorrow", "typed")
    assert a._last_user_text == "what's on my calendar tomorrow"
    assert a.exchanges == [("what's on my calendar tomorrow", "Nothing on tomorrow, sir.")]


def test_claim_uncertain_settles_every_open_card():
    from jarvis.events import UncertainResolved, bus
    a = object.__new__(app_mod.JarvisApp)
    a._uncertain_lock = threading.Lock()
    a._pending_uncertain = {"r1": "play some jazz"}
    got = []
    bus.subscribe(UncertainResolved, got.append)
    try:
        assert a._claim_uncertain(True) is True
        assert a._pending_uncertain == {}
        assert a._claim_uncertain(False) is False
        deadline = time.monotonic() + 2
        while not got and time.monotonic() < deadline:
            time.sleep(0.01)
        assert got and got[0].request_id == "r1" and got[0].yes is True
        assert got[0].source == "voice"
    finally:
        bus.unsubscribe(UncertainResolved, got.append)


# ------------------------------------ creation actions on a shaky transcript
# 2026-08-30: shaky_transcript() was consumed only by the bulk cancels above,
# so a misheard "5:15" vs "5:50" set the wrong alarm silently -- the daily
# cost of the confidence gate not being wired to the creation paths. Alarms,
# timers and reminders now read the PARSED result back when the transcript
# scraped in under confirm.shaky_logprob, through the same stash_destructive /
# _try_destructive_confirm machinery (so the offer expires and a change of
# subject drops it). A confident transcript is untouched.
@pytest.fixture
def clock(rich):
    c, svc = rich
    svc.timekeeper.parse_when.return_value = datetime(2026, 8, 31, 7, 15)
    svc.timekeeper.describe_due.return_value = "at 7:15 am tomorrow"
    return c, svc


def test_a_shaky_alarm_is_read_back_and_a_yes_sets_it(clock):
    c, svc = clock
    res = c.handle("set an alarm for seven fifteen", source="voice", confidence=-0.9)
    assert res.reply == "An alarm at 7:15 am tomorrow, sir?" and res.speak
    assert res.status == "Confirm?"
    svc.timekeeper.add_alarm.assert_not_called()

    res = c.handle("yes", source="voice", confidence=-0.2)
    svc.timekeeper.add_alarm.assert_called_once()
    assert res.reply == "Alarm at 7:15 am tomorrow, sir."


def test_a_confident_alarm_is_set_outright(clock):
    c, svc = clock
    res = c.handle("set an alarm for seven fifteen", source="voice", confidence=-0.3)
    svc.timekeeper.add_alarm.assert_called_once()
    assert res.reply == "Alarm at 7:15 am tomorrow, sir."
    # and a typed turn, which carries no confidence at all
    svc.timekeeper.add_alarm.reset_mock()
    c.handle("set an alarm for seven fifteen", source="typed")
    svc.timekeeper.add_alarm.assert_called_once()


def test_a_repeat_is_named_in_the_question(clock):
    c, svc = clock
    res = c.handle("set an alarm for seven fifteen every day", source="voice",
                   confidence=-0.9)
    assert res.reply == "An alarm at 7:15 am tomorrow, every day, sir?"
    c.handle("yes", source="voice")
    assert svc.timekeeper.add_alarm.call_args.args[2] == "daily"


def test_a_shaky_timer_is_read_back(clock):
    c, svc = clock
    res = c.handle("set a timer for five minutes", source="voice", confidence=-0.9)
    assert res.reply == "A timer for 5 minutes, sir?"
    svc.timekeeper.add_timer.assert_not_called()
    res = c.handle("yes", source="voice")
    svc.timekeeper.add_timer.assert_called_once()
    assert res.reply == "5 minutes, sir; I'll let you know."


def test_a_shaky_reminder_is_read_back(clock):
    c, svc = clock
    res = c.handle("remind me to call mum at 5 pm", source="voice", confidence=-0.9)
    assert res.reply == "A reminder to call mum at 7:15 am tomorrow, sir?"
    svc.timekeeper.add_reminder.assert_not_called()
    res = c.handle("yes", source="voice")
    svc.timekeeper.add_reminder.assert_called_once_with(
        datetime(2026, 8, 31, 7, 15).timestamp(), "call mum")
    assert res.reply.startswith("Very good, sir; I'll remind you to call mum")


def test_a_no_sets_nothing(clock):
    c, svc = clock
    c.handle("set an alarm for seven fifteen", source="voice", confidence=-0.9)
    res = c.handle("no", source="voice")
    assert res.reply == "Very good, sir." and res.status == "Dropped"
    c.handle("yes", source="voice")
    svc.timekeeper.add_alarm.assert_not_called()


def test_a_new_subject_drops_the_creation_offer(clock):
    """An unanswered read-back must never set an alarm later: the same rule
    as a bulk cancel, because it is the same machinery."""
    c, svc = clock
    c.handle("set an alarm for seven fifteen", source="voice", confidence=-0.9)
    c.handle("what's the weather tomorrow", source="typed")
    c.handle("yes", source="voice")
    svc.timekeeper.add_alarm.assert_not_called()


def test_read_back_off_sets_a_shaky_alarm_anyway(clock):
    c, svc = clock
    svc.assistant.data["confirm.read_back"] = False
    res = c.handle("set an alarm for seven fifteen", source="voice", confidence=-0.9)
    svc.timekeeper.add_alarm.assert_called_once()
    assert res.reply == "Alarm at 7:15 am tomorrow, sir."
