"""Turn-flow defects confirmed by the 2026-08-30 review of the feature set.

Each test names the live symptom it pins. The app is the real JarvisApp
with only hardware and peers stubbed, as in tests/test_assistant_features.
"""
import json
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import jarvis.app as app_mod
from jarvis.config import CONFIG
from jarvis.events import SpeakingState


def _app(monkeypatch, tmp_path, **over):
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(get=lambda k, d=None: {"briefing.on_first_wake": True,
                                                         "briefing.after": "06:00"}.get(k, d),
                                  user_name="Hunter")
    a._init_assistant_state()
    a.recorder = SimpleNamespace(endpointer=object(), recording=False,
                                 start=lambda followup=False: a.starts.append(followup))
    a.starts = []
    a._audio_busy = threading.Event()
    a._turn_busy = threading.Event()
    a._turn_timer = a._turn_watchdog = None
    a._pending_uncertain = {}
    a._uncertain_lock = threading.Lock()
    a._thinking_i = 0
    a._thinking_delay_s = 0.05
    a._turn_timeout_s = 30.0
    a._tts_active = False
    a._turn_filler_pending = False
    a._last_source, a._last_user_text = "voice", "hello"
    a.said = []
    a._say = a.said.append
    a.context = SimpleNamespace(add_exchange=lambda u, j: a.exchanges.append((u, j)))
    a.exchanges = []
    a.services = SimpleNamespace(brain=SimpleNamespace(chat=lambda t, **kw: a.chats.append((t, kw))))
    a.chats = []
    a.marks = []
    a.turns = SimpleNamespace(mark=lambda *x, **k: a.marks.append(x),
                              abandon=lambda r: a.marks.append(("abandon", r)))
    state = tmp_path / "briefing.json"
    a._briefing_state_path = lambda: state
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _delivered_today(tmp_path):
    (tmp_path / "briefing.json").write_text(
        json.dumps({"delivered": datetime.now().date().isoformat()}))


def _due(monkeypatch, a):
    """A briefing is due: the clock is past 06:00 and nothing was delivered."""
    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, 30, 9, 0, 0)
    monkeypatch.setattr(app_mod, "datetime", _Clock)


# ---------------------------------------------------------------- filler
def test_the_first_streamed_sentence_disarms_the_thinking_line(monkeypatch, tmp_path):
    """Heard live: "First sentence. Checking right now, sir. Second sentence."
    -- the filler timer kept running under a streamed reply."""
    monkeypatch.setattr(CONFIG, "talkback", True)
    a = _app(monkeypatch, tmp_path)
    a._turn_start()
    a._on_stream_sentence("First sentence, sir.")
    time.sleep(0.2)                                    # past the 0.05 s filler delay
    a._on_stream_sentence("Second sentence, sir.")
    a._turn_finished()
    assert a.said == ["First sentence, sir.", "Second sentence, sir."], a.said


def test_a_bursts_falling_edge_clears_the_filler_label(monkeypatch, tmp_path):
    a = _app(monkeypatch, tmp_path)
    a._turn_filler_pending = True
    a._tts_active = True
    a._turn_on_speaking(SpeakingState(active=False))
    assert not a._turn_filler_pending
    a._turn_on_speaking(SpeakingState(active=True))
    assert a.marks[-1][0] == "audio", "the next burst is the answer, not a filler"


# -------------------------------------------------------------- briefing
def test_the_briefing_waits_for_the_answers_own_falling_edge(monkeypatch, tmp_path):
    """An ack ("Looking that up, sir") ends in a falling edge while the
    brain is still busy; delivering there collided with the in-flight call
    and marked the day delivered for a briefing nobody heard."""
    monkeypatch.setattr(CONFIG, "talkback", True)
    a = _app(monkeypatch, tmp_path)
    _due(monkeypatch, a)
    a._after_dispatch("what is the weather", "voice",
                      SimpleNamespace(reply="Looking that up, sir.", speak=True,
                                      done=False, ack=True, status="Looking it up…"))
    assert a._briefing_pending
    a._turn_busy.set()                                  # the answer is still coming
    a._after_speech()                                   # the ack's falling edge
    assert a.chats == [] and a._briefing_pending
    a._turn_busy.clear()
    a._after_speech()                                   # the answer's falling edge
    assert [t for t, _ in a.chats] == ["my morning briefing"]
    assert not a._briefing_pending
    assert json.loads((tmp_path / "briefing.json").read_text())["delivered"] == "2026-08-30"


def test_an_uncertain_or_ignored_turn_does_not_arm_the_briefing(monkeypatch, tmp_path):
    a = _app(monkeypatch, tmp_path)
    _due(monkeypatch, a)
    for status in ("Was that for me?", "Ignored (background chat)"):
        a._after_dispatch("mumble", "voice",
                          SimpleNamespace(reply=None, speak=False, done=False, ack=False, status=status))
        assert not a._briefing_pending, status


def test_a_busy_model_leaves_the_day_unmarked(monkeypatch, tmp_path):
    a = _app(monkeypatch, tmp_path, brain=SimpleNamespace(is_busy=True))
    _due(monkeypatch, a)
    a._deliver_first_wake_briefing()
    assert a.chats == [] and not (tmp_path / "briefing.json").exists()
    a.brain = SimpleNamespace(is_busy=False)
    a._deliver_first_wake_briefing()
    assert len(a.chats) == 1 and (tmp_path / "briefing.json").exists()


def test_the_briefing_is_not_delivered_into_the_yes_no_window(monkeypatch, tmp_path):
    a = _app(monkeypatch, tmp_path)
    _due(monkeypatch, a)
    a._briefing_pending = True
    a._pending_uncertain = {"rid": object()}
    a._after_speech()                                   # "Was that for me?" finished
    assert a.chats == [] and a._briefing_pending


# ------------------------------------------------------------- follow-up
def test_the_follow_up_mic_waits_for_the_queue_to_drain(monkeypatch, tmp_path):
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(CONFIG, "followup_window", 4.0)
    monkeypatch.setattr(app_mod.MACHINE, "has_mic", True, raising=False)
    _delivered_today(tmp_path)
    tts = SimpleNamespace(is_speaking=False, pending=1)
    a = _app(monkeypatch, tmp_path, tts=tts)
    a._followup_after_speech = True
    a._after_speech()                                   # a filler's edge, answer queued
    assert a.starts == [] and a._followup_after_speech
    tts.pending = 0
    a._after_speech()
    time.sleep(0.3)
    assert a.starts == [True] and not a._followup_after_speech


# ------------------------------------------------------------- barge-in
def test_a_wake_word_over_a_streamed_reply_cuts_the_model_too(monkeypatch, tmp_path):
    """With the turn open for the whole generation, the old gate refused
    the wake word and barge-in was unreachable on any long reply."""
    monkeypatch.setattr(CONFIG, "barge_in", True)
    monkeypatch.setattr(CONFIG, "sound", False)
    cancelled, cut = [], []
    a = _app(monkeypatch, tmp_path, brain=SimpleNamespace(cancel=lambda: cancelled.append(1)))
    a.interrupt_speech = lambda: cut.append(1) or True
    a._turn_busy.set()
    a._tts_active = True
    a._on_hotword(0.9)
    assert cancelled == [1] and cut == [1]
    assert not a._turn_busy.is_set() and a._stream_muted
    assert ("wake",) in a.marks
    a._on_stream_sentence("...and a late sentence, sir.")
    assert a.said == [], "a cancelled reply's tail is never spoken"
    time.sleep(0.1)                                     # the Timer(0) hop
    assert a.starts == [False]
    a._turn_start()
    assert not a._stream_muted, "the next turn's stream is fresh"


def test_without_speech_playing_the_turn_gate_still_holds(monkeypatch, tmp_path):
    monkeypatch.setattr(CONFIG, "barge_in", True)
    a = _app(monkeypatch, tmp_path)
    a._turn_busy.set()
    a._tts_active = False
    a._on_hotword(0.9)
    assert a.marks == [] and a._turn_busy.is_set()


# ------------------------------------------------------------- confidence
def _process(a, monkeypatch, accepted, text="by Agenda 4.2.6"):
    monkeypatch.setattr(CONFIG, "speaker_verify", False)
    a.transcriber = SimpleNamespace(transcribe=lambda audio: SimpleNamespace(
        text=text, confidence=-1.2, accepted=accepted))
    a._audio_busy.set()
    a._process_audio(b"\x00" * 32000)


def test_say_that_again_is_asked_once_then_jarvis_goes_quiet(monkeypatch, tmp_path):
    a = _app(monkeypatch, tmp_path)
    _process(a, monkeypatch, accepted=False)
    assert a.said == [app_mod.SAY_AGAIN_LINE] and a._followup_after_speech
    a._followup_after_speech = False
    _process(a, monkeypatch, accepted=False)
    assert a.said == [app_mod.SAY_AGAIN_LINE], "no second ask: the room is talking"
    assert not a._followup_after_speech
    assert ("abandon", "rejected:confidence") in a.marks
    a._on_hotword.__func__  # a wake resets the counter (see _on_hotword)
    a._say_again_count = 0
    _process(a, monkeypatch, accepted=False)
    assert a.said == [app_mod.SAY_AGAIN_LINE] * 2


# --------------------------------------------------------- guests, memory
def test_his_own_voice_is_not_a_guest(monkeypatch, tmp_path):
    monkeypatch.setattr(CONFIG, "talkback", True)
    a = _app(monkeypatch, tmp_path)
    a._tts_active = True
    a._on_guest(0.95)
    assert a.said == []
    a._tts_active = False
    a._on_guest(0.95)
    assert a.said == [a._guest_line]


def test_guest_and_learning_gates_are_open_right_after_boot(monkeypatch, tmp_path):
    a = _app(monkeypatch, tmp_path)
    assert time.monotonic() - a._last_guest_ts > 180.0
    assert time.monotonic() - a._last_learn_ts > 600.0


def test_the_app_does_not_record_a_chat_exchange_twice(monkeypatch, tmp_path):
    """brain._remember records every chat; the app recorded it again."""
    from jarvis import events
    replies = []
    unsub = events.bus.subscribe(events.JarvisReply, replies.append)
    a = _app(monkeypatch, tmp_path)
    a.tts = SimpleNamespace(is_speaking=False, pending=0)
    a._on_brain_tags([("SPEAK", "It is ten, sir.")])
    try:
        assert a.exchanges == []
        assert a.said == ["It is ten, sir."]
    finally:
        try:
            events.bus.unsubscribe(events.JarvisReply, replies.append)
        except Exception:
            if callable(unsub):
                unsub()
