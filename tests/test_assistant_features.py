"""The conversational features added 2026-08-30: per-turn tool subsets,
conversation memory, the confidence gate's spoken retry, the follow-up
window, guest voices, passive enrolment, the first-wake briefing, meeting
heads-ups, diagnostics and barge-in.
"""
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

import jarvis.app as app_mod
import jarvis.recorder as recorder_mod
from jarvis.config import CONFIG
from jarvis.headsup import MeetingHeadsUp
from jarvis.tools.cues import CORE, compile_cues
from jarvis.tools.registry import ToolRegistry, ToolResult, ToolSpec


# ------------------------------------------------------------ tool subsets
def _reg(names):
    reg = ToolRegistry()
    for n in names:
        reg.register(ToolSpec(n, "desc", handler=lambda **k: ToolResult(text="x")))
    reg.set_cues(compile_cues())
    return reg


def test_only_cued_tools_ride_with_the_utterance():
    reg = _reg(["get_time", "get_weather", "get_calendar", "notes", "get_mail",
                "spotify_play", "spotify_control", "get_briefing", "set_timer"])
    def names(text):
        return {s["function"]["name"] for s in reg.schemas_for(text)}
    assert names("what's the weather") == set(CORE)
    assert names("play some daft punk") == set(CORE) | {"spotify_play"}
    assert names("what was my last email about") == set(CORE) | {"get_mail"}
    assert names("set a timer for five minutes") == set(CORE) | {"set_timer"}
    assert len(reg.schemas_for("who are you")) == len(CORE)


def test_a_tool_without_cues_always_rides():
    reg = _reg(["get_time", "brand_new_tool"])
    assert {s["function"]["name"] for s in reg.schemas_for("anything")} == {"get_time", "brand_new_tool"}


def test_without_cues_installed_everything_rides():
    reg = ToolRegistry()
    reg.register(ToolSpec("get_mail", "d", handler=lambda **k: ToolResult(text="x")))
    assert len(reg.schemas_for("hello")) == 1


# ------------------------------------------------------ conversation memory
def test_the_model_now_sees_the_last_exchanges():
    from jarvis.context import ContextEngine
    ctx = ContextEngine.__new__(ContextEngine)
    ctx._conversation = []
    ctx.add_exchange("what's on my calendar tomorrow", "Nothing tomorrow, sir.")
    text = ContextEngine.format_for_prompt(ctx, {"time": "now", "conversation": ctx._recent_conversation()})
    assert "User: what's on my calendar tomorrow" in text
    assert "Jarvis: Nothing tomorrow, sir." in text


def test_stale_exchanges_drop_out():
    from jarvis.context import ContextEngine
    ctx = ContextEngine.__new__(ContextEngine)
    old = (datetime.now() - timedelta(seconds=ContextEngine.CONVERSATION_TTL_S + 5)).isoformat()
    ctx._conversation = [{"time": old, "user": "old", "jarvis": "old reply"}]
    ctx.add_exchange("fresh", "fresh reply")
    recent = ctx._recent_conversation()
    assert [e["user"] for e in recent] == ["fresh"]


# ----------------------------------------------------- follow-up capture
def _rec(endpointer):
    rec = object.__new__(recorder_mod.Recorder)
    rec.recording = True
    rec.endpointer = endpointer
    rec._ep_cursor = 0
    rec._record_rate = 16000
    rec._record_start_time = time.monotonic() - 10.0
    rec._audio_frames = [np.zeros((1600, 1), dtype=np.float32)] * 20
    rec._stop_endpoint, rec._stop_dead_air = "", None
    rec._voice_stopped = False
    rec._followup = True
    rec.aborted = []
    rec.abort = lambda: rec.aborted.append(True) or setattr(rec, "recording", False)
    rec.stops = []
    rec.stop = lambda reason="manual", **kw: rec.stops.append(reason)
    return rec


def test_a_follow_up_window_with_no_speech_closes_quietly(monkeypatch):
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_vad", True)
    monkeypatch.setattr(recorder_mod.CONFIG, "followup_window", 4.0)
    class Vad:
        silence_since_speech = None
        def feed(self, a, r): pass
    rec = _rec(Vad())
    assert rec._check_endpoint() is True and rec.aborted and rec.stops == []


def test_a_follow_up_that_is_answered_ends_like_any_capture(monkeypatch):
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_vad", True)
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_silence", 0.8)
    monkeypatch.setattr(recorder_mod.CONFIG, "silence_grace", 0.0)
    class Vad:
        silence_since_speech = 0.9
        audio_seconds = 3.0
        def feed(self, a, r): pass
    rec = _rec(Vad())
    assert rec._check_endpoint() is True and rec.stops == ["silence"] and not rec.aborted


def _app(monkeypatch, **over):
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
    a.said = []
    a._say = a.said.append
    a.context = SimpleNamespace(add_exchange=lambda u, j: a.exchanges.append((u, j)))
    a.exchanges = []
    a.services = SimpleNamespace(brain=SimpleNamespace(chat=lambda t, **kw: a.chats.append((t, kw))))
    a.chats = []
    a._briefing_state_path = lambda: over.get("state")
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_a_spoken_reply_arms_the_follow_up_and_records_the_exchange(monkeypatch, tmp_path):
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(CONFIG, "followup_window", 4.0)
    monkeypatch.setattr(app_mod.MACHINE, "has_mic", True, raising=False)
    a = _app(monkeypatch, state=tmp_path / "b.json")
    (tmp_path / "b.json").write_text(json.dumps({"delivered": datetime.now().date().isoformat()}))
    res = SimpleNamespace(reply="It is ten, sir.", speak=True, done=True, ack=False, status="")
    a._after_dispatch("what time is it", "voice", res)
    assert a.exchanges == [("what time is it", "It is ten, sir.")]
    assert a._followup_after_speech
    a._after_speech()                                    # the reply finished playing
    time.sleep(0.3)
    assert a.starts == [True] and not a._followup_after_speech


def test_fillers_and_acks_do_not_arm_a_follow_up(monkeypatch, tmp_path):
    monkeypatch.setattr(CONFIG, "talkback", True)
    a = _app(monkeypatch, state=tmp_path / "b.json")
    (tmp_path / "b.json").write_text(json.dumps({"delivered": datetime.now().date().isoformat()}))
    a._after_dispatch("look up x", "voice",
                      SimpleNamespace(reply="Looking that up, sir.", speak=True, done=False, ack=True, status=""))
    assert not a._followup_after_speech and a.exchanges == []


def test_a_wake_word_cancels_a_pending_follow_up_and_barges_in(monkeypatch, tmp_path):
    monkeypatch.setattr(CONFIG, "barge_in", True)
    monkeypatch.setattr(CONFIG, "sound", False)
    a = _app(monkeypatch, state=tmp_path / "b.json")
    a.turns = SimpleNamespace(mark=lambda *x, **k: None)
    a._tts_active = True
    a.cut = []
    a.interrupt_speech = lambda: a.cut.append(True)
    a._followup_after_speech = True
    monkeypatch.setattr(app_mod.threading, "Timer", lambda d, fn: SimpleNamespace(start=lambda: None))
    a._on_hotword(0.9)
    assert a.cut == [True] and not a._followup_after_speech


# ------------------------------------------------------------ guests
def test_a_guest_is_declined_politely_and_not_nagged(monkeypatch, tmp_path):
    monkeypatch.setattr(CONFIG, "talkback", True)
    a = _app(monkeypatch, state=tmp_path / "b.json")
    a._on_guest(0.95)
    a._on_guest(0.95)
    assert a.said == ["I only answer to Hunter, sir."]
    a._on_guest(0.5)                                     # a weak wake: silence
    assert len(a.said) == 1


# ------------------------------------------------------ passive learning
def test_passive_learning_takes_a_clear_match_once_in_a_while(monkeypatch, tmp_path):
    monkeypatch.setattr(CONFIG, "speaker_verify", True)
    monkeypatch.setattr(CONFIG, "speaker_threshold", 0.3)
    a = _app(monkeypatch, state=tmp_path / "b.json")
    learned = []
    done = threading.Event()
    a.speaker = SimpleNamespace(add_sample=lambda audio: learned.append(len(audio)) or done.set())
    audio = np.zeros(16000 * 2, dtype=np.float32)
    a._maybe_learn_voice(audio, {"scores": [0.62, -0.06]})
    assert done.wait(2) and learned == [32000]
    a._maybe_learn_voice(audio, {"scores": [0.62]})       # rate-limited
    a._maybe_learn_voice(audio, {"scores": [0.35]})       # not clear enough
    time.sleep(0.05)
    assert learned == [32000]


# ---------------------------------------------------- first-wake briefing
class _Clock:
    """datetime stand-in: now() is 07:00 on a fixed day."""
    fixed = datetime(2026, 8, 30, 7, 0)

    @classmethod
    def now(cls, tz=None):
        return cls.fixed

    fromtimestamp = staticmethod(datetime.fromtimestamp)


def test_first_wake_briefing_is_due_once_a_day_after_the_hour(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "datetime", _Clock)
    a = _app(monkeypatch, state=tmp_path / "b.json")
    assert a._briefing_due(datetime(2026, 8, 30, 5, 59)) is False
    assert a._briefing_due(datetime(2026, 8, 30, 7, 0)) is True
    a._after_dispatch("what time is it", "voice",
                      SimpleNamespace(reply="x", speak=True, done=True, ack=False, status=""))
    assert a._briefing_pending is True
    a._after_speech()
    assert a.chats == [("my morning briefing", {"force_tool": "get_briefing"})]
    assert a.said[0].startswith("Your briefing")
    assert a._briefing_due(_Clock.fixed) is False, "delivered today: not again"


def test_asking_for_the_briefing_counts_as_delivered(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "datetime", _Clock)
    a = _app(monkeypatch, state=tmp_path / "b.json")
    a._after_dispatch("give me my briefing", "voice",
                      SimpleNamespace(reply=None, speak=False, done=False, ack=False, status="Briefing…"))
    assert a._briefing_due(_Clock.fixed) is False and not a._briefing_pending


# -------------------------------------------------------- meeting heads-up
class _Cal:
    def __init__(self, events):
        self._events = events
    def configured(self):
        return True
    def events(self):
        return self._events


def test_heads_up_files_one_reminder_per_timed_event(tmp_path):
    tz = timezone.utc
    now = datetime(2026, 8, 31, 8, 30, tzinfo=tz)
    def ev(title, start, all_day=False):
        return SimpleNamespace(title=title, start=start, all_day=all_day)
    cal = _Cal([ev("BIOSENSORS", now + timedelta(minutes=40)),
                ev("Holiday", now + timedelta(minutes=50), all_day=True),
                ev("Gone", now - timedelta(minutes=5)),
                ev("Far", now + timedelta(hours=5))])
    filed = []
    tk = SimpleNamespace(add_reminder=lambda due, text, repeat="": filed.append((due, text)))
    h = MeetingHeadsUp(lambda: cal, tk, lead_min=10, state_path=tmp_path / "s.json",
                       now=lambda tzinfo=None: now)
    assert h.tick() == 1
    (due, text), = filed
    assert text == "BIOSENSORS in 10 minutes"
    assert abs(due - (now + timedelta(minutes=30)).timestamp()) < 1
    assert h.tick() == 0, "filed twice"
    h2 = MeetingHeadsUp(lambda: cal, tk, lead_min=10, state_path=tmp_path / "s.json",
                        now=lambda tzinfo=None: now)
    assert h2.tick() == 0, "a restart re-filed the same meeting"


def test_heads_up_inside_the_lead_says_the_real_minutes(tmp_path):
    tz = timezone.utc
    now = datetime(2026, 8, 31, 8, 30, tzinfo=tz)
    cal = _Cal([SimpleNamespace(title="Standup", start=now + timedelta(minutes=4), all_day=False)])
    filed = []
    tk = SimpleNamespace(add_reminder=lambda due, text, repeat="": filed.append((due, text)))
    MeetingHeadsUp(lambda: cal, tk, lead_min=10, state_path=tmp_path / "s.json",
                   now=lambda tzinfo=None: now).tick()
    assert filed[0][1] == "Standup in 4 minutes"
    assert filed[0][0] - now.timestamp() < 10


# -------------------------------------------------------------- barge-in
def test_barge_in_leaves_the_wake_word_live_while_speaking(monkeypatch):
    import jarvis.tts as tts_mod
    t = object.__new__(tts_mod.TTS)
    t._mic_hold = "would-be-a-stack"
    acquired = []
    t._arbiter = SimpleNamespace(acquire=lambda owner: acquired.append(owner))
    monkeypatch.setattr(tts_mod.CONFIG, "barge_in", True)
    t._acquire_mic()
    assert t._mic_hold is None and acquired == []
    t._release_mic()                                     # nothing to release: no raise
