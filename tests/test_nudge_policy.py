"""The "did not catch that" policy (jarvis/app.py, JarvisApp._nudge).

After a wake word the user expects a reply, but several outcomes used to end
in nothing: a clip the speaker gate dropped, an empty transcript, "No audio
captured" / "Too short" (status text only), a second garbled clip in a row.
The code itself said a silent no-op "looks exactly like a dead microphone".
Policy by cause: nothing heard gets a short spoken "Sir?" and a follow-up
window; a speaker rejection or a second garbled clip gets a non-verbal
earcon; follow-up windows, the mic button and guests stay silent; and the
whole thing is rate-limited so a noisy room cannot make him chatter.

The app is the real JarvisApp with hardware and peers stubbed, as in
tests/test_turn_flow_fixes.py; the recorder is stubbed as in
tests/test_endpoint.py.
"""
import threading
import time
from types import SimpleNamespace

import numpy as np

import jarvis.app as app_mod
import jarvis.events as events_mod
from jarvis.app import NUDGE_LINE, SAY_AGAIN_LINE
from jarvis.config import CONFIG
from jarvis.events import RecordingStarted, RecordingStopped, Transcribed


class FakeTranscriber:
    loaded = True

    def __init__(self, text="what time is it", accepted=True):
        self.text, self.accepted = text, accepted

    def transcribe(self, audio):
        return SimpleNamespace(text=self.text, confidence=-0.2, accepted=self.accepted)


def _app(monkeypatch, cfg=None, config=None, **over):
    """cfg: assistant.json overrides; config: jarvis.config.CONFIG overrides."""
    settings = {"talkback": True, "speaker_verify": False, "endpoint_vad": True}
    settings.update(config or {})
    for k, v in settings.items():
        monkeypatch.setattr(CONFIG, k, v)
    options = {"listening.nudge": True, "listening.nudge_cooldown_s": 30}
    options.update(cfg or {})
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(get=lambda k, d=None: options.get(k, d), user_name="Hunter")
    a._init_assistant_state()
    a.recorder = SimpleNamespace(recording=False, endpointer=None, last_audio=None,
                                 snapshot_audio=lambda: None)
    a.transcriber = FakeTranscriber()
    a.speaker = SimpleNamespace(enrolled=False)
    a._audio_busy = threading.Event()
    a._tts_active = False
    a.said, a.dispatched, a.marks, a.beeps = [], [], [], []
    a._say = a.said.append
    a._dispatch = lambda text, source: a.dispatched.append((text, source))
    a._maybe_learn_voice = lambda audio, stats: None
    a.turns = SimpleNamespace(mark=lambda *x, **k: a.marks.append(x),
                              abandon=lambda r: a.marks.append(("abandon", r)))
    a.events = []
    monkeypatch.setattr(events_mod.bus, "publish", a.events.append)
    beeped = threading.Event()

    def play_beep(kind):
        a.beeps.append(kind)
        beeped.set()
    a.beeped = beeped
    monkeypatch.setattr(app_mod, "play_beep", play_beep)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _wake_capture(a, followup=False):
    """A capture opens the way a wake word (or a follow-up window) opens it."""
    a._wake_pending = not followup
    a._on_recording_started(RecordingStarted())


def _stopped(a, audio=None, followup=False, endpoint="vad"):
    """The recorder stopped with nothing usable (last_audio None)."""
    a.recorder.last_audio = audio
    a._on_recording_stopped(RecordingStopped(reason="silence", endpoint=endpoint,
                                             dead_air_s=0.8, followup=followup))


def _stopped_with_audio(a, followup=False):
    """The recorder stopped with a clip; the test then runs _process_audio
    itself (the real handler would start it on a worker thread)."""
    a._stop_event = RecordingStopped(reason="silence", endpoint="vad",
                                     dead_air_s=0.8, followup=followup)


def _clip():
    return np.zeros(16000 * 2, dtype=np.float32)          # 2 s at 16 kHz


# -------------------------------------------------------- nothing heard
def test_nothing_captured_after_a_wake_word_gets_the_spoken_cue(monkeypatch):
    """"No audio captured" and "Too short" both leave last_audio None."""
    a = _app(monkeypatch)
    _wake_capture(a)
    _stopped(a, audio=None)
    assert a.said == [NUDGE_LINE]
    assert a._followup_after_speech, "the question should be repeatable without the wake word"
    assert ("abandon", "no_audio") in a.marks


def test_an_empty_transcript_gets_the_spoken_cue(monkeypatch):
    a = _app(monkeypatch)
    a.transcriber = FakeTranscriber(text="   ", accepted=True)
    _wake_capture(a)
    _stopped_with_audio(a)
    a._process_audio(_clip())
    assert a.said == [NUDGE_LINE] and a.dispatched == []
    assert not a._audio_busy.is_set()


# ---------------------------------------------------------- stays silent
def test_a_follow_up_window_that_hears_nothing_stays_silent(monkeypatch):
    """Nothing said into the follow-up window is the normal case; a cue
    there would re-open the mic without end."""
    a = _app(monkeypatch)
    _wake_capture(a, followup=True)             # opened by _start_followup
    _stopped(a, audio=None, followup=True)
    assert a.said == [] and a.beeps == []
    # and the recorder's own flag on the event wins even if the app's did not
    _wake_capture(a)
    _stopped(a, audio=None, followup=True)
    assert a.said == [] and a.beeps == []


def test_the_follow_up_opener_clears_the_wake_flag(monkeypatch):
    monkeypatch.setattr(CONFIG, "followup_window", 4.0)
    monkeypatch.setattr(app_mod.MACHINE, "has_mic", True)
    a = _app(monkeypatch)
    a.recorder = SimpleNamespace(recording=False, endpointer=object(),
                                 start=lambda followup=False: None)
    a._turn_busy = threading.Event()
    a._wake_pending = True                      # a stale flag from a wake refused elsewhere
    a._start_followup()
    assert a._wake_pending is False


def test_the_mic_button_gets_no_cue_and_a_wake_flag_does_not_leak(monkeypatch):
    """The button shows its own status; and the flag a wake word set must be
    consumed by ITS capture, never by the next one."""
    a = _app(monkeypatch)
    a._on_recording_started(RecordingStarted())          # button: no wake pending
    _stopped(a, audio=None)
    assert a.said == [] and a.beeps == []
    a._wake_pending = True
    a._on_recording_started(RecordingStarted())          # the wake's capture
    assert a._turn_from_wake and not a._wake_pending
    a._on_recording_started(RecordingStarted())          # the next button press
    assert not a._turn_from_wake


def test_no_cue_while_he_is_already_talking(monkeypatch):
    a = _app(monkeypatch, _tts_active=True)
    _wake_capture(a)
    _stopped(a, audio=None)
    assert a.said == [] and a.beeps == []


def test_the_policy_can_be_switched_off(monkeypatch):
    a = _app(monkeypatch, cfg={"listening.nudge": False})
    _wake_capture(a)
    _stopped(a, audio=None)
    assert a.said == [] and a.beeps == []


# ------------------------------------------------------------- earcons
def test_a_speaker_rejection_gets_an_earcon_not_a_line(monkeypatch):
    """Not every rejection is a guest -- several were the user under the old
    0.40 threshold -- so silence is wrong; but a spoken line would answer a
    guest, so the cue is non-verbal."""
    a = _app(monkeypatch, config={"speaker_verify": True})
    a.speaker = SimpleNamespace(enrolled=True,
                                filter_segments=lambda audio: (None, {"best_score": 0.1}))
    _wake_capture(a)
    _stopped_with_audio(a)
    a._process_audio(_clip())
    assert a.beeped.wait(2.0) and a.beeps == ["nudge"]
    assert a.said == [] and not a._followup_after_speech
    (ev,) = [e for e in a.events if isinstance(e, Transcribed)]
    assert (ev.accepted, ev.reject_reason) == (False, "speaker")


def test_the_second_garbled_clip_gets_an_earcon(monkeypatch):
    """The first is asked again (SAY_AGAIN_LINE); the second in a row is the
    room reaching the mic, so no re-opened window -- but not silence either."""
    a = _app(monkeypatch)
    a.transcriber = FakeTranscriber(text="by agenda four point two", accepted=False)
    _wake_capture(a)
    _stopped_with_audio(a)
    a._process_audio(_clip())
    assert a.said == [SAY_AGAIN_LINE] and a.beeps == []
    a._followup_after_speech = False
    _wake_capture(a)
    _stopped_with_audio(a)
    a._process_audio(_clip())
    assert a.beeped.wait(2.0) and a.beeps == ["nudge"]
    assert a.said == [SAY_AGAIN_LINE] and not a._followup_after_speech
    assert ("abandon", "rejected:confidence") in a.marks


def test_the_earcon_falls_back_when_talkback_is_off(monkeypatch):
    a = _app(monkeypatch, config={"talkback": False})
    _wake_capture(a)
    _stopped(a, audio=None)
    assert a.beeped.wait(2.0) and a.beeps == ["nudge"] and a.said == []


# ---------------------------------------------------------- rate limit
def test_the_cue_is_rate_limited_across_causes(monkeypatch):
    a = _app(monkeypatch)
    _wake_capture(a)
    _stopped(a, audio=None)
    a.transcriber = FakeTranscriber(text="", accepted=True)
    _wake_capture(a)
    _stopped_with_audio(a)
    a._process_audio(_clip())                   # "empty", 0 s after the last cue
    assert a.said == [NUDGE_LINE], "chattered within the cooldown"
    a._last_nudge_ts = time.monotonic() - 31.0  # the cooldown has passed
    _wake_capture(a)
    _stopped(a, audio=None)
    assert a.said == [NUDGE_LINE, NUDGE_LINE]


def test_the_cooldown_comes_from_the_config(monkeypatch):
    a = _app(monkeypatch, cfg={"listening.nudge_cooldown_s": 0})
    for _ in range(3):
        _wake_capture(a)
        _stopped(a, audio=None)
    assert a.said == [NUDGE_LINE] * 3


def test_a_suppressed_cue_does_not_restart_the_cooldown(monkeypatch):
    a = _app(monkeypatch)
    _wake_capture(a)
    _stopped(a, audio=None)
    first = a._last_nudge_ts
    _wake_capture(a)
    _stopped(a, audio=None)                     # suppressed
    assert a._last_nudge_ts == first


# ----------------------------------------------------------- plumbing
def test_the_cue_line_is_prewarmed():
    """An uncached line took 12.6 s to render on 2026-08-27; "Sir?" must be
    in the speech cache with the other fixed persona lines."""
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(setup_lines=lambda: [], user_name="Hunter")
    a._guest_line = "I only answer to Hunter, sir."
    phrases = a._canned_phrases()
    assert NUDGE_LINE in phrases and SAY_AGAIN_LINE in phrases


def test_the_hotword_arms_the_wake_flag_after_its_refusal_checks(monkeypatch):
    """A refused wake ("still on the last one") must not arm the flag for
    whatever capture opens next."""
    a = _app(monkeypatch)
    a.recorder = SimpleNamespace(recording=False, endpointer=None, start=lambda: None)
    a._turn_busy = threading.Event()
    a._audio_busy.set()                         # still transcribing the previous clip
    a._on_hotword(0.9)
    assert a._wake_pending is False
    a._audio_busy.clear()
    monkeypatch.setattr(CONFIG, "sound", False)
    monkeypatch.setattr(CONFIG, "barge_in", False)
    a._on_hotword(0.9)
    assert a._wake_pending is True


def test_no_nudge_while_a_reply_is_still_rendering(monkeypatch):
    """The SpeakingState edge now rises at first AUDIO, so tts.busy is the
    truthful "he is talking" predicate during the render gap."""
    a = _app(monkeypatch)
    a.tts = SimpleNamespace(busy=True)
    a._turn_from_wake = True
    a._stop_event = None
    a._nudge("empty")
    assert a.said == [] and a.beeps == []
