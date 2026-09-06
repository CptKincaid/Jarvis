"""Every path that opens the microphone asks first. All of them, not most.

FOUND by the adversary that cleared the Users-tab enrolment lane, 2026-09-05:
the new exclusion covers the face run and the voice run, and ``runs_live``
appears nowhere in jarvis/app.py -- so the two OTHER things in that file that
open a capture were never taught to ask.

  * ``_ask_uncertain`` speaks "Was that for me?" and then calls
    ``recorder.record_fixed(UNCERTAIN_LISTEN_S)``. A turn already in flight
    when a voice run starts opens a second capture underneath it.
  * ``train_wakeword`` takes three fixed 3 s captures with no gate and no
    undo -- the same shape as the drawer button that was just removed.

WHY THE ARBITER IS NOT THE ANSWER, and it is worth writing down because it
looks like it should be: ``recorder.MicArbiter`` is a re-entrant DEPTH
COUNTER under an RLock whose only job is pausing the hotword on the first
acquire. Two consumers on two threads both get their context manager and
both proceed. ``Recorder.record_fixed`` guards only on ``self.recording``,
a flag it never sets, so two ``record_fixed`` calls do not exclude each
other either. The exclusion has to be decided BEFORE a device opens, which
is what ``voicerun.runs_live`` is for.

THE HARM IS BOUNDED AND IT IS STILL WORTH CLOSING. A take Jarvis talked
through is dropped at the far edge, so his voiceprint stays his voice; what
is lost is a take, a prompt and his patience. The census census of this
class is the point: one door left unasked is how the whole exclusion
becomes decorative.
"""
import pytest

from jarvis import app as app_mod


class _Rec:
    """A recorder that records nothing and remembers being asked."""

    def __init__(self):
        self.calls = 0
        self.recording = False

    def record_fixed(self, seconds):
        self.calls += 1
        return None


class _TTS:
    def __init__(self):
        self.said = []

    def speak(self, text, block=False):
        self.said.append(text)


class _Services:
    """Both enrolment slots, parked or not."""

    def __init__(self, face=None, voice=None):
        self.enrol_run = face
        self.voice_run = voice


def _app(*, face=None, voice=None):
    a = object.__new__(app_mod.JarvisApp)
    a.services = _Services(face=face, voice=voice)
    a.recorder = _Rec()
    a.tts = _TTS()
    return a


def _bind(a, name):
    return getattr(app_mod.JarvisApp, name).__get__(a)


# ------------------------------------------------------- the seam itself
def test_runs_live_sees_a_parked_run_of_either_kind():
    from jarvis import voicerun
    assert voicerun.runs_live(_Services()) == ()
    assert voicerun.runs_live(_Services(face=object())) == ("face",)
    assert voicerun.runs_live(_Services(voice=object())) == ("voice",)
    both = voicerun.runs_live(_Services(face=object(), voice=object()))
    assert set(both) == {"face", "voice"}


# ------------------------------------------------- train_wakeword asks first
@pytest.mark.parametrize("kind", ["face", "voice"])
def test_training_the_wake_word_does_not_open_the_mic_under_a_run(kind):
    a = _app(**{kind: object()})
    _bind(a, "train_wakeword")()
    assert a.recorder.calls == 0, (
        "train_wakeword opened the microphone while a %s enrolment was live"
        % kind)


def test_training_the_wake_word_still_works_with_nothing_running(monkeypatch):
    """The guard must be a guard, not an off switch."""
    a = _app()
    a.recorder.record_fixed = lambda s: __import__("numpy").zeros(16000,
                                                                  dtype="float32")
    seen = {}
    monkeypatch.setattr("jarvis.hotword.train_verifier",
                        lambda samples: seen.setdefault("n", len(samples)))
    _bind(a, "train_wakeword")()
    assert seen.get("n") == 3


# ------------------------------------------------ _ask_uncertain asks first
@pytest.mark.parametrize("kind", ["face", "voice"])
def test_the_uncertain_prompt_does_not_open_the_mic_under_a_run(kind):
    a = _app(**{kind: object()})
    a.UNCERTAIN_LISTEN_S = 4
    _bind(a, "_ask_uncertain")("rid-1")
    assert a.recorder.calls == 0, (
        "_ask_uncertain opened the microphone while a %s enrolment was live"
        % kind)
    assert a.tts.said == [], (
        "it also spoke over the run -- the refusal must come first")


# ------------------------------------------------------------- the census
def test_every_record_fixed_caller_in_app_asks_runs_live_first():
    """AST census, not a grep: any method of JarvisApp that calls
    record_fixed must also mention runs_live. A new door added tomorrow
    fails here rather than silently becoming the next one."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(app_mod))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = ast.dump(node)
        if "'record_fixed'" in body and "'runs_live'" not in body:
            offenders.append(node.name)
    assert offenders == [], (
        "these open the microphone without asking runs_live first: %s"
        % offenders)
