"""ONE MICROPHONE, ONE CONSUMER -- and the pool may never hold Jarvis's voice.

DEFECT 1, as the adversary put it: the face run and the voice run do not
exclude each other, so both can be live at once and a voice pool of JARVIS'S
OWN SPEECH gets saved.

WHY THE ARBITER CANNOT BE THE ANSWER, written down because it is the thing
that looks like a fix and is not. ``recorder.MicArbiter`` is a re-entrant
DEPTH COUNTER guarded by an RLock: ``acquire`` increments, pauses the hotword
on the first one and resumes it on the last. It serialises nothing. Two
consumers on two threads both get their context manager and both proceed, and
a deeper or longer acquire only makes the hotword deafer. So the exclusion has
to be decided in the PREFLIGHT, before either device is opened, and it has to
be decided the same way through every door.

AND THE HARM, which is a different claim from the exclusion. ``jarvis/tts.py``
holds ``acquire("tts")`` for a whole spoken burst and there is no AEC on this
box, so anything recording while Jarvis talks records Jarvis. The face run's
entire progress channel is SPEECH -- five stations, each announced -- so a
voice run overlapping a face run is not a tidiness problem, it is eight takes
of Jarvis's own voice written to the file his identity rests on.

NOTHING IN THIS FILE OPENS A DEVICE. Every recorder, every camera worker and
every TTS is injected, and every claim is a number or a returned sentence.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from jarvis import app as app_mod
from jarvis import consent as cs
from jarvis import enrolrun as er
from jarvis import gate as gt
from jarvis import voicerun as vr
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER, Person, Registry



# ------------------------------------------------------------- the stand-ins
class LiveRun:
    """What a started run looks like from outside. Captures nothing."""

    def __init__(self, alive=True):
        self.alive = bool(alive)
        self.aborted = []

    def abort(self, reason=""):
        self.aborted.append(reason)
        self.alive = False

    @property
    def running(self):
        return self.alive


class Arbiter:
    """The real MicArbiter's SHAPE -- re-entrant, counting, excluding
    nothing -- so a test cannot accidentally lean on it for the guarantee."""

    def __init__(self):
        self.depth = 0
        self.peak = 0
        self.owners = []

    def acquire(self, owner):
        arb = self

        class _Ctx:
            def __enter__(self):
                arb.depth += 1
                arb.peak = max(arb.peak, arb.depth)
                arb.owners.append(owner)
                return arb

            def __exit__(self, *exc):
                arb.depth -= 1
                return False

        return _Ctx()


class Recorder:
    """A microphone that yields numbers. ``talk_during`` makes Jarvis start
    speaking in the middle of a capture -- the exact race the run must lose
    the take over rather than keep it."""

    def __init__(self, tts=None, talk_during=()):
        self.mic_available = True
        self.recording = False
        self.arbiter = Arbiter()
        self.calls = 0
        self._tts = tts
        self._talk_during = set(talk_during)

    def record_fixed(self, seconds):
        self.calls += 1
        if self._tts is not None and self.calls in self._talk_during:
            self._tts.speaking = True
        return np.full(int(16000 * max(seconds, 0.1)), 0.2, dtype=np.float32)


class RealShapeTts:
    """``jarvis.tts.TTS`` EXPOSES PROPERTIES, NOT METHODS -- ``busy`` is
    "speaking now or lines still queued" and ``is_speaking`` is the bare
    flag. A seam that calls ``tts.is_speaking()`` against this raises
    TypeError, which is the shipped bug this file pins."""

    def __init__(self, speaking=False, queued=0):
        self.speaking = bool(speaking)
        self.queued = int(queued)

    @property
    def is_speaking(self):
        return self.speaking

    @property
    def busy(self):
        return bool(self.speaking or self.queued)


class MethodTts:
    """The older seam shape some callers still hand in."""

    def __init__(self, speaking=False):
        self.speaking = bool(speaking)

    def is_speaking(self):
        return self.speaking


class Services:
    def __init__(self, worker=None):
        self.preview_worker = worker if worker is not None else object()
        self.preview_lease = self._lease
        self.enrol_run = None
        self.voice_run = None
        self.enrol_offer = None
        self.leased = []

    def _lease(self, on):
        self.leased.append(bool(on))


class Stub:
    """The app's enrolment seams, with nothing else of the app attached."""

    def __init__(self, gate, services, cfg=None, sensing=None,
                 recorder=None, tts=None, speaker=None):
        self.gate = gate
        self.services = services
        self.assistant = cfg or {}
        self.sensing = sensing
        self.recorder = recorder
        self.tts = tts
        self.speaker = speaker

    _people_registry = app_mod.JarvisApp._people_registry
    _people_unlock_left = app_mod.JarvisApp._people_unlock_left
    _people_open_unlock = app_mod.JarvisApp._people_open_unlock
    _people_gate = app_mod.JarvisApp._people_gate
    _people_decide = app_mod.JarvisApp._people_decide
    _stop_run = app_mod.JarvisApp._stop_run
    _enrol_say = app_mod.JarvisApp._enrol_say
    face_enrol_start = app_mod.JarvisApp.face_enrol_start
    voice_enrol_start = app_mod.JarvisApp.voice_enrol_start


def _app(tmp_path, *, services=None, recorder=None, tts=None):
    path = tmp_path / "people.json"
    reg = Registry(path=path)
    reg.add_person(Person(label="alderman", name="Alderman", role=ROLE_OWNER))
    reg.add_person(Person(label="heather", name="Heather", role=ROLE_KNOWN,
                          consent="typed"))
    assert reg.save()
    gate = gt.OwnerGate(registry=Registry.load(path), owner="alderman",
                        get_option=lambda k, d=None: d)
    return Stub(gate, services if services is not None else Services(),
                recorder=recorder, tts=tts)


@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    from jarvis.commander import Commander, IntentClassifier
    from jarvis.config import CONFIG
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG",
                       str(tmp_path / "assistant.json"))
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG",
                        tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    svc = SimpleNamespace(desktop=MagicMock(), workflows=MagicMock(),
                          memory=MagicMock(), context=MagicMock(),
                          tts=MagicMock(), enrol_offer=None, enrol_run=None,
                          voice_run=None, preview_worker=object(),
                          preview_lease=None, assistant={})
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.brain = SimpleNamespace(think=MagicMock(), chat=MagicMock())
    return Commander(svc)


# =============================================== A: the two preflights refuse
def test_the_voice_preflight_refuses_while_a_face_run_is_live():
    """The face run's whole progress channel is SPEECH. Recording through it
    stores Jarvis reading the stations out."""
    svc = Services()
    svc.enrol_run = LiveRun()
    out = vr.preflight(recorder=Recorder(), tts=RealShapeTts(),
                       services=svc, label="alderman", owner="alderman")
    assert out["ok"] is False
    assert "face" in out["reply"].lower()


def test_the_face_preflight_refuses_while_a_voice_run_is_live():
    """The mirror. A face run started under a live voice run puts five
    spoken station prompts straight into the takes being kept."""
    svc = Services()
    svc.voice_run = LiveRun()
    out = er.preflight({"camera.identity": True}, sensing=None,
                       worker=svc.preview_worker, services=svc)
    assert out["ok"] is False
    assert "voice" in out["reply"].lower()


def test_neither_refusal_is_the_other_run_s_own_slot_sentence():
    """Two different walls need two different sentences: "stop the face run"
    and "one is already running" send him to different buttons."""
    face_busy = Services()
    face_busy.enrol_run = LiveRun()
    voice_busy = Services()
    voice_busy.voice_run = LiveRun()
    a = vr.preflight(recorder=Recorder(), tts=RealShapeTts(),
                     services=face_busy)["reply"]
    b = vr.preflight(recorder=Recorder(), tts=RealShapeTts(),
                     services=voice_busy)["reply"]
    assert a and b and a != b


# ================================= B: every door, not just the one that leaks
class TestEveryEntryPoint:
    """A GUARANTEE IS ONLY AS GOOD AS ITS WEAKEST DOOR. There are four ways
    into these two runs -- the tab's two buttons, the spoken offer, and the
    typed word that commits it -- and each is walked here against a live run
    of the other kind."""

    def test_the_tab_button_will_not_start_a_face_run_over_a_voice_run(
            self, tmp_path):
        svc = Services()
        svc.voice_run = LiveRun()
        app = _app(tmp_path, services=svc)
        ok, line = app.face_enrol_start()
        assert ok is False
        assert "voice" in line.lower()
        assert svc.enrol_run is None
        assert svc.leased == [], "a refused press asked for the lens"

    def test_the_tab_button_will_not_start_a_voice_run_over_a_face_run(
            self, tmp_path):
        svc = Services()
        svc.enrol_run = LiveRun()
        rec = Recorder()
        app = _app(tmp_path, services=svc, recorder=rec,
                   tts=RealShapeTts())
        ok, line = app.voice_enrol_start()
        assert ok is False
        assert "face" in line.lower()
        assert svc.voice_run is None
        assert rec.calls == 0, "a refused press opened the microphone"

    def test_the_shared_launch_will_not_start_a_face_run_over_a_voice_run(
            self, tmp_path):
        """``enrolrun.launch`` is the sequence BOTH face doors run, so it is
        pinned on its own -- a door added later inherits the refusal."""
        svc = Services()
        svc.voice_run = LiveRun()
        built = []
        ok, reply, run = er.launch(
            cfg={"camera.identity": True}, services=svc, sensing=None,
            say=lambda _t: None,
            build_fn=lambda **kw: built.append(kw) or LiveRun())
        assert ok is False
        assert run is None
        assert built == [], "a run was built before the refusal"
        assert svc.enrol_run is None
        assert "voice" in reply.lower()

    def test_the_spoken_offer_is_not_made_while_a_voice_run_is_live(
            self, cmdr):
        """The OFFER is the third door: it parks "type enrol to start", and
        ninety seconds later that word opens a lens. It must not be parked
        against a microphone that is already recording."""
        from jarvis import commander as cm
        cmdr.services.voice_run = LiveRun()
        cmdr.services.assistant = {"camera.identity": True}
        res = cm._enrol_offer(cmdr)
        assert cmdr.services.enrol_offer is None, \
            "an offer to open the lens was parked over a live voice run"
        assert res is None or "voice" in str(res.reply).lower()

    def test_the_typed_word_will_not_commit_over_a_voice_run(self, cmdr):
        """The fourth door. An offer parked BEFORE the voice run started is
        still on the namespace, and the word is typed after it."""
        cmdr.services.enrol_offer = {"made_at": time.time()}
        cmdr.services.assistant = {"camera.identity": True}
        cmdr.services.voice_run = LiveRun()
        res = cmdr._try_enrol("enrol", "typed")
        assert res is not None
        assert cmdr.services.enrol_run is None, \
            "a stale offer opened the lens over a live voice run"
        assert "voice" in str(res.reply).lower()

    def test_the_drawer_button_is_gone_and_records_nothing(self, tmp_path):
        """The fifth door is the one being REMOVED, and it stays removed:
        ``enroll_speaker`` refuses rather than appending fifteen unjudged
        seconds to voiceprint.npz."""
        import inspect
        src = inspect.getsource(app_mod.JarvisApp.enroll_speaker)
        # The DOC still names what it used to do, deliberately. The BODY is
        # what must no longer reach a device or the file.
        body = src.split('"""')[-1]
        assert "record_fixed" not in body
        assert "enroll_from_audio" not in body
        assert "ENROLL_SPEAKER_GONE" in body


# ================================== C: the seam that reads "am I talking now"
class TestTheTalkingSeam:
    """THE PREFLIGHT'S MOST IMPORTANT CHECK MUST WORK AGAINST THE REAL TTS.

    ``jarvis.tts.TTS.is_speaking`` is a PROPERTY returning a bool, and
    ``TTS.busy`` is the wider "speaking now or still queued". A seam that
    writes ``tts.is_speaking()`` gets ``TypeError: 'bool' object is not
    callable`` against the shipped object -- which the preflight catches and
    turns into "I can't tell whether I'm talking", so the button he was told
    to use instead of the dangerous one refuses every single time.
    """

    def test_a_real_shaped_tts_that_is_talking_is_seen_as_talking(self):
        out = vr.preflight(recorder=Recorder(),
                           tts=RealShapeTts(speaking=True), services=None)
        assert out["ok"] is False
        assert out["reason"] == "tts is speaking", out
        assert "talking" in out["reply"].lower()

    def test_a_real_shaped_tts_that_is_quiet_lets_the_run_through(self):
        out = vr.preflight(recorder=Recorder(), tts=RealShapeTts(),
                           services=None, label="alderman", owner="alderman")
        assert out["ok"] is True, out

    def test_lines_still_queued_count_as_talking(self):
        """``busy`` is the honest predicate: a queued burst will be spoken
        into the very first take. ``is_speaking`` alone says idle during the
        render window."""
        out = vr.preflight(recorder=Recorder(),
                           tts=RealShapeTts(speaking=False, queued=2),
                           services=None)
        assert out["ok"] is False
        assert out["reason"] == "tts is speaking", out

    def test_the_older_method_shaped_seam_still_works(self):
        assert vr.preflight(recorder=Recorder(), tts=MethodTts(speaking=True),
                            services=None)["reason"] == "tts is speaking"
        assert vr.preflight(recorder=Recorder(), tts=MethodTts(),
                            services=None, label="a", owner="a")["ok"] is True

    def test_a_seam_that_cannot_answer_still_fails_closed(self):
        class Broken:
            @property
            def busy(self):
                raise RuntimeError("no engine")

        out = vr.preflight(recorder=Recorder(), tts=Broken(), services=None)
        assert out["ok"] is False
        assert out["reason"] in ("tts unreadable", "no tts seam")


# ================================= D: the harm itself, not just the exclusion
class TestNoPoolIsSavedWhileJarvisTalks:
    """THE ACCEPTANCE BAR IS THE HARM, NOT THE ORDERING. Excluding the two
    runs closes the door the adversary found; this closes the one behind it,
    because Jarvis can also start talking from anywhere else -- a timer, a
    reminder, an arriving message -- in the middle of a take."""

    def _run(self, tmp_path, rec, tts, **kw):
        from jarvis import voicegallery as vg
        from tests.synthvoice import Voices
        gallery = vg.VoiceGallery(root=tmp_path / "voice_gallery")
        world = Voices(seed=11)
        said = []
        return vr.VoiceRun(
            label="alderman", gallery=gallery, recorder=rec,
            embed=lambda _a: world.take("alderman"),
            tts=tts, say=said.append, card=lambda _t: None,
            earcon=lambda _n: None, takes=3, seconds=1.0, settle_s=0.0,
            consent_how=cs.HOW_OWNER, now=lambda: 0.0,
            sleep=lambda _s: None, **kw), gallery, said

    def test_a_take_that_jarvis_talked_through_is_thrown_away(self, tmp_path):
        """Quiet at the tone, talking by the time the capture ends: the take
        holds Jarvis's voice and may not be kept."""
        tts = RealShapeTts()
        rec = Recorder(tts=tts, talk_during=(1,))
        run, _gallery, _said = self._run(tmp_path, rec, tts)
        take = run._take_once("read this")
        assert take is None, "a take Jarvis spoke through was kept"

    def test_no_generation_is_written_when_jarvis_talks_through_every_take(
            self, tmp_path):
        tts = RealShapeTts()
        rec = Recorder(tts=tts, talk_during=set(range(1, 40)))
        run, gallery, said = self._run(tmp_path, rec, tts, max_tries=6)
        run._run()
        assert gallery.generations() == [], \
            "a pool recorded over Jarvis's own voice reached the disk"
        assert run.kept == 0

    def test_a_quiet_run_still_saves(self, tmp_path):
        """The guard must not be a blanket refusal: silence still enrols."""
        tts = RealShapeTts()
        rec = Recorder(tts=tts)
        run, gallery, _said = self._run(tmp_path, rec, tts)
        run._run()
        assert run.kept == 3
        assert gallery.generations(), "a quiet run wrote nothing"
