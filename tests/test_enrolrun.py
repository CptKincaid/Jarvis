"""In-app face enrolment (jarvis/enrolrun.py) and the rung that starts it.

WHAT THESE TESTS ARE ALLOWED TO DO. No camera, no display, no microphone, no
Tk, no weights. Every frame is noise this process generated, every detector
and recogniser is a stub from tests/test_faceenrol.py, and no test looks at,
saves or asserts on the content of anything. The gallery is the throwaway one
conftest forces through ``JARVIS_FACE_GALLERY``.

WHAT IS BEING PINNED, and each is a promise the design cannot keep by
intention alone:

* **A partial run cannot destroy a good gallery.** Abort at any phase, a
  camera taken away mid-run, a verdict of "not usable" -- none of them may
  reach ``gallery.save()``. This is the invariant the whole feature is
  gated on, so it is asserted at every phase separately rather than once.
* **The window has no route to a forced save.** ``--force`` exists for the
  terminal; there must be no code path from a spoken sentence to a
  non-``ok`` write, and that is checked structurally as well as behaviourally.
* **The label is HIS, from his config, never from what was said.** So a
  misrouted turn cannot write somebody else's face under his name.
* **No config is written.** The CLI flips ``camera.identity`` after typed
  consent; the in-app path refuses instead, because an aborted run must not
  leave "faces may be written down" switched on behind it.
* **The camera is given back on every path, including the exception path.**
* **The commit is TYPED.** ``owner.mode`` is 'shadow' on his live box, so
  the owner gate refuses nothing -- the typed word is the bar that actually
  stands between a stranger's sentence and a biometric write.
"""
from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from jarvis import enrolrun as er
from jarvis import faceenrol as fe
from jarvis import visionrig as vr
from jarvis.facegallery import FaceGallery
from jarvis.facemodels import LIFECAM_CINEMA
from tests.test_faceenrol import (PLAN_YAWS, ScriptedDetector,
                                  ScriptedRecogniser, base_vec, same_face)


# ------------------------------------------------------------------ fakes
class FakeSource:
    """``read() -> (ok, frame)`` -- cv2's contract, which is what the real
    FrameTap presents. Substituted for the tap so the run is deterministic
    and single-threaded; the tap's own plumbing is tests/test_enroltap.py.

    ``on_read`` is how "he said stop halfway" and "the camera went away" are
    injected at an exact frame number.
    """

    def __init__(self, stop_after: int = 0, on_read=None, seed: int = 3,
                 width: int = 1280, height: int = 720):
        self.stop_after = int(stop_after)
        self.on_read = on_read
        self.reads = 0
        self.released = 0
        self.reason = ""
        # CAPTURE-SIZED, and it has to be. The quality gate crops the frame
        # at the detected box to measure sharpness, so a frame smaller than
        # the lens would put every box outside it and reject every sample
        # for a reason that has nothing to do with what is being tested.
        self.width, self.height = int(width), int(height)
        self._rng = np.random.default_rng(seed)

    def read(self):
        if self.on_read is not None:
            self.on_read(self.reads)
        if self.stop_after and self.reads >= self.stop_after:
            return False, None
        self.reads += 1
        return True, self._rng.integers(0, 256,
                                        (self.height, self.width, 3),
                                        dtype=np.uint8)

    def release(self):
        self.released += 1

    def abort(self, reason):
        self.reason = str(reason)

    def deny(self, reason):
        self.reason = str(reason)


class FakeWorker:
    """``campreview.PreviewWorker``'s two-method surface, counting calls."""

    def __init__(self, set_tap_raises: bool = False):
        self.taps = []
        self.stops = 0
        self.set_tap_raises = bool(set_tap_raises)

    def set_tap(self, tap):
        self.taps.append(tap)
        if self.set_tap_raises and tap is not None:
            raise RuntimeError("the pipeline is gone")

    def stop(self, join=True):
        self.stops += 1


class FakeConfig:
    """``AssistantConfig``'s read surface, and a ``set`` that RECORDS rather
    than writes -- which is how "no config is written" is asserted."""

    def __init__(self, **values):
        self.values = {"camera.min_conf": 0.6, "camera.identity": True,
                       "camera.identity_min": 0.363, "user.name": "hunter"}
        self.values.update(values)
        self.sets = []

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.sets.append((key, value))
        return True


class Recorder:
    """The four output channels, captured in ORDER so a test can assert that
    the numbers-only check ran before anything was emitted."""

    def __init__(self):
        self.said = []
        self.cards = []
        self.tones = []
        self.clips = []
        self.lease = []
        self.order = []

    def say(self, text):
        self.said.append(text)
        self.order.append(("say", text))

    def card(self, text):
        self.cards.append(text)
        self.order.append(("card", text))

    def tone(self, name):
        self.tones.append(name)

    def clip(self, text):
        self.clips.append(text)
        self.order.append(("clip", text))
        return True

    def set_lease(self, on):
        self.lease.append(bool(on))


def build_run(tmp_path, monkeypatch, *, yaws=None, vectors=None,
              source=None, worker=None, cfg=None, gallery=None,
              min_conf=0.6):
    """An EnrolRun wired to stubs, with the real faceenrol underneath it.

    ``fe.build_models`` and the three ``camera`` config readers are the only
    things patched: everything the run actually decides -- the plan, the
    quality bars, the judge, the save guard -- is the real code.
    """
    rec = Recorder()
    gallery = gallery if gallery is not None else \
        FaceGallery(root=tmp_path / "gallery")
    det = ScriptedDetector(yaws or PLAN_YAWS)
    vecs = vectors if vectors is not None else same_face(base_vec(), 13)
    recog = ScriptedRecogniser(vecs, min_conf=min_conf)
    monkeypatch.setattr(fe, "build_models", lambda _cfg: (det, recog, ""))
    from jarvis import camera as cam
    monkeypatch.setattr(cam, "lens_from_config", lambda _cfg: LIFECAM_CINEMA)
    monkeypatch.setattr(cam, "head_from_config", lambda _cfg: None)
    monkeypatch.setattr(cam, "face_backend_from_config", lambda _cfg: "sface")
    services = SimpleNamespace(enrol_run=None, enrol_offer=None)
    run = er.EnrolRun(
        cfg=cfg if cfg is not None else FakeConfig(),
        worker=worker if worker is not None else FakeWorker(),
        say=rec.say, card=rec.card, lease=rec.set_lease, earcon=rec.tone,
        clipboard=rec.clip, gallery=gallery, services=services,
        # No settle and no near-duplicate gap: both are real sleeps, and
        # thirteen of each would make this file minutes long for pacing that
        # tests/test_faceenrol.py already pins.
        settle_s=0.0, gap_s=0.0, station_frames=40)
    services.enrol_run = run
    run.tap = source if source is not None else FakeSource()
    return run, rec, gallery, services


# ================================================================= the run
class TestAGoodRun:
    def test_a_clean_run_saves_one_generation_and_says_so(self, tmp_path,
                                                          monkeypatch):
        run, rec, gallery, services = build_run(tmp_path, monkeypatch)
        run._run()
        assert gallery.generations(), "a usable run must write a generation"
        assert any(t.startswith("Usable, and saved as generation")
                   for t in rec.said), rec.said
        # ...and the camera went back and the run unparked itself.
        assert rec.lease == [True, False]
        assert run.worker.taps[-1] is None
        assert services.enrol_run is None
        assert run.tap.released == 1

    def test_the_full_report_goes_to_the_clipboard_and_the_card_is_short(
            self, tmp_path, monkeypatch):
        """Two channels with two jobs. The card is the eight lines he reads
        at a glance; the clipboard is the forty-one he pastes to somebody."""
        run, rec, _g, _s = build_run(tmp_path, monkeypatch)
        run._run()
        assert len(rec.cards) == 1
        assert len(rec.cards[0].splitlines()) == 8
        assert len(rec.clips) == 1
        assert len(rec.clips[0].splitlines()) > 20

    def test_the_earcons_are_the_progress_channel(self, tmp_path,
                                                  monkeypatch):
        """He follows this with his head turned away, so a kept sample must
        tick and a finished station must chime."""
        run, rec, _g, _s = build_run(tmp_path, monkeypatch)
        run._run()
        assert rec.tones.count(er.TONE_CAPTURE) == 5      # five stations
        assert rec.tones.count(er.TONE_KEPT) == run.samples_kept
        assert run.samples_kept >= fe.MIN_SAMPLES
        assert er.TONE_DONE in rec.tones

    def test_the_preview_is_restarted_so_it_reloads_the_new_generation(
            self, tmp_path, monkeypatch):
        """A generation that landed thirty seconds ago must not be invisible
        to the running console until the next restart."""
        run, _rec, _g, _s = build_run(tmp_path, monkeypatch)
        run._run()
        assert run.worker.stops == 1

    def test_the_card_is_numbers_only_and_the_check_runs_before_it(
            self, tmp_path, monkeypatch):
        """assert_numbers_only is the mechanical half of the privacy rule,
        and it is worthless if it runs after the emit. Patched to record."""
        calls = []
        real = vr.assert_numbers_only

        def spy(obj, path="report"):
            # Only the OUTERMOST call is the run's; the real function
            # recurses into itself for every key.
            if path == "report":
                calls.append(sorted(obj) if isinstance(obj, dict) else obj)
            return real(obj, path)

        monkeypatch.setattr(vr, "assert_numbers_only", spy)
        run, rec, _g, _s = build_run(tmp_path, monkeypatch)
        run._run()
        assert calls, "the numbers-only assertion never ran"
        assert "saved_generation" in calls[0]
        # It ran before the card and the clipboard were touched.
        assert rec.order[0][0] == "say"
        assert [k for k, _v in rec.order].index("card") > 0


# ======================================================= nothing is saved
class TestNothingIsSavedUnlessItIsUsable:
    def test_an_abort_before_the_first_frame_writes_nothing(self, tmp_path,
                                                            monkeypatch):
        run, rec, gallery, _s = build_run(tmp_path, monkeypatch)
        run.abort()
        run._run()
        assert gallery.generations() == []
        assert er.E27 in rec.said
        # THE LENS IS NEVER CLAIMED AT ALL. Loading ArcFace is not instant,
        # so an abort that landed while the weights were loading must not
        # light the camera for the moment it would take the loop to notice --
        # "I said stop and it came on anyway" is what would make him stop
        # trusting the abort word.
        assert True not in rec.lease
        assert rec.lease == [False], rec.lease   # released anyway, harmlessly

    def test_an_abort_halfway_through_writes_nothing(self, tmp_path,
                                                     monkeypatch):
        """The realistic one: he says "Jarvis, stop" at station three."""
        run = None

        def stop_at_six(reads):
            if reads == 6:
                run.abort()

        run, rec, gallery, _s = build_run(
            tmp_path, monkeypatch, source=FakeSource(on_read=stop_at_six))
        run._run()
        assert gallery.generations() == []
        assert er.E27 in rec.said
        assert er.TONE_STOP in rec.tones

    def test_the_camera_being_taken_away_mid_run_writes_nothing(
            self, tmp_path, monkeypatch):
        """The curfew edge, or "offline mode" said mid-capture. The tap's
        deny turns into a False read, which run_enrolment treats as fatal."""
        source = FakeSource(stop_after=4)
        source.reason = "sensing said no"
        run, rec, gallery, _s = build_run(tmp_path, monkeypatch,
                                          source=source)
        run._run()
        assert gallery.generations() == []
        assert er.E33 in rec.said
        assert not any("Usable" in t for t in rec.said)

    def test_a_pool_the_judge_refuses_is_not_written(self, tmp_path,
                                                     monkeypatch):
        """TOO TIGHT: thirteen takes of one head position. The gallery would
        know him at the lens and refuse him the moment he turned to work, so
        it may not be saved -- and the OLD one must survive."""
        gallery = FaceGallery(root=tmp_path / "gallery")
        for vec in same_face(base_vec(5), 9, seed=4):
            gallery.add("hunter", vec)
        before = gallery.save(reason="the good one he already had")
        run, rec, _g, _s = build_run(
            tmp_path, monkeypatch, gallery=gallery,
            yaws=[0.0] * 13, vectors=same_face(base_vec(), 13, k=0.02))
        run._run()
        assert gallery.generations() == [before], \
            "a refused run must not add a generation"
        assert any(t.startswith("Not saved") for t in rec.said), rec.said

    def test_a_run_under_the_sample_floor_is_refused_by_count(
            self, tmp_path, monkeypatch):
        """Eight is the floor. A run that kept fewer says so, and says the
        remedy (more light, sit closer) rather than a verdict word."""
        run, rec, gallery, _s = build_run(
            tmp_path, monkeypatch, source=FakeSource(stop_after=200),
            yaws=[0.0, 90.0, 90.0, 90.0, 90.0, 90.0])
        run._run()
        assert gallery.generations() == []
        assert any(t.startswith("Not saved") for t in rec.said), rec.said

    def test_the_in_app_path_never_asks_for_force(self, tmp_path,
                                                  monkeypatch):
        """BEHAVIOURALLY: save_enrolment is called with force at its default
        False. The CLI's --force is the only caller that may pass True."""
        seen = {}
        real = fe.save_enrolment

        def spy(gallery, report, **kw):
            seen.update(kw)
            return real(gallery, report, **kw)

        monkeypatch.setattr(fe, "save_enrolment", spy)
        run, _rec, _g, _s = build_run(tmp_path, monkeypatch)
        run._run()
        assert seen, "save_enrolment was never reached"
        assert seen.get("force", False) is False
        assert "allow_shrink" not in seen or seen["allow_shrink"] is False

    def test_there_is_no_force_in_the_module_at_all(self):
        """STRUCTURALLY, which is the half that survives the next edit: the
        window has no route to a non-ok save because the word does not
        appear in the code."""
        from pathlib import Path
        import io
        import re
        import tokenize
        path = Path(er.__file__)
        out = []
        with open(path, "rb") as fh:
            for tok in tokenize.tokenize(io.BytesIO(fh.read()).readline):
                if tok.type in (tokenize.COMMENT, tokenize.STRING):
                    continue
                out.append(tok.string)
        code = " ".join(out)
        assert not re.search(r"\bforce\b", code)
        assert not re.search(r"\ballow_shrink\b", code)


# ================================================ the label and the config
class TestTheLabelAndTheConfig:
    def test_the_label_is_his_config_name_and_not_anything_spoken(
            self, tmp_path, monkeypatch):
        """A misrouted "enrol Heather's face" that somehow reached this run
        must still write HIS label -- so it cannot put somebody else's face
        under his name, or his name over somebody else's."""
        cfg = FakeConfig(**{"user.name": "hunter"})
        run, _rec, gallery, _s = build_run(tmp_path, monkeypatch, cfg=cfg)
        run._run()
        from jarvis import identity as identity_mod
        assert run.label == identity_mod.owner_label(cfg)
        assert list(gallery.labels()) == [run.label]

    def test_no_config_key_is_ever_written(self, tmp_path, monkeypatch):
        """camera.identity is CHECKED and never set. The CLI flips it after
        typed consent; an aborted in-app run would otherwise leave "faces may
        be written down" switched on with no terminal output to notice it."""
        cfg = FakeConfig()
        run, _rec, _g, _s = build_run(tmp_path, monkeypatch, cfg=cfg)
        run._run()
        assert cfg.sets == []

    def test_identity_switched_off_refuses_before_anything_opens(self):
        cfg = FakeConfig(**{"camera.identity": False})
        out = er.preflight(cfg, worker=FakeWorker())
        assert out["ok"] is False
        assert out["reply"] == er.E7


# ================================================== the camera always goes back
class TestTheCameraAlwaysGoesBack:
    def test_the_lease_is_released_on_the_exception_path(self, tmp_path,
                                                         monkeypatch):
        """A traceback on a daemon thread that left the lens open is the one
        failure this module may not have."""
        run, rec, gallery, services = build_run(
            tmp_path, monkeypatch, worker=FakeWorker(set_tap_raises=True))
        run._run()
        assert rec.lease == [True, False], rec.lease
        assert er.E35 in rec.said
        assert gallery.generations() == []
        assert services.enrol_run is None

    def test_the_tap_is_released_even_when_the_run_blows_up(self, tmp_path,
                                                            monkeypatch):
        source = FakeSource()
        run, _rec, _g, _s = build_run(
            tmp_path, monkeypatch, source=source,
            worker=FakeWorker(set_tap_raises=True))
        run._run()
        assert source.released == 1

    def test_the_run_unparks_itself_from_services_on_every_path(
            self, tmp_path, monkeypatch):
        run, _rec, _g, services = build_run(tmp_path, monkeypatch)
        run.abort()
        run._run()
        assert services.enrol_run is None

    def test_start_runs_on_its_own_thread_and_finishes(self, tmp_path,
                                                       monkeypatch):
        """The Tk pump may never be the thread that waits ninety seconds for
        a face, so the run owns a daemon thread of its own."""
        run, _rec, gallery, _s = build_run(tmp_path, monkeypatch)
        assert run.start() is True
        assert run.start() is False           # idempotent while it runs
        assert run.finished.wait(20.0), "the run never finished"
        assert run.thread.name == "face-enrol"
        assert run.thread.daemon is True
        assert gallery.generations()


# ===================================================== nothing on the disk
class TestNothingButTheEmbeddingReachesTheDisk:
    def test_the_full_report_is_never_logged(self, tmp_path, monkeypatch,
                                             caplog):
        """The enrolment banner promises the embedding is the only thing that
        reaches the disk. Forty-one lines of numbers in jarvis.log would pass
        the numbers-only check and still break that promise, so the log gets
        ONE line of counts."""
        caplog.set_level(logging.DEBUG)
        run, rec, _g, _s = build_run(tmp_path, monkeypatch)
        run._run()
        full = rec.clips[0]
        for record in caplog.records:
            message = record.getMessage()
            assert full not in message
            # No single log line carries the report's body either.
            assert "cohesion   min" not in message
        assert sum("in-app enrolment:" in r.getMessage()
                   for r in caplog.records) == 1

    def test_the_run_holds_no_frame_after_it_finishes(self, tmp_path,
                                                      monkeypatch):
        run, _rec, _g, _s = build_run(tmp_path, monkeypatch)
        run._run()
        for value in vars(run).values():
            assert not isinstance(value, np.ndarray)


# ========================================================= the preflight
class TestThePreflight:
    def test_offline_and_the_curfew_get_different_sentences(self):
        """They have different remedies -- "come back online" versus "wait
        until seven" -- so telling him the wrong one costs him the fix."""
        offline = SimpleNamespace(status=lambda: {
            "camera": False, "offline": True, "reason": "offline mode"})
        curfew = SimpleNamespace(status=lambda: {
            "camera": False, "offline": False, "reason": "curfew 21:00-07:00"})
        assert er.preflight(FakeConfig(), sensing=offline,
                            worker=FakeWorker())["reply"] == er.E5
        assert er.preflight(FakeConfig(), sensing=curfew,
                            worker=FakeWorker())["reply"] == er.E6

    def test_no_camera_console_means_the_terminal(self):
        out = er.preflight(FakeConfig(), worker=None)
        assert out["ok"] is False
        assert out["reply"] == er.E9

    def test_a_second_run_is_refused(self):
        services = SimpleNamespace(enrol_run=object())
        out = er.preflight(FakeConfig(), worker=FakeWorker(),
                           services=services)
        assert out["reply"] == er.E10

    def test_a_nonsense_confidence_bar_stops_before_anything_opens(self):
        out = er.preflight(FakeConfig(**{"camera.min_conf": 0.0}),
                           worker=FakeWorker())
        assert out["ok"] is False
        assert out["reply"] == er.E12


# ==================================================== abort, pause and skip
class TestTheControls:
    def test_a_pause_holds_before_the_next_station_not_mid_capture(
            self, tmp_path, monkeypatch):
        run, rec, gallery, _s = build_run(tmp_path, monkeypatch)
        run.pause()
        threading.Timer(0.15, run.skip).start()
        run._run()
        assert er.E29 in rec.said            # "Holding. Say ready when you are."
        assert er.E30 in rec.said            # "Right."
        assert gallery.generations()         # and it went on to finish

    def test_a_pause_that_is_never_released_stops_the_run(self, tmp_path,
                                                          monkeypatch):
        """A hold is not a licence to leave the lens open."""
        run, rec, gallery, _s = build_run(tmp_path, monkeypatch)
        run.pause_ttl_s = 0.05
        run.pause()
        run._run()
        assert er.E31 in rec.said
        assert gallery.generations() == []
        assert rec.lease == [True, False]

    def test_the_run_ceiling_stops_a_run_that_will_not_end(self, tmp_path,
                                                           monkeypatch):
        run, _rec, gallery, _s = build_run(tmp_path, monkeypatch)
        run.run_max_s = -1.0                 # already past it
        run._run()
        assert run._should_stop() == er.STOP_TIME
        assert gallery.generations() == []


# ============================================== the commander: offer + commit
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
                          preview_worker=None, preview_lease=None)
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.brain = SimpleNamespace(think=MagicMock(), chat=MagicMock())
    return Commander(svc)


class TestTheTypedCommit:
    """THE SAFETY ARGUMENT OF THE WHOLE FEATURE, and the reason it is here
    rather than left to the owner gate: ``owner.mode`` is 'shadow' on his
    live config, and shadow downgrades every refusal to admit. So the gate
    refuses nothing, and a spoken sentence would be the only thing between a
    stranger and a biometric write that replaces his gallery under his own
    label -- after which the gate's face leg would name that stranger as him.
    """

    def _offer(self, cmdr, made=None):
        cmdr.services.enrol_offer = {
            "made_at": time.time() if made is None else made}

    def test_the_word_said_out_loud_does_not_start_it(self, cmdr):
        self._offer(cmdr)
        res = cmdr._try_enrol("enrol", "voice")
        assert res is not None and res.reply == er.E3
        assert cmdr.services.enrol_run is None
        # ...and the offer is still parked, because he has just been told
        # what to do and the ninety seconds are still running.
        assert cmdr.services.enrol_offer is not None

    def test_the_word_typed_reaches_the_start(self, cmdr, monkeypatch):
        started = []
        monkeypatch.setattr(
            cmdr, "_start_enrol",
            lambda: (started.append(1),
                     __import__("jarvis.commander", fromlist=["x"])
                     .CommandResult(handled=True, reply=er.E2))[1])
        self._offer(cmdr)
        res = cmdr._try_enrol("enrol", "typed")
        assert started == [1]
        assert res.reply == er.E2
        assert cmdr.services.enrol_offer is None

    def test_both_spellings_are_accepted(self, cmdr, monkeypatch):
        for word in ("enrol", "enroll", "Enrol."):
            monkeypatch.setattr(cmdr, "_start_enrol", lambda: None)
            self._offer(cmdr)
            # _start_enrol returning None is not the point; reaching it is.
            cmdr._try_enrol(word, "typed")
            assert cmdr.services.enrol_offer is None, word

    def test_a_stale_offer_does_not_open_a_camera(self, cmdr):
        self._offer(cmdr, made=time.time() - er.OFFER_TTL_S - 10)
        res = cmdr._try_enrol("enrol", "typed")
        assert res.reply == er.E4
        assert cmdr.services.enrol_run is None

    def test_an_unrelated_sentence_leaves_the_offer_alone(self, cmdr):
        self._offer(cmdr)
        assert cmdr._try_enrol("what time is it", "typed") is None
        assert cmdr.services.enrol_offer is not None

    def test_with_no_offer_the_rung_is_invisible(self, cmdr):
        assert cmdr._try_enrol("enrol", "typed") is None


class TestTheMidRunControls:
    def _live(self, cmdr):
        run = SimpleNamespace(running=True, aborted=[], skipped=[],
                              paused=[])
        run.abort = lambda reason="": run.aborted.append(reason)
        run.skip = lambda: run.skipped.append(1)
        run.pause = lambda: run.paused.append(1)
        cmdr.services.enrol_run = run
        return run

    @pytest.mark.parametrize("phrase", [
        "stop", "jarvis, stop", "cancel", "abort", "never mind",
        "stop the enrolment", "forget it"])
    def test_stop_reaches_the_run_from_any_source(self, cmdr, phrase):
        """A stop must take the widest door there is: it shuts a lens, and
        stopping is the safe direction."""
        for source in ("voice", "typed", "socket"):
            run = self._live(cmdr)
            res = cmdr._try_enrol(phrase, source)
            assert res is not None and res.speak is False
            assert run.aborted, (phrase, source)

    @pytest.mark.parametrize("phrase", ["ready", "next", "go", "i'm ready"])
    def test_ready_skips_the_settle(self, cmdr, phrase):
        run = self._live(cmdr)
        assert cmdr._try_enrol(phrase, "voice") is not None
        assert run.skipped == [1], phrase

    @pytest.mark.parametrize("phrase", ["wait", "hold on", "not yet",
                                        "hang on"])
    def test_wait_holds_it(self, cmdr, phrase):
        run = self._live(cmdr)
        assert cmdr._try_enrol(phrase, "voice") is not None
        assert run.paused == [1], phrase

    def test_a_live_run_does_not_swallow_the_rest_of_the_turn(self, cmdr):
        """He can still ask the time with his head turned away."""
        self._live(cmdr)
        assert cmdr._try_enrol("what time is it", "voice") is None

    def test_a_finished_run_stops_answering(self, cmdr):
        run = self._live(cmdr)
        run.running = False
        assert cmdr._try_enrol("stop", "voice") is None
        assert run.aborted == []
