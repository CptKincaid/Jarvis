"""Enrolling his face WITHOUT stopping Jarvis: the run that lives inside the
running app.

WHAT THIS REPLACES. Until now, enrolling a face while Jarvis was up was
impossible in both directions. With the app running it holds /dev/video0 and
``scripts/face_enrol.py`` could not open the camera at all; with the app
stopped, ``SensingPolicy`` refused on the grounds that sensing was offline.
The only path through was: come back online, kill Jarvis, enrol, restart. He
was walked through that on 2026-09-03 and asked for it to go away.

WHAT IT IS NOT. It is not a second enrolment. ``faceenrol.run_enrolment``
still walks the stations, ``faceenrol.judge_gallery`` still decides, and
``faceenrol.save_enrolment`` still refuses to write a pool that failed --
this module supplies a frame source, a spoken progress channel and a set of
controls, and nothing else. There is no second copy of the station loop, no
second set of quality bars and no second way to save.

THE PROGRESS CHANNEL IS AUDIO, AND THAT IS THE WHOLE DESIGN. Two of the five
stations are "turn your head the other way" and "sit back, further than
usual". At those two he can see no screen and reach no keyboard, so anything
that advances on a key press or a click solves the three easy stations and
fails the two the five-station plan exists for. So Jarvis SAYS the station,
pauses for him to get there, plays a rising tone meaning hold still, captures,
plays a three-note chime meaning he may move, and says the count. He can
follow the entire run with his eyes shut.

ADVANCE IS ON A BUDGET, NOT ON A POSE GATE. A station is given a bounded
number of frames and then the run moves on whether or not it got what it
wanted. Requiring the pose before advancing would make a station he cannot
physically reach on his camera mount cost him minutes of standing still
instead of costing him one note -- and it is unnecessary, because
``run_enrolment`` already says when a yaw is outside the station's window and
``judge_gallery`` judges the distribution that actually came out.

IT IS OWNER-ONLY, AND THE COMMIT IS TYPED. Two separate reasons, both
measured rather than assumed:

* THIRD PARTIES KEEP THE TERMINAL. ``scripts/face_enrol.consent`` requires
  stdin AND stdout to be real TTYs and makes the person type their own name
  before their biometrics are stored. A window can reproduce "read it and
  type it" but not "and nobody may type it for them" -- the dialog is driven
  by whoever is already logged in. So the label here is forced to
  ``identity.owner_label`` and is NEVER read from spoken words; "enrol
  Heather" keeps today's hand-over (jarvis/enrolentry.py) unchanged.
* THE OWNER GATE CANNOT CARRY THIS ALONE. ``owner.mode`` is 'shadow' on his
  live config, and in shadow every refusal is downgraded to admit -- so the
  gate refuses nothing today, and even in enforce it fails open several
  documented ways and its own header says a photograph defeats the face leg.
  A spoken sentence would then be the only thing between a stranger and a
  biometric write that REPLACES his gallery under his own label, after which
  the gate's face leg would name that stranger as him. ``gate.GATED_SOURCES``
  is ("voice",), i.e. ``typed`` is exempt BY CONSTRUCTION on the argument
  that typing means somebody is physically at the keyboard -- so voice makes
  the offer and one typed word commits it. The four letters happen before the
  camera opens and long before station four puts the keyboard out of reach.

THE MICROPHONE IS NEVER ACQUIRED. ``MicArbiter`` is a re-entrant depth
counter that pauses the hotword on the first acquire, so holding it for a
ninety-second run would be ninety seconds of deafness -- and his abort word
would be the thing that could not be heard. Abort, pause and skip ride the
always-live hotword through a commander rung instead.

NOTHING BUT THE EMBEDDING REACHES THE DISK. The eight-line card goes out as a
display-only reply (which does not enter the plaintext journal), the full
report goes to the clipboard, and the log gets one line of counts. That is
the enrolment banner's promise kept in the window as well as at the terminal.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from jarvis import faceenrol as fe
from jarvis import visionrig as vr
from jarvis.enroltap import FrameTap
from jarvis.logs import get_logger

log = get_logger("enrolrun")

# ------------------------------------------------------------- the pacing
# How long he is given to GET into a position before the tone says hold
# still. Station one is "look at the lens", which he is already doing; the
# expensive one is station three, "turn your head the other way", and 3 s is
# a comfortable head turn without being a wait. "Ready" cuts it short.
SETTLE_S = 3.0
# The frame budget per station. At the preview's 7.5-15 fps this is roughly
# eight to sixteen seconds of trying, which is generous for the three samples
# a station wants and bounded enough that five stations cannot become a
# five-minute stand. A read that TIMES OUT ends the run rather than eating
# the budget, so the worst case here is frames, not frames x timeout.
STATION_FRAMES = 120
# Minimum spacing between two accepted samples -- the near-duplicate guard,
# and the same default run_enrolment documents.
GAP_S = 0.35
# How long a "hold on" may hold the camera open before the run gives up. He
# asked for a pause, not for an open lens he has forgotten about.
PAUSE_TTL_S = 120.0
# The whole run's ceiling, checked by should_stop. A five-station run is
# about ninety seconds; this is the backstop for a station that somehow
# never exhausts its budget.
RUN_MAX_S = 240.0
# The offer's life. Long enough to read the sentence, walk back and type
# four letters; short enough that a stale "enrol" typed an hour later is a
# new subject rather than a camera coming on.
OFFER_TTL_S = 90.0
# What he types to commit. One word, in the command bar he already has.
CONFIRM_WORD = "enrol"
# ...and the spelling he will half the time actually type.
CONFIRM_WORDS = ("enrol", "enroll")

# ---------------------------------------------------------- the sentences
# Authored fresh rather than reused from ``Station.prompt``. Those carry
# "--" and words in CAPS for a terminal reader and land badly through TTS;
# they stay exactly as they are for the CLI and the report.

# The offer and the commit.
E1 = ("I can do it here, sir. Five positions, about a minute, no terminal. "
      "Type enrol in the box to start -- nothing is captured until you do.")
E2 = ("Right. The camera is coming on. Hold still when you hear the tone, "
      "and move when you hear the chime. Say Jarvis, stop at any point and "
      "nothing is written.")
E3 = ("That has to be typed, sir, not said. Type enrol in the box and I'll "
      "start.")
E4 = ("That offer has gone stale, sir. Ask me to enrol your face again.")

# Preflight refusals. No camera is opened on any of these paths.
E5 = ("I can't, sir. You told me to go offline, so the camera stays shut. "
      "Say come back online and ask me again.")
E6 = ("I can't, sir. The camera curfew is on, and nothing opens the lens "
      "before seven.")
E7 = ("Face identity is switched off, sir. Turn on camera identity in "
      "settings and ask me again. I won't flip that one for you from a "
      "spoken command.")
E8 = ("The face models aren't on disk, sir, so there's nothing to enrol "
      "into. The names are on the card.")
E9 = ("I can't run it in the window on this box, sir. The command is on "
      "your clipboard.")
E10 = "An enrolment is already running, sir. Say stop to end it."
E11 = ("{name} has to type their own name at a terminal before I store "
       "their face, sir. That isn't one I'll take by voice. The command is "
       "on your clipboard.")
E12 = ("The camera confidence bar in your config won't make sense, sir, so "
       "I've stopped before opening anything. It's on the card.")

# The stations, in plan order.
STATION_LINES = {
    "lens": "One of five. Look straight into the lens.",
    "screen": ("Two of five. Now look at your screen, exactly as you do "
               "when you're working."),
    "across": ("Three of five. Turn your head the other way, as if you "
               "were looking at the far side of the desk."),
    "back": ("Four of five. Back to the camera, but sit back -- further "
             "away than usual."),
    "close": ("Five of five. Lean in closer than usual, still looking at "
              "the camera."),
}

# Between stations -- spoken as one breath with the next station line.
E20 = "Nothing usable there."
E21 = "That's all five. Judging it now."

# The verdicts.
E22 = ("Usable, and saved as generation {n}. {takes} takes. The full report "
       "is on your clipboard.")
E23 = ("Not saved, sir. The takes are too alike -- that gallery would know "
       "you in one position and refuse you in every other. Your old one is "
       "untouched, and the report is on your clipboard.")
E24 = ("Not saved, sir. One of the takes doesn't match the others, so "
       "something in that pool isn't you. Your old one is untouched, and "
       "the report is on your clipboard.")
E25 = ("Not saved, sir. Eight takes are the floor and I only kept {n}. Your "
       "old one is untouched. Try again with more light, or sit a little "
       "closer.")
E26 = ("Not saved, sir. The detector didn't find a usable face in any "
       "position. Your old one is untouched, and the numbers are on your "
       "clipboard.")

# Abort, pause and mid-run failure.
E27 = "Stopped. Nothing was written and your gallery is untouched."
E29 = "Holding. Say ready when you are."
E30 = "Right."
E31 = ("You've been holding two minutes, sir, so I've stopped and turned "
       "the camera off. Nothing was written.")
E33 = ("You've gone offline, so the camera has shut. Nothing was saved, and "
       "your gallery is untouched.")
E34 = ("The camera stopped answering, sir. Nothing was saved, and your "
       "gallery is untouched.")
E35 = ("Something went wrong in the middle of that, sir. The camera is off, "
       "nothing was written, and your gallery is untouched.")

# Why an abort happened, in the words should_stop hands to run_enrolment.
STOP_SPOKEN = "he said stop"
STOP_PAUSE = "the hold ran out"
STOP_TIME = "the run ran past its ceiling"

# ------------------------------------------------------------ the earcons
# The progress channel he can follow with his head turned away. Every play
# passes cooldown_s=0.0: the default four-second same-tone cooldown exists to
# stop a false-wake tone repeating, and here it would silently swallow the
# second and third "kept one" ticks of a three-sample station -- which are
# the ticks he is counting.
TONE_CAPTURE = "heard-you"     # capturing now, hold still
TONE_KEPT = "thinking"         # one sample kept -- he counts these
TONE_DONE = "done"             # position finished, you may move
TONE_EMPTY = "held-back"       # this position gave nothing, moving on
TONE_STOP = "warning"          # the run has stopped


def _count_line(got: int, wanted: int) -> str:
    """What the station just produced, in the fewest words that are true."""
    if got <= 0:
        return E20
    if got >= wanted:
        return "%d." % got
    return "%d of %d." % (got, wanted)


class EnrolRun:
    """One in-app enrolment. Built by the commander, parked on
    ``services.enrol_run``, and thrown away when it finishes.

    Every collaborator is injected, which is what makes this testable without
    a camera, a display, a microphone or Tk: ``say`` speaks, ``card``
    publishes display-only text, ``earcon`` plays a tone, ``clipboard`` takes
    the long report, ``lease`` holds the preview open, and ``worker`` is the
    live ``campreview.PreviewWorker`` that actually holds the lens.
    """

    def __init__(self, *, cfg, worker, say: Callable[[str], None],
                 card: Callable[[str], None],
                 lease: Optional[Callable[[bool], None]] = None,
                 earcon: Optional[Callable[[str], None]] = None,
                 clipboard: Optional[Callable[[str], bool]] = None,
                 sensing=None, gallery=None, services=None,
                 settle_s: float = SETTLE_S, gap_s: float = GAP_S,
                 station_frames: int = STATION_FRAMES,
                 pause_ttl_s: float = PAUSE_TTL_S,
                 run_max_s: float = RUN_MAX_S,
                 now: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.worker = worker
        self.sensing = sensing
        self.services = services
        self._say = say
        self._card = card
        self._lease = lease
        self._earcon = earcon
        self._clipboard = clipboard
        self.settle_s = float(settle_s)
        # The near-duplicate guard, passed through rather than read
        # from the constant at the call site so the suite can drive a
        # whole thirteen-sample run without thirteen real sleeps.
        self.gap_s = float(gap_s)
        self.station_frames = int(station_frames)
        self.pause_ttl_s = float(pause_ttl_s)
        self.run_max_s = float(run_max_s)
        self._now = now
        self.gallery = gallery
        self.label = ""
        self.tap = FrameTap()
        self.thread: Optional[threading.Thread] = None
        self.started_at = 0.0
        # The controls. _advance cuts a settle short; _paused holds before
        # the NEXT station and never mid-capture; _stop ends the run.
        self._advance = threading.Event()
        self._paused = threading.Event()
        self._stop_reason = ""
        self._lock = threading.Lock()
        # What the last station produced, said in one breath with the next
        # station's line so the audio is a cadence rather than a stream.
        self._pending_count = ""
        # The station currently being captured, and what it has produced.
        self._cur = None
        self._cur_got = 0
        self._cur_wanted = 0
        self.finished = threading.Event()
        # Numbers a caller or a test may read without touching a frame.
        self.stations_done = 0
        self.samples_kept = 0

    # ------------------------------------------------------------- speaking
    def _speak(self, text: str) -> None:
        try:
            self._say(text)
        except Exception:            # noqa: BLE001 - TTS is not the run
            log.debug("enrolrun: the voice failed", exc_info=True)

    def _tone(self, name: str) -> None:
        """Play an earcon, never letting the audio path end the run."""
        if self._earcon is None:
            return
        try:
            self._earcon(name)
        except Exception:            # noqa: BLE001 - sound is a courtesy
            log.debug("enrolrun: the earcon failed", exc_info=True)

    # ------------------------------------------------------------ controls
    @property
    def running(self) -> bool:
        thread = self.thread
        return thread is not None and thread.is_alive()

    def abort(self, reason: str = STOP_SPOKEN) -> None:
        """He said stop. Ends the run at the next frame, at the latest.

        Both waits this can be blocked in -- the tap's read and the settle
        pause -- are woken, so an abort costs at most one frame interval and
        not a whole station.
        """
        with self._lock:
            if not self._stop_reason:
                self._stop_reason = str(reason or STOP_SPOKEN)
        self._advance.set()
        self._paused.clear()
        try:
            self.tap.abort(self._stop_reason)
        except Exception:            # noqa: BLE001
            log.debug("enrolrun: the tap refused an abort", exc_info=True)

    def skip(self) -> None:
        """"Ready" / "next" / "go" -- stop settling and capture now."""
        self._paused.clear()
        self._advance.set()

    def pause(self) -> None:
        """"Wait" / "hold on" -- hold BEFORE the next station.

        Never mid-capture: a pause that landed between two samples of one
        station would leave him frozen in a position with the camera on and
        no tone to tell him which. The hold is taken at the station boundary,
        where a pause means something he can act on.
        """
        self._paused.set()
        self._advance.clear()

    def _stopping(self) -> str:
        with self._lock:
            return self._stop_reason

    # ----------------------------------------------------------- the seams
    def _should_stop(self) -> str:
        """Asked once per frame by ``run_enrolment``."""
        reason = self._stopping()
        if reason:
            return reason
        if self.started_at and \
                (self._now() - self.started_at) > self.run_max_s:
            self.abort(STOP_TIME)
            return STOP_TIME
        return ""

    def _on_station(self, station, index: int, total: int) -> None:
        """Announce a position, hold for him to reach it, then say hold still.

        The previous station's count is spoken HERE rather than when that
        station ended, so the two arrive as one utterance -- "Three. Two of
        five, now look at your screen" -- instead of as two announcements
        with a gap he would read as the run having stalled.
        """
        self._finish_station()
        if self._hold_if_paused():
            return
        if self._stopping():
            return
        self._cur = station
        self._cur_got = 0
        self._cur_wanted = int(getattr(station, "samples", 0))
        line = STATION_LINES.get(station.key) or str(station.prompt)
        count, self._pending_count = self._pending_count, ""
        self._speak(("%s %s" % (count, line)) if count else line)
        # THE SETTLE. Interruptible from both directions: "ready" cuts it
        # short, "stop" ends it. A plain sleep here would make the abort word
        # take up to three seconds to land at every station boundary.
        self._advance.clear()
        self._advance.wait(self.settle_s)
        if self._stopping():
            return
        # The rising tone: capturing now, hold still. It is the LAST thing
        # before the frames start, so what he hears and what the loop does
        # are the same instant.
        self._tone(TONE_CAPTURE)

    def _on_sample(self, sample, station, got: int, wanted: int) -> None:
        """One judged frame. A tick per KEPT sample and nothing per reject.

        A tone per rejected frame would be a machine-gun at 10 fps in bad
        light, and it would say "something is wrong" at precisely the moment
        the honest message is "keep still, I am still trying".
        """
        if not getattr(sample, "accepted", False):
            return
        self._cur_got = int(got) + 1
        self.samples_kept += 1
        self._tone(TONE_KEPT)

    def _finish_station(self) -> None:
        """Close the station that was running: the chime, and the count held
        back to be spoken in one breath with the next station's line.

        ``run_enrolment`` has no end-of-station callback and does not need
        one -- a station ends exactly when the next one is announced, or when
        the walk returns -- so this is called from both of those places and
        is idempotent through ``self._cur``.
        """
        if self._cur is None:
            return
        got, wanted, self._cur = self._cur_got, self._cur_wanted, None
        self.stations_done += 1
        # The three-note rise means "you may move". It is the tone that lets
        # him stop holding a position he cannot see a screen from.
        self._tone(TONE_DONE if got else TONE_EMPTY)
        self._pending_count = _count_line(got, wanted)

    def _hold_if_paused(self) -> bool:
        """Block while he is holding. True if the run must now end."""
        if not self._paused.is_set():
            return False
        self._speak(E29)
        deadline = self._now() + self.pause_ttl_s
        while self._paused.is_set() and not self._stopping():
            if self._now() >= deadline:
                # A HOLD IS NOT A LICENCE TO LEAVE THE LENS OPEN. Two
                # minutes and the run ends itself, camera off, nothing
                # written.
                self._speak(E31)
                self.abort(STOP_PAUSE)
                return True
            self._advance.wait(0.2)
        if self._stopping():
            return True
        self._speak(E30)
        return False

    # --------------------------------------------------------------- the run
    def start(self) -> bool:
        """Spawn the capture on its own thread. True if it began.

        A DAEMON THREAD, and not the Tk thread and not a worker pool: the run
        is ninety seconds of speaking, sleeping and waiting on frames, and
        anything of that shape on the Tk thread freezes the console -- the
        window would stop repainting, the command bar would stop taking the
        very word that aborts it, and he would read a frozen console as a
        crash.
        """
        if self.running:
            return False
        self.started_at = self._now()
        self.thread = threading.Thread(target=self._run, name="face-enrol",
                                       daemon=True)
        self.thread.start()
        return True

    def _run(self) -> None:
        """The whole run, and every exit from it goes through the finally."""
        saved = False
        # Normally start() has already stamped this. Set here as well so that
        # the run ceiling holds for a caller that drives _run directly -- a
        # ceiling that only exists when you came in through the front door is
        # not a ceiling.
        if not self.started_at:
            self.started_at = self._now()
        try:
            saved = self._walk()
        except Exception:                # noqa: BLE001 - nothing may escape
            # A traceback here would leave the lens open and the tap live,
            # which is the one failure this module may not have. The finally
            # below is what actually closes them; this exists so the failure
            # is spoken rather than dying silently on a daemon thread.
            log.exception("enrolrun: the run failed")
            self._tone(TONE_STOP)
            self._speak(E35)
        finally:
            # ORDER MATTERS, AND EVERY STEP IS UNCONDITIONAL.
            try:
                self.tap.release()
            except Exception:            # noqa: BLE001
                log.debug("enrolrun: the tap refused release", exc_info=True)
            try:
                if self.worker is not None:
                    self.worker.set_tap(None)
            except Exception:            # noqa: BLE001
                log.debug("enrolrun: the worker kept the tap", exc_info=True)
            if saved:
                # THE PREVIEW IS STILL HOLDING THE OLD GALLERY IN MEMORY. Its
                # pipeline resolved an identity gallery when it was built, so
                # without this the console would go on failing to recognise
                # him until the next restart -- against a generation that
                # landed thirty seconds ago. Stopping is enough: releasing
                # the lease re-applies, which builds a fresh pipeline.
                try:
                    self.worker.stop(join=False)
                except Exception:        # noqa: BLE001 - provider edge
                    log.debug("enrolrun: the worker would not stop",
                              exc_info=True)
            if self._lease is not None:
                try:
                    self._lease(False)
                except Exception:        # noqa: BLE001 - a dead window
                    log.debug("enrolrun: the lease would not release",
                              exc_info=True)
            if self.services is not None:
                try:
                    if getattr(self.services, "enrol_run", None) is self:
                        self.services.enrol_run = None
                except Exception:        # noqa: BLE001 - a slim services
                    log.debug("enrolrun: could not unpark the run",
                              exc_info=True)
            self.finished.set()

    def _walk(self) -> bool:
        """Preflight, capture, judge, save. True if a generation was written.

        THE GALLERY IS THIS RUN'S OWN OBJECT and never the preview's. The
        live pipeline is handed a gallery by ``campreview.resolve_identity``,
        and ``forget(label)`` below is an IN-MEMORY delete -- so sharing the
        object would mean an aborted run silently un-recognising him in the
        running console until a restart. A cancelled action must not leave a
        regression behind it.
        """
        cfg = self.cfg
        from jarvis import identity as identity_mod   # noqa: PLC0415 - lazy
        # THE LABEL IS HIS, FROM HIS CONFIG, AND NEVER FROM SPOKEN WORDS.
        # This is what makes a misrouted turn unable to write somebody else's
        # face under his name, or his name over somebody else's face.
        self.label = identity_mod.owner_label(cfg)
        if not self.label:
            self._speak(E35)
            return False
        detector, recogniser, why = fe.build_models(cfg)
        if detector is None or recogniser is None:
            log.warning("enrolrun: no models (%s)", why)
            self._speak(E8)
            return False
        from jarvis import camera as cam              # noqa: PLC0415 - lazy
        from jarvis.facegallery import default_gallery  # noqa: PLC0415
        gallery = self.gallery
        if gallery is None:
            gallery = default_gallery(
                backend=cam.face_backend_from_config(cfg))
        limits = fe.SampleLimits(
            min_conf=float(cfg.get("camera.min_conf", 0.6)))
        lens = cam.lens_from_config(cfg)

        # NOTHING IS ON DISK YET AND NOTHING WILL BE UNTIL save_enrolment.
        # load() then forget() is the CLI's ordering and it matters for the
        # same reason: save() writes the whole in-memory pool, so a run that
        # started from an empty object would drop every OTHER enrolled person
        # the moment it landed.
        gallery.load()
        gallery.forget(self.label)
        plan, plan_why = fe.choose_plan([], mode="full")
        if not plan:
            log.warning("enrolrun: no plan (%s)", plan_why)
            self._speak(E35)
            return False
        session = fe.EnrolmentSession(gallery, self.label, lens, detector,
                                      recogniser, limits,
                                      head=cam.head_from_config(cfg))

        # THE LENS IS CLAIMED HERE AND NOT ONE LINE EARLIER. Everything above
        # is config, weights and arithmetic; a refusal costs him a sentence
        # and no camera at all.
        #
        # ...and an abort that landed while the weights were loading must not
        # light it for the instant it would take the loop to notice. Loading
        # ArcFace is not instant, and "I said stop and the camera came on
        # anyway" is exactly the thing that would make him stop trusting the
        # abort word.
        if self._stopping():
            self._ended_early(session.report(fe.SFACE_COSINE_SAME))
            return False
        if self._lease is not None:
            self._lease(True)
        self.worker.set_tap(self.tap)
        rep, _run = fe.run_enrolment(
            session, self.tap, plan=plan,
            say=_swallow, wait=None,
            frames_per_station=self.station_frames, gap_s=self.gap_s,
            identity_min=float(cfg.get("camera.identity_min",
                                       fe.SFACE_COSINE_SAME)),
            on_station=self._on_station, on_sample=self._on_sample,
            should_stop=self._should_stop)
        self._finish_station()

        if self._stopping() or not rep.detector_ok:
            self._ended_early(rep)
            return False
        # The LAST station's count has nowhere else to go -- there is no
        # sixth station line to ride out on -- so it is spoken here, in one
        # breath with "that's all five". He is counting these.
        count, self._pending_count = self._pending_count, ""
        self._speak(("%s %s" % (count, E21)) if count else E21)
        out = fe.save_enrolment(
            gallery, rep,
            reason="in-app enrolment %s (typed at the console)" % self.label)
        self._report(rep, out)
        return bool(out.get("saved_generation"))

    def _ended_early(self, rep) -> None:
        """A run that stopped before judging. NOTHING was written -- and say
        which of the three ways it was, because they have three different
        remedies."""
        self._tone(TONE_STOP)
        reason = self._stopping() or str(getattr(rep, "reason", "") or "")
        low = reason.lower()
        if reason in (STOP_SPOKEN, ""):
            self._speak(E27)
        elif reason == STOP_PAUSE:
            pass            # E31 was already said the moment it expired
        elif "sensing" in low or "offline" in low or "handed back" in low:
            self._speak(E33)
        else:
            self._speak(E34)
        log.info("in-app enrolment stopped: %s; kept %d over %d station(s), "
                 "nothing saved", reason or "no reason", self.samples_kept,
                 self.stations_done)

    def _report(self, rep, out: dict) -> None:
        """The verdict, the card and the clipboard. NUMBERS ONLY, and the
        full report never touches the disk.

        Three channels, each chosen for what it does NOT do:

        * the SPOKEN verdict is one sentence, because he is still sitting
          there and a read-out of forty lines is not an answer;
        * the CARD goes out display-only, which is the one reply path that
          does not call ``context.add_exchange`` and therefore does not land
          in the plaintext journal;
        * the FULL report goes to the clipboard, so the paste-it-to-somebody
          workflow the CLI was built around still works.

        It is deliberately NOT written to jarvis.log. Forty-one lines of
        numbers would pass the numbers-only assertion, but the enrolment
        banner promises "the only thing that reaches the disk is the
        embedding" and the clipboard already covers the convenience. One INFO
        line of counts, which is what facegallery already logs.
        """
        gen = int(out.get("saved_generation") or 0)
        if gen:
            self._speak(E22.format(n=gen, takes=int(rep.pool_total)))
        else:
            self._speak(self._refusal(rep))
        # THE ASSERTION RUNS BEFORE ANYTHING IS EMITTED, exactly as the CLI
        # does it. Honest about what it proves: it is a STRUCTURAL check over
        # the payload -- it guarantees no array, no crop and no object
        # escaped into the report; it does not vet the wording of a string.
        payload = rep.to_dict()
        payload["saved_generation"] = gen
        payload["gallery_total"] = int(out.get("gallery_total") or 0)
        vr.assert_numbers_only(payload)
        try:
            self._card("\n".join(self._card_lines(rep, gen)))
        except Exception:                # noqa: BLE001 - the card is a view
            log.debug("enrolrun: the card could not be shown", exc_info=True)
        if self._clipboard is not None:
            try:
                self._clipboard("\n".join(rep.lines()))
            except Exception:            # noqa: BLE001 - xclip is optional
                log.debug("enrolrun: the clipboard refused", exc_info=True)
        log.info("in-app enrolment: %d kept over %d station(s), ok=%s, "
                 "generation %d, pool %d", self.samples_kept,
                 self.stations_done, rep.ok, gen, int(rep.pool_total))

    @staticmethod
    def _refusal(rep) -> str:
        """Which refusal, from the CHECK that actually failed.

        Ordered by what he can act on: the sample floor first (sit closer,
        more light), then cohesion (something in that pool is not him), then
        variation (it is all one pose). A detector that never saw a face at
        all is a different sentence again, because none of those remedies
        apply to it.
        """
        failed = {c.name for c in getattr(rep, "checks", ())
                  if c.ok is False}
        if not getattr(rep, "detector_ok", False) or int(rep.accepted) == 0:
            return E26
        if "samples" in failed:
            return E25.format(n=int(rep.pool_total))
        if "cohesion" in failed:
            return E24
        if "variation" in failed or "pose_spread" in failed:
            return E23
        return E26

    def _card_lines(self, rep, gen: int) -> list:
        """The eight-line card. Counts, angles, scores and a verdict."""
        return [
            "Face enrolment -- %s" % ("saved" if gen else "NOT saved"),
            "samples    %d kept of %d offered over %d station(s)"
            % (int(rep.accepted), int(rep.offered), int(rep.stations_run)),
            "pool       %d take(s) judged" % int(rep.pool_total),
            "confidence min %.2f  p50 %.2f  (bar %.2f)"
            % (rep.conf_min, rep.conf_p50, rep.min_conf),
            "yaw        %+.0f..%+.0f deg  spread %.0f"
            % (rep.yaw_min, rep.yaw_max, rep.yaw_spread),
            "cosine     p05 %.3f  p50 %.3f" % (rep.cos_p05, rep.cos_p50),
            "cohesion   min %.3f  (bar %.3f)"
            % (rep.cohesion_min, rep.identity_min),
            ("generation %d, %d in the gallery"
             % (gen, int(rep.gallery_total))
             if gen else "nothing was written; your old gallery is intact"),
        ]


def _swallow(_line: str) -> None:
    """``run_enrolment``'s terminal channel, discarded.

    The in-app run's progress IS the spoken cadence and the earcons; these
    are the CLI's printed lines and they are dropped rather than logged.
    ``Sample.line()`` is numbers only and would pass the assertion, but a
    line per frame at 10 fps is a hundred lines a station in a logfile whose
    contract says the embedding is the only thing enrolment writes down.
    """


def preflight(cfg, *, sensing=None, worker=None, services=None) -> dict:
    """Everything that must be true BEFORE a lens opens. Numbers and a verb.

    Returns ``{"ok": bool, "reply": str, "reason": str}``. Ordered so that
    the most authoritative refusal wins: a run told "the weights are missing"
    when the real answer was "you are in the curfew" sends him to fix the
    wrong thing, which is the exact mistake ``do_enrol`` documents at its own
    top and this mirrors.

    NOTHING HERE WRITES. In particular ``camera.identity`` is CHECKED and
    never set: the CLI flips it after typed consent, but an in-app run that
    was aborted halfway would leave "faces may be written down" switched on
    behind it with no terminal output to notice it in.
    """
    if getattr(services, "enrol_run", None) is not None:
        return {"ok": False, "reply": E10, "reason": "a run is already live"}
    if worker is None:
        return {"ok": False, "reply": E9, "reason": "no preview worker"}
    # 1. SENSING, FIRST AND ALWAYS.
    st = {}
    if sensing is not None:
        try:
            st = dict(sensing.status())
        except Exception:            # noqa: BLE001 - a provider edge
            log.warning("enrolrun: the sensing owner failed", exc_info=True)
            return {"ok": False, "reply": E5, "reason": "sensing unreadable"}
        if not st.get("camera"):
            why = str(st.get("reason") or "")
            # The curfew and the offline switch send him to two different
            # remedies, so they get two different sentences.
            offline = bool(st.get("offline")) or "offline" in why.lower()
            return {"ok": False, "reply": E5 if offline else E6,
                    "reason": "sensing: %s" % (why or "denied")}
    # 2. THE BARS, before anything is opened. min_conf comes from a
    #    user-editable key and at 0 it removes the gate this lane rests on.
    try:
        fe.SampleLimits(min_conf=float(cfg.get("camera.min_conf", 0.6)))
    except (ValueError, TypeError) as exc:
        return {"ok": False, "reply": E12, "reason": str(exc)}
    # 3. THE PHASE GATE. Checked, never flipped.
    if not bool(cfg.get("camera.identity", False)):
        return {"ok": False, "reply": E7,
                "reason": "camera.identity is false"}
    # 4. THE MODELS.
    try:
        from jarvis import camera as cam      # noqa: PLC0415 - lazy
        from jarvis import facedetect         # noqa: PLC0415
        probe = facedetect.probe(
            model_dir=str(cfg.get("camera.model_dir", "") or "") or None,
            backend=cam.face_backend_from_config(cfg))
    except Exception as exc:         # noqa: BLE001 - absence is not a crash
        return {"ok": False, "reply": E8, "reason": str(exc)}
    if not probe.get("ready"):
        missing = sorted(k for k, m in (probe.get("models") or {}).items()
                         if not m.get("ok"))
        return {"ok": False, "reply": E8,
                "reason": "models not usable: %s" % ", ".join(missing)}
    return {"ok": True, "reply": "", "reason": ""}
