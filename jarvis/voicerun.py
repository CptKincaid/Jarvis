"""Enrolling a VOICE inside the running app, judged before anything is stored.

WHAT THIS REPLACES, and the replacement is the whole point.
``JarvisApp.enroll_speaker`` -- one press in the settings drawer, next to a
slider -- records a fixed fifteen seconds of whatever the room contained and
hands it to ``speaker.enroll_from_audio``, which APPENDS one embedding to
``voiceprint.npz``. No loudness floor. No cohesion check. No separation check.
No consent, no gate, no owner check, no label, and no undo: that file is
written tmp->replace with no generations and nothing to roll back to. It is
one click deep and it is irreversible, and it is the thing this module exists
to take out of his hands.

WHAT IT IS NOT. It is not a second enrolment and it is not a second set of
bars. ``jarvis/voiceenrol.py`` owns the loudness floor, the cohesion bar, the
separation bar and the owner's anchor -- the same functions
``scripts/voice_enrol.py`` calls, moved into the package so there is one copy
rather than two -- and ``jarvis/voicegallery.py`` owns the store, its
generations and its rollback. This module supplies a microphone turn-taking
discipline, a spoken progress channel and a set of controls, and nothing else.

THE MICROPHONE IS HELD FOR THE WHOLE RUN, AND THAT IS THE OPPOSITE OF WHAT
``jarvis/enrolrun.py`` DOES. The face run never acquires the mic, because
``MicArbiter`` pauses the hotword on the first acquire and the abort WORD has
to stay hearable -- the camera is not the abort channel, so the two do not
collide. Here they do: the thing being recorded is his voice, and a live wake
word on an open capture is a wake word firing into an enrolment take. So the
arbiter is taken ONCE for the run, the hotword is deaf for the duration, and
he is TOLD that before the first take, because it makes his abort a button
rather than a word.

AND THE REFUSAL THAT MATTERS MOST: NOT WHILE JARVIS IS TALKING. ``MicArbiter``
is a re-entrant DEPTH COUNTER and not a mutex -- it will happily let two
consumers hold at once -- and ``jarvis/tts.py`` takes ``acquire("tts")`` for
talkback. There is no AEC on this box. So a voice enrolment started while
Jarvis is mid-sentence records Jarvis's own voice out of the speakers and
stores it as his, which is exactly what the drawer button does today. The
preflight refuses on ``tts.is_speaking()`` and the run waits for quiet before
each take.

NOTHING BUT THE EMBEDDING REACHES THE DISK, and no recording is kept anywhere
at any point: the audio becomes 192 numbers and is dropped on the floor. That
is the promise ``consent.VOICE_LINES`` makes, kept in the window as well as at
the terminal.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Sequence

import numpy as np

from jarvis import consent as cs
from jarvis import voiceenrol as ve
from jarvis import voicegallery as vg
from jarvis.logs import get_logger

log = get_logger("voicerun")

# The bars are ve's, BOUND rather than re-declared, so ``vr.MIN_RMS is
# ve.MIN_RMS`` and a change to the floor cannot leave this module behind.
MIN_RMS = ve.MIN_RMS
TAKES = ve.TAKES
SECONDS = ve.SECONDS
PROMPTS = ve.PROMPTS

# The sample rate every embedding in this tree is taken at.
RATE = 16000
# How long he is given to get ready before the tone. Shorter than the face
# run's: he is already sitting where he speaks from.
SETTLE_S = 2.0
# How long the run will wait for Jarvis to stop talking before a take. A
# spoken prompt is a second or two; anything past this is a stuck TTS and the
# run says so rather than recording over it.
QUIET_WAIT_S = 8.0
# How often the quiet wait looks again.
POLL_S = 0.1
# How many attempts a run may make in total. Eight kept takes with a few
# retries is normal; an unbounded loop on a muted microphone is a run that
# never ends, and every loop in this tree carries a hard bound.
MAX_TRIES = 20
# The whole run's ceiling. Eight takes of eight seconds plus prompts is about
# two and a half minutes; this is the backstop.
RUN_MAX_S = 420.0

STOP_PRESSED = "stopped from the tab"
STOP_SPOKEN = "stopped"

# ---------------------------------------------------------- the sentences
# Authored for TTS, the way jarvis/enrolrun.py authors its own: the script's
# printed lines carry "--" and words in CAPS for a terminal reader and land
# badly spoken.
V1 = ("Right, sir. Eight takes of about eight seconds. I'm taking the "
      "microphone for the whole run, so the wake word can't hear you -- press "
      "Stop on the tab if you want to end it. Nothing is written until all "
      "eight are judged.")
V2 = "{n} of {total}. {heading} Say: {line}"
V3 = "Kept. {n} of {total}."
V4 = "Saved, sir. {takes} takes, generation {gen}. Your previous one is still there."
V5 = "Not saved, sir, and nothing was written. {why}"
V6 = ("Stopped. Nothing was written and your voice gallery is untouched.")
V7 = ("I couldn't get a usable take, sir, after {tries} tries. Nothing was "
      "written.")
V8 = ("Something went wrong in the middle of that, sir. The microphone is "
      "back, nothing was written, and your gallery is untouched.")
V9 = ("I'm still talking, sir, so I can't record yet -- try again in a "
      "moment.")
V10 = ("I started talking over that one, sir, so I've thrown it away. Again.")

# Preflight refusals. No capture is opened on any of these paths.
P1 = ("Not while I'm talking, sir -- I'd enrol my own voice. Try again in a "
      "moment.")
P2 = "There's a recording open already, sir. Finish that one first."
P3 = "A voice enrolment is already running, sir. Press Stop to end it."
P4 = "There's no microphone on this box, sir, so there's nothing to record."
P5 = ("I can't tell whether I'm talking, sir, so I won't open the microphone "
      "and record myself. That one needs a terminal.")
P6 = ("That would store a measurement of somebody else, sir, and only they "
      "can agree to it. The command is on your clipboard.")
# THE CROSS-CHECK. The face run's whole progress channel is SPEECH -- five
# stations, each announced -- so recording through one stores Jarvis reading
# the stations out, under his label.
P7 = ("I'm enrolling your face just now, sir, and I talk my way through that "
      "-- recording now would enrol my own voice. Stop that one first.")


def _speakable(why: str, cap: int = 400) -> str:
    """One of the shared bars' refusals, said out loud.

    THE WORDS ARE NOT REWRITTEN, only made speakable: the numbers in them are
    the numbers he can act on, and a fresh sentence per refusal would be a
    second description of a decision this module does not make. Em-dashes and
    newlines are what TTS stumbles on, so those go and nothing else does.
    """
    text = " ".join(str(why or "").split())
    text = text.replace(" -- ", ", ").replace("--", ",")
    if len(text) > cap:
        text = text[:cap].rsplit(" ", 1)[0] + "..."
    return text


def _arbiter_of(recorder):
    """The mic arbiter, whichever name this recorder exposes it under.

    ``Recorder`` holds it as ``_arbiter``; the injected ones in the suite
    expose ``arbiter``. A recorder with neither is refused rather than
    recorded from without turn-taking.
    """
    for name in ("arbiter", "_arbiter"):
        got = getattr(recorder, name, None)
        if got is not None and hasattr(got, "acquire"):
            return got
    return None


# ------------------------------------------------- who owns the devices
# THE SLOT NAMES ARE HERE ONCE. Both preflights ask this function rather than
# each reading the namespace itself, because the defect that shipped was
# exactly two preflights each looking only at its own slot.
FACE_SLOT = "enrol_run"
VOICE_SLOT = "voice_run"


def runs_live(services) -> tuple:
    """Which in-app enrolments are parked right now: ``()``, ``("face",)``,
    ``("voice",)`` or both.

    WHY THIS EXISTS AND WHY IT IS NOT A LOCK. ``recorder.MicArbiter`` looks
    like the thing that should stop two consumers and it is not: it is a
    re-entrant DEPTH COUNTER whose only job is pausing the hotword on the
    first acquire and resuming it on the last. Two runs on two threads both
    get their context manager and both proceed, and a deeper or longer
    acquire only makes the wake word deafer. So the exclusion is decided
    BEFORE a device opens, in the preflight, and both preflights ask here.

    PARKED COUNTS AS LIVE, deliberately. A run clears its own slot in a
    ``finally``, so the window in which a finished run is still parked is
    short -- and erring the other way means starting a second consumer
    against a run that is one instruction from its last capture. The cost of
    being wrong in this direction is one sentence; in the other it is a pool
    of Jarvis's own voice.
    """
    if services is None:
        return ()
    out = []
    if getattr(services, FACE_SLOT, None) is not None:
        out.append("face")
    if getattr(services, VOICE_SLOT, None) is not None:
        out.append("voice")
    return tuple(out)


def talking(tts):
    """Is Jarvis making sound, or about to? ``True``/``False``, or ``None``
    when the seam cannot be read at all.

    THE SHAPE IS READ, NOT ASSUMED, and that is a bug fix rather than
    politeness. ``jarvis.tts.TTS`` exposes ``busy`` and ``is_speaking`` as
    PROPERTIES returning bools; this module called ``tts.is_speaking()``,
    which against the shipped object raises ``TypeError: 'bool' object is
    not callable``. The preflight caught it, fell to "I can't tell whether
    I'm talking" and refused -- so the safe button that replaces the drawer's
    dangerous one refused every press on his live box, and only the suite's
    method-shaped stand-in ever said otherwise.

    ``busy`` IS PREFERRED over ``is_speaking``: it is "speaking now OR lines
    still queued", and a queued burst is a burst that will land inside the
    take. ``SpeakingState`` marks first AUDIO, so during the render window
    ``is_speaking`` alone says idle while a sentence is on its way to the
    speakers.
    """
    if tts is None:
        return None
    for name in ("busy", "is_speaking"):
        try:
            got = getattr(tts, name)
        except AttributeError:
            continue
        except Exception:            # noqa: BLE001 - a property that raises
            log.warning("voicerun: tts.%s could not be read", name,
                        exc_info=True)
            return None
        try:
            return bool(got() if callable(got) else got)
        except Exception:            # noqa: BLE001 - a seam that cannot say
            log.warning("voicerun: tts.%s could not be read", name,
                        exc_info=True)
            return None
    return None


def _rms(audio) -> float:
    try:
        arr = np.asarray(audio, dtype=np.float64).ravel()
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(arr))))
    except Exception:                # noqa: BLE001 - an unmeasurable take
        return 0.0


def preflight(*, recorder=None, tts=None, services=None, label="",
              owner="", consent_how="") -> dict:
    """Everything that must be true BEFORE a capture opens. ``{ok, reply,
    reason}``.

    Ordered so the most authoritative refusal wins, the rule
    ``enrolrun.preflight`` already writes down: a run told "no microphone"
    when the real answer was "I am talking" sends him to fix the wrong thing.

    NOTHING HERE WRITES and nothing here opens a device.
    """
    live = runs_live(services)
    if "voice" in live:
        return {"ok": False, "reply": P3, "reason": "a run is already live"}
    # THE CROSS-CHECK, and it is the whole of defect 1. A face run holds no
    # microphone, but it TALKS -- five stations, each announced -- and with
    # no AEC on this box a capture underneath that is a capture of Jarvis.
    if "face" in live:
        return {"ok": False, "reply": P7, "reason": "a face run is live"}
    if recorder is None:
        return {"ok": False, "reply": P4, "reason": "no recorder"}
    if not getattr(recorder, "mic_available", False):
        return {"ok": False, "reply": P4, "reason": "no microphone"}
    if _arbiter_of(recorder) is None:
        return {"ok": False, "reply": P4, "reason": "no mic arbiter"}
    # THE TALKING CHECK IS FAIL-CLOSED. A tts seam that cannot answer is not
    # evidence of silence, and the cost of guessing wrong is his voiceprint
    # holding Jarvis's voice.
    if tts is None:
        return {"ok": False, "reply": P5, "reason": "no tts seam"}
    speaking = talking(tts)
    if speaking is None:
        return {"ok": False, "reply": P5, "reason": "tts unreadable"}
    if speaking:
        return {"ok": False, "reply": P1, "reason": "tts is speaking"}
    if getattr(recorder, "recording", False):
        return {"ok": False, "reply": P2, "reason": "a capture is open"}
    # SOMEBODY ELSE'S VOICE NEEDS THEIR OWN CONSENT, and a window cannot take
    # it: the dialog is driven by whoever is already logged in, which is the
    # one thing consent.take_at_terminal exists to make impossible. The owner
    # enrolling HIMSELF is the single exemption and it is already spelled.
    if owner and str(label or "") != str(owner):
        return {"ok": False, "reply": P6, "reason": "not the owner's voice"}
    if consent_how and consent_how == cs.HOW_TERMINAL:
        # A false attestation on disk is worse than none: a reader later
        # believes a terminal ceremony happened. Nothing in the window may
        # write "typed".
        return {"ok": False, "reply": P6,
                "reason": "a console run may not attest a terminal ceremony"}
    return {"ok": True, "reply": "", "reason": ""}


class VoiceRun:
    """One in-app voice enrolment. Built by the app, parked on
    ``services.voice_run``, thrown away when it finishes.

    Every collaborator is injected, which is what makes it testable with no
    microphone and no model: ``recorder`` records, ``embed`` turns audio into
    192 numbers, ``tts`` says whether Jarvis is talking, ``say`` speaks,
    ``card`` publishes display-only text and ``earcon`` plays a tone.
    """

    def __init__(self, *, label: str, gallery, recorder, embed: Callable,
                 tts=None, say: Callable[[str], None] = lambda _t: None,
                 card: Callable[[str], None] = lambda _t: None,
                 earcon: Optional[Callable[[str], None]] = None,
                 services=None, owner: str = "",
                 consent_how: str = cs.HOW_OWNER,
                 takes: int = TAKES, seconds: float = SECONDS,
                 settle_s: float = SETTLE_S,
                 max_tries: int = MAX_TRIES,
                 run_max_s: float = RUN_MAX_S,
                 quiet_wait_s: float = QUIET_WAIT_S,
                 owner_vectors: Optional[Sequence] = None,
                 speech_seconds: Optional[Callable] = None,
                 now: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.label = str(label or "")
        self.gallery = gallery
        self.recorder = recorder
        self.embed = embed
        self.tts = tts
        self.services = services
        self.owner = str(owner or "")
        self.consent_how = str(consent_how or "")
        self.takes = int(takes)
        self.seconds = float(seconds)
        self.settle_s = float(settle_s)
        self.max_tries = int(max_tries)
        self.run_max_s = float(run_max_s)
        self.quiet_wait_s = float(quiet_wait_s)
        self.owner_vectors = owner_vectors
        self._speech_seconds = speech_seconds
        self._now = now
        self._sleep = sleep
        self._say = say
        self._card = card
        self._earcon = earcon
        self._stop_reason = ""
        self._lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.finished = threading.Event()
        self.started_at = 0.0
        # Numbers a caller or a test may read without touching a device.
        self.kept = 0
        self.tries = 0
        self.saved_generation = 0

    # ------------------------------------------------------------- speaking
    def _speak(self, text: str) -> None:
        try:
            self._say(text)
        except Exception:            # noqa: BLE001 - TTS is not the run
            log.debug("voicerun: the voice failed", exc_info=True)

    def _show(self, text: str) -> None:
        try:
            self._card(text)
        except Exception:            # noqa: BLE001 - a card is not the run
            log.debug("voicerun: the card failed", exc_info=True)

    def _tone(self, name: str) -> None:
        if self._earcon is None:
            return
        try:
            self._earcon(name)
        except Exception:            # noqa: BLE001 - sound is a courtesy
            log.debug("voicerun: the earcon failed", exc_info=True)

    # ------------------------------------------------------------ controls
    @property
    def running(self) -> bool:
        t = self.thread
        return bool(t is not None and t.is_alive())

    def abort(self, reason: str = STOP_PRESSED) -> None:
        """End the run. Safe from any thread and safe before it starts --
        stopping is the direction that must always work."""
        with self._lock:
            if not self._stop_reason:
                self._stop_reason = str(reason or STOP_PRESSED)

    def _stopping(self) -> str:
        with self._lock:
            return self._stop_reason

    def _should_stop(self) -> str:
        stop = self._stopping()
        if stop:
            return stop
        if self.started_at and (self._now() - self.started_at) > self.run_max_s:
            return "the run ran past its ceiling"
        return ""

    def start(self) -> bool:
        """Spawn the run on its own daemon thread. True if it began.

        NOT the Tk thread: this is minutes of recording and sleeping, and
        anything of that shape on Tk freezes the console -- and the console is
        where the Stop button he was just told to press lives.
        """
        if self.running:
            return False
        self.started_at = self._now()
        self.thread = threading.Thread(target=self.walk, name="voice-enrol",
                                       daemon=True)
        self.thread.start()
        return True

    # ---------------------------------------------------------------- the run
    def walk(self) -> bool:
        """The whole run. Never raises, and always gives the microphone back.

        THE ARBITER IS TAKEN ONCE, around everything. ``record_fixed`` takes
        it again inside -- that is what re-entrancy is for and it is harmless
        -- but between two separate acquires the hotword would come back live
        on a microphone that is about to be recorded into, which is the state
        this run may not be in.
        """
        if not self.started_at:
            self.started_at = self._now()
        arbiter = _arbiter_of(self.recorder)
        if arbiter is None:
            self._speak(P4)
            self.finished.set()
            return False
        saved = False
        try:
            with arbiter.acquire("voice-enrol"):
                saved = self._run()
        except Exception:            # noqa: BLE001 - nothing may escape
            log.exception("voicerun: the run failed")
            self._speak(V8)
        finally:
            self.finished.set()
            self._unpark()
        return saved

    def _unpark(self) -> None:
        try:
            if getattr(self.services, "voice_run", None) is self:
                self.services.voice_run = None
        except Exception:            # noqa: BLE001 - a slim services
            log.debug("voicerun: could not unpark", exc_info=True)

    def _run(self) -> bool:
        if self._should_stop():
            self._speak(V6)
            return False
        self._speak(V1)
        staged = self._collect()
        stop = self._should_stop()
        if stop:
            self._speak(V6)
            log.info("voicerun: %s -- %d takes discarded", stop, len(staged))
            return False
        if len(staged) < self.takes:
            self._speak(V7.format(tries=self.tries))
            log.info("voicerun: only %d of %d takes in %d tries; nothing "
                     "written", len(staged), self.takes, self.tries)
            return False
        return self._judge_and_save(staged)

    def _collect(self) -> list:
        """The takes, staged in memory. NOTHING is written in here."""
        staged: list = []
        name = self.label.capitalize()
        while len(staged) < self.takes and self.tries < self.max_tries:
            if self._should_stop():
                return staged
            heading, line = PROMPTS[len(staged) % len(PROMPTS)]
            self.tries += 1
            self._speak(V2.format(n=len(staged) + 1, total=self.takes,
                                  heading=heading, line=line.format(name=name)))
            take = self._take_once(heading)
            if take is None:
                continue
            staged.append(take)
            self.kept = len(staged)
            self._speak(V3.format(n=len(staged), total=self.takes))
            self._show("take %d: rms %.4f, %.2f s of speech"
                       % (len(staged), take[2], take[1]))
        return staged

    def _take_once(self, heading):
        """One capture, with Jarvis proved silent at BOTH ENDS of it.

        THE BEFORE-CHECK ALONE IS NOT ENOUGH, and that is the second half of
        defect 1. Excluding the face run closes the door the adversary found;
        this closes the one behind it, because Jarvis speaks from a dozen
        places that have nothing to do with this run -- a timer going off, an
        arriving message, a reminder he set this morning. A burst that begins
        one second into an eight-second capture is a take of his voice with
        seven seconds of Jarvis's underneath it, and with no AEC on this box
        it cannot be salvaged. So it is DROPPED and the run tries again,
        which costs him one prompt rather than his voiceprint.

        FAIL CLOSED at the far edge too: a seam that cannot answer after the
        capture is not evidence of silence.
        """
        if not self._wait_for_quiet():
            self._speak(V9)
            return None
        if self.settle_s > 0:
            self._sleep(self.settle_s)
        if self._should_stop():
            return None
        self._tone("listen")
        audio = self._record()
        self._tone("done")
        if self.tts is not None and talking(self.tts) is not False:
            log.info("voicerun: a take was discarded -- Jarvis spoke through it")
            self._speak(V10)
            return None
        return self._keep(audio, heading)

    def _wait_for_quiet(self) -> bool:
        """Do not record over Jarvis. Bounded, and a seam that cannot answer
        is treated as still talking -- fail closed, for the same reason the
        preflight does."""
        if self.tts is None:
            return True
        deadline = self._now() + self.quiet_wait_s
        # BOUNDED BOTH WAYS. The clock is injected, so a frozen ``now`` in a
        # test would make a while-on-the-clock loop run for ever; every loop
        # in this tree carries a hard count as well.
        for _ in range(max(1, int(self.quiet_wait_s / POLL_S) + 1)):
            speaking = talking(self.tts)
            if speaking is None:
                return False
            if not speaking:
                return True
            if self._should_stop():
                return False
            if self._now() >= deadline:
                return False
            self._sleep(POLL_S)
        return False

    def _record(self):
        try:
            return self.recorder.record_fixed(self.seconds)
        except Exception:            # noqa: BLE001 - a device edge
            log.warning("voicerun: the capture failed", exc_info=True)
            return None

    def _keep(self, audio, heading):
        """One take, measured. ``(embedding, speech_s, rms, heading)`` or None.

        A REFUSED TAKE IS NOT A KEPT TAKE, and the floor is judged against the
        kept ones: eight tries with six usable takes is a six-take pool, not
        an eight-take one.
        """
        if audio is None or len(audio) == 0:
            self._speak("Nothing came through on that one, sir. Again.")
            return None
        rms = _rms(audio)
        good, why = ve.take_ok(rms)
        if not good:
            self._speak(_speakable(why))
            return None
        try:
            emb = self.embed(audio)
        except Exception:            # noqa: BLE001 - the model's boundary
            log.exception("voicerun: the embedder failed")
            raise
        if emb is None:
            self._speak("There wasn't enough speech in that one, sir. Again.")
            return None
        return (emb, self._speech_s(audio), rms, heading)

    def _speech_s(self, audio) -> float:
        if self._speech_seconds is None:
            try:
                return float(len(audio)) / RATE
            except Exception:        # noqa: BLE001
                return 0.0
        try:
            return float(self._speech_seconds(audio))
        except Exception:            # noqa: BLE001 - a seam
            log.debug("voicerun: the speech measure failed", exc_info=True)
            return 0.0

    def _judge_and_save(self, staged) -> bool:
        """THE POOL IS JUDGED BEFORE A SINGLE WRITE, by the shared bars.

        ``voiceenrol.pool_ok`` is the same function ``scripts/voice_enrol.py``
        calls: cohesion, the same-name-is-the-same-person check, separation
        against everybody already enrolled, and the owner's anchor against
        voiceprint.npz. Nothing here decides anything it decides.
        """
        vectors = [e for e, _s, _r, _h in staged]
        try:
            good, why = ve.pool_ok(self.gallery, self.label, vectors,
                                   owner=self.owner,
                                   owner_vectors=self.owner_vectors)
        except Exception:            # noqa: BLE001 - a gallery that cannot say
            log.exception("voicerun: the pool could not be judged")
            self._speak(V8)
            return False
        if not good:
            self._speak(V5.format(why=_speakable(why)))
            self._show(str(why))
            log.info("voicerun: pool refused for %s; nothing written",
                     self.label)
            return False
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            for emb, speech_s, rms, heading in staged:
                self.gallery.add(self.label, emb, len_s=speech_s, rms=rms,
                                 note=heading, at=stamp, src="enrol")
            # THE ATTESTATION IS STORED WITH THE POOL, and it is whatever
            # ceremony actually happened -- never "typed" from a window.
            self.gallery.set_consent(self.label, self.consent_how)
            gen = int(self.gallery.save(
                "in-app enrol %s (%d takes, consent %s)"
                % (self.label, len(staged), self.consent_how)))
        except ValueError as exc:
            # The gallery's own last refusal (a shrink, a collapsed pool).
            self._speak(V5.format(why=_speakable(str(exc))))
            self._show(str(exc))
            return False
        except Exception:            # noqa: BLE001 - a store edge
            log.exception("voicerun: the pool could not be saved")
            self._speak(V8)
            return False
        self.saved_generation = gen
        self._speak(V4.format(takes=len(staged), gen=gen))
        self._show("\n".join(self._card_lines(staged, gen)))
        log.info("voicerun: %s enrolled, %d takes, generation %d",
                 self.label, len(staged), gen)
        return True

    def _card_lines(self, staged, gen: int) -> list:
        """Numbers, and nothing that could be played back."""
        rmss = [r for _e, _s, r, _h in staged]
        speech = [s for _e, s, _r, _h in staged]
        med = vg.median_pairwise([e for e, _s, _r, _h in staged])
        return [
            "voice enrolment -- %s" % self.label,
            "takes      %d kept of %d tried (floor %d)"
            % (len(staged), self.tries, self.takes),
            "rms        %.4f - %.4f  (floor %.4f)"
            % (min(rmss), max(rmss), MIN_RMS),
            "speech     %.1f - %.1f s per take" % (min(speech), max(speech)),
            "cohesion   median pairwise cosine %.3f  (bar %.2f)"
            % (med if med is not None else float("nan"),
               vg.COLLAPSED_MEDIAN_COSINE),
            "consent    %s" % self.consent_how,
            "generation %d written; the previous one is still on disk" % gen,
        ]
