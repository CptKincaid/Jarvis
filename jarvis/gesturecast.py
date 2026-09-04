"""Grab, throw, and what happens next: the courier between the hand and the
sinks.

Hunter, verbatim: "lets have a gesture added where i basically reach out and
grab at the screen (in the air) where the camera is and then gesture towards
almost throwing the cast onto the HPCOMPUTER. that would be cool i think"

Three modules already exist and know nothing of each other on purpose:
``jarvis/gesture.py`` turns coordinates into ``grab`` / ``throw`` / ``drop``
events; ``jarvis/cast.py`` knows what a fist is holding (a ``CastSubject``)
and where an open hand can honestly send it (a sink); ``jarvis/handstage.py``
runs the hand tracker on the preview's own frame. THIS module is the glue,
and it holds no frame either: it is handed events, subjects and sinks --
strings and numbers -- and it decides what he hears, sees and is asked.

WHAT HE EXPERIENCES, in order, and where each line comes from:

* He reaches at the lens with an open hand.  ``prepare()`` resolves the
  subject speculatively (HELD > DOCUMENT > TRACK > SCREEN) so the grab is
  instant; nothing is said.
* He closes his hand and holds it still for three frames.  The GRAB tone
  plays FIRST (``heard-you``), the carry chip appears in the console header
  with the subject's name, and Jarvis says ``cast.subject_line``: "Holding
  the thesis draft, sir."  With nothing to hold there is no carry at all:
  one ``held-back`` tone, and only a second empty grab inside 10 s earns
  "I've nothing in hand, sir."
* He flings it left or right and opens his hand.  The chip slides that way;
  the sink for that side is looked up in ``gesture.sinks`` -- which SHIPS
  EMPTY, so every throw lands on the board until he has said which side
  HPCOMPUTER is on, and Jarvis tells him so once: "That went to the board,
  sir. Tell me which side HPCOMPUTER is on and I'll send it there."  A
  board landing plays ``done`` and raises the board (a BoardCommand on
  Jarvis's own surface -- never a new window); a throw at HPCOMPUTER today
  is HELD, plays ``warning``, is said out loud with the live reason, and
  the payload falls back to the board so he is not left holding it.
* He opens his hand where it is, pulls it back still closed, says "drop
  it", or waits 8 s.  ``held-back`` plays and the chip says "dropped".

EVERYTHING HERE ALSO WORKS BY VOICE WITH THE CAMERA OFF. "throw this on
HPCOMPUTER", "put it on the board", "drop it", "what am I holding" and
"HPCOMPUTER is on my right" all route through the same courier from
jarvis/commander.py, resolving the subject fresh when no carry is live.

THE THREADS. ``on_event`` and the payload callbacks arrive on the CAPTURE
thread inside ``PreviewPipeline.grab()``, so they do the cheap things --
a tone, a chip update marshalled through ``post`` -- and hand the cast to a
worker: ``cast()`` blocks on a screen capture and a 0.35 s TCP probe, and a
130 ms grab must not wait on either. The voice methods run on the
commander's worker thread and speak through the CommandResult they return.

THE EARCON COOLDOWN. ``earcons.play`` drops a repeat of the same tone inside
``sound.cooldown_s`` (4 s), which is right for a false-wake metronome and
wrong here: grab-drop-grab inside 4 s would make the second grab silent and
he would conclude the gesture is unreliable. The four gesture tones are
played with a 0.6 s cooldown of their own (``earcons.play(cooldown_s=)``);
the gesture's own 8-frame cooldown bounds the rate.

NOTHING IRREVERSIBLE HAPPENS ON A WAVE. A sink that needs a read-back is
PROPOSED through ``commander.stash_destructive`` and runs only on a spoken
yes inside 60 s, exactly the outbox pattern; with no commander to stash on,
``cast()`` refuses out loud. No frame, no path and no URL is logged here.
"""
from __future__ import annotations

import inspect
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

from jarvis import cast as cast_mod
from jarvis import identity as identity_mod
from jarvis.cast import (
    NOTHING_LINE,
    ROUTABLE_DIRECTIONS,
    UNTAUGHT_LINE,
    BoardSink,
    CastSubject,
    HandoffSink,
    HpcomputerSink,
    cast,
    earcon_for,
    is_taught,
    pick_sink,
    resolve_subject,
    sink_alias,
    sink_label,
    sink_map,
    subject_line,
    teach_sink,
)
from jarvis.gesture import CallablePayload, CastEvent, CastGesture, CastState
from jarvis.logs import get_logger

log = get_logger("gesturecast")

OPTION_SPEAK_GRAB = "gesture.speak_grab"
GESTURE_TONE_COOLDOWN_S = 0.6    # see the module docstring; GUESSED but bounded
NOTHING_REPEAT_S = 10.0          # a second empty grab inside this is answered
PREPARE_TTL_S = 5.0              # a prepared subject older than this is re-resolved
PREPARE_MIN_GAP_S = 1.0          # reach flicker must not spawn xdotool repeatedly
RECENT_TTL_S = 1800.0            # how long the board's CAST slab shows the last one
EVENT_LOG = 32                   # events kept for status()

DROPPED_LINE = "Put down, sir."
SIDE_LINE = "{Target} is on your {side}, sir."
SIDE_UNKNOWN_LINE = ("You haven't told me which side {target} is on, sir. "
                     "Say \"{target} is on my left\" or \"on my right\".")
NO_VOICE_SINK_LINE = "I don't know a target called {name}, sir."

# Tones. The four statuses that speak instead of chiming come back "" from
# cast.earcon_for; the grab is the one tone this module owns outright.
GRAB_TONE = "heard-you"
DROP_TONE = "held-back"


def _call(obj):
    """A service handed in directly, or a zero-arg RESOLVER for one -- a
    plain function or lambda -- because the app builds the commander AFTER
    its services bag. A service object that happens to be callable is not
    a resolver."""
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        try:
            return obj()
        except Exception:                        # noqa: BLE001 - a resolver
            return None
    return obj


class GestureCast:
    """The courier. One per app; the commander, the console and the
    capture thread all hold it.

    Every outward edge is injected: ``speak(text)``, ``earcon(name)``,
    ``board_show()`` (the app's BoardCommand publisher), ``capture()``
    (context.capture_screen), ``identity()`` (the eye's name for whoever
    is in frame, "" for no opinion), ``set_option`` / ``get_option``
    (assistant.json). ``worker(fn)`` runs a blocking cast off the capture
    thread; the suite passes one that runs inline.
    """

    def __init__(self, *, get_option: Optional[Callable] = None,
                 set_option: Optional[Callable] = None,
                 commander: Any = None, spotify: Any = None,
                 capture: Optional[Callable[[], Optional[dict]]] = None,
                 identity: Optional[Callable[[], str]] = None,
                 speak: Optional[Callable[[str], None]] = None,
                 earcon: Optional[Callable[[str], None]] = None,
                 board_show: Optional[Callable[[], Any]] = None,
                 console_visible: Optional[Callable[[], bool]] = None,
                 transfer: Optional[Callable[[str], object]] = None,
                 transport: Optional[Callable[[CastSubject], object]] = None,
                 probe: Optional[Callable[[], tuple]] = None,
                 handoff: Optional[dict] = None,
                 providers: Optional[dict] = None,
                 worker: Optional[Callable[[Callable[[], None]], None]] = None,
                 now: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time,
                 thresholds=None, mirrored: Optional[bool] = None,
                 preview_fps: float = 6.0) -> None:
        self._get = get_option
        self._set = set_option
        self._commander = commander
        self._spotify = spotify
        self._capture = capture
        self._identity_fn = identity
        self._speak_fn = speak
        self._earcon_fn = earcon
        self._board_show = board_show
        self._console_visible = console_visible
        self._providers = providers if isinstance(providers, dict) else None
        self._worker = worker or self._thread
        self._now = now
        self._wall = wall
        self._lock = threading.Lock()
        self._held: Optional[CastSubject] = None
        self._prepared: Optional[CastSubject] = None
        self._prepared_at = -1e9
        self._last_nothing = -1e9
        self._silence_drops = 0
        self._untaught_said = False
        self._recent: Optional[dict] = None
        self._board_row: Optional[dict] = None
        self._events: deque = deque(maxlen=EVENT_LOG)
        self._chip = None
        self._post: Callable[[Callable[[], None]], None] = lambda fn: fn()
        self.casts = 0

        self.registry = {
            "board": BoardSink(self._board_publish,
                               console_visible=self._console_is_visible),
            "hpcomputer": HpcomputerSink(now=now, transfer=transfer,
                                         transport=transport, probe=probe),
            "handoff": HandoffSink(now=now, publish_url=self._board_url,
                                   **(handoff or {})),
        }
        self.payload = CallablePayload(pick_up=self._pick_up,
                                       put_back=self._put_back,
                                       prepare=self._prepare)
        if thresholds is None:
            from jarvis.handstage import thresholds_from_options   # noqa: PLC0415
            thresholds = thresholds_from_options(get_option)
        if mirrored is None:
            mirrored = bool(self._option("gesture.mirrored", False))
        self.machine = CastGesture(thresholds, mirrored=bool(mirrored),
                                   payload=self.payload,
                                   on_event=self.on_event, now=now,
                                   preview_fps=preview_fps)

    # ---------------------------------------------------------- wiring
    def attach_ui(self, *, chip=None, post: Optional[Callable] = None,
                  console_visible: Optional[Callable[[], bool]] = None) -> None:
        """The console hands over its carry chip and a Tk marshaller.
        ``post(fn)`` must run ``fn`` on the Tk thread (root.after(0, fn))."""
        self._chip = chip
        if post is not None:
            self._post = post
        if console_visible is not None:
            self._console_visible = console_visible

    def stage(self, *, get_option: Optional[Callable] = None, **kw):
        """The hand stage for ``PreviewPipeline(hands=...)``, bound to this
        courier's machine. Built here so the console needs to know nothing
        about trackers or thresholds."""
        from jarvis.handstage import HandStage             # noqa: PLC0415
        return HandStage(self.machine, get_option=get_option or self._get,
                         hold_off=self.question_open, now=self._now, **kw)

    def _option(self, key: str, default):
        if not callable(self._get):
            return default
        try:
            value = self._get(key, default)
        except TypeError:
            try:
                value = self._get(key)
            except Exception:                    # noqa: BLE001 - config boundary
                return default
        except Exception:                        # noqa: BLE001 - config boundary
            return default
        return default if value is None else value

    def commander(self):
        return _call(self._commander)

    def spotify(self):
        return _call(self._spotify)

    def question_open(self) -> bool:
        c = self.commander()
        fn = getattr(c, "question_open", None)
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:                        # noqa: BLE001 - a predicate
            return False

    @staticmethod
    def _thread(fn: Callable[[], None]) -> None:
        threading.Thread(target=fn, daemon=True, name="gesture-cast").start()

    # ---------------------------------------------------------- outputs
    def _speak(self, text: str) -> None:
        if not text:
            return
        if self._speak_fn is None:
            log.info("gesture (unspoken): %s", text)
            return
        try:
            self._speak_fn(text)
        except Exception:                        # noqa: BLE001 - the TTS edge
            log.exception("gesture: speak failed")

    def _earcon(self, name: str) -> None:
        if not name:
            return
        if self._earcon_fn is not None:
            try:
                self._earcon_fn(name)
            except Exception:                    # noqa: BLE001 - a beep
                log.debug("gesture: earcon %s failed", name, exc_info=True)
            return
        from jarvis import earcons                         # noqa: PLC0415
        threading.Thread(target=earcons.play, args=(name,),
                         kwargs={"cooldown_s": GESTURE_TONE_COOLDOWN_S},
                         daemon=True, name="earcon-" + name).start()

    def _chip_do(self, method: str, *args) -> None:
        chip = self._chip
        if chip is None:
            return
        fn = getattr(chip, method, None)
        if not callable(fn):
            return

        def run():
            try:
                fn(*args)
            except Exception:                    # noqa: BLE001 - a dead widget
                log.debug("gesture: chip.%s failed", method, exc_info=True)
        try:
            self._post(run)
        except Exception:                        # noqa: BLE001 - a dead window
            log.debug("gesture: chip post failed", exc_info=True)

    def _console_is_visible(self) -> bool:
        fn = self._console_visible
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:                        # noqa: BLE001 - a UI read
            return False

    def _identity(self) -> str:
        fn = self._identity_fn
        if not callable(fn):
            return ""
        try:
            return str(fn() or "")
        except Exception:                        # noqa: BLE001 - the eye
            return ""

    # ------------------------------------------------------- the payload
    def _resolve(self) -> CastSubject:
        with self._lock:
            held = self._held
        return resolve_subject(self.commander(), self.spotify(),
                               now=self._wall(), providers=self._providers,
                               held=held)

    def _prepare(self) -> None:
        """The reach: resolve speculatively so the grab is instant. Rate
        limited, because a reach that flickers at the arm bar must not
        spawn xdotool seven times a second on the capture thread."""
        now = self._now()
        if now - self._prepared_at < PREPARE_MIN_GAP_S:
            return
        self._prepared_at = now
        try:
            self._prepared = self._resolve()
        except Exception:                        # noqa: BLE001 - never raises, but
            log.debug("gesture: prepare failed", exc_info=True)
            self._prepared = None

    def _pick_up(self) -> Optional[tuple]:
        now = self._now()
        subject = self._prepared
        self._prepared = None
        if subject is None or (now - self._prepared_at) > PREPARE_TTL_S:
            subject = self._resolve()
        if subject is None or not subject.holdable:
            return None
        with self._lock:
            self._held = subject
        return subject.kind, subject.spoken

    def _put_back(self, handle: str) -> None:
        with self._lock:
            self._held = None

    def _take_held(self) -> Optional[CastSubject]:
        with self._lock:
            subject, self._held = self._held, None
        return subject

    @property
    def held(self) -> Optional[CastSubject]:
        with self._lock:
            return self._held

    @property
    def carrying(self) -> bool:
        return self.machine.state is CastState.CARRYING

    # --------------------------------------------------------- the events
    def on_event(self, ev: CastEvent) -> None:
        """From the capture thread, inside the machine's lock. Cheap only."""
        self._events.append(ev.numbers_only())
        if ev.kind == "grab":
            self._earcon(GRAB_TONE)
            subject = self.held
            name = subject.spoken if subject is not None else ev.payload
            # The rule depletes over the cap that will ACTUALLY end the
            # carry (the frame cap, 4.1 s at 7.5 fps), and the chip keeps
            # the wall-clock backstop as its own timer for the case where
            # no frame ever comes to end it.
            self._chip_do("hold", name, self.machine.carry_cap_s(),
                          float(self.machine.t.carry_max_s))
            if subject is not None and bool(self._option(OPTION_SPEAK_GRAB, True)):
                line = subject_line(subject)
                self._worker(lambda: self._speak(line))
            return
        if ev.kind == "throw":
            subject = self._take_held()
            self._chip_do("thrown", ev.sector)
            sector = ev.sector
            self._worker(lambda: self._throw(sector, subject, by="gesture"))
            return
        if ev.kind == "drop":
            with self._lock:
                silent = self._silence_drops > 0
                if silent:
                    self._silence_drops -= 1
            if ev.why == "nothing to carry":
                self._earcon(DROP_TONE)
                now = self._now()
                if now - self._last_nothing <= NOTHING_REPEAT_S:
                    self._worker(lambda: self._speak(NOTHING_LINE))
                self._last_nothing = now
                return
            if not silent:
                self._earcon(DROP_TONE)
            self._chip_do("dropped", ev.toward)

    # ---------------------------------------------------------- the cast
    def _propose_with(self, speak: Callable[[str], None]):
        """cast()'s ``propose`` seam, or None when nothing can hold a yes.
        With None, cast() refuses out loud rather than doing it anyway."""
        c = self.commander()
        stash = getattr(c, "stash_destructive", None)
        if not callable(stash):
            return None

        def propose(run, line: str) -> None:
            from jarvis.commander import CommandResult     # noqa: PLC0415

            def confirmed():
                res = run()
                spoken = getattr(res, "spoken", "") or ""
                return CommandResult(handled=True, reply=spoken, speak=True,
                                     status="Cast")
            speak(line)
            stash(confirmed, line)
        return propose

    def _cast(self, sink, subject: CastSubject,
              speak: Callable[[str], None]) -> str:
        self.casts += 1
        return cast(sink, subject, speak=speak,
                    propose=self._propose_with(speak),
                    capture=self._capture, identity=self._identity(),
                    # HIS label, from HIS config -- jarvis/identity.py's one
                    # derivation. Passed rather than looked up inside cast()
                    # so a courier under test says who it means.
                    owner=identity_mod.owner_label(self._get),
                    fallback=self.registry["board"])

    def _record(self, status: str, sink, subject: Optional[CastSubject],
                by: str) -> None:
        name = getattr(sink, "name", "")
        self._recent = {
            "status": status, "sink": name, "target": sink_label(name),
            "kind": subject.kind if subject else "empty",
            "spoken": subject.spoken if subject else "",
            "by": by, "at": float(self._wall()),
        }

    def _throw(self, sector: str, subject: Optional[CastSubject],
               by: str = "gesture") -> str:
        """Off the capture thread. Picks the sink for the side, casts,
        speaks what cast() said, plays the tone for the outcome."""
        sink = pick_sink(sector, get_option=self._get, registry=self.registry)
        lines: list = []
        if subject is None or not subject.holdable:
            status = "empty"
            lines.append(NOTHING_LINE)
        else:
            try:
                status = self._cast(sink, subject, lines.append)
            except Exception:                    # noqa: BLE001 - never raises, but
                log.exception("gesture: cast failed")
                status = "held"
                lines.append(cast_mod.SINK_RAISED_LINE.format(
                    Target=cast_mod._cap(getattr(sink, "label", "it"))))
        if by == "gesture" and status == "landed" and \
                sector in ROUTABLE_DIRECTIONS and \
                getattr(sink, "name", "") == "board" and \
                not is_taught(self._get) and not self._untaught_said:
            self._untaught_said = True
            lines.append(UNTAUGHT_LINE)
        self._record(status, sink, subject, by)
        tone = earcon_for(status)
        if tone:
            self._earcon(tone)
        self._chip_do("landed" if status in ("landed", "proposed") else
                      "held", getattr(sink, "label", ""))
        for line in lines:
            self._speak(line)
        log.info("gesture: throw %s -> %s: %s (%s)", sector,
                 getattr(sink, "name", "?"), status, by)
        return status

    # ------------------------------------------------------ board surface
    def _board_publish(self, subject: CastSubject) -> None:
        """BoardSink.publish: a row on Jarvis's OWN surface. Raises the
        board through the app's BoardCommand seam (its feed's first tick is
        immediate and reads ``recent()``), never a window of its own."""
        self._board_row = {"kind": subject.kind, "spoken": subject.spoken,
                           "at": float(self._wall())}
        if callable(self._board_show):
            self._board_show()

    def _board_url(self, url: str, subject: CastSubject) -> None:
        """HandoffSink.publish_url: the address goes on the board row,
        never into a log line."""
        self._board_row = {"kind": "handoff", "spoken": subject.spoken,
                           "url": url, "at": float(self._wall())}
        if callable(self._board_show):
            self._board_show()

    def recent(self) -> Optional[dict]:
        """The last cast for the board's CAST slab (jarvis/board.py), or
        None once it is old news. Strings and numbers only."""
        rec = self._recent
        if rec is None:
            return None
        if self._wall() - float(rec.get("at", 0.0)) > RECENT_TTL_S:
            return None
        out = dict(rec)
        row = self._board_row
        if isinstance(row, dict) and row.get("url"):
            out["url"] = str(row["url"])
        return out

    # ------------------------------------------------------------ voice
    def throw_by_voice(self, sink_name: str) -> tuple:
        """"throw this on HPCOMPUTER" -> (line to speak, status). The held
        subject if a carry is live, else the subject resolved now; the
        machine's carry is ended silently, because the sentence is the
        throw. The carry caps run only when a frame arrives to test them,
        so ``sweep()`` comes first: a carry whose frames simply stopped
        stayed live for as long as the silence lasted, and the spoken
        throw then cast the STALE subject (MEASURED at 60 s)."""
        self.machine.sweep()
        name = sink_alias(sink_name)
        sink = self.registry.get(name)
        if sink is None:
            return NO_VOICE_SINK_LINE.format(name=str(sink_name).strip()), "refused"
        subject = self._take_held()
        if self.carrying:
            with self._lock:
                self._silence_drops += 1
            self.machine.cancel("thrown by voice")
        if subject is None:
            subject = self._resolve()
        lines: list = []
        if not subject.holdable:
            return NOTHING_LINE, "empty"
        status = self._cast(sink, subject, lines.append)
        self._record(status, sink, subject, "voice")
        tone = earcon_for(status)
        if tone:
            self._earcon(tone)
        self._chip_do("landed" if status in ("landed", "proposed") else
                      "held", getattr(sink, "label", ""))
        line = " ".join(s for s in lines if s).strip()
        if not line and status == "landed":
            line = cast_mod.BOARD_LINE if name == "board" else ""
        return line, status

    def drop_by_voice(self) -> str:
        """"drop it" / "put it down": ends a live carry (the held-back tone
        and the chip follow through on_event) or clears a held subject.
        The wall-clock cap is applied first, so a carry the camera stopped
        feeding is already down before the sentence lands."""
        self.machine.sweep()
        if self.carrying:
            self.machine.cancel("spoken")
            return DROPPED_LINE
        if self._take_held() is not None:
            self._chip_do("dropped", "")
            return DROPPED_LINE
        return NOTHING_LINE

    def spoken_over(self) -> None:
        """ANY other spoken command supersedes a live carry -- a sentence
        outranks a gesture -- quietly: no tone, the chip just goes."""
        if not self.carrying:
            return
        with self._lock:
            self._silence_drops += 1
        self.machine.cancel("spoken over")

    def holding_line(self) -> str:
        """"what are you holding" -- and an expired carry is not held."""
        self.machine.sweep()
        subject = self.held
        return subject_line(subject) if subject is not None else NOTHING_LINE

    def teach(self, side: str, sink_name: str) -> str:
        if not callable(self._set):
            return cast_mod.TAUGHT_FAILED_LINE
        return teach_sink(side, sink_name, set_option=self._set,
                          get_option=self._get, registry=self.registry)

    def side_line(self, sink_name: str) -> str:
        name = sink_alias(sink_name)
        label = sink_label(name)
        for side, sink in sink_map(self._get).items():
            if sink == name and side in ROUTABLE_DIRECTIONS:
                return SIDE_LINE.format(Target=cast_mod._cap(label), side=side)
        return SIDE_UNKNOWN_LINE.format(target=label)

    # ----------------------------------------------------------- status
    def status(self) -> dict:
        held = self.held
        return {"state": self.machine.state.value,
                "held": held.spoken if held else "",
                "held_kind": held.kind if held else "",
                "casts": self.casts, "taught": is_taught(self._get),
                "sinks": dict(sink_map(self._get)),
                "recent": self.recent() or {},
                "events": list(self._events)}


__all__ = [
    "DROPPED_LINE", "DROP_TONE", "GESTURE_TONE_COOLDOWN_S", "GRAB_TONE",
    "GestureCast", "NOTHING_REPEAT_S", "NO_VOICE_SINK_LINE",
    "OPTION_SPEAK_GRAB", "RECENT_TTL_S", "SIDE_LINE", "SIDE_UNKNOWN_LINE",
]
