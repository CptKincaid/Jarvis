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
from jarvis import castview as view_mod
from jarvis import identity as identity_mod
from jarvis import screens as screens_mod
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

# --- the screen cast (jarvis/screens.py, jarvis/castview.py) --------------
# How stale the last hand row may be and still name the screen he grabbed
# at. At 6-7.5 fps the grab lands on the very frame the row came from; this
# is the backstop for a capture thread that stalled between the two, and it
# is what makes "the preview shut" also mean "no screen cast".
SOURCE_TTL_S = 1.5
# Head yaw is read from the last clean face row in the window BEFORE the
# fist closes, never at the grab instant: handstage.py's own docstring says
# the reaching arm crosses the face at exactly the moment the gesture
# matters, which is why attention is LATCHED for 3 s rather than sampled. A
# design that read yaw at the grab would read it with the arm in the way.
YAW_WINDOW_S = 1.0
YAW_ROWS = 24                    # ~3-4 s of face rows at 6-7.5 fps

DROPPED_LINE = "Put down, sir."
SIDE_LINE = "{Target} is on your {side}, sir."
SIDE_UNKNOWN_LINE = ("You haven't told me which side {target} is on, sir. "
                     "Say \"{target} is on my left\" or \"on my right\".")
NO_VOICE_SINK_LINE = "I don't know a target called {name}, sir."
NOTHING_THAT_WAY_LINE = "There's nothing that way, sir."
NO_SCREEN_CAST_LINE = "Screen casting is off, sir."

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


class _Source:
    """A spoken cast names the destination, so the source is the OTHER
    machine and there is no grab to score. This is that source, in the one
    shape ``_cast_view`` reads -- rather than a second code path."""

    __slots__ = ("machine",)

    def __init__(self, machine: str) -> None:
        self.machine = str(machine)


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
                 view_launch: Optional[Callable[[str], object]] = None,
                 view_stop: Optional[Callable[[], object]] = None,
                 view_alive: Optional[Callable[[], bool]] = None,
                 view_ack_wait: Optional[Callable[[float], object]] = None,
                 view_settle: Optional[Callable[[float], object]] = None,
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

        # --- the screen cast -------------------------------------------
        # OFF BY DEFAULT and UNLEARNED by default: both keys ship absent.
        # Casting a screen puts a desktop on a monitor in his office; the
        # board cast this sits beside costs three seconds of his
        # attention. The two do not deserve the same default.
        self.screens = screens_mod.load(self._get)
        self.learner = screens_mod.ScreenLearner()
        self.relay = view_mod.CastRelay(
            now=now, on_layout=self.note_layout)
        self.view_state = view_mod.ViewState(now=now)
        self._last_hand: Optional[dict] = None
        self._yaws: deque = deque(maxlen=YAW_ROWS)
        self._source = None
        self._unarmed_said = False
        self.yaw_misses = 0
        self.source_reads = 0
        self.refusals = 0

        self.registry = {
            "board": BoardSink(self._board_publish,
                               console_visible=self._console_is_visible),
            "hpcomputer": HpcomputerSink(now=now, transfer=transfer,
                                         transport=transport, probe=probe),
            "handoff": HandoffSink(now=now, publish_url=self._board_url,
                                   **(handoff or {})),
            # The two screen-view sinks. They live in jarvis/castview.py
            # rather than jarvis/cast.py because
            # test_the_module_cannot_open_a_window_or_a_lens greps that
            # file for exactly the launcher this needs -- and a viewer
            # window is what that test exists to keep out. Both launchers
            # are INJECTED and default to None, so a courier built by a
            # test opens nothing.
            # ``view_alive`` is not optional decoration: without it the
            # sink cannot tell whether the viewer it spawned is still
            # there, so it reports itself unavailable rather than claiming
            # a landing off a Popen that merely forked.
            "spark-view": view_mod.SparkViewSink(
                launch=view_launch, stop=view_stop, alive=view_alive,
                settle=view_settle, state=self.view_state, now=now),
            # ``view_ack_wait`` is how the sink waits for HPCOMPUTER to
            # acknowledge the verb; None is the real one -- the relay's own
            # event, set when the poll route answers the Windows script. A
            # test injects the helper's half of the round trip here rather
            # than reaching inside the sink.
            "hp-view": view_mod.HpViewSink(relay=self.relay,
                                           state=self.view_state,
                                           wait=view_ack_wait, now=now),
        }
        self.views = view_mod.registry(self.registry["spark-view"],
                                       self.registry["hp-view"])
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
                         hold_off=self.question_open, watch=self.note_frame,
                         now=self._now, **kw)

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

    # ------------------------------------------------- which screen he grabbed
    def note_frame(self, row: dict) -> None:
        """One numbers-only frame row from the hand stage. Capture thread.

        Two ring-buffers and nothing else. The hand row is kept only when a
        hand was actually present, so a stretch of empty frames leaves the
        last real one in place and ``SOURCE_TTL_S`` -- not a stale
        coordinate -- decides whether it still counts. The face row is kept
        separately because yaw must be read from BEFORE the fist closes,
        and by then the reaching arm may well be across his face.

        This must stay cheap: it runs inside ``PreviewPipeline.grab()``.
        """
        try:
            at = float(row.get("at", 0.0))
        except (TypeError, ValueError):
            return
        if row.get("present") and float(row.get("palm_diag", 0.0)) > 0.0:
            self._last_hand = {"at": at, "cx": float(row.get("cx", 0.0)),
                               "palm_diag": float(row.get("palm_diag", 0.0)),
                               "frame_w": float(row.get("frame_w", 0.0))}
        if row.get("face_ok"):
            self._yaws.append(
                (at, screens_mod.yaw_t_from_deg(row.get("yaw_deg", 0.0))))

    def _yaw_before(self, at: float) -> Optional[float]:
        """The newest clean yaw sample in the ``YAW_WINDOW_S`` before the
        grab, or None for NO OPINION.

        THE NUMBER I WOULD LOOK AT FIRST is how often this answers None:
        if a clean pre-reach face row is missing on a quarter of grabs then
        yaw is a veto that abstains often and nothing more, whatever the
        clusters look like at rest. ``yaw_misses`` counts it.
        """
        for t, yaw in reversed(self._yaws):
            if t <= at and (at - t) <= YAW_WINDOW_S:
                return yaw
            if t < at - YAW_WINDOW_S:
                break
        return None

    def _read_source(self, at: float):
        """Which machine he grabbed at, or None for no opinion.

        None is what a shut preview produces, and that is the point: with
        no fresh hand row there is no source, with no source there is no
        screen cast, and the throw lands on the board exactly as it does
        today.
        """
        row = self._last_hand
        if row is None or (at - row["at"]) > SOURCE_TTL_S:
            return None
        if not float(row.get("frame_w", 0.0)) > 0.0:
            return None
        hand_u = screens_mod.hand_x_u(row["cx"], row["palm_diag"],
                                      row["frame_w"],
                                      mirrored=self.machine.mirrored)
        yaw = self._yaw_before(at)
        if yaw is None:
            self.yaw_misses += 1
        self.source_reads += 1
        return screens_mod.score(self.screens, hand_u, yaw)

    @property
    def screen_cast_on(self) -> bool:
        return screens_mod.enabled(self._get)

    def _view_decision(self, sector: str, source, by: str):
        """``None`` = not a screen cast, fall through to today's behaviour.

        Every gate that returns None here is a gate that leaves the board
        cast exactly as it is: the feature switched off, nothing learned, a
        map that would not arm, a grab too near the boundary, a preview
        that stopped feeding. The one thing that is NOT None-and-fall-
        through is a valid source thrown at a direction with nothing in it,
        which is a REFUSAL he hears.
        """
        if by != "gesture" or not self.screen_cast_on:
            return None
        if self.screens is None or not self.screens.armed:
            if not self._unarmed_said:
                self._unarmed_said = True
                line = screens_mod.unarmed_line(self.screens)
                if line:
                    self._speak(line)
            return None
        if source is None or not source.ok:
            return None
        dest = screens_mod.route(source.machine, sector)
        if not dest:
            return ("refuse", source.machine, source)
        return ("cast", dest, source)

    def _cast_view(self, decision, by: str) -> tuple:
        """``(line to say, status)``. Fires, or refuses. Never speaks
        itself, so the spoken path can hand the line back to the commander
        the way ``throw_by_voice`` already does."""
        kind, target, source = decision
        if kind == "refuse":
            self.refusals += 1
            self._earcon(DROP_TONE)
            self._chip_do("dropped", "")
            log.info("gesture: %s has nothing that way", target)
            return NOTHING_THAT_WAY_LINE, "refused"
        sink = self.views.get(target)
        subject = view_mod.view_subject(source.machine, at=self._wall())
        if sink is None or subject is None:
            return NO_VOICE_SINK_LINE.format(name=str(target)), "refused"
        lines: list = []
        status = self._cast(sink, subject, lines.append)
        self._record(status, sink, subject, by)
        tone = earcon_for(status)
        if tone:
            self._earcon(tone)
        self._chip_do("landed" if status in ("landed", "proposed") else
                      "held", getattr(sink, "label", ""))
        log.info("gesture: screen cast %s -> %s: %s (%s)", source.machine,
                 target, status, by)
        return " ".join(s for s in lines if s).strip(), status

    # ------------------------------------------------------ stop the cast
    @property
    def cast_live(self) -> str:
        """The name of the view sink that has a cast up, or ""."""
        return self.view_state.live

    def stop_cast(self) -> str:
        """The stop, and it must be as easy as the start.

        The harm here is a window appearing on a screen he is using, not
        bytes leaving, and no tone undoes that. Reachable three ways: a
        fling at the desk while a cast is up (DOWN is already the cancel
        sector, so this costs no new gesture and no new vocabulary), "stop
        the cast" by voice, and this method.
        """
        live = self.view_state.live
        for sink in self.views.values():
            if getattr(sink, "name", "") == live and sink.stop_cast():
                self._chip_do("dropped", "")
                log.info("gesture: cast stopped (%s)", live)
                return view_mod.STOPPED_LINE
        return view_mod.NOTHING_UP_LINE

    def cast_screen(self, target: str) -> tuple:
        """"cast the spark to HPCOMPUTER" -- the spoken way in.

        His ruling gives two ways in, the gesture and a sentence, and
        neither asks for confirmation. This one names the DESTINATION
        outright, so it needs no map at all and works with the camera off.
        """
        if not self.screen_cast_on:
            return NO_SCREEN_CAST_LINE, "refused"
        name = str(target or "").strip().lower()
        dest = name if name in screens_mod.MACHINES else \
            (screens_mod.HPCOMPUTER
             if cast_mod.sink_alias(name) == "hpcomputer" else "")
        if dest not in self.views:
            return NO_VOICE_SINK_LINE.format(name=str(target).strip()), "refused"
        source = [m for m in screens_mod.MACHINES if m != dest][0]
        return self._cast_view(("cast", dest, _Source(source)), "voice")

    # --------------------------------------------------------- the events
    def on_event(self, ev: CastEvent) -> None:
        """From the capture thread, inside the machine's lock. Cheap only."""
        self._events.append(ev.numbers_only())
        if ev.kind == "grab":
            # WHICH SCREEN HE GRABBED AT, read from the frame the fist
            # closed on and latched now, because by the time the throw
            # lands his hand is somewhere else entirely.
            self._source = self._read_source(float(ev.at))
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
            source, self._source = self._source, None
            self._worker(lambda: self._throw(sector, subject, by="gesture",
                                             source=source))
            return
        if ev.kind == "drop":
            self._source = None
            with self._lock:
                silent = self._silence_drops > 0
                if silent:
                    self._silence_drops -= 1
            if ev.toward == "down" and self.cast_live:
                # A FLING AT THE DESK STOPS THE CAST. Down is already the
                # cancel sector, so this costs no new gesture and no new
                # vocabulary -- and a stop has to be as easy as a start,
                # because what a start puts on his screen no tone takes off.
                self._earcon(DROP_TONE)
                self._worker(lambda: self._speak(self.stop_cast()))
                return
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
               by: str = "gesture", source=None) -> str:
        """Off the capture thread. Picks the sink for the side, casts,
        speaks what cast() said, plays the tone for the outcome.

        The SCREEN CAST is asked first and answers None for every reason
        there could be not to do one -- switched off, nothing learned, a
        map that would not arm, a grab too near the boundary, a preview
        that stopped feeding -- and every one of those Nones leaves the
        board cast below exactly as it was.
        """
        view = self._view_decision(sector, source, by)
        if view is not None:
            line, status = self._cast_view(view, by)
            if line:
                self._speak(line)
            return status
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
                "screens": self.screens_status(),
                "events": list(self._events)}

    def screens_status(self) -> dict:
        """The screen cast, in numbers only -- for the console readout and
        for scripts/screen_selfcheck.py.

        ``yaw_miss_pct`` is the number to look at first. If a clean
        pre-reach face row is missing on a quarter of grabs then head yaw
        is a veto that abstains often and nothing more, whatever the
        clusters look like at rest, and that is the single likeliest way
        this whole idea fails.
        """
        m = self.screens
        reads = max(int(self.source_reads), 1)
        out = {"on": bool(self.screen_cast_on),
               "armed": bool(m is not None and m.armed),
               "reason": m.reason if m is not None else "nothing learned",
               "screens": len(m.screens) if m is not None else 0,
               "machines": list(m.machines) if m is not None else [],
               "boundary_u": round(float(m.boundary_u), 4) if m else 0.0,
               "margin_sigma": round(float(m.margin_sigma), 3) if m else 0.0,
               "source_reads": int(self.source_reads),
               "yaw_misses": int(self.yaw_misses),
               "yaw_miss_pct": round(100.0 * self.yaw_misses / reads, 1),
               "refusals": int(self.refusals),
               "live": self.cast_live}
        out.update(self.relay.numbers_only())
        out["deck"] = self.view_state.numbers_only()
        out["learner"] = self.learner.numbers_only()
        return out

    def relearn(self, why: str = "") -> None:
        """Forget the map and say so once. The three tripwires all land
        here: the layout changed, the medians drifted, or grabs kept
        disagreeing with his head. Disarmed is not broken -- with no map
        every throw goes to the board, which is a real destination."""
        self.screens = None
        self._source = None
        self._unarmed_said = False
        self.learner = screens_mod.ScreenLearner()
        log.info("gesture: the screen map is disarmed (%s)", why or "asked")

    def note_layout(self, layout: str) -> bool:
        """The Windows helper's layout string, checked against the stored
        one. A change is an EXACT signal that he unplugged, added or moved
        a monitor, and the map disarms at once rather than routing on a
        room that no longer exists."""
        if not screens_mod.layout_changed(self.screens, layout):
            return False
        self.relearn("the monitor layout changed")
        self._speak(screens_mod.LAYOUT_CHANGED_LINE)
        return True


__all__ = [
    "DROPPED_LINE", "DROP_TONE", "GESTURE_TONE_COOLDOWN_S", "GRAB_TONE",
    "GestureCast", "NOTHING_REPEAT_S", "NOTHING_THAT_WAY_LINE",
    "NO_SCREEN_CAST_LINE", "NO_VOICE_SINK_LINE", "OPTION_SPEAK_GRAB",
    "RECENT_TTL_S", "SIDE_LINE", "SIDE_UNKNOWN_LINE", "SOURCE_TTL_S",
    "YAW_WINDOW_S",
]
