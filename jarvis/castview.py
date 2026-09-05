"""Casting a SCREEN VIEW between his two machines. No transport of its own.

Hunter, verbatim: "if i pull from the spark (right screen) and throw to the
middle, it will cast the spark to HPCOMPUTER. if i pull from theHPCOMPUTER
(middle screen) and throw to the right then it will cast HPCOMPTUER to
spark". The grab names the SOURCE, the throw names the DESTINATION;
``jarvis/screens.py`` turns a grab into a machine and ``ROUTES`` turns the
pair into a destination. THIS module is what happens after that decision,
and it is deliberately two thin sinks and one piece of shared state.

WHY THIS IS NOT IN jarvis/cast.py.
``tests/test_cast.py::test_the_module_cannot_open_a_window_or_a_lens`` greps
that file for ``launch_app``, ``xdg-open``, ``Toplevel`` and ``tkinter``,
and a viewer window is exactly what that test exists to keep out. It is
also the window churn that froze his desktop on 2026-08-26. So the viewer
lives here, and even here it is an INJECTED callable: this file spawns no
process and opens no connection of any kind, which
``test_the_module_holds_no_transport_of_its_own`` pins at source level. The
real launcher is built in jarvis/app.py; the suite passes a recorder.
NOTHING IN THE DESIGN OR TEST SESSION STARTED A RUSTDESK SESSION.

THE TWO DIRECTIONS ARE NOT SYMMETRICAL, and that is the whole shape here.

* HPCOMPUTER -> SPARK is the easy one and needs NO WINDOWS CHANGE AT ALL.
  "Cast HPCOMPUTER to the Spark" means the Spark runs the RustDesk VIEWER
  pointed at 192.168.50.114 -- outbound from the Spark, which the Windows
  firewall does not touch, on the 21118 path measured open in both
  directions. ``SparkViewSink``.
* SPARK -> HPCOMPUTER cannot be pushed: inbound to Windows is firewalled.
  His ruling, which is not re-opened here: a script running as HIM at logon
  POLLS the Spark and receives a VERB FROM A CLOSED SET -- show-spark,
  stop, none -- never a command string; the Windows side holds its own
  fixed command lines. That line is what keeps this from being remote
  execution into his live session. ``HpViewSink`` sets a verb on a
  ``CastRelay``; ``jarvis/webapp.py`` serves it on one gated route.

THE CLOSED SET IS ENFORCED TWICE. ``CastRelay.set_verb`` raises on anything
outside ``VERBS``, and ``poll`` re-checks before answering, so a bug
elsewhere in Jarvis that manages to park a command string on the relay
still cannot put one on the wire. Two checks for one rule is not
belt-and-braces here: the first is a programming error caught loudly, the
second is what actually goes out.

ONE CAST AT A TIME, AND A STOP AS EASY AS A START. He overruled a
confirmation step -- "no its ok with teh gesture or if i tell jarvis to cast
directly" -- and that ruling is right, because with two machines the two
valid throw directions are disjoint and a source misread always lands on a
refusal rather than the wrong desktop. But the harm that remains is a RIGHT
cast at a WRONG MOMENT: a window appearing on a screen he is using, which
no tone undoes. So ``ViewState`` allows exactly one live cast across both
directions, suppresses a second for ``CAST_SUPPRESS_S`` (longer than the
gesture's own 8-frame cooldown, so a double fling cannot open two viewers),
and every sink can stop what it started.

THE FINDING I OWE HIM, AND IT IS NOT SETTLED BY ANY TEST HERE. I do not
know whether the RustDesk viewer steals focus when it opens, whether it can
be launched minimised, or whether ``--connect`` honours a window-state
flag. I did not start a session and will not. That is the one part of this
lane that cannot be verified from numbers, and it is his to try once,
deliberately, when he is not mid-sentence in something.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from jarvis.cast import CastSubject, held_result, landed_result
from jarvis.logs import get_logger
from jarvis.screens import HPCOMPUTER, SPARK
from jarvis.screens import MACHINES as _MACHINES

log = get_logger("castview")

# THE CLOSED SET. Sorted, so the tuple itself reads as an enumeration
# rather than a priority. "none" is first because it is what a poll gets
# when there is nothing to do, which is almost always.
VERBS = ("none", "show-spark", "stop")
VERB_NONE = "none"
VERB_SHOW = "show-spark"
VERB_STOP = "stop"

# The two ends, as measured on 2026-09-04: RustDesk 1.4.9 both ends,
# LAN-direct on 21118, verified open from both directions.
HPCOMPUTER_HOST = "192.168.50.114"
SPARK_HOST = "192.168.50.109"

# How long after the Windows helper's last poll it still counts as running.
# It long-polls with a 25 s hold, so a minute is two missed rounds.
HELPER_ALIVE_S = 60.0
# A second cast is refused for this long after one starts or stops. The
# gesture's own cooldown is 8 frames, about 1.1 s at 7.5 fps, which is not
# enough to stop a double fling opening two viewers.
CAST_SUPPRESS_S = 3.0
# The helper's layout string is stored verbatim as a tripwire and never
# parsed for meaning here, so it needs a length bound and nothing else.
MAX_LAYOUT_CHARS = 200

VIEW_LABELS = {SPARK: "the Spark", HPCOMPUTER: "HPCOMPUTER"}
VIEW_SCREENS = {SPARK: "the Spark's screen",
                HPCOMPUTER: "HPCOMPUTER's screen"}

SHOWN_LINE = "{What} is on {target}, sir."
NO_HELPER_LINE = ("HPCOMPUTER isn't listening for me, sir — its startup "
                  "script isn't running. I've not cast anything.")
NO_VIEWER_LINE = "I've no viewer to open here, sir; I've not cast anything."
VIEWER_FAILED_LINE = "The viewer wouldn't open, sir."
BUSY_LINE = "There's a cast up already, sir"
BUSY_FULL_LINE = BUSY_LINE + "; say stop the cast and try again."
STOPPED_LINE = "Cast stopped, sir."
NOTHING_UP_LINE = "There's nothing cast, sir."

NO_HELPER_REASON = "the startup script isn't running on HPCOMPUTER"
NO_VIEWER_REASON = "there is no viewer wired on this box"


def view_subject(machine: str, *, at: float) -> Optional[CastSubject]:
    """The SOURCE screen, as the subject that travels.

    He grabs at a screen and throws it; what travels is the screen, not
    whatever document happened to be under his hand at the time. Naming the
    subject for the source is what makes the spoken line match what he
    actually did.
    """
    name = str(machine or "").strip().lower()
    if name not in _MACHINES:
        return None
    return CastSubject("screen", VIEW_SCREENS[name], at=float(at))


# ------------------------------------------------------------ the relay
class CastRelay:
    """What the Windows startup script polls, and the only thing it gets.

    A verb from ``VERBS`` and a sequence number, both small. In the other
    direction the helper reports two things it computed on Windows -- the
    INDEX of the monitor holding the mouse pointer, and the x-offset and
    width of each monitor. No cursor coordinate, no window handle, no
    title, no process name and no path crosses the wire in either
    direction. The layout is stored verbatim as an exact tripwire (a
    monitor added, removed or moved) and is never parsed for meaning here.

    ``seq`` is what makes the poll idempotent: without it a held
    "show-spark" would relaunch the viewer on every round.
    """

    def __init__(self, *, now: Callable[[], float] = time.monotonic,
                 alive_s: float = HELPER_ALIVE_S) -> None:
        self._now = now
        self._alive_s = float(alive_s)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._verb = VERB_NONE
        self._seq = 0
        self.mon = -1
        self.layout = ""
        self.seen_at: Optional[float] = None
        self.polls = 0

    # -- what Jarvis sets ---------------------------------------------
    @property
    def verb(self) -> str:
        with self._lock:
            return self._verb if self._verb in VERBS else VERB_NONE

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    def set_verb(self, verb) -> int:
        """Queue one verb. RAISES on anything outside the closed set --
        loudly, because a caller trying to send a command string is a
        programming error and not a runtime condition to be swallowed."""
        if not isinstance(verb, str) or verb not in VERBS:
            raise ValueError("a cast verb must be one of %s; got %r"
                             % (", ".join(VERBS), verb))
        with self._lock:
            self._verb = verb
            self._seq += 1
            seq = self._seq
        self._wake.set()
        return seq

    # -- what the helper reports --------------------------------------
    def note(self, *, mon=None, layout=None, at: Optional[float] = None) -> None:
        """One poll from the Windows side. Rubbish is DROPPED, not stored."""
        self.seen_at = float(at) if at is not None else float(self._now())
        self.polls += 1
        try:
            index = int(mon)
        except (TypeError, ValueError):
            index = -1
        self.mon = index if -1 <= index <= 15 else -1
        text = layout if isinstance(layout, str) else ""
        self.layout = text.strip()[:MAX_LAYOUT_CHARS]

    def alive(self, at: Optional[float] = None) -> bool:
        """Is the Windows helper actually polling? Never assumed."""
        seen = self.seen_at
        if seen is None:
            return False
        now = float(at) if at is not None else float(self._now())
        return (now - seen) <= self._alive_s

    # -- what goes out ------------------------------------------------
    def poll(self, seq: int, timeout_s: float = 0.0,
             wait: Optional[Callable[[float], bool]] = None) -> dict:
        """The helper's long poll: block up to ``timeout_s``, answer the
        instant a verb is set.

        LONG-POLLING IS NOT AN OPTIMISATION HERE, IT IS THE ONLY WAY IT
        FITS. ``webapp.RATE_MAX`` is 60 requests per 60 seconds, so a
        one-second poll would sit exactly on the limit and trip
        intermittently. A 25 s hold is about 2.4 requests a minute -- 4% of
        the budget -- and still lands a cast in well under a second.

        The reply is two fields. ``verb`` is re-checked against ``VERBS``
        here, not only where it was set: this is the last line before the
        wire, and it is what makes "no command string can ever be returned"
        a property of the code rather than of every caller.
        """
        try:
            since = int(seq)
        except (TypeError, ValueError):
            since = -1
        waited = float(timeout_s)
        if waited > 0.0 and self.seq == since:
            self._wake.clear()
            if self.seq == since:
                waiter = wait if callable(wait) else self._wake.wait
                waiter(waited)
        with self._lock:
            verb, current = self._verb, self._seq
        if verb not in VERBS:                       # a bug upstream, contained
            log.error("cast relay: refusing to send %r; it is not a verb",
                      verb)
            verb = VERB_NONE
        if current == since:
            verb = VERB_NONE
        return {"seq": int(current), "verb": str(verb)}

    def numbers_only(self) -> dict:
        return {"verb": self.verb, "seq": self.seq, "mon": int(self.mon),
                "layout": str(self.layout), "polls": int(self.polls),
                "alive": bool(self.alive())}


# ------------------------------------------------------- one at a time
class ViewState:
    """Exactly one live cast across BOTH directions, and a suppression.

    Shared by both sinks on purpose: "a cast cannot fire while one is up"
    has to mean any cast, not one per direction, or a fling each way opens
    two windows on two machines.
    """

    def __init__(self, *, now: Callable[[], float] = time.monotonic,
                 suppress_s: float = CAST_SUPPRESS_S) -> None:
        self._now = now
        self.suppress_s = float(suppress_s)
        self._lock = threading.Lock()
        self.live = ""
        self.at = 0.0
        self._last_end = -1e9
        self.starts = 0
        self.refusals = 0

    def take(self, name: str) -> tuple:
        """``(ok, why)``. Refusing is the normal outcome, not an error."""
        now = float(self._now())
        with self._lock:
            if self.live:
                self.refusals += 1
                return False, "a cast is already up (%s)" % self.live
            if now - self._last_end < self.suppress_s:
                self.refusals += 1
                return False, "a cast ended less than %.0fs ago" % self.suppress_s
            self.live = str(name)
            self.at = now
            self.starts += 1
            return True, ""

    def release(self) -> str:
        """Give the deck back. Returns what was up, or ""."""
        with self._lock:
            was, self.live = self.live, ""
            self.at = 0.0
            self._last_end = float(self._now())
        return was

    def numbers_only(self) -> dict:
        return {"live": str(self.live), "at": round(float(self.at), 3),
                "starts": int(self.starts), "refusals": int(self.refusals),
                "suppress_s": float(self.suppress_s)}


class _ViewSink:
    """What the two view sinks share: the deck, and holding honestly.

    ``needs_readback`` is False -- HIS RULING, taken and not re-opened: "no
    its ok with teh gesture or if i tell jarvis to cast directly". The
    safety this rests on is the disjoint-direction property in
    ``screens.ROUTES``, not a prompt. ``needs_identity`` is True so that
    ``cast()`` demands a positive name the day the read-back path is ever
    turned on; today it is inert, because ``camera.identity`` ships off and
    an empty identity is NO OPINION. What is NOT inert today is the veto
    already in ``cast()``: a DIFFERENT name in frame refuses any sink,
    which is exactly the "it shows a desktop to whoever is in the room"
    case.

    A held view NEVER falls back to the board. A row on the board is not a
    smaller version of a desktop on a monitor; it is a different thing, and
    quietly substituting one for the other would teach him the gesture had
    worked.
    """

    name = "view"
    label = "there"
    reversible = False
    needs_identity = True
    wants_bytes = False

    def __init__(self, *, state: Optional[ViewState] = None,
                 now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self.state = state if state is not None else ViewState(now=now)

    def needs_readback(self, subject: Optional[CastSubject] = None) -> bool:
        return False

    def available(self) -> tuple:
        return True, ""

    def _busy(self) -> Optional[object]:
        ok, why = self.state.take(self.name)
        if ok:
            return None
        return held_result(BUSY_FULL_LINE, sink=self.name, detail=why)

    def stop_cast(self) -> bool:
        """Stop what THIS sink started. False when nothing of ours is up."""
        if self.state.live != self.name:
            return False
        self._stop()
        self.state.release()
        return True

    def _stop(self) -> None:                        # pragma: no cover - base
        pass


class SparkViewSink(_ViewSink):
    """HPCOMPUTER's desktop, on the Spark's screen. The easy direction.

    ``launch(host)`` and ``stop()`` are INJECTED and default to None -- the
    same discipline ``HpcomputerSink(transport=None)`` already holds, and
    for the same stated reason: a transport that cannot be run against the
    real host is a guess dressed as a feature. With nothing wired this sink
    HOLDS and says so, which is a visible, honest, recoverable state.

    It needs no Windows helper: the connection is outbound from this box.
    What it DOES need, and only he can do, is RustDesk's "Enable direct IP
    access" and an unattended password on HPCOMPUTER -- without both, a
    direct-IP connect waits for someone to accept it on the Windows side
    and the cast silently does nothing.
    """

    name = "spark-view"
    label = VIEW_LABELS[SPARK]

    def __init__(self, *, launch: Optional[Callable[[str], object]] = None,
                 stop: Optional[Callable[[], object]] = None,
                 host: str = HPCOMPUTER_HOST,
                 state: Optional[ViewState] = None,
                 now: Callable[[], float] = time.monotonic) -> None:
        super().__init__(state=state, now=now)
        self._launch = launch if callable(launch) else None
        self._stop_fn = stop if callable(stop) else None
        self.host = str(host)

    def available(self) -> tuple:
        if self._launch is None:
            return False, NO_VIEWER_REASON
        return True, ""

    def deliver(self, subject: CastSubject):
        if self._launch is None:
            return held_result(NO_VIEWER_LINE, sink=self.name,
                               detail=NO_VIEWER_REASON)
        busy = self._busy()
        if busy is not None:
            return busy
        try:
            self._launch(self.host)
        except Exception as exc:                    # noqa: BLE001 - the seam
            log.warning("castview: the viewer would not open: %s", exc)
            self.state.release()
            return held_result(VIEWER_FAILED_LINE, sink=self.name,
                               detail=str(exc))
        return landed_result(
            SHOWN_LINE.format(What=_cap(VIEW_SCREENS[HPCOMPUTER]),
                              target=self.label), sink=self.name)

    def _stop(self) -> None:
        if self._stop_fn is None:
            return
        try:
            self._stop_fn()
        except Exception:                           # noqa: BLE001 - the seam
            log.debug("castview: stopping the viewer raised", exc_info=True)


class HpViewSink(_ViewSink):
    """The Spark's desktop, on HPCOMPUTER. Through a verb, never a command.

    It HOLDS whenever the Windows helper is not actually polling, and it
    says why. That is the honest degradation: nothing is queued for a
    machine that is not listening, so a startup script installed an hour
    later does not suddenly execute an hour-old intention.
    """

    name = "hp-view"
    label = VIEW_LABELS[HPCOMPUTER]

    def __init__(self, *, relay: Optional[CastRelay] = None,
                 state: Optional[ViewState] = None,
                 now: Callable[[], float] = time.monotonic) -> None:
        super().__init__(state=state, now=now)
        self.relay = relay

    def available(self) -> tuple:
        if self.relay is None or not self.relay.alive():
            return False, NO_HELPER_REASON
        return True, ""

    def deliver(self, subject: CastSubject):
        ok, why = self.available()
        if not ok:
            return held_result(NO_HELPER_LINE, sink=self.name, detail=why)
        busy = self._busy()
        if busy is not None:
            return busy
        try:
            self.relay.set_verb(VERB_SHOW)
        except Exception as exc:                    # noqa: BLE001 - the seam
            log.warning("castview: the relay refused the verb: %s", exc)
            self.state.release()
            return held_result(VIEWER_FAILED_LINE, sink=self.name,
                               detail=str(exc))
        return landed_result(
            SHOWN_LINE.format(What=_cap(VIEW_SCREENS[SPARK]),
                              target=self.label), sink=self.name)

    def _stop(self) -> None:
        if self.relay is None:
            return
        try:
            self.relay.set_verb(VERB_STOP)
        except Exception:                           # noqa: BLE001 - the seam
            log.debug("castview: stopping the cast raised", exc_info=True)


def _cap(text: str) -> str:
    text = str(text or "")
    return text[:1].upper() + text[1:] if text else text


def registry(spark: SparkViewSink, hp: HpViewSink) -> dict:
    """DESTINATION machine -> the sink that shows a screen ON it.

    Exactly two entries, and the keys are ``screens.MACHINES``. If this
    ever grows a third the disjoint-direction safety property is void --
    ``screens.build`` refuses to arm a third machine for that reason.
    """
    return {SPARK: spark, HPCOMPUTER: hp}


__all__ = [
    "BUSY_FULL_LINE", "BUSY_LINE", "CAST_SUPPRESS_S", "CastRelay",
    "HELPER_ALIVE_S", "HPCOMPUTER_HOST", "HpViewSink", "MAX_LAYOUT_CHARS",
    "NOTHING_UP_LINE", "NO_HELPER_LINE", "NO_HELPER_REASON",
    "NO_VIEWER_LINE", "NO_VIEWER_REASON", "SHOWN_LINE", "SPARK_HOST",
    "STOPPED_LINE", "SparkViewSink", "VERBS", "VERB_NONE", "VERB_SHOW",
    "VERB_STOP", "VIEWER_FAILED_LINE", "VIEW_LABELS", "VIEW_SCREENS",
    "ViewState", "registry", "view_subject",
]
