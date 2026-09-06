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

NEITHER DIRECTION MAY SAY IT LANDED UNTIL IT HAS. Both sinks used to, and
both were wrong in the same way -- they reported the ATTEMPT. ``HpViewSink``
said "The Spark's screen is on HPCOMPUTER, sir" the instant the verb was
parked on the relay, though ``alive()`` only means the helper polled some
time in the last minute and it may last have polled 59 seconds ago; nothing
ever confirmed receipt. ``SparkViewSink`` said the mirror of it whenever
``launch()`` failed to raise, and the real launcher is a ``Popen``, so a
viewer that started and died a moment later reported success. cast.py's
oldest promise is that ``CastResult.landed`` and ``.held`` are separate
booleans, never inferred from each other; reporting an attempt as a landing
breaks the spirit of it, and his ruling that there is NO CONFIRMATION STEP
makes the spoken line the only thing he has to go on.

So each direction now has one piece of evidence behind the word "landed",
and it is the cheapest true one available:

* HPCOMPUTER: the helper's NEXT POLL, quoting the sequence it ACTED ON --
  and round 3 changed the Windows script so that this is true. It used to
  commit ``$seq`` before it attempted ``Start-Process``, under
  ``$ErrorActionPreference = 'SilentlyContinue'``, so RustDesk not being
  at the fixed path produced a receipt for a launch that never happened
  (MEASURED: 1 launch attempt, 0 windows up, and Jarvis said it had
  landed). Now the receipt moves only after a launch the script has
  watched survive, and a launch that fails sends a CODE from
  ``HELPER_FAILS`` instead, which Jarvis repeats.
  No receipt inside ``HELPER_ACK_S`` and the verb is taken back off
  the relay as a ``stop`` -- so a helper that was merely slow puts the
  window away rather than leaving one up that Jarvis has already said it
  did not open -- the deck is released, and the cast HOLDS out loud.
  ...AND A RECEIPT IS STILL NOT A DESKTOP, WHICH IS THE ROUND-4 FINDING
  AND THE ONE WITH LIVE EXPOSURE. The gesture ships ``enabled: False``;
  the spoken cast does not, so this is the direction he can hit today. A
  RustDesk viewer sitting on an accept-or-password prompt survives the
  script's own 700 ms settle perfectly well, and this sink said "The
  Spark's screen is on HPCOMPUTER, sir." over it with nothing on any
  monitor -- and nothing revisited it, so the deck stayed held and the
  next genuine cast was refused as busy (MEASURED, round-3 attack). So
  this direction takes the same tri-state ``connected`` probe, the same
  second look and the same spoken retraction as the Spark's. Its
  evidence is the MIRROR of the Spark's, because the viewer is on his
  machine and there is no pid here: jarvis/app.py reads an ESTABLISHED
  socket INBOUND from HPCOMPUTER on ``RUSTDESK_PORTS`` and bytes leaving
  over it.
* The Spark: the viewer is still running ``VIEWER_SETTLE_S`` after launch
  AND it is actually receiving something. Round 2 stopped at the first
  half, and the module's own docstring had already predicted what that
  costs: without direct-IP access and an unattended password on
  HPCOMPUTER a direct-IP connect waits for someone to accept it, and a
  viewer parked on that prompt is a LIVE PROCESS. ``Popen.poll() is None``
  said yes and Jarvis said "HPCOMPUTER's screen is on the Spark, sir."
  with nothing cast. So there is a second, injected probe -- ``connected``
  -- and it is a TRI-STATE: True, False, or None for cannot tell, which
  becomes an honest "I can't say it landed" rather than a landing. With no
  such probe wired EITHER SINK reports itself UNAVAILABLE, on exactly the
  rule the launcher is already held to -- and round 4 moved that rule, the
  probe, the settle, the second look and the retraction onto the shared
  ``_ViewSink`` base, because two copies of it is how round 3 came to ship
  one honest direction and one that lied. jarvis/app.py implements it out
  of /proc: an ESTABLISHED socket the viewer itself owns to that host,
  and bytes arriving over it. It opens no window and no capture device.
  And it is ASKED AGAIN at ``VIEWER_CONFIRM_S``, because a viewer alive
  at 0.7 s and gone at 3 s was claimed landed and never revisited -- the
  deck stayed held by a cast that was not there and the next genuine cast
  was refused as busy. The second look gives the deck back and RETRACTS
  the sentence out loud.

WHAT THE CONNECTION PROBE STILL CANNOT SAY, and he should have it plainly:
it proves a live connection carrying a stream. It does not prove a window
is visible, or on the right monitor, or in front of what he was reading.
That last step has no local evidence and is his to look at once.

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
# How long a cast waits for the helper to ACKNOWLEDGE the verb before it
# gives up and says so. Being inside HELPER_ALIVE_S is not receipt: the
# helper may last have polled 59 s ago, and a verb parked for a machine
# that is not at the door is not a screen on a monitor. The helper is
# parked in a 25 s long poll almost all of the time, so the round trip is
# its wake plus one Start-Process; 2 s is generous for a LAN and short
# enough to say out loud. GUESSED -- nothing here has been run against
# the real helper -- AND SINCE ROUND 5 THE STOP BLOCKS ON IT TOO: a helper
# that takes longer than this to wake and kill the viewer turns a stop
# that WORKED into "HPCOMPUTER didn't come back for that, sir", with the
# window gone and the deck held until the second look notices. The
# measurement it wants is the real wake-to-receipt gap: the clock between
# ``set_verb`` parking a verb and the poll that quotes its sequence back,
# over a handful of real casts, with the value set from the slowest of
# those plus headroom. Nothing here has that number yet.
HELPER_ACK_S = 2.0
# How long after launching the Spark's own viewer before asking whether it
# is still there. A viewer that cannot start (no display, a bad binary, a
# refused direct-IP connect) dies almost at once; one that survives this
# window has at least got up. GUESSED, and it is a floor not a guarantee:
# a viewer that dies at three seconds was reported as landed at one.
VIEWER_SETTLE_S = 0.7
# ...and then again, later, because a viewer that dies at three seconds was
# reported as landed at one and NOTHING ever revisited it: the deck stayed
# held by a cast that was not there and the next genuine cast was refused
# as busy (MEASURED, round-3 attack). This is when the second look happens.
# GUESSED, and it is a compromise: long enough that a viewer which is
# going to fall over has done so, short enough that he is still in the
# same moment when Jarvis takes the sentence back.
VIEWER_CONFIRM_S = 4.0
# The throughput a viewer must be pulling to count as CONNECTED rather than
# merely running. A RustDesk viewer showing a desktop pulls hundreds of
# kilobytes a second; one sitting on an accept-or-password prompt pulls a
# keepalive and nothing else. 40 kB/s is an order of magnitude above the
# prompt and an order below the stream. GUESSED from those two orders, not
# measured against his HPCOMPUTER -- the probe that reads it is in
# jarvis/app.py and it reads /proc, never the picture.
VIEWER_STREAM_BPS = 40000.0
# How many times the ack wait may re-enter before giving up, whatever the
# clock says. A hard bound, because an unbounded wait loop on this box has
# already cost a session.
ACK_WAIT_ROUNDS = 8
# What the Windows helper is allowed to say went wrong. A CLOSED SET, for
# exactly the reason the verbs are one: a free-text Windows error would put
# a path, a window title or a process name on the wire, and this lane has
# always sent nothing but an index and a layout.
HELPER_FAILS = ("no-viewer", "launch-failed", "viewer-exited",
                "stop-failed")
# ROUND 4 ADDED ``stop-failed``. ``Stop-Cast`` swallowed a failed
# ``Stop-Process`` in an empty catch and returned 0 unconditionally, so a
# viewer the script could NOT kill was orphaned on his middle monitor
# while the receipt told Jarvis the stop had happened -- and a following
# show-spark would stack a second viewer on top of it. A stop that did
# not happen is now a code like any other.
# THE PORTS THE SPARK IS VIEWED ON, and the reason this constant exists at
# all. The Spark -> HPCOMPUTER cast has no viewer process on this box: the
# viewer runs on HIS Windows machine and the Spark is the end being VIEWED.
# So the local evidence is an INBOUND connection from HPCOMPUTER -- and
# "is there a connection to HPCOMPUTER" is TRUE ALMOST ALWAYS, because the
# Windows helper's own long poll is one, held open 25 s at a time about
# 2.4 times a minute. A probe that is true whatever happens is a rubber
# stamp on a sentence, so the probe is scoped to these ports and the poll
# port (jarvis/webapp.py's 8765) is deliberately not among them. 21118 is
# the direct-IP path measured open in both directions on 2026-09-04; the
# others are RustDesk's own defaults and are GUESSED to be worth watching.
RUSTDESK_PORTS = (21115, 21116, 21117, 21118, 21119)
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
# ...and this is what an unavailable sink says when what is MISSING is the
# EVIDENCE rather than the machinery. Round 3 spoke NO_VIEWER_LINE and
# NO_HELPER_LINE for both cases alike, so a sink with no connection probe
# told him HPCOMPUTER's startup script was not running -- which was a
# second untrue sentence sitting behind the first. The reason a cast did
# not happen has to be the real one.
NO_EVIDENCE_LINE = ("I've no way to check that would actually land, sir, so "
                    "I'll not say it did. Nothing's cast.")
VIEWER_FAILED_LINE = "The viewer wouldn't open, sir."
VIEWER_DIED_LINE = ("The viewer opened and closed again, sir. Nothing's "
                    "cast.")
VIEWER_NOT_CONNECTED_LINE = ("The viewer opened but never connected, sir — "
                             "HPCOMPUTER didn't let it in. Nothing's cast.")
CANNOT_TELL_LINE = ("I can't tell whether that viewer connected, sir, so I "
                    "won't say it did. Nothing's cast.")
CAST_GONE_LINE = ("The cast has dropped, sir. What I told you a moment ago "
                  "is no longer true.")
LAUNCH_FAILED_LINE = "HPCOMPUTER couldn't open the viewer, sir"
HELPER_FAIL_LINES = {
    "no-viewer": LAUNCH_FAILED_LINE + " — RustDesk isn't where it expects it.",
    "launch-failed": LAUNCH_FAILED_LINE + " — it wouldn't start.",
    "viewer-exited": LAUNCH_FAILED_LINE + " — it closed straight away.",
    "stop-failed": ("There's a viewer still up on HPCOMPUTER that I "
                    "couldn't close, sir. I've not opened another."),
}
# What the SPARK -> HPCOMPUTER direction says when the thing it started is
# not there any more. Two different facts, two different sentences: the
# machine stopped answering, or the viewer it started never carried a
# picture.
HELPER_GONE_LINE = ("HPCOMPUTER stopped answering me, sir. I've not left "
                    "anything cast.")
HP_NOT_CONNECTED_LINE = ("HPCOMPUTER opened the viewer but nothing's coming "
                         "through, sir — it never connected back to the "
                         "Spark. Nothing's cast.")
NO_RECEIPT_LINE = ("HPCOMPUTER didn't come back for it, sir. I've taken it "
                   "back rather than leave it queued — nothing's cast.")
BUSY_LINE = "There's a cast up already, sir"
BUSY_FULL_LINE = BUSY_LINE + "; say stop the cast and try again."
STOPPED_LINE = "Cast stopped, sir."
NOTHING_UP_LINE = "There's nothing cast, sir."
# ROUND 5. WHAT A STOP SAYS WHEN IT DID NOT HAPPEN, which until now was
# "Cast stopped, sir." in every case. ``HELPER_FAIL_LINES["stop-failed"]``
# already existed for the Spark -> HPCOMPUTER direction and was reachable
# only through a LATER show-spark; these two are the cases it does not
# cover -- the helper that never came back for the stop at all, and the
# local viewer on THIS box that would not die.
NO_STOP_RECEIPT_LINE = ("HPCOMPUTER didn't come back for that, sir, so I "
                        "can't say the cast has stopped. Say it again and "
                        "I'll ask once more.")
STOP_FAILED_LINE = ("I couldn't close that viewer, sir — it's still up. "
                    "I've left the cast marked as live.")
# ...and what a landing says when the thing that would CATCH it dropping
# could not be started. Failing to arm the watchdog holds; it never lands.
NO_WATCHDOG_LINE = ("I can't set the check that would tell me if that cast "
                    "drops, sir, so I'll not say it landed. Nothing's cast.")

NO_HELPER_REASON = "the startup script isn't running on HPCOMPUTER"
NO_VIEWER_REASON = "there is no viewer wired on this box"
NO_PROBE_REASON = "there is no way to tell whether the viewer is up"
NO_CONNECTION_PROBE_REASON = ("there is no way to tell whether the viewer "
                              "actually connected")
NOT_CONNECTED_REASON = "the viewer was running but nothing was flowing"
CANNOT_TELL_REASON = "the connection probe had no opinion"
CAST_GONE_REASON = "the viewer was gone when it was checked again"
NO_RECEIPT_REASON = "HPCOMPUTER never acknowledged the verb"
VIEWER_DIED_REASON = "the viewer was gone a moment after it started"
NO_STOP_RECEIPT_REASON = "HPCOMPUTER never acknowledged the stop"
STOP_FAILED_REASON = "the viewer was still there after it was told to close"
NO_WATCHDOG_REASON = "the second look could not be armed"
NOTHING_OF_OURS_REASON = "nothing of ours is up"


class StopReport:
    """What a stop actually did. TRUTHY ONLY WHEN IT HAPPENED.

    Round 4 left ``stop_cast`` returning a bare bool that was True the
    moment the verb was PARKED, and the courier could not tell "that
    wasn't mine" from "I asked and it did not work" -- so a stop the
    helper reported as failed fell through to "There's nothing cast,
    sir." while the window was still on his middle monitor.

    ``ok`` is the only thing ``bool()`` reads, so every existing
    ``assert sink.stop_cast()`` still means what it meant. ``mine`` is
    what the courier needs: it says the sink OWNED the cast, which
    separates the two silences.
    """

    __slots__ = ("ok", "mine", "line", "detail")

    def __init__(self, ok: bool, mine: bool, line: str = "",
                 detail: str = "") -> None:
        self.ok = bool(ok)
        self.mine = bool(mine)
        self.line = str(line)
        self.detail = str(detail)

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:                       # pragma: no cover
        return ("StopReport(ok=%r, mine=%r, line=%r, detail=%r)"
                % (self.ok, self.mine, self.line, self.detail))


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
                 alive_s: float = HELPER_ALIVE_S,
                 on_layout: Optional[Callable[[str], object]] = None) -> None:
        self._now = now
        self._alive_s = float(alive_s)
        # THE TRIPWIRE. A layout string that differs from the last one is an
        # exact signal that he unplugged, added or moved a monitor, and it
        # arrives from the machine that owns the layout -- which is the one
        # thing I could not enumerate over SSH. The courier disarms its map
        # on it rather than routing on a room that no longer exists.
        self._on_layout = on_layout
        self._lock = threading.Lock()
        self._wake = threading.Event()
        # RECEIPT, not hope. The helper long-polls carrying the sequence it
        # last ACTED ON, so a poll quoting our current sequence is the
        # Windows side saying "I have this one" -- the only evidence that
        # crosses back, and the only thing that may turn a cast into a
        # landing. It is deliberately not a new field on the wire: the
        # helper already sends it and its script is not being changed.
        self._ack_wake = threading.Event()
        self.acked = -1
        # What the helper said went WRONG, from HELPER_FAILS, and which
        # sequence it failed on. The shipped script committed its receipt
        # before it even attempted the launch and swallowed the error, so
        # a Jarvis that read the receipt said a cast had landed on a
        # machine where RustDesk was not installed (MEASURED: 1 launch
        # attempt, 0 windows up, "The Spark's screen is on HPCOMPUTER,
        # sir."). Now a failed launch never becomes a receipt AND says so.
        self.fail = ""
        self.failseq = -1
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
    def note(self, *, mon=None, layout=None, at: Optional[float] = None,
             fail=None, failseq=None) -> None:
        """One poll from the Windows side. Rubbish is DROPPED, not stored.

        ``fail`` is enumerated against ``HELPER_FAILS`` here for the same
        reason ``poll`` re-checks the verb: this is the line the wire
        crosses, and a free-text Windows error carries paths and window
        titles that this lane has never sent.
        """
        self.seen_at = float(at) if at is not None else float(self._now())
        self.polls += 1
        code = fail if isinstance(fail, str) else ""
        self.fail = code if code in HELPER_FAILS else ""
        try:
            self.failseq = int(failseq)
        except (TypeError, ValueError):
            self.failseq = -1
        if self.fail:
            log.warning("cast relay: HPCOMPUTER reports %r for seq %d",
                        self.fail, self.failseq)
            # A failure ends the wait as surely as a receipt does, and it
            # ends it with the truth instead of a timeout.
            self._ack_wake.set()
        try:
            index = int(mon)
        except (TypeError, ValueError):
            index = -1
        self.mon = index if -1 <= index <= 15 else -1
        text = layout if isinstance(layout, str) else ""
        was, self.layout = self.layout, text.strip()[:MAX_LAYOUT_CHARS]
        if self.layout and self.layout != was and callable(self._on_layout):
            try:
                self._on_layout(self.layout)
            except Exception:                    # noqa: BLE001 - the courier
                log.debug("cast relay: the layout watcher raised",
                          exc_info=True)

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
        # Take the receipt FIRST, before any wait: the helper is telling us
        # what it holds as it arrives, and it holds that whether or not
        # there is anything new to send it.
        with self._lock:
            if 0 <= since <= self._seq and since > self.acked:
                self.acked = since
                fresh = True
            else:
                fresh = False
        if fresh:
            self._ack_wake.set()
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

    def failed_for(self, seq) -> str:
        """The helper's reason for not acting on ``seq``, or "". Never a
        guess: only a code the helper actually sent for that sequence."""
        try:
            want = int(seq)
        except (TypeError, ValueError):
            return ""
        with self._lock:
            return self.fail if self.failseq == want else ""

    def acked_through(self, seq) -> bool:
        """Has the helper confirmed it holds ``seq``? Never assumed."""
        try:
            want = int(seq)
        except (TypeError, ValueError):
            return False
        with self._lock:
            return self.acked >= want

    def await_ack(self, seq, timeout_s: float = HELPER_ACK_S,
                  wait: Optional[Callable[[float], object]] = None) -> bool:
        """Block up to ``timeout_s`` for the helper to acknowledge ``seq``.

        The wait is INJECTED for the same reason every other outward edge
        in this file is: the suite must never sleep, and a test needs to
        drive the helper's side of the round trip by hand. Returns whether
        the receipt actually arrived -- never whether it probably did.
        """
        deadline = float(self._now()) + float(timeout_s)
        waiter = wait if callable(wait) else self._ack_wake.wait
        rounds = 0
        while True:
            if self.acked_through(seq) or self.failed_for(seq):
                return self.acked_through(seq)
            left = deadline - float(self._now())
            if left <= 0.0 or rounds >= ACK_WAIT_ROUNDS:
                return self.acked_through(seq)
            # RE-ARM AND RE-CHECK EVERY ROUND. Round 2 waited ONCE on a
            # shared event, so the receipt for the PREVIOUS verb woke the
            # wait for this one and spent it: waiting on seq 2 and woken
            # by the receipt for seq 1, it reported NO RECEIPT and held --
            # it called a genuine cast failed (MEASURED). The clock and
            # the round cap are what bound the loop; a waiter that
            # neither wakes nor advances the clock ends it on the first
            # pass.
            self._ack_wake.clear()
            if self.acked_through(seq) or self.failed_for(seq):
                return self.acked_through(seq)
            rounds += 1
            before = float(self._now())
            try:
                woke = waiter(left)
            except Exception:               # noqa: BLE001 - the injected seam
                log.debug("cast relay: the ack wait raised", exc_info=True)
                return self.acked_through(seq)
            if not woke and float(self._now()) <= before:
                # An injected wait that reports nothing and moves no clock
                # has, by contract, consumed the whole budget.
                return self.acked_through(seq)

    def numbers_only(self) -> dict:
        return {"verb": self.verb, "seq": self.seq, "mon": int(self.mon),
                "layout": str(self.layout), "polls": int(self.polls),
                "acked": int(self.acked), "alive": bool(self.alive()),
                "fail": str(self.fail), "failseq": int(self.failseq)}


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

    ROUND 4 PUT THE HONESTY HERE RATHER THAN IN ONE SINK. Round 3 gave
    ``SparkViewSink`` a tri-state connection probe, a second look and a
    spoken retraction, and left ``HpViewSink`` claiming a landing off a
    receipt -- and the receipt only ever meant "the Windows script acted",
    never "a desktop arrived". Two copies of "probe, give the deck back,
    retract" is two places for the directions to drift apart, and they had
    already drifted once. So the probe, the tri-state, the settle, the
    second look, the retraction and the give-back are all on this base and
    neither subclass owns a copy. What legitimately differs is exactly two
    things, and each is one overridden hook: what STARTS the cast, and what
    "still there" means -- a process this app owns, versus a machine that
    polls us.
    """

    name = "view"
    label = "there"
    reversible = False
    needs_identity = True
    wants_bytes = False
    # What this direction says when the thing it started is gone. One line
    # each, because "the viewer died" and "HPCOMPUTER stopped answering"
    # are different facts and he should be told which one happened.
    gone_line = VIEWER_DIED_LINE
    gone_reason = VIEWER_DIED_REASON
    not_connected_line = VIEWER_NOT_CONNECTED_LINE

    def __init__(self, *, state: Optional[ViewState] = None,
                 now: Callable[[], float] = time.monotonic,
                 connected: Optional[Callable[[], Optional[bool]]] = None,
                 settle: Optional[Callable[[float], object]] = None,
                 later: Optional[Callable[[float, Callable], object]] = None,
                 retract: Optional[Callable[[str], object]] = None,
                 arm: Optional[Callable[[], object]] = None,
                 settle_s: float = VIEWER_SETTLE_S,
                 confirm_s: float = VIEWER_CONFIRM_S) -> None:
        self._now = now
        self.state = state if state is not None else ViewState(now=now)
        # THE TRI-STATE: True, False, or None for CANNOT TELL. It is
        # injected because this module owns no transport and reads no
        # /proc; jarvis/app.py implements one per direction.
        self._connected = connected if callable(connected) else None
        self._settle = settle if callable(settle) else time.sleep
        # How the SECOND look is scheduled. Injected so the suite neither
        # sleeps nor spawns; the default is a daemon timer.
        self._later = later if callable(later) else _default_later
        self._retract = retract if callable(retract) else None
        # HOW THE PROBE IS TIED TO THIS CAST. The Spark -> HPCOMPUTER probe
        # answers "is there an established RustDesk socket from HPCOMPUTER",
        # which is true of a session HE opened himself before Jarvis was
        # asked for anything. ``arm`` is called BEFORE the cast starts, so
        # the probe can answer about what appeared AFTER it. Optional: a
        # sink with no arm behaves exactly as round 4 did.
        self._arm = arm if callable(arm) else None
        self.settle_s = float(settle_s)
        self.confirm_s = float(confirm_s)
        self._watch = None
        # THE GENERATION COUNTER, and it is what makes a stop he asked for
        # unable to be retracted. ``_confirm`` reads the deck, then spends
        # VIEWER_STREAM sampling time inside the real probe (0.35 s), and a
        # "stop the cast" inside that window used to still get "The cast has
        # dropped, sir." -- and to re-stamp the suppression window with it.
        # Every end of a cast bumps this; the timer carries the value it was
        # armed with and does nothing at all if it has moved.
        self._epoch = 0

    def needs_readback(self, subject: Optional[CastSubject] = None) -> bool:
        return False

    def available(self) -> tuple:
        """THE RULE BOTH DIRECTIONS ARE NOW HELD TO. With no honest local
        evidence that the cast ARRIVED, this sink cannot know whether it
        landed -- so it reports itself unavailable rather than saying a
        sentence it cannot support. Subclasses check their own wiring
        first and then call this."""
        if self._connected is None:
            return False, NO_CONNECTION_PROBE_REASON
        return True, ""

    # What this direction says when it is not wired up at all. The
    # EVIDENCE case is separate and shared, because "I have no viewer" and
    # "I have no way to tell" are different sentences and only one of them
    # is true at a time.
    unavailable_line = NO_VIEWER_LINE

    def _unavailable(self, why: str):
        """Hold, and say the true reason. A missing PROBE is not a missing
        helper and not a missing viewer."""
        line = (NO_EVIDENCE_LINE
                if why in (NO_PROBE_REASON, NO_CONNECTION_PROBE_REASON)
                else self.unavailable_line)
        return held_result(line, sink=self.name, detail=why)

    def _busy(self) -> Optional[object]:
        ok, why = self.state.take(self.name)
        if ok:
            return None
        return held_result(BUSY_FULL_LINE, sink=self.name, detail=why)

    def stop_cast(self) -> StopReport:
        """Stop what THIS sink started, AND WAIT FOR THE RECEIPT.

        ROUND 5, AND IT IS THE ONE HE CAN HIT TODAY. This method used to
        call ``_stop()``, release the deck and return True unconditionally,
        so Jarvis said "Cast stopped, sir." at the instant the verb was
        PARKED -- measured at that moment: viewers up on HPCOMPUTER 1, kill
        attempts 0. The helper then failed to kill it, reported
        ``stop-failed``, and NOTHING SPOKE IT: the deck was already
        released and ``failed_for(seq)`` was read by nobody. Asking again
        did not help, because ``state.live`` was already "" and the second
        "stop the cast" answered "There's nothing cast, sir." with the
        window still on his middle monitor.

        This is the same PARKED-IS-NOT-RECEIVED rule round 4 applied to the
        show verb, and it uses the same three pieces: ``await_ack``,
        ``failed_for`` and ``HELPER_FAIL_LINES``. The other direction has
        the mirror of it -- after telling the local viewer to die, ask the
        same ``alive`` probe the landing claim already trusts whether it
        did.

        THE DECK IS NOT GIVEN BACK BY A STOP THAT DID NOT HAPPEN. The deck
        is the record of what is on his screens; releasing it for a viewer
        that is still up is precisely what made the second ask unanswerable.
        A stop that failed leaves the cast live, says why, KEEPS WATCHING
        IT, and can be asked again.
        """
        if self.state.live != self.name:
            return StopReport(False, False, NOTHING_UP_LINE,
                              NOTHING_OF_OURS_REASON)
        ok, line, why = self._stop(confirm=True)
        if not ok:
            log.warning("castview: %s would not stop (%s); the deck stays "
                        "held", self.name, why)
            # ROUND 6, AND IT IS ONE CALL. ``_stop`` cancelled the watchdog
            # and moved the generation on, as it must for every end of a
            # cast -- but this cast did NOT end. Keeping the deck for a
            # viewer that is still up is right; keeping it with nothing
            # watching is how Jarvis's last word stayed "there's a viewer
            # still up that I couldn't close" for ever. MEASURED, both
            # directions: watchdogs armed 1, live 0, cancelled 1; he closes
            # the window himself, and 3000 s later retractions 0, deck
            # held, every "cast my screen" refused as busy. So the second
            # look is re-armed here, in the generation ``_stop`` just moved
            # to, and it is what takes the sentence back when the cast
            # ends by any route other than a stop that works.
            if not self._arm_confirm():
                log.warning("castview: %s is still up and can no longer be "
                            "watched", self.name)
            return StopReport(False, True, line or STOP_FAILED_LINE, why)
        self.state.release()
        return StopReport(True, True, STOPPED_LINE, "")

    # -- the evidence ---------------------------------------------------
    def _settle_and_confirm(self) -> Optional[object]:
        """Wait, look, and refuse to claim anything that is not there.

        Returns None when the cast landed -- and only then is the second
        look armed. Anything else is a ``held_result`` with the deck
        already given back, because a cast that cannot be shown to have
        arrived must not hold the deck against the next one.
        """
        try:
            self._settle(self.settle_s)
        except Exception:                           # noqa: BLE001 - the seam
            log.debug("castview: the settle wait raised", exc_info=True)
        if not self._still_there():
            log.warning("castview: %s was gone %.1fs after it started",
                        self.name, self.settle_s)
            return self._give_back(self.gone_line, self.gone_reason)
        linked = self._is_connected()
        if linked is False:
            return self._give_back(self.not_connected_line,
                                   NOT_CONNECTED_REASON)
        if linked is None:
            return self._give_back(CANNOT_TELL_LINE, CANNOT_TELL_REASON)
        if not self._arm_confirm():
            # ROUND 5: A LANDING IS NOT CLAIMED WHEN THE WATCHDOG CANNOT BE
            # ARMED. ``_arm_confirm`` used to swallow the exception from
            # ``_later`` and return, and ``deliver`` still said landed and
            # still held the deck -- so the one thing that would notice the
            # cast dropping was gone and nothing said so. The shipped
            # ``_later`` is ``threading.Timer(...).start()``, which raises
            # under thread exhaustion; that is not hypothetical on a box
            # that has had an OOM kill. Failing to arm it HOLDS.
            return self._give_back(NO_WATCHDOG_LINE, NO_WATCHDOG_REASON)
        return None

    def _give_back(self, line: str, why: str):
        """Give the deck back and say why. Best-effort on the stop itself.

        This is the path for a cast that did NOT land, so the deck must
        come back whatever the stop does -- holding it here is what round 3
        did, and the next genuine cast was refused as busy for ever. A stop
        that fails on the way out is logged and named in the detail; it is
        not allowed to strand the deck.
        """
        ok, _line, stop_why = self._stop()
        if not ok:
            log.warning("castview: %s could not be stopped on the way out "
                        "(%s)", self.name, stop_why)
            why = "%s; %s" % (why, stop_why)
        self.state.release()
        return held_result(line, sink=self.name, detail=why)

    def _arm_probe(self) -> None:
        """Tell the connection probe that THIS cast is what it is about.

        Called before anything is started, and never allowed to stop a
        cast: a probe that cannot take a baseline is a probe with less
        evidence, which the tri-state already knows how to say.
        """
        if self._arm is None:
            return
        try:
            self._arm()
        except Exception:                           # noqa: BLE001 - the seam
            log.debug("castview: arming the connection probe raised",
                      exc_info=True)

    def _is_connected(self) -> Optional[bool]:
        """True, False, or None for CANNOT TELL. A probe that raises has
        no opinion; it is never a yes."""
        if self._connected is None:
            return None
        try:
            got = self._connected()
        except Exception:                           # noqa: BLE001 - the seam
            log.debug("castview: the connection probe raised", exc_info=True)
            return None
        return None if got is None else bool(got)

    def _still_there(self) -> bool:
        """Is the thing this sink started still up? Overridden per
        direction; the base has nothing of its own to look at."""
        return True

    def _arm_confirm(self) -> bool:
        """The SECOND LOOK. A cast alive at the settle and gone at three
        seconds was claimed and never revisited, so the deck stayed held
        by something that was not there and the next genuine cast was
        refused as busy. This gives the deck back and takes the sentence
        back.

        ROUND 5 MADE IT REPORT AND MADE IT REPEAT. It returned None
        whether it armed or not, and the caller claimed a landing either
        way; now the caller HOLDS when it cannot be armed. And the timer
        it arms carries the generation it was armed in, so a cast he
        stopped in the meantime cannot be spoken about by a timer that was
        already in flight.
        """
        epoch = self._epoch
        try:
            self._watch = self._later(self.confirm_s,
                                      lambda: self._confirm(epoch))
        except Exception:                           # noqa: BLE001 - the seam
            log.warning("castview: the confirm timer would not arm for %s",
                        self.name, exc_info=True)
            self._watch = None
            return False
        return True

    def _confirm(self, epoch: Optional[int] = None) -> None:
        """Look again -- AND KEEP LOOKING.

        ROUND 5, TWO DEFECTS IN ONE METHOD.

        IT WAS ONE-SHOT. MEASURED: the confirm fired at 4.0 s, found
        everything well, and no timer was ever armed again. Run the clock
        200 s with the viewer gone and the deck was still held, retractions
        0, timers 0, and the next genuine cast refused as busy -- which is
        the exact failure the second look was added to close, arriving one
        confirm later. It re-arms itself now for as long as the cast is
        both live and ours.

        AND IT COULD RETRACT A STOP HE ASKED FOR. It read ``state.live``,
        then spent 0.35 s inside the real probe, and a "stop the cast" in
        that window still got "The cast has dropped, sir. What I told you a
        moment ago is no longer true." -- and pushed the suppression window
        forward with it, so the cast he asked for next was refused as too
        soon. The deck and the generation are BOTH re-read after the probe,
        because the probe is where the time goes.
        """
        if epoch is not None and epoch != self._epoch:
            return                                  # a stale timer
        self._watch = None
        if self.state.live != self.name:
            return                                  # already stopped
        there = self._still_there()
        linked = self._is_connected()
        if epoch is not None and epoch != self._epoch:
            return                                  # he stopped it mid-probe
        if self.state.live != self.name:
            return
        if there and linked is True:
            if not self._arm_confirm():
                # The cast IS there; refusing to believe it because a timer
                # would not start would be a second untrue sentence. Say so
                # in the log and stop watching, which is a degradation and
                # not a lie: ``stop_cast`` still works.
                log.warning("castview: %s is up but can no longer be "
                            "watched", self.name)
            return
        log.warning("castview: the cast was gone %.1fs after it was claimed",
                    self.confirm_s)
        ok, _line, why = self._stop()
        if not ok:
            log.warning("castview: ...and it would not stop either (%s)", why)
        self.state.release()
        if self._retract is not None:
            try:
                self._retract(CAST_GONE_LINE)
            except Exception:                       # noqa: BLE001 - the seam
                log.debug("castview: the retraction raised", exc_info=True)

    # -- stopping -------------------------------------------------------
    def _stop(self, *, confirm: bool = False) -> tuple:
        """``(ok, line, detail)``. ``ok`` is whether it actually stopped.

        ``confirm`` is what separates the two callers. HIS stop asks for a
        receipt and must not lie about it. The internal give-backs -- a
        cast that never landed, a second look that found it gone -- are
        cleanup on a cast that is already not there, and they must not
        spend the ack budget or hold the deck; they take the best-effort
        path and log what went wrong.

        The generation moves HERE, on every end of a cast, which is what
        makes an in-flight second look harmless.
        """
        self._epoch += 1
        watch, self._watch = self._watch, None
        cancel = getattr(watch, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:                       # noqa: BLE001 - the seam
                log.debug("castview: the confirm timer would not cancel",
                          exc_info=True)
        try:
            ok, line, why = self._stop_here(confirm=confirm)
        except Exception as exc:                    # noqa: BLE001 - the seam
            # ROUND 5: A RAISING ``_stop_here`` MUST NOT BECOME A TRUE.
            # It was swallowed here, so a kill that raised still returned,
            # still said "Cast stopped, sir.", still released the deck --
            # and left a viewer alive that nothing was tracking, so the
            # next cast opened a second one on top of it.
            log.warning("castview: stopping %s raised: %s", self.name, exc,
                        exc_info=True)
            return False, self.stop_failed_line, str(exc) or STOP_FAILED_REASON
        return bool(ok), str(line or ""), str(why or "")

    # What THIS direction says when the thing it started would not close.
    stop_failed_line = STOP_FAILED_LINE

    def _stop_here(self, *, confirm: bool = False) -> tuple:
        """``(ok, line, detail)``. The base has nothing of its own to stop,
        so there is nothing that can fail."""
        return True, "", ""


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
                 alive: Optional[Callable[[], bool]] = None,
                 connected: Optional[Callable[[], Optional[bool]]] = None,
                 settle: Optional[Callable[[float], object]] = None,
                 later: Optional[Callable[[float, Callable], object]] = None,
                 retract: Optional[Callable[[str], object]] = None,
                 arm: Optional[Callable[[], object]] = None,
                 settle_s: float = VIEWER_SETTLE_S,
                 confirm_s: float = VIEWER_CONFIRM_S,
                 host: str = HPCOMPUTER_HOST,
                 state: Optional[ViewState] = None,
                 now: Callable[[], float] = time.monotonic) -> None:
        super().__init__(state=state, now=now, connected=connected,
                         settle=settle, later=later, retract=retract,
                         arm=arm, settle_s=settle_s, confirm_s=confirm_s)
        self._launch = launch if callable(launch) else None
        self._stop_fn = stop if callable(stop) else None
        # ``alive`` answers one question a moment after the launch: is the
        # viewer still there? The real launcher is a Popen, so "it did not
        # raise" means the fork succeeded and nothing more.
        self._alive = alive if callable(alive) else None
        # ...AND ALIVE IS NOT CONNECTED, which is the round-3 finding on
        # this sink. Without direct-IP access and an unattended password
        # on HPCOMPUTER a direct-IP connect waits for someone to accept it
        # on the Windows side, and a viewer sitting on that prompt is a
        # live process: ``Popen.poll() is None`` said yes and round 2 said
        # "HPCOMPUTER's screen is on the Spark, sir." with nothing cast.
        # The tri-state ``connected`` probe is on ``_ViewSink`` now, and
        # jarvis/app.py implements this direction's out of /proc: an
        # ESTABLISHED socket the viewer itself owns to that host, and
        # sustained bytes coming in over it. It never opens a window, a
        # capture device or the picture.
        self.host = str(host)

    def available(self) -> tuple:
        if self._launch is None:
            return False, NO_VIEWER_REASON
        if self._alive is None:
            # The same rule the launcher is held to. A viewer whose state
            # cannot be read is a viewer whose landing cannot be claimed,
            # and this sink may not claim one.
            return False, NO_PROBE_REASON
        # ...and the connection probe, which the base owns because BOTH
        # directions are held to it now.
        return super().available()

    def deliver(self, subject: CastSubject):
        ok, why = self.available()
        if not ok:
            return self._unavailable(why)
        busy = self._busy()
        if busy is not None:
            return busy
        self._arm_probe()
        try:
            self._launch(self.host)
        except Exception as exc:                    # noqa: BLE001 - the seam
            log.warning("castview: the viewer would not open: %s", exc)
            self.state.release()
            return held_result(VIEWER_FAILED_LINE, sink=self.name,
                               detail=str(exc))
        held = self._settle_and_confirm()
        if held is not None:
            return held
        return landed_result(
            SHOWN_LINE.format(What=_cap(VIEW_SCREENS[HPCOMPUTER]),
                              target=self.label), sink=self.name)

    def _still_there(self) -> bool:
        """Ask whether the process is there. A probe that raises is not a
        yes."""
        try:
            return bool(self._alive())
        except Exception:                           # noqa: BLE001 - the seam
            log.debug("castview: the viewer probe raised", exc_info=True)
            return False

    def _stop_here(self, *, confirm: bool = False) -> tuple:
        """Close the viewer THIS app started, and -- when he asked for it --
        CHECK. The mirror of the helper's ``stop-failed`` code, made out of
        the probe the landing claim already trusts: after telling it to die,
        ask whether it did.

        A ``_stop_fn`` that raises is caught by ``_stop`` and is never a
        True; before round 5 it was swallowed, so a kill that raised still
        said "Cast stopped, sir." with the viewer alive and the deck given
        back, and the next cast opened a second viewer on top of it.
        """
        if self._stop_fn is None:
            return True, "", ""
        self._stop_fn()
        if not confirm or self._alive is None:
            return True, "", ""
        if self._still_there():
            return False, STOP_FAILED_LINE, STOP_FAILED_REASON
        return True, "", ""


class HpViewSink(_ViewSink):
    """The Spark's desktop, on HPCOMPUTER. Through a verb, never a command.

    It HOLDS whenever the Windows helper is not actually polling, and it
    says why. That is the honest degradation: nothing is queued for a
    machine that is not listening, so a startup script installed an hour
    later does not suddenly execute an hour-old intention.

    ROUND 4: A RECEIPT IS NOT A DESKTOP, AND THIS IS THE DIRECTION HE CAN
    ACTUALLY HIT TODAY -- the gesture ships off, the spoken cast does not.
    Round 3 made the receipt honest about the LAUNCH: the Windows script
    now moves its sequence only after a viewer it watched survive 700 ms.
    That is the strongest thing the script can say and it is still not a
    landing. A RustDesk viewer sitting on an accept-or-password prompt
    survives 700 ms perfectly well, and MEASURED against the real relay
    and the real sink it gave landed=True, held=False and "The Spark's
    screen is on HPCOMPUTER, sir." with nothing on any monitor -- and
    nothing re-checked it, so ``state.live`` stayed held by a cast that
    was not there and the next genuine cast was refused as busy.

    So this direction now needs what the other one needed: a tri-state
    connection probe (see ``_ViewSink``), a second look, and a spoken
    retraction. THE EVIDENCE IS THE MIRROR IMAGE of the Spark's, because
    the viewer is on HIS machine and there is no pid here to ask about:
    the Spark is the end being VIEWED, so what jarvis/app.py reads is an
    ESTABLISHED socket INBOUND from HPCOMPUTER on the screen-sharing port
    and bytes leaving over it. It opens nothing.

    AND WHAT IT STILL CANNOT SAY, unchanged and not quietly widened: that
    proves a live connection carrying a stream. It does not prove a
    window is visible, or on the middle monitor, or in front of what he
    was reading. That last step has no local evidence and is his.
    """

    name = "hp-view"
    label = VIEW_LABELS[HPCOMPUTER]
    unavailable_line = NO_HELPER_LINE
    gone_line = HELPER_GONE_LINE
    gone_reason = NO_HELPER_REASON
    not_connected_line = HP_NOT_CONNECTED_LINE

    def __init__(self, *, relay: Optional[CastRelay] = None,
                 state: Optional[ViewState] = None,
                 ack_s: float = HELPER_ACK_S,
                 wait: Optional[Callable[[float], object]] = None,
                 connected: Optional[Callable[[], Optional[bool]]] = None,
                 settle: Optional[Callable[[float], object]] = None,
                 later: Optional[Callable[[float, Callable], object]] = None,
                 retract: Optional[Callable[[str], object]] = None,
                 arm: Optional[Callable[[], object]] = None,
                 settle_s: float = VIEWER_SETTLE_S,
                 confirm_s: float = VIEWER_CONFIRM_S,
                 now: Callable[[], float] = time.monotonic) -> None:
        super().__init__(state=state, now=now, connected=connected,
                         settle=settle, later=later, retract=retract,
                         arm=arm, settle_s=settle_s, confirm_s=confirm_s)
        self.relay = relay
        self.ack_s = float(ack_s)
        # How the receipt is waited for. None is the real one -- the
        # relay's own event, set by the poll route when the helper answers.
        self._wait = wait if callable(wait) else None

    def available(self) -> tuple:
        if self.relay is None or not self.relay.alive():
            return False, NO_HELPER_REASON
        return super().available()

    def deliver(self, subject: CastSubject, *,
                wait: Optional[Callable[[float], object]] = None):
        """PARKED IS NOT RECEIVED, AND RECEIVED IS NOT SHOWN. Those are two
        separate steps and this method is both of them.

        ``alive()`` only says the helper polled within the last minute; it
        may last have polled 59 seconds ago, and a verb sitting on the
        relay for a machine that is not at the door is not a screen on a
        monitor. So the verb goes on, and then this waits for the helper's
        NEXT poll to quote that sequence back -- which is the Windows
        script saying it has acted on it.

        With no receipt the verb is TAKEN BACK OFF THE RELAY (as a stop,
        the closed set's own word for it, so that a helper which did get it
        and was merely slow to say so puts the window away again) and the
        deck is released. Nothing is left queued for a machine that did not
        answer -- the same rule this sink already held for a helper that
        was never running at all.

        AND THEN THE SAME AGAIN FOR THE PICTURE. The receipt means the
        script launched something and watched it survive its own 700 ms;
        it does not mean a desktop arrived, and a viewer on a password
        prompt satisfies it exactly. So the connection is probed, and a
        cast that cannot be shown to be carrying a stream gives the deck
        back and says so rather than claiming a landing.
        """
        ok, why = self.available()
        if not ok:
            return self._unavailable(why)
        busy = self._busy()
        if busy is not None:
            return busy
        # BEFORE THE VERB, so the probe can tell this cast's connection from
        # a RustDesk session he had open already.
        self._arm_probe()
        try:
            seq = self.relay.set_verb(VERB_SHOW)
        except Exception as exc:                    # noqa: BLE001 - the seam
            log.warning("castview: the relay refused the verb: %s", exc)
            self.state.release()
            return held_result(VIEWER_FAILED_LINE, sink=self.name,
                               detail=str(exc))
        if not self.relay.await_ack(seq, self.ack_s,
                                    wait=wait if callable(wait)
                                    else self._wait):
            # THE HELPER MAY NOW SAY WHY. A failed launch is not a
            # receipt, and the shipped script used to commit one anyway;
            # when it reports a reason instead, Jarvis repeats it rather
            # than saying "it didn't come back for it".
            why = self.relay.failed_for(seq)
            line = HELPER_FAIL_LINES.get(why, NO_RECEIPT_LINE)
            log.warning("castview: no receipt for verb %d after %.1fs (%s)",
                        seq, self.ack_s, why or "silence")
            self._stop()
            self.state.release()
            return held_result(line, sink=self.name,
                               detail=why or NO_RECEIPT_REASON)
        held = self._settle_and_confirm()
        if held is not None:
            return held
        return landed_result(
            SHOWN_LINE.format(What=_cap(VIEW_SCREENS[SPARK]),
                              target=self.label), sink=self.name)

    def _still_there(self) -> bool:
        """A machine that has stopped answering is a machine that is gone,
        whatever a socket says. This is the slow half of the evidence and
        the connection probe is the fast half."""
        return self.relay is not None and bool(self.relay.alive())

    stop_failed_line = HELPER_FAIL_LINES["stop-failed"]

    def _stop_here(self, *, confirm: bool = False) -> tuple:
        """Park the stop verb -- AND, WHEN HE ASKED FOR IT, WAIT FOR THE
        RECEIPT. PARKED IS NOT RECEIVED, exactly as for the show verb.

        The Windows script has reported ``stop-failed`` since round 4 and
        deliberately does NOT move its receipt on one, so the two outcomes
        are already distinguishable on the wire: an ack means the viewer is
        gone, a code means it is still up, and silence means the machine
        never came back for it. All three now reach a sentence.
        """
        if self.relay is None:
            return True, "", ""
        seq = self.relay.set_verb(VERB_STOP)
        if not confirm:
            return True, "", ""
        if self.relay.await_ack(seq, self.ack_s, wait=self._wait):
            return True, "", ""
        why = self.relay.failed_for(seq)
        line = HELPER_FAIL_LINES.get(why, NO_STOP_RECEIPT_LINE)
        log.warning("castview: no receipt for the stop %d after %.1fs (%s)",
                    seq, self.ack_s, why or "silence")
        return False, line, (why or NO_STOP_RECEIPT_REASON)


def _default_later(delay_s: float, fn: Callable[[], object]):
    """The real second look: one daemon timer, cancellable. It is the only
    thread this module owns and it does nothing but call back in."""
    timer = threading.Timer(float(delay_s), fn)
    timer.daemon = True
    timer.start()
    return timer


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
    "ACK_WAIT_ROUNDS", "CANNOT_TELL_LINE", "CANNOT_TELL_REASON",
    "CAST_GONE_LINE", "CAST_GONE_REASON", "HELPER_FAILS",
    "HELPER_FAIL_LINES", "LAUNCH_FAILED_LINE",
    "HELPER_GONE_LINE", "HP_NOT_CONNECTED_LINE", "RUSTDESK_PORTS",
    "NOT_CONNECTED_REASON", "NO_CONNECTION_PROBE_REASON",
    "VIEWER_CONFIRM_S", "VIEWER_NOT_CONNECTED_LINE", "VIEWER_STREAM_BPS",
    "BUSY_FULL_LINE", "BUSY_LINE", "CAST_SUPPRESS_S", "CastRelay",
    "HELPER_ACK_S", "HELPER_ALIVE_S", "HPCOMPUTER_HOST", "HpViewSink",
    "MAX_LAYOUT_CHARS",
    "NOTHING_UP_LINE", "NO_EVIDENCE_LINE", "NO_HELPER_LINE",
    "NO_HELPER_REASON", "NOTHING_OF_OURS_REASON",
    "NO_PROBE_REASON", "NO_RECEIPT_LINE", "NO_RECEIPT_REASON",
    "NO_STOP_RECEIPT_LINE", "NO_STOP_RECEIPT_REASON", "NO_WATCHDOG_LINE",
    "NO_WATCHDOG_REASON", "STOP_FAILED_LINE", "STOP_FAILED_REASON",
    "StopReport",
    "NO_VIEWER_LINE", "NO_VIEWER_REASON", "SHOWN_LINE", "SPARK_HOST",
    "STOPPED_LINE", "SparkViewSink", "VERBS", "VERB_NONE", "VERB_SHOW",
    "VERB_STOP", "VIEWER_DIED_LINE", "VIEWER_DIED_REASON",
    "VIEWER_FAILED_LINE", "VIEWER_SETTLE_S", "VIEW_LABELS", "VIEW_SCREENS",
    "ViewState", "registry", "view_subject",
]
