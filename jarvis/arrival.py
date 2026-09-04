"""Arrival and departure: the room notices the door.

The delta this module owns is ORDERING, not new faculties. "Welcome back,
sir" and the "While you were out, sir: ..." catch-up already shipped; what
was missing was that they landed as a bare line into a dead room. Arrival
is now a composed cue in a fixed order --

    panel -> earcon -> greeting -> catch-up

-- and departure is its mirror in the one way that matters: it is SILENT.
He is walking out of the door; a valediction spoken to an empty room is a
notification pretending to be a presence.

Everything here is a pure function of its arguments so the ordering is
testable without a phone, a mic or a Tk window; ``run()`` takes the actions
as callables and returns the steps it actually performed. (``DoorWatch`` is
the one object, and it holds a single bool and takes no clock, no config
and no socket -- the rule it enforces is still the pure ``door_arrival``.)

THREE THINGS WERE ADDED 2026-09-03, and all three say LESS than they could:

* **the kitchen is a door sensor.** ``door_arrival`` / ``DoorWatch`` --
  the room next to his front door going occupied after a whole-home
  absence, routed through the same ``run()`` and the same once-per-return
  damper as the phone and desk probes.
* **"Welcome back from X"** -- ``outing`` names where he was ONLY when a
  calendar event honestly covered the absence. Two events that both fit is
  no name at all: a guessed event name is worse than no event name.
* **the catch-up OFFERS** -- ``catch_up_offer`` is a count and a question,
  never a sender or a subject. His ruling: "He should offer." A standing
  FAULT is told rather than offered: there is nothing to "go through" in
  a broken disk, and the first cut asked anyway.

Three things this module deliberately does NOT do.

* **It does not speak on departure.** Not even quietly.
* **It does not move music to his phone.** ``spotify.transfer_playback``
  defaults to ``force_play=True``, and an unprompted cue that STARTS audio
  in the pocket of a man walking to his car is a far worse failure than a
  paused song. If it is ever built it must take the ``force_play=False``
  path and be opt-in; today it is simply absent.
* **It does not put the panel into standby.** The night surface is owned
  elsewhere; two features writing one surface is how it ends up flickering.

DEPARTURE IS ASYMMETRIC BY CONSTRUCTION. Arrival fires on first sight,
because being late to notice him is the whole failure mode. Departure has
to clear three gates: the sentinel's own away grace (``away_after_min``,
which clamps at 60 s and defaults to 12 minutes), then a further confirm
window, then a hard veto if the microphone heard him inside
``mic_silence_s``. The veto is a veto and not a weight: a sleeping phone
radio faking a departure while he is sitting in the room is the one
outcome worth spending latency to avoid.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("arrival")

# The order is the feature. Do not reorder without a reason in the commit:
# the panel comes up first so the greeting is not spoken at a dark screen,
# the earcon precedes the words the way a knock precedes a sentence, and
# the catch-up follows the greeting because it is the answer to it.
ARRIVAL_STEPS = ("panel", "earcon", "greeting", "catch-up")
# ...and the two halves it SPLITS INTO when the catch-up is deferred to his
# desk. Declared as the split of ARRIVAL_STEPS rather than as two hand-typed
# tuples, and pinned by
# tests/test_arrival_desk.py::test_the_door_and_the_desk_together_are_still_the_whole_cue,
# so a step added to the contract and to neither half fails the suite
# instead of silently never running.
DESK_STEPS = ("catch-up",)
DOOR_STEPS = tuple(s for s in ARRIVAL_STEPS if s not in DESK_STEPS)
DEPARTURE_STEPS = ("settle",)

ARRIVAL_EARCON = "arrival"
DEFAULT_CONFIRM_MIN = 5
DEFAULT_MIC_SILENCE_MIN = 10


def arrival_plan(*, returned: bool = False, home: bool = True,
                 quiet_reason: str = "", cue: bool = True,
                 defer_catch_up: bool = False) -> list[str]:
    """The ordered steps for a Presence event, or [] for no cue at all.

    ``quiet_reason`` is quiet.py's answer, and it is honoured exactly as
    ``app._on_presence`` always honoured it: "you're out" is stale by
    definition on a returned event and never defers anything, while any
    OTHER reason (quiet hours, DND, a meeting) leaves the panel to come up
    and holds the voice for the policy's own tick.

    ``defer_catch_up`` drops the last step and ONLY the last step: he is
    greeted at the door exactly as before, and the question about his mail
    is owed until he reaches his desk (``settle_plan``, ``DeskWatch``).
    """
    if not home or not returned:
        return []
    reason = str(quiet_reason or "").strip()
    if reason and reason != "you're out":
        return ["panel"]
    skip = set(DESK_STEPS) if defer_catch_up else set()
    return [s for s in ARRIVAL_STEPS
            if (cue or s != "earcon") and s not in skip]


def departure_plan(*, home: bool = True) -> list[str]:
    """The steps for a departure. Silent by design -- see the docstring."""
    return [] if home else list(DEPARTURE_STEPS)


def confirm_s(cfg_get: Callable, default_min: float = DEFAULT_CONFIRM_MIN) -> float:
    """``presence.departure_confirm_min`` in seconds, floored at 60.

    Floored rather than clamped to the away grace: the grace has already
    elapsed when the event fires, so this window is purely additional and a
    minute of it is the least that means anything.
    """
    try:
        minutes = float(cfg_get("presence.departure_confirm_min", default_min))
    except (TypeError, ValueError):
        minutes = float(default_min)
    return max(60.0, minutes * 60.0)


def mic_silence_s(cfg_get: Callable, default_min: float = DEFAULT_MIC_SILENCE_MIN) -> float:
    try:
        minutes = float(cfg_get("presence.departure_mic_silence_min", default_min))
    except (TypeError, ValueError):
        minutes = float(default_min)
    return max(60.0, minutes * 60.0)


def departure_ready(*, now: float, since: float, last_turn: Optional[float],
                    confirm: float, mic_silence: float,
                    still_away: bool = True) -> tuple[bool, str]:
    """May the room settle? (ok, why-not). Every gate is a hard veto."""
    if not still_away:
        return False, "he came back"
    if last_turn is not None and now - float(last_turn) < mic_silence:
        # The turn ledger, not a new field: the mic heard him, so whatever
        # his phone's radio is doing he is in the room.
        return False, f"the mic heard him {int(now - float(last_turn))}s ago"
    waited = now - float(since or 0.0)
    if waited < confirm:
        return False, f"only {int(waited)}s past the away grace"
    return True, ""


def run(steps, actions: dict) -> list[str]:
    """Perform ``steps`` in order; returns those that actually ran.

    A step with no action registered is skipped silently (the panel hook is
    optional), and a step that raises is logged and does NOT abort the rest
    -- a missing earcon must never cost him the greeting.
    """
    done = []
    for step in steps:
        fn = actions.get(step)
        if not callable(fn):
            continue
        try:
            if fn() is False:
                continue          # the step declined (nothing held, say)
        except Exception:         # noqa: BLE001 - one broken step, not the cue
            log.exception("arrival: step %r failed", step)
            continue
        done.append(step)
    return done


# ======================================================================
# THE DOOR: a room whose presence, after a whole-home absence, IS arrival
# ======================================================================
# His words for why the kitchen and not the office: "kitchen to see if i
# enter my apartment since the kitchen and door are next to each other".
# The phone leg cannot do this job. It notices him when his phone's radio
# next answers an ARP, which on an iPhone in Wi-Fi power-save is whenever
# it feels like it; the radar sees a body on the doorstep inside one poll.
#
# THE GATE IS THE ABSENCE, NOT THE ROOM. A kitchen going occupied is only
# a door opening if the house was empty before it. He makes coffee three
# times an evening, and a welcome per cup is the exact nuisance the
# arrival rework was built to avoid. So ``away`` here is the presence
# sentinel's own verdict -- "away", not "unknown" -- and nothing else in
# this module tries to re-derive it.
DEFAULT_DOOR_ROOM = "kitchen"


def _room_key(value) -> str:
    """A room name as it is compared: slugged the way roomfabric slugs it,
    so a hand-written config key and a fabric event name meet.

    THE SAME RULE, not merely a similar one. ``roomfabric._slug`` also
    strips every character that is not alphanumeric, space, hyphen or
    underscore, and the first cut here did not -- so a room called
    "Kitchen!" became the fabric room "kitchen" while a ``door_room`` of
    "Kitchen!" matched nothing, silently, for ever. It is copied rather
    than imported because this module's promise is that it pulls in
    nothing that owns a thread or a socket; the two are pinned to each
    other by
    tests/test_arrival_kitchen.py::test_the_room_key_slugs_exactly_as_the_fabric_does.
    """
    text = " ".join(str(value or "").split()).lower()
    return "".join(ch if (ch.isalnum() or ch in " -_") else ""
                   for ch in text).strip()


def door_arrival(*, room, door: str = DEFAULT_DOOR_ROOM,
                 away: bool = False) -> bool:
    """Is THIS room becoming occupied a door opening?

    True only when the house was away and the room is the door room.
    Deliberately NOT consulted: which room was active before. The fabric
    drops its active room after ``rooms_stale_after_s`` (90 s) of nothing,
    so an absence measured in minutes always arrives with no previous
    room -- but a radar stuck on by a fan would leave one, and refusing
    the arrival in that case would mean the fan silently costs him the
    greeting. The away gate is the evidence; the previous room is trivia.

    ``room`` may be None (no room is active, a sensor with no opinion) and
    that is not a door either.
    """
    key = _room_key(room)
    return bool(away) and bool(key) and key == _room_key(door)


class DoorWatch:
    """The rising edge of ``door_arrival``, and nothing else.

    Pure: it holds one string and takes no clock, no config and no socket.
    The fabric republishes RoomChanged whenever the ACTIVE room moves, and
    a walk kitchen -> office -> kitchen inside one return would otherwise
    put a second welcome on the floor. This is the first guard; the app's
    ``GREET_DAMPER_S`` (600 s, shared with the phone and desk probes) is
    the second, and the one that catches two SENTINELS crossing on the
    same walk. Both exist on purpose -- see app._greet_return, where the
    damper was orphaned once already.

    ``left()`` re-arms it, so a genuine second outing is greeted again.
    """

    def __init__(self, door: str = DEFAULT_DOOR_ROOM):
        self.door = door
        self._fired = False

    def left(self) -> None:
        """He went out. The next door opening is a new arrival."""
        self._fired = False

    def observe(self, *, room, away: bool = False) -> bool:
        """One RoomChanged. True exactly once per return."""
        if not door_arrival(room=room, door=self.door, away=away):
            return False
        if self._fired:
            return False
        self._fired = True
        return True


# ======================================================================
# THE DESK: he is greeted at the door and ASKED when he sits down
# ======================================================================
# His words, 2026-09-03: "Welcome back sir at the door and then when I'm in
# my office he can ask about stuff. This can be through the sensor logic
# and the camera logic if he sees me (if applicable)."
#
# So the cue splits. panel -> earcon -> greeting still run together the
# moment the kitchen sees him; the catch-up is ARMED there and DELIVERED
# when he settles. What it fixes is not a bug in any one line, it is the
# timing of the whole second half: today "shall I go through your mail" is
# put to a man who is still taking his shoes off.
#
# TWO LEGS, WHICHEVER ARRIVES FIRST, because at his desk his back is to the
# radar and his face is to the camera and both of those are NORMAL:
#
#   radar   -- the office zone verdict reaches its desk band. Measured
#              2026-09-03: he sits at 3.13 m median and the office map's
#              "at the desk" band is 2.25-3.75 m.
#   camera  -- the lens recognises him in the office. AN IDENTITY LABEL AND
#              A SCORE. There is no argument in this module an image could
#              be passed through, and there never will be.
#
# ONE DELIVERY, NEVER TWO, however many times either leg fires: DeskWatch
# is the latch, and it holds a lock because the two legs are two lanes and
# neither owns the Tk pump.
#
# DEFERRING IS NOT DRAINING. Nothing here touches quiet.release_fragments:
# the held lines stay held, in quiet.py, exactly where they were. If he
# never reaches the desk -- straight to bed, the office module unplugged,
# the camera inside its 21:00-07:00 curfew -- the watch simply stays armed
# and the backlog stays held for the policy's own next quiet window. That
# is the whole answer to "does a deferred catch-up expire": IT DOES NOT.
# Nothing is captured at the door to go stale. The offer is a live read of
# the mailbox and the fault board taken at DELIVERY, so an offer made at
# eleven at night is as true as one made at six; and the alternative --
# expiring it -- is a timer whose only possible effect is to throw away
# lines that are his and have no second copy anywhere.
DEFAULT_DESK_ROOM = "office"
# zones.DEFAULT_CAMERA_ZONE / the office map's far band, spelled here rather
# than imported for the reason _room_key is copied rather than imported:
# this module pulls in nothing that owns a thread, a file or a socket, and
# jarvis/zones.py owns a log file and a config reader. The two are pinned
# to each other by
# tests/test_arrival_desk_app.py::test_the_desk_zone_is_the_one_zones_py_names.
DEFAULT_DESK_ZONE = "at the desk"

LEG_RADAR = "radar"
LEG_CAMERA = "camera"


def defer_catch_up(*, legs=(), enabled: bool = True) -> bool:
    """Should the catch-up be held for the desk rather than asked at the door?

    ``legs`` is the legs the caller believes can actually FIRE right now
    (app._settle_legs). The check is not ceremony: a deferral no leg can
    ever fire is the catch-up silently never happening, which is a worse
    outcome than asking him about his mail in the hallway. With no zone
    source attached and no camera feed, this returns False and the cue is
    byte for byte the one that shipped.
    """
    return bool(enabled) and bool(tuple(legs))


def settle_plan(*, quiet_reason: str = "") -> list[str]:
    """The steps for a settle, or [] to leave the catch-up OWED.

    THE ONE NEW WAY THIS FEATURE COULD SPEAK OVER A QUIET HOUR, and it is
    closed here. At the door a quiet house makes the plan panel-only
    (``arrival_plan``) so the catch-up never ran during quiet hours at
    all. Deferred, it can arrive at his desk an hour later -- and
    ``app._say`` is called for the digest with ``proactive=False``, which
    does not consult quiet.py and would pierce the hold.

    Held is NOT dropped: the caller leaves the watch armed, so the next
    settle after the window closes delivers it.

    "you're out" is the one reason ignored, for the same reason
    ``arrival_plan`` ignores it: he is demonstrably at his own desk, so a
    sentinel still saying he is out is stale by the time this is read.
    """
    reason = str(quiet_reason or "").strip()
    if reason and reason != "you're out":
        return []
    return list(DESK_STEPS)


def _camera_names_him(camera) -> bool:
    """Does this camera opinion NAME somebody? A label, or a
    ``zones.CameraOpinion``; never a frame, and there is no third form.

    ``known=False`` is the lens saying "I looked and recognised nobody",
    which zones.py is explicit is NOT a claim the chair is empty -- so it
    is not a settle either. Anything falsy is no opinion at all.
    """
    if camera is None:
        return False
    known = getattr(camera, "known", None)
    if known is not None:                      # a CameraOpinion
        return bool(known) and bool(str(getattr(camera, "label", "") or "").strip())
    return bool(str(camera or "").strip())     # an identity label


def settle(*, room="", verdict=None, camera=None,
           desk_room: str = DEFAULT_DESK_ROOM,
           desk_zone: str = DEFAULT_DESK_ZONE) -> str:
    """Has he settled at his desk, and by WHICH LEG? ``LEG_RADAR``,
    ``LEG_CAMERA``, or "" for not yet.

    Pure, and pure in the strong sense: it reads three arguments and
    returns a string. No clock, no config, no socket, no sensor -- which
    is what lets the whole of this feature be tested with no phone, no
    radar, no lens and no window.

    ``verdict`` is a ``zones.Verdict`` (anything with ``.zone``, and
    optionally ``.rule`` and ``.room``) or a plain zone name. ``camera``
    is a ``zones.CameraOpinion`` or an identity label -- ``app._eye_identity``
    hands out a name and nothing else, and that string is the entire
    camera-side interface of this feature.

    **IT MAY NOT LIE ABOUT WHICH LEG DECIDED.** ``zones.verdict`` lets the
    camera overrule the radar and the camera-ruled verdict then carries
    the desk zone, so reading ``.zone`` alone would credit the radar for a
    decision the lens made -- with his back to the radar and no range
    reading involved at all. ``.rule`` is what settles it. With both legs
    honestly true at once the CAMERA is named: it identified him, where
    the radar saw a body in a band.

    Never raises. A sensor lane that explodes on attribute access costs
    this observation and nothing else -- it must not raise through a bus
    subscriber and take the deferred catch-up with it.
    """
    try:
        return _settle(room, verdict, camera, desk_room, desk_zone)
    except Exception:  # noqa: BLE001 - a broken lane is not a homecoming
        log.debug("arrival: the settle verdict could not be read", exc_info=True)
        return ""


def _settle(room, verdict, camera, desk_room, desk_zone) -> str:
    want = _room_key(desk_room)
    here = _room_key(room) or _room_key(getattr(verdict, "room", ""))
    if not want or here != want:
        return ""
    zone = verdict if isinstance(verdict, str) else getattr(verdict, "zone", "")
    at_desk = _room_key(zone) == _room_key(desk_zone) and bool(_room_key(zone))
    # THE CAMERA FIRST, and by RULE rather than by zone -- see the docstring.
    if _camera_names_him(camera):
        return LEG_CAMERA
    if at_desk and str(getattr(verdict, "rule", "") or "") == LEG_CAMERA:
        return LEG_CAMERA
    return LEG_RADAR if at_desk else ""


class DeskWatch:
    """The deferred catch-up: armed at the door, fired ONCE at the desk.

    Holds one bool and a lock and nothing else -- no clock, no config, no
    socket, and deliberately no copy of the digest. What is deferred is
    the QUESTION, not its answer: the mail count and the fault board are
    read at delivery, which is why a deferral can sit for hours without
    going stale and why there is no expiry to get wrong.

    The lock is not decoration. The radar leg and the camera leg are two
    lanes on two threads and neither owns the Tk pump, so "exactly once"
    cannot rest on the GIL landing between the read and the write of a
    bare bool.

    ``arm()`` is idempotent (two sentinels crossing on one walk through
    the door both reach the greeter), ``clear()`` gives the catch-up up,
    and only a REAL settle spends the arm -- walking through the office to
    the bedroom leaves it owed.
    """

    def __init__(self, room: str = DEFAULT_DESK_ROOM,
                 zone: str = DEFAULT_DESK_ZONE):
        self.room = room
        self.zone = zone
        self._armed = False
        self._lock = threading.Lock()

    @property
    def armed(self) -> bool:
        return self._armed

    def arm(self) -> None:
        """A catch-up is owed. Idempotent."""
        with self._lock:
            self._armed = True

    def clear(self) -> None:
        """It was delivered, or given up on. Idempotent."""
        with self._lock:
            self._armed = False

    def observe(self, *, room="", verdict=None, camera=None) -> str:
        """One reading from either leg. The leg that settled him, EXACTLY
        once per ``arm()``; "" every other time, unarmed included.
        """
        leg = settle(room=room, verdict=verdict, camera=camera,
                     desk_room=self.room, desk_zone=self.zone)
        if not leg:
            return ""
        with self._lock:
            if not self._armed:
                return ""
            self._armed = False
        return leg


# ======================================================================
# "WELCOME BACK FROM X" -- named only on evidence, never on a guess
# ======================================================================
WELCOME_FROM_LINE = "Welcome back from {what}, sir."
# He must have been out for most of the event before it is called the
# reason he was out. A 3-hour lab he caught the last twenty minutes of is
# not where he was all afternoon.
OUTING_COVER = 0.5
# ...and it must have ended shortly before he walked in, or still be
# running. Without this a 9 a.m. lecture names a 6 p.m. homecoming, which
# is a guess wearing a fact's clothes. 45 minutes is a commute with a stop.
OUTING_ENDED_WITHIN_S = 45 * 60.0
# Under this there is nothing to be back FROM: he took the bins out.
OUTING_MIN_ABSENCE_S = 5 * 60.0
# ...and the event has to explain most of the time he was ACTUALLY GONE.
# This is the second half of the coverage rule and it was missing on the
# first cut: OUTING_COVER alone measures the overlap against the EVENT, so
# a four-minute entry that ended sixteen minutes before he walked in named
# a four-hour absence -- measured, "Welcome back from take the bins out,
# sir." after four hours out. Both directions now have to hold: he was at
# most of the event, AND the event was most of his absence.
#
# WHAT THIS REFUSES, and it is the honest cost: a one-hour class inside a
# two-and-a-half-hour absence is 0.4 and gets the plain line. A long
# commute either side of a short event is exactly the case where the
# calendar cannot prove where he was, and his rule is that a guessed event
# name is worse than no event name. Raise it toward 0.0 to name more and
# guess more; it is a keyword argument for that reason.
OUTING_ABSENCE_COVER = 0.5
# A calendar title can be a paragraph (his Canvas feed's are). Spoken as
# the tail of a two-second greeting, a long one is worse than the plain
# line, so it disqualifies the match rather than being truncated -- a
# half-read title is the same guess with fewer words.
OUTING_MAX_TITLE = 48


def speakable_title(value, cap: int = OUTING_MAX_TITLE) -> str:
    """A calendar title as it may be SPOKEN, or "" when it may not."""
    text = " ".join(str(value or "").split())
    return text if text and len(text) <= int(cap) else ""


def outing(events, *, left, back, cover: float = OUTING_COVER,
           ended_within_s: float = OUTING_ENDED_WITHIN_S,
           min_absence_s: float = OUTING_MIN_ABSENCE_S,
           max_title: int = OUTING_MAX_TITLE,
           absence_cover: float = OUTING_ABSENCE_COVER) -> str:
    """What he was out AT, or "" -- and "" is the common answer.

    ``events`` is whatever ``CalendarSource.events()`` returns (anything
    with ``start`` / ``end`` / ``all_day`` / ``title``); ``left`` and
    ``back`` are datetimes bracketing the absence. An event qualifies when
    every one of these holds:

    * it is not all-day (a "Fall break" spanning the absence is not
      somewhere he went);
    * he was out for at least ``cover`` of it;
    * **it accounts for at least ``absence_cover`` of the absence** -- the
      symmetric half, without which a four-minute errand named a four-hour
      outing (the docstring's own counter-example is what the code
      produced before 2026-09-03);
    * it ended no more than ``ended_within_s`` before he walked in, or was
      still running;
    * its title is short enough to say.

    **A GUESSED EVENT NAME IS WORSE THAN NO EVENT NAME**, so TWO
    qualifying events is "" as surely as none: the calendar honestly
    cannot say which he went to, and the plain welcome is not a failure.
    The same event carried by two feeds is one title and still matches --
    ``merge_events`` does not always fold an iCloud copy and a subscribed
    copy together, and a duplicate must not disqualify a real answer.

    Never raises. Every failure -- a naive datetime meeting an aware one,
    a caldav object that throws on attribute access -- is "", because the
    cost of an exception here is the whole arrival cue.
    """
    try:
        return _outing(events, left, back, float(cover), float(ended_within_s),
                       float(min_absence_s), int(max_title),
                       float(absence_cover))
    except Exception:  # noqa: BLE001 - a name is never worth the greeting
        log.debug("arrival: the outing match failed; plain welcome",
                  exc_info=True)
        return ""


def _outing(events, left, back, cover, ended_within_s, min_absence_s,
            max_title, absence_cover) -> str:
    if left is None or back is None:
        return ""                       # no recorded departure: nothing to match
    absence = (back - left).total_seconds()
    if absence < min_absence_s:
        return ""
    found: dict = {}
    for event in events or ():
        if getattr(event, "all_day", False):
            continue
        start, end = getattr(event, "start", None), getattr(event, "end", None)
        if start is None or end is None:
            continue
        length = (end - start).total_seconds()
        if length <= 0:
            continue
        overlap = (min(end, back) - max(start, left)).total_seconds()
        if overlap <= 0 or overlap < cover * length:
            continue
        # THE OTHER DIRECTION. The line above asks "was he at the event?";
        # this one asks "was the event where he was?". Only both together
        # rule out a short entry standing in for hours nobody can account
        # for -- see OUTING_ABSENCE_COVER.
        if overlap < absence_cover * absence:
            continue
        if (back - end).total_seconds() > ended_within_s:
            continue
        title = speakable_title(getattr(event, "title", ""), max_title)
        if title:
            found.setdefault(title.lower(), title)
    if len(found) == 1:
        return next(iter(found.values()))
    if found:
        log.info("arrival: %d events could be where he was; plain welcome",
                 len(found))
    return ""


def welcome_line(what: str = "") -> str:
    """The greeting. ``what`` empty (the usual case) is the plain line.

    ``WELCOME_LINE`` is imported lazily so this module keeps its promise of
    importing nothing that starts a thread or opens a socket -- and so
    presence.py stays free to reach for arrival.py later without a cycle.
    """
    from jarvis.presence import WELCOME_LINE
    text = " ".join(str(what or "").split())
    return WELCOME_FROM_LINE.format(what=text) if text else WELCOME_LINE


# ======================================================================
# THE CATCH-UP OFFERS. IT DOES NOT DELIVER.
# ======================================================================
# His ruling, recorded 2026-09-02 after a five-word request was answered
# with 40 seconds of monologue: "He should offer." So this half of the
# arrival cue is a COUNT and a question -- never a sender, never a
# subject, never a body. He gets the contents when he answers yes, and the
# yes is resolved by the offer protocol that already exists
# (app._offer_first_wake_briefing parks it, Commander._try_briefing_offer
# answers it), so a yes cannot mean different things on different rungs.
CATCH_UP_QUESTION = "Shall I go through {it}, sir?"


def catch_up_offer(*, unread=None, major: str = "") -> str:
    """The offer, or "" when there is nothing honest to ask about.

    ``unread`` is a count or None. **None is SILENCE about mail -- never
    "no mail"** -- the same rule ``RoomSensor.read`` follows, for the same
    reason: a mailbox that timed out must not be reported as an empty one.
    (It does not say "I could not look" either; the first cut's docstring
    claimed a line that has never existed. There is no honest short way to
    say it at the door, and an unreachable mailbox is not news.) Zero
    unread and a clear board is silence too, not "no email, sir".

    ``major`` is one clause about anything that actually went wrong while
    he was out, in the words its own source wrote (jarvis/faults.py builds
    that line; nothing here paraphrases it).

    **THE QUESTION IS ABOUT THE MAIL AND NOTHING ELSE.** With no unread
    count and a fault standing, this returns the fault as a STATEMENT with
    no question on the end. "The disk is full. Shall I go through it,
    sir?" was what the first cut said, and there is nothing to go through:
    answering yes just spoke the same sentence back. A fault is told, not
    offered. ``offers_to_read`` is how the caller tells the two apart, and
    only a question is worth parking on the offer protocol.

    The fault clause is its own sentence and its case is LEFT ALONE. The
    first cut lower-cased its first letter to splice it in after "and",
    which turned "Ollama is unreachable" into "ollama is unreachable" and
    "I have lent the GPU to your trainer" -- health.py's real wording --
    into "i have lent". The digest is shown on the card as well as spoken,
    and no rule can tell "Memory" (safe to lower) from "Ollama" (not) by
    looking at it, so nothing is re-cased at all.
    """
    return " ".join(catch_up_fragments(unread=unread, major=major))


def catch_up_fragments(*, unread=None, major: str = "") -> list:
    """``catch_up_offer`` before it is joined: 0, 1 or 2 whole lines.

    The fault is its own fragment and the mail question is another,
    because that is the invariant ``jarvis/address.py`` thinning needs --
    "each fragment must be a whole authored Jarvis line". health.py's real
    wording carries its own "sir" ("I have lent the GPU to your trainer,
    sir; quick answers only..."), and as one three-sentence blob the burst
    would arrive at the door addressing him twice with nothing able to
    take one out.
    """
    try:
        count = None if unread is None else max(0, int(unread))
    except (TypeError, ValueError):
        count = None
    fault = " ".join(str(major or "").split())
    parts = []
    if fault:
        parts.append(fault if fault.endswith((".", "!", "?")) else fault + ".")
    if not count:
        # No mail to go through: the fault stands alone, or there is
        # nothing at all to say.
        return parts
    # "Shall I go through THEM" for three emails, "IT" for one. A pronoun
    # that does not agree is the tell that a line was assembled rather
    # than written, and this one is spoken at the door. It counts the
    # MAIL, because the mail is all a yes delivers.
    parts.append("You've %d unread email%s. %s" % (
        count, "" if count == 1 else "s",
        CATCH_UP_QUESTION.format(it="them" if count > 1 else "it")))
    return parts


def offers_to_read(line: str) -> bool:
    """Did ``catch_up_offer`` ask a question, or just tell him something?

    The caller has to know: only a question may be parked on
    ``services.briefing_offer``, and parking a statement would leave a
    "yes" hanging on nothing. A fault-only line is a statement.
    """
    return str(line or "").strip().endswith("?")
