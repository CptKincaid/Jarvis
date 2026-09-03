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
  never a sender or a subject. His ruling: "He should offer."

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

from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("arrival")

# The order is the feature. Do not reorder without a reason in the commit:
# the panel comes up first so the greeting is not spoken at a dark screen,
# the earcon precedes the words the way a knock precedes a sentence, and
# the catch-up follows the greeting because it is the answer to it.
ARRIVAL_STEPS = ("panel", "earcon", "greeting", "catch-up")
DEPARTURE_STEPS = ("settle",)

ARRIVAL_EARCON = "arrival"
DEFAULT_CONFIRM_MIN = 5
DEFAULT_MIC_SILENCE_MIN = 10


def arrival_plan(*, returned: bool = False, home: bool = True,
                 quiet_reason: str = "", cue: bool = True) -> list[str]:
    """The ordered steps for a Presence event, or [] for no cue at all.

    ``quiet_reason`` is quiet.py's answer, and it is honoured exactly as
    ``app._on_presence`` always honoured it: "you're out" is stale by
    definition on a returned event and never defers anything, while any
    OTHER reason (quiet hours, DND, a meeting) leaves the panel to come up
    and holds the voice for the policy's own tick.
    """
    if not home or not returned:
        return []
    reason = str(quiet_reason or "").strip()
    if reason and reason != "you're out":
        return ["panel"]
    steps = [s for s in ARRIVAL_STEPS if cue or s != "earcon"]
    return steps


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
    so a hand-written config key and a fabric event name meet."""
    return " ".join(str(value or "").split()).strip().lower()


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
           max_title: int = OUTING_MAX_TITLE) -> str:
    """What he was out AT, or "" -- and "" is the common answer.

    ``events`` is whatever ``CalendarSource.events()`` returns (anything
    with ``start`` / ``end`` / ``all_day`` / ``title``); ``left`` and
    ``back`` are datetimes bracketing the absence. An event qualifies when
    every one of these holds:

    * it is not all-day (a "Fall break" spanning the absence is not
      somewhere he went);
    * he was out for at least ``cover`` of it;
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
                       float(min_absence_s), int(max_title))
    except Exception:  # noqa: BLE001 - a name is never worth the greeting
        log.debug("arrival: the outing match failed; plain welcome",
                  exc_info=True)
        return ""


def _outing(events, left, back, cover, ended_within_s, min_absence_s,
            max_title) -> str:
    if left is None or back is None:
        return ""                       # no recorded departure: nothing to match
    if (back - left).total_seconds() < min_absence_s:
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

    ``unread`` is a count or None, and **None is "I could not look", never
    "no mail"** -- the same rule ``RoomSensor.read`` follows, for the same
    reason: a mailbox that timed out must not be reported as an empty one.
    Zero unread and a clear board is silence, not "no email, sir".

    ``major`` is one clause about anything that actually went wrong while
    he was out, in the words its own source wrote (jarvis/faults.py builds
    that line; nothing here paraphrases it).
    """
    clauses = []
    try:
        count = None if unread is None else max(0, int(unread))
    except (TypeError, ValueError):
        count = None
    if count:
        clauses.append("you've %d unread email%s" % (count, "" if count == 1 else "s"))
    clause = " ".join(str(major or "").split()).rstrip(".")
    if clause:
        clauses.append(clause[0].lower() + clause[1:] if clauses else clause)
    if not clauses:
        return ""
    # "Shall I go through THEM" for three emails, "IT" for one thing. A
    # pronoun that does not agree is the tell that a line was assembled
    # rather than written, and this one is spoken at the door.
    things = (count or 0) + (1 if clause else 0)
    lead = ", and ".join(clauses)
    lead = lead[0].upper() + lead[1:]
    return "%s. %s" % (lead, CATCH_UP_QUESTION.format(
        it="them" if things > 1 else "it"))
