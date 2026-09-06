"""Three legs vote on one question: is Hunter in the flat?

HIS DESIGN, verbatim, 2026-09-05:

    "Ok this should be the logic then. If phone isn't on WiFi and camera
     doesn't see but sensor is on, assume not there
     If WiFi connected and no camera and no sensor assume bedroom
     I think that covers the three main cases?
     Also if you see office sensor then kitchen then phone disconnect
     assume he left the building"

WHY THREE LEGS AND NOT ONE. On 2026-09-05 at 20:43:13 he walked in and
Jarvis said nothing. The office radar had read occupied continuously since
at least 19:07:11 with the flat empty; ``HouseView`` calls the house
occupied if ANY room is, so one latched room pinned the whole house
"home"; the sentinel therefore never said "away"; and ``door_arrival``
needs away and only away. One leg wedged and the greeting was gone. Three
legs that can outvote each other is the right shape, and it is his.

THE TWO PRINCIPLES the filled cells obey, both of them his:

  P1  A LEG THAT NAMES HIM WINS OUTRIGHT. The camera is the only leg that
      can identify; the radar has one bit and no name, the phone tracks a
      device and not a man. So a camera that says "hunter 0.71" ends the
      vote, in every sensor state, whatever the other two say.
  P2  SENSOR ON IS NEVER SUFFICIENT ON ITS OWN. That is exactly tonight.
      The radar is the only leg that can LATCH, so it may confirm a
      verdict and never carry one alone.

THE ASYMMETRY THAT MATTERS MOST, and it is why this module is not simply
majority voting: **a leg that CANNOT answer votes UNKNOWN, never "no".**

  * A camera inside its curfew, switched off, or -- on his box today --
    never wired to a feed at all, has not "failed to see him". It did not
    look. ``CAM_BLIND`` is not ``CAM_LOOKED``, and only ``CAM_LOOKED``
    can reach his rule 1. If silence counted as "doesn't see", rule 1
    would decide he is out while he sits in the office at ten at night,
    and a FALSE AWAY IS WORSE THAN TONIGHT'S FALSE HOME -- it fires a
    greeting at a man already sitting down, and it makes quiet.py answer
    "you're out" to a man in the room.
  * An unreachable radar is evidence of nothing. ``ROOMS_UNREACHABLE``
    falls through to the phone, which is byte for byte what this box did
    before any radar was bought.
  * A phone inside ``away_after_min`` is UNKNOWN, not absent. presence.py:
    "away_after_min is the grace before out -- iPhones nap off Wi-Fi."

EVERY VERDICT CARRIES ITS REASON. ``Verdict.reason`` is a sentence, and
the caller logs it at INFO. The next time he asks "why did he not greet
me", the answer is already written down -- which is the whole complaint
about tonight: the day's log held seven "presence: home" lines, zero
"away" lines, and not one line saying why.

WHAT THIS MODULE IS NOT. It owns no thread, no socket, no clock and no
config reader. It is a pure function over three enums plus two hints, so
every one of the 27 cells is testable without a network, and
tests/test_presencevote.py writes the table out in full rather than
deriving it. Cell 6 alone takes two more inputs -- an age in seconds and
the mic's word -- and ``cell6`` below is the whole of that, so the table
stays 27 rows a person can read against his design.

A NOTE ON "AWAY" AND WHO IT IS ABOUT. This answers "is HUNTER in the
flat", not "is the flat empty". A guest in the flat with him out is
correctly AWAY here (roomfabric.others() already refuses to count bodies:
an LD2410 has one bit and sees through plasterboard). Nothing downstream
may read AWAY as "the flat is empty" -- a security or heating action hung
off this leg would be wrong, and that is a boundary, not a bug.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from jarvis.logs import get_logger

log = get_logger("presencevote")

# ------------------------------------------------------------ the legs
PHONE_YES = "phone-yes"          # answers on the Wi-Fi
PHONE_NO = "phone-no"            # asked, no answer, past the grace
PHONE_UNKNOWN = "phone-unknown"  # could not ask, or still inside the grace

CAM_SAW = "cam-saw"              # named him: a label and a score
CAM_LOOKED = "cam-looked"        # looked at a live feed and saw nobody
CAM_BLIND = "cam-blind"          # could not look: off, curfew, no feed

ROOMS_ON = "rooms-on"            # some room reads occupied
ROOMS_CLEAR = "rooms-clear"      # every configured room answered, all clear
ROOMS_UNREACHABLE = "rooms-unreachable"   # no room answered

# THE FOURTH LEG, and it only ever speaks inside cell 6. It is not part of
# the 27-cell table and must not become part of it: the table is his three
# legs in his order, and adding a fourth would make it 81 cells nobody can
# read against the design. This leg exists because cell 6 is the ONE cell
# whose three legs are identical in two situations with opposite truths --
# a latched radar with him out, and him sitting still at his desk with his
# phone asleep -- and a spoken turn is the one piece of evidence that can
# tell them apart. A radar can latch and a phone can nap; a microphone
# cannot hallucinate a sentence.
MIC_HEARD = "mic-heard"          # a turn in the ledger inside the window
MIC_SILENT = "mic-silent"        # the ledger was read; the last turn is older
MIC_UNKNOWN = "mic-unknown"      # no ledger, unreadable, nonsense

_PHONES = (PHONE_YES, PHONE_NO, PHONE_UNKNOWN)
_CAMERAS = (CAM_SAW, CAM_LOOKED, CAM_BLIND)
_ROOMS = (ROOMS_ON, ROOMS_CLEAR, ROOMS_UNREACHABLE)
_MICS = (MIC_HEARD, MIC_SILENT, MIC_UNKNOWN)

# ------------------------------------------------- the recency window
# HOW FRESH AN AGREEMENT HAS TO BE for cell 6 to keep saying "he is here".
#
# DERIVED, NOT MEASURED, and the derivation is the whole justification.
# He was asked how long his short trips are -- the shop, a walk, the bins
# -- and has not answered, so there is no ground truth to fit. What there
# IS is a number he already owns: ``presence.away_after_min`` is 12
# minutes, and it exists for exactly one reason -- an iPhone drops off
# Wi-Fi power-save for minutes at a time and 12 minutes is the nap length
# the box is willing to forgive. A recency window SHORTER than that would
# call a phone's silence stale faster than the sentinel is willing to call
# the same phone absent, which is incoherent, and it would manufacture
# false aways at the desk. So the floor is 12 minutes. The stamp itself is
# up to one home poll (60 s) old on top, and a minute of cadence slack is
# cheap, so: 12 + 1 + 2 = 15.
#
# WHAT THE NUMBER BUYS AND COSTS, both MEASURED on the sentinel end to end
# (scripts/presence_cliff.py; tests/test_presence_recency.py pins them):
#   * a latched radar plus a trip now reaches "away" at 26 minutes -- the
#     window, then the 12-minute grace, less the one poll still fresh --
#     instead of 56, so a half-hour errand is greeted where it was not.
#   * he must be out for 26 minutes to be greeted at all. A five-minute
#     bin trip earns silence, which is the right answer anyway.
#   * THE RESIDUAL, same cell, opposite truth: at his desk with the phone
#     asleep, the camera dark and NO spoken turn for 26 minutes, he is
#     called away. Any turn inside that resets it (the mic leg below); a
#     live camera ends it by his rule 1. The first cut did the same at 13
#     minutes on an unstamped run; jarvis-v3 never does while a room is on.
# ONE EDIT CHANGES IT: presence.corroboration_recency_min.
DEFAULT_RECENCY_S = 15 * 60.0
RECENCY_S_PROVENANCE = (
    "DERIVED, not measured: he has not said how long his short trips are. "
    "Floored on presence.away_after_min (12 min), which is his own tolerance "
    "for an iPhone napping off Wi-Fi -- a shorter window would call a nap "
    "stale faster than the sentinel will call the phone absent -- plus one "
    "60 s home poll for stamp granularity and two minutes of cadence slack. "
    "Revise it from his own answer: presence.corroboration_recency_min.")

# The mic window is NOT a new number. arrival.departure_ready already vetoes
# a departure on ``presence.departure_mic_silence_min`` (10 min) read off the
# same turn ledger, and it is the same physical question -- how long does a
# spoken turn go on proving he is in the flat. One number, one meaning.
DEFAULT_MIC_WINDOW_S = 10 * 60.0

# --------------------------------------------------------- the verdicts
HOME = "home"
AWAY = "away"
BED = "bed"          # a HOME sub-state: in the flat, in a room with no sensor
UNKNOWN = "unknown"

# The bedroom hint goes stale. A room that last saw him before lunch says
# nothing about where he is now, so past this the split is not attempted
# and his bare rule 2 answers instead.
HINT_MAX_AGE_S = 1800.0          # 30 minutes


@dataclass(frozen=True)
class Verdict:
    """One answer, the cell it came from, and why -- in a sentence."""

    state: str = UNKNOWN
    reason: str = ""
    cell: int = 0
    hold: bool = False           # keep the previous verdict; this is not news

    @property
    def home(self) -> bool:
        """True unless he has been ESTABLISHED to be out.

        Matches ``PresenceSentinel.is_home()``: unknown reads as home so a
        misconfigured leg can never mute him.
        """
        return self.state != AWAY

    @property
    def strictly_away(self) -> bool:
        """The gate ``arrival.door_arrival`` needs, and only this."""
        return self.state == AWAY

    @property
    def greetable(self) -> bool:
        """May a return from THIS state earn a greeting? Only from away.

        BED and HOME are both "he is in the flat", so neither arms the
        door: he never left, and there is nothing to welcome him back
        from.
        """
        return self.state == AWAY


def _slug(value) -> str:
    """A room name as roomfabric and arrival both slug it."""
    text = " ".join(str(value or "").split()).lower()
    return "".join(ch if (ch.isalnum() or ch in " -_") else ""
                   for ch in text).strip()


# ====================================================================
# THE LEG BUILDERS -- turning what the box actually has into three enums
# ====================================================================
def phone_leg(*, answer: Optional[bool], unseen_s: float,
              grace_s: float) -> str:
    """The phone's vote.

    ``answer`` is ``presence.probe``'s: True seen, False asked-and-silent,
    None could-not-ask. **A False inside the grace is UNKNOWN, not NO.**
    An iPhone drops off Wi-Fi power-save for minutes at a time and a
    napping radio is not a departure -- that grace is the reason
    ``away_after_min`` exists and it is honoured here rather than
    re-derived.

    ``answer=None`` is the honest unknown the probe cannot yet send: today
    it returns a plain bool and flattens "the Spark's own Wi-Fi dropped"
    into "he left". See the report's phoneSetup. This function is ready
    for the fix on the day the probe learns to say None.
    """
    if answer is None:
        return PHONE_UNKNOWN
    if answer:
        return PHONE_YES
    try:
        unseen, grace = float(unseen_s), float(grace_s)
    except (TypeError, ValueError):
        return PHONE_UNKNOWN
    return PHONE_NO if unseen >= grace else PHONE_UNKNOWN


def camera_leg(*, identity: str = "", faces: Optional[int] = None,
               live: bool = False) -> str:
    """The camera's vote -- FROM A NAME AND A COUNT, never from a frame.

    Nothing in this module, or in any module it can reach, takes an image.
    ``identity`` is the label the local recogniser already produced and
    ``faces`` is a count. Both are numbers-and-strings by the time they
    arrive here, and they must stay that way.

    ``live`` is the whole point of the leg. A camera that is off, inside
    its curfew, or (on his box today) never attached to
    ``services.camera_feed`` at all is BLIND -- it did not look, so its
    silence is not evidence of an empty room. Only a live feed reporting
    zero faces has actually LOOKED.
    """
    if str(identity or "").strip():
        return CAM_SAW
    if not live or faces is None:
        return CAM_BLIND
    try:
        return CAM_SAW if int(faces) > 0 else CAM_LOOKED
    except (TypeError, ValueError):
        return CAM_BLIND


def rooms_leg(readings, faulted=()) -> str:
    """The radar's vote from ``{room: True | False | None}``.

    CLEAR ONLY WHEN EVERY ROOM ANSWERED, which is ``roomfabric.anywhere``'s
    rule and the right one: with three rooms "empty" is a statement about
    coverage, and a room whose breaker is open means we do not have it.

    ``faulted`` rooms are DROPPED, not counted as clear. A stuck sensor
    removes a vote; it never casts one (jarvis/stuckroom.py). So the
    latched office plus a genuinely clear kitchen reads CLEAR, and the
    latched office alone reads UNREACHABLE -- in both cases the house
    falls through to the phone, which is the honest leg, and which is
    exactly what would have saved tonight.
    """
    drop = {_slug(r) for r in (faulted or ())}
    live = {k: v for k, v in dict(readings or {}).items()
            if _slug(k) not in drop}
    if not live:
        return ROOMS_UNREACHABLE
    if any(v is True for v in live.values()):
        return ROOMS_ON
    if all(v is False for v in live.values()):
        return ROOMS_CLEAR
    return ROOMS_UNREACHABLE


# ====================================================================
# THE TABLE
# ====================================================================
# (rooms, phone, camera) -> (cell, state, reason). Written out in full and
# in his order, because this table is the specification: a reader has to
# be able to check it against the design line by line. Two cells are
# resolved by a function instead of a constant and both are marked.
_R1 = "his rule 1"
_R2 = "his rule 2"
_P1 = "a camera that names him wins outright"
_P2 = "sensor ON never carries a verdict alone"

_CELL6 = "cell6"
_CELL_BED = "bedroom"

_TABLE = {
    # ---------------------------------------------------- SENSOR = ON
    (ROOMS_ON, PHONE_YES, CAM_SAW):
        (1, HOME, "all three legs agree he is here"),
    (ROOMS_ON, PHONE_YES, CAM_LOOKED):
        (2, HOME, "his phone answers and a room sees somebody; the lens "
                  "looked elsewhere, and faces 0 is not an empty flat"),
    (ROOMS_ON, PHONE_YES, CAM_BLIND):
        (3, HOME, "his phone answers and a room sees somebody; the camera "
                  "could not look"),
    (ROOMS_ON, PHONE_NO, CAM_SAW):
        (4, HOME, "the lens named him, so %s -- the radio napped or is off"
                  % _P1),
    (ROOMS_ON, PHONE_NO, CAM_LOOKED):
        (5, AWAY, "%s: his phone did not answer past the grace and the "
                  "camera looked and saw nobody; only the radar dissents "
                  "and it is the one leg that can latch" % _R1),
    (ROOMS_ON, PHONE_NO, CAM_BLIND): (6, _CELL6, ""),
    (ROOMS_ON, PHONE_UNKNOWN, CAM_SAW):
        (7, HOME, "the lens named him, so %s" % _P1),
    (ROOMS_ON, PHONE_UNKNOWN, CAM_LOOKED):
        (8, HOME, "a room sees somebody, the phone could not be asked, and "
                  "no leg that can name him disagrees"),
    (ROOMS_ON, PHONE_UNKNOWN, CAM_BLIND):
        (9, UNKNOWN, "only the radar has an opinion and it is the leg that "
                     "can latch, so %s; holding the last verdict rather "
                     "than deriving one" % _P2),
    # --------------------------------------------- SENSOR = ALL CLEAR
    (ROOMS_CLEAR, PHONE_YES, CAM_SAW):
        (10, HOME, "the lens named him, so %s -- the radar missed a still "
                   "body, which at his measured 303 cm in a band ending at "
                   "375 cm is ordinary" % _P1),
    (ROOMS_CLEAR, PHONE_YES, CAM_LOOKED): (11, _CELL_BED, ""),
    (ROOMS_CLEAR, PHONE_YES, CAM_BLIND): (12, _CELL_BED, ""),
    (ROOMS_CLEAR, PHONE_NO, CAM_SAW):
        (13, HOME, "the lens named him, so %s" % _P1),
    (ROOMS_CLEAR, PHONE_NO, CAM_LOOKED):
        (14, AWAY, "all three legs answered and none of them sees him; "
                   "none of the three can latch"),
    (ROOMS_CLEAR, PHONE_NO, CAM_BLIND):
        (15, AWAY, "his phone did not answer past the grace and every room "
                   "answered clear; the camera could not look, but the two "
                   "legs that did agree and neither can latch"),
    (ROOMS_CLEAR, PHONE_UNKNOWN, CAM_SAW):
        (16, HOME, "the lens named him, so %s" % _P1),
    (ROOMS_CLEAR, PHONE_UNKNOWN, CAM_LOOKED):
        (17, UNKNOWN, "only the lens says nobody and it cannot see the "
                      "bedroom, which has no sensor; never away on that"),
    (ROOMS_CLEAR, PHONE_UNKNOWN, CAM_BLIND):
        (18, UNKNOWN, "only the rooms answered and they are blind to the "
                      "bedroom, which has no sensor; never away on that"),
    # ------------------------------------------- SENSOR = UNREACHABLE
    (ROOMS_UNREACHABLE, PHONE_YES, CAM_SAW):
        (19, HOME, "the lens named him, so %s" % _P1),
    (ROOMS_UNREACHABLE, PHONE_YES, CAM_LOOKED):
        (20, HOME, "his phone answers; no room could be read"),
    (ROOMS_UNREACHABLE, PHONE_YES, CAM_BLIND):
        (21, HOME, "his phone answers and nothing else can be asked -- the "
                   "phone-only verdict this box gave before any radar"),
    (ROOMS_UNREACHABLE, PHONE_NO, CAM_SAW):
        (22, HOME, "the lens named him, so %s" % _P1),
    (ROOMS_UNREACHABLE, PHONE_NO, CAM_LOOKED):
        (23, AWAY, "his phone did not answer past the grace and the camera "
                   "looked and saw nobody; neither leg can latch"),
    (ROOMS_UNREACHABLE, PHONE_NO, CAM_BLIND):
        (24, AWAY, "phone-only away past the grace, no room readable -- the "
                   "leg presence.py was built on"),
    (ROOMS_UNREACHABLE, PHONE_UNKNOWN, CAM_SAW):
        (25, HOME, "the lens named him, so %s" % _P1),
    (ROOMS_UNREACHABLE, PHONE_UNKNOWN, CAM_LOOKED):
        (26, UNKNOWN, "only the lens has an opinion and it is blind to the "
                      "bedroom; never away on that"),
    (ROOMS_UNREACHABLE, PHONE_UNKNOWN, CAM_BLIND):
        (27, UNKNOWN, "nothing can see and nothing can be asked"),
}


def bedroom_split(*, last_room: str = "", age_s: Optional[float] = None,
                  door_room: str = "kitchen",
                  desk_room: str = "office") -> tuple:
    """Cells 11 and 12: phone on the Wi-Fi, every room clear. Where is he?

    HIS RULE 2 says bedroom, and it is the right instinct. The last room
    that saw anybody makes it materially better, because his flat is a
    corridor -- front door, kitchen, office, with the BEDROOM off the
    KITCHEN and no sensor in it -- so "phone home, all rooms clear" is not
    one situation but two, with OPPOSITE correct answers:

      last room = KITCHEN -> he walked out of the kitchen, and the only
        unsensored places reachable from there are the bedroom and the
        bathroom. BEDROOM is well supported. His rule is right.

      last room = OFFICE -> he did NOT pass through the kitchen. The
        office is only reachable through the kitchen, so he cannot have
        got to the bedroom without lighting the kitchen sensor first. The
        honest reading is the OPPOSITE of his rule: he is still in the
        office and the radar dropped a still body. Measured 2026-09-05:
        his chest at the desk reads 303 cm still, and the max still gate
        is 4 -- a band ending at 375 cm. He sits in the LAST active gate.
        A still-target drop at that range is ordinary, not a fault.

    That branch answers HOME rather than BED, and it must NOT re-arm the
    door greeting, because he never left.

    THE SOURCE IS ``Room.last_true``, NOT ``fabric.where().room``. The
    active room is dropped after ``rooms_stale_after_s`` (90 s) measured
    from its last_true, and with the office's 10 s absence delay on top
    the hint would survive about 100 seconds after he leaves a room. Every
    bedroom trip that matters is longer than that, so ``where()`` would
    hand back "" exactly when it is needed. ``last_true`` is per room,
    maintained on every tick, never cleared while the process lives, and
    unbounded in age.

    IT CANNOT TELL BEDROOM FROM BATHROOM and does not try. Both are off
    the kitchen, both unsensored, and for every consumer in the tree "he
    is in the flat, in a room I cannot see" is the same answer.
    """
    key = _slug(last_room)
    if not key or age_s is None or float(age_s) > HINT_MAX_AGE_S:
        return (BED, "%s: his phone is on the Wi-Fi and no room sees him, "
                     "and no room hint is fresh enough to place him"
                % _R2)
    if key == _slug(desk_room):
        return (HOME, "his phone is on the Wi-Fi, every room is clear, and "
                      "the last room to see anybody was the office -- which "
                      "is only reachable through the kitchen, and the "
                      "kitchen never lit, so he never left the office and "
                      "the radar dropped a still body")
    if key == _slug(door_room):
        return (BED, "%s: his phone is on the Wi-Fi, every room is clear, "
                     "and the kitchen saw him last -- the bedroom is off "
                     "the kitchen and has no sensor" % _R2)
    return (BED, "%s: his phone is on the Wi-Fi and no room sees him" % _R2)


def mic_leg(*, s_ago, window_s: float = DEFAULT_MIC_WINDOW_S) -> str:
    """The mic's vote -- FROM ONE NUMBER OFF THE TURN LEDGER, never audio.

    ``s_ago`` is ``TurnLedger.idle_s()``: seconds since the microphone last
    completed a turn, or None when it never has (a fresh process, an empty
    box). Nothing here, or in anything it reaches, opens a capture device
    or touches a sample -- the leg takes a float and returns a word.

    A turn inside ``window_s`` is HEARD; older is SILENT. No ledger,
    nonsense, or a negative age is UNKNOWN -- the camera's rule again: a
    leg that could not be READ has not heard nothing.
    """
    try:
        age, window = float(s_ago), float(window_s)
    except (TypeError, ValueError):
        return MIC_UNKNOWN
    if age != age or window != window or age < 0.0:      # NaN, negative
        return MIC_UNKNOWN
    return MIC_HEARD if age < window else MIC_SILENT


def _mins(seconds) -> str:
    """The number IN the sentence: '44 min', '30 s'. A verdict that says
    'recently' while the code asks 'ever' is the bug this file had."""
    try:
        s = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "an unknown time"
    if s == float("inf"):
        return "for ever"
    return ("%d s" % int(round(s))) if s < 60.0 else ("%d min" % int(s // 60.0))


def cell6(*, agreed_s_ago, recency_s: float = DEFAULT_RECENCY_S,
          mic: str = MIC_UNKNOWN, mic_s_ago=None) -> tuple:
    """Cell 6: a room reads occupied, his phone did not answer past the
    grace, and the camera could not look. Returns ``(state, reason)``.

    THE CELL 2026-09-05 LANDED IN, and the one his three rules cannot reach.
    Rule 1 needs a camera that LOOKED; a camera that could not look is not
    evidence of an empty room, so R1's precondition was never met and the
    rules fall silent -- which is that night, again.

    It is also the NAPPING-PHONE-AT-THE-DESK cell, where the truth is the
    opposite: he is sitting still in the office with his phone asleep. The
    three legs are IDENTICAL in both situations. Only history separates
    them, and the first cut asked history the wrong question -- "has
    anything EVER agreed with this run?" -- while printing "recently". A
    radar that latches has almost always seen a real body first, so "ever"
    is yes for the rest of the run and a shop trip reproduces 2026-09-05.

    TWO CLOCKS, AND THE ASYMMETRY IS THE DESIGN:

      * ``agreed_s_ago`` -- seconds since anything independent (phone,
        camera, mic) last agreed with the occupied run, or since the run
        began if nothing has yet. Fresh means inside ``recency_s``.
      * ``mic`` -- the turn ledger's word, on its own window.

    HOME when EITHER clock is fresh. AWAY only when BOTH have run out. A
    false away greets him mid-sentence and can hand a guest his turn; a
    false home merely delays a greeting. So away must be the harder verdict
    to reach, and it is: of the six (agreement x mic) states, two are away.

    ``agreed_s_ago=None`` -- no run on the books at all -- is HOME: absence
    of history is not evidence of a fault.
    """
    try:
        window = float(recency_s)
    except (TypeError, ValueError):
        window = DEFAULT_RECENCY_S
    if window != window or window <= 0.0:
        window = DEFAULT_RECENCY_S
    win = _mins(window)
    if mic == MIC_HEARD:
        when = ("%s ago" % _mins(mic_s_ago)) if mic_s_ago is not None \
            else "inside its window"
        return (HOME, "a room sees somebody and the mic heard him %s, so he "
                      "is in the flat whatever his phone's radio is doing"
                % when)
    if agreed_s_ago is None:
        return (HOME, "a room sees somebody and its run has no history yet "
                      "to hold against it (%s window); absence of history "
                      "is not a fault" % win)
    try:
        age = float(agreed_s_ago)
    except (TypeError, ValueError):
        return (HOME, "a room sees somebody and its history could not be "
                      "read; never away on that")
    if age < window:
        return (HOME, "a room sees somebody and something else agreed with "
                      "that run %s ago, inside the %s window, so the radar "
                      "is telling the truth and his phone is napping"
                % (_mins(age), win))
    silence = ("the mic has been silent past its own window"
               if mic == MIC_SILENT else "the mic ledger could not be read")
    return (AWAY, "a room reads occupied but nothing has agreed with that "
                  "run for %s -- not his phone, not the camera, not the mic "
                  "(%s) -- past the %s window, and the radar is the one leg "
                  "that can latch; %s" % (_mins(age), silence, win, _P2))


def decide(*, phone: str, camera: str, rooms: str,
           agreed_s_ago: Optional[float] = None,
           recency_s: float = DEFAULT_RECENCY_S,
           mic: str = MIC_UNKNOWN, mic_s_ago: Optional[float] = None,
           last_room: str = "", last_room_age_s: Optional[float] = None,
           door_room: str = "kitchen", desk_room: str = "office") -> Verdict:
    """The vote. Never raises; an unrecognised leg value is UNKNOWN.

    ``agreed_s_ago``, ``recency_s``, ``mic`` and ``mic_s_ago`` reach ONE
    cell -- see ``cell6``. The other 26 never read them, and
    tests/test_presencevote.py pins that they cannot leak.
    """
    if phone not in _PHONES or camera not in _CAMERAS or rooms not in _ROOMS:
        return Verdict(UNKNOWN, "a leg reported a value this voter does not "
                                "recognise; answering unknown rather than "
                                "guessing", 0, hold=True)
    cell, state, reason = _TABLE[(rooms, phone, camera)]

    if state == _CELL6:
        # The mic is not one of the three; a bad value there cannot cost
        # the vote, it only silences the fourth leg.
        state, reason = cell6(agreed_s_ago=agreed_s_ago, recency_s=recency_s,
                              mic=(mic if mic in _MICS else MIC_UNKNOWN),
                              mic_s_ago=mic_s_ago)
        return Verdict(state, reason, cell)

    if state == _CELL_BED:
        state, reason = bedroom_split(last_room=last_room,
                                      age_s=last_room_age_s,
                                      door_room=door_room, desk_room=desk_room)
        return Verdict(state, reason, cell)

    # Cell 9 is the only unknown with a leg still claiming occupancy, so it
    # is the only one with anything worth holding on to. The rest have
    # nothing: presence._forget() is the right shape for those.
    return Verdict(state, reason, cell, hold=(state == UNKNOWN and
                                              rooms == ROOMS_ON))


def decide_rooms_only(*, rooms: str, camera: str = CAM_BLIND) -> Verdict:
    """The verdict on a box with NO PHONE LEG CONFIGURED AT ALL.

    NOT his box -- ``presence.phone_ip`` is set on the Spark, so the 27
    cells above are what run for him. This exists because P2 ("sensor ON
    never carries a verdict alone") is a rule about OUTVOTING, and on an
    install with one leg there is nothing to outvote it with. Applying
    cell 9 there would hold "unknown" for ever and presence would simply
    never work.

    So the rooms answer, exactly as ``RoomOrPhone`` always made them
    answer on a sensor-only install: a room seeing somebody is home, every
    room clear is out, no room readable is unknown. The honesty cost is
    stated rather than hidden -- with one leg, a latched radar cannot be
    caught by voting, and only jarvis/stuckroom.py can catch it.
    """
    if camera == CAM_SAW:
        return Verdict(HOME, "the lens named him, so %s" % _P1, 0)
    if rooms == ROOMS_ON:
        return Verdict(HOME, "a room sees somebody and no phone leg is "
                             "configured, so the rooms are the only evidence "
                             "there is -- nothing can outvote them here", 0)
    if rooms == ROOMS_CLEAR:
        return Verdict(AWAY, "every room answered clear and no phone leg is "
                             "configured to fall back to", 0)
    return Verdict(UNKNOWN, "no room could be read and no phone leg is "
                            "configured", 0, hold=True)


# ====================================================================
# HIS DEPARTURE SEQUENCE: office, then kitchen, then the phone drops
# ====================================================================
SEQ_IDLE = ""
SEQ_DESK = "office"       # the office saw him
SEQ_DOOR = "kitchen"      # ... and then the kitchen, soon enough after
SEQ_LEFT = "left"         # ... and then his phone went

# THE TWO WINDOWS, and why these numbers.
#
# SEQ_STEP_S -- office to kitchen. MEASURED: the fabric holds a new room
# 2 s before publishing (rooms_enter_hold_s) and the office's own absence
# delay is 10 s, so a walk from the desk to the kitchen shows up as a room
# change inside ~15 s. 120 s is eight times that, which covers him
# stopping to pick up his keys, and is still far short of "he was in the
# kitchen twenty minutes ago", which is not a walk to the door.
SEQ_STEP_S = 120.0
#
# SEQ_WINDOW_S -- kitchen to the phone dropping. The phone stays on the
# Wi-Fi until he is out of range, and then ``away_after_min`` (12 min =
# 720 s) has to expire before the leg says NO at all. The away poll is
# 10 s. 900 s is the grace plus three minutes of slack for the cadence. If
# his phone has not gone by then he did not leave the building -- he made
# a coffee -- and the sequence expires rather than waiting all evening for
# a drop it can blame on a walk an hour ago.
SEQ_WINDOW_S = 900.0


class DepartureSequence:
    """"office sensor then kitchen then phone disconnect" -- his words.

    AN ORDERED SEQUENCE WITH A WINDOW, not three independent facts. Three
    facts that happen to be true at the same time would fire on him making
    coffee and then his phone napping in his pocket; the ORDER is what
    makes it a departure, because his flat is a corridor and leaving means
    passing the kitchen after the office and then going out of range.

    Pure: no clock, no thread, no config. The caller passes ``at``.

    IT IS SILENT. arrival.py is explicit -- "a valediction spoken to an
    empty room is a notification pretending to be a presence" -- so this
    class has no speak path and ``speaks`` is False for ever. What
    completing the sequence buys is a LOG LINE and an earlier, better
    grounded "away" than the phone grace alone would give: the door watch
    can be re-armed knowing he actually walked out, rather than inferring
    it from a radio that went quiet.
    """

    speaks = False

    def __init__(self, desk_room: str = "office", door_room: str = "kitchen",
                 step_s: float = SEQ_STEP_S, window_s: float = SEQ_WINDOW_S):
        self.desk_room = desk_room
        self.door_room = door_room
        self.step_s = float(step_s)
        self.window_s = float(window_s)
        self.stage = SEQ_IDLE
        self.at = 0.0

    def reset(self) -> None:
        self.stage, self.at = SEQ_IDLE, 0.0

    def home(self, at: float = 0.0) -> None:
        """He is back. The next walk out is a new departure."""
        self.reset()

    def room(self, *, room, at: float) -> bool:
        """One RoomChanged. Returns False always -- a room is never the
        whole sequence -- but advances or breaks it."""
        key, at = _slug(room), float(at)
        if key == _slug(self.desk_room):
            # Back at the desk: whatever the sequence was, he did not
            # leave. This is also the re-arm for the next walk out.
            self.stage, self.at = SEQ_DESK, at
            return False
        if key == _slug(self.door_room):
            if self.stage == SEQ_DESK and at - self.at <= self.step_s:
                self.stage, self.at = SEQ_DOOR, at
            else:
                # The kitchen without the office first, or too slow after
                # it, is a kitchen trip and not a walk to the door.
                self.stage, self.at = SEQ_IDLE, at
            return False
        return False

    def phone_gone(self, at: float) -> bool:
        """His phone stopped answering. True exactly when that completes
        the sequence, meaning he left the building."""
        at = float(at)
        if self.stage != SEQ_DOOR or at - self.at > self.window_s:
            return False
        self.stage, self.at = SEQ_LEFT, at
        return True

    @property
    def left(self) -> bool:
        return self.stage == SEQ_LEFT


def departure_note(seq: "DepartureSequence") -> str:
    """The line the log gets. There is no spoken counterpart, on purpose."""
    if seq.stage == SEQ_LEFT:
        return ("departure: office then %s then his phone dropped, in that "
                "order and inside the window -- he left the building"
                % seq.door_room)
    return "departure: not the sequence (stage %r)" % (seq.stage or "idle")
