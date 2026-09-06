"""Who is being addressed, and therefore which form of address to use.

ONE QUESTION, ASKED IN ONE PLACE. "Sir" or "ma'am" is a property OF A
PERSON, stored in ``people.json`` because Hunter typed it there, and it is
resolved at SPEAK TIME from whoever Jarvis currently believes he is talking
to. It is never a global constant, never derived from a name, and never
guessed.

WHAT THIS MODULE WILL NOT DO
    There is no name list here, no gender table, and no heuristic. A person
    the registry does not hold resolves to "" -- NO form of address at all,
    so Jarvis says "Signed in." rather than picking one and being wrong at
    somebody. Being briefly formless is recoverable; calling a woman "sir"
    in her own kitchen is the thing this whole feature exists to stop.

WHY THERE IS A TIME LIMIT ON THE ANSWER
    ``JarvisApp._gate_who`` is set by the owner gate on every judged turn
    and then simply stays there. Without a limit, one turn in which the
    camera named Mara would make every proactive line for the rest of the
    day -- a reminder at midnight, tomorrow's briefing -- come out addressed
    to her. So an attribution older than ``ADDRESSEE_TTL`` is not an
    addressee any more, and the resolver falls back TO THE OWNER, which is
    the state the whole rest of the codebase is authored for.

PURE AND CLOCK-INJECTED. ``for_addressee`` takes callables and a number; it
touches no app, no config and no file, so it tests with neither.
"""
from __future__ import annotations

from typing import Callable, Optional

from jarvis.identity import HONORIFICS, ROLE_UNKNOWN
from jarvis.logs import get_logger

log = get_logger("honorific")

# How long a gate attribution is still "who I am talking to". Two minutes
# is longer than a follow-up window and far shorter than a session, which
# is the band a turn-to-turn attribution is actually good for.
ADDRESSEE_TTL = 120.0

# What every caller falls back to when there is nothing to resolve from --
# no registry, no attribution, a stale one. It is "sir" because that is
# what this codebase says TODAY, on every one of its authored lines, and a
# fallback that changed behaviour would make the feature unsafe to ship.
SIR_DEFAULT = "sir"


def for_label(registry, label) -> str:
    """The stored form of address for ``label``, or "".

    "" covers three different things ON PURPOSE -- no registry, no such
    person, and a person who chose no form of address -- because the
    behaviour for all three is identical and distinguishing them here would
    invite a caller to treat one of them as "so use sir".
    """
    try:
        person = registry.person(str(label or ""))
    except Exception:  # noqa: BLE001 - a registry that cannot say says nothing
        log.debug("honorific: the registry could not be asked", exc_info=True)
        return ""
    if person is None:
        return ""
    value = str(getattr(person, "honorific", "") or "")
    if value not in HONORIFICS:
        # Storage that says something this build does not know is not a
        # licence to invent one.
        return ""
    return value


def addressee(get_who: Callable, get_when: Callable, owner: str,
              now: float, ttl: float = ADDRESSEE_TTL) -> str:
    """WHOSE label the next spoken line is aimed at.

    Stale, empty, or unreadable -> the owner. Everything in this codebase
    is authored for him, so falling back to him is falling back to the
    behaviour that shipped.
    """
    try:
        who = str(get_who() or "")
        when = float(get_when() or 0.0)
    except Exception:  # noqa: BLE001 - an app that cannot say is his
        return str(owner or "")
    if not who:
        return str(owner or "")
    if not when or (now - when) > float(ttl):
        return str(owner or "")
    return who


def for_addressee(registry, get_who: Callable, get_when: Callable,
                  owner: str, now: float, ttl: float = ADDRESSEE_TTL) -> str:
    """``for_label`` of ``addressee`` -- the one call ``_say`` makes.

    THE OWNER SHORT-CIRCUITS TO "sir" ONLY IF THAT IS WHAT HIS ROW SAYS.
    Until he registers himself there is no row at all, ``for_label``
    returns "", and this returns "sir" from the fallback below -- which is
    the behaviour Jarvis has today, unchanged, and is why nothing moves
    until he runs the enrolment commands.
    """
    who = addressee(get_who, get_when, owner, now, ttl)
    value = for_label(registry, who)
    if value:
        return value
    if not who or who == str(owner or ""):
        # No registry, or an owner who has not been registered yet. This is
        # the state the live app is in TODAY, and "sir" is what it says
        # today; the swap is then a no-op and not one byte moves.
        return SIR_DEFAULT
    return ""


def known_person(registry, label) -> Optional[object]:
    """The row for ``label``, or None. A convenience so callers do not each
    write their own try/except round a registry that may not be usable."""
    try:
        return registry.person(str(label or ""))
    except Exception:  # noqa: BLE001
        return None


def role_of(registry, label) -> str:
    try:
        return registry.role_of(str(label or ""))
    except Exception:  # noqa: BLE001
        return ROLE_UNKNOWN
