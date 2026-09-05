"""WHOSE turn this is -- one reading, and every path that reads HIS data
asks it the same question.

THE TWO HALVES OF ONE ANSWER
    The owner gate (jarvis/gate.py) attributes a VOICE turn to a person.
    That attribution used to be written straight into ``brain`` as module
    state by ``app._gate_admits`` and then simply stayed there: after one
    admitted guest turn, every typed / CLI / phone / Discord turn of
    Hunter's, and every proactive ``brain.chat`` (the first-wake briefing),
    ran scoped as the guest -- offered the time and the weather, refused
    his own briefing with "That one's Hunter's, Mara." (round-2 review,
    09-04). And commander's Tier-1 handlers never asked at all: a known
    person got his next class, his to-dos, his held notifications, his
    memory of who his doctor is, all without a model and without a scope.

    So the attribution now lives HERE, and it is PER TURN:

      * a voice turn sets it (``set_addressee``) with a stamp, and it
        EXPIRES after ``honorific.ADDRESSEE_TTL`` -- the same two minutes
        the spoken honorific gives an attribution, so the prompt, the tool
        scope, the Tier-1 scope and the spoken "ma'am" all stop believing
        in the same guest at the same moment;
      * every NON-VOICE dispatch and every PROACTIVE call is the OWNER'S:
        ``app._dispatch`` clears it for any source the gate does not judge,
        and the briefing clears it before it asks;
      * each consumer reads it ONCE per turn -- ``brain._chat_sync`` at the
        top, ``Commander.handle`` at the top -- and carries that reading,
        so a gate flip mid-turn cannot give one person's prompt with
        another's scope.

A READING IS A VALUE, NOT A LOOK-UP (round-3 review, 09-05). The module
below is process-global mutable state, and round 3 measured what that
costs when the write and the read sit either side of a WAIT: the writers
(``app._gate_admits``, ``app._the_turn_is_his``) run outside
``Commander._turn_lock`` and the reader ran inside it, after an unbounded
queue on that lock. With a third turn holding the lock 5 ms, 200/200
trials answered a GUEST from his notes, and 198/200 refused HIM his own
notes. The control -- the same contention with nobody flipping the scope
-- leaked 0/200, so the contention was never the bug: the unsynchronised
read-after-wait was.

So the writers now RETURN the reading they install, and every consumer
takes that value as an argument (``Commander.handle(addressee=...)``,
``app._debrief_reply``, ``app._after_dispatch``, ``brain.chat``). The
module state remains, because the honorific swap and the prompt cache
still ask it ambiently, but nothing that guards HIS DATA looks it up
after a wait any more. ``_LOCK`` makes each snapshot consistent; carrying
the value is what makes it correct.

DEFAULT-DENY, IN ONE PLACE. ``owner_only`` returns "" for him and the
authored refusal for anybody else. There is no allow-list of HIS things
to keep up to date: the question is "is this turn his", and anything a
caller would not hand a stranger asks it. What a known person MAY have is
the short list in ``brain.KNOWN_TOOLS`` and ``commander._handle_known``.

PURE ENOUGH TO TEST WITH A NUMBER. ``now`` is injectable everywhere; the
module holds one small dict and touches no app, no config, no file.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

from jarvis.honorific import ADDRESSEE_TTL, SIR_DEFAULT
from jarvis.logs import get_logger

log = get_logger("scope")

# The owner: no name, and the address this codebase says on every line.
OWNER: Tuple[str, str] = ("", SIR_DEFAULT)

# (display name, honorific, monotonic stamp). An empty name is the owner.
_TURN = {"name": "", "honorific": SIR_DEFAULT, "since": float("-inf")}
# Writers run on the gate's thread and readers on the commander's; without
# this a reader could see a half-written attribution (the name swapped, the
# stamp not yet). It does NOT make a read-after-wait safe -- only carrying
# the value does that -- it makes each snapshot self-consistent.
_LOCK = threading.RLock()

# Spoken when a turn that is not his reaches for something of his: a tool
# the model was not offered, a commander short-cut, a Tier-1 handler that
# reads his calendar. Authored, so it ends the turn; nothing of his is
# rendered. It names the person -- the gate greeted them by name, so a
# refusal that says the name back is a refusal and not a snub -- and
# carries no honorific, so there is nothing for the swap to get wrong.
HIS_LINE = ("That one's Hunter's, {name}. I can give you the time and the "
            "weather.")


def _now(now: Optional[float]) -> float:
    return time.monotonic() if now is None else float(now)


def set_addressee(name: str = "", honorific: str = SIR_DEFAULT,
                  now: Optional[float] = None) -> Tuple[str, str]:
    """WHO the turn is for. Empty name = the owner, and the stamp is
    irrelevant. Called from ``app._gate_admits`` with what the gate said,
    and from ``app._dispatch`` / the proactive callers with nothing.

    RETURNS THE READING IT INSTALLS, so the caller that knows whose turn
    this is never has to come back and ask. That return value is the
    whole fix for round-3 blockers 1 and 2: the attributor takes the
    reading at the instant it attributes and hands it down the call chain
    as an argument, instead of every consumer re-reading this dict after
    an unbounded wait on somebody else's lock.
    """
    who = str(name or "")
    hon = str(honorific if honorific is not None else SIR_DEFAULT)
    with _LOCK:
        _TURN["name"] = who
        _TURN["honorific"] = hon
        _TURN["since"] = _now(now)
    return (who, hon) if who else OWNER


def clear_addressee() -> Tuple[str, str]:
    """The owner. What every non-voice path and every proactive call
    says before it asks the model or a handler for anything -- and it
    hands back ``OWNER`` so the caller carries that as its reading."""
    return set_addressee(*OWNER)


def addressee(now: Optional[float] = None) -> Tuple[str, str]:
    """``(display name, honorific)`` for THIS moment. ``("", "sir")`` is
    the owner -- and so is a guest attribution older than the TTL: the
    room does not stay hers because she spoke once at noon.

    AMBIENT, and therefore the fallback only. A caller guarding his data
    must be handed the turn's reading, not call this after a wait.
    """
    with _LOCK:
        name = _TURN["name"]
        hon = _TURN["honorific"]
        since = _TURN["since"]
    if not name:
        return OWNER
    if (_now(now) - since) > float(ADDRESSEE_TTL):
        return OWNER
    return name, hon


def reading(addressee_in=None, now: Optional[float] = None) -> Tuple[str, str]:
    """THE ONE WAY A CONSUMER GETS ITS READING.

    ``addressee_in`` is what the attributor passed down for THIS turn --
    a ``(name, honorific)`` pair. When it is None the caller had nobody
    to tell it (a stand-in commander in a test, an older call site) and
    the ambient state answers, which is the pre-round-3 behaviour and is
    only safe because it is taken before any wait.
    """
    if addressee_in is None:
        return addressee(now)
    pair = tuple(addressee_in)
    return (str(pair[0] or ""), str(pair[1] or SIR_DEFAULT)) \
        if pair[0] else OWNER


def is_owner(now: Optional[float] = None) -> bool:
    return not addressee(now)[0]


def refusal(who: str) -> str:
    """The authored line for ``who``; "there" when the name is somehow
    empty, so the sentence still parses aloud."""
    return HIS_LINE.format(name=who or "there")


def owner_only(what: str = "", who: Optional[str] = None,
               now: Optional[float] = None) -> str:
    """THE ONE QUESTION. "" when the turn is his; else the refusal line.

    ``who`` is the caller's own reading for this turn when it took one
    (Commander.handle, brain._chat_sync); a caller with no reading gets
    the current one. ``what`` is for the log only -- which of his things
    was reached for -- so a wrong refusal can be traced.
    """
    if who is None:
        who = addressee(now)[0]
    if not who:
        return ""
    log.info("scope: %s is %s's; refused for %s", what or "that", "Hunter",
             who)
    return refusal(who)
