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

DEFAULT-DENY, IN ONE PLACE. ``owner_only`` returns "" for him and the
authored refusal for anybody else. There is no allow-list of HIS things
to keep up to date: the question is "is this turn his", and anything a
caller would not hand a stranger asks it. What a known person MAY have is
the short list in ``brain.KNOWN_TOOLS`` and ``commander._handle_known``.

PURE ENOUGH TO TEST WITH A NUMBER. ``now`` is injectable everywhere; the
module holds one small dict and touches no app, no config, no file.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

from jarvis.honorific import ADDRESSEE_TTL, SIR_DEFAULT
from jarvis.logs import get_logger

log = get_logger("scope")

# The owner: no name, and the address this codebase says on every line.
OWNER: Tuple[str, str] = ("", SIR_DEFAULT)

# (display name, honorific, monotonic stamp). An empty name is the owner.
_TURN = {"name": "", "honorific": SIR_DEFAULT, "since": float("-inf")}

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
                  now: Optional[float] = None) -> None:
    """WHO the turn is for. Empty name = the owner, and the stamp is
    irrelevant. Called from ``app._gate_admits`` with what the gate said,
    and from ``app._dispatch`` / the proactive callers with nothing."""
    _TURN["name"] = str(name or "")
    _TURN["honorific"] = str(honorific if honorific is not None
                             else SIR_DEFAULT)
    _TURN["since"] = _now(now)


def clear_addressee() -> None:
    """The owner. What every non-voice path and every proactive call
    says before it asks the model or a handler for anything."""
    set_addressee(*OWNER)


def addressee(now: Optional[float] = None) -> Tuple[str, str]:
    """``(display name, honorific)`` for THIS moment. ``("", "sir")`` is
    the owner -- and so is a guest attribution older than the TTL: the
    room does not stay hers because she spoke once at noon."""
    name = _TURN["name"]
    if not name:
        return OWNER
    if (_now(now) - _TURN["since"]) > float(ADDRESSEE_TTL):
        return OWNER
    return name, _TURN["honorific"]


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
