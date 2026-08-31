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
as callables and returns the steps it actually performed.

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
