"""Pick a speaker-verification threshold from measured scores.

`SpeakerVerifier.threshold` defaults to 0.40 (`jarvis/speaker.py:40`). That is
a starting guess, not a measurement: ECAPA cosine similarity shifts with the
microphone, the room and how far away you sit, so the only honest way to set
it is to score real clips of you and real clips of the room, then look at where
the two populations actually separate.

Pure arithmetic on purpose -- no audio, no model, no I/O -- so the decision
rule is unit-testable and the CLI (`scripts/tune_speaker_threshold.py`) stays a
thin shell over it.

The accept rule mirrors `jarvis/speaker.py:266` exactly::

    is_match = score >= threshold
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SweepRow:
    """Outcome of applying one candidate threshold to the measured scores."""
    threshold: float
    far: float              # fraction of room clips wrongly ACCEPTED (0..1)
    frr: float              # fraction of your clips wrongly REJECTED (0..1)
    accepted_room: int
    rejected_me: int


@dataclass(frozen=True)
class Recommendation:
    threshold: float
    far: float
    frr: float
    clean_separation: bool  # every room clip scored below every clip of you
    reason: str


def sweep(me_scores: Sequence[float],
          room_scores: Sequence[float],
          thresholds: Iterable[float]) -> list[SweepRow]:
    """Score every candidate threshold against both populations.

    Raises ValueError when either class is empty: a threshold tuned against
    only one of them is not tuned at all, and silently returning something
    plausible would be worse than failing.
    """
    me = list(me_scores)
    room = list(room_scores)
    if not me:
        raise ValueError("no clips of you to tune against")
    if not room:
        raise ValueError("no room/background clips to tune against")

    rows = []
    for t in thresholds:
        rejected_me = sum(1 for s in me if s < t)
        accepted_room = sum(1 for s in room if s >= t)
        rows.append(SweepRow(
            threshold=t,
            far=accepted_room / len(room),
            frr=rejected_me / len(me),
            accepted_room=accepted_room,
            rejected_me=rejected_me,
        ))
    return rows


def recommend(rows: Sequence[SweepRow], max_far: float = 0.0) -> Recommendation:
    """Choose a threshold: keep the room out, then be as forgiving as possible.

    Among thresholds whose false-accept rate is within `max_far`, take the one
    with the lowest false-reject rate, breaking ties toward the LOWEST
    threshold. Tightening beyond what the data requires buys nothing and
    rejects the user on an off day -- a hoarse morning, a different chair.

    When nothing meets `max_far` the populations overlap; rather than pretend,
    return the best total-error compromise and say so in `reason`.
    """
    if not rows:
        raise ValueError("nothing to recommend from")

    ordered = sorted(rows, key=lambda r: r.threshold)
    clean = _has_clean_gap(ordered)

    # A clean gap leaves many thresholds tied at 0% / 0%, and the table cannot
    # break that tie -- so margin does. Sit in the MIDDLE of the usable band:
    # the bottom edge keeps no rejection margin (and the negatives that define
    # the gap are usually easy ones -- a TV through speakers scores near zero,
    # where a real second person would score far higher), while the top edge
    # rejects the user on a hoarse morning or from a different chair.
    band = [r for r in ordered if r.far == 0.0 and r.frr == 0.0]
    if band:
        best = band[len(band) // 2]
        return Recommendation(
            best.threshold, best.far, best.frr, True,
            f"clean separation: every room clip scored below every clip of you. "
            f"{len(band)} thresholds tie at 0%/0% ({band[0].threshold:.2f}-"
            f"{band[-1].threshold:.2f}); this is the midpoint, which keeps the "
            f"most margin on both sides")

    eligible = [r for r in ordered if r.far <= max_far]
    if eligible:
        best = min(eligible, key=lambda r: (r.frr, r.threshold))
        if clean:                       # unreachable: handled by `band` above
            reason = "clean separation"
        else:
            # A FAR-clean threshold can still exist when the populations
            # overlap -- it just buys that cleanliness by rejecting the user.
            # Say so, or the number looks better than it is.
            reason = (f"populations OVERLAP -- a room clip scored above one of "
                      f"your own takes. Holding FAR at {max_far:.0%} costs "
                      f"{best.frr:.0%} false rejects of you; more takes, or "
                      f"moving the mic, would widen the gap")
        return Recommendation(best.threshold, best.far, best.frr, clean, reason)

    best = min(ordered, key=lambda r: (r.far + r.frr, r.threshold))
    return Recommendation(
        best.threshold, best.far, best.frr, clean,
        "populations OVERLAP -- a room voice scores as high as you do, so no "
        f"threshold is clean. Best compromise admits {best.far:.0%} of the room "
        f"and rejects {best.frr:.0%} of you; collect more takes or move the mic.")


def _has_clean_gap(rows: Sequence[SweepRow]) -> bool:
    """True when some swept threshold achieves both FAR and FRR of zero."""
    return any(r.far == 0.0 and r.frr == 0.0 for r in rows)


def format_table(rows: Sequence[SweepRow]) -> str:
    """Human-readable sweep, for the CLI."""
    out = [f"{'thresh':>7}  {'FAR':>6}  {'FRR':>6}   room in / you out",
           "-" * 46]
    for r in sorted(rows, key=lambda x: x.threshold):
        out.append(f"{r.threshold:>7.2f}  {r.far:>6.1%}  {r.frr:>6.1%}"
                   f"   {r.accepted_room:>3d}      / {r.rejected_me:>3d}")
    return "\n".join(out)
