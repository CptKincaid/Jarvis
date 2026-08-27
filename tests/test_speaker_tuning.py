"""Threshold sweep for speaker verification.

`SpeakerVerifier.threshold` ships at 0.40 (`jarvis/speaker.py:40`), which is a
guess -- ECAPA cosine scores shift with microphone, room and distance. These
tests pin the arithmetic that turns measured scores into a defensible number,
so the choice comes from data rather than from the default.

The accept rule must match `jarvis/speaker.py:266` exactly:

    is_match = score >= self.threshold

An off-by-one on that boundary silently changes who gets in, so it is tested
directly rather than assumed.
"""
from __future__ import annotations

import pytest

from jarvis.speaker_tuning import recommend, sweep


def test_accept_rule_is_inclusive_at_the_boundary():
    """score == threshold is a MATCH, per speaker.py:266."""
    rows = sweep(me_scores=[0.50], room_scores=[0.50], thresholds=[0.50])
    row = rows[0]
    assert row.frr == 0.0, "a score equal to the threshold must be accepted"
    assert row.far == 1.0, "...which also means the room clip gets in"


def test_clean_separation_recommends_a_gap_threshold():
    me = [0.72, 0.78, 0.81, 0.75]
    room = [0.10, 0.22, 0.31, 0.05]
    rec = recommend(sweep(me, room, _grid()))
    assert rec.clean_separation is True
    assert rec.far == 0.0 and rec.frr == 0.0
    assert max(room) < rec.threshold <= min(me), rec


def test_overlap_is_reported_not_hidden():
    """When a room voice scores as high as the user, no threshold is clean."""
    me = [0.40, 0.55, 0.70]
    room = [0.35, 0.58, 0.44]        # 0.58 sits above one genuine take
    rec = recommend(sweep(me, room, _grid()))
    assert rec.clean_separation is False
    assert rec.far > 0.0 or rec.frr > 0.0
    assert "overlap" in rec.reason.lower()


def test_far_falls_and_frr_rises_with_threshold():
    me = [0.60, 0.70, 0.80]
    room = [0.10, 0.30, 0.50]
    rows = sweep(me, room, _grid())
    fars = [r.far for r in rows]
    frrs = [r.frr for r in rows]
    assert fars == sorted(fars, reverse=True), "FAR must be non-increasing"
    assert frrs == sorted(frrs), "FRR must be non-decreasing"


def test_counts_match_the_rates():
    me = [0.9, 0.2]                  # one take will be rejected at 0.5
    room = [0.6, 0.1]                # one room clip gets in at 0.5
    row = sweep(me, room, [0.5])[0]
    assert row.rejected_me == 1 and row.frr == 0.5
    assert row.accepted_room == 1 and row.far == 0.5


def test_clean_separation_sits_in_the_MIDDLE_of_the_usable_band():
    """Not the bottom edge.

    Every threshold in the gap scores 0% FAR and 0% FRR against the measured
    clips, so the table alone cannot choose between them. The bottom edge is
    the worst pick: it satisfies the data while keeping no rejection margin at
    all, and the clips that define the gap are an easy negative (TV through
    speakers scores near zero). A real second person scores far higher than a
    TV does, and only margin protects against them. Take the midpoint.
    """
    me = [0.90, 0.95]
    room = [0.10, 0.12]
    rec = recommend(sweep(me, room, [0.2, 0.3, 0.4, 0.5, 0.6]))
    assert rec.threshold == 0.4, "must not hug the bottom edge of the gap"


def test_refuses_to_tune_without_both_classes():
    with pytest.raises(ValueError):
        sweep(me_scores=[], room_scores=[0.1], thresholds=[0.5])
    with pytest.raises(ValueError):
        sweep(me_scores=[0.9], room_scores=[], thresholds=[0.5])


def test_max_far_allows_a_deliberate_tradeoff():
    """Letting a little room audio through can be worth a big FRR win."""
    me = [0.30, 0.85, 0.90]          # 0.30 is a bad take
    room = [0.10, 0.20]
    strict = recommend(sweep(me, room, _grid()), max_far=0.0)
    assert strict.frr == 0.0
    assert strict.threshold <= 0.30


def _grid():
    return [round(0.05 * i, 2) for i in range(4, 17)]      # 0.20 .. 0.80
