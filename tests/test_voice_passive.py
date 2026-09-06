"""Passive learning, and the measurement that turned it off.

THE PLAN WAS A FROZEN REFERENCE CENTROID AND A CAP OF 20, and the plan was
wrong -- not about freezing, which is right and is implemented, but about what
freezing buys. This file is the adversary that showed it, run in full so the
numbers in ``voicegallery.MAX_PASSIVE``'s comment cannot rot.

THE ATTACK, and it needs no microphone. Every accepted sample moves the pool's
centroid. An attacker who can get samples accepted therefore steers it: craft
each one to sit exactly on the bar and point as far toward a target voice as
that allows. Under the old rule the bar is measured against the centroid the
sample is about to move, so each accepted sample buys the next one more room
-- a WALK. Freezing the reference stops the walk. It does not stop the SWING,
because twenty new vectors against fourteen originals is a 59% shift in a mean
whatever each one scores.

Synthetic vectors only, at his measured within-person spread. No microphone,
no recording, no real voiceprint used as an oracle.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis import voicegallery as vg
from tests.synthvoice import Voices, centroid, cos

BAR = vg.ACCEPT_DEFAULT
SPEECH = 3.0


def _world():
    world = Voices(seed=42)
    pool = [np.asarray(v, dtype=np.float64) for v in world.takes("hunter", 14)]
    intruder = np.asarray(world.take("intruder"), dtype=np.float64)
    return pool, intruder


def _craft(reference, intruder, want, norm):
    """A vector scoring exactly ``want`` against ``reference``, pushed as far
    toward ``intruder`` as that score allows. The worst case, deliberately."""
    u = np.asarray(reference, dtype=np.float64)
    u = u / np.linalg.norm(u)
    d = intruder - u * float(intruder @ u)
    d = d / np.linalg.norm(d)
    v = want * u + np.sqrt(max(1e-12, 1.0 - want ** 2)) * d
    return v / np.linalg.norm(v) * norm


def _drift(pool, intruder, bar, steps, frozen):
    live = list(pool)
    start = centroid(pool)
    norm = float(np.mean([np.linalg.norm(e) for e in pool]))
    for _ in range(steps):
        ref = start if frozen else centroid(live)
        live.append(_craft(ref, intruder, bar + 1e-9, norm))
    now = centroid(live)
    return {"rotation": cos(now, start),
            "intruder": cos(intruder, now),
            "his_worst": min(cos(e, now) for e in pool)}


# ------------------------------------------------- the measurement, in full
def test_the_old_rule_lets_the_centroid_be_walked_off_him():
    """Gating against the centroid the sample is about to move bounds one
    step and not the walk."""
    pool, intruder = _world()
    assert cos(intruder, centroid(pool)) < BAR, "the intruder starts outside"
    got = _drift(pool, intruder, BAR, 20, frozen=False)
    assert got["rotation"] < 0.70, got
    assert got["intruder"] > 0.80, got
    assert got["his_worst"] < 0.40, got


def test_freezing_the_reference_helps_and_is_not_enough():
    """THE CORRECTION. Freezing improves every number and still ends with a
    stranger comfortably inside the accept bar, because the problem is the
    COUNT, not the bar."""
    pool, intruder = _world()
    frozen = _drift(pool, intruder, BAR, 20, frozen=True)
    walked = _drift(pool, intruder, BAR, 20, frozen=False)
    assert frozen["rotation"] > walked["rotation"], (frozen, walked)
    assert frozen["intruder"] < walked["intruder"], (frozen, walked)
    assert frozen["intruder"] > BAR, (
        "freezing was expected to leave the stranger inside; it did not, and "
        "MAX_PASSIVE's comment needs re-measuring: %r" % (frozen,))


@pytest.mark.parametrize("bar", [0.30, 0.55, 0.80])
def test_no_bar_in_a_plausible_range_closes_it_at_twenty_samples(bar):
    pool, intruder = _world()
    got = _drift(pool, intruder, bar, 20, frozen=True)
    assert got["intruder"] > BAR, (bar, got)


def test_only_the_count_bounds_the_drift():
    """At the label's own genuine floor, one sample is safe, two are marginal,
    three are not. That is the whole argument for zero."""
    pool, intruder = _world()
    g = vg.VoiceGallery()
    for e in pool:
        g.add("hunter", e)
    floor = g.genuine_floor("hunter")
    assert floor is not None and floor > 0.5, floor
    got = {n: _drift(pool, intruder, floor, n, frozen=True) for n in (1, 2, 3)}
    assert got[1]["rotation"] > 0.99, got[1]
    assert got[1]["intruder"] < BAR, got[1]
    assert got[3]["intruder"] > BAR, got[3]


# ------------------------------------------------------------- the decision
def test_passive_learning_is_off_by_default():
    assert vg.MAX_PASSIVE == 0
    pool, _intruder = _world()
    g = vg.VoiceGallery()
    for e in pool:
        g.add("hunter", e)
    v = g.identify(pool[0], SPEECH)
    assert v.who == "hunter"
    ok, why = g.passive_ok("hunter", pool[0], v)
    assert ok is False and "off" in why


# ------------------------------- and the rules a caller who turns it on gets
def test_a_sample_weaker_than_his_own_worst_take_is_refused():
    """The bar is HIS genuine floor, not the accept bar. Something that only
    just clears 0.30 is nowhere near as characteristic of him as his own
    weakest enrolment take, and adding it is how a pool stops being his."""
    pool, intruder = _world()
    g = vg.VoiceGallery()
    for e in pool:
        g.add("hunter", e)
    floor = g.genuine_floor("hunter")
    weak = _craft(centroid(pool), intruder, BAR + 0.02,
                  float(np.mean([np.linalg.norm(e) for e in pool])))
    v = g.identify(weak, SPEECH)
    assert v.who == "hunter", "the weak sample should still be ACCEPTED"
    ok, why = g.passive_ok("hunter", weak, v, max_passive=5)
    assert ok is False and "genuine floor" in why
    assert "%.3f" % floor in why


def test_a_strong_sample_is_allowed_when_it_is_deliberately_turned_on():
    pool, _intruder = _world()
    g = vg.VoiceGallery()
    for e in pool:
        g.add("hunter", e)
    strong = centroid(pool)
    v = g.identify(strong, SPEECH)
    ok, why = g.passive_ok("hunter", strong, v, max_passive=5)
    assert ok is True, why


def test_a_provisional_label_never_teaches_itself():
    world = Voices(seed=7, apart=0.3)
    g = vg.VoiceGallery()
    for e in world.takes("mara", 5):
        g.add("mara", e)
    probe = g.centroid("mara")
    v = g.identify(probe, SPEECH)
    assert v.provisional == "mara" and v.who == ""
    ok, why = g.passive_ok("mara", probe, v, max_passive=5)
    assert ok is False and "did not name" in why


def test_a_failed_margin_teaches_nobody():
    world = Voices(seed=8, apart=3.0)
    g = vg.VoiceGallery()
    for e in world.takes("hunter", 14):
        g.add("hunter", e)
    for e in world.takes("mara", 12):
        g.add("mara", e)
    probe = g.centroid("hunter")
    v = g.identify(probe, SPEECH)
    assert v.who == "" and v.margin is not None and v.margin < vg.MARGIN
    ok, _why = g.passive_ok("hunter", probe, v, max_passive=5)
    assert ok is False


def test_an_abstention_teaches_nobody():
    pool, _intruder = _world()
    g = vg.VoiceGallery()
    for e in pool:
        g.add("hunter", e)
    v = g.identify(pool[0], 0.9)
    assert v.abstained is True
    ok, _why = g.passive_ok("hunter", pool[0], v, max_passive=5)
    assert ok is False


def test_the_cap_is_read_from_the_stored_takes_not_remembered():
    """The thing voiceprint.npz cannot do. Without per-sample provenance the
    "frozen enrolment centroid" is only frozen until the next restart, after
    which the walk resumes from wherever it got to."""
    pool, _intruder = _world()
    g = vg.VoiceGallery()
    for e in pool:
        g.add("hunter", e, src="enrol")
    strong = centroid(pool)
    for _ in range(2):
        g.add("hunter", strong, src="passive")
    v = g.identify(strong, SPEECH)
    ok, why = g.passive_ok("hunter", strong, v, max_passive=2)
    assert ok is False and "cap" in why
    # And the enrolment centroid ignores the passive takes entirely.
    assert cos(g.enrolment_centroid("hunter"), centroid(pool)) > 0.999


def test_the_genuine_floor_needs_two_takes():
    g = vg.VoiceGallery()
    g.add("mara", Voices(seed=9).take("mara"))
    assert g.genuine_floor("mara") is None
    assert g.genuine_floor("nobody") is None
