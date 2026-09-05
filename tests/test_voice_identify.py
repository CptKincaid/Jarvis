"""``VoiceGallery.identify`` -- the rule, branch by branch.

UNKNOWN IS THE DEFAULT RETURN. Most of this file is about the ways nobody
gets named, because a false accept hands somebody else his assistant with
owner scope and a false reject costs him one repeat. Wherever the two
conflicted, the reject was picked.

Every vector is SYNTHETIC (tests/synthvoice.py). No microphone, no recording,
no real voiceprint used as an oracle.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis import voicegallery as vg
from tests.synthvoice import Voices, centroid, cos

SPEECH = 3.0            # comfortably past the abstain window


def _gal(world, counts, apart=None):
    g = vg.VoiceGallery()
    for label, n in counts.items():
        for e in world.takes(label, n):
            g.add(label, e)
    assert apart is None or apart
    return g


# ----------------------------------------------------------- naming somebody
def test_the_only_enrolled_person_is_named(gal_world=None):
    world = Voices(seed=1)
    g = _gal(world, {"hunter": 14})
    v = g.identify(world.take("hunter"), SPEECH)
    assert v.who == "hunter"
    assert v.score > vg.ACCEPT_DEFAULT
    assert v.margin is None, "a margin was applied with one label enrolled"


def test_two_people_far_apart_are_each_named():
    """`apart` is the DIAL described in tests/synthvoice.py, not a measurement
    of two real humans. At 0.3 the two are well separated and the rule should
    name each of them; that is a statement about the code, not about Mara."""
    world = Voices(seed=2, apart=0.3)
    g = _gal(world, {"hunter": 14, "mara": 10})
    assert g.identify(world.take("hunter"), SPEECH).who == "hunter"
    assert g.identify(world.take("mara"), SPEECH).who == "mara"


def test_the_score_is_against_a_centroid_not_a_best_sample():
    """THE MEASURED DEPARTURE FROM THE FACE GALLERY. His own pairwise minimum
    is 0.283, below the 0.30 bar that admits him, while sample-to-centroid
    runs 0.632-0.831. Copying facegallery.match here rejects him on day one.
    """
    world = Voices(seed=3)
    pool = world.takes("hunter", 14)
    g = vg.VoiceGallery()
    for e in pool:
        g.add("hunter", e)
    probe = pool[0]
    others = pool[1:]
    best_sample = max(cos(probe, e) for e in others)
    to_centroid = cos(probe, centroid(others))
    assert to_centroid > best_sample + 0.1, (to_centroid, best_sample)
    assert g.identify(probe, SPEECH).who == "hunter"


# --------------------------------------------------------------- the margin
def test_two_people_too_close_to_separate_produce_UNKNOWN():
    """THE SENTENCE THAT REPLACES A COIN FLIP. When the top two centroids sit
    inside the margin, nobody is named -- not the nearer one."""
    world = Voices(seed=4, apart=3.0)      # deliberately confusable
    g = _gal(world, {"hunter": 14, "mara": 12})
    probe = world.take("hunter")
    v = g.identify(probe, SPEECH)
    assert v.margin is not None and v.margin < vg.MARGIN
    assert v.who == "", "the nearer label was named on a failed margin"
    assert "margin" in v.why
    # And it still says WHO it could not choose between, so the near-miss
    # sentence can be honest without naming anybody.
    assert {v.scores[0][0], v.scores[1][0]} == {"hunter", "mara"}


def test_a_failed_margin_never_resolves_to_the_owner():
    """"closest is probably him" is a false accept dressed as convenience,
    and it is exactly the direction the weighting forbids."""
    world = Voices(seed=5, apart=3.0)
    g = _gal(world, {"hunter": 14, "mara": 12})
    for _ in range(20):
        v = g.identify(world.take("mara"), SPEECH)
        if v.margin is not None and v.margin < vg.MARGIN:
            assert v.who == ""
            break
    else:
        pytest.skip("this seed produced no near miss to check")


def test_the_margin_is_the_measured_floor_on_meaning():
    """0.20 sits just above the p99 of the SAME-PERSON margin distribution --
    p99 0.194 over 2800 trials, max 0.257 (2026-09-04, his own pool split into
    two pretend people). Reproduced here on the calibrated fixture: split ONE
    person's takes into two labels and confirm the rule almost never separates
    them, which is what makes the bar a floor rather than a discrimination
    threshold."""
    world = Voices(seed=6)
    pool = world.takes("hunter", 14)
    named = 0
    trials = 400
    rng = np.random.default_rng(11)
    for _ in range(trials):
        idx = rng.permutation(len(pool))
        probe, rest = pool[idx[0]], idx[1:]
        half = len(rest) // 2
        g = vg.VoiceGallery()
        for i in rest[:half]:
            g.add("a", pool[i])
        for i in rest[half:]:
            g.add("b", pool[i])
        if g.identify(probe, SPEECH).who:
            named += 1
    # A same-person split must almost never produce a confident name. It is
    # not zero and the constant's comment says so: 0.75% of measured trials
    # exceed 0.20.
    assert named / trials < 0.05, named / trials


def test_the_margin_bar_is_marked_provisional_in_code():
    """Until a second real person enrols there is no between-people evidence
    on this box at all, and nothing may quietly start describing 0.20 as a
    validated bar."""
    assert vg.MARGIN_IS_PROVISIONAL is True
    assert vg.MARGIN == 0.20


# ------------------------------------------------------------- the accept bar
def test_a_voice_below_the_accept_bar_names_nobody():
    world = Voices(seed=7, apart=0.05)
    g = _gal(world, {"hunter": 14})
    stranger = world.take("someone-else")
    v = g.identify(stranger, SPEECH)
    if v.score >= vg.ACCEPT_DEFAULT:
        pytest.skip("this fixture's stranger is not far enough away")
    assert v.who == "" and "below" in v.why


def test_the_live_threshold_is_honoured():
    world = Voices(seed=8)
    g = _gal(world, {"hunter": 14})
    probe = world.take("hunter")
    assert g.identify(probe, SPEECH, threshold=0.30).who == "hunter"
    assert g.identify(probe, SPEECH, threshold=0.99).who == ""


def test_no_bar_lives_anywhere_but_this_module():
    """gate.py stays threshold-free; the accept bar is speaker's measured
    0.30 and the margin is here. Two copies of a bar is two bars that drift."""
    import inspect

    from jarvis import gate as gt
    src = inspect.getsource(gt)
    found = []
    for tok in src.replace("(", " ").replace(")", " ").replace(",", " ").split():
        try:
            val = float(tok)
        except ValueError:
            continue
        if 0.0 < val < 1.0:
            found.append(tok)
    assert found == [], found


# ---------------------------------------------------------------- abstaining
def test_a_short_utterance_abstains_and_scores_nothing():
    """Every "Yes." he says arrives under 1.5 s. This is a fail-open, and the
    owner fallback for it lives in gate._voice_leg where the owner is known."""
    world = Voices(seed=9)
    g = _gal(world, {"hunter": 14})
    v = g.identify(world.take("hunter"), 0.8)
    assert v.abstained is True
    assert v.who == "" and v.score == 0.0 and v.scores == ()
    assert "abstain" in v.why


def test_the_abstain_window_is_not_shortened_by_multi_speaker():
    assert vg.ABSTAIN_SECONDS == 1.5


def test_an_abstention_is_never_a_recognition():
    """It must not be narratable as one: nothing was measured."""
    world = Voices(seed=10)
    g = _gal(world, {"hunter": 14, "mara": 10})
    v = g.identify(world.take("hunter"), 1.4)
    assert v.abstained and not v.who and not v.provisional and v.margin is None


# --------------------------------------------------------------- provisional
def test_a_label_with_too_few_takes_scores_but_never_names():
    world = Voices(seed=11, apart=0.3)
    g = _gal(world, {"mara": 4})
    v = g.identify(world.take("mara"), SPEECH)
    assert v.score > vg.ACCEPT_DEFAULT, "it should still SCORE"
    assert v.who == "", "a provisional label named somebody"
    assert v.provisional == "mara"
    assert "take" in v.why


def test_the_owners_migrated_pool_is_not_provisional():
    world = Voices(seed=12)
    g = _gal(world, {"hunter": 14})
    assert g.provisional("hunter") is False
    assert g.identify(world.take("hunter"), SPEECH).who == "hunter"


def test_eight_takes_is_where_naming_starts():
    world = Voices(seed=13)
    g = vg.VoiceGallery()
    takes = world.takes("mara", 8)
    for i, e in enumerate(takes):
        g.add("mara", e)
        v = g.identify(world.take("mara"), SPEECH)
        if i + 1 < vg.MIN_TAKES_TO_NAME:
            assert v.who == "" and v.provisional == "mara"
        else:
            assert v.who == "mara"


# ------------------------------------------------------------ nothing at all
def test_an_empty_gallery_names_nobody_and_says_so():
    v = vg.VoiceGallery().identify(np.ones(192, dtype=np.float32), SPEECH)
    assert v.who == "" and "nobody is enrolled" in v.why
    assert v.abstained is False


def test_a_degenerate_probe_names_nobody():
    world = Voices(seed=14)
    g = _gal(world, {"hunter": 14})
    v = g.identify(np.zeros(192, dtype=np.float32), SPEECH)
    assert v.who == "" and "no usable embedding" in v.why
    v = g.identify(np.ones(512, dtype=np.float32), SPEECH)
    assert v.who == "" and "wrong dimension" in v.why


def test_unknown_is_reachable_by_construction_not_by_exception():
    """Every refusal builds the same object with an empty ``who``. Nothing
    here raises to say "I do not know"."""
    world = Voices(seed=15)
    g = _gal(world, {"hunter": 14})
    for probe, speech in ((world.take("hunter"), 0.5),
                          (np.zeros(192, dtype=np.float32), SPEECH),
                          (world.take("hunter"), SPEECH)):
        v = g.identify(probe, speech)
        assert isinstance(v, vg.VoiceVerdict)
        assert isinstance(v.who, str)
