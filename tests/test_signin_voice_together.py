"""THE JUNCTION OF THE TWO LANES, which neither of them could test alone.

people-signin and voice-multispeaker each passed their own security suite and
each was verdicted safe to merge; merged together they broke, and the reason
was not in any conflict hunk. This file is the standing guard against the
class: properties that only EXIST once both lanes are in the tree, and that a
future change to either one could quietly remove while both lanes' own suites
stayed green.

THREE JUNCTIONS, and each is a real widening.

1. THE VOICE LEG CAN NOW NAME A GUEST. Before voice-multispeaker,
   ``gate._voice_leg`` returned the owner's label or nothing at all, so
   people-signin's per-turn scope was only ever reached down the FACE leg --
   the camera, the office, daylight. A second enrolled voice means a guest
   can be named in the dark, on the leg that runs all the time, and every
   guard people-signin built has to hold on that path too: KNOWN scope, the
   refusal that names her, and a refused turn attributing nobody.

2. A NAME CAN NOW BE SAID OUT LOUD AT A VOICE VERDICT THAT NAMES NOBODY.
   people-signin reads a claimed name off the transcript; voice-multispeaker
   has four separate ways of measuring "somebody, but not anybody I will
   name" -- a fault, a failed margin, a best guess who is somebody else, a
   pool that is somebody else's. Saying "it's Mara Quinn" must not move any
   of them one inch. It may only choose a kinder sentence.

3. ENFORCE IS NOT ENFORCE UNTIL SOMEBODY CAN GET BACK IN. people-signin
   downgrades enforce to shadow when no owner carries a typed override code.
   Shadow admits everybody with ``who=""``, so a voice-lane assertion of the
   form "she was never admitted as him" is VACUOUSLY TRUE there -- which is
   exactly how this merge first went green in the places that mattered least
   and red in the places that shouted. Both guards are pinned here together.

Synthetic stats dictionaries and invented people (Mara Quinn, Heather Vance).
No microphone, no camera, no recording, no real registry, no real voiceprint.
"""
from __future__ import annotations

import pytest

from jarvis import gate as gt
from jarvis import identity as ident
from jarvis import passphrase as pp
from jarvis import scope as scope_mod
from jarvis import signinlines as sl
from tests.test_owner_gate_wiring import _stand_in

MARA = "Mara Quinn"


def _registry(tmp_path, *, code=True):
    """One owner and one KNOWN guest with a first and last name, so the
    sign-in leg has something to narrow to and the honorific has something
    to read."""
    r = ident.Registry(path=tmp_path / "people.json")
    r.add_person(ident.Person(label="hunter", name="Hunter", role=ident.ROLE_OWNER,
                              voice=True, honorific="sir"))
    # NO ``voice=True`` ON HER ROW, and that is a finding of this merge
    # rather than a fixture detail. identity.Registry.add_person REFUSES a
    # voice-enrolled non-owner, on the stated ground that speaker.py holds
    # one voiceprint and one centroid -- which voice-multispeaker is exactly
    # the end of. The flag is bookkeeping only (nothing reads it as an input
    # to any decision; the voice leg reads the GALLERY), so the merge leaves
    # the guard alone and the widening is Hunter's call, not the merge
    # author's. What matters here is that her row's ROLE is what scopes her,
    # and it does.
    r.add_person(ident.Person(label="mara", first="Mara", last="Quinn",
                              role=ident.ROLE_KNOWN, honorific="ma'am",
                              consent="typed"))
    if code:
        # Without this the gate under test is a SHADOW gate: see the third
        # junction in the module docstring.
        r.set_secret("hunter", "code_hash", pp.hash_secret("xxx000code"))
    r.save()
    return ident.Registry.load(tmp_path / "people.json")


def _gate(tmp_path, *, mode="enforce", code=True, camera=False):
    reg = _registry(tmp_path, code=code)
    opts = {"owner.mode": mode, "camera.identity": camera}
    return gt.OwnerGate(registry=reg, owner="hunter",
                        get_option=lambda k, d=None: opts.get(k, d))


def _stats(**kw):
    """A voice-lane stats dict in its full shape, defaulting to a clip that
    matched and named nobody."""
    base = {"total": 1, "matched": 1, "scores": [0.41], "who": "",
            "who_scores": {}, "labels": (), "abstained": False,
            "who_fault": "", "matched_label": "", "top": "",
            "provisional": "", "near_miss": False}
    base.update(kw)
    return base


# ------------------- junction 3 first: the gate under test must be enforcing
def test_the_fixture_is_an_enforcing_gate_and_says_why_when_it_is_not():
    """THE ASSERTION THAT KEEPS THE REST OF THIS FILE HONEST. Every "she was
    refused" below is worthless in shadow, where nothing is refused and
    nobody is named. Both halves of the guard are pinned: with a code the
    mode is enforce, without one it is shadow -- people-signin's rule that
    he cannot lock himself out, still live after the voice lane landed."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        assert _gate(Path(d), code=True).effective_mode() == gt.MODE_ENFORCE
    with tempfile.TemporaryDirectory() as d:
        assert _gate(Path(d), code=False).effective_mode() == gt.MODE_SHADOW


# ------------------------------- junction 1: a guest named by the VOICE leg
def test_a_guest_the_voice_leg_names_gets_known_scope_not_his(tmp_path):
    """The widening in one test. The voice leg names Mara -- something it
    could not do before voice-multispeaker -- and people-signin's scope
    still holds her to the time and the weather."""
    g = _gate(tmp_path)
    ok = g.judge("voice", "what time is it", stats=_stats(who="mara"))
    assert ok.admit is True
    assert (ok.who, ok.role, ok.how) == ("mara", ident.ROLE_KNOWN, gt.HOW_VOICE)

    his = g.judge("voice", "read me my mail", stats=_stats(who="mara"))
    assert his.admit is False
    assert his.who == "mara" and his.role == ident.ROLE_KNOWN
    assert his.line == gt.NOT_YOURS_LINE.format(name=MARA)


def test_the_voice_leg_cannot_hand_a_guest_the_owners_scope(tmp_path):
    """The escalation the voice lane exists to prevent, asked at the layer
    people-signin added. Every one of the voice lane's nobody shapes, put to
    a gate that now has a scope to give away."""
    g = _gate(tmp_path)
    for why, stats in (
            ("a fault in the naming instrument", _stats(who_fault="boom")),
            ("a failed margin", _stats(near_miss=True, top="mara")),
            ("the best guess is somebody else", _stats(top="mara")),
            ("her pool", _stats(matched_label="mara")),
            ("a provisional label", _stats(top="mara", provisional="mara")),
    ):
        d = g.judge("voice", "read me my mail", stats=stats)
        assert d.admit is False, why
        assert d.who == "" and d.role != ident.ROLE_OWNER, why


def test_a_refused_voice_turn_attributes_nobody_in_the_app(tmp_path):
    """people-signin's round-3 fix, driven down the leg the voice lane
    added: a REFUSED turn must leave no attribution behind, or the guest
    owns the process-wide scope for the whole TTL."""
    # Her row carries no ``voice`` flag and needs none: the voice leg reads
    # the GALLERY, and the registry row supplies the role that scopes her.
    a = _stand_in(tmp_path, known=True)
    named = {"total": 1, "matched": 1, "scores": [0.41], "who": "heather",
             "who_scores": {"heather": 0.41}}
    assert a._gate_admits("read me my mail", named) is False
    assert a._gate_who == "" and a._gate_how == ""
    assert a.abandoned == ["gate:known"]
    assert a.spoken and "Heather" in a.spoken[0]


def test_an_admitted_voice_guest_carries_her_own_reading(tmp_path):
    """And the other half: an ADMITTED guest turn installs HER reading, so
    the commander, the debrief and the record all see a turn that is not
    his. The voice leg reaching this line at all is the merge."""
    a = _stand_in(tmp_path, known=True)
    named = {"total": 1, "matched": 1, "scores": [0.41], "who": "heather",
             "who_scores": {"heather": 0.41}}
    assert a._gate_admits("what time is it", named) is True
    assert a._gate_who == "heather"
    who, hon = scope_mod.reading(a._gate_addressee)
    assert who == "Heather" and hon == "ma'am"   # Person.display()
    assert scope_mod.reading(a._gate_addressee) != scope_mod.OWNER


# ------------------------ junction 2: a claimed name moves nothing it may not
CLAIM = "hi, this is Mara Quinn"


@pytest.mark.parametrize("why,stats", [
    ("a fault in the naming instrument", {"who_fault": "boom"}),
    ("a failed margin", {"near_miss": True, "top": "mara"}),
    ("the best guess is somebody else", {"top": "mara"}),
    ("her pool with no name", {"matched_label": "mara"}),
    ("a rejected clip", {"matched": 0}),
])
def test_saying_a_name_admits_nobody_the_voice_leg_would_not(tmp_path, why,
                                                             stats):
    """THE HEART OF THE MERGE. people-signin reads a name off the words;
    voice-multispeaker decides these five shapes name nobody. The name may
    change the SENTENCE and it may never change the VERDICT."""
    g = _gate(tmp_path)
    quiet = g.judge("voice", "read me my mail", stats=_stats(**stats))
    claimed = g.judge("voice", CLAIM, stats=_stats(**stats))
    assert quiet.admit is False and claimed.admit is False, why
    assert claimed.who == "" and claimed.role == "", why
    assert claimed.how == gt.HOW_NOBODY, why


def test_the_claim_is_carried_but_never_confirmed_without_a_leg(tmp_path):
    """It buys one kinder sentence and nothing else -- and NOT the welcome,
    which claims two instruments."""
    # matched=0: the clip cleared nothing, so the voice leg names nobody.
    # A bare _stats() is a NAMELESS MATCH ON HIS POOL, which is the
    # documented fail-open and is HIM -- not a blank verdict.
    g = _gate(tmp_path, camera=False)
    d = g.judge("voice", CLAIM, stats=_stats(matched=0))
    assert d.admit is False
    assert d.line == gt.SIGNIN_NO_LEG_LINE.format(name=MARA)
    assert d.line != gt.SIGNIN_OK_LINE.format(first="Mara")


def test_an_owners_name_said_out_loud_is_never_carried(tmp_path):
    """A stranger saying "Hunter" learns nothing: the refusal names no owner,
    which is the line UNKNOWN_LINE was rewritten to keep."""
    g = _gate(tmp_path)
    d = g.judge("voice", "hi, this is Hunter", stats=_stats(matched=0))
    assert d.admit is False
    assert "hunter" not in (d.line or "").lower()


def test_the_welcome_needs_a_leg_and_the_leg_may_be_the_voice(tmp_path):
    """The two lanes' one shared sentence, at the junction: the claim
    narrows, the VOICE leg confirms, and the welcome is spoken -- with the
    wording owned by jarvis/signinlines.py and nowhere else."""
    g = _gate(tmp_path)
    d = g.judge("voice", CLAIM, stats=_stats(who="mara"))
    assert d.admit is True and d.who == "mara"
    assert d.line == sl.BOTH_LEGS_LINE.format(first="Mara")


def test_the_two_lanes_hold_one_copy_of_the_welcome(tmp_path):
    """The merge resolution itself, pinned. gate.SIGNIN_OK_LINE is the SAME
    OBJECT as signinlines.BOTH_LEGS_LINE -- an alias, not a second literal
    that a later edit could let drift."""
    assert gt.SIGNIN_OK_LINE is sl.BOTH_LEGS_LINE
    assert gt.first_name is sl.first_name
    assert gt.NEAR_MISS_LINE is sl.NEAR_MISS_LINE


def test_both_lanes_prewarmed_lines_survived_the_union(tmp_path):
    """Each lane added one sentence to PREWARM_LINES and a union that lost
    either would cost a first-hearing latency nobody would notice in a
    test."""
    assert gt.ENROL_AT_KEYBOARD_LINE in gt.PREWARM_LINES
    assert gt.NEAR_MISS_LINE in gt.PREWARM_LINES
    assert not any("{" in line for line in gt.PREWARM_LINES)


def test_a_near_miss_with_him_on_top_is_still_nobody_at_this_gate(tmp_path):
    """THE CASE THE OTHER PARAMETRISATION CANNOT CATCH. With ``top`` naming
    somebody else the "best guess is not him" rule refuses anyway, so that
    row would stay green with the margin rule deleted. HIM on top is the one
    the voice lane measured (a confusable guest minted as him 22 of 150 at
    the widest synthetic overlap), and it is nobody here too."""
    g = _gate(tmp_path)
    d = g.judge("voice", "read me my mail",
                stats=_stats(near_miss=True, top="hunter"))
    assert d.admit is False
    assert d.who == "" and d.role != ident.ROLE_OWNER


def test_the_near_miss_sentence_still_reaches_a_speaker_who_claims_nothing(
        tmp_path):
    """The voice lane's wording survives the merge on the path it was
    written for: a near miss with no name said out loud."""
    g = _gate(tmp_path)
    d = g.judge("voice", "open the front door",
                stats=_stats(near_miss=True, top="hunter"))
    assert d.line == gt.NEAR_MISS_LINE


def test_a_claimed_name_chooses_the_sentence_at_a_near_miss(tmp_path):
    """A DECISION OF THIS MERGE, PINNED SO IT IS NOT AN ACCIDENT. Both lanes
    write a sentence for "somebody, unconfirmed", and in judge() the CLAIM is
    read first. That is the right way round -- the speaker told us a name, so
    answering the name is more useful than "I can't tell which of you" -- and
    both outcomes refuse, so nothing about admission turns on it.

    THE COST, SAID OUT LOUD: the sign-in sentence is rate-limited per claimed
    name (SIGNIN_COOLDOWN_S), so a second near-miss turn inside a minute from
    somebody repeating the same name is refused SILENTLY, where the voice lane
    alone would have said the near-miss line again. A silent refusal is the
    behaviour people-signin already chose for a repeated claim; it is recorded
    here rather than discovered."""
    g = _gate(tmp_path)
    near = _stats(near_miss=True, top="hunter")
    first = g.judge("voice", CLAIM, stats=near, now=1000.0)
    assert first.admit is False
    assert first.line == gt.SIGNIN_NO_LEG_LINE.format(name=MARA)
    again = g.judge("voice", CLAIM, stats=near, now=1001.0)
    assert again.admit is False and again.line == ""
