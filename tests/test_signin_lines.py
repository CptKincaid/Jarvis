"""The sentences, and the four preconditions behind the one that is his.

"Voice and identity recognized, welcome back Hunter." CLAIMS TWO INSTRUMENTS,
so it may only be said when two instruments actually confirmed. Most of this
file enumerates the ways it must NOT be said -- a dark camera under the night
curfew, an abstention, two legs naming different people, a leg that was never
switched on. A sentence that claims something that did not happen is the kind
of small lie that makes a whole feature untrustworthy.

AND IT IS THE LINE FOR ANYBODY RECOGNISED, not the owner alone: his words,
with the recognised person's first name in them. The first version greeted a
guest with "Hello Mara, I recognize you."; he asked for his sentence.

Pure functions over dataclasses. No microphone, no camera, no model.
"""
from __future__ import annotations

import itertools

import pytest

from jarvis import signinlines as sl
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER, Person
from jarvis.recognise import Legs

HUNTER = Person(label="hunter", name="Hunter Peyrovi", role=ROLE_OWNER)
MARA = Person(label="mara", name="Mara", role=ROLE_KNOWN)

BOTH = Legs(voice_says="hunter", voice_running=True,
            face_says="hunter", face_running=True)


# ------------------------------------------------------- the verbatim line
def test_his_wording_is_exact():
    assert sl.line_for(BOTH, HUNTER) == (
        "Voice and identity recognized, welcome back Hunter. "
        "How may I be of assistance today?")


def test_the_first_name_is_the_first_token():
    assert sl.first_name(HUNTER) == "Hunter"
    assert sl.first_name(Person(label="mara", name="", role=ROLE_KNOWN)) == "Mara"
    assert sl.first_name(None, "there") == "there"


def test_a_typed_first_name_wins_over_the_display_name():
    """people-signin's Person row carries a typed ``first``; the ONE
    resolver of the first name honours it, so that branch's greeting
    becomes an import of this module and nothing else at the merge."""
    from types import SimpleNamespace
    typed = SimpleNamespace(label="hunter", name="Hunter Peyrovi",
                            first="Hunt")
    assert sl.first_name(typed) == "Hunt"
    blank = SimpleNamespace(label="hunter", name="Hunter Peyrovi", first="  ")
    assert sl.first_name(blank) == "Hunter"
    assert sl.BOTH_LEGS_LINE.format(first=sl.first_name(typed)) == (
        "Voice and identity recognized, welcome back Hunt. "
        "How may I be of assistance today?")


def test_the_sentence_lives_in_one_file():
    """SETTLED 09-04 for both branches: jarvis/signinlines.py OWNS the line
    and the rule -- his words, spoken TO the person recognised, addressed
    by first name, honorifics elsewhere. Neither branch carries its own
    copy, so the integration merge cannot disagree; a second copy anywhere
    under jarvis/ fails here, on purpose."""
    from pathlib import Path
    root = Path(sl.__file__).resolve().parent
    holders = sorted(p.name for p in root.rglob("*.py")
                     if "welcome back {first}" in p.read_text(encoding="utf-8"))
    assert holders == ["signinlines.py"], holders
    doc = sl.__doc__ or ""
    assert "TO THE PERSON RECOGNISED" in doc.upper()
    assert "FIRST NAME" in doc.upper()


def test_the_gate_takes_its_near_miss_line_from_here():
    from jarvis import gate
    assert gate.NEAR_MISS_LINE is sl.NEAR_MISS_LINE


@pytest.mark.parametrize("legs,why", [
    (Legs(voice_says="hunter", voice_running=True,
          face_says="", face_running=True), "the face leg named nobody"),
    (Legs(voice_says="", voice_running=True,
          face_says="hunter", face_running=True), "the voice leg named nobody"),
    (Legs(voice_says="hunter", voice_running=True,
          face_says="mara", face_running=True), "the legs disagree"),
    (Legs(voice_says="hunter", voice_running=False,
          face_says="hunter", face_running=True), "the voice leg was not running"),
    (Legs(voice_says="hunter", voice_running=True,
          face_says="hunter", face_running=False), "the camera was dark"),
])
def test_the_verbatim_line_is_unreachable_without_all_four(legs, why):
    assert sl.both_legs_line(legs, HUNTER) == "", why
    assert sl.line_for(legs, HUNTER) != sl.BOTH_LEGS_LINE.format(first="Hunter")


def test_the_verbatim_line_is_unreachable_for_a_different_person():
    """The legs agreed with each other and disagreed with the row handed in.
    Naming the row anyway would let a mismatch become an identity."""
    assert sl.both_legs_line(BOTH, MARA) == ""


def test_it_is_unreachable_by_exhaustion():
    """EVERY combination of the four inputs, not a sample of five. The line
    appears in exactly one of the sixteen and it is the one where all four
    hold."""
    said = []
    for v, vr, f, fr in itertools.product(["", "hunter", "mara"], [True, False],
                                          ["", "hunter", "mara"], [True, False]):
        legs = Legs(voice_says=v, voice_running=vr, face_says=f, face_running=fr)
        if sl.both_legs_line(legs, HUNTER):
            said.append((v, vr, f, fr))
    assert said == [("hunter", True, "hunter", True)]


# -------------------------------------------------------------- one leg
def test_a_dark_camera_says_so_rather_than_claiming_it():
    """The night curfew turns the camera off every night; the sentence has to
    be honest about which instrument actually saw him."""
    legs = Legs(voice_says="hunter", voice_running=True,
                face_says="", face_running=False)
    line = sl.line_for(legs, HUNTER)
    assert line == ("I recognize your voice, Hunter. The camera isn't "
                    "confirming right now — welcome back anyway.")


def test_seen_but_not_yet_heard():
    legs = Legs(voice_says="", voice_running=False,
                face_says="hunter", face_running=True)
    assert sl.line_for(legs, HUNTER) == (
        "I recognize your face, Hunter. I haven't heard you yet — "
        "welcome back.")


def test_a_recognised_guest_gets_his_exact_words_with_her_name():
    """HIS RULING: the sentence is the sign-in line for a recognised person,
    not "Hello Mara, I recognize you." Scope is the gate's business."""
    legs = Legs(voice_says="mara", voice_running=True,
                face_says="mara", face_running=True)
    assert sl.line_for(legs, MARA) == (
        "Voice and identity recognized, welcome back Mara. "
        "How may I be of assistance today?")
    assert "Hello Mara" not in sl.line_for(legs, MARA)
    assert not hasattr(sl, "KNOWN_LINE")


def test_a_guest_on_one_leg_is_told_which_leg_saw_her():
    """The same honesty he gets: the both-legs sentence is not said when
    only one instrument ran."""
    legs = Legs(voice_says="mara", voice_running=True,
                face_says="", face_running=False)
    line = sl.line_for(legs, MARA)
    assert line == ("I recognize your voice, Mara. The camera isn't "
                    "confirming right now — welcome back anyway.")
    seen = Legs(voice_says="", voice_running=False,
                face_says="mara", face_running=True)
    assert sl.line_for(seen, MARA) == (
        "I recognize your face, Mara. I haven't heard you yet — welcome back.")


def test_the_guest_line_needs_the_same_four_preconditions():
    said = []
    for v, vr, f, fr in itertools.product(["", "hunter", "mara"], [True, False],
                                          ["", "hunter", "mara"], [True, False]):
        legs = Legs(voice_says=v, voice_running=vr, face_says=f, face_running=fr)
        if sl.both_legs_line(legs, MARA):
            said.append((v, vr, f, fr))
    assert said == [("mara", True, "mara", True)]


# ------------------------------------------------------------- near miss
def test_two_people_too_close_to_separate_name_nobody():
    line = sl.line_for(Legs(voice_running=True), None, near_miss=True)
    assert line == ("I can hear someone I know, but I can't tell which of "
                    "you — say a little more and I'll catch up.")
    assert "hunter" not in line.lower() and "mara" not in line.lower()


def test_a_near_miss_outranks_nothing_but_is_outranked_by_a_name():
    """A confirmed identity beats a near miss. The near-miss line may never
    appear beside a name."""
    assert sl.line_for(BOTH, HUNTER, near_miss=True) != sl.NEAR_MISS_LINE


# ----------------------------------------------------------- provisional
def test_a_provisional_label_is_spoken_only_in_shadow():
    assert sl.line_for(Legs(), None, provisional=MARA, mode="shadow") == (
        "I think that's Mara, but I've only heard them a few times — "
        "I'll keep listening.")
    assert sl.line_for(Legs(), None, provisional=MARA, mode="enforce") == ""
    assert sl.line_for(Legs(), None, provisional=MARA) == ""


def test_a_near_miss_outranks_a_provisional_guess():
    assert sl.line_for(Legs(), None, near_miss=True, provisional=MARA,
                       mode="shadow") == sl.NEAR_MISS_LINE


# --------------------------------------------------------- silence counts
def test_an_abstention_says_nothing_about_identity():
    """Below 1.5 s of speech nothing was measured. Narrating it would be
    claiming an instrument that did not run."""
    legs = Legs(voice_says="", voice_running=True, face_says="",
                face_running=False)
    assert sl.line_for(legs, None) == ""


def test_nothing_enrolled_says_nothing():
    assert sl.line_for(Legs(), None) == ""


def test_a_leg_that_named_somebody_who_is_not_the_row_says_nothing():
    legs = Legs(voice_says="heather", voice_running=True)
    assert sl.line_for(legs, HUNTER) == ""


# ------------------------------------------------- and never as security
@pytest.mark.parametrize("line", [sl.BOTH_LEGS_LINE, sl.VOICE_ONLY_LINE,
                                  sl.FACE_ONLY_LINE,
                                  sl.NEAR_MISS_LINE, sl.PROVISIONAL_LINE])
def test_no_sentence_is_worded_as_security(line):
    """identity.py's opening paragraph: a recording defeats the voice check
    and the user was told so. Nothing here may imply otherwise."""
    low = line.lower()
    for word in ("authenticat", "verified", "secure", "unlock", "password",
                 "access granted", "identity confirmed"):
        assert word not in low, (word, line)


def test_the_module_is_pure():
    """No clock, no file, no config, no model, no logger -- the same contract
    recognise.py holds, and for the same reason: it is the only way the whole
    decision can be tested on a box with no second voice."""
    import ast
    import inspect
    src = inspect.getsource(sl)
    imported = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported.append(getattr(node, "module", "") or "")
            imported += [a.name for a in node.names]
    joined = " ".join(imported)
    for forbidden in ("time", "os", "logging", "numpy", "jarvis.config",
                      "jarvis.logs", "jarvis.events"):
        assert forbidden not in joined, forbidden
