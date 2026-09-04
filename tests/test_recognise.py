"""The verdict, as a pure function -- no mic, no lens, no model, no clock.

Every row of the design's table is here, and so is the invariant the whole
feature rests on: THE ABSENCE OF A LEG IS NEVER EVIDENCE AGAINST HIM. No
camera, no gallery, the curfew, his back turned, a clip too short to score --
each is "no opinion", and no combination of them may produce a negative.
"""
import itertools

from jarvis import recognise as rec
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER

OWNER = "hunter"
ROLES = {OWNER: ROLE_OWNER, "heather": ROLE_KNOWN}


def legs(**kw):
    return rec.Legs(**kw)


def call(**kw):
    return rec.recognise(legs(**kw), ROLES, OWNER)


# ------------------------------------------------------------- the table
def test_row1_voice_alone_carries_the_night():
    """Curfew, camera off, lens blocked: voice is the only leg and it is
    enough. This is the row that must never regress -- it is the one that
    holds every night between 23:00 and 07:00."""
    v = call(voice_says=OWNER, voice_running=True)
    assert (v.who, v.role, v.how) == (OWNER, ROLE_OWNER, rec.HOW_VOICE)


def test_row2_face_alone_when_the_clip_could_not_be_scored():
    v = call(face_says=OWNER, face_running=True)
    assert (v.who, v.how) == (OWNER, rec.HOW_FACE)


def test_row3_both_legs_agree_and_voice_is_named_as_the_decider():
    """Voice is named first because it is the leg that is always there;
    the pane and the log have to say which leg decided."""
    v = call(voice_says=OWNER, voice_running=True,
             face_says=OWNER, face_running=True)
    assert v.how == rec.HOW_VOICE


def test_row4_highest_role_among_the_positives_wins():
    v = call(voice_says=OWNER, voice_running=True,
             face_says="heather", face_running=True)
    assert (v.who, v.role) == (OWNER, ROLE_OWNER)


def test_row5_a_known_person_is_recognised_by_face():
    v = call(face_says="heather", face_running=True)
    assert (v.who, v.role, v.how) == ("heather", ROLE_KNOWN, rec.HOW_FACE)


def test_row6_the_passphrase_never_demotes_a_positive():
    v = call(voice_says=OWNER, voice_running=True, phrase_says=OWNER)
    assert v.how == rec.HOW_VOICE


def test_row7_face_rescues_him_when_his_voice_is_hoarse():
    """EITHER LEG SUFFICES: a voice leg naming somebody else does not
    cancel a face leg naming him."""
    v = call(voice_says="heather", voice_running=True,
             face_says=OWNER, face_running=True)
    assert (v.who, v.role, v.how) == (OWNER, ROLE_OWNER, rec.HOW_FACE)


def test_row9_a_voice_negative_never_cancels_a_face_positive():
    v = call(voice_says="someone-else", voice_running=True,
             face_says="heather", face_running=True)
    assert (v.who, v.role) == ("heather", ROLE_KNOWN)


def test_row10_the_passphrase_works_with_the_camera_off_and_the_gallery_empty():
    """Ill, in the dark, turned away. Neither leg is running at all."""
    v = call(phrase_says=OWNER)
    assert (v.who, v.role, v.how) == (OWNER, ROLE_OWNER, rec.HOW_PHRASE)


def test_rows_11_and_12_are_byte_identical():
    """A near miss must not be distinguishable from an unrelated sentence:
    no extra field, no different reason, nothing to time or to read."""
    near_miss = call(phrase_says="")          # offered, did not match
    unrelated = call()                        # nothing offered at all
    assert near_miss == unrelated
    assert near_miss.how == rec.HOW_NOBODY


def test_an_unenrolled_name_from_either_leg_is_no_opinion_not_a_person():
    """A leg may name somebody the registry has never heard of. That is
    not an identity and must not become one."""
    assert call(voice_says="stranger", voice_running=True).how == rec.HOW_NOBODY
    assert call(face_says="stranger", face_running=True).how == rec.HOW_NOBODY


def test_the_passphrase_only_ever_resolves_to_an_owner():
    """It is the OWNER's way back in. A KNOWN person's row carrying a
    phrase hash must not promote them."""
    v = call(phrase_says="heather")
    assert v.how == rec.HOW_NOBODY


# -------------------------------------------------- the safety invariant
def test_absence_is_never_evidence_against_anyone():
    """Exhaustive: over every combination of legs, turning a leg OFF can
    only ever lose a positive -- it can never turn one into a refusal of
    somebody the other leg named."""
    values = ("", OWNER, "heather", "stranger")
    for vs, fs, ps in itertools.product(values, values, ("", OWNER)):
        full = rec.recognise(rec.Legs(voice_says=vs, voice_running=True,
                                      face_says=fs, face_running=True,
                                      phrase_says=ps), ROLES, OWNER)
        blind = rec.recognise(rec.Legs(voice_says=vs, voice_running=True,
                                       phrase_says=ps), ROLES, OWNER)
        if full.who and full.how == rec.HOW_FACE:
            continue            # the face leg was the decider; losing it is a loss
        assert blind.who == full.who, (vs, fs, ps)


def test_a_dark_leg_and_a_silent_leg_are_the_same_thing():
    """`voice_running=False` (the verifier is switched off) must read
    exactly like `voice_says=""` (it ran and had nothing to say)."""
    off = call(face_says=OWNER, face_running=True, voice_running=False)
    quiet = call(face_says=OWNER, face_running=True, voice_running=True)
    assert off.who == quiet.who == OWNER


# ------------------------------------------------------------- purity
def test_the_module_holds_no_threshold_of_its_own():
    """A second copy of a bar is how a gate silently stops meaning what it
    says. Every threshold stays where it already lives -- speaker.threshold
    and camera.identity_min -- and none of them is duplicated here."""
    import inspect
    src = inspect.getsource(rec)
    floats = []
    for tok in src.replace("(", " ").replace(")", " ").replace(",", " ").split():
        try:
            val = float(tok)
        except ValueError:
            continue
        if 0.0 < val < 1.0:
            floats.append(tok)
    assert floats == [], "recognise.py grew a threshold: %r" % (floats,)


def test_it_is_total_and_never_raises():
    for bad in (None, 0, "", [], {"who": 1}):
        v = rec.recognise(rec.Legs(voice_says=str(bad), voice_running=True),
                          ROLES, OWNER)
        assert isinstance(v, rec.Verdict)
    assert rec.recognise(rec.Legs(), {}, "").how == rec.HOW_NOBODY


def test_it_reads_no_clock_and_opens_no_file():
    import inspect
    src = inspect.getsource(rec)
    for forbidden in ("import time", "datetime", "open(", "Path(", "log."):
        assert forbidden not in src, forbidden
