"""THE MIRROR RULE, as a pin: every answer site says which way it fails.

The guard-one-half pattern has now happened FIVE times on this project.
Round 4's occurrence: strip_fillers gained a "?" carry so that "yes, uh?"
could not send a file; the carry lives in the CANONICAL strip, so it also
reached every grammar where REFUSING is the unsafe direction, and the
fixer saw the shape on ONE rung (the ringing alarm), pinned it as an open
question, and never enumerated its twins.  Measured through the real
handler: 676 flashcards marked wrong, 429 working sessions swallowing a
stop as an ANSWER, a camera run 0/195 aborted, and an undo lane that
REGRESSED against mainline.  The whole 15,684-test suite was green with
that in it, because every test was written by somebody looking at one
half.

So the rule is not a sentence in a report this time.  tests/stopconsent.py
DERIVES, for every site the census finds, which direction is unsafe:

  STOP     not matching is unsafe  -> must tolerate a carried "?" and "…"
  CONSENT  matching is unsafe and  -> must refuse a carried "?", itself or
           cannot be taken back       through a named BAR for its lane
  NEITHER  both directions inert   -> named below, with the reason

and this file asserts it.  A NEW site that the derivation cannot settle
and the table does not name FAILS -- the census fails closed, and so does
this.  That is the point: the sixth occurrence has to be a deliberate
edit to a table, in front of somebody, with a reason typed next to it.

The behavioural half lives next door and is the evidence this file is
about the real app: test_stop_is_the_safe_direction.py drives 1,948 stop
forms through Commander.handle, and test_hesitation_teeth.py drives the
520 rising consent forms that must still act on nothing.
"""
from __future__ import annotations

import pathlib

import pytest

from tests.answercensus import census
from tests.stopconsent import CONSENT, NEITHER, STOP, verdicts
from tests.test_answer_census import KNOWN_STRIPPERS

ROOT = pathlib.Path(__file__).resolve().parent.parent / "jarvis"

#: key -> (kind, why).  Only for sites the vocabulary cannot settle: an
#: ambiguous head, a bare refusal, a lead-stripping extractor, or a
#: grammar that is not a regex at all.  A KNOWN row also OVERRIDES a
#: derivation, and every override says what the derivation got wrong.
KNOWN = {
    # ---- STOP, but with no halt word of its own ----------------------
    "commander.py:Commander._enrol_control:_ENROL_READY_RX": (STOP,
        "'ready' / 'next' / 'go on' paces a live CAMERA run. The words are "
        "ambiguous ('go on' consents elsewhere); the RUNG is not -- "
        "_enrol_control's docstring says 'Stopping is the safe direction' "
        "and 'uh, stop not stopping a CAMERA is the unsafe direction'. Its "
        "twin _ENROL_WAIT_RX derives STOP on 'wait'; shipping one widened "
        "and not the other is the guard-one-half pattern itself."),
    "commander.py:_undo_match:_UNDO_TAIL_RX": (STOP,
        "the undo lane's TAIL: what may follow 'scratch that' and still be "
        "an undo. It matches courtesies and the empty string, so it has no "
        "witness of its own; it is half of _UNDO_RX, which derives STOP."),

    # ---- CONSENT, barred for the whole lane rather than at the site --
    "commander.py:_send_near_yes:_SEND_YES_RX": (CONSENT,
        "the send read-back's yes. It tolerates a '?' at its own anchor and "
        "is barred for the lane by _SEND_YES_BAR_RX, which refuses a '?' "
        "ANYWHERE in the sentence; test_hesitation_teeth drives 156 rising "
        "forms through handle() and 0 send. Its '…' is refused at the "
        "anchor, which is why _SEND_NO_TAIL had to be split out in round 4 "
        "instead of widening the shared _SEND_YES_TAIL."),
    "commander.py:parse_send_answer:_SEND_YES_RX": (CONSENT,
        "the same grammar reached from the parser; the same bar."),
    "commander.py:_send_fold_pending:_SEND_YES_RX": (CONSENT,
        "the same grammar reached from the fold-in backstop; the same bar."),

    # ---- NEITHER: a bare refusal -------------------------------------
    "commander.py:Commander._try_approval:_NO_RX": (NEITHER,
        "a DENIAL. Matching it denies; missing it leaves the permission "
        "unanswered. Neither direction acts, so no carried punctuation can "
        "hurt -- and widening it would be a change with no measurement "
        "behind it."),
    "commander.py:Commander._try_terminal_offer:_NO_RX": (NEITHER,
        "the same denial over the terminal offer."),
    "router.py:answer_kind:_ANSWER_LOCAL_RX": (NEITHER,
        "'no' routes the question to the local model instead of Claude. "
        "Both directions answer him; neither acts on the world."),

    # ---- NEITHER: reversible offers ----------------------------------
    "commander.py:briefing_answer:_BRIEFING_YES_RX": (NEITHER,
        "OVERRIDES a CONSENT derivation, and deliberately. Its worst case "
        "is that Jarvis reads a summary aloud -- reversible, and nothing "
        "leaves the flat. This is also the rung the whole branch exists "
        "for: 09-05 12:24, 'Uh, yeah.' to 'Shall I run your briefing, "
        "sir?' was thrown away. Making it refuse a rising yes would undo "
        "the fix that started all four rounds."),
    "commander.py:news_answer:_NEWS_ASK_RX": (NEITHER,
        "The news follow-up rides the briefing offer (kind 'news'): 'read "
        "them' / 'what are they' / 'go on then' read three headlines aloud. "
        "Reversible, nothing leaves the flat, no device moves; a false yes "
        "costs one sentence of stories, a false no costs him asking twice. "
        "Parked only for 60 s after a briefing that summarised the news."),
    "commander.py:Commander._try_teach_offer:_TAKE_QUIZ_RX": (NEITHER,
        "'go on' / 'do it' takes a teach offer and builds a quiz from "
        "chunks already retrieved. Nothing is sent, opened or written "
        "outside the flashcard store."),
    "commander.py:parse_yes_no:in(_YES_WORDS)": (NEITHER,
        "the PLAIN destructive read-back. stash_destructive's own "
        "docstring draws this line: 'Cancelling three alarms that way is a "
        "bad afternoon; putting a file on another machine that way cannot "
        "be taken back' -- and the irreversible half is strict=True, which "
        "is answered by parse_send_answer, not by this. Measured: "
        "parse_send_answer('yes?') is None, parse_yes_no('yes?') is True, "
        "and that difference is the design."),
    "commander.py:parse_yes_no:in(_NO_WORDS)": (NEITHER,
        "its refusal half; inert in both directions."),
    "commander.py:Commander._try_send_confirm:_SEND_MAYBE_RX": (NEITHER,
        "the VAGUE leg: 'okay?' / 'I think so' earns the lane's one "
        "RE-ASK, never a send. Matching it asks him again."),
    "commander.py:Commander._try_destructive_confirm:_SEND_MAYBE_RX": (NEITHER,
        "the same vague leg on the destructive read-back; the same re-ask."),

    # ---- NEITHER: extractors, normalisers, lead-strippers ------------
    # These take a piece OUT of what he said. They neither act nor refuse:
    # the punctuation rides on to the grammar that judges the piece, which
    # is a site of its own above. A key ending in "(" names an inline
    # grammar by PREFIX -- see stopconsent.known_row for why.
    **{k: (NEITHER, why) for k, why in {
        "commander.py:_file_answer:_FILE_ANSWER_LEAD_RX":
            "strips a lead off 'the lab report'; the phrase is judged after",
        "commander.py:_account_answer:_ACCOUNT_ANSWER_LEAD_RX":
            "strips a lead off an account label",
        "commander.py:_recipient_answer:_RECIPIENT_ANSWER_LEAD_RX":
            "strips a lead off a recipient. It derives STOP on \"that's "
            "enough\", which is a lead it REMOVES, not a meaning it has",
        "commander.py:_send_clean:_SEND_FILLER_LEAD_RX":
            "the send lane's own pre-existing filler list",
        "commander.py:day_shift_followup:_DAY_SHIFT_RX":
            "'make it tuesday' after a plan; an extractor over the whole line",
        "commander.py:day_shift_followup:_DAY_SHIFT_TAIL_RX": "its tail",
        "commander.py:pick_from_answer:_PICK_ORDINAL_RX":
            "'the first one' -> an index",
        "commander.py:_person_from_answer:_PICK_ORDINAL_RX":
            "the same ordinal against the candidate list",
        "commander.py:correction_kind:_CORRECTION_RX":
            "'no, I said Heather' -> the corrected words",
        "commander.py:correction_kind:_CORRECTION_NOT_RX":
            "'not Dana, Heather' -> the corrected words",
        "commander.py:feedback_kind:_FEEDBACK_YES_RX":
            "'that was helpful' -> a thumbs-up on the last answer",
        "commander.py:feedback_kind:_FEEDBACK_NO_RX":
            "'that was wrong' -> a thumbs-down on the last answer",
        "commander.py:_send_correction:_SEND_CORRECT_TO_RX":
            "'no, to Dana' -> the corrected recipient",
        "commander.py:_send_correction_malformed:_SEND_CORRECT_TO_RX":
            "the same grammar, asking whether the correction parsed",
        "commander.py:_send_names_someone:_SEND_CORRECT_TO_RX":
            "the same grammar, asking whether a person was named",
        "commander.py:_send_second_recipient:_SEND_SECOND_RX":
            "finds a SECOND recipient; a hit is a re-ask, never a send",
        "commander.py:quiz_kind:_QUIZ_RX":
            "'quiz me on X' -> a topic; starting a quiz is reversible",
        "commander.py:review_kind:_REVIEW_RX":
            "'review my flashcards'; reversible",
        "commander.py:strip_address:_ADDRESS_RX": "takes 'jarvis,' off the front",
        "commander.py:strip_jarvis_prefix:startswith(": "the same, as a startswith",
        "contacts.py:normalise:_LEAD_RX": "takes 'my'/'the' off a contact name",
        "contacts.py:which_line:_LEAD_RX":
            "the same, phrasing the 'which one' question",
        "dialogue.py:WeekPlanner.settle:_MOVE_RX":
            "'move it to tuesday' inside the planner; an extractor",
        "outbox.py:strip_honorific:_HONORIFIC_RX": "takes 'Dr'/'Mrs' off a name",
        "outbox.py:gender_from_honorific:_HONORIFIC_RX":
            "the same, reading a gender off it",
        "outbox.py:resolve:re.sub(": "takes 'my'/'the' off a recipient",
        "outbox.py:said_gender:re.sub(": "the same",
        "outbox.py:recipient_gender:re.sub(": "the same",
        "outbox.py:prepare:startswith(":
            "reads back a sentinel outbox.prepare itself wrote",
        "commander.py:pick_from_answer:re.sub(":
            "takes 'jarvis, the' off an ordinal",
        "commander.py:_person_from_answer:re.sub(":
            "takes a lead off a named person",
        "router.py:normalise:_PREFIX_RX":
            "takes a wake prefix off a routed question",
        "leavetime.py:parse_minutes:re.fullmatch(":
            "'about ten' -> an int; a sink, and answer_minutes strips first",
        "leavetime.py:answer_minutes:startswith(":
            "its own courtesy list, looping leads off",
        "leavetime.py:answer_minutes:endswith(": "the same, off tails",
        "filephrase.py:denied:startswith(":
            "a path prefix test on a RESOLVED path, not on his words",
        "filephrase.py:resolve:startswith(": "the same",
        "tools/filepick.py:pick:startswith(":
            "the same, choosing between candidate paths",
        "tools/timekeeper.py:normalize_kind:in(":
            "'alarm' | 'timer' | 'reminder' -- a kind, never his answer",
    }.items()},
}


@pytest.fixture(scope="module")
def cen():
    return census(ROOT, KNOWN_STRIPPERS)


@pytest.fixture(scope="module")
def verd(cen):
    return verdicts(ROOT, cen, KNOWN)


# ===================================================================
# 1. FAILS CLOSED: nothing is unclassified
# ===================================================================
def test_every_site_says_which_direction_is_unsafe(verd):
    """A new answer grammar arrives classified or it fails. That is the
    whole mechanism: the fifth occurrence happened because a rule lived
    in a report and the twins lived in the source."""
    loose = {k: v.why for k, v in verd.items() if not v.kind}
    assert not loose, (
        "these sites are neither derived nor named -- add a row to KNOWN "
        "in this file saying which direction is unsafe and why:\n"
        + "\n".join(f"  {k}\n      {w}" for k, w in sorted(loose.items())))


def test_the_known_table_has_no_stale_rows(verd):
    """A row for a site that no longer exists is a rule about nothing --
    and worse, it is a rule somebody will read as still holding. A key
    ending in "(" matches by prefix, exactly as known_row resolves it."""
    stale = []
    for k in sorted(KNOWN):
        if k.endswith("("):
            if not any(key.startswith(k) for key in verd):
                stale.append(k)
        elif k not in verd:
            stale.append(k)
    assert not stale, f"KNOWN names sites the census no longer finds: {stale}"


# ===================================================================
# 2. THE MIRROR: every STOP tolerates the carry, every CONSENT refuses it
# ===================================================================
def test_every_STOP_site_tolerates_a_carried_question_mark_and_ellipsis(verd):
    """strip_fillers("stop, uh?") == "stop?" and ("stop, uh…") == "stop…".
    Every grammar where NOT matching is the unsafe direction must take
    both. This is the assertion that would have caught round 4's defect
    on the day it was written."""
    bad = []
    for key, v in sorted(verd.items()):
        if v.kind != STOP or not v.tolerates:
            continue
        for punct, ok in v.tolerates.items():
            if not ok:
                bad.append(f"  {key}\n      {v.witness!r} + {punct!r} is REFUSED "
                           f"({v.why})")
    assert not bad, ("a STOP grammar refuses the punctuation a filled pause "
                     "carries onto it:\n" + "\n".join(bad))


def test_every_CONSENT_site_refuses_a_carried_question_mark(verd):
    """The other half, and the reason the fix is at the anchors rather
    than in strip_fillers: the carry must SURVIVE for these."""
    bad = []
    for key, v in sorted(verd.items()):
        if v.kind != CONSENT or not v.tolerates:
            continue
        if v.tolerates.get("?") and "BAR" not in v.why and "bar" not in v.why:
            bad.append(f"  {key}: {v.witness!r} + '?' is ACCEPTED ({v.why})")
    assert not bad, ("a CONSENT grammar takes a rising hesitated form and "
                     "names no bar:\n" + "\n".join(bad))


# ===================================================================
# 3. THE FLOOR: the fourteen anchors round 4 changed, by name
# ===================================================================
STOP_FLOOR = {
    "commander.py:quiet_kind:_QUIET_RX",
    "commander.py:cancel_kind:_CANCEL_TASK_RX",
    "dialogue.py:enough_kind:_ENOUGH_RX",
    "commander.py:Commander._enrol_control:_ENROL_STOP_RX",
    "commander.py:Commander._enrol_control:_ENROL_READY_RX",
    "commander.py:Commander._enrol_control:_ENROL_WAIT_RX",
    "commander.py:Commander._try_ringing:_RING_STOP_RX",
    "commander.py:Commander._try_ringing:_SNOOZE_RX",
    "commander.py:_undo_match:_UNDO_RX",
    "commander.py:_undo_match:_UNDO_TAIL_RX",
    "commander.py:Commander._try_quiz_answer:_QUIZ_STOP_RX",
    "commander.py:Commander._try_quiz_answer:_QUIZ_SKIP_RX",
    "commander.py:_lecture_end:_LECTURE_END_RX",
    "commander.py:read_control_kind:_READ_CTL_RX",
    "commander.py:parse_send_answer:_SEND_NO_RX",
    "commander.py:_send_near_yes:_SEND_NO_RX",
    "commander.py:briefing_answer:_BRIEFING_NO_RX",
    "dialogue.py:WeekPlanner.settle:_SKIP_RX",
}
CONSENT_FLOOR = {
    "commander.py:Commander._try_approval:_YES_RX",
    "commander.py:Commander._try_terminal_offer:_YES_RX",
    "commander.py:Commander._try_terminal_offer:_OPEN_IT_RX",
    "commander.py:Commander._try_enrol:_ENROL_CONFIRM_RX",
    "commander.py:_send_near_yes:_SEND_YES_RX",
    "commander.py:parse_send_answer:_SEND_YES_RX",
}


def test_the_floor_of_stop_sites_is_still_stop(verd):
    for key in sorted(STOP_FLOOR):
        assert key in verd, f"the census lost {key}"
        assert verd[key].kind == STOP, f"{key} is now {verd[key].kind}: {verd[key].why}"


def test_the_floor_of_consent_sites_is_still_consent(verd):
    for key in sorted(CONSENT_FLOOR):
        assert key in verd, f"the census lost {key}"
        assert verd[key].kind == CONSENT, \
            f"{key} is now {verd[key].kind}: {verd[key].why}"


def test_the_biometric_gate_is_a_consent_and_was_NOT_widened(verd):
    """FLAGGED IN THE REPORT. _ENROL_CONFIRM_RX was in round 4's brief as
    one of the nine stop-side anchors to widen. It is not one: the word
    'enrol' STARTS a camera run that REPLACES his face gallery, and
    _try_enrol's own docstring calls it 'the one case where a mistaken
    identity GRANTS rather than denies'. The brief's own rule -- the bar
    stays where acting is irreversible -- says leave it alone, so it was
    left alone and this pin says so out loud."""
    v = verd["commander.py:Commander._try_enrol:_ENROL_CONFIRM_RX"]
    assert v.kind == CONSENT
    assert v.tolerates == {"?": False, "…": False}


# ===================================================================
# 4. THE CARRY ITSELF IS UNCHANGED -- the fix is at the anchors
# ===================================================================
def test_strip_fillers_still_carries_the_tail(verd):
    from jarvis.endpoint import strip_fillers
    assert strip_fillers("yes, uh?") == "yes?"
    assert strip_fillers("stop, uh?") == "stop?"
    assert strip_fillers("stop, uh…") == "stop…"
    assert strip_fillers("yes, uh") == "yes"
