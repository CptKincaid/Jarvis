"""The transcript confidence gate, and the two ways out of it.

The 2026-08-31 voice session lost NINE separate features to one line of
code -- ``TranscribeResult.accepted``, the Whisper avg_logprob gate in
``jarvis/transcriber.py`` -- which ran in ``app._process_audio`` BEFORE
the commander saw a syllable. Hunter's notes, and the log lines behind
them (/tmp/vss_voice/jarvis.log, all times 2026-08-31):

  #24  "belay that (low confidence)"      20:50:24  'Belay that'   -0.88
  #33  "Yes (low confidence, didnt do it)" 20:58:35 'Yes.'         -0.95
       ...answering his OWN "Clear all three off your shopping list, sir?"
  #144 "Scratch that gets rejected"       21:02:26  'scratch that' -0.89
  #67  "all have low confidence errors"   21:14:59  'Play my liked songs.' -0.97
  #68  "rejected confidence issues"       21:16:53  'volume 40'    -0.87
                                          20:32:53  'max volume'   -2.08
  #101 "low confidence errors again"      21:26:37  'Full brightness' -0.99

Every one of those transcripts is CORRECT. Whisper's avg_logprob is
length-biased -- few tokens, so the end-of-text token dominates the mean
-- so the gate was worst at exactly the utterances a command and an
answer look like: short ones.

Three fixes, one test section each:
  1. the threshold, moved from -0.85 to -2.90 on measured data;
  2. a salvage that runs a sub-threshold transcript anyway when Jarvis
     asked the question or the words are a verbatim Tier-1 command;
  3. the second garbled clip in a row, which used to be a
     cooldown-suppressed earcon, i.e. silence.
"""
from __future__ import annotations

import time

import pytest
from types import SimpleNamespace

import jarvis.app as app_mod
from jarvis.config import CONFIG
from jarvis.transcriber import MIN_AVG_LOGPROB, TranscribeResult
from tests.test_app_wiring import build, paths, seams  # noqa: F401  (fixtures)


@pytest.fixture
def app(build):          # noqa: F811 - the shared factory, one app per test
    return build()


def _result(text: str, conf: float) -> TranscribeResult:
    """A TranscribeResult the way the transcriber builds one: `accepted`
    is a property over `segments`, so a result with no segments skips the
    gate entirely and would make these tests pass for the wrong reason."""
    return TranscribeResult(text=text, confidence=conf,
                            segments=[(text, conf)])


def _feed(app, monkeypatch, text: str, conf: float):   # noqa: F811
    """Push one clip through _process_audio and return what was dispatched."""
    dispatched: list[tuple] = []
    spoken: list[str] = []
    monkeypatch.setattr(CONFIG, "speaker_verify", False)
    monkeypatch.setattr(app, "_dispatch",
                        lambda t, source, confidence=None:
                        dispatched.append((t, source, confidence)))
    monkeypatch.setattr(app, "_say", lambda text, **kw: spoken.append(text))
    monkeypatch.setattr(app.transcriber, "transcribe",
                        lambda audio: _result(text, conf))
    app._process_audio(object())
    return dispatched, spoken


# --------------------------------------------------- 1. the threshold
#
# Measured over all 196 `Transcribed:` lines of the session log. The band
# -2.90..-0.85 holds 18 real utterances and 1 hallucination; below -2.90
# there is not a single actionable one. Anything that re-tightens this
# past -2.64 ("Yes", the worst-scoring CORRECT transcript in the session)
# starts eating commands again, which is what these cases pin.
HIS_REJECTED_TURNS = [
    ("Belay that", -0.88),                  # #24  20:50:24
    ("scratch that", -0.89),                # #144 21:02:26
    ("volume 40", -0.87),                   # #68  21:16:53
    ("Play my liked songs.", -0.97),        # #67  21:14:59
    ("Full brightness", -0.99),             # #101 21:26:37
    ("Cancel.", -0.92),                     # #20  cancel-alarm follow-up
    ("any emails from blend", -0.93),
    ("Yes. Thank you.", -1.45),             # #33  20:58:31
    ("Yes.", -0.95),                        # #33  20:58:35
    ("max volume", -2.08),                  # #68  20:32:53
    ("Yes", -2.64),                         # the worst true transcript seen
]


@pytest.mark.parametrize("text,conf", HIS_REJECTED_TURNS)
def test_the_short_commands_he_actually_said_are_no_longer_rejected(text, conf):
    assert _result(text, conf).accepted, (
        f"{text!r} at {conf} is a correct transcript of a real command; "
        f"the gate at {MIN_AVG_LOGPROB} threw it away")


@pytest.mark.parametrize("text,conf", [
    ("", -3.15),
    ("certain seal seal seal seal seal", -3.46),
    ("Aaa Open two of the ちょっとおよそ", -5.19),
    ("Meet ki Again君 Zoom Check learn a couple Check look", -6.13),
])
def test_the_real_hallucinations_are_still_rejected(text, conf):
    """The gate still has a job: these are the same session's garbage."""
    assert not _result(text, conf).accepted


# ------------------------------------------- 2. salvage: Jarvis asked
#
# The worst of the nine. "Clear the shopping list." -> "Clear all three
# off your shopping list, sir?" -> "Yes" -> dropped for confidence,
# nothing happened. A yes/no answer to a question Jarvis himself put must
# never die in the gate: he has already promised to act on the next word.
def test_a_yes_answering_his_own_question_survives_a_bad_score(app, monkeypatch):
    # the read-back is live: this is exactly what _ask_destructive parks
    ran: list[str] = []
    app.commander._pending_destructive = (
        lambda: ran.append("cleared"), "Clear all three, sir?", time.monotonic())
    assert app._question_open(app.commander), "fixture did not arm the question"

    dispatched, _ = _feed(app, monkeypatch, "Yes.", -3.9)   # far below the gate

    assert [t for t, _s, _c in dispatched] == ["Yes."], (
        "the answer to Jarvis's own question was dropped for low confidence")


def test_with_no_question_open_the_same_garble_is_still_refused(app, monkeypatch):
    """The salvage is narrow: it is the open question that rescues the
    words, not a blanket amnesty for anything sub-threshold."""
    app.commander._pending_destructive = None
    dispatched, spoken = _feed(app, monkeypatch, "Yes.", -3.9)

    assert dispatched == []
    assert spoken == [app_mod.SAY_AGAIN_LINE]


def test_the_routers_hand_it_to_claude_question_holds_the_floor(app):
    """#32: "Cross the second one off the list." ended in "Shall I hand
    that to Claude, sir?" (21:57:30) -- a question Jarvis asked, but one
    the router owns, so question_open() did not know about it and the
    answer got the strict gate."""
    assert not app.commander.question_open()
    # an unconfident "claude" guess is what parks the ask (router._tie_break)
    app.services.router.classify = lambda text, timeout=None: ("claude", 0.2)
    decision = app.services.router.route("cross the second one off the list")
    assert decision.kind == "ask", f"expected the question, got {decision.kind}"
    assert app.services.router.pending() is not None, "no ask was parked"
    assert app.commander.question_open(), (
        "an unanswered 'Shall I hand that to Claude, sir?' is an open question")


def test_the_uncertain_intent_card_also_rescues_its_answer(app, monkeypatch):
    """"Was that for me?" is app-side, not a commander rung, and its whole
    point is that the next word resolves it."""
    app.commander._pending_destructive = None
    app._pending_uncertain = {"req-1": "turn the lights down"}
    dispatched, _ = _feed(app, monkeypatch, "Yes.", -3.9)
    assert [t for t, _s, _c in dispatched] == ["Yes."]


# ------------------------------------ 2b. salvage: a verbatim Tier-1 command
# "cancel my alarm" is #20 and "full brightness" is #101 -- both are
# ASSISTANT_TIER1 matches, so the classifier is already spared them; the
# confidence gate was overruling them a step earlier.
@pytest.mark.parametrize("text,conf", [
    ("cancel my alarm", -4.2),
    ("full brightness", -0.99),
])
def test_an_exact_tier1_command_is_not_second_guessed(app, monkeypatch, text, conf):
    """A confidence score has no business overruling an exact whole-
    utterance match: if the words ARE "cancel my alarm", they are."""
    assert app.commander._match_assistant(text), \
        f"fixture: expected a Tier-1 matcher for {text!r}"
    dispatched, _ = _feed(app, monkeypatch, text, conf)
    assert [t for t, _s, _c in dispatched] == [text]


def test_character_salad_does_not_match_a_tier1_command(app, monkeypatch):
    """The Tier-1 probe is a whole-utterance matcher, so the rescue cannot
    be tripped by a hallucination that merely contains a command word."""
    junk = "ọch Zoom Check learn a couple cancel my Check look"
    dispatched, spoken = _feed(app, monkeypatch, junk, -6.13)
    assert dispatched == []
    assert spoken == [app_mod.SAY_AGAIN_LINE]


# ------------------------------- 3. the second strike is not silence
#
# "Say that again, sir?" then, on the next garble, an EARCON -- and
# _nudge rate-limits the earcon on a 30 s cooldown, so the honest outcome
# was usually nothing at all. That is indistinguishable from being
# ignored, which is how #33 read to Hunter.
def test_a_second_garbled_clip_says_so_instead_of_going_quiet(app, monkeypatch):
    app.commander._pending_destructive = None
    app._turn_from_wake = True
    app._last_nudge_ts = time.monotonic()      # the earcon would be suppressed

    _feed(app, monkeypatch, "Aaa Open two of the", -5.19)      # first strike
    dispatched, spoken = _feed(app, monkeypatch, "Kamen-427.st", -4.85)

    assert dispatched == []
    assert spoken == [app_mod.NOT_CAUGHT_LINE], \
        "the second rejection in a row left him with silence"


def test_the_second_strike_does_not_re_open_the_microphone(app, monkeypatch):
    """The reason the second strike was mute in the first place: a
    television reaching the follow-up mic loops the exchange. Speaking is
    safe, re-arming the mic is not."""
    app.commander._pending_destructive = None
    app._turn_from_wake = True
    _feed(app, monkeypatch, "Aaa Open two of the", -5.19)
    app._followup_after_speech = False
    _feed(app, monkeypatch, "Kamen-427.st", -4.85)
    assert app._followup_after_speech is False


def test_an_accepted_turn_clears_the_strike_count(app, monkeypatch):
    app.commander._pending_destructive = None
    app._turn_from_wake = True
    _feed(app, monkeypatch, "Aaa Open two of the", -5.19)
    _feed(app, monkeypatch, "what time is it", -0.3)
    assert app._say_again_count == 0


# ------------------------------------------------ the event the UI reads
def test_a_salvaged_transcript_is_not_published_as_rejected(app, monkeypatch):
    """status[warn]: Rejected (confidence) is what he SAW while the command
    was in fact running; the Transcribed event must agree with the outcome."""
    from jarvis.events import Transcribed, bus

    seen: list[Transcribed] = []
    bus.subscribe(Transcribed, seen.append)
    try:
        app.commander._pending_destructive = (
            lambda: None, "Clear all three, sir?", time.monotonic())
        _feed(app, monkeypatch, "Yes.", -3.9)
        bus.drain()
    finally:
        bus.unsubscribe(Transcribed, seen.append)

    assert seen and seen[-1].accepted is True
    assert seen[-1].reject_reason == ""


# ------------------------------------------------------ what files, refuses
# Added after review: the salvage treated an open question as a licence to
# dispatch anything, and a flashcard FILES what it is given.
def test_a_live_flashcard_is_never_salvaged_into(app):
    """_quiz_store.record marks the card wrong and burns it, so a garbled
    answer must cost him a repeat and nothing else -- the same objection
    that already excludes the open debrief."""
    app.commander = SimpleNamespace(question_open=lambda: True,
                                    _pending_quiz={"id": "card-1"},
                                    _match_assistant=lambda t: None)
    assert app._salvage_low_confidence("mmm\u0435\u043d-4222,902,902,2 de la", -5.68) == ""
    assert app._salvage_low_confidence("Yes.", -3.0) == ""
    app.commander._pending_quiz = None
    assert app._salvage_low_confidence("Yes.", -3.0) == "answering a question Jarvis asked"
