"""Knightfall, at the gate (Hunter, 2026-09-04 20:50: "a and b ... leave
enforcement").

(a) THE SPOKEN PHRASE IS CONSUMED IN EVERY MODE. Before this, the phrase
was tried only when nobody was recognised, and a recognised owner saying it
had the sentence answered as an ordinary turn -- which, for the phrase he
chose, means the Oracle lane ("is knightfall up") or the model. Now a
phrase-shaped utterance that matches an owner's hash comes back
``consumed=True`` whether or not a leg named him, in shadow as well as in
enforce, and the app dispatches nothing for it.

(b) ONE WINDOW OPENER, ``open_window(who, how)``, used by the phrase and by
the typed code; a turn inside a code-opened window is admitted on the
``code`` leg exactly as a phrase-opened one is admitted on ``grant``.

No mic, no lens, no model, no real registry: the same fake registry in
tmp_path that tests/test_owner_gate.py builds. The fake phrase and code
never appear in a test name.
"""
import logging

import pytest

from jarvis import gate as gt
from jarvis import passphrase as pp
from tests.test_owner_gate import (ABSTAINED, FAKE_PHRASE, MATCHED,
                                   NOT_RUNNING, _gate)

SAID = "Xxx, not a real phrase xxx."       # what Whisper would hand over
WRONG = "xxx not a real phrase yyy"


def _consumed(d):
    return (d.consumed, d.how, d.line, d.redact)


# ------------------------------------------------ (a) the phrase, consumed
@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_a_recognised_owner_saying_the_phrase_has_the_turn_consumed(
        tmp_path, mode):
    """The voice leg named him AND the words are the phrase: the gate
    answers it itself. It must not travel on as a sentence."""
    g = _gate(tmp_path, mode=mode, phrase=True)
    d = g.judge("voice", SAID, stats=MATCHED)
    assert _consumed(d) == (True, gt.HOW_PHRASE, gt.PHRASE_OK_LINE,
                            gt.REDACTED_TEXT)
    assert d.who == "hunter" and d.admit is True


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_an_unrecognised_voice_saying_the_phrase_has_the_turn_consumed(
        tmp_path, mode):
    g = _gate(tmp_path, mode=mode, phrase=True)
    d = g.judge("voice", SAID, stats=MATCHED, rejected=True)
    assert _consumed(d) == (True, gt.HOW_PHRASE, gt.PHRASE_OK_LINE,
                            gt.REDACTED_TEXT)
    assert d.who == "hunter"


def test_the_phrase_is_consumed_even_when_no_leg_is_running(tmp_path):
    """Verification off, camera dark: still his phrase, still consumed --
    otherwise it would name the Oracle service on a box with no legs."""
    g = _gate(tmp_path, mode="shadow", phrase=True)
    d = g.judge("voice", SAID, stats=NOT_RUNNING, face_running=False)
    assert d.consumed is True and d.how == gt.HOW_PHRASE


def test_the_consumed_turn_opens_the_window_for_grant_s(tmp_path):
    g = _gate(tmp_path, mode="enforce", phrase=True)
    g.judge("voice", SAID, stats=MATCHED, now=0.0)
    after = g.judge("voice", "read me my mail", stats=MATCHED, rejected=True,
                    now=5.0)
    assert after.admit is True and after.how == gt.HOW_GRANT
    assert after.consumed is False and after.redact == ""
    shut = g.judge("voice", "read me my mail", stats=MATCHED, rejected=True,
                   now=gt.GRANT_S + 1)
    assert shut.admit is False


def test_shadow_logs_the_window_with_the_mode_and_only_who_and_which_path(
        tmp_path, caplog):
    g = _gate(tmp_path, mode="shadow", phrase=True)
    with caplog.at_level(logging.INFO):
        g.judge("voice", SAID, stats=MATCHED, rejected=True)
    assert "the phrase opened the floor to hunter for 300s (mode=shadow)" \
        in caplog.text
    assert FAKE_PHRASE.lower() not in caplog.text.lower()
    assert "not a real phrase" not in caplog.text.lower()


def test_an_ordinary_sentence_is_never_consumed(tmp_path):
    g = _gate(tmp_path, mode="enforce", phrase=True)
    d = g.judge("voice", "what time is it", stats=MATCHED)
    assert d.consumed is False and d.admit is True and d.how == gt.HOW_VOICE
    assert d.line == "" and d.redact == ""
    short = g.judge("voice", "Yes.", stats=ABSTAINED)
    assert short.consumed is False and short.admit is True


def test_a_wrong_phrase_pays_no_window(tmp_path):
    g = _gate(tmp_path, mode="enforce", phrase=True)
    d = g.judge("voice", WRONG, stats=MATCHED, rejected=True, now=0.0)
    assert d.consumed is False and d.admit is False
    assert g.judge("voice", "hello", stats=MATCHED, rejected=True,
                   now=1.0).admit is False
    # ...and a recognised owner's near miss is just his sentence
    d = g.judge("voice", WRONG, stats=MATCHED, now=2.0)
    assert d.consumed is False and d.admit is True and d.how == gt.HOW_VOICE


def test_without_a_phrase_set_nothing_is_consumed_and_nothing_is_derived(
        tmp_path):
    g = _gate(tmp_path, mode="enforce", phrase=False)
    d = g.judge("voice", SAID, stats=MATCHED)
    assert d.consumed is False and d.how == gt.HOW_VOICE
    assert g.kdf_calls == 0


def test_the_consumed_decision_carries_nothing_of_the_phrase(tmp_path):
    g = _gate(tmp_path, mode="shadow", phrase=True)
    d = g.judge("voice", SAID, stats=MATCHED)
    for value in vars(d).values():
        assert "not a real phrase" not in str(value).lower()
        assert FAKE_PHRASE.lower() not in str(value).lower()


# ------------------------------------------------ the limiter, kept as it is
def test_a_strangers_attempts_are_still_limited_exactly_as_before(tmp_path):
    g = _gate(tmp_path, mode="enforce", phrase=True)
    assert (pp.PHRASE_LIMIT, pp.PHRASE_WINDOW) == (5, 300.0)
    for i in range(pp.PHRASE_LIMIT):
        g.judge("voice", WRONG, stats=MATCHED, rejected=True, now=float(i))
    d = g.judge("voice", SAID, stats=MATCHED, rejected=True, now=10.0)
    assert d.consumed is False and d.admit is False
    d = g.judge("voice", SAID, stats=MATCHED, rejected=True,
                now=10.0 + pp.PHRASE_WINDOW + 1)
    assert d.consumed is True


def test_a_recognised_owners_ordinary_sentences_never_burn_an_attempt(
        tmp_path):
    """THE LINE THAT KEEPS THE FEATURE USABLE. The phrase check now runs
    on every phrase-shaped turn; if his own sentences counted against the
    five-per-five-minutes limiter, the sixth sentence would close the
    phrase for the rest of the window and "Knightfall protocol" would
    reach the Oracle lane. A recognised owner is not attempting anything."""
    g = _gate(tmp_path, mode="enforce", phrase=True)
    for i in range(3 * pp.PHRASE_LIMIT):
        g.judge("voice", "read me the weather for tomorrow please",
                stats=MATCHED, now=float(i))
    ok, wait = g.phrase_attempts.allow(now=100.0)
    assert ok is True and wait == 0.0
    assert g.judge("voice", SAID, stats=MATCHED, now=101.0).consumed is True


def test_the_recognised_path_pays_the_derivation_but_only_when_phrase_shaped(
        tmp_path):
    """The cost, stated: a recognised owner's phrase-shaped sentence pays
    one scrypt per owner with a phrase (~20 ms measured on this box); a
    short one ("yes", "no", "pause") pays nothing, as before."""
    g = _gate(tmp_path, mode="enforce", phrase=True)
    g.judge("voice", "no", stats=MATCHED)
    g.judge("voice", "pause", stats=MATCHED)
    assert g.kdf_calls == 0
    g.judge("voice", "what time is it", stats=MATCHED)
    assert g.kdf_calls == 1


# ------------------------------------------------ (b) one window opener
def test_open_window_is_what_the_phrase_path_calls(tmp_path, monkeypatch):
    g = _gate(tmp_path, mode="shadow", phrase=True)
    calls = []
    real = g.open_window

    def spy(who, how, now=None):
        calls.append((who, how))
        return real(who, how, now=now)
    monkeypatch.setattr(g, "open_window", spy)
    g.judge("voice", SAID, stats=MATCHED, rejected=True, now=0.0)
    assert calls == [("hunter", gt.HOW_PHRASE)]


def test_open_window_from_the_typed_path_admits_on_the_code_leg(tmp_path):
    """The code opens the same window the phrase does; the turn inside it
    is attributed to the code so the log says which path let him in."""
    assert gt.HOW_CODE == "code"
    g = _gate(tmp_path, mode="enforce", phrase=True)
    g.open_window("hunter", gt.HOW_CODE, now=0.0)
    d = g.judge("voice", "read me my mail", stats=MATCHED, rejected=True,
                now=5.0)
    assert d.admit is True and d.who == "hunter" and d.how == gt.HOW_CODE
    assert d.redact == "" and d.consumed is False
    assert g.judge("voice", "hello", stats=MATCHED, rejected=True,
                   now=gt.GRANT_S + 1).admit is False


def test_the_code_window_is_logged_with_who_and_path_and_mode(tmp_path,
                                                                caplog):
    g = _gate(tmp_path, mode="shadow")
    with caplog.at_level(logging.INFO):
        g.open_window("hunter", gt.HOW_CODE)
    assert "the code opened the floor to hunter for 300s (mode=shadow)" \
        in caplog.text


def test_granted_is_unchanged_it_answers_who_and_nothing_else(tmp_path):
    g = _gate(tmp_path, mode="enforce")
    assert g._granted(0.0) == ""
    g.open_window("hunter", gt.HOW_CODE, now=0.0)
    assert g._granted(1.0) == "hunter"
    assert g._granted(gt.GRANT_S + 1) == ""


def test_the_code_leg_is_admitted_like_grant_in_the_rescue_set():
    """The app rescues a dropped clip on the face, grant and code legs."""
    from pathlib import Path
    src = (Path(gt.__file__).parent / "app.py").read_text()
    assert "gate_mod.HOW_CODE" in src


def test_no_new_threshold_and_no_lock_wording():
    from tests.test_owner_gate import test_the_gate_owns_no_voice_threshold
    from tests.test_owner_registry import no_overstatement
    test_the_gate_owns_no_voice_threshold(None)
    no_overstatement(gt)
