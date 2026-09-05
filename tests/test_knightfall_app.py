"""Knightfall, in the app (Hunter, 2026-09-04: "a and b ... leave
enforcement").

Part 1 -- THE SPOKEN PHRASE DISPATCHES NOTHING. On both paths into the gate
(the admitted turn in ``_process_audio`` and the dropped clip in
``_gate_rescue``) a consumed decision ends the turn: the line is spoken,
the mic re-opens, the ledger closes the turn as ``gate:phrase``, the bus
sees ``REDACTED_TEXT`` and nothing else -- no commander, no model, no
UserUtterance. In shadow as well as in enforce, because that is how he
tests it tonight.

Bound methods on the stand-in tests/test_owner_gate_wiring.py builds: no
microphone, no lens, no Whisper, no Tk. The fake phrase never appears in a
test name.
"""
import logging
import threading
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
from jarvis import gate as gate_mod
from jarvis.events import Transcribed, UserUtterance, bus
from tests.test_owner_gate_wiring import FAKE_PHRASE, MATCHED, _stand_in

SAID = "Xxx, not a real phrase xxx."


def _result(text, accepted=True):
    return SimpleNamespace(text=text, confidence=-0.2, accepted=accepted,
                           looping=False)


def _app(tmp_path, *, mode="shadow", phrase=True, said=SAID, rejected=False):
    a = _stand_in(tmp_path, mode=mode, phrase=phrase, said=said)
    a.dispatched = []
    a.nudges = []
    a._process_audio = app_mod.JarvisApp._process_audio.__get__(a)
    a._gate_judge = app_mod.JarvisApp._gate_judge.__get__(a)
    a._gate_consumed = app_mod.JarvisApp._gate_consumed.__get__(a)
    a._take_speculation = lambda: None
    a._decode_clip = lambda audio: (audio, MATCHED, rejected,
                                    None if rejected else _result(said))
    a._audio_busy = threading.Event()
    a._audio_busy.set()
    a._nudge = a.nudges.append
    a._maybe_learn_voice = lambda audio, stats: None
    a._dispatch = lambda text, source, **k: a.dispatched.append((text, source))
    a._salvage_low_confidence = lambda text, conf: ""
    a._say_again_count = 0
    a._stop_event = SimpleNamespace(t=0.0)
    return a


@pytest.fixture
def events():
    seen = []
    subs = [(Transcribed, seen.append), (UserUtterance, seen.append)]
    for etype, fn in subs:
        bus.subscribe(etype, fn)
    yield seen
    for etype, fn in subs:
        bus.unsubscribe(etype, fn)


# ------------------------------------------------ the admitted path
@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_a_recognised_owner_saying_the_phrase_dispatches_nothing(
        tmp_path, events, mode, caplog):
    a = _app(tmp_path, mode=mode)
    with caplog.at_level(logging.DEBUG):
        a._process_audio(object())
    assert a.dispatched == []
    assert a.spoken == [gate_mod.PHRASE_OK_LINE]
    assert a._followup_after_speech is True
    assert a.abandoned == ["gate:phrase"]
    assert [type(e).__name__ for e in events] == ["Transcribed"]
    assert events[0].text == gate_mod.REDACTED_TEXT and events[0].accepted
    assert not a._audio_busy.is_set()
    joined = (caplog.text + " ".join(str(vars(e)) for e in events)).lower()
    assert FAKE_PHRASE.lower() not in joined
    assert "not a real phrase" not in joined
    assert "the phrase opened the floor to hunter" in caplog.text


def test_an_ordinary_turn_still_dispatches_exactly_as_before(tmp_path, events):
    a = _app(tmp_path, said="what time is it")
    a._process_audio(object())
    assert a.dispatched == [("what time is it", "voice")]
    assert a.spoken == [] and a.abandoned == []
    assert [type(e).__name__ for e in events] == ["Transcribed", "UserUtterance"]
    assert events[0].text == "what time is it"
    assert (a._gate_who, a._gate_how) == ("hunter", gate_mod.HOW_VOICE)


def test_the_phrase_is_consumed_even_on_a_low_confidence_transcript(
        tmp_path, events):
    """An exact match after normalisation is stronger evidence than
    Whisper's length-biased score; and NOT checking would leave the phrase
    on the bus as a rejected transcript."""
    a = _app(tmp_path)
    a._decode_clip = lambda audio: (audio, MATCHED, False,
                                    _result(SAID, accepted=False))
    a._process_audio(object())
    assert a.dispatched == [] and a.spoken == [gate_mod.PHRASE_OK_LINE]
    assert events[0].text == gate_mod.REDACTED_TEXT


def test_the_gate_is_asked_once_per_turn_not_twice(tmp_path, monkeypatch):
    """One judge, one derivation: the admitted path reuses the decision."""
    a = _app(tmp_path, said="read me the weather for tomorrow please")
    calls = []
    real = a.gate.judge

    def counting(*args, **kw):
        calls.append(args[1])
        return real(*args, **kw)
    monkeypatch.setattr(a.gate, "judge", counting)
    a._process_audio(object())
    assert len(calls) == 1
    assert a.gate.kdf_calls == 1


def test_without_a_phrase_set_the_admitted_path_is_untouched(tmp_path,
                                                             monkeypatch):
    a = _app(tmp_path, phrase=False, said="what time is it")
    a._process_audio(object())
    assert a.dispatched == [("what time is it", "voice")]
    assert a.gate.kdf_calls == 0


def test_a_gate_that_could_not_be_built_consumes_nothing(tmp_path):
    a = _app(tmp_path)
    a.gate = None
    assert a._gate_judge(SAID, MATCHED) is None
    a._process_audio(object())
    assert a.dispatched == [(SAID, "voice")]


# ------------------------------------------------ the dropped-clip path
def test_an_unrecognised_voice_saying_the_phrase_in_shadow_is_consumed(
        tmp_path):
    """Shadow used to drop a rejected clip without a decode. The phrase
    is the ONE thing that now acts in shadow: one decode, when an owner
    has a phrase set, and nothing else changes."""
    a = _stand_in(tmp_path, mode="shadow", phrase=True, said=SAID)
    assert a._gate_rescue(object(), MATCHED, False) is app_mod.PHRASE_CONSUMED
    assert a.transcriber.calls == 1
    assert a.spoken == [gate_mod.PHRASE_OK_LINE]
    assert a._followup_after_speech is True
    assert a.abandoned == ["gate:phrase"]


def test_shadow_still_rescues_nothing_and_refuses_nothing_out_loud(tmp_path):
    a = _stand_in(tmp_path, mode="shadow", phrase=True, said="hello there")
    assert a._gate_rescue(object(), MATCHED, False) is None
    assert a.transcriber.calls == 1
    assert a.spoken == []
    b = _stand_in(tmp_path, mode="shadow", phrase=False, said="hello there")
    assert b._gate_rescue(object(), MATCHED, False) is None
    assert b.transcriber.calls == 0


def test_a_consumed_dropped_clip_publishes_redacted_and_never_a_rejection(
        tmp_path, events):
    a = _app(tmp_path, mode="enforce", rejected=True)
    a._process_audio(object())
    assert [type(e).__name__ for e in events] == ["Transcribed"]
    assert events[0].text == gate_mod.REDACTED_TEXT and events[0].accepted
    assert a.nudges == [] and a.dispatched == []
    assert a.spoken == [gate_mod.PHRASE_OK_LINE]


def test_the_code_window_rescues_a_dropped_clip_in_enforce(tmp_path, caplog):
    a = _stand_in(tmp_path, mode="enforce", phrase=False, said="read me my mail")
    a.gate.open_window("hunter", gate_mod.HOW_CODE)      # the real clock
    with caplog.at_level(logging.INFO):
        out = a._gate_rescue(object(), MATCHED, False)
    assert out is not None and out is not app_mod.PHRASE_CONSUMED
    assert a.transcriber.calls == 1
    assert "the code leg rescued" in caplog.text


def test_the_code_window_in_shadow_only_says_it_would_have(tmp_path, caplog):
    a = _stand_in(tmp_path, mode="shadow", phrase=False, said="read me my mail")
    a.gate.open_window("hunter", gate_mod.HOW_CODE)      # the real clock
    with caplog.at_level(logging.INFO):
        assert a._gate_rescue(object(), MATCHED, False) is None
    assert "WOULD have rescued" in caplog.text and "code" in caplog.text
    assert a.transcriber.calls == 0
