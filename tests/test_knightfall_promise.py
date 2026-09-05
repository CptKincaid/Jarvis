"""WHAT THE CONSUMED TURN LEAVES BEHIND: the glass, and the promise.

Three claims are measured here, and each was RED on ecb5abc before the
commit that carries this file.

1. THE WORDS STAY ON SCREEN. ``_partial_loop`` publishes
   ``PartialText(text=<the words>)`` while he speaks and the transcript pane
   draws a ghost card. ``_gate_consumed`` then publishes
   ``Transcribed(text=REDACTED_TEXT, accepted=True)`` -- and
   ``MainWindow._ev_transcribed`` clears the partial ONLY when the event is
   NOT accepted or is empty, while ``add_user`` (the pane's other clearer)
   never runs because no ``UserUtterance`` follows. So the redaction the log
   was so careful about is undone by the screen it is standing in front of.

2. "I'M LISTENING" IS NOT TRUE IN SHADOW -- and shadow is his live mode.
   The phrase opens a five-minute window, the next clip is dropped by the
   SPEAKER FILTER (which is not the gate and does not care what mode it is
   in), and ``_gate_rescue_inner`` refuses to act on anything but
   ``MODE_ENFORCE``. The window is honoured in enforce and inert in shadow,
   so the sentence he hears is a promise the code keeps in one mode of
   three. The same is true of the window the TYPED CODE opens.

3. THE LIMITER IS DISARMED BY ANY NAMED FACE. ``judge`` passes
   ``limited=not named.who``, so a KNOWN non-owner in view -- a guest, the
   cleaner, anybody the camera has a row for -- turns the five-per-window
   passphrase limiter off entirely and every guess pays a key derivation.

No real secret is anywhere near this file: the phrase under test is
tests/test_owner_gate_wiring.FAKE_PHRASE, which is x's.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis import gate as gate_mod
from jarvis import passphrase as pp
from jarvis.events import PartialText, Transcribed, UserUtterance, bus
from jarvis.identity import ROLE_KNOWN
from tests.test_knightfall_log_leak import SPOKEN, _app, _audio, _real_transcriber
from tests.test_owner_gate_wiring import MATCHED, _stand_in


# ================================================== 1. the glass
class _Pane:
    """The transcript pane as MainWindow drives it: every call, in order."""

    def __init__(self):
        self.calls: list[tuple] = []

    def show_partial(self, text):
        self.calls.append(("show", text))

    def clear_partial(self):
        self.calls.append(("clear", ""))

    def add_user(self, text, confidence=None):
        self.calls.append(("user", text))

    def add_jarvis(self, text, rtt=None):
        self.calls.append(("jarvis", text))

    def clear_all(self):
        self.calls.append(("clear_all", ""))

    def showing(self) -> str:
        """The words currently on the glass, by replaying the calls."""
        text = ""
        for what, arg in self.calls:
            if what == "show":
                text = arg
            elif what in ("clear", "clear_all"):
                text = ""
            elif what == "user":
                text = arg
        return text


@pytest.fixture
def pane():
    """The REAL MainWindow handlers, bound to a stand-in, on the REAL bus --
    no Tk root, which is the idiom the rest of the UI tests use."""
    from jarvis.ui import main_window as mw

    w = SimpleNamespace()
    w.transcript = _Pane()
    w._last_confidence = None
    w._utter_ts = None
    w.set_status = lambda *a, **k: None
    w._note_activity = lambda: None
    for name in ("_ev_partial", "_ev_transcribed", "_ev_user"):
        setattr(w, name, getattr(mw.MainWindow, name).__get__(w))
    subs = [(PartialText, w._ev_partial), (Transcribed, w._ev_transcribed),
            (UserUtterance, w._ev_user)]
    for etype, fn in subs:
        bus.subscribe(etype, fn)
    yield w.transcript
    for etype, fn in subs:
        bus.unsubscribe(etype, fn)


@pytest.mark.parametrize("mode", ["shadow", "enforce", "off"])
def test_the_consumed_turn_takes_the_words_off_the_glass(
        tmp_path, monkeypatch, pane, mode):
    """The ghost card carries the plaintext; consuming the turn must take it
    down. Published exactly as ``_partial_loop`` publishes it."""
    a = _app(tmp_path, monkeypatch, mode=mode, matches=True)
    bus.publish(PartialText(text=SPOKEN))
    assert pane.showing() == SPOKEN, "the ghost card is the premise"
    a._process_audio(_audio())
    assert pane.showing() == "", (
        "the spoken phrase is still on screen: %s" % (pane.calls,))
    assert not any(SPOKEN.lower() in str(arg).lower()
                   for what, arg in pane.calls if what in ("user", "jarvis"))


def test_an_ordinary_turn_still_replaces_the_ghost_with_the_sentence(
        tmp_path, monkeypatch, pane):
    """The clearing must not cost an ordinary turn its transcript card."""
    said = "what time is it"
    a = _app(tmp_path, monkeypatch, mode="shadow", matches=True, said=said)
    bus.publish(PartialText(text=said))
    a._process_audio(_audio())
    assert pane.showing() == said


# ================================================== 2. the promise
def _second_clip(a, said, monkeypatch, tmp_path):
    """The turn AFTER the phrase: the speaker filter drops it, exactly as it
    dropped the one that made him say the phrase in the first place."""
    a.transcriber = _real_transcriber(said, monkeypatch, tmp_path)
    a.speaker.matches = False
    a.dispatched.clear()
    a._process_audio(_audio())
    return a.dispatched


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_the_phrase_opens_a_floor_that_actually_takes_the_next_clip(
        tmp_path, monkeypatch, mode):
    """"I'm listening." has to be true. The window the phrase opens must
    rescue the clip the speaker filter drops, in EVERY mode that opens
    one -- his live mode is shadow, where it did nothing at all."""
    a = _app(tmp_path, monkeypatch, mode=mode, matches=True)
    a._process_audio(_audio())                     # he says the phrase
    assert a.spoken and a.spoken[0] == gate_mod.PHRASE_OK_LINE
    assert _second_clip(a, "what time is it", monkeypatch, tmp_path) == [
        ("what time is it", "voice")], (
        "the phrase said it was listening and then dropped his next clip")


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_the_typed_code_opens_the_same_floor(tmp_path, monkeypatch, mode):
    """The drawer's code opens the window through the same one opener, so
    it must buy the same thing the phrase buys."""
    a = _app(tmp_path, monkeypatch, mode=mode, matches=True)
    a.gate.open_window("hunter", gate_mod.HOW_CODE)
    assert _second_clip(a, "what time is it", monkeypatch, tmp_path) == [
        ("what time is it", "voice")]


def test_off_does_not_promise_a_floor_it_never_opened(tmp_path, monkeypatch):
    """MODE_OFF consumes the phrase (so it stays off the bus) and opens
    NOTHING -- deliberately, see Gate._phrase_consumed. Then it must not say
    the sentence that means a floor was opened: the next clip is dropped by
    the speaker filter exactly as before, measured here."""
    a = _app(tmp_path, monkeypatch, mode="off", matches=True)
    a._process_audio(_audio())
    assert a.dispatched == []                      # still consumed
    assert _second_clip(a, "what time is it", monkeypatch, tmp_path) == [], (
        "off rescues nothing -- that is the premise of this test")
    assert a.spoken[0] != gate_mod.PHRASE_OK_LINE, (
        "off promised a floor it did not open")


# ================================================== 3. the limiter
def _guesses(gate, n, **kw):
    """n wrong phrase-shaped guesses; the key derivations they cost."""
    for i in range(n):
        gate.judge("voice", "open the door please number %d" % i,
                   stats={"total": 2, "matched": 0, "scores": [0.1, 0.1]},
                   rejected=True, **kw)
    return gate.kdf_calls


def test_a_known_stranger_in_view_cannot_disarm_the_phrase_limiter(tmp_path):
    """``limited=not named.who`` reads "somebody was named" as "the owner was
    named". A KNOWN non-owner in front of the lens is somebody -- and turning
    the limiter off for them hands a guesser unlimited attempts and one
    scrypt each, with the owner nowhere in the room."""
    a = _stand_in(tmp_path, mode="enforce", phrase=True, known=True)
    nobody = _guesses(a.gate, 40, face="", face_running=False)
    a2 = _stand_in(tmp_path, mode="enforce", phrase=True, known=True)
    known = _guesses(a2.gate, 40, face="heather", face_running=True)
    assert nobody <= pp.PHRASE_LIMIT, "the limiter's own baseline"
    assert known <= pp.PHRASE_LIMIT, (
        "a KNOWN face disarmed the limiter: %d derivations for 40 guesses, "
        "against %d with nobody named" % (known, nobody))


def test_the_owner_himself_is_still_not_rate_limited(tmp_path):
    """The reason the exemption exists at all: his own sentences must not
    burn attempts, or the sixth in five minutes closes his way back in."""
    a = _stand_in(tmp_path, mode="enforce", phrase=True)
    n = 0
    for i in range(12):
        a.gate.judge("voice", "please open the door number %d" % i,
                     stats=dict(MATCHED), face="", face_running=False)
        n = a.gate.kdf_calls
    assert n > pp.PHRASE_LIMIT, (
        "the owner's own turns must not be rate limited (%d)" % n)


def test_a_known_person_is_not_promoted_by_a_phrase_hash(tmp_path):
    """Unchanged rule, asserted beside the limiter change so a future edit
    cannot quietly widen it: only an OWNER's phrase opens anything."""
    a = _stand_in(tmp_path, mode="enforce", phrase=True, known=True)
    d = a.gate.judge("voice", "hello there", stats=dict(MATCHED),
                     face="heather", face_running=True)
    assert d.role != ROLE_KNOWN or not d.consumed
