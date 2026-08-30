"""A briefing turn must show what Jarvis SAID, not only the card.

Live, 2026-08-29 22:14: "what's going on on Monday, and give me my daily
brief." The model called get_calendar and get_briefing, the reply covered
both, and all of it was spoken -- but the transcript showed only the
briefing card. _on_brain_tags publishes no JarvisReply when a BRIEFING tag is
present (the card is the reply), and _ev_briefing rendered ev.sections and
dropped ev.spoken, so the Monday half of what he said never appeared.

No Tk here: the handlers are called unbound on a stub window, which is how
they run anyway -- they touch only self.transcript and self._utter_ts.
"""
import time
from types import SimpleNamespace

from jarvis.events import BriefingReady, JarvisReply
from jarvis.ui.main_window import MainWindow


class FakeTranscript:
    def __init__(self):
        self.calls = []

    def add_jarvis(self, text, rtt=None):
        self.calls.append(("jarvis", text, rtt))

    def add_briefing(self, sections):
        self.calls.append(("card", sections))


def _win(utter_ts=None):
    """The two attributes the handlers touch, plus the real _take_rtt."""
    w = SimpleNamespace(transcript=FakeTranscript(), _utter_ts=utter_ts)
    w._take_rtt = lambda: MainWindow._take_rtt(w)
    return w


def test_the_spoken_sentence_is_shown_before_the_card():
    w = _win(utter_ts=time.monotonic() - 2.0)
    sections = {"weather": "72°F", "news": []}
    MainWindow._ev_briefing(w, BriefingReady(
        sections=sections,
        spoken="On Monday you have Biosensors at nine ten. It is a clear evening, sir."))
    kinds = [c[0] for c in w.transcript.calls]
    assert kinds == ["jarvis", "card"], kinds
    assert w.transcript.calls[0][1].startswith("On Monday you have Biosensors")
    assert 1.5 < w.transcript.calls[0][2] < 3.0, "the turn's RTT belongs on the spoken line"
    assert w.transcript.calls[1][1] is sections
    assert w._utter_ts is None, "the utterance stamp must be consumed once"


def test_a_card_with_nothing_spoken_is_still_just_a_card():
    w = _win(utter_ts=time.monotonic())
    MainWindow._ev_briefing(w, BriefingReady(sections={"weather": "x"}, spoken=""))
    assert [c[0] for c in w.transcript.calls] == ["card"]
    assert w._utter_ts is None


def test_plain_replies_keep_their_rtt_behaviour():
    w = _win(utter_ts=time.monotonic() - 1.0)
    MainWindow._ev_reply(w, JarvisReply(text="Right away, sir.", speak=True))
    (kind, text, rtt), = w.transcript.calls
    assert kind == "jarvis" and text == "Right away, sir." and 0.5 < rtt < 2.0
    MainWindow._ev_reply(w, JarvisReply(text="", speak=False))
    assert len(w.transcript.calls) == 1, "an empty reply adds nothing"
