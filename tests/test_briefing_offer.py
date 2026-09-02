"""The first-wake briefing OFFERS itself and waits for a yes.

2026-09-02 14:29:30, live log: Hunter said five words ("Say hello to my
family.") and got a greeting, then spoken telemetry ("Yesterday: 24 turns,
median wait 1.4 seconds ... I dropped 7 clips of yours at the speaker
gate"), then a full weather/calendar/deadlines/news briefing -- about 40
seconds of monologue he had not asked for. His ruling: "He should offer.
We will have auto briefing changed later with presence and camera stuff."

All three deliveries in the two retained logs (2026-08-31 14:33, 09-01
15:00, 09-02 14:29) followed something else entirely, and all three landed
in the AFTERNOON while the code asked the model for "my morning briefing".

Real JarvisApp (hardware and peers stubbed, as in tests/test_turn_flow_fixes)
and a real Commander for the answer rung. No audio, no network, no model.
"""
import json
import threading
import time as _t
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.app as app_mod
from jarvis.commander import (BRIEFING_DECLINED_LINE, Commander,
                              IntentClassifier)
from jarvis.config import CONFIG
from jarvis.tools.briefing import OFFER_TTL_S


# ===================================================================
# the app half: arming, offering, marking the day
# ===================================================================
def _app(monkeypatch, tmp_path, **over):
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(get=lambda k, d=None: {"briefing.on_first_wake": True,
                                                         "briefing.after": "06:00"}.get(k, d),
                                  user_name="Hunter")
    a._init_assistant_state()
    a.recorder = SimpleNamespace(endpointer=object(), recording=False,
                                 start=lambda followup=False, **k: a.starts.append(followup))
    a.starts = []
    a._audio_busy = threading.Event()
    a._turn_busy = threading.Event()
    a._pending_uncertain = {}
    a._uncertain_lock = threading.Lock()
    a._last_source, a._last_user_text = "voice", "hello"
    a.said = []
    a._say = lambda text, proactive=False, kind="message": a.said.append(text)
    a.context = SimpleNamespace(add_exchange=lambda u, j: None)
    a.chats = []
    a.services = SimpleNamespace(
        brain=SimpleNamespace(chat=lambda t, **kw: a.chats.append((t, kw))),
        briefing_offer=None)
    a.brain = SimpleNamespace(is_busy=False)
    a.marks = []
    a.turns = SimpleNamespace(mark=lambda *x, **k: a.marks.append(x),
                              abandon=lambda r: None)
    a._briefing_state_path = lambda: tmp_path / "briefing.json"
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _clock(monkeypatch, hh, mm=0):
    """Freeze app.py's datetime -- both the due-check and the arc word the
    briefing asks the model for read it."""
    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 2, hh, mm, 0)
    monkeypatch.setattr(app_mod, "datetime", _Clock)


def _arm(a):
    """A normal answered voice turn: the briefing arms behind it."""
    a._after_dispatch("say hello to my family", "voice",
                      SimpleNamespace(reply="Hello to your family, sir.", speak=True,
                                      done=True, ack=False, status="Chat"))


class TestTheOfferReplacesTheDelivery:
    def test_a_settled_burst_offers_instead_of_reading_the_briefing(self, monkeypatch, tmp_path):
        a = _app(monkeypatch, tmp_path)
        _clock(monkeypatch, 14, 29)
        _arm(a)
        assert a._briefing_pending
        a._after_speech()
        assert a.said == ["Shall I run your afternoon briefing, sir?"], a.said
        assert a.chats == [], "nothing may be read out until he says yes"
        assert not a._briefing_pending

    def test_the_offer_arms_the_mic_so_yes_needs_no_wake_word(self, monkeypatch, tmp_path):
        a = _app(monkeypatch, tmp_path)
        _clock(monkeypatch, 14, 29)
        _arm(a)
        a._after_speech()
        assert a._followup_after_speech, "a question nobody listens for is a dead end"
        assert isinstance(a.services.briefing_offer, dict)
        assert callable(a.services.briefing_offer.get("deliver"))

    def test_the_offer_names_the_hour_it_is_actually_made_in(self, monkeypatch, tmp_path):
        for hour, word in ((7, "morning"), (14, "afternoon"), (19, "evening")):
            a = _app(monkeypatch, tmp_path)
            _clock(monkeypatch, hour)
            (tmp_path / "briefing.json").unlink(missing_ok=True)
            _arm(a)
            a._after_speech()
            assert a.said == [f"Shall I run your {word} briefing, sir?"], (hour, a.said)

    def test_the_wind_down_is_undone_by_the_offer_not_by_the_yes(self, monkeypatch, tmp_path):
        """"The first wake of the day is the morning even when he never
        said the word" -- true whether or not he wants the news read to
        him. Hung off the delivery, a "no" left the house in night mode
        all day."""
        a = _app(monkeypatch, tmp_path)
        _clock(monkeypatch, 7)
        restored = []
        a.winddown = SimpleNamespace(restore=lambda: restored.append(1))
        _arm(a)
        a._after_speech()
        assert restored == [1]

    def test_the_offer_waits_while_another_question_is_on_the_table(self, monkeypatch, tmp_path):
        a = _app(monkeypatch, tmp_path)
        _clock(monkeypatch, 14, 29)
        a.commander = SimpleNamespace(question_open=lambda: True)
        _arm(a)
        a._after_speech()
        assert a.said == [] and a._briefing_pending, \
            "two open questions at once is how a yes lands on the wrong one"
        a.commander = SimpleNamespace(question_open=lambda: False)
        a._after_speech()
        assert a.said == ["Shall I run your afternoon briefing, sir?"]


class TestAskedOnceIsTheDay:
    def test_making_the_offer_closes_the_day(self, monkeypatch, tmp_path):
        """The day is marked when the question is PUT, not when it is
        answered: _after_dispatch re-arms after any answered turn, so an
        offer that only marked on delivery would ask again seconds after
        every decline. Nagging is the failure mode he complained about."""
        a = _app(monkeypatch, tmp_path)
        _clock(monkeypatch, 14, 29)
        _arm(a)
        a._after_speech()
        assert json.loads((tmp_path / "briefing.json").read_text())["delivered"] == "2026-09-02"
        assert not a._briefing_due()
        _arm(a)
        assert not a._briefing_pending, "one offer a day, not one an utterance"

    def test_quiet_hours_still_suppress_the_whole_thing(self, monkeypatch, tmp_path):
        a = _app(monkeypatch, tmp_path)
        _clock(monkeypatch, 6, 30)
        a.quiet = SimpleNamespace(is_quiet=lambda: True)
        assert not a._briefing_due()
        assert a._briefing_block() == "quiet hours"
        _arm(a)
        a._after_speech()
        assert a.said == [] and not (tmp_path / "briefing.json").exists()

    def test_the_state_file_still_prevents_a_second_pass(self, monkeypatch, tmp_path):
        a = _app(monkeypatch, tmp_path)
        _clock(monkeypatch, 14, 29)
        (tmp_path / "briefing.json").write_text(json.dumps({"delivered": "2026-09-02"}))
        assert a._briefing_block() == "already raised today"
        _arm(a)
        assert not a._briefing_pending

    def test_the_gates_are_a_list_a_presence_signal_can_join(self, monkeypatch, tmp_path):
        """Hunter: "We will have auto briefing changed later with presence
        and camera stuff." The eventual "he is here and awake" condition
        must be one more entry, not a rewrite of the caller."""
        a = _app(monkeypatch, tmp_path)
        _clock(monkeypatch, 5, 0)
        assert a._briefing_block() == "before the hour"
        names = [n for n, _ in a._briefing_gates()]
        assert names == ["switched off", "before the hour", "quiet hours",
                         "already raised today"]
        a._briefing_gates = lambda: (("nobody home", lambda now: False),)
        assert a._briefing_block() == "nobody home" and not a._briefing_due()


class TestWhatIsActuallyDelivered:
    def test_the_model_is_asked_for_the_briefing_of_this_hour(self, monkeypatch, tmp_path):
        """brain.chat("my morning briefing") was asked at 14:33, 15:00 and
        14:29 in the two retained logs. It has never once been morning."""
        a = _app(monkeypatch, tmp_path)
        _clock(monkeypatch, 14, 29)
        assert a._deliver_first_wake_briefing() is True
        assert a.chats == [("my afternoon briefing", {"force_tool": "get_briefing"})]
        assert a.said == ["Your briefing for today, sir."]

    def test_the_day_review_telemetry_is_not_read_out_unprompted(self, monkeypatch, tmp_path):
        """"Yesterday: 24 turns, median wait 1.4 seconds ... I dropped 7
        clips of yours at the speaker gate" is instrumentation, not news.
        It stays reachable by asking (app.day_review_text / the "day
        review" command) and is out of the briefing."""
        a = _app(monkeypatch, tmp_path)
        _clock(monkeypatch, 14, 29)
        a.dayreviewer = SimpleNamespace(review=lambda day: {
            "has_data": True, "turns": 24, "median_wait_s": 1.4,
            "worst_wait_s": 6.1, "speaker_rejections": 7})
        a._deliver_first_wake_briefing()
        assert a.said == ["Your briefing for today, sir."], a.said
        assert "speaker gate" in a.day_review_text("yesterday")

    def test_a_busy_model_says_so_rather_than_eating_the_yes(self, monkeypatch, tmp_path):
        a = _app(monkeypatch, tmp_path, brain=SimpleNamespace(is_busy=True))
        _clock(monkeypatch, 14, 29)
        assert a._deliver_first_wake_briefing() is False
        assert a.chats == []


# ===================================================================
# the commander half: answering the offer
# ===================================================================
@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    svc = SimpleNamespace(desktop=MagicMock(), workflows=MagicMock(), memory=MagicMock(),
                          context=MagicMock(), tts=MagicMock(), briefing_offer=None)
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.brain = SimpleNamespace(think=MagicMock(), chat=MagicMock())
    return Commander(svc)


def _offer(cmdr, made=None, ok=True):
    ran = []
    cmdr.services.briefing_offer = {
        "made_at": _t.time() if made is None else made,
        "deliver": lambda: (ran.append(1), ok)[1]}
    return ran


class TestAnsweringTheOffer:
    @pytest.mark.parametrize("said", ["yes", "yes please", "go ahead", "please do"])
    def test_a_yes_delivers_it(self, cmdr, said):
        ran = _offer(cmdr)
        res = cmdr.handle(said, "voice")
        assert res is not None and res.handled and ran == [1], said
        assert res.status.startswith("Briefing")
        assert cmdr.services.briefing_offer is None

    @pytest.mark.parametrize("said", ["no", "no thanks", "not now", "later",
                                      "maybe later", "skip it"])
    def test_a_no_declines_without_reading_anything(self, cmdr, said):
        ran = _offer(cmdr)
        res = cmdr.handle(said, "voice")
        assert res is not None and res.handled and ran == [], said
        assert res.reply == BRIEFING_DECLINED_LINE
        assert cmdr.services.briefing_offer is None

    def test_a_new_subject_silently_drops_the_offer(self, cmdr):
        ran = _offer(cmdr)
        cmdr.handle("what's the time", "voice")
        assert cmdr.services.briefing_offer is None and ran == []

    def test_a_stale_offer_is_not_taken(self, cmdr):
        ran = _offer(cmdr, made=_t.time() - OFFER_TTL_S - 10)
        res = cmdr.handle("yes", "voice")
        assert ran == []
        assert res is None or res.status != "Briefing…"

    def test_an_unrelated_no_is_not_swallowed(self, cmdr):
        """He says no to a great many things. With no offer armed -- and
        with a stale one -- a bare "no" must reach whatever comes next."""
        assert cmdr.services.briefing_offer is None
        res = cmdr.handle("no", "voice")
        assert res is None or res.reply != BRIEFING_DECLINED_LINE
        _offer(cmdr, made=_t.time() - OFFER_TTL_S - 10)
        res = cmdr.handle("no", "voice")
        assert res is None or res.reply != BRIEFING_DECLINED_LINE

    def test_a_busy_model_is_reported_not_swallowed(self, cmdr):
        _offer(cmdr, ok=False)
        res = cmdr.handle("yes", "voice")
        assert res is not None and res.handled and res.speak and res.reply

    def test_the_offer_counts_as_a_question_on_the_table(self, cmdr):
        _offer(cmdr)
        assert cmdr.question_open()
        cmdr.services.briefing_offer = None
        assert not cmdr.question_open()
