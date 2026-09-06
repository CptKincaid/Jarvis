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
from jarvis.commander import (BRIEFING_DECLINED_LINE, BRIEFING_OFFER_TTL_S,
                              Commander, IntentClassifier)
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


YES_ANSWERS = [
    "yes", "Yes.", "yes please", "Yes, please.", "yeah", "sure", "okay", "go ahead",
    "yes go ahead", "Yes, go ahead.", "yeah sure", "Yeah, sure.", "yes please do",
    "Yes, please do.", "sure go ahead", "Sure, go ahead.", "yes do it", "Yes, do it.",
    "yes run it", "okay go ahead", "Okay, do it.", "yes let's hear it",
    "Yes, let's hear it.", "yeah go for it", "go for it", "why not", "sure why not",
    "yes that would be great", "please", "yes thank you", "yes sir", "ok sure",
    "yeah okay", "yes yes", "yep go ahead", "let's do it", "do it please"]
NO_ANSWERS = [
    "no", "No.", "no thanks", "No, thank you.", "not now", "no not now", "later",
    "No, maybe later.", "nah", "no I'm good", "skip it", "No, skip it.",
    "not today thanks", "no leave it", "not right now", "no, not right now, thanks",
    "I'm fine", "no need", "never mind", "no thank you jarvis"]


@pytest.mark.parametrize("text", YES_ANSWERS)
def test_a_composed_yes_is_still_a_yes(text):
    """26 of these 37 were refused by a grammar that took ONE yes-word plus a
    courtesy, and "yes go ahead" then went to the model as a fresh command
    -- the day's only offer gone (F36, measured 09-03)."""
    from jarvis.commander import _BRIEFING_NO_RX, _BRIEFING_YES_RX
    assert _BRIEFING_YES_RX.match(text.strip()), text
    assert not _BRIEFING_NO_RX.match(text.strip()), text


@pytest.mark.parametrize("text", NO_ANSWERS)
def test_a_decline_is_still_a_decline(text):
    from jarvis.commander import _BRIEFING_NO_RX, _BRIEFING_YES_RX
    assert _BRIEFING_NO_RX.match(text.strip()), text
    assert not _BRIEFING_YES_RX.match(text.strip()), text


@pytest.mark.parametrize("text", ["yes, turn the lights off", "yes and what is the weather",
                                  "sure, set a timer for ten minutes"])
def test_a_yes_that_carries_a_command_falls_through(text):
    from jarvis.commander import _BRIEFING_YES_RX
    assert not _BRIEFING_YES_RX.match(text)


def test_a_held_offer_keeps_the_follow_up_mic_for_the_open_question(monkeypatch, tmp_path):
    """First voice turn of the day ends in a read-back. The offer is armed
    behind it and correctly HELD on the falling edge -- but the hold used to
    return before the follow-up block, so the "yes" that read-back needed
    was never heard without a wake word (F35, 09-03)."""
    monkeypatch.setattr(CONFIG, "talkback", True)
    a = _app(monkeypatch, tmp_path)
    _clock(monkeypatch, 9, 0)
    opened = []
    a._start_followup = lambda: opened.append(1)
    a.commander = SimpleNamespace(question_open=lambda: True)   # the read-back is live
    a._after_dispatch("cancel all my alarms", "voice",
                      SimpleNamespace(reply="Cancel all three alarms, sir?", speak=True,
                                      done=True, ack=False, status="Read-back"))
    assert a._briefing_pending and a._followup_after_speech
    a._after_speech()                                            # the read-back's falling edge
    assert a.said == [] and a._briefing_pending                  # offer held...
    assert opened == [1], "...and the mic for the read-back opened"
    assert a._followup_after_speech is False


class TestTheOfferReplacesTheDelivery:
    def test_a_settled_burst_offers_instead_of_reading_the_briefing(self, monkeypatch, tmp_path):
        a = _app(monkeypatch, tmp_path)
        _clock(monkeypatch, 14, 29)
        _arm(a)
        assert a._briefing_pending
        a._after_speech()
        assert a.said == ["Shall I run your briefing, sir?"], a.said
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

    def test_the_offer_says_the_same_hour_free_line_whenever_it_is_put(
            self, monkeypatch, tmp_path):
        """The hour word belongs to the MODEL ask, not to the spoken
        question. Jarvis must not offer a phrase his own grammar refuses:
        "run my afternoon briefing" reaches nothing and "my evening
        briefing" is the TOMORROW preview (both measured below)."""
        for hour in (7, 14, 19):
            a = _app(monkeypatch, tmp_path)
            _clock(monkeypatch, hour)
            (tmp_path / "briefing.json").unlink(missing_ok=True)
            _arm(a)
            a._after_speech()
            assert a.said == ["Shall I run your briefing, sir?"], (hour, a.said)
        for word in ("morning", "afternoon", "evening"):
            assert word not in app_mod.BRIEFING_OFFER_LINE

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
        assert a.said == ["Shall I run your briefing, sir?"]


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


class TestTheOfferDoesNotHoldTheFloorForThreeMinutes:
    def test_the_open_question_expires_with_the_window_it_opened(
            self, monkeypatch, tmp_path):
        """_question_open gates _salvage_low_confidence as well as the mic
        window (the argument app.py already writes out against
        _pending_leave). The offer opens a 15 s follow-up window, so
        holding the floor for the wake alarm's 180 s meant three minutes of
        force-accepted sub-threshold garble after one question a day."""
        a = _app(monkeypatch, tmp_path)
        slim = SimpleNamespace()          # no commander.question_open
        a.services.briefing_offer = {"made_at": _t.time(), "deliver": lambda: True}
        assert a._question_open(slim)
        a.services.briefing_offer["made_at"] = _t.time() - BRIEFING_OFFER_TTL_S - 5
        assert not a._question_open(slim)


class TestWhatIsActuallyDelivered:
    def test_the_model_is_asked_for_the_briefing_of_this_hour(self, monkeypatch, tmp_path):
        """brain.chat("my morning briefing") was asked at 14:33, 15:00 and
        14:29 in the two retained logs. It has never once been morning."""
        a = _app(monkeypatch, tmp_path)
        _clock(monkeypatch, 14, 29)
        assert a._deliver_first_wake_briefing() is True
        assert a.chats == [("my afternoon briefing",
                            {"force_tool": "get_briefing",
                             "addressee": ("", "sir")})]
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
# ===================================================================
# F37 (09-03): three yes-taking rungs question_open() did not count
# ===================================================================
def _slim_cmdr(tmp_path, monkeypatch):
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


def test_the_teach_offer_is_an_open_question_until_it_expires(tmp_path, monkeypatch):
    """"...say quiz me and I'll test you on it, sir" (rung 3a'') takes the
    next plain yes and was not counted, so the briefing offer could land
    on top of it."""
    c = _slim_cmdr(tmp_path, monkeypatch)
    assert c.question_open() is False
    c._pending_teach = ("thermodynamics", "body", _t.monotonic())
    assert c.question_open() is True
    c._pending_teach = ("thermodynamics", "body", _t.monotonic() - OFFER_TTL_S - 1)
    assert c.question_open() is False, "a dead offer must not hold the floor"


def test_the_terminal_offer_is_an_open_question_until_it_expires(tmp_path, monkeypatch):
    """Claude's "say the word and I'll open the terminal there instead"
    (rung 3b) had no stamp at all: it took a yes for ever and was never
    counted. It now carries one, and both ends read it."""
    c = _slim_cmdr(tmp_path, monkeypatch)
    c._pending_terminal_slug = "vss"
    c._pending_terminal_made = _t.monotonic()
    assert c.question_open() is True
    c._pending_terminal_made = _t.monotonic() - OFFER_TTL_S - 1
    assert c.question_open() is False
    # ...and the rung itself lets a late yes go rather than opening a
    # terminal he asked for an hour ago.
    c.services.claude = SimpleNamespace(open_terminal=lambda slug: True)
    assert c._try_terminal_offer("yes") is None
    assert c._pending_terminal_slug == ""


def test_the_event_read_back_is_an_open_question_until_it_expires(tmp_path, monkeypatch):
    """"Dentist, tomorrow at 3 pm. Shall I add it, sir?" (rung 3c) is the
    one he hits most. calendar.add_event now stamps made_at; a dict with
    no stamp (every older caller and test) still counts as live."""
    c = _slim_cmdr(tmp_path, monkeypatch)
    c.services.calendar = SimpleNamespace(
        pending_event={"title": "Dentist", "start": None, "end": None})
    assert c.question_open() is True, "unstamped: a live question"
    c.services.calendar.pending_event["made_at"] = _t.monotonic()
    assert c.question_open() is True
    c.services.calendar.pending_event["made_at"] = _t.monotonic() - OFFER_TTL_S - 1
    assert c.question_open() is False
    # ...and a late "yes" to it writes nothing: the read-back is spent.
    written = []
    import jarvis.commander as cmd_mod
    monkeypatch.setattr(cmd_mod, "add_event",
                        lambda *a, **kw: written.append(1) or ("Added", None))
    c.services.calendar.icloud_calendars = lambda: ["CAL"]
    assert c._try_event_confirm("yes") is None
    assert written == [] and c.services.calendar.pending_event is None


def test_the_calendar_tool_stamps_the_read_back_it_parks():
    """The producing end of the stamp above."""
    import inspect
    from jarvis.tools import calendar as cal_mod
    src = inspect.getsource(cal_mod)
    assert '"made_at": time.monotonic()' in src


def test_the_offer_is_held_behind_an_event_read_back(monkeypatch, tmp_path):
    """The live consequence. First voice turn of the day: "add a dentist
    appointment tomorrow at three" -> "Dentist, tomorrow at 3 pm. Shall I
    add it, sir?". The settled burst used to put "Shall I run your
    briefing, sir?" straight on top of it -- two questions on the table --
    and his yes landed on the event while the day's only offer was
    spent. The real Commander's question_open() now counts it."""
    monkeypatch.setattr(CONFIG, "talkback", True)
    a = _app(monkeypatch, tmp_path)
    _clock(monkeypatch, 9, 0)
    c = _slim_cmdr(tmp_path, monkeypatch)
    c.services.calendar = SimpleNamespace(
        pending_event={"title": "Dentist", "start": None, "end": None,
                       "made_at": _t.monotonic()})
    a.commander = c
    opened = []
    a._start_followup = lambda: opened.append(1)
    a._after_dispatch("add a dentist appointment tomorrow at three", "voice",
                      SimpleNamespace(reply="Dentist, tomorrow at 3 pm. Shall I add it, sir?",
                                      speak=True, done=True, ack=False, status="Read-back"))
    a._after_speech()
    assert a.said == [], "the offer must wait behind the read-back"
    assert a._briefing_pending, "...and stay armed for after it"
    assert c.services.calendar.pending_event is not None
    assert opened == [1], "the read-back keeps its follow-up mic (F35)"


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
    # "okay" / "ok" / "alright" are the commonest casual yesses in the live
    # log ("jarvis ok delete my last email") and are absent from
    # commander._YES_WORDS, so they used to answer this question with
    # COMPLETE SILENCE -- and the day was already closed, so there was no
    # second ask. They are widened HERE, on this rung, exactly as the
    # decline side already was.
    @pytest.mark.parametrize("said", [
        "yes", "yes please", "yes, sir", "yeah", "yep", "sure", "sure thing",
        "go ahead", "please do", "do it", "okay", "ok", "OK.", "alright",
        "all right", "sounds good", "absolutely", "of course", "certainly",
        "let's hear it", "run it", "okay then", "yes jarvis"])
    def test_a_yes_delivers_it(self, cmdr, said):
        ran = _offer(cmdr)
        res = cmdr.handle(said, "voice")
        assert res is not None and res.handled and ran == [1], said
        assert res.status.startswith("Briefing")
        assert cmdr.services.briefing_offer is None

    @pytest.mark.parametrize("said", ["no", "no thanks", "no thank you",
                                      "not now", "not today", "not just now",
                                      "later", "later, jarvis", "maybe later",
                                      "another time", "in a bit", "skip it",
                                      "skip that", "leave it", "never mind",
                                      "forget it", "no, not now", "nope"])
    def test_a_no_declines_without_reading_anything(self, cmdr, said):
        ran = _offer(cmdr)
        res = cmdr.handle(said, "voice")
        assert res is not None and res.handled and ran == [], said
        assert res.reply == BRIEFING_DECLINED_LINE
        assert cmdr.services.briefing_offer is None

    # THE REGRESSION. The decline regex was start-anchored only, so it
    # matched as a PREFIX and every one of these was answered "Very good,
    # sir." with the command inside it lost. "skip" is in the live log
    # verbatim (jarvis.log.1:20928, routed to local:music), and rung 4b's
    # own comment says the read control must outrank the chains that would
    # "otherwise eat 'skip' / 'back' / 'pause' / 'go on'" -- this rung sits
    # above it and was eating them first.
    @pytest.mark.parametrize("said", [
        "skip", "skip this song", "skip the intro", "skip to the next track",
        "later today remind me to call mom",
        "later on set a timer for ten minutes", "later gator",
        "leave it alone", "leave it running",
        "no, turn the lights off", "no, set a timer for ten minutes",
        "no, delete my last email", "no music please", "no way",
        "no problem", "no worries"])
    def test_a_command_that_merely_begins_with_a_decline_is_not_swallowed(
            self, cmdr, said):
        ran = _offer(cmdr)
        res = cmdr.handle(said, "voice")
        assert ran == [], said
        assert res is None or res.reply != BRIEFING_DECLINED_LINE, said
        assert res is None or not str(res.status or "").startswith("Briefing"), said

    # The other half of the same defect: parse_yes_no waives its own
    # >6-word overheard-speech guard when the first word is a yes/no word,
    # so a sentence that merely OPENS with "yeah" delivered the whole
    # 40-second briefing. The first line here is real: jarvis.log.1:19499,
    # handle 'Yeah, so you should be able to look that up.' source=voice --
    # not an answer to anything; it routed to a web lookup.
    @pytest.mark.parametrize("said", [
        "Yeah, so you should be able to look that up.",
        "yes I already told him it was fine this morning",
        "no I do not think that is what she meant at all",
        "sure but only after the presentation on Thursday"])
    def test_overheard_speech_that_opens_with_yes_or_no_is_not_an_answer(
            self, cmdr, said):
        ran = _offer(cmdr)
        res = cmdr.handle(said, "voice")
        assert ran == [], said
        assert res is None or res.reply != BRIEFING_DECLINED_LINE, said

    def test_a_new_subject_silently_drops_the_offer(self, cmdr):
        ran = _offer(cmdr)
        cmdr.handle("what's the time", "voice")
        assert cmdr.services.briefing_offer is None and ran == []

    @pytest.mark.parametrize("source", ["discord", "cli", "api"])
    def test_a_turn_from_another_room_neither_answers_nor_eats_the_offer(
            self, cmdr, source):
        """_handle_inner runs this rung for EVERY source, and it used to
        clear the offer whatever the source. The day is closed when the
        question is put, so a Discord message inside the window spent the
        only briefing offer of the day."""
        ran = _offer(cmdr)
        cmdr.handle("what's the weather", source)
        assert isinstance(cmdr.services.briefing_offer, dict), source
        assert ran == []
        res = cmdr.handle("yes", "voice")       # still answerable by voice
        assert ran == [1] and res.status.startswith("Briefing")

    def test_a_reading_in_progress_keeps_its_own_skip(self, cmdr):
        """"skip it" is a decline here and a read control at rung 4b, which
        is BELOW this one. While something is actually being read the
        reader owns it, and the offer is left standing."""
        cmdr.services.reader = SimpleNamespace(
            active=True, skip=lambda: SimpleNamespace(ok=True, message="Skipped."))
        ran = _offer(cmdr)
        res = cmdr.handle("skip it", "voice")
        assert ran == []
        assert res is not None and res.reply != BRIEFING_DECLINED_LINE
        assert isinstance(cmdr.services.briefing_offer, dict), "the offer still stands"
        assert cmdr.handle("yes", "voice").status.startswith("Briefing")

    def test_a_stale_offer_is_not_taken(self, cmdr):
        ran = _offer(cmdr, made=_t.time() - OFFER_TTL_S - 10)
        res = cmdr.handle("yes", "voice")
        assert ran == []
        assert res is None or res.status != "Briefing…"

    def test_it_goes_stale_with_its_own_window_not_the_wake_alarms(self, cmdr):
        """The offer opens the mic for its own answer (quiz.window_s, 15 s).
        An answer that arrives a minute and a half later is not an answer,
        and the wake alarm's 180 s is what made the false-yes window large."""
        assert BRIEFING_OFFER_TTL_S < OFFER_TTL_S
        ran = _offer(cmdr, made=_t.time() - BRIEFING_OFFER_TTL_S - 5)
        res = cmdr.handle("yes", "voice")
        assert ran == []
        assert res is None or res.status != "Briefing…"
        assert not cmdr.question_open()

    def test_a_bare_no_outside_a_live_offer_reaches_what_comes_next(self, cmdr):
        """He says no to a great many things. With no offer armed -- and
        with a stale one -- a bare "no" must reach whatever comes next.
        (A bare "no" to a LIVE offer IS the decline; it is
        "no, <command>" that must survive, above.)"""
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


class TestTheOfferSaysSomethingHeCanRepeat:
    """Finding from review, verified against a real Commander: the line
    Jarvis spoke was not a phrase his own command grammar accepts.

    After a decline -- or after the offer is dropped by a new subject --
    echoing his own words back at him is the natural retry, and it landed
    on nothing (afternoon) or on TOMORROW's preview (evening).
    """

    def test_the_offer_can_be_echoed_straight_back_and_runs_the_briefing(self, cmdr):
        echo = (app_mod.BRIEFING_OFFER_LINE.lower()
                .replace("shall i ", "").replace("your ", "my ")
                .replace(", sir?", "").strip())
        res = cmdr.handle(echo, "voice")
        assert res is not None and res.status == "Briefing…", (echo, res)
        cmdr.services.brain.chat.assert_called_once()
        assert cmdr.services.brain.chat.call_args.kwargs.get("force_tool") == "get_briefing"

    @pytest.mark.parametrize("said", ["my briefing", "run my briefing", "briefing"])
    def test_the_briefing_is_still_one_sentence_away_after_a_decline(self, cmdr, said):
        _offer(cmdr)
        assert cmdr.handle("no thanks", "voice").reply == BRIEFING_DECLINED_LINE
        assert cmdr.handle(said, "voice").status == "Briefing…", said

    @pytest.mark.parametrize("word", ["morning", "afternoon", "evening"])
    def test_an_hour_worded_offer_would_not_have_reached_todays_briefing(
            self, cmdr, word):
        """Why the arc word is not in the spoken line. "my afternoon
        briefing" matches nothing (_BRIEFING_RX carries no arc words) and
        "my evening briefing" is _PREVIEW_RX, which forces when=tomorrow --
        registered BELOW _BRIEFING_RX, so widening that one to take the arc
        words would steal the evening preview instead."""
        res = cmdr.handle(f"my {word} briefing", "voice")
        assert res is None or res.status != "Briefing…", word
