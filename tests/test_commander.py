"""Tests for jarvis.commander — registry precedence, routing, CommandResult.

Pure logic: services are mocked, no audio/X11/model dependencies.
"""
import re
import types
from unittest.mock import MagicMock

import pytest

import jarvis.commander as commander
from jarvis.commander import (
    Command,
    Commander,
    CommandResult,
    IntentClassifier,
    QUICK_COMMANDS,
    REGISTRY,
    _apply_voice_commands,
    strip_jarvis_prefix,
)
from jarvis.config import CONFIG, PATHS


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def services():
    svc = types.SimpleNamespace(
        desktop=MagicMock(),
        workflows=MagicMock(),
        brain=MagicMock(),
        memory=MagicMock(),
        context=MagicMock(),
        tts=MagicMock(),
    )
    # Neutral defaults so nothing swallows commands unexpectedly.
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    return svc


@pytest.fixture
def cmdr(services, tmp_path, monkeypatch):
    # Keep the intent classifier off the real ~/.aiws_trainer log.
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG",
                        tmp_path / "intent_log.json")
    # Deterministic config for routing tests.
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "auto_type", True)
    monkeypatch.setattr(CONFIG, "talkback", False)
    # Run background threads inline so mocks are visible immediately.
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    return Commander(services)


# ------------------------------------------------------------ CommandResult
def test_command_result_contract():
    r = CommandResult(handled=True)
    assert r.handled is True
    assert r.reply is None
    assert r.speak is False
    assert r.status is None
    assert r.done is True


def test_registry_is_ordered_command_list():
    assert all(isinstance(c, Command) for c in REGISTRY)
    names = [c.name for c in REGISTRY]
    # Known landmines: answer-question fallback must precede the
    # QUICK_COMMANDS table lookup; find-file must exist as one combined
    # regex entry ahead of the catch-alls.
    assert names.index("answer question") < names.index("quick command")
    assert names.index("find file") < names.index("answer question")
    # deploy/autonomous wiring precedes workflow lookup
    assert names.index("autonomous") < names.index("workflow")


def test_quick_commands_use_paths_vss_env():
    assert str(PATHS.VSS_ENV) in QUICK_COMMANDS["commit"]
    assert str(PATHS.VSS_ENV) in QUICK_COMMANDS["run tests"]
    assert "/home/hunterp/vss_env" not in repr(QUICK_COMMANDS).replace(
        str(PATHS.VSS_ENV), "")


# ------------------------------------------------------- voice command table
def test_apply_voice_commands_punctuation():
    assert _apply_voice_commands("hello comma world period") == "hello, world."


def test_apply_voice_commands_newline_and_question():
    out = _apply_voice_commands("first line new line second line question mark")
    assert out == "first line\nsecond line?"


def test_apply_voice_commands_backspace_deletes_previous_char():
    # "backspace" removes exactly the character before it (here: the space)
    assert _apply_voice_commands("helloo backspace") == "helloo"
    # at position 0 it eats itself and leading whitespace is stripped
    assert _apply_voice_commands("backspace hello") == "hello"


def test_apply_voice_commands_action_phrases():
    assert _apply_voice_commands("delete that") == \
        "__ACTION__delete_last_sentence"
    assert _apply_voice_commands("please scratch that") == \
        "__ACTION__delete_last_sentence"
    assert _apply_voice_commands("clear all") == "__ACTION__clear_all"


# ------------------------------------------------------------ prefix helper
def test_strip_jarvis_prefix():
    assert strip_jarvis_prefix("Jarvis, commit.") == "commit"
    assert strip_jarvis_prefix("hey jarvis check gpu") == "check gpu"
    assert strip_jarvis_prefix("commit now") is None


# ------------------------------------------------------- registry precedence
def test_find_file_regex_wins_over_generic_branches(cmdr, services):
    services.context.find_file.return_value = ["/a/settings.json"]
    res = cmdr.handle("jarvis find file settings.json")
    services.context.find_file.assert_called_once_with("settings.json")
    assert res.handled and "settings.json" in res.reply


def test_bare_find_also_routes_to_find_file(cmdr, services):
    services.context.find_file.return_value = []
    res = cmdr.handle("jarvis find voiceprint")
    services.context.find_file.assert_called_once_with("voiceprint")
    assert res.handled


def test_answer_question_runs_before_quick_commands(cmdr, services,
                                                    monkeypatch):
    ran = []
    monkeypatch.setattr(commander.subprocess, "run",
                        lambda *a, **k: ran.append(a))
    services.context.answer_question.return_value = "Up eight days, sir."
    # "uptime" is a system fact the legacy agent still owns; "check gpu" is a
    # QUICK_COMMANDS phrase that must never get the chance to run.
    res = cmdr.handle("jarvis uptime, and check gpu")
    services.context.answer_question.assert_called_once_with(
        "uptime, and check gpu")
    assert res.reply == "Up eight days, sir."
    assert ran == []          # quick-command shell never executed


def test_answer_question_no_longer_owns_weather_or_the_clock(cmdr, services):
    # spec 5.2: weather and time are tools of the local model now.
    for text in ("what's the weather like", "what's the temperature outside"):
        cmdr.handle(text, source="typed")
    services.context.answer_question.assert_not_called()


def test_quick_command_runs_when_no_local_answer(cmdr, services, monkeypatch):
    ran = {}

    def fake_run(cmd, **kw):
        ran["cmd"] = cmd
        return types.SimpleNamespace(stdout="ok\n", stderr="")

    monkeypatch.setattr(commander.subprocess, "run", fake_run)
    res = cmdr.handle("jarvis check gpu")
    assert res.handled and not res.done
    assert "nvidia-smi" in ran["cmd"]


def test_desktop_stage_runs_before_registry(cmdr, services):
    # If the desktop parser claims the phrase, the registry never sees it.
    services.desktop.parse_action = lambda part: ("key", "ctrl+f")
    res = cmdr.handle("jarvis find file settings.json")
    services.desktop.execute_actions.assert_called_once_with(
        [("key", "ctrl+f")])
    services.context.find_file.assert_not_called()
    assert res.handled


def test_registry_requires_jarvis_prefix(cmdr, services):
    cmdr.handle("find file settings.json", source="typed")
    services.context.find_file.assert_not_called()
    # unprefixed text falls through to the brain (jarvis mode on)
    services.brain.think.assert_called_once()


# ----------------------------------------------------- memory routing (fix)
def test_remember_routes_to_persistent_memory(cmdr, services):
    res = cmdr.handle("jarvis remember that I parked on level 3")
    assert services.memory.remember.call_count == 1
    key, value = services.memory.remember.call_args[0]
    # 2026-09-04: filed in the SECOND person (memory.store_fact_from_speech)
    # -- under "Known facts" the model is Jarvis, and "I parked" said that
    # Jarvis did. The key is the first six words of the value.
    assert value == "You parked on level 3"
    # the key is the CASE-FOLDED head of the value (tests/test_memory_rung.py
    # (h)): the value keeps Whisper's capital, the key does not
    assert key == "you parked on level 3"
    assert res.handled and "You parked on level 3" in res.reply
    services.brain.think.assert_not_called()


@pytest.mark.parametrize("said,fact", [
    ("jarvis put it in your memory that i graduate december 10th 2026 with an electrical engineering degree",
     "you graduate december 10th 2026 with an electrical engineering degree"),
    ("jarvis keep in mind that heather prefers email", "heather prefers email"),
    ("jarvis don't forget that the lab moved to room 049", "the lab moved to room 049"),
])
def test_the_ways_he_actually_says_remember_reach_the_store(cmdr, services, said, fact):
    """2026-09-02 23:25: "Put it in your memory that i graduate December 10th
    2026" missed the rung, reached the model, and the model said "I have noted
    that, sir" with nothing stored. Two days later Jarvis searched his
    documents for the date. Every phrasing here must hit the store, not the
    model."""
    res = cmdr.handle(said)
    assert services.memory.remember.call_count == 1
    key, value = services.memory.remember.call_args[0]
    assert value == fact
    assert res.handled and fact in res.reply
    services.brain.think.assert_not_called()


def test_remember_to_is_still_not_a_fact(cmdr, services):
    cmdr.handle("jarvis don't forget to call mum")
    assert services.memory.remember.call_count == 0


@pytest.mark.parametrize("said", [
    "jarvis make a note of milk",
    "jarvis make a note of milk and eggs",
    "jarvis note that down",
])
def test_a_note_is_still_a_note_not_a_fact(cmdr, services, said):
    """Measured 2026-09-04 on 7539478: 'note that' / 'make a note (of)' in the
    remember rung pulled these out of his notes list into facts.json. The
    notes rung owns them; nothing here may reach long-term memory."""
    res = cmdr.handle(said)
    assert services.memory.remember.call_count == 0
    assert res.handled
    services.brain.think.assert_not_called()


def test_recall_routes_to_memory(cmdr, services):
    services.memory.recall.return_value = [
        {"key": "parking", "value": "parked on level 3", "time": "t"}]
    res = cmdr.handle("jarvis recall parking")
    services.memory.recall.assert_called_once_with("parking")
    assert "parked on level 3" in res.reply


def test_workflow_lookup_falls_through_when_unknown(cmdr, services):
    res = cmdr.handle("jarvis what year is it")
    services.workflows.get.assert_called_with("what year is it")
    services.workflows.run.assert_not_called()
    # nothing in the registry claims it: it ends up at the assistant
    services.brain.think.assert_called_once_with("jarvis what year is it")
    assert res.handled and not res.done


def test_deploy_routes_to_autonomous_brain(cmdr, services):
    res = cmdr.handle("jarvis deploy")
    services.brain.execute_autonomous.assert_called_once_with("deploy")
    services.workflows.run.assert_not_called()
    assert res.handled and not res.done


# ----------------------------------------------------------- dictation mode
def test_dictation_toggle_and_typing(cmdr, services, monkeypatch):
    typed = []
    monkeypatch.setattr(cmdr, "_type_raw", lambda t: typed.append(t))

    res = cmdr.handle("jarvis dictate")
    assert cmdr.dictation is True
    assert "Dictation mode: ON" in res.reply

    res = cmdr.handle("hello world")
    assert typed == ["hello world "]
    assert res.handled

    res = cmdr.handle("okay end dictation")
    assert cmdr.dictation is False
    assert res.reply == "Dictation mode: OFF"
    # dictation bypasses everything else
    services.brain.think.assert_not_called()


# --------------------------------------------------------------- targeting
def test_voice_targeting(cmdr, services):
    res = cmdr.handle("switch to opera", source="typed")
    services.desktop.target_window.assert_called_once_with("opera")
    assert res.handled

    res = cmdr.handle("reset target", source="typed")
    services.desktop.reset_target.assert_called_once()
    assert res.status == "Target: auto"


# ------------------------------------------------------------ intent gating
def test_background_chat_is_ignored(cmdr, services):
    res = cmdr.handle("she said no way lol haha dude", source="voice")
    assert res.handled and "Ignored" in res.status
    services.brain.think.assert_not_called()


def test_uncertain_intent_calls_ui_hook(cmdr, services):
    prompts = []
    cmdr.on_uncertain = prompts.append
    res = cmdr.handle("banana purple elephant dancing", source="voice")
    assert prompts == ["banana purple elephant dancing"]
    assert res.handled and not res.done
    services.brain.think.assert_not_called()


def test_resolve_uncertain_yes_routes_to_brain(cmdr, services):
    cmdr.on_uncertain = lambda t: None
    cmdr.handle("banana purple elephant dancing", source="voice")
    res = cmdr.resolve_uncertain("banana purple elephant dancing", True)
    services.brain.think.assert_called_once_with(
        "banana purple elephant dancing")
    assert res.handled
    # feedback was learned
    assert cmdr.intent.num_examples == 1


def test_typed_input_skips_intent_classification(cmdr, services):
    # Casual text that voice would discard goes to the brain when typed.
    cmdr.handle("she said no way lol haha dude", source="typed")
    services.brain.think.assert_called_once()


# ------------------------------------------------------------ brain routing
def test_jarvis_mode_routes_to_brain(cmdr, services):
    res = cmdr.handle("summarize my day for me", source="typed")
    services.brain.think.assert_called_once_with("summarize my day for me")
    assert res.handled and not res.done and res.status == "Thinking..."


def test_fallback_types_to_window_when_jarvis_mode_off(cmdr, services,
                                                       monkeypatch):
    monkeypatch.setattr(CONFIG, "jarvis_mode", False)
    services.context.interpret_intent.return_value = None
    res = cmdr.handle("write the summary now please", source="typed")
    services.desktop.type_text.assert_called_once_with(
        "write the summary now please")
    assert res.handled


def test_screenshot_phrase_stripped_and_flagged(cmdr, services, monkeypatch):
    monkeypatch.setattr(CONFIG, "jarvis_mode", False)
    services.context.interpret_intent.return_value = None
    cmdr.handle("fix the login bug and take a screenshot", source="typed")
    services.desktop.screenshot.assert_called_once_with(
        text="fix the login bug")


# -------------------------------------------------------- reminders/timers
def test_timer_routes_to_workflows(cmdr, services):
    res = cmdr.handle("jarvis timer for 5 minutes")
    services.workflows.set_reminder.assert_called_once_with(
        300, "Timer for 5 minutes")
    assert res.handled


def test_remind_me_routes_to_workflows(cmdr, services):
    res = cmdr.handle("jarvis remind me in 10 minutes to stretch")
    services.workflows.set_reminder.assert_called_once_with(600, "stretch")
    assert res.handled


# ------------------------------------------------------------ Tier 1 clock
def test_clock_reply_is_spoken_twelve_hour_in_voice():
    from datetime import datetime
    from jarvis.commander import clock_kind, clock_reply
    at = datetime(2026, 8, 26, 16, 5)
    assert clock_reply(at, "time") == "It's 4:05 in the afternoon, sir."
    assert clock_reply(datetime(2026, 8, 26, 0, 30), "time") == \
        "It's 12:30 at night, sir."
    assert clock_reply(datetime(2026, 8, 3, 9, 0), "time") == \
        "It's 9:00 in the morning, sir."
    assert clock_reply(at, "date") == \
        "It's Wednesday the 26th of August, sir."
    assert clock_reply(datetime(2026, 9, 1, 19, 0), "date") == \
        "It's Tuesday the 1st of September, sir."
    assert clock_reply(at, "day") == "It's Wednesday, sir."
    assert clock_kind("what time is it") == "time"
    assert clock_kind("What's the time, Jarvis?") == "time"
    assert clock_kind("have you got the time") == "time"
    assert clock_kind("what's the date today") == "date"
    assert clock_kind("what day is it") == "day"
    assert clock_kind("what year is it") is None
    assert clock_kind("set a timer for 5 minutes") is None
    assert clock_kind("it's time to go") is None


def test_prefixed_clock_question_is_answered_locally(cmdr, services):
    res = cmdr.handle("jarvis what time is it")
    assert res.handled and res.speak
    assert re.search(r"It's \d{1,2}:\d{2} (in the|at) \w+, sir\.",
                     res.reply), res.reply
    services.brain.think.assert_not_called()
    services.context.answer_question.assert_not_called()


def test_unprefixed_clock_question_never_reaches_tier2(cmdr, services):
    res = cmdr.handle("what's the date today?", source="typed")
    assert res.handled and res.speak
    assert re.match(r"It's \w+day the \d+(st|nd|rd|th) of \w+, sir\.",
                    res.reply), res.reply
    services.brain.think.assert_not_called()


def test_unprefixed_text_with_a_clock_word_still_routes_to_brain(cmdr,
                                                                  services):
    cmdr.handle("this took a long time to build", source="typed")
    services.brain.think.assert_called_once()


# --------------------------------------------------------- Tier 1 courtesy
def test_courtesy_kind_matches_whole_utterances_only():
    from jarvis.commander import courtesy_kind
    assert courtesy_kind("Jarvis, are you there?") == "presence"
    assert courtesy_kind("you there jarvis") == "presence"
    assert courtesy_kind("Thank you, Jarvis.") == "thanks"
    assert courtesy_kind("thanks a lot") == "thanks"
    # thank-you paraphrases the 3B model misreads (round-3 samples:
    # "nice one" was answered as sarcasm) are answered here too
    assert courtesy_kind("Appreciate it, Jarvis.") == "thanks"
    assert courtesy_kind("much appreciated") == "thanks"
    assert courtesy_kind("nice one jarvis") == "thanks"
    assert courtesy_kind("Jarvis, well done!") == "thanks"
    assert courtesy_kind("good job") == "thanks"
    assert courtesy_kind("Good night, Jarvis.") == "goodnight"
    assert courtesy_kind("goodnight") == "goodnight"
    assert courtesy_kind("I'm off to bed, Jarvis") == "goodnight"
    # not swallowed when the courtesy is part of a longer utterance
    assert courtesy_kind("thanks, now open the terminal") is None
    assert courtesy_kind("nice one, now open the terminal") is None
    assert courtesy_kind("appreciate it if you opened the terminal") is None
    assert courtesy_kind("are you there any tests for this module") is None
    assert courtesy_kind("good night mode") is None
    assert courtesy_kind("what can you do") is None


def test_courtesy_replies_are_in_voice_and_answered_locally(cmdr, services):
    import random
    from jarvis.commander import COURTESY_REPLIES, courtesy_reply
    for kind, lines in COURTESY_REPLIES.items():
        for line in lines:
            assert " sir" in line and line.endswith(".")
            assert len(line) < 60
        assert courtesy_reply(kind, random.Random(1)) in lines
    res = cmdr.handle("Good night, Jarvis.", source="typed")
    assert res.handled and res.speak
    assert res.reply in COURTESY_REPLIES["goodnight"]
    res = cmdr.handle("jarvis thank you")
    assert res.reply in COURTESY_REPLIES["thanks"] and res.speak
    res = cmdr.handle("jarvis are you there?")
    assert res.reply in COURTESY_REPLIES["presence"] and res.speak
    services.brain.think.assert_not_called()
    # a paraphrase still reaches the brain
    cmdr.handle("are you still with me, Jarvis?", source="typed")
    services.brain.think.assert_called_once()


def test_greeting_kind_matches_whole_utterances_only():
    """Greetings were the last courtesy shape still costing a model turn
    (2026-08-26 sweep, Tier-1 fast paths)."""
    from jarvis.commander import greeting_kind
    assert greeting_kind("hello") == "greeting"
    assert greeting_kind("Hi there, Jarvis!") == "greeting"
    assert greeting_kind("Good morning") == "greeting"
    assert greeting_kind("good afternoon jarvis") == "greeting"
    assert greeting_kind("morning") == "greeting"
    assert greeting_kind("how are you") == "wellbeing"
    assert greeting_kind("How's it going, Jarvis?") == "wellbeing"
    assert greeting_kind("what's up") == "wellbeing"
    assert greeting_kind("are you busy") == "availability"
    assert greeting_kind("got a minute") == "availability"
    # "good night" keeps its own answer, and a longer sentence is not a
    # greeting at all
    assert greeting_kind("good night") is None
    assert greeting_kind("hey jarvis stop that") is None
    assert greeting_kind("hello can you fix the parser") is None
    assert greeting_kind("what's up with the build") is None
    assert greeting_kind("good morning briefing please") is None


def test_greetings_are_answered_locally_in_voice(cmdr, services):
    from jarvis.commander import COURTESY_REPLIES
    for text, kind in (("hello", "greeting"), ("how are you", "wellbeing"),
                       ("are you busy", "availability")):
        res = cmdr.handle(text, source="typed")
        assert res.handled and res.speak, text
        assert res.reply in COURTESY_REPLIES[kind], (text, res.reply)
    services.brain.think.assert_not_called()
    services.brain.chat.assert_not_called()


def test_the_greeting_line_follows_the_clock():
    from datetime import datetime

    from jarvis.commander import _greeting_line
    assert _greeting_line(datetime(2026, 8, 26, 8, 0)) == "Good morning, sir."
    assert _greeting_line(datetime(2026, 8, 26, 14, 0)) == "Good afternoon, sir."
    assert _greeting_line(datetime(2026, 8, 26, 21, 0)) == "Good evening, sir."
    assert _greeting_line(datetime(2026, 8, 26, 2, 0)) == "Good evening, sir."


# ---------------------------------------------------- graceful degradation
def test_missing_service_falls_through(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG",
                        tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    svc = types.SimpleNamespace(brain=MagicMock())   # only the brain exists
    c = Commander(svc)
    res = c.handle("jarvis remember that the sky is blue", source="typed")
    # memory branch skipped (service missing) → falls through to the brain
    svc.brain.think.assert_called_once()
    assert res.handled


def test_empty_text_not_handled(cmdr):
    res = cmdr.handle("   ")
    assert res.handled is False


# ==================================================================
# The assistant services (spec 5.2): ringing alarm, approvals, the
# router question, the skill map and the router dispatch itself.
# ==================================================================
from datetime import datetime, timedelta          # noqa: E402

from jarvis.commander import (                    # noqa: E402
    ALLOWED_LINE,
    ASSISTANT_TIER1,
    CLAUDE_ACK_FALLBACK,
    DECLINED_LINE,
    STOPPED_LINE,
)
from jarvis.router import (                        # noqa: E402
    DEFAULT_SKILL_PHRASES,
    ROUTER_QUESTION,
    Router,
)


class FakeAssistantCfg:
    """The slice of AssistantConfig the commander and the router read."""

    def __init__(self, **over):
        self.data = {
            "claude.skill_phrases": dict(DEFAULT_SKILL_PHRASES),
            "claude.big_model": "fable",
            "alarms.snooze_min": 10,
            "briefing.enabled": False,
        }
        self.data.update(over)

    @property
    def skill_phrases(self):
        return self.data["claude.skill_phrases"]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setup_line(self, section):
        return f"I'll need {section} set up, sir."


@pytest.fixture
def rich(services, tmp_path, monkeypatch):
    """A commander wired to the full spec-2.2 services namespace.

    The router is the real one (its decisions are the thing under test);
    everything it dispatches to is a mock, and the model tie-breaker is a
    stub — no network, no Ollama, no Claude.
    """
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG",
                        tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "auto_type", True)
    monkeypatch.setattr(CONFIG, "talkback", False)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())

    services.assistant = FakeAssistantCfg()
    services.timekeeper = MagicMock()
    services.timekeeper.ringing = False
    services.timekeeper.ringing = None                    # nothing ringing
    services.timekeeper.list_text.return_value = "Nothing set, sir."
    services.timekeeper.cancel.return_value = 1
    services.timekeeper.parse_when.return_value = \
        datetime.now() + timedelta(hours=9)
    services.timekeeper.describe_due.return_value = "at 7:00 am tomorrow"
    services.notes = MagicMock()
    services.approvals = MagicMock()
    services.approvals.pending.return_value = []          # nothing pending
    services.claude = MagicMock()
    services.claude.active_project = "jarvis"
    services.claude.submit.return_value = types.SimpleNamespace(task_id="t1")
    services.brain.local_line.return_value = "Right away, sir."
    services.classify_calls = []

    def classify(text):
        services.classify_calls.append(text)
        return ("local", 0.2)                             # always ambiguous

    services.router = Router(services.assistant, classify=classify)
    return Commander(services)


def _submitted(services):
    assert services.claude.submit.call_count == 1, \
        services.claude.submit.call_args_list
    return services.claude.submit.call_args


# ------------------------------------------------- (a) a ringing alarm wins
def test_ringing_alarm_words_beat_everything(rich, services):
    services.timekeeper.ringing = types.SimpleNamespace(id="a1", label="Get up")
    res = rich.handle("stop", source="typed")
    services.timekeeper.stop_ringing.assert_called_once_with("dismiss")
    assert res.handled and res.status == "Alarm dismissed"
    services.brain.chat.assert_not_called()
    services.claude.submit.assert_not_called()


def test_ringing_alarm_beats_a_router_action(rich, services):
    # "cancel" is a Claude action when nothing rings and a dismissal when
    # something does.
    services.timekeeper.ringing = types.SimpleNamespace(id="a1", label="Get up")
    rich.handle("jarvis cancel")
    services.timekeeper.stop_ringing.assert_called_once_with("dismiss")
    services.claude.cancel.assert_not_called()


def test_ringing_alarm_beats_a_pending_approval(rich, services):
    services.timekeeper.ringing = types.SimpleNamespace(id="a1", label="Get up")
    services.approvals.pending.return_value = [object()]
    rich.handle("okay", source="typed")
    services.timekeeper.stop_ringing.assert_called_once_with("dismiss")
    services.approvals.answer.assert_not_called()


def test_snooze_uses_the_spoken_number_then_the_config_default(rich, services):
    services.timekeeper.ringing = types.SimpleNamespace(id="a1", label="Get up")
    res = rich.handle("snooze 5", source="typed")
    services.timekeeper.snooze.assert_called_once_with(5)
    assert res.speak and "5 minutes" in res.reply
    services.timekeeper.snooze.reset_mock()
    rich.handle("snooze", source="typed")
    services.timekeeper.snooze.assert_called_once_with(10)


def test_words_that_are_not_alarm_words_still_route_while_ringing(rich,
                                                                  services):
    services.timekeeper.ringing = types.SimpleNamespace(id="a1", label="Get up")
    rich.handle("what's the weather", source="typed")
    services.timekeeper.stop_ringing.assert_not_called()
    services.brain.chat.assert_called_once()


# --------------------------------------------- (b) a pending approval wins
def test_approval_yes_and_no_beat_the_registry(rich, services):
    services.approvals.pending.return_value = [object()]
    res = rich.handle("yes", source="typed")
    services.approvals.answer.assert_called_once_with(True, source="typed")
    assert res.reply == ALLOWED_LINE and res.speak
    services.approvals.answer.reset_mock()
    res = rich.handle("no", source="discord")
    services.approvals.answer.assert_called_once_with(False, source="discord")
    assert res.reply == DECLINED_LINE
    services.brain.chat.assert_not_called()


def test_approval_only_swallows_yes_no_words(rich, services):
    services.approvals.pending.return_value = [object()]
    rich.handle("set a timer for 5 minutes", source="typed")
    services.approvals.answer.assert_not_called()
    services.timekeeper.add_timer.assert_called_once()


def test_nothing_pending_means_yes_is_an_ordinary_utterance(rich, services):
    rich.handle("yes", source="typed")
    services.approvals.answer.assert_not_called()


# ------------------------------------------- (c) the pending router question
def test_router_question_is_asked_once_then_resolved_to_claude(rich, services):
    res = rich.handle("sort out the thing we talked about", source="typed")
    assert res.reply == ROUTER_QUESTION and res.speak
    assert services.classify_calls == ["sort out the thing we talked about"]
    services.claude.submit.assert_not_called()
    res = rich.handle("yes", source="typed")
    args, kwargs = _submitted(services)
    assert args[0] == "sort out the thing we talked about"
    assert res.speak and res.reply == "Right away, sir."
    assert services.router.pending() is None


def test_router_question_resolved_to_local_goes_to_the_brain(rich, services):
    rich.handle("sort out the thing we talked about", source="typed")
    rich.handle("no, you do it", source="typed")
    services.brain.chat.assert_called_once_with(
        "sort out the thing we talked about")
    services.claude.submit.assert_not_called()


def test_a_new_subject_drops_the_router_question(rich, services):
    rich.handle("sort out the thing we talked about", source="typed")
    rich.handle("what's the weather", source="typed")
    # one chat call for the new subject (the route short-cut may add
    # force_tool/force_args; tests/test_route_shortcut.py pins those)
    assert services.brain.chat.call_count == 1
    assert services.brain.chat.call_args.args == ("what's the weather",)
    assert services.router.pending() is None
    services.claude.submit.assert_not_called()


# ------------------------------------------------------- the skill map
@pytest.mark.parametrize("text,prompt", [
    ("review this code", "/code-review"),
    ("commit this", "/commit"),
    ("simplify that", "/simplify"),
    ("security review", "/security-review"),
    ("run a ralph loop on the calendar parser", "/ralph-loop the calendar parser"),
    ("plan a feature voice barge-in", "/feature-dev voice barge-in"),
    ("run the playwright skill on the login page", "/playwright the login page"),
])
def test_skill_phrases_reach_claude_as_slash_commands(rich, services, text,
                                                      prompt):
    res = rich.handle(text, source="typed")
    args, kwargs = _submitted(services)
    assert args[0] == prompt
    assert kwargs["project"] == "jarvis"
    assert res.speak and res.reply == "Right away, sir."
    assert services.classify_calls == []          # no model round trip


def test_unknown_slash_command_passes_through(rich, services):
    rich.handle("/deploy-preview staging", source="typed")
    args, _ = _submitted(services)
    assert args[0] == "/deploy-preview staging"


# --------------------------------------------------- router → services
def test_local_route_goes_to_brain_chat(rich, services):
    res = rich.handle("what's on my calendar tomorrow", source="typed")
    assert services.brain.chat.call_count == 1
    assert services.brain.chat.call_args.args == ("what's on my calendar tomorrow",)
    assert res.handled and not res.done
    services.claude.submit.assert_not_called()


def test_claude_route_acknowledges_in_persona_then_submits(rich, services):
    services.brain.local_line.return_value = \
        "Right away, sir — fixing the failing test."
    res = rich.handle("fix the failing test in the parser", source="typed")
    args, kwargs = _submitted(services)
    assert args[0] == "fix the failing test in the parser"
    assert kwargs["parallel"] is False
    assert res.reply == "Right away, sir — fixing the failing test."
    assert res.speak and not res.done


def test_claude_ack_falls_back_when_the_local_model_is_down(rich, services):
    services.brain.local_line.side_effect = RuntimeError("ollama down")
    res = rich.handle("fix the failing test in the parser", source="typed")
    assert res.reply == CLAUDE_ACK_FALLBACK


def test_a_queue_line_from_the_manager_is_spoken_verbatim(rich, services):
    services.claude.submit.return_value = \
        "Claude's still on the last one for jarvis, sir; I've queued it."
    res = rich.handle("fix the failing test in the parser", source="typed")
    assert res.reply.startswith("Claude's still on the last one")
    assert res.speak and not res.done


def test_large_tasks_escalate_to_the_big_model(rich, services):
    rich.handle("refactor the parser and the router in the vss project",
                source="typed")
    args, kwargs = _submitted(services)
    assert kwargs["model"] == "fable"
    assert kwargs["project"] == "vss"


def test_a_spoken_model_override_sticks_and_is_passed(rich, services):
    rich.handle("fix the failing test with opus", source="typed")
    services.claude.set_model.assert_called_once_with("opus")
    _, kwargs = _submitted(services)
    assert kwargs["model"] == "opus"


def test_parallel_is_passed_through(rich, services):
    rich.handle("run the tests at the same time", source="typed")
    _, kwargs = _submitted(services)
    assert kwargs["parallel"] is True


@pytest.mark.parametrize("text,method,kwargs", [
    ("cancel", "cancel", {}),
    ("stop that", "cancel", {}),
    ("work on the haymaker project", "work_on", {"name": "haymaker"}),
    ("use fable", "set_model", {"alias": "fable"}),
    ("fast mode on", "set_fast_mode", {"on": True}),
    ("start a new project called weather station", "new_project",
     {"name": "weather station"}),
])
def test_router_actions_dispatch_to_the_session_manager(rich, services, text,
                                                        method, kwargs):
    res = rich.handle(text, source="typed")
    fn = getattr(services.claude, method)
    assert fn.call_count == 1, fn.call_args_list
    for k, v in kwargs.items():
        assert fn.call_args.kwargs.get(k) == v, (k, fn.call_args)
    assert res.handled and res.speak
    services.claude.submit.assert_not_called()


def test_cancel_replies_with_the_persona_line(rich, services):
    services.claude.cancel.return_value = True
    assert rich.handle("cancel", source="typed").reply == STOPPED_LINE
    services.claude.cancel.return_value = False
    assert rich.handle("cancel", source="typed").reply != STOPPED_LINE


def test_resume_passes_the_whole_utterance(rich, services):
    services.claude.resume.return_value = "Picking up the VSS labeler, sir."
    res = rich.handle("pick up where we left off", source="typed")
    assert services.claude.resume.call_count == 1
    assert services.claude.resume.call_args.kwargs["utterance"] == \
        "pick up where we left off"
    assert res.reply == "Picking up the VSS labeler, sir."


def test_switch_to_a_window_is_still_a_window_target(rich, services):
    rich.handle("switch to opera", source="typed")
    services.desktop.target_window.assert_called_once_with("opera")
    services.claude.work_on.assert_not_called()


def test_missing_claude_service_speaks_a_setup_line(services, tmp_path,
                                                    monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG",
                        tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    services.assistant = FakeAssistantCfg()
    services.router = Router(services.assistant,
                             classify=lambda t: ("local", 0.2))
    c = Commander(services)                      # no claude service
    res = c.handle("fix the failing test in the parser", source="typed")
    assert res.speak and "set up" in res.reply


# ---------------------------------------------------- Tier 1 precedence
def test_tier1_timer_beats_the_router(rich, services):
    res = rich.handle("timer for 5 minutes", source="typed")
    services.timekeeper.add_timer.assert_called_once_with(300, "5 minutes timer")
    assert res.speak and res.reply.startswith("5 minutes")
    services.brain.chat.assert_not_called()
    services.claude.submit.assert_not_called()
    assert services.classify_calls == []


def test_tier1_reminder_beats_the_router(rich, services):
    rich.handle("remind me in 10 minutes to stretch", source="typed")
    assert services.timekeeper.add_reminder.call_count == 1
    due, text = services.timekeeper.add_reminder.call_args.args
    assert text == "stretch" and due > 0
    services.brain.chat.assert_not_called()


def test_tier1_alarm_beats_the_router(rich, services):
    res = rich.handle("wake me up at 7 tomorrow", source="typed")
    assert services.timekeeper.add_alarm.call_count == 1
    assert services.timekeeper.add_alarm.call_args.args[2] == "once"
    assert "7:00 am tomorrow" in res.reply
    services.brain.chat.assert_not_called()


def test_tier1_schedule_list_and_cancel(rich, services):
    res = rich.handle("what timers do I have", source="typed")
    services.timekeeper.list_text.assert_called_once_with("timer")
    assert res.reply == "Nothing set, sir."
    res = rich.handle("cancel all my alarms", source="typed")
    services.timekeeper.cancel.assert_called_once_with("all", "alarm")
    assert res.reply == "Cancelled, sir."


def test_tier1_notes_and_todos(rich, services):
    services.notes.add.return_value = 1
    rich.handle("take a note that the boiler is broken", source="typed")
    services.notes.add.assert_called_once()
    assert services.notes.add.call_args.args[0] == "note"
    services.notes.add.reset_mock()
    # A bare "list" is still the to-do list; a NAMED one is not (spec 14 --
    # "add milk to my shopping list" used to land among the to-dos).
    rich.handle("add milk to my list", source="typed")
    assert services.notes.add.call_args.args[0] == "todo"
    services.brain.chat.assert_not_called()


def test_briefing_only_when_enabled(rich, services):
    # off: "good morning" is an ordinary greeting, answered by Tier 1 in
    # Jarvis's own voice (2026-08-26: it used to cost a full model turn)
    from jarvis.commander import COURTESY_REPLIES
    res = rich.handle("good morning", source="typed")
    assert res.reply in COURTESY_REPLIES["greeting"] and res.speak
    services.brain.chat.assert_not_called()
    # an explicit request still reaches the tool (which says it is off)
    rich.handle("what's my briefing", source="typed")
    services.brain.chat.assert_called_once_with("what's my briefing",
                                                force_tool="get_briefing")
    services.brain.chat.reset_mock()
    services.assistant.data["briefing.enabled"] = True
    rich.handle("good morning", source="typed")
    services.brain.chat.assert_called_once_with("good morning",
                                                force_tool="get_briefing")


def test_last_mail_by_voice_reaches_tier1_and_pins_the_read_flag(rich, services,
                                                                   monkeypatch):
    """"What was my last email about?" arrives with NO "jarvis" prefix -- the
    hotword consumed it -- and that path runs ASSISTANT_TIER1, not REGISTRY.
    The command was first added to REGISTRY alone, so it never fired by
    voice, which is the only way it is used. The handler-level tests in
    test_last_mail_routing.py could not see that; this one drives handle()."""
    from jarvis.commander import _LAST_MAIL_HOURS
    monkeypatch.setattr(rich.intent, "classify",
                        lambda text: (IntentClassifier.YES, 0.9))
    res = rich.handle("What was my last email about?", source="voice")
    # Tier 1 hands the handler the lower-cased, de-punctuated text.
    services.brain.chat.assert_called_once_with(
        "what was my last email about", force_tool="get_mail",
        force_args={"limit": 1, "since_hours": _LAST_MAIL_HOURS,
                    "unread_only": False})
    assert res.handled and res.done is False
    # and typed, which skips the intent gate entirely
    services.brain.chat.reset_mock()
    rich.handle("my latest email", source="typed")
    assert services.brain.chat.call_args.kwargs["force_args"]["unread_only"] is False
    # "message" is not mail here: Discord, notes and sessions all use the word
    services.brain.chat.reset_mock()
    rich.handle("what was the last message you sent", source="typed")
    for call in services.brain.chat.call_args_list:
        assert call.kwargs.get("force_tool") != "get_mail", call


def test_mail_write_intents_are_not_answered_with_a_read(rich, services):
    """"reply to my latest email" names the newest message but wants
    something get_mail cannot do; forcing a read would answer the wrong
    question with confidence. These must fall through to the router."""
    for said in ("reply to my latest email", "delete my last email",
                 "forward the last mail to bob", "archive the most recent email"):
        services.brain.chat.reset_mock()
        rich.handle(said, source="typed")
        for call in services.brain.chat.call_args_list:
            assert call.kwargs.get("force_tool") != "get_mail", said


def test_a_time_question_naming_a_place_goes_to_get_time(rich, services):
    """"What's the time in London?" was answered with the home time: the
    Tier-1 clock matcher fired on "what's the time" and never let the model
    see "in London". get_time(location=...) already geocodes a city."""
    from jarvis.commander import clock_kind
    assert clock_kind("what's the time in london") is None
    assert clock_kind("what time is it in tokyo right now") is None
    assert clock_kind("what's the date in sydney") is None
    assert clock_kind("what's the time") == "time"
    assert clock_kind("what time is it in the morning") == "time"      # not a place
    assert clock_kind("what's the date today") == "date"
    services.brain.chat.reset_mock()
    rich.handle("what's the time in london", source="typed")
    assert services.brain.chat.call_count == 1, "the local clock answered instead of the router"


def test_assistant_tier1_is_a_subset_of_the_registry_in_order():
    names = [c.name for c in ASSISTANT_TIER1]
    reg = [c.name for c in REGISTRY]
    assert names and set(names) <= set(reg)
    assert names == [n for n in reg if n in set(names)]


# ------------------------------------------------------------- sources
def test_discord_text_skips_the_intent_gate(rich, services):
    # Voice would discard this as background chat; "discord" is typed.
    res = rich.handle("she said no way lol haha dude", source="discord")
    assert services.classify_calls == ["she said no way lol haha dude"]
    assert res.reply == ROUTER_QUESTION
    rich.handle("what's the weather", source="discord")
    assert services.brain.chat.call_count == 1
    assert services.brain.chat.call_args.args == ("what's the weather",)


def test_voice_still_gates_background_chat(rich, services):
    res = rich.handle("she said no way lol haha dude", source="voice")
    assert "Ignored" in res.status
    services.brain.chat.assert_not_called()


def test_a_ringing_alarm_outranks_dictation_but_only_for_ring_words(
        rich, services, monkeypatch):
    """Reversed 2026-08-30 (review finding #7): this test used to pin the
    OLD order -- dictation swallowed "stop", so a ringing alarm could not
    be silenced by voice until "end dictation". Now the ring words win
    while an alarm is actually ringing; everything else is still typed,
    and the approvals stage still never sees dictated text."""
    typed = []
    monkeypatch.setattr(rich, "_type_raw", lambda t: typed.append(t))
    services.timekeeper.ringing = types.SimpleNamespace(id="a1", label="Get up")
    services.approvals.pending.return_value = [object()]
    rich.handle("jarvis dictate")
    rich.handle("stop")
    assert typed == [], "the ring-stop was typed into the window"
    assert services.timekeeper.stop_ringing.called
    rich.handle("dear sir stop me if you have heard this")
    assert typed == ["dear sir stop me if you have heard this "]
    services.approvals.answer.assert_not_called()


def test_manager_calls_are_filtered_to_the_real_signatures(rich, services):
    """The router's arg bag is wider than the manager's signatures; the
    call is made once, with only the arguments the method accepts."""
    from jarvis.commander import _call_manager

    seen = []

    def resume(utterance=""):                    # spec 7.2 signature
        seen.append(utterance)
        return "Picking it up, sir."

    assert _call_manager(resume, {"utterance": "pick up where we left off",
                                  "when": "yesterday", "name": ""}) == \
        "Picking it up, sir."
    assert seen == ["pick up where we left off"]

    def cancel(project=None):
        return project
    assert _call_manager(cancel, {}) is None

    # a method that raises TypeError itself is not retried
    calls = []

    def boom(name):
        calls.append(name)
        raise TypeError("inside")
    with pytest.raises(TypeError):
        _call_manager(boom, {"name": "x"})
    assert calls == ["x"]


def test_dispatch_action_survives_a_manager_error(rich, services):
    services.claude.work_on.side_effect = RuntimeError("no such project")
    res = rich.handle("work on the haymaker project", source="typed")
    assert res.handled and res.speak and res.reply.endswith("sir.")


# ------------------------------------------------- the address is not chatter
def test_addressed_voice_commands_skip_the_intent_gate(rich, services):
    """The classifier calls every one- or two-word phrase background chat;
    an explicit "jarvis" is address enough to reach the router."""
    for text, method in (("jarvis cancel", "cancel"),
                         ("hey jarvis abort", "cancel"),
                         ("jarvis use fable", "set_model")):
        services.claude.reset_mock()
        services.claude.active_project = "jarvis"
        res = rich.handle(text, source="voice")
        assert getattr(services.claude, method).call_count == 1, text
        assert "Ignored" not in (res.status or ""), text


def test_unaddressed_background_chat_is_still_ignored(rich, services):
    res = rich.handle("she said no way lol haha dude", source="voice")
    assert "Ignored" in res.status
    services.claude.submit.assert_not_called()


def test_the_local_model_never_sees_the_address(rich, services):
    from jarvis.commander import strip_address
    assert strip_address("Hey Jarvis, what's the weather?") == \
        "what's the weather?"
    assert strip_address("jarvis: what's on my calendar") == \
        "what's on my calendar"
    assert strip_address("what's the weather?") == "what's the weather?"
    assert strip_address("jarvis") == "jarvis"        # nothing but the name
    rich.handle("hey jarvis what's the weather", source="voice")
    assert services.brain.chat.call_count == 1
    assert services.brain.chat.call_args.args == ("what's the weather",)


# ------------------------------------------- (h) an open terminal (rule 1b)
def _terminals(services, *open_slugs):
    """Wire services.claude.terminal_open like ClaudeSessionManager's, and
    record what the commander asks about."""
    asked = []

    def terminal_open(project=None):
        asked.append(project)
        return (project or "") in open_slugs
    services.claude.terminal_open = terminal_open
    return asked


def test_in_the_terminal_queues_into_that_session_and_says_which(rich, services):
    asked = _terminals(services, "jarvis")
    res = rich.handle("in the terminal, run the tests", source="typed")
    args, kwargs = _submitted(services)
    assert args[0] == "run the tests"
    assert kwargs["project"] == "jarvis"
    assert asked == ["jarvis"], asked
    assert res.reply == "Through to the jarvis session, sir."
    assert res.speak and not res.done
    # the acknowledgement names the session itself: no model turn needed
    services.brain.local_line.assert_not_called()
    services.brain.chat.assert_not_called()


def test_tell_it_to_reaches_the_open_session(rich, services):
    _terminals(services, "jarvis")
    res = rich.handle("tell it to run the tests", source="typed")
    args, _ = _submitted(services)
    assert args[0] == "run the tests"
    assert res.reply == "Through to the jarvis session, sir."


def test_a_named_terminal_beats_the_active_project(rich, services):
    asked = _terminals(services, "haymaker")
    res = rich.handle("fix the parser in the haymaker terminal", source="typed")
    _, kwargs = _submitted(services)
    assert kwargs["project"] == "haymaker"
    assert asked == ["haymaker"]
    assert res.reply == "Through to the haymaker session, sir."


def test_with_no_terminal_open_the_utterance_routes_as_before(rich, services):
    asked = _terminals(services)                      # nothing attached
    res = rich.handle("in the terminal, run the tests", source="typed")
    args, _ = _submitted(services)
    assert args[0] == "in the terminal, run the tests"   # nothing stripped
    assert asked == ["jarvis"]
    assert res.reply == "Right away, sir."               # the usual ack
    services.brain.local_line.assert_called_once()


def test_with_no_terminal_open_tell_it_to_stays_local(rich, services):
    _terminals(services)
    rich.handle("tell it to stop", source="typed")
    services.claude.submit.assert_not_called()
    services.brain.chat.assert_called_once_with("tell it to stop")


def test_a_session_manager_without_terminal_open_cannot_crash_the_router(
        rich, services):
    del services.claude.terminal_open                 # older wiring
    res = rich.handle("in the terminal, run the tests", source="typed")
    args, _ = _submitted(services)
    assert args[0] == "in the terminal, run the tests"
    assert res.handled and res.speak


def test_a_broken_terminal_probe_is_survivable(rich, services):
    def boom(project=None):
        raise RuntimeError("tmux is not running")
    services.claude.terminal_open = boom
    res = rich.handle("in the terminal, run the tests", source="typed")
    args, _ = _submitted(services)
    assert args[0] == "in the terminal, run the tests"
    assert res.handled


def test_a_queue_line_from_the_manager_still_wins_over_the_terminal_line(
        rich, services):
    _terminals(services, "jarvis")
    services.claude.submit.return_value = \
        "Claude's still on the last one for jarvis, sir; I've queued it."
    res = rich.handle("in the terminal, run the tests", source="typed")
    assert res.reply.startswith("Claude's still on the last one")


# ------------------------------------------------ confirming a calendar add
#
# add_event writes outright when the parse is unambiguous and otherwise reads
# its interpretation back. The user's plain "yes" then has to mean THAT event,
# not a new command -- the same shape as the pending terminal offer.
def _pending(services, title="Lab presentation"):
    start = datetime(2026, 8, 31, 16, 10).astimezone()
    cal = types.SimpleNamespace(
        pending_event={"title": title, "start": start,
                       "end": start + timedelta(hours=1), "calendar": None},
        icloud_calendars=lambda: ["CAL"], written=[])
    services.calendar = cal
    return cal


def _fake_add(cal, undo=None):
    """Stand in for calendar.add_event -> (line, undo)."""
    def _add(cals, title, s, e, calendar_name=None):
        cal.written.append(title)
        return f"Added {title}, sir.", (undo or (lambda: "Taken back, sir."))
    return _add


def test_yes_writes_the_event_that_was_read_back(cmdr, services, monkeypatch):
    cal = _pending(services)
    import jarvis.commander as cmd_mod
    monkeypatch.setattr(cmd_mod, "add_event", _fake_add(cal))

    res = cmdr.handle("yes", "voice")

    assert cal.written == ["Lab presentation"]
    assert "Added" in res.reply and res.speak
    assert cal.pending_event is None, "the offer must not linger"


def test_scratch_that_takes_the_event_back_off_the_calendar(cmdr, services,
                                                            monkeypatch):
    """The add used to be add-only, so "scratch that" fell through in
    silence -- which reads as success while the event sits in his calendar."""
    cal = _pending(services)
    removed = []
    import jarvis.commander as cmd_mod
    monkeypatch.setattr(cmd_mod, "add_event",
                        _fake_add(cal, undo=lambda: removed.append(1) or
                                  "Taken back off your calendar, sir."))
    cmdr.handle("yes", "voice")

    res = cmdr.handle("scratch that", "voice")

    assert removed == [1]
    assert res.handled and "Taken back" in res.reply


def test_scratch_that_reaches_an_add_made_through_the_tool(cmdr, services):
    """The confident path writes from inside the TOOL and parks its undo on
    the calendar source; a ToolResult has no undo slot to carry one."""
    import time as _time
    removed = []
    services.calendar = types.SimpleNamespace(
        pending_event=None,
        last_add={"undo": lambda: removed.append(1) or "Taken back off your "
                                                       "calendar, sir.",
                  "at": _time.monotonic(), "title": "Standup"})

    res = cmdr.handle("scratch that", "voice")

    assert removed == [1] and "Taken back" in res.reply
    assert services.calendar.last_add is None, "a second scratch must not repeat it"


def test_a_stale_calendar_add_is_not_undone_by_a_later_scratch(cmdr, services):
    """A minute later "scratch that" is about something else entirely."""
    import time as _time
    removed = []
    services.calendar = types.SimpleNamespace(
        pending_event=None,
        last_add={"undo": lambda: removed.append(1) or "gone",
                  "at": _time.monotonic() - commander.UNDO_WINDOW_S - 1,
                  "title": "Standup"})

    cmdr.handle("scratch that", "voice")

    assert removed == []


def test_an_add_the_server_cannot_undo_says_so(cmdr, services, monkeypatch):
    """calendar._undo_add's third outcome: no delete path, so the honest
    answer is that it cannot be taken back -- never a bare "done"."""
    from jarvis.tools.calendar import CANNOT_UNDO_LINE
    cal = _pending(services)
    import jarvis.commander as cmd_mod
    monkeypatch.setattr(cmd_mod, "add_event", _fake_add(
        cal, undo=lambda: CANNOT_UNDO_LINE.format(title="Lab presentation")))
    cmdr.handle("yes", "voice")

    res = cmdr.handle("scratch that", "voice")

    assert "can't take one back" in res.reply
    assert "Lab presentation" in res.reply


def test_no_drops_it_without_writing(cmdr, services, monkeypatch):
    cal = _pending(services)
    import jarvis.commander as cmd_mod
    monkeypatch.setattr(cmd_mod, "add_event",
                        lambda *a, **kw: cal.written.append("SHOULD NOT HAPPEN"))

    res = cmdr.handle("no", "voice")

    assert cal.written == []
    assert cal.pending_event is None
    assert res.handled


def test_an_unrelated_utterance_drops_the_offer_rather_than_writing(
        cmdr, services, monkeypatch):
    """Changing the subject must never be read as consent."""
    cal = _pending(services)
    import jarvis.commander as cmd_mod
    monkeypatch.setattr(cmd_mod, "add_event",
                        lambda *a, **kw: cal.written.append("SHOULD NOT HAPPEN"))

    cmdr.handle("what's the weather tomorrow", "voice")

    assert cal.written == []
    assert cal.pending_event is None


def test_a_write_failure_is_reported_not_swallowed(cmdr, services, monkeypatch):
    cal = _pending(services)
    import jarvis.commander as cmd_mod

    def boom(*a, **kw):
        raise RuntimeError("412 precondition failed")

    monkeypatch.setattr(cmd_mod, "add_event", boom)
    res = cmdr.handle("yes", "voice")

    assert "couldn't add" in res.reply.lower()
    assert cal.pending_event is None


def test_a_web_cue_by_voice_skips_the_intent_gate(rich, services, monkeypatch):
    """Live: "Lookup who won the last Formula One [race]" was called uncertain
    and answered "Was that for me?". A web cue is addressed by construction."""
    def _boom(text):
        raise AssertionError("reached the intent classifier")
    monkeypatch.setattr(rich.intent, "classify", _boom)
    services.brain.web_answer.return_value = object()
    res = rich.handle("look up who won the last formula one race", source="voice")
    services.brain.web_answer.assert_called_once()
    assert res.done is False and res.ack


# ------------------------------------------------------------ next exam
# "When's my next exam?" without a model turn: Canvas (token set) merged
# with the calendar cache through tools/canvas.find_next_exam.
import jarvis.commander as _cmd_mod  # noqa: E402


@pytest.mark.parametrize("text, query", [
    ("when's my next exam", "exam"),
    ("when is my next midterm?", "midterm"),
    ("When is the next quiz", "quiz"),
    ("how long until the biosensors midterm", "biosensors midterm"),
    ("how many days until my final exam", "final exam"),
    ("how long till the circuits quiz?", "circuits quiz"),
    ("when's the biosensors midterm", "biosensors midterm"),
])
def test_next_exam_regex_hands_over_the_query(text, query):
    m = _cmd_mod._NEXT_EXAM_RX.match(text.lower().rstrip(".!?"))
    assert m, text
    assert (m.group("q1") or m.group("q2")).strip() == query


@pytest.mark.parametrize("text", [
    "when's my next meeting", "how long until dinner", "exam", "when is the exam hall open",
    "how long until the exam results come out",
])
def test_next_exam_regex_leaves_other_questions_alone(text):
    assert not _cmd_mod._NEXT_EXAM_RX.match(text)


def _exam_cal(*events):
    return types.SimpleNamespace(configured=True, events=lambda: list(events))


def _event(title, start, all_day=False):
    return types.SimpleNamespace(title=title, start=start, all_day=all_day, calendar="Canvas")


# 2026-09-05: the exam tests below built their instant as
#     (datetime.now().astimezone() + timedelta(days=3)).replace(hour=13)
# which carries TODAY's UTC offset onto a date three days away. When a DST
# change falls inside that horizon the instant is an hour out, and the
# product -- which reads real local dates -- then disagrees about both "in
# 3 days" and "at 1:00 pm". Measured red on 2026-10-30/31 and 2027-03-13/14,
# and reproduced live under TZ=America/Santiago and TZ=Pacific/Easter, both
# of which change over this weekend. The cure is to do the wall-clock
# arithmetic NAIVE and attach the zone LAST, so Python resolves the offset
# that actually applies to the day we land on.
def _in_days(days, hour, minute=0):
    """A local wall-clock instant `days` from now, at hour:minute."""
    return (datetime.now() + timedelta(days=days)).replace(
        hour=hour, minute=minute, second=0, microsecond=0).astimezone()


def test_next_exam_answers_from_the_calendar_without_a_token(rich, services, monkeypatch):
    import jarvis.tools.canvas as cv
    monkeypatch.setattr(cv, "fetch_due", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("Canvas must not be asked without a token")))
    start = _in_days(3, 13)
    services.calendar = _exam_cal(_event("Physics exam", start), _event("Dentist", start))
    res = rich.handle("when's my next exam", source="typed")
    assert res.handled and res.speak
    assert res.reply.startswith("Your next exam is the Physics exam, in 3 days, ")
    assert res.reply.endswith(" at 1:00 pm, sir.")
    services.brain.chat.assert_not_called()
    res = rich.handle("how long until the physics exam", source="typed")
    assert res.reply.startswith("The Physics exam, in 3 days, ")


def test_next_exam_without_a_token_and_no_calendar_hit_falls_through(rich, services):
    """Nothing on the calendar and no token: the router's model turn reaches
    canvas_due, whose setup line names what is missing. A flat "nothing on
    the books" here would vouch for a source that was never read."""
    services.calendar = _exam_cal(
        _event("Dentist", (datetime.now() + timedelta(days=1)).astimezone()))
    res = rich.handle("when's my next exam", source="typed")
    assert res.status != "No exam found"
    # the router's rule path routes a question locally: a model turn ran
    services.brain.chat.assert_called_once()
    assert services.brain.chat.call_args[0][0] == "when's my next exam"


def test_next_exam_with_a_token_merges_canvas_and_speaks_the_course(rich, services, monkeypatch):
    import jarvis.tools.canvas as cv
    services.assistant = FakeAssistantCfg(**{"canvas.token": "7~abcDEF123secret"})
    seen = {}

    def fake_fetch_due(settings, days, fetch, when):
        seen["days"] = days
        assert settings["token"] == "7~abcDEF123secret"
        return [{"course": "BIOSENSORS", "title": "Midterm 1",
                 "due": _in_days(6, 9)},
                {"course": "CIRCUITS", "title": "Quiz 2",
                 "due": _in_days(1, 17)}]
    monkeypatch.setattr(cv, "fetch_due", fake_fetch_due)
    services.calendar = None
    res = rich.handle("how long until the biosensors midterm", source="typed")
    assert seen["days"] == cv.EXAM_LOOKAHEAD_DAYS
    assert res.reply.startswith("The Midterm 1 for BIOSENSORS, in 6 days, ") and res.speak
    res = rich.handle("when is my next quiz", source="typed")
    assert res.reply == "Your next quiz is the Quiz 2 for CIRCUITS, tomorrow at 5:00 pm, sir."
    # Canvas was read and holds nothing of the kind: say so, no model turn
    res = rich.handle("when's my next final", source="typed")
    assert res.reply == cv.NO_EXAM_LINE and res.status == "No exam found"
    monkeypatch.setattr(cv, "fetch_due", lambda *a, **k: [])
    res = rich.handle("how long until the quiz", source="typed")
    assert res.reply == cv.NO_QUIZ_LINE
    services.brain.chat.assert_not_called()


def test_next_exam_is_tier1_in_the_assistant_and_a_canvas_outage_falls_through(
        rich, services, monkeypatch):
    assert "next exam" in [c.name for c in ASSISTANT_TIER1]
    import jarvis.tools.canvas as cv
    services.assistant = FakeAssistantCfg(**{"canvas.token": "7~abcDEF123secret"})
    monkeypatch.setattr(cv, "fetch_due",
                        lambda *a, **k: (_ for _ in ()).throw(cv.CanvasError("unreachable")))
    services.calendar = None
    res = rich.handle("when's my next exam", source="typed")
    # Canvas was consulted (and failed quietly): "nothing on the books" is
    # honest about what could be read; nothing crashed the handler
    assert res.reply == cv.NO_EXAM_LINE
    monkeypatch.setattr(cv, "find_next_exam",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = rich.handle("when's my next exam", source="typed")
    assert res.status != "No exam found" and "Command failed" not in (res.reply or "")

# --------------------------------------------------- "what's Claude doing?"
def test_the_status_question_speaks_the_managers_digest(rich, services):
    """Router action status_text -> ClaudeSessionManager.status_text(), and
    the string it returns IS the reply (no persona paraphrase, no task)."""
    digest = "Claude's working on jarvis, sir; started just now, 2 files touched so far."
    services.claude.status_text.return_value = digest
    for text in ("what's claude doing?", "jarvis, how's claude getting on",
                 "is claude still working"):
        services.claude.reset_mock()
        services.claude.status_text.return_value = digest
        res = rich.handle(text, source="typed")
        assert services.claude.status_text.call_count == 1, text
        assert res.handled and res.speak and res.reply == digest, text
        assert res.done is True
        services.claude.submit.assert_not_called()


# ------------------------------------------------- clipboard -> Claude
TRACEBACK = ("Traceback (most recent call last):\n"
             "  File \"/home/hunterp/Jarvis/jarvis/router.py\", line 12, in route\n"
             "    raise ValueError(\"boom\")\n"
             "ValueError: boom\n")


@pytest.fixture
def clip_project(rich, services, tmp_path, monkeypatch):
    """An active project on tmp and a scripted xclip."""
    proj = tmp_path / "proj"
    proj.mkdir()
    services.claude.active_project = "proj"
    services.claude.project_for.return_value = types.SimpleNamespace(
        slug="proj", path=str(proj))
    clips = {"clipboard": TRACEBACK, "primary": "def f():\n    return 1\n"}
    asked = []

    def xclip(selection, run=None):
        asked.append(selection)
        return clips[selection]
    monkeypatch.setattr(commander.reader_mod, "_xclip", xclip)
    return types.SimpleNamespace(path=proj, clips=clips, asked=asked)


def _clip_files(proj):
    return sorted((proj / ".jarvis" / "clips").glob("*.txt"))


def test_the_clip_goes_to_claude_by_file_never_inline(rich, services, clip_project):
    """A traceback cannot ride the prompt: sanitize_keys collapses it to one
    line for send-keys, and the prompt is echoed to .prompt files, the bus
    and Discord.  So the clip is a 0600 file under the project and the
    prompt names it."""
    res = rich.handle("have Claude fix what I copied", source="typed")
    _, kwargs = _submitted(services)
    prompt = services.claude.submit.call_args.args[0]
    files = _clip_files(clip_project.path)
    assert len(files) == 1 and files[0].read_text() == TRACEBACK
    assert oct(files[0].stat().st_mode & 0o777) == "0o600"
    assert (clip_project.path / ".jarvis" / ".gitignore").read_text() == "*\n"
    assert prompt.startswith(f"fix the text in {files[0]}.")
    assert "read it first" in prompt and "boom" not in prompt
    assert kwargs["project"] == "proj"
    assert clip_project.asked == ["clipboard"]
    assert res.handled and res.speak and res.reply == "Right away, sir."


def test_the_clip_command_beats_the_bare_clipboard_reader(rich, services, clip_project):
    """REGISTRY's "clipboard" entry is a substring match that runs on any
    jarvis-prefixed text BEFORE the router; unordered, "Jarvis, have Claude
    fix the clipboard" would be read aloud instead of handed over."""
    names = [c.name for c in REGISTRY]
    assert names.index("clip to claude") < names.index("clipboard")
    res = rich.handle("jarvis, have claude fix the clipboard", source="typed")
    assert services.claude.submit.call_count == 1
    assert not (res.reply or "").startswith("Clipboard:")
    # ...and the same words unprefixed, as the hotword delivers them, land
    # in Tier 1 rather than in the router as the bare prompt "fix the clipboard"
    services.claude.reset_mock()
    services.claude.active_project = "proj"
    rich.handle("send what I copied to Claude and fix the error", source="typed")
    prompt = services.claude.submit.call_args.args[0]
    assert prompt.startswith("fix the error the text in ")


@pytest.mark.parametrize("text,selection,head", [
    ("ask claude about the clipboard", "clipboard", "Have a look at the text in "),
    ("have claude look at the selection and explain it", "primary", "look at the text in "),
    ("tell claude to fix what I highlighted", "primary", "fix the text in "),
    ("give claude the clipboard", "clipboard", "Have a look at the text in "),
    ("send the highlighted text to claude", "primary", "Have a look at the text in "),
    ("have Claude fix the copied traceback in TurnLedger", "clipboard", "fix the text in "),
])
def test_clip_phrasings(rich, services, clip_project, text, selection, head):
    rich.handle(text, source="typed")
    prompt = services.claude.submit.call_args.args[0]
    assert prompt.startswith(head), (text, prompt)
    assert clip_project.asked == [selection], text
    if "explain it" in text:
        assert " and explain it." in prompt
    if "TurnLedger" in text:
        assert "in TurnLedger" in prompt            # casing kept from the raw text


def test_code_nouns_are_not_a_clip(rich, services, clip_project):
    """"the selection logic" is code, not the X selection: routes as a
    plain Claude task with the words intact and reads no clipboard."""
    rich.handle("have claude change the selection logic in the picker", source="typed")
    prompt = services.claude.submit.call_args.args[0]
    assert prompt == "change the selection logic in the picker"
    assert clip_project.asked == []


def test_an_empty_clip_is_said_not_sent(rich, services, clip_project):
    clip_project.clips["clipboard"] = "  \n"
    res = rich.handle("have claude fix what i copied", source="typed")
    assert res.reply == commander.CLIP_EMPTY_LINE and res.speak
    clip_project.clips["primary"] = ""
    res = rich.handle("have claude fix the selection", source="typed")
    assert res.reply == commander.SELECTION_EMPTY_LINE
    services.claude.submit.assert_not_called()
    assert _clip_files(clip_project.path) == []


def test_no_active_project_means_no_file_and_the_project_line(rich, services, clip_project):
    from jarvis.claude_session import NO_PROJECT_LINE
    services.claude.active_project = None
    res = rich.handle("have claude fix what i copied", source="typed")
    assert res.reply == NO_PROJECT_LINE and res.speak
    services.claude.submit.assert_not_called()
    assert _clip_files(clip_project.path) == []


# --------------------------------------------------------------- the Board
@pytest.fixture
def board_svc(rich, services):
    """services.board as the app wires it: show/hide reach the window over
    the bus, read() answers with the panel's one-line spoken state."""
    svc = types.SimpleNamespace(shown=0, hidden=0, asked=[])

    def show():
        svc.shown += 1
        return svc.shown > 1                 # True == it was already up

    def hide():
        svc.hidden += 1
        return True

    def read(panel):
        svc.asked.append(panel)
        return ("Two tasks running, sir." if "session" in panel else "")

    services.board = types.SimpleNamespace(show=show, hide=hide, read=read)
    return svc


def test_bring_up_the_board_raises_it_and_says_so(rich, board_svc):
    res = rich.handle("bring up the board", source="voice")
    assert res.handled and res.speak and board_svc.shown == 1
    assert res.reply == "The board, sir."


def test_asking_twice_does_not_pretend_it_just_appeared(rich, board_svc):
    rich.handle("show me the board", source="voice")
    res = rich.handle("board up", source="voice")
    assert res.reply == "Already up, sir." and board_svc.shown == 2


def test_close_the_board_takes_it_down(rich, board_svc):
    res = rich.handle("close the board", source="voice")
    assert res.handled and board_svc.hidden == 1
    assert res.reply == "Board down, sir."


def test_focus_on_a_panel_speaks_its_state_rather_than_lighting_it(
        rich, board_svc):
    res = rich.handle("focus on the sessions", source="voice")
    assert res.reply == "Two tasks running, sir." and res.speak
    assert board_svc.asked == ["sessions"]


def test_focus_on_something_that_is_not_a_panel_stays_with_its_old_owner(
        rich, board_svc):
    """The bare word "focus" belongs to the voice-targeting chain, and
    "focus session on the thesis" to the pomodoro. Only a name that
    RESOLVES to a panel is carved out of that chain, so an unrelated
    "focus on ..." reaches neither the board nor a wrong answer."""
    res = rich.handle("focus on the thesis", source="typed")
    assert board_svc.asked == []
    assert res.status == "Target: on the thesis"


def test_a_board_that_refuses_to_raise_says_so_instead_of_failing(
        rich, services):
    def boom():
        raise RuntimeError("no display")
    services.board = types.SimpleNamespace(show=boom, hide=lambda: True,
                                           read=lambda p: "")
    res = rich.handle("bring up the board", source="voice")
    assert res.handled and "couldn't raise the board" in res.reply


def test_without_the_board_service_the_verb_is_simply_not_claimed(rich,
                                                                  services):
    services.board = None
    res = rich.handle("bring up the board", source="typed")
    assert res.status != "Board"


def test_a_manager_refusal_is_spoken_as_is(rich, services, clip_project):
    services.claude.submit.return_value = "Claude's still on the last one for proj, sir; I've queued it."
    res = rich.handle("have claude fix what i copied", source="typed")
    assert res.reply.startswith("Claude's still on the last one") and res.speak


# ---------------------------------------------------------------------------
# Review round 2026-08-30: the intent gate vs the unprefixed Tier-1 commands.
# "standup" and "review my flashcards" were classified NO and dropped
# silently -- the third recurrence of the silent-drop bug (media words
# 08-27, study words 08-30). A Tier-1 match now bypasses the gate, and this
# table forces every FUTURE Tier-1 command to prove its phrase survives.
# ---------------------------------------------------------------------------
TIER1_SAMPLES = {
    "explain document": "explain the biosensors lab handout",
    "quiz": "quiz me on chapter three",
    "scan syllabus": "scan my syllabus for dates",
    "review flashcards": "review my flashcards",
    "stop quiz": "stop the quiz",
    "teach me": "teach me biosensors",
    "board show": "bring up the board",
    "board hide": "close the board",
    "board focus": "focus on the sessions",
    "plan week": "let's plan the week",
    "focus start": "start a focus session",
    "focus left": "how long left",
    "focus end": "end the session",
    "study total": "how much did i study this week",
    "study streak": "what's my streak",
    "lecture notes": "notes for biosensors",
    "timer": "set a timer for five minutes",
    "alarm": "set an alarm for seven",
    "no asides": "no more asides",
    "list schedule": "any timers running",
    "cancel schedule": "cancel the timer",
    "adjust schedule": "extend that timer by ten minutes",
    "briefing": "give me my briefing",
    "preview": "what does tomorrow look like",
    "week": "how's my week looking",
    "briefing section": "no news in the morning",
    "verbosity": "shorter briefings",
    "last mail": "what was my last email",
    # The send-a-file family. The sample is his own phrasing from the
    # brief; it arms a read-back and never a send, so the gate test can
    # run it safely.
    "send file": "email the lab report to heather",
    "sent files": "what did i email today",
    "liked songs": "play my liked songs",
    "music resume": "start playing my spotify",   # the 20:56:42 clause
    "diagnostics": "run diagnostics",
    "next exam": "when's my next exam",
    "next class": "what's my next class",       # the 10:10:33 misroute
    "leave time": "it takes ten minutes to get to wisenbaker",
    "leave time amend": "make that ten next time",
    "leave time query": "how long to wisenbaker",
    "leave time forget": "forget the walk to wisenbaker",
    "greeting": "good morning",
    "courtesy": "good night",
    "day review": "how did yesterday go",
    "list add": "add milk to the shopping list",
    "list read": "read my packing list",
    "list strike": "take milk off the shopping list",
    "list strike anon": "cross the second one off the list",
    "list clear": "clear the shopping list",
    # grab and throw by voice (jarvis/gesturecast.py)
    "cast throw": "throw this on hpcomputer",
    "cast put": "put it on the board",
    "cast drop": "drop it",
    "cast holding": "what am i holding",
    "cast side": "which side is hpcomputer on",
    "cast teach": "hpcomputer is on my right",
    "lists": "what lists do i have",
    "week review": "how was my week",
    "garden report": "memory report",
    "garden undo": "forget the last garden pass",
    "last seen": "when did i last talk to my advisor",
    "register": "formal mode",
    "todo done": "mark buy milk as done",
    "todo add": "add buy milk to my todo list",
    "todo list": "what's on my todo list",
    "take note": "take a note buy milk",
    "show notes": "show my notes",
    "answer question": "what's your ip address",
    "remind me": "remind me to call mum at five",
    "person": "my advisor is Dr Peyrovi",
    "remember": "remember that my dentist is dr patel",
    "recall": "what did i say about the thesis",
    "who is": "who's my advisor",
    "recap": "recap my day",
    "quiet status": "are you on do not disturb",
    "quiet hours off": "turn off quiet hours",
    "quiet hours": "quiet hours from eleven to seven",
    "do not disturb": "do not disturb for an hour",
    "free": "i am free",
    "room tone": "room tone on",
    # offline mode (jarvis/sensing.py) -- his own words for four of these
    "sensing off": "offline mode",
    "face enrol": "enrol my face",
    "face forget": "forget heather's face",
    "face gallery": "who do you recognise",
    "sensing on": "come back online",
    "sensing status": "are you watching",
    "sensing hold": "no cameras for the next two hours",
    "sensing curfew": "camera curfew from nine to seven",
    "ui look": "switch to classic visuals",
    # 2026-09-02 23:26, verbatim -- the gate called it background chat
    "clear transcript": "clear the transcript",
    "audio out": "where's your voice coming out",
    "standup": "standup",
    "oracle status": "how's the oracle box",
    "oracle logs": "show me the haymaker logs",
    "oracle action": "restart the haymaker bot",
    "oracle service": "is knightfall up",
    "oracle freeform": "run deploy on the oracle box",
    # HPCOMPUTER's five doors, registered in Tier 1 for exactly the reason
    # the Oracle five are: the hotword eats the wake word, so every one of
    # them arrives bare and the prefixed registry pass never runs on it.
    "remote push": "put the budget on hpcomputer",
    "remote pull": "get the budget from hpcomputer",
    "remote status": "is hpcomputer up",
    "remote query": "what's the disk on hpcomputer",
    "remote freeform": "run the build on hpcomputer",
    "gpu reclaim": "take the gpu back",
    "gpu lend": "lend the gpu",
    "log triage": "anything wrong in your log",
    "whats wrong": "what's wrong",
    "quietly": "quietly please",
    "slow turn": "why was that slow",
    "clip to claude": "have claude fix what i copied",
    "read control": "skip",
    "math": "what's 18 percent of 74",
    "room light": "dim it a little",
    "scene": "power down the workshop",
}


def test_the_audio_out_question_takes_the_phrasing_people_actually_use():
    """Live 2026-08-31: "which speaker are you coming out of" fell through to
    the model while "coming out of" already sat in the gate vocabulary --
    the words were let past the classifier and then matched nothing."""
    import jarvis.commander as C
    for phrase in ("which speaker are you coming out of",
                   "what speaker are you playing through",
                   "which speaker are you on",
                   "where is your voice coming out"):
        assert C._AUDIO_OUT_RX.match(phrase), phrase
    assert not C._AUDIO_OUT_RX.match("which speaker do you like")


def test_every_tier_one_command_has_a_gate_sample():
    from jarvis.commander import ASSISTANT_TIER1
    names = {c.name for c in ASSISTANT_TIER1}
    missing = names - set(TIER1_SAMPLES)
    extra = set(TIER1_SAMPLES) - names
    assert not missing and not extra, (
        f"add a sample for {sorted(missing)} / drop {sorted(extra)}")


def test_tier_one_samples_survive_the_intent_gate(tmp_path, monkeypatch):
    """Every sample either matches its Tier-1 regex outright (gate bypass)
    or at least classifies as not-NO. Fails when a future command's words
    are missing from both the matchers' reach and the vocabulary."""
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "l.json")
    c = object.__new__(Commander)
    ic = IntentClassifier()
    for name, phrase in TIER1_SAMPLES.items():
        if c._match_assistant(phrase):
            continue
        verdict, conf = ic.classify(phrase)
        assert verdict != IntentClassifier.NO, (name, phrase, conf)


# The two Tier-1 families that landed in the same merge (integration-0903):
# enrol-in-app's face rungs sit HIGH in the registry so "remember Heather's
# face" beats "remember", "forget Heather's face" beats "leave time forget"
# and "add Heather's face" beats "list add"; file-and-remote's send/remote
# rungs sit LOWER and key on send/email/put/get verbs with a file or a
# machine. Each branch proved its own ladder; nobody had run the UNION. The
# gate test above only asks that a sample matches *some* rung, which would
# not notice "forget heather's face" being swallowed by "leave time forget",
# so this one asks which rung answers FIRST -- the rung that actually runs.
UNION_FIRST_RUNG = {
    # face family, incl. the verbs it shares with older rungs
    "enrol my face": "face enrol",
    "add heather's face to the gallery": "face enrol",
    "remember heather's face": "face enrol",
    "register me": "face enrol",
    "forget heather's face": "face forget",
    "delete heather's face": "face forget",
    "remove ali's face from the gallery": "face forget",
    "who do you recognise": "face gallery",
    "am i enrolled": "face gallery",
    # send/remote family
    "email the lab report to heather": "send file",
    "put the budget on hpcomputer": "remote push",
    "copy the lab report to hpcomputer": "remote push",
    "get the budget from hpcomputer": "remote pull",
    "is hpcomputer up": "remote status",
    "what's the disk on hpcomputer": "remote query",
    "run the build on hpcomputer": "remote freeform",
    # a sentence with BOTH families' words in it is a transfer, not an
    # enrolment: the verb decides, and the send/remote claim rules then
    # judge whether "face" names a file at all
    "send heather's face to hpcomputer": "remote push",
    "email my face to heather": "send file",
    # the cast verbs (gesture-cast) sit ABOVE the remote five in the
    # registry, pinned here so the merge order is a decision and not an
    # accident: an object that is only this/it/that, aimed at a sink the
    # cast table knows, is the throw by voice (the held subject, else the
    # one resolved now); a NAMED file to the same host is still the
    # transfer with its read-back
    "put this on hpcomputer": "cast put",
    "send it to the hp": "cast throw",
    "put this file on hpcomputer": "remote push",
    "send this file to hpcomputer": "remote push",
    # ...and the neighbours each family had to beat still answer their own
    "forget the walk to wisenbaker": "leave time forget",
    "remember that the lab is on tuesday": "remember",
    "add milk to the shopping list": "list add",
}


@pytest.mark.parametrize("phrase,expected", sorted(UNION_FIRST_RUNG.items()))
def test_face_and_remote_families_do_not_shadow_each_other(phrase, expected):
    from jarvis.commander import ASSISTANT_TIER1
    t = phrase.strip().lower().rstrip(".!?")
    first = next((c.name for c in ASSISTANT_TIER1 if c.matcher(t)), None)
    assert first == expected, (phrase, first)


# The dialogue-board phrasings that MISS their Tier-1 regex and so fall
# through to the classifier. The exact-match bypass cannot help here, which
# is precisely why the vocabulary has to carry them: the silent drop this
# guards against has already recurred three times (media words 08-27, study
# words and the review round 08-30). A dropped session opener is the worst
# of the family -- it is turn one of a conversation, so the silence reads as
# him ignoring you rather than mishearing you.
LOOSE_GATE_PHRASES = [
    "sort out my week for me",
    "can we plan the week out tonight",
    "map out the week when you get a chance",
    "anything wrong over there",
    "tell me what went wrong last night",
    "a bit more quietly please",
    "keep it down while the run is going",
    "stop narrating the epochs",
]


@pytest.mark.parametrize("phrase", LOOSE_GATE_PHRASES)
def test_loose_dialogue_phrasings_survive_the_intent_gate(
        phrase, tmp_path, monkeypatch):
    """These deliberately do NOT match their Tier-1 regex, so they reach the
    classifier and only the vocabulary can save them."""
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "l.json")
    c = object.__new__(Commander)
    assert not c._match_assistant(phrase), (
        f"{phrase!r} now matches Tier-1 outright -- it no longer tests the "
        "vocabulary; pick a phrasing that still falls through")
    verdict, conf = IntentClassifier().classify(phrase)
    assert verdict != IntentClassifier.NO, (phrase, verdict, conf)


def test_a_tier_one_match_never_consults_the_classifier(rich, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the classifier was consulted for a Tier-1 match")
    monkeypatch.setattr(rich.intent, "classify", boom)
    res = rich.handle("good morning", source="voice")
    assert res.handled and res.speak


def test_intent_log_is_firewalled_for_the_whole_suite():
    import os as _os
    assert str(IntentClassifier.INTENT_LOG) == _os.environ["JARVIS_INTENT_LOG"]
    assert ".aiws_trainer" not in str(IntentClassifier.INTENT_LOG)


# ------------------------------------------------ ringing vs sticky modes
def test_ringing_alarm_is_reachable_during_lecture_notes(rich):
    import types as _t
    rich.lecture_course = "biosensors"
    added = []
    rich._lecture = _t.SimpleNamespace(add=lambda b: added.append(b) or 1,
                                       close=lambda: "closed")
    rich.services.timekeeper.ringing = True
    res = rich.handle("stop", source="voice")
    assert res.handled and added == [], "the ring-stop went into the notes"
    res = rich.handle("impedance is the ratio", source="voice")
    assert added == ["impedance is the ratio"], "a note line was eaten"


def test_lecture_recovery_keeps_the_source_and_skips_the_gate(rich, monkeypatch):
    rich.lecture_course = "ghost"
    rich._lecture = None
    def boom(*a, **k):
        raise AssertionError("a typed turn reached the voice gate")
    monkeypatch.setattr(rich.intent, "classify", boom)
    res = rich.handle("tell me something wonderful today", source="typed")
    assert rich.lecture_course is None and res is not None


# ------------------------------------------------ "what did I miss?"
def test_what_did_i_miss_reads_without_ending_the_window(rich):
    import types as _t
    q = _t.SimpleNamespace(
        free=MagicMock(return_value="free!"),
        release=MagicMock(return_value="While you were busy, sir: one reminder."))
    rich.services.quiet = q
    res = rich.handle("what did i miss", source="typed")
    assert q.release.called and not q.free.called
    assert res.reply.startswith("While you were busy")
    q.release.return_value = ""
    res = rich.handle("anything i missed", source="typed")
    from jarvis.quiet import NOTHING_HELD_LINE
    assert res.reply == NOTHING_HELD_LINE and not q.free.called
    rich.handle("i am free", source="typed")
    assert q.free.called


# ------------------------------------------------ turn serialization
def test_handle_serializes_concurrent_turns(rich, monkeypatch):
    import threading as _th
    import time as _time
    seen = []
    def slow_inner(text, source):
        before = rich._confidence
        _time.sleep(0.05)
        seen.append((text, before, rich._confidence))
        return CommandResult(handled=True, status="ok")
    monkeypatch.setattr(rich, "_handle_inner", slow_inner)
    ts = [ _th.Thread(target=rich.handle, args=(t, "voice"), kwargs={"confidence": c})
           for t, c in (("a", -0.1), ("b", -0.9)) ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for text, before, after in seen:
        want = -0.1 if text == "a" else -0.9
        assert before == want == after, seen


# ------------------------------------------------ feedback log bound
def test_feedback_log_is_bounded(rich, tmp_path, monkeypatch):
    fl = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(Commander, "FEEDBACK_LOG", fl)
    pad = '{"pad": "' + "x" * 120 + '"}'
    fl.write_text("\n".join(pad for _ in range(2500)) + "\n")
    assert fl.stat().st_size > 262144
    rich._feedback_line("the text", "the status", True, "card")
    lines = fl.read_text().splitlines()
    assert len(lines) == 1000 and '"how": "card"' in lines[-1]


# ------------------------------------------ spoken "good night" (2026-08-31)
# The hotword eats the wake word, so a courtesy arrives bare: strip_jarvis_prefix
# returns None, the whole prefixed-registry pass is skipped, and the intent gate
# called two words background chat and dropped the turn in silence. That left
# the entire wind-down (music fade, screen dim, do-not-disturb) unreachable by
# voice -- _h_courtesy on "goodnight" is its only call site. "good morning" was
# never affected, because "greeting" was in ASSISTANT_TIER1 and "courtesy" was
# not: the asymmetry is the whole bug.
def test_the_intent_gate_alone_would_drop_a_spoken_good_night(tmp_path,
                                                              monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "l.json")
    verdict, _conf = IntentClassifier().classify("good night")
    assert verdict == IntentClassifier.NO         # why the bypass has to exist


def test_a_spoken_good_night_reaches_the_wind_down(rich):
    wd = MagicMock()
    wd.start.return_value = True
    rich.services.winddown = wd
    res = rich.handle("good night", source="voice")
    assert wd.start.called
    assert res.handled and res.status != "Ignored (background chat)"


def test_the_courtesy_bypasses_the_gate_by_name(rich):
    """The bypass is a Tier-1 probe, so it is the same rung "good morning"
    has always used -- and it must not swallow anything else."""
    assert rich._match_assistant("good night") == "courtesy"
    assert rich._match_assistant("thank you") == "courtesy"
    assert rich._match_assistant("the roof is leaking") is None


def test_a_spoken_thank_you_is_still_answered_not_dropped(rich):
    res = rich.handle("thank you", source="voice")
    assert res.status == "Courtesy" and res.speak


# --------------------------------------------- the window look (2026-09-01)
# "Switch to classic visuals" writes console.look for the NEXT start (the
# window reads it once in main_window.create) and says so. Two grabs already
# own every "switch to X": the desktop chain on the prefixed path and
# TARGET_PATTERN on the bare one -- both would have targeted a window called
# "classic visuals", so both carve the look phrases out.
class _LookCfg(FakeAssistantCfg):
    def __init__(self, **over):
        super().__init__(**over)
        self.sets = []

    def set(self, key, value):
        self.sets.append((key, value))
        self.data[key] = value
        return True


@pytest.mark.parametrize("text, name, line", [
    ("switch to classic visuals", "classic", commander.UI_LOOK_LINES["classic"]),
    ("switch to the classic look", "classic", commander.UI_LOOK_LINES["classic"]),
    ("classic mode visuals", "classic", commander.UI_LOOK_LINES["classic"]),
    ("go back to the classic look, please", "classic", commander.UI_LOOK_LINES["classic"]),
    ("switch the visuals to classic", "classic", commander.UI_LOOK_LINES["classic"]),
    ("use the holo visuals", "holo", commander.UI_LOOK_LINES["holo"]),
    ("use the holographic look", "holo", commander.UI_LOOK_LINES["holo"]),
    ("holographic visuals", "holo", commander.UI_LOOK_LINES["holo"]),
    ("change the look to holographic", "holo", commander.UI_LOOK_LINES["holo"]),
    ("Switch to holo mode.", "holo", commander.UI_LOOK_LINES["holo"]),
])
def test_look_phrases_write_the_option_and_speak_the_restart_line(rich, text, name, line):
    rich.services.assistant = _LookCfg()
    for source in ("typed", "voice"):
        rich.services.assistant.sets.clear()
        res = rich.handle(text, source=source)
        assert res.handled and res.speak and res.reply == line, (text, source)
        assert rich.services.assistant.sets == [("console.look", name)], (text, source)
        assert "restart" in res.reply
        assert res.status == f"Visuals: {name} (restart)"
    rich.services.desktop.target_window.assert_not_called()
    rich.services.brain.chat.assert_not_called()
    rich.services.claude.submit.assert_not_called()


def test_a_prefixed_look_switch_is_not_a_window_target(rich):
    """The registry pass runs AFTER the desktop chain, whose parser owns
    "switch to X". With the real parser this became target_window("classic
    visuals"); the chain now steps aside for the look phrases."""
    from jarvis.desktop import parse_desktop_action
    rich.services.desktop.parse_action = parse_desktop_action
    rich.services.assistant = _LookCfg()
    res = rich.handle("jarvis, switch to classic visuals", source="voice")
    assert res.reply == commander.UI_LOOK_LINES["classic"]
    assert rich.services.assistant.sets == [("console.look", "classic")]
    rich.services.desktop.execute_actions.assert_not_called()
    rich.services.desktop.target_window.assert_not_called()
    # ...while an actual window switch still is one
    rich.handle("jarvis, switch to firefox", source="voice")
    assert rich.services.desktop.execute_actions.called


def test_a_bare_look_switch_is_not_a_window_target_either(rich):
    """TARGET_PATTERN in _route_text grabs every bare "switch to X" before
    Tier 1 runs; the look phrases are exempted like the project switch."""
    rich.services.assistant = _LookCfg()
    res = rich.handle("switch to the holographic look", source="typed")
    assert res.reply == commander.UI_LOOK_LINES["holo"]
    rich.services.desktop.target_window.assert_not_called()
    res = rich.handle("switch to opera", source="typed")
    assert res.status == "Target: opera"
    rich.services.desktop.target_window.assert_called_once_with("opera")


def test_look_switch_without_a_config_says_so_aloud(rich):
    """No needs=("assistant",): a silent fall-through would hand "switch to
    classic visuals" to the model, which cannot do it."""
    rich.services.assistant = None
    res = rich.handle("classic visuals", source="typed")
    assert res.handled and res.speak
    assert res.reply == commander.UI_LOOK_NO_CONFIG_LINE
    assert res.status == "Visuals: no config"
    rich.services.brain.chat.assert_not_called()
    # a config that cannot take the write is reported, not swallowed
    broken = _LookCfg()

    def boom(key, value):
        raise OSError("disk full")
    broken.set = boom
    rich.services.assistant = broken
    res = rich.handle("holographic visuals", source="typed")
    assert res.reply == commander.UI_LOOK_SAVE_FAILED_LINE and res.speak


@pytest.mark.parametrize("text", [
    "play classic rock", "switch to classic rock", "what is a hologram",
    "classic", "holographic", "look at the screen", "switch to the other monitor",
    "use the classic rock playlist", "look up holograms", "set the visuals",
    "take a look", "switch to the vss project",
    # 09-01 review: the soft openers matched WITHOUT a surface noun, and
    # "ui look" is Tier 1 (no intent gate), so an overheard "give me classic"
    # in an open listening window rewrote the config. Only switch/change/
    # set/flip + "to" may drop the noun.
    "give me classic", "i want the classic", "use classic", "show me the classic",
    "go to classic", "take me to classic", "put it to classic", "i'd like holo",
    "let's have classic",
])
def test_look_regex_leaves_everything_else_alone(text):
    assert commander._UI_LOOK_RX.match(text) is None, text


@pytest.mark.parametrize("text, name", [
    ("switch to classic", "classic"), ("flip back to holo", "holo"),
    ("set it to classic mode", "classic"), ("change to the holographic", "holo"),
    ("give me the classic visuals", "classic"), ("i want the holo look", "holo"),
    ("take me back to the classic visuals", "classic"),
])
def test_switching_verbs_may_drop_the_noun_but_soft_openers_may_not(rich, text, name):
    rich.services.assistant = _LookCfg()
    res = rich.handle(text, source="voice")
    assert res.reply == commander.UI_LOOK_LINES[name], text
    assert rich.services.assistant.sets == [("console.look", name)]


def test_a_pinning_env_var_is_confessed_not_promised_away(rich, monkeypatch):
    """theme.resolve_look lets JARVIS_LOOK outrank console.look, so with the
    env exported the "applies after a restart" line would be false. The
    write still happens (the pin may be lifted later); the reply says why
    nothing will change."""
    rich.services.assistant = _LookCfg()
    monkeypatch.setenv(commander.UI_LOOK_ENV, "holo")
    res = rich.handle("switch to classic visuals", source="voice")
    assert rich.services.assistant.sets == [("console.look", "classic")]
    assert res.handled and res.speak
    assert res.reply == commander.UI_LOOK_PINNED_LINE.format(
        pinned="holographic", env="JARVIS_LOOK")
    assert "restart" not in res.reply
    assert res.status == "Visuals: classic saved, env pins holo"
    # an env that agrees with the write, or that names junk, is not a pin
    rich.services.assistant.sets.clear()
    res = rich.handle("switch to holo visuals", source="voice")
    assert res.reply == commander.UI_LOOK_LINES["holo"]
    monkeypatch.setenv(commander.UI_LOOK_ENV, "neon")
    res = rich.handle("switch to classic visuals", source="voice")
    assert res.reply == commander.UI_LOOK_LINES["classic"]
    monkeypatch.delenv(commander.UI_LOOK_ENV)
    res = rich.handle("switch to classic visuals", source="voice")
    assert res.reply == commander.UI_LOOK_LINES["classic"]


def test_ui_look_is_a_tier_one_command_and_precedes_the_router(rich):
    assert rich._match_assistant("switch to classic visuals") == "ui look"
    assert rich._match_assistant("holographic visuals") == "ui look"
    names = [c.name for c in REGISTRY]
    assert "ui look" in names
    assert names.index("ui look") < names.index("quiet status")


# =========================================================================
# Extend / shorten a timer, reminder or alarm (live 2026-09-01 20:41)
# =========================================================================
# 20:40:54 "Set a timer for 10 minutes to pack up the chicken." -> Tier-1.
# 20:41:49 "Extend that timer by 10 minutes." -> "route local" -> the model,
# whose manage_schedule had no extend action, picked `list` and read the
# timer back: "One timer, sir: pack up the chicken in 9 minutes." The route
# is now a Tier-1 regex ("adjust schedule"), so it never depends on the
# model's choice again.
def _adjusted_item(**over):
    base = dict(id="tm-9", kind="timer", label="pack up the chicken",
                due=1_800_000_000.0, effective_due=1_800_000_000.0, state="pending")
    base.update(over)
    return types.SimpleNamespace(**base)


@pytest.mark.parametrize("text,which,kind,delta", [
    ("extend that timer by 10 minutes", "last", "timer", 600),
    ("extend the timer by ten minutes", "last", "timer", 600),
    ("extend my alarm by 15 minutes", "last", "alarm", 900),
    ("extend it by 5 minutes", "last", "all", 300),
    ("extend that by ten minutes", "last", "all", 600),
    ("extend the timer for the chicken by 10 minutes", "the chicken", "timer", 600),
    ("add 10 minutes to the timer", "last", "timer", 600),
    ("add ten minutes to that timer", "last", "timer", 600),
    ("add 5 minutes to my reminder", "last", "reminder", 300),
    ("put another 10 minutes on the timer", "last", "timer", 600),
    ("give the timer another 10 minutes", "last", "timer", 600),
    ("give me 10 more minutes on the timer", "last", "timer", 600),
    ("10 more minutes on the timer", "last", "timer", 600),
    ("push the timer back 10 minutes", "last", "timer", 600),
    ("push back my alarm by 15 minutes", "last", "alarm", 900),
    ("push that back ten minutes", "last", "all", 600),
    ("delay the reminder by 20 minutes", "last", "reminder", 1200),
    ("move my alarm back 30 minutes", "last", "alarm", 1800),
    ("make it 10 minutes longer", "last", "all", 600),
    ("make the timer ten minutes longer", "last", "timer", 600),
    ("shorten the timer by 5 minutes", "last", "timer", -300),
    ("shorten that by five minutes", "last", "all", -300),
    ("cut the timer by 2 minutes", "last", "timer", -120),
    ("take 5 minutes off the timer", "last", "timer", -300),
    ("take five minutes off that", "last", "all", -300),
    ("knock 5 minutes off the timer", "last", "timer", -300),
    ("bring the alarm forward 15 minutes", "last", "alarm", -900),
    ("bring my alarm forward by fifteen minutes", "last", "alarm", -900),
    ("make it 5 minutes shorter", "last", "all", -300),
    ("make the timer five minutes earlier", "last", "timer", -300),
    ("move the alarm up 10 minutes", "last", "alarm", -600),
    # punctuation, address and courtesy tails; other units
    ("Extend that timer by 10 minutes.", "last", "timer", 600),
    ("extend the timer by ten minutes, jarvis", "last", "timer", 600),
    ("extend the timer by 30 seconds please", "last", "timer", 30),
    ("push the alarm back an hour", "last", "alarm", 3600),
    ("extend my timers by 5 minutes", "all", "timer", 300),
])
def test_adjust_schedule_phrasings_reach_the_timekeeper(rich, services, text,
                                                        which, kind, delta):
    tk = services.timekeeper
    tk.adjust.return_value = [_adjusted_item(kind=kind if kind != "all" else "timer")]
    assert rich._match_assistant(text) == "adjust schedule", text
    res = rich.handle(text, source="typed")
    tk.adjust.assert_called_once_with(which, kind, float(delta))
    assert res.handled and res.speak
    assert res.reply.endswith(".") and "sir" in res.reply
    assert ("added" in res.reply) == (delta > 0)
    assert ("off" in res.reply) == (delta < 0)
    assert res.status.endswith("extended" if delta > 0 else "shortened")
    assert res.undo is not None
    services.brain.chat.assert_not_called()
    services.claude.submit.assert_not_called()
    tk.add_timer.assert_not_called()
    tk.cancel.assert_not_called()


def test_adjust_schedule_by_voice_bypasses_the_intent_gate(rich, services, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the classifier was consulted")
    monkeypatch.setattr(rich.intent, "classify", boom)
    services.timekeeper.adjust.return_value = [_adjusted_item()]
    res = rich.handle("Extend that timer by 10 minutes.", source="voice")
    services.timekeeper.adjust.assert_called_once_with("last", "timer", 600.0)
    assert res.handled and res.speak and res.status == "Timer extended"
    services.brain.chat.assert_not_called()


@pytest.mark.parametrize("text", [
    "set a timer for 10 minutes",
    "extend the deadline by a week",
    "add milk to the shopping list",
    "10 more minutes",
    "give me 10 more minutes",
    "remind me in 10 minutes to extend the lease",
    "how long is left on the timer",
    "did they extend the timer",
])
def test_adjust_schedule_leaves_other_utterances_alone(rich, services, text):
    assert rich._match_assistant(text) != "adjust schedule", text
    rich.handle(text, source="typed")
    services.timekeeper.adjust.assert_not_called()


def test_the_timer_route_still_wins_for_setting_one(rich, services):
    res = rich.handle("set a timer for 10 minutes", source="typed")
    services.timekeeper.add_timer.assert_called_once_with(600, "10 minutes timer")
    services.timekeeper.adjust.assert_not_called()
    assert res.reply.startswith("10 minutes, sir")
    names = [c.name for c in REGISTRY]
    assert names.index("timer") < names.index("adjust schedule")
    assert names.index("cancel schedule") + 1 == names.index("adjust schedule")


def test_adjust_schedule_with_nothing_to_move_says_so(rich, services):
    services.timekeeper.adjust.return_value = []
    res = rich.handle("extend the timer by ten minutes", source="typed")
    assert res.handled and res.speak
    assert res.reply == "No timer to extend, sir." and res.status == "Nothing to adjust"
    assert res.undo is None
    res = rich.handle("bring my alarm forward 5 minutes", source="typed")
    assert res.reply == "No alarm to bring forward, sir."
    res = rich.handle("shorten it by five minutes", source="typed")
    assert res.reply == "Nothing to shorten, sir."
    services.brain.chat.assert_not_called()


def test_adjust_schedule_undo_moves_it_back_by_id(rich, services):
    tk = services.timekeeper
    tk.adjust.return_value = [_adjusted_item(id="tm-9", kind="timer")]
    res = rich.handle("extend that timer by 10 minutes", source="typed")
    assert res.undo is not None
    tk.adjust.reset_mock()
    tk.adjust.return_value = [_adjusted_item(id="tm-9")]
    res = rich.handle("scratch that", source="typed")
    tk.adjust.assert_called_once_with("tm-9", "timer", -600)
    assert res.reply == "Back to where it was, sir." and res.status == "Undone"


def test_adjust_schedule_speaks_the_new_due(rich, services):
    """A MagicMock's _describe_item is not a string, so the handler falls
    back to the label plus describe_due -- the reply still carries the due."""
    services.timekeeper.adjust.return_value = [_adjusted_item(kind="alarm", label="Gym")]
    services.timekeeper.describe_due.return_value = "at 7:10 am tomorrow"
    res = rich.handle("push my alarm back 10 minutes", source="typed")
    assert res.reply == "Ten minutes added, sir: Gym at 7:10 am tomorrow."
    assert res.status == "Alarm extended"


# ------------------------------------------------------ the incident, replayed
class _Clock:
    def __init__(self, t):
        self.t = t

    def now(self):
        return self.t


@pytest.fixture
def real_tk(rich, services, tmp_path):
    """The rich commander with a REAL Timekeeper on a tmp db: fake clock,
    no ringer, no notifications, no subprocesses (run= records)."""
    from jarvis.tools.timekeeper import Timekeeper
    clock = _Clock(datetime(2026, 9, 1, 20, 40, 54).timestamp())
    runs = []
    tk = Timekeeper(tmp_path / "tk.db", say=lambda *a, **k: None,
                    cfg=services.assistant, now=clock.now,
                    run=lambda argv, **k: runs.append(list(argv)),
                    tick_s=0.01, ring=False, cache_dir=tmp_path / "cache",
                    notify=False)
    services.timekeeper = tk
    yield rich, tk, clock
    tk.close()


def test_incident_replay_extend_that_timer_by_ten_minutes(real_tk, services, monkeypatch):
    c, tk, clock = real_tk

    def boom(*a, **k):
        raise AssertionError("the classifier was consulted")
    monkeypatch.setattr(c.intent, "classify", boom)

    res = c.handle("Set a timer for 10 minutes to pack up the chicken.", source="voice")
    assert res.handled and res.speak
    assert res.reply == "10 minutes to pack up the chicken, sir; I'll let you know."
    items = tk.list("timer")
    assert len(items) == 1 and items[0].label == "pack up the chicken"
    assert items[0].due == pytest.approx(clock.now() + 600)

    clock.t += 55                                            # 20:41:49
    res = c.handle("Extend that timer by 10 minutes.", source="voice")
    assert res.handled and res.speak
    assert res.status == "Timer extended"
    assert res.reply == "Ten minutes added, sir: pack up the chicken in 19 minutes."
    assert re.search(r"\b(19|nineteen)\b", res.reply)
    services.brain.chat.assert_not_called()
    services.claude.submit.assert_not_called()
    item = tk.list("timer")[0]
    assert item.due == pytest.approx(item.created + 1200, abs=2)
    assert item.state == "pending"

    # ...and "scratch that" puts it back where it was
    res = c.handle("scratch that", source="voice")
    assert res.reply == "Back to where it was, sir."
    item = tk.list("timer")[0]
    assert item.due == pytest.approx(item.created + 600, abs=2)


def test_incident_replay_add_minutes_to_a_named_timer(real_tk, services):
    c, tk, clock = real_tk
    c.handle("set a timer for 5 minutes for the tea", source="typed")
    c.handle("set a timer for 10 minutes to pack up the chicken", source="typed")
    res = c.handle("add 5 minutes to the timer for the tea", source="typed")
    assert res.reply == "Five minutes added, sir: the tea in 10 minutes."
    by_label = {i.label: i for i in tk.list("timer")}
    assert by_label["the tea"].due == pytest.approx(clock.now() + 600)
    assert by_label["pack up the chicken"].due == pytest.approx(clock.now() + 600)
    res = c.handle("take 4 minutes off the timer for the chicken", source="typed")
    assert res.reply == "Four minutes off, sir: pack up the chicken in 6 minutes."


# =========================================================================
# Timer label grammar (live 2026-09-01 20:40 and 21:03)
# =========================================================================
# "Set a timer for 8 minutes to put chicken away." was answered "8 minutes
# for the put chicken away, sir": the label was always framed as a thing.
@pytest.mark.parametrize("text,seconds,label,spoken", [
    ("Set a timer for 8 minutes to put chicken away.", 480, "put chicken away",
     "8 minutes to put chicken away, sir; I'll let you know."),
    ("Set a timer for 10 minutes to pack up the chicken.", 600, "pack up the chicken",
     "10 minutes to pack up the chicken, sir; I'll let you know."),
    ("set a timer for 5 minutes for the tea", 300, "the tea",
     "5 minutes for the tea, sir; I'll let you know."),
    ("5 minute timer called laundry", 300, "laundry",
     "5 minutes for the laundry, sir; I'll let you know."),
    ("timer for 3 minutes to flip the steak", 180, "flip the steak",
     "3 minutes to flip the steak, sir; I'll let you know."),
    # the connector was heard as "for" but the label is plainly an action
    ("set a timer for 2 minutes for stir the sauce", 120, "stir the sauce",
     "2 minutes to stir the sauce, sir; I'll let you know."),
])
def test_timer_label_grammar(rich, services, text, seconds, label, spoken):
    res = rich.handle(text, source="typed")
    services.timekeeper.add_timer.assert_called_once_with(seconds, label)
    assert res.reply == spoken and res.speak


def test_timer_label_grammar_in_the_shaky_read_back(rich, services, monkeypatch):
    monkeypatch.setattr(rich, "shaky_transcript", lambda: True)
    res = rich.handle("Set a timer for 8 minutes to put chicken away.", source="typed")
    assert res.reply == "An 8-minute timer to put chicken away, sir?"
    assert res.status == "Confirm?"
    services.timekeeper.add_timer.assert_not_called()
    res = rich.handle("yes", source="typed")
    services.timekeeper.add_timer.assert_called_once_with(480, "put chicken away")
    assert res.reply == "8 minutes to put chicken away, sir; I'll let you know."
    services.timekeeper.add_timer.reset_mock()
    res = rich.handle("set a timer for 5 minutes for the tea", source="typed")
    assert res.reply == "A 5-minute timer for the tea, sir?"


def test_timer_label_phrase_helper():
    from jarvis.commander import timer_label_phrase
    assert timer_label_phrase("to", "put chicken away") == "to put chicken away"
    assert timer_label_phrase("for", "the tea") == "for the tea"
    assert timer_label_phrase("for", "tea") == "for the tea"
    assert timer_label_phrase("called", "laundry") == "for the laundry"
    assert timer_label_phrase("called", "make tea") == "to make tea"
    assert timer_label_phrase("for", "my eggs") == "for my eggs"
    assert timer_label_phrase("to", "") == ""


# =========================================================================
# "Belay that last order" (live 2026-09-01 21:10:23)
# =========================================================================
# Whisper wrote "BELAY THAT LAST Uhhh... ORDER". undo_kind took "belay that"
# only with an empty tail, so the object noun and the filler sent it to the
# intent classifier (Uncertain, 0.50), "Was that for me?", yes -- and the
# MODEL answered "Understood, sir; I'll stand down." and did nothing. He had
# to say "JARVIS, CANCEL TIMER!".
def test_strip_inline_fillers():
    from jarvis.commander import strip_inline_fillers as strip_fillers
    assert strip_fillers("BELAY THAT LAST Uhhh... ORDER") == "BELAY THAT LAST ORDER"
    assert strip_fillers("scratch, um, that") == "scratch that"
    assert strip_fillers("undo that… please") == "undo that please"
    assert strip_fillers("erm belay that hmm") == "belay that"
    assert strip_fillers("the umbrella is here") == "the umbrella is here"   # not a filler
    assert strip_fillers("") == ""


@pytest.mark.parametrize("text", [
    "BELAY THAT LAST Uhhh... ORDER",
    "belay that last order",
    "scratch that last command",
    "cancel that last one",
    "belay that order",
    "undo my last request",
    "belay the last instruction",
    "belay that last order, jarvis",
    "um, belay that last order please",
    # the phrases that already worked
    "scratch that", "undo that", "undo", "no, scratch that", "scratch that last one",
    "take that back", "belay that", "actually, scratch that", "undo the last one",
])
def test_undo_kind_accepts_the_object_nouns_and_fillers(text):
    assert commander.undo_kind(text), text


@pytest.mark.parametrize("text", [
    "did they belay that climb",
    "belay that climb",
    "cancel the timer",
    "scratch my head",
    "undo the last commit",
    "never mind",
    "scratch buy milk off the list",
    "cancel that meeting",
    "",
])
def test_undo_kind_still_leaves_the_rest_alone(text):
    assert not commander.undo_kind(text), text


def test_undo_explicit_names_its_object():
    assert commander.undo_explicit("belay that last order")
    assert commander.undo_explicit("scratch that last command")
    assert commander.undo_explicit("undo my last request")
    assert not commander.undo_explicit("scratch that")
    assert not commander.undo_explicit("undo")
    assert not commander.undo_explicit("cancel the timer")
    # "cancel"/"forget" are not pure undo words: they are still HEARD as an
    # undo ("cancel that last one"), but they never short-circuit the model
    # when there is nothing to take back -- he may well mean the schedule.
    assert commander.undo_kind("cancel that last one")
    assert not commander.undo_explicit("cancel that last one")
    assert not commander.undo_explicit("forget the last one")


def test_belay_that_last_order_by_voice_never_reaches_the_classifier(rich, services,
                                                                     monkeypatch):
    services.timekeeper.add_timer.return_value = types.SimpleNamespace(id="tm-1")
    rich.handle("set a timer for 8 minutes to put chicken away", source="voice")

    def boom(*a, **k):
        raise AssertionError("the classifier was consulted")
    monkeypatch.setattr(rich.intent, "classify", boom)
    res = rich.handle("BELAY THAT LAST Uhhh... ORDER", source="voice")
    services.timekeeper.cancel.assert_called_once_with(which="tm-1", kind="timer")
    assert res.reply == "Timer scrapped, sir." and res.speak and res.status == "Undone"
    services.brain.chat.assert_not_called()


def test_an_explicit_undo_outlives_the_bare_window(rich, services, monkeypatch):
    """Set at 21:03:18, belayed at 21:10:23: seven minutes. A bare "scratch
    that" that old is a stray transcript; "belay that last order" is not."""
    import time as _time
    services.timekeeper.add_timer.return_value = types.SimpleNamespace(id="tm-1")
    rich.handle("set a timer for 8 minutes to put chicken away", source="voice")
    undo, at = rich._last_undo
    rich._last_undo = (undo, at - 7 * 60)                    # seven minutes ago
    res = rich.handle("BELAY THAT LAST Uhhh... ORDER", source="voice")
    services.timekeeper.cancel.assert_called_once_with(which="tm-1", kind="timer")
    assert res.reply == "Timer scrapped, sir."
    # ...but not for ever, and a bare "scratch that" keeps the short window
    services.timekeeper.cancel.reset_mock()
    rich.handle("set a timer for 8 minutes to put chicken away", source="voice")
    undo, at = rich._last_undo
    rich._last_undo = (undo, at - 7 * 60)
    res = rich.handle("scratch that", source="voice")
    services.timekeeper.cancel.assert_not_called()
    assert _time.monotonic() > 0 and res.status != "Undone"
    rich._last_undo = (undo, at - commander.UNDO_EXPLICIT_WINDOW_S - 5)
    res = rich.handle("belay that last order", source="voice")
    services.timekeeper.cancel.assert_not_called()
    assert res.reply == "Nothing to take back, sir." and res.speak


def test_an_explicit_undo_with_nothing_to_undo_is_answered_not_handed_on(rich, services,
                                                                          monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the classifier was consulted")
    monkeypatch.setattr(rich.intent, "classify", boom)
    res = rich.handle("belay that last order", source="voice")
    assert res.handled and res.reply == "Nothing to take back, sir." and res.speak
    services.brain.chat.assert_not_called()
    services.claude.submit.assert_not_called()


def test_incident_replay_belay_that_last_order(real_tk, services, monkeypatch):
    c, tk, clock = real_tk

    def boom(*a, **k):
        raise AssertionError("the classifier was consulted")
    monkeypatch.setattr(c.intent, "classify", boom)
    res = c.handle("Set a timer for 8 minutes to put chicken away.", source="voice")
    assert res.reply == "8 minutes to put chicken away, sir; I'll let you know."
    assert len(tk.list("timer")) == 1
    res = c.handle("BELAY THAT LAST Uhhh... ORDER", source="voice")
    assert res.handled and res.reply == "Timer scrapped, sir." and res.speak
    assert tk.list("timer") == []
    assert tk.list(include_done=True)[0].state == "cancelled"
    services.brain.chat.assert_not_called()
    # a "belay" inside a sentence about something else is not an undo
    res = c.handle("did they belay that climb", source="typed")
    assert res.status != "Undone"


# =========================================================================
# Review pass 2026-09-02: the holes the first cut of the route left
# =========================================================================
@pytest.mark.parametrize("text,which,kind,delta", [
    # the label BEFORE the kind word -- as natural in speech as "the timer
    # for the chicken", and it went to the model, the path that failed
    ("extend the chicken timer by five minutes", "chicken", "timer", 300),
    ("shorten the tea timer by 2 minutes", "tea", "timer", -120),
    ("give me 5 more minutes on the tea timer", "tea", "timer", 300),
    ("10 more minutes on the chicken timer", "chicken", "timer", 600),
    ("put 5 minutes on the chicken timer", "chicken", "timer", 300),
    ("take 4 minutes off the chicken timer", "chicken", "timer", -240),
    ("push my 7 am alarm back 10 minutes", "7 am", "alarm", 600),
    ("give the pack up the chicken timer another 5 minutes",
     "pack up the chicken", "timer", 300),
    # "all my timers" is the plural cancel already understands
    ("extend all my timers by 5 minutes", "all", "timer", 300),
    ("push back all of my alarms by ten minutes", "all", "alarm", 600),
    # the unit left off after a verb that can mean nothing else
    ("extend the timer by 5", "last", "timer", 300),
    ("extend that by five", "last", "all", 300),
    ("shorten the timer by 2", "last", "timer", -120),
])
def test_adjust_schedule_second_pass_phrasings(rich, services, text, which, kind, delta):
    tk = services.timekeeper
    tk.adjust.return_value = [_adjusted_item(kind=kind if kind != "all" else "timer")]
    assert rich._match_assistant(text) == "adjust schedule", text
    res = rich.handle(text, source="typed")
    tk.adjust.assert_called_once_with(which, kind, float(delta))
    assert res.handled and res.speak
    services.brain.chat.assert_not_called()


@pytest.mark.parametrize("text", [
    # "bump it up five minutes" is as often "five MORE" as it is "sooner":
    # the confident opposite is worse than no Tier-1 match
    "bump that timer up 5 minutes",
    "move the timer up 10 minutes",
    "bump it up five minutes",
    # a pronoun with a verb that is not a schedule verb, nothing scheduled
    "give it another 5 minutes",
    "add 5 minutes to that",
    "make that 5 minutes longer",
    "put 30 seconds on it",
    "10 more minutes on it",
])
def test_adjust_schedule_leaves_the_ambiguous_ones_to_the_model(rich, services, text):
    services.timekeeper.adjust.return_value = []
    res = rich.handle(text, source="typed")
    assert res is None or res.status != "Nothing to adjust", text
    assert not (res and res.status and res.status.endswith(("extended", "shortened")))


def test_move_the_alarm_up_is_still_the_calendar_idiom(rich, services):
    services.timekeeper.adjust.return_value = [_adjusted_item(kind="alarm")]
    rich.handle("move the alarm up 10 minutes", source="typed")
    services.timekeeper.adjust.assert_called_once_with("last", "alarm", -600.0)


def test_an_unambiguous_verb_with_a_pronoun_still_answers_for_the_schedule(rich, services):
    """"Extend that by five" names no object but can mean nothing else."""
    services.timekeeper.adjust.return_value = []
    res = rich.handle("extend that by five minutes", source="typed")
    assert res.handled and res.reply == "Nothing to extend, sir."
    res = rich.handle("shorten it by five minutes", source="typed")
    assert res.reply == "Nothing to shorten, sir."


def test_adjust_undo_restores_the_snapshot_when_there_is_one(rich, services):
    """A shorten clamped to now cannot be undone by adding the delta back."""
    tk = services.timekeeper
    prev = (1_800_000_000.0, None, "pending")
    tk.adjust.return_value = [_adjusted_item(id="tm-9", previous=prev, applied=-120.0)]
    tk.restore.return_value = True
    res = rich.handle("take 5 minutes off the timer", source="typed")
    assert res.undo is not None
    tk.adjust.reset_mock()
    res = rich.handle("scratch that", source="typed")
    tk.restore.assert_called_once_with("tm-9", prev)
    tk.adjust.assert_not_called()
    assert res.reply == "Back to where it was, sir."


def test_adjust_undo_still_falls_back_to_the_opposite_delta(rich, services):
    """No snapshot (an older timekeeper, or a ring that cannot be un-killed):
    the opposite delta is still the best there is."""
    tk = services.timekeeper
    tk.adjust.return_value = [_adjusted_item(id="tm-9")]
    rich.handle("extend that timer by 10 minutes", source="typed")
    tk.adjust.reset_mock()
    tk.restore.reset_mock()
    tk.adjust.return_value = [_adjusted_item(id="tm-9")]
    res = rich.handle("scratch that", source="typed")
    tk.restore.assert_not_called()
    tk.adjust.assert_called_once_with("tm-9", "timer", -600)
    assert res.reply == "Back to where it was, sir."


@pytest.mark.parametrize("text", [
    "cancel that one", "cancel that request", "cancel that step",
    "cancel that action", "forget that thing",
])
def test_cancel_that_one_is_not_an_undo(rich, services, text):
    """d38b493 sent these to the model (manage_schedule cancel); the widened
    undo grammar turned "cancel that one" -- a natural way to pick an item
    out of a read-out -- into a rewind of the last action."""
    assert not commander.undo_kind(text), text
    services.timekeeper.add_timer.return_value = types.SimpleNamespace(
        id="tm-1", label="the tea", due=1_800_000_000.0, duration=600)
    rich.handle("set a timer for 10 minutes for the tea", source="typed")
    res = rich.handle(text, source="typed")
    assert res is None or res.status != "Undone", text


def test_a_weak_undo_with_nothing_to_take_back_reaches_the_model(rich, services):
    """"Belay that last order" with nothing on the books is still addressed
    to Jarvis and is answered; "cancel the last one" is not -- he may mean
    the schedule, and the model can still cancel it."""
    res = rich.handle("cancel that last one", source="typed")
    assert res is None or res.status != "Nothing to undo"
    res = rich.handle("belay that last order", source="voice")
    assert res.handled and res.reply == "Nothing to take back, sir."


def test_incident_replay_undo_of_a_clamped_shorten(real_tk, services):
    """Real wiring: take five minutes off a two-minute timer and "scratch
    that" claimed "back to where it was" while leaving it five minutes out."""
    c, tk, clock = real_tk
    c.handle("set a timer for 2 minutes for the eggs", source="typed")
    created = tk.list("timer")[0].created
    res = c.handle("take 5 minutes off the timer", source="typed")
    assert res.reply == "Two minutes off, sir: the eggs now."   # not "five minutes off"
    assert tk.list("timer")[0].due == pytest.approx(clock.now())
    res = c.handle("scratch that", source="typed")
    assert res.reply == "Back to where it was, sir."
    assert tk.list("timer")[0].due == pytest.approx(created + 120)


def test_incident_replay_extend_a_timer_named_before_the_kind_word(real_tk, services,
                                                                   monkeypatch):
    c, tk, clock = real_tk

    def boom(*a, **k):
        raise AssertionError("the classifier was consulted")
    monkeypatch.setattr(c.intent, "classify", boom)
    c.handle("set a timer for 30 minutes to pack up the chicken", source="voice")
    c.handle("set a timer for 5 minutes for the tea", source="voice")
    res = c.handle("extend the chicken timer by five minutes", source="voice")
    assert res.status == "Timer extended"
    assert res.reply == "Five minutes added, sir: pack up the chicken in 35 minutes."
    res = c.handle("shorten the tea timer by 2 minutes", source="voice")
    assert res.reply == "Two minutes off, sir: the tea in 3 minutes."
    services.brain.chat.assert_not_called()


def test_incident_replay_push_back_then_forward_leaves_the_series_alone(real_tk, services):
    """2026-09-02 review: "push my alarm back thirty... no, bring it forward
    thirty" left the wake-up reading "at 7:00 am tomorrow, then back to
    7:00 am" -- self-contradictory, and ", every day" was gone from every
    listing and briefing until it next rang."""
    c, tk, clock = real_tk
    tk.add_alarm(datetime(2026, 9, 2, 7, 0).timestamp(), "wake up", "daily")
    res = c.handle("push my alarm back 30 minutes", source="voice")
    assert res.reply == ("30 minutes added, sir: wake up at 7:30 am tomorrow, "
                         "then back to 7:00 am.")
    res = c.handle("bring my alarm forward 30 minutes", source="voice")
    assert res.reply == "30 minutes off, sir: wake up at 7:00 am tomorrow, every day."
    it = tk.list("alarm")[0]
    assert it.snooze_until is None and not it.shifted
    assert tk.list_text("alarm") == "One alarm, sir: wake up at 7:00 am tomorrow, every day."


# ------------------------------------------------ grab and throw by voice
class FakeCourier:
    """jarvis/gesturecast.GestureCast's voice face, recording every call."""

    def __init__(self):
        self.throws, self.drops, self.taught, self.sides = [], 0, [], []
        self.holding, self.spoken_over_calls = 0, 0

    def throw_by_voice(self, sink):
        self.throws.append(sink)
        return "On the board, sir.", "landed"

    def drop_by_voice(self):
        self.drops += 1
        return "Put down, sir."

    def holding_line(self):
        self.holding += 1
        return "Holding the thesis draft, sir."

    def teach(self, side, sink):
        self.taught.append((side, sink))
        return "Right is HPCOMPUTER from now on, sir."

    def side_line(self, sink):
        self.sides.append(sink)
        return "HPCOMPUTER is on your right, sir."

    def spoken_over(self):
        self.spoken_over_calls += 1


@pytest.fixture
def courier(rich, services):
    c = FakeCourier()
    services.gesture = c
    return c


def test_the_cast_verbs_reach_the_courier(rich, courier):
    res = rich.handle("throw this on hpcomputer", source="voice")
    assert res.handled and res.speak and res.reply == "On the board, sir."
    assert courier.throws == ["hpcomputer"]
    res = rich.handle("put it on the board", source="voice")
    assert res.reply == "On the board, sir." and courier.throws[-1] == "the board"
    res = rich.handle("drop it", source="voice")
    assert res.reply == "Put down, sir." and courier.drops == 1
    res = rich.handle("what am I holding", source="voice")
    assert res.reply == "Holding the thesis draft, sir." and courier.holding == 1
    res = rich.handle("HPCOMPUTER is on my right", source="voice")
    assert res.reply.startswith("Right is HPCOMPUTER") and courier.taught == [("right", "hpcomputer")]
    res = rich.handle("which side is hpcomputer on", source="voice")
    assert res.reply == "HPCOMPUTER is on your right, sir." and courier.sides == ["hpcomputer"]


def test_without_a_courier_the_cast_verbs_do_not_claim_the_turn(rich, services):
    services.gesture = None
    assert rich._match_assistant("drop it") == "cast drop"
    res = rich._try_assistant("drop it")
    assert res is None


def test_the_cast_verbs_do_not_collide_with_the_list_and_board_families(rich, courier):
    """The words overlap half the registry: the object is pinned to
    this/it/that and the target to the sink table, in both directions."""
    for phrase, name in (
            ("put milk on the shopping list", "list add"),
            ("throw milk on the shopping list", "list add"),
            ("cross the second one off the list", "list strike anon"),
            ("drop the board", "board hide"),
            ("put the board down", "board hide"),
            ("put up the board", "board show"),
            ("drop the timer", "cancel schedule"),
            ("throw this on hpcomputer", "cast throw"),
            ("send this to the pc", "cast throw"),
            ("put that on the board", "cast put"),
            ("put it down", "cast drop"),
            ("let go", "cast drop"),
            ("hpcomputer is on my left", "cast teach"),
            ("the right is the board", "cast teach")):
        assert rich._match_assistant(phrase) == name, phrase
    # "the desktop" is a folder on this box (the file lane's ruling), so it
    # must never resolve to HPCOMPUTER as a cast sink.
    assert not str(rich._match_assistant("send this to the desktop") or "").startswith("cast")
    assert not str(rich._match_assistant("put this on my desktop") or "").startswith("cast")
    for phrase in ("throw a party for my sister", "send this to mom",
                   "right is fine", "what is left", "drop everything",
                   "put it on my calendar", "throw it away",
                   "cast a wide net", "let it be"):
        got = rich._match_assistant(phrase)
        assert not (got or "").startswith("cast"), (phrase, got)
    assert courier.throws == []


def test_a_sentence_puts_a_live_carry_down_but_a_cast_verb_does_not(rich, courier):
    rich.handle("what time is it", source="voice")
    assert courier.spoken_over_calls == 1
    rich.handle("drop it", source="voice")
    rich.handle("throw this on the board", source="voice")
    rich.handle("hpcomputer is on my right", source="voice")
    rich.handle("what am i holding", source="voice")
    assert courier.spoken_over_calls == 1
    rich.handle("bring up the board", source="typed")
    assert courier.spoken_over_calls == 2


def test_a_slim_commander_with_no_services_survives_the_carry_hook():
    c = object.__new__(Commander)
    c._cast_spoken_over("anything at all")       # no services: a no-op
