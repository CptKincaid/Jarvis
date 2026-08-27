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
    assert value == "i parked on level 3"
    assert res.handled and "i parked on level 3" in res.reply
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
    services.brain.chat.assert_called_once_with("what's the weather")
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
    services.brain.chat.assert_called_once_with("what's on my calendar tomorrow")
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
    rich.handle("add milk to my shopping list", source="typed")
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
    services.brain.chat.assert_called_once_with("what's the weather")


def test_voice_still_gates_background_chat(rich, services):
    res = rich.handle("she said no way lol haha dude", source="voice")
    assert "Ignored" in res.status
    services.brain.chat.assert_not_called()


def test_dictation_still_beats_every_assistant_stage(rich, services,
                                                     monkeypatch):
    typed = []
    monkeypatch.setattr(rich, "_type_raw", lambda t: typed.append(t))
    services.timekeeper.ringing = types.SimpleNamespace(id="a1", label="Get up")
    services.approvals.pending.return_value = [object()]
    rich.handle("jarvis dictate")
    rich.handle("stop")
    assert typed == ["stop "]
    services.timekeeper.stop_ringing.assert_not_called()
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
    services.brain.chat.assert_called_once_with("what's the weather")


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
