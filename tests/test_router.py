"""Tests for jarvis.router — the local-vs-Claude decision (spec 5.1).

Pure: no model, no network, no services. The 60-utterance table (20 local,
20 claude, 10 actions, 10 ambiguous) must route as labelled WITHOUT the
model; the ambiguous ten reach the classify stub and, at low confidence,
produce exactly one question. Firewall: the router never touches config
files or /tmp/vss_voice (conftest sets JARVIS_LOG_DIR; the assistant
config env is pointed at a throwaway path here too).
"""
import os
import tempfile

os.environ.setdefault(
    "JARVIS_ASSISTANT_CONFIG",
    os.path.join(tempfile.mkdtemp(prefix="jarvis-router-"), "assistant.json"))

import pytest  # noqa: E402

from jarvis.router import (  # noqa: E402
    ASK_TTL_S,
    DEFAULT_SKILL_PHRASES,
    ROUTER_QUESTION,
    RouteDecision,
    Router,
    answer_kind,
    estimate_size,
    normalise,
    slugify,
)


class FakeCfg:
    """The slice of AssistantConfig the router reads."""

    def __init__(self, phrases=None):
        self.data = {"claude.skill_phrases": phrases if phrases is not None
                     else dict(DEFAULT_SKILL_PHRASES),
                     "claude.big_model": "fable"}

    @property
    def skill_phrases(self):
        return self.data["claude.skill_phrases"]

    def get(self, key, default=None):
        return self.data.get(key, default)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def make_router(classify=lambda t: ("local", 0.2), phrases=None, clock=None):
    return Router(FakeCfg(phrases), classify=classify,
                  now=clock or Clock())


# ------------------------------------------------------------ the table
LOCAL = [
    "what's the weather like today",
    "will it rain tomorrow",
    "what time is it",
    "what's the date today",
    "what's on my calendar tomorrow",
    "do I have any meetings this afternoon",
    "remind me at 3 pm to call mum",
    "set a timer for 10 minutes",
    "wake me up at 7 tomorrow",
    "take a note that the boiler is broken",
    "add milk to my todo list",
    "check my email",
    "what's my briefing",
    "where am I",
    "good morning",
    "tell me a joke",
    "who wrote Hamlet",
    "how many ounces in a pound",
    "what can you do",
    "remind me to run the tests at 5",        # wrapper: the tail is free text
]
CLAUDE = [
    "fix the failing test in the parser",
    "refactor the router module",
    "add a unit test for the timekeeper",
    "why is the build failing",
    "run the tests",
    "commit my changes and push the branch",
    "install the caldav package",
    "write a script that renames my downloads by date",
    "explain this code",
    "have a look at jarvis/brain.py",
    "debug the traceback in the logs",
    "review this code",                        # skill phrase
    "commit this",                             # skill phrase
    "run a ralph loop on the calendar parser",  # skill phrase with $1
    "run the playwright skill on the login page",  # passthrough skill
    "have claude tidy up the imports",         # explicit cue
    "ask claude to write the docs",            # explicit cue
    "rename the class to AlarmClock",
    "merge the feature branch",
    "fix the bug in the weather tool",         # coding pair beats a topic noun
]
ACTIONS = [
    ("cancel", "cancel", {}),
    ("stop that", "cancel", {}),
    ("abort", "cancel", {}),
    ("work on the jarvis project", "work_on", {"name": "jarvis"}),
    ("switch to the haymaker project", "work_on", {"name": "haymaker"}),
    ("pick up where we left off", "resume", {"when": "", "name": ""}),
    ("what were we working on yesterday", "resume", {"when": "yesterday", "name": ""}),
    ("continue the vss project", "resume", {"when": "", "name": "vss"}),
    ("start a new project called weather station", "new_project",
     {"name": "weather station"}),
    ("use fable", "set_model", {"alias": "fable"}),
]
AMBIGUOUS = [
    "sort out the thing we talked about",
    "can you take care of the downloads folder situation",
    "I need the numbers from last week put together somewhere",
    "handle the mess in my home directory",
    "get everything ready for the demo tomorrow",
    "put together a summary of what happened this week",
    "look into the noise the fans are making",
    "clean things up a bit around here",
    "the tests are slow",                         # bare coding noun: weak cue
    "should I use focal loss for the rare classes?",
]


def test_table_has_sixty_rows():
    assert len(LOCAL) == 20 and len(CLAUDE) == 20
    assert len(ACTIONS) == 10 and len(AMBIGUOUS) == 10


@pytest.mark.parametrize("text", LOCAL)
def test_local_utterances_route_local_without_the_model(text):
    calls = []

    def classify(t):
        calls.append(t)
        return ("claude", 0.99)          # would mislead if it were consulted

    d = make_router(classify).route(text, "jarvis")
    assert d.kind == "local", (text, d)
    assert calls == []


@pytest.mark.parametrize("text", CLAUDE)
def test_claude_utterances_route_claude_without_the_model(text):
    calls = []

    def classify(t):
        calls.append(t)
        return ("local", 0.99)

    d = make_router(classify).route(text, "jarvis")
    assert d.kind == "claude", (text, d)
    assert d.prompt, text
    assert calls == []


@pytest.mark.parametrize("text,action,args", ACTIONS)
def test_actions_route_to_the_session_manager(text, action, args):
    d = make_router().route(text, "jarvis")
    assert d.kind == "action" and d.action == action, (text, d)
    for k, v in args.items():
        assert d.args.get(k) == v, (text, k, d.args)


@pytest.mark.parametrize("text", AMBIGUOUS)
def test_ambiguous_utterances_ask_exactly_one_question(text):
    calls = []

    def classify(t):
        calls.append(t)
        return ("local", 0.2)

    r = make_router(classify)
    d = r.route(text, "jarvis")
    assert d.kind == "ask", (text, d)
    assert len(calls) == 1
    assert r.pending() is not None and r.pending().text == d.prompt
    # asking again while pending does not stack questions
    assert r.pending() is r.pending()


# ------------------------------------------------------------ rule 6/7
def test_confident_classifier_routes_silently():
    # The local direction is free to be wrong: one cheap Ollama turn.
    r = make_router(lambda t: ("local", 0.8))
    d = r.route("sort out the thing we talked about", None)
    assert d.kind == "local" and d.reason == "classify"
    # The claude direction bills the user's account, so it takes the higher
    # bar (fixed 2026-08-26, finding C1): 0.9 asks, 0.95 dispatches.
    r = make_router(lambda t: ("claude", 0.9))
    d = r.route("sort out the thing we talked about", None)
    assert d.kind == "ask" and d.reason == "classify-claude-unconfirmed"
    assert r.pending() is not None
    r = make_router(lambda t: ("claude", 0.95))
    d = r.route("sort out the thing we talked about", None)
    assert d.kind == "claude" and d.reason == "classify"
    assert d.confidence == 0.95 and r.pending() is None


def test_a_confident_claude_still_needs_a_coding_cue():
    """Rule 6 may not spend money on a sentence with no coding cue in it at
    all, however sure the classifier says it is (finding C1)."""
    r = make_router(lambda t: ("claude", 1.0))
    d = r.route("an apple a day keeps the doctor away or so they say", None)
    assert d.kind == "ask", (d.kind, d.reason)
    # the control: "sort out" IS a coding verb, so this one may dispatch
    d = r.route("sort out the thing we talked about", None)
    assert d.kind == "claude", (d.kind, d.reason)


# The household / idiom corpus behind finding C1: ordinary English, each
# paired with the verdict the REAL gemma4:26b classifier returned for it on
# 2026-08-26 (an 80-utterance labelled measurement).  Six of these used to
# dispatch a billed CLI task with no confirmation.  None may reach the paid
# CLI, whatever the classifier says.
ORDINARY_ENGLISH = [
    ("put on some jazz from the seventies please", "claude", 0.90),
    ("my code of conduct is to always be kind", "claude", 0.80),
    ("we should test the waters before committing to it", "claude", 0.85),
    ("could you play something a bit more upbeat please", "claude", 0.80),
    ("build me up a shopping list for sunday lunch", "claude", 0.90),
    ("the only real error is not learning from the last one", "claude", 0.85),
    ("we need to patch things up with the neighbours", "claude", 0.60),
    ("the kids want to build a den in the garden", "claude", 0.70),
    ("I think I will branch out and try a new restaurant tonight", "local", 0.90),
    ("let us not commit to anything until we have slept on it", "local", 0.90),
    ("the test of a good friend is who calls when things go wrong", "local", 0.90),
    ("the dress code for the wedding is black tie apparently", "local", 0.95),
    ("run me a bath in twenty minutes would you", "local", 0.95),
    ("we always split the bill when we go out", "local", 0.90),
    ("his memory is going but his humour is intact", "local", 0.90),
]
# The other half of the same measurement: genuine coding work that the
# classifier decides (the rule ladder never sees a cue it recognises).
# These must still reach Claude without a question.
CODING_TIE_BREAKS = [
    ("write a unit test for the timekeeper snooze path", 1.0),
    ("profile the startup path and tell me where the time goes", 0.95),
    ("the linter is complaining about line length everywhere", 1.0),
    ("port the old monolith intent table over to the new commander", 1.0),
    ("set up a github action that runs pytest on every push", 1.0),
    ("bump the ruff version and fix whatever it complains about", 1.0),
    ("trace where the status strip gets its project name from", 1.0),
]


@pytest.mark.parametrize("text,route,conf", ORDINARY_ENGLISH)
def test_ordinary_english_never_dispatches_to_the_paid_cli(text, route, conf):
    r = make_router(lambda t: (route, conf))
    d = r.route(text, "jarvis")
    assert d.kind != "claude", (text, d.kind, d.reason, d.confidence)


@pytest.mark.parametrize("text,conf", CODING_TIE_BREAKS)
def test_real_coding_work_still_reaches_claude_without_a_question(text, conf):
    r = make_router(lambda t: ("claude", conf))
    d = r.route(text, "jarvis")
    assert d.kind == "claude", (text, d.kind, d.reason, d.confidence)


def test_media_requests_are_local_without_the_model():
    """Music is the one local skill whose vocabulary is ordinary English,
    so it needs cues of its own (finding C1 b)."""
    calls = []
    r = make_router(lambda t: calls.append(t) or ("claude", 1.0))
    for text in ("put on some jazz from the seventies please",
                 "could you play something a bit more upbeat please",
                 "put on the beatles while I make dinner",
                 "can you queue up something calm for the drive home",
                 "stick on some classical while I read",
                 "skip this track it is dreadful",
                 "turn the volume down a bit please"):
        d = r.route(text, "jarvis")
        assert d.kind == "local", (text, d.kind, d.reason)
    assert calls == []
    # ... without swallowing the coding sense of the same verbs
    assert r.route("play back the log", None).kind != "claude"
    assert r.route("debug the spotify tool", None).kind == "claude"


def test_the_ask_remembers_the_modifiers_it_was_given():
    """Answering "yes" must run the task the user described, on the model
    the user named (finding M1)."""
    r = make_router(lambda t: ("local", 0.1))
    d = r.route("the calendar module is broken, use haiku", None)
    assert d.kind == "ask" and d.args["model"] == "haiku"
    assert r.pending().args == {"model": "haiku"}


def test_classifier_failure_or_absence_asks():
    def boom(t):
        raise RuntimeError("ollama down")
    assert make_router(boom).route("sort out the thing we talked about",
                                   None).kind == "ask"
    assert make_router(None).route("sort out the thing we talked about",
                                   None).kind == "ask"
    # garbage answers are treated as no answer
    assert make_router(lambda t: ("maybe", 0.99)).route(
        "sort out the thing we talked about", None).kind == "ask"


def test_short_sentence_with_no_cue_is_local_without_asking():
    calls = []
    r = make_router(lambda t: calls.append(t) or ("local", 0.0))
    for text in ("make it nicer", "tidy up", "hmm", "sort the photos from the trip"):
        assert r.route(text, None).kind == "local", text
    assert calls == []


def test_claude_needs_a_cue():
    # seven words, no cue: neither table fires, the model is asked
    r = make_router(lambda t: ("local", 0.1))
    d = r.route("make the downloads folder less of a disaster please", None)
    assert d.kind == "ask"


# ------------------------------------------------------------ resolve
def test_resolve_answer_yes_and_no():
    r = make_router()
    d = r.route("sort out the thing we talked about", "jarvis")
    assert d.kind == "ask"
    assert r.resolve_answer("yes") == "claude"
    assert r.pending() is None
    r.route("sort out the thing we talked about", "jarvis")
    assert r.resolve_answer("just you") == "local"
    r.route("sort out the thing we talked about", "jarvis")
    assert r.resolve_answer("have claude do it") == "claude"
    r.route("sort out the thing we talked about", "jarvis")
    # a new subject is not an answer: the caller drops the question
    assert r.resolve_answer("have claude look at the parser instead") is None
    assert r.resolve_answer("go ahead") == "claude"
    r.route("sort out the thing we talked about", "jarvis")
    assert r.resolve_answer("quick one") == "local"


def test_resolve_answer_needs_a_pending_question():
    r = make_router()
    assert r.resolve_answer("yes") is None
    assert answer_kind("yes") == "claude" and answer_kind("no") == "local"
    assert answer_kind("Claude, please.") == "claude"
    assert answer_kind("you answer it") == "local"
    assert answer_kind("no, keep it local") == "local"
    assert answer_kind("what's the weather") is None
    assert answer_kind("fix the parser instead") is None


def test_pending_question_expires_after_ninety_seconds():
    clock = Clock(100.0)
    r = make_router(clock=clock)
    r.route("sort out the thing we talked about", "jarvis")
    clock.t += ASK_TTL_S - 1
    assert r.pending() is not None
    clock.t += 2
    assert r.pending() is None
    assert r.resolve_answer("yes") is None


def test_a_determiner_is_not_a_project_name():
    """"in my repo" means the project already open, not one called "my"."""
    r = make_router()
    for text in ("fix the parser in my repo", "fix the parser in this repo",
                 "fix the parser in that codebase",
                 "fix the parser in the current project"):
        d = r.route(text, "jarvis")
        assert d.kind == "claude" and d.project == "", (text, d)
    # a real name after the determiner is still captured
    assert r.route("fix the parser in my jarvis repo", None).project == "jarvis"


def test_pending_remembers_the_active_project():
    r = make_router()
    r.route("sort out the thing we talked about", "haymaker")
    assert r.pending().project == "haymaker"
    r.clear_pending()
    assert r.pending() is None


def test_router_question_is_the_fixed_persona_line():
    assert ROUTER_QUESTION == \
        "Shall I hand that to Claude, sir, or is it a quick one for me?"


# ------------------------------------------------------------ skills
def test_skill_phrase_expands_with_group_substitution():
    r = make_router()
    d = r.route("run a ralph loop on the calendar parser", None)
    assert d.kind == "claude" and d.reason == "skill"
    assert d.prompt == "/ralph-loop the calendar parser"
    d = r.route("plan a feature voice barge-in", None)
    assert d.prompt == "/feature-dev voice barge-in"
    for text, want in (("review my code", "/code-review"),
                       ("commit this", "/commit"),
                       ("simplify that", "/simplify"),
                       ("security review", "/security-review")):
        assert r.route(text, None).prompt == want, text


def test_skill_phrases_come_from_config_and_can_be_extended():
    r = make_router(phrases={"^ship it$": "/commit && /pr",
                             "^lint (.+)$": "/lint $1"})
    assert r.route("ship it", None).prompt == "/commit && /pr"
    assert r.route("lint the parser", None).prompt == "/lint the parser"
    # the defaults are gone when the user replaced the map
    assert r.route("review my code", None).prompt != "/code-review"
    # an invalid regex is skipped, not fatal
    r = make_router(phrases={"(": "/broken", "^ship it$": "/commit"})
    assert r.route("ship it", None).prompt == "/commit"


def test_unknown_skill_names_pass_through():
    r = make_router()
    d = r.route("run the playwright skill on the login page", None)
    assert d.kind == "claude" and d.prompt == "/playwright the login page"
    d = r.route("run the code-review skill", None)
    assert d.prompt == "/code-review"
    d = r.route("/simplify jarvis/router.py", None)
    assert d.prompt == "/simplify jarvis/router.py"


def test_explicit_cue_is_stripped_and_skill_still_applies():
    r = make_router()
    d = r.route("have claude review my code", None)
    assert d.kind == "claude" and d.prompt == "/code-review"
    d = r.route("ask claude to tidy up the imports", None)
    assert d.reason == "explicit" and d.prompt == "tidy up the imports"
    d = r.route("Jarvis, tidy up the imports with claude", None)
    assert d.kind == "claude" and d.prompt == "tidy up the imports"


# ------------------------------------------------------------ arguments
def test_model_parallel_and_project_modifiers_are_extracted():
    r = make_router()
    d = r.route("refactor the parser and the router in the jarvis project, "
                "use fable, in parallel", None)
    assert d.kind == "claude"
    assert d.prompt == "refactor the parser and the router"
    assert d.project == "jarvis"
    assert d.args["model"] == "fable" and d.args["parallel"] is True
    assert d.args["size"] == "large"
    d = r.route("this is a big one: migrate the config to toml", None)
    assert d.args["model"] == "fable" and d.prompt == "migrate the config to toml"
    d = r.route("think hard and fix the failing test", None)
    assert d.kind == "claude" and d.args["model"] == "fable"
    assert d.prompt == "fix the failing test"
    d = r.route("fix the failing test at the same time", None)
    assert d.args.get("parallel") is True


def test_model_phrase_alone_is_a_set_model_action():
    r = make_router()
    for text, alias in (("use opus", "opus"), ("use sonnet for this", "sonnet"),
                        ("use haiku", "haiku"), ("think hard", "fable"),
                        ("this is a big one", "fable")):
        d = r.route(text, None)
        assert d.kind == "action" and d.action == "set_model", text
        assert d.args == {"alias": alias}, text


def test_fast_mode_action():
    r = make_router()
    d = r.route("fast mode on", None)
    assert d.action == "set_fast_mode" and d.args == {"on": True}
    d = r.route("turn fast mode off", None)
    assert d.action == "set_fast_mode" and d.args == {"on": False}
    d = r.route("disable fast mode", None)
    assert d.args == {"on": False}


def test_resume_args():
    r = make_router()
    d = r.route("continue the haymaker digest project", None)
    assert d.action == "resume" and d.args["name"] == "haymaker digest"
    assert d.project == "haymaker-digest"
    assert d.args["utterance"] == "continue the haymaker digest project"
    d = r.route("what we were working on yesterday", None)
    assert d.action == "resume" and d.args["when"] == "yesterday"
    d = r.route("pick up where we left off on the vss project", None)
    assert d.action == "resume" and d.args["name"] == "vss"
    d = r.route("resume the jarvis project", None)
    assert d.action == "resume" and d.args["name"] == "jarvis"


def test_work_on_is_a_project_not_a_task():
    r = make_router()
    d = r.route("work on haymaker", None)
    assert d.action == "work_on" and d.args == {"name": "haymaker"}
    d = r.route("let's work on the Weather Station project", None)
    assert d.action == "work_on" and d.args["name"] == "Weather Station"
    assert d.project == "weather-station"
    # a coding object after "work on" is a task for Claude
    d = r.route("work on the login bug", None)
    assert d.kind == "claude"
    # a bare "switch to X" (no "project") is not a project switch (window target)
    assert r.route("switch to opera", None).kind != "action"


def test_new_project_variants():
    r = make_router()
    for text in ("start a new project called weather station",
                 "create a new project named weather station",
                 "new project weather station"):
        d = r.route(text, None)
        assert d.action == "new_project" and d.args == {"name": "weather station"}, text
        assert d.project == "weather-station"


def test_cancel_variants_and_jarvis_prefix():
    r = make_router()
    for text in ("Jarvis, cancel.", "cancel that", "abort the task",
                 "hey jarvis stop that", "cancel claude"):
        assert r.route(text, None).action == "cancel", text


def test_normalise_and_slugify():
    assert normalise("  Hey Jarvis, Fix the Tests!  ") == "Fix the Tests"
    assert normalise("jarvis what's the weather?") == "what's the weather"
    assert slugify("Weather Station 2") == "weather-station-2"


# ------------------------------------------------------------ size
@pytest.mark.parametrize("prompt,size", [
    ("fix the failing test in the parser", "small"),
    ("run the tests", "small"),
    ("add a unit test for the timekeeper", "small"),
    ("refactor the parser and the router", "large"),
    ("refactor the router module", "large"),
    ("add a feature flag for the briefing", "large"),
    ("rename the helper across all the modules", "large"),
    ("migrate the config to toml", "large"),
    ("debug the alarm ringer and the snooze path", "large"),
    ("update the parser, the router and the tests", "large"),
    ("", "small"),
])
def test_estimate_size(prompt, size):
    assert estimate_size(prompt) == size


def test_decision_defaults():
    d = RouteDecision("local", "x")
    assert d.prompt == "" and d.project == "" and d.action == ""
    assert d.args == {} and d.confidence == 1.0


def test_empty_text_is_local():
    assert make_router().route("", None).kind == "local"
    assert make_router().route("jarvis", None).kind == "local"


# ---------------------------------------------- modifier positions (H1)
# A cross-cutting modifier ("use opus", "in parallel") is extracted and
# DELETED from the prompt.  Deleting one that was part of the sentence
# sends Claude an instruction the user never gave and, for a model alias,
# spends his credits on a model he never asked for -- so this table pins
# every position the phrase can occupy.  Each row is
# (utterance, prompt Claude receives, model, parallel); a prompt equal to
# the utterance means "no directive here, hands off".
NO_DIRECTIVE = object()          # marker: the prompt must come through whole

MODIFIER_TABLE = [
    # --- directive: the phrase stands alone (rule 1 set_model action) ---
    ("use opus", "", "opus", False),
    ("use haiku please", "", "haiku", False),
    ("use sonnet for this one", "", "sonnet", False),
    ("think hard", "", "fable", False),
    ("this is a big one", "", "fable", False),
    # --- directive: opens the utterance, delimited or joined to the task -
    ("use opus, rewrite the parser", "rewrite the parser", "opus", False),
    ("use opus; rewrite the parser", "rewrite the parser", "opus", False),
    ("use fable: refactor the router", "refactor the router", "fable", False),
    ("use sonnet and fix the failing test", "fix the failing test", "sonnet", False),
    ("use opus to rewrite the parser", "rewrite the parser", "opus", False),
    ("this is a big one: migrate the config to toml",
     "migrate the config to toml", "fable", False),
    ("think hard and fix the failing test", "fix the failing test", "fable", False),
    ("in parallel, refactor the parser", "refactor the parser", None, True),
    ("in parallel and rewrite the docs", "rewrite the docs", None, True),
    ("in parallel then rewrite the docs", "rewrite the docs", None, True),
    # --- directive: trailing, or delimited mid-sentence ------------------
    ("fix the failing test with opus", "fix the failing test", "opus", False),
    ("clean up the imports, use haiku", "clean up the imports", "haiku", False),
    ("refactor brain.py in parallel", "refactor brain.py", None, True),
    ("run the tests at the same time", "run the tests", None, True),
    ("port the handler to rust simultaneously", "port the handler to rust",
     None, True),
    ("fix the parser in parallel with the other task", "fix the parser",
     None, True),
    ("rewrite the parser, use fable, in parallel", "rewrite the parser",
     "fable", True),
    # --- NOT a directive: the alias is adjectival ------------------------
    ("use opus level reasoning and rewrite the parser", NO_DIRECTIVE, None, False),
    ("write opus-style docstrings for the parser", NO_DIRECTIVE, None, False),
    ("give the module a fable grade summary", NO_DIRECTIVE, None, False),
    ("use sonnet style comments in the parser", NO_DIRECTIVE, None, False),
    ("add haiku like brevity to the docstrings", NO_DIRECTIVE, None, False),
    ("rewrite the readme with opus level detail", NO_DIRECTIVE, None, False),
    # --- NOT a directive: "in parallel" is prepositional ------------------
    ("in parallel with the release, write the changelog", NO_DIRECTIVE, None, False),
    ("in parallel to the release, write the changelog", NO_DIRECTIVE, None, False),
    ("run the migration in parallel with the backfill", NO_DIRECTIVE, None, False),
    ("benchmark the parser in parallel as the build runs", NO_DIRECTIVE, None, False),
    # --- NOT a directive: mid-sentence / subordinate clause ---------------
    ("document how we use opus in the pipeline", NO_DIRECTIVE, None, False),
    ("rewrite the docs to use sonnet as the example model", NO_DIRECTIVE, None, False),
    ("refactor the parser so the workers run in parallel", NO_DIRECTIVE, None, False),
    ("fix the bug where tasks run at the same time", NO_DIRECTIVE, None, False),
    ("add a test that checks we use haiku for cheap tasks", NO_DIRECTIVE, None, False),
    ("update the readme where it says use opus for everything",
     NO_DIRECTIVE, None, False),
    ("tell the team to use opus for code reviews", NO_DIRECTIVE, None, False),
    ("the parser should use sonnet under the hood", NO_DIRECTIVE, None, False),
    ("look into why the workers use fable in production", NO_DIRECTIVE, None, False),
    ("make the queue run jobs concurrently when the box is idle",
     NO_DIRECTIVE, None, False),
]


@pytest.mark.parametrize("text,prompt,model,parallel", MODIFIER_TABLE)
def test_modifier_is_a_directive_only_where_one_was_meant(text, prompt, model,
                                                          parallel):
    d = Router(None, classify=None).route(text)
    if prompt is NO_DIRECTIVE:
        # Nothing may be deleted: what Claude is asked to do is byte-for-byte
        # what the user said.
        assert d.kind in ("claude", "ask"), (text, d.kind, d.reason)
        assert d.prompt == text, (text, d.prompt)
    elif prompt == "":
        assert d.kind == "action" and d.action == "set_model", (text, d)
        assert d.args == {"alias": model}, (text, d.args)
        return
    else:
        assert d.prompt == prompt, (text, d.prompt)
    assert d.args.get("model") == model, (text, d.args)
    assert bool(d.args.get("parallel")) is parallel, (text, d.args)


def test_the_modifier_table_covers_every_position():
    assert len(MODIFIER_TABLE) >= 25
    assert sum(1 for r in MODIFIER_TABLE if r[1] is NO_DIRECTIVE) >= 10
    assert sum(1 for r in MODIFIER_TABLE if r[2]) >= 10
    assert sum(1 for r in MODIFIER_TABLE if r[3]) >= 6


# ------------------------------------------- rule 1b: an open terminal
class FakeTerminals:
    """Stands in for ClaudeSessionManager.terminal_open (tmux list-clients).

    open: the set of slugs with a client attached; "" means "whatever is
    active".  Every call is recorded so the tests can prove the router
    asked about the right project and asked at most once.
    """

    def __init__(self, *open_slugs, boom=False):
        self.open = set(open_slugs)
        self.calls = []
        self.boom = boom

    def __call__(self, project=None):
        self.calls.append(project)
        if self.boom:
            raise RuntimeError("tmux is not running")
        return (project or "") in self.open


TERMINAL_UTTERANCES = [
    ("in the terminal, run the tests", "run the tests", ""),
    ("in the terminal run the tests", "run the tests", ""),
    ("In the terminal, fix the failing test", "fix the failing test", ""),
    ("run the tests in the terminal", "run the tests", ""),
    ("fix the parser in the terminal", "fix the parser", ""),
    ("in the terminal, what's the git status", "what's the git status", ""),
    ("tell it to run the tests", "run the tests", ""),
    ("tell it to stop", "stop", ""),
    ("have it fix the failing test", "fix the failing test", ""),
    ("have it rerun the suite", "rerun the suite", ""),
    ("ask it to commit that", "commit that", ""),
    ("tell the terminal to run the tests", "run the tests", ""),
    ("in the jarvis terminal, fix the parser", "fix the parser", "jarvis"),
    ("fix the parser in the haymaker terminal", "fix the parser", "haymaker"),
    ("in the terminal session, run the tests", "run the tests", ""),
]


@pytest.mark.parametrize("text,prompt,project", TERMINAL_UTTERANCES)
def test_addressing_an_open_terminal_queues_into_that_session(text, prompt,
                                                              project):
    term = FakeTerminals(project or "jarvis")
    d = Router(None, classify=None).route(text, "jarvis", terminal_open=term)
    assert d.kind == "claude" and d.reason == "terminal", (text, d)
    assert d.prompt == prompt, (text, d.prompt)
    assert d.project == project, (text, d.project)
    assert d.args.get("terminal") is True, (text, d.args)
    assert term.calls == [project or "jarvis"], (text, term.calls)


@pytest.mark.parametrize("text,prompt,project", TERMINAL_UTTERANCES)
def test_with_no_terminal_open_those_phrasings_keep_todays_behaviour(
        text, prompt, project):
    """Closed terminal == the router as it was: same decision as the call
    that never mentions terminals at all."""
    before = Router(None, classify=None).route(text, "jarvis")
    term = FakeTerminals()                       # nothing attached
    after = Router(None, classify=None).route(text, "jarvis", terminal_open=term)
    assert after == before, (text, after, before)
    assert after.reason != "terminal" and not after.args.get("terminal")
    assert term.calls, "the router never asked whether a terminal was open"


def test_a_missing_or_broken_terminal_probe_never_changes_the_route():
    r = Router(None, classify=None)
    plain = r.route("in the terminal, run the tests", "jarvis")
    # older / partly wired session manager: no callable at all
    assert r.route("in the terminal, run the tests", "jarvis",
                   terminal_open=None) == plain
    assert r.route("in the terminal, run the tests", "jarvis",
                   terminal_open="not callable") == plain
    boom = FakeTerminals("jarvis", boom=True)
    assert r.route("in the terminal, run the tests", "jarvis",
                   terminal_open=boom) == plain
    assert boom.calls == ["jarvis"]


def test_terminal_routing_does_not_swallow_ordinary_utterances():
    term = FakeTerminals("jarvis")
    r = Router(None, classify=None)
    for text in ("open the terminal", "what time is it", "fix the parser",
                 "have it ready by noon", "tell me a joke",
                 "remind me to check the terminal at six",
                 "cancel that", "use opus"):
        d = r.route(text, "jarvis", terminal_open=term)
        assert d.reason != "terminal", (text, d)
        assert not d.args.get("terminal"), (text, d.args)


def test_modifiers_still_apply_to_a_terminal_utterance():
    term = FakeTerminals("jarvis")
    d = Router(None, classify=None).route(
        "in the terminal, use opus and rewrite the parser", "jarvis",
        terminal_open=term)
    assert d.reason == "terminal" and d.prompt == "rewrite the parser"
    assert d.args["model"] == "opus" and d.args["terminal"] is True


def test_a_session_action_beats_the_terminal_address():
    """"in the terminal, cancel that" is still Jarvis's cancel: queued as a
    prompt it would sit behind the task the user is trying to stop."""
    term = FakeTerminals("jarvis")
    r = Router(None, classify=None)
    for text in ("in the terminal, cancel that", "tell it to cancel that",
                 "cancel that in the terminal"):
        d = r.route(text, "jarvis", terminal_open=term)
        assert d.kind == "action" and d.action == "cancel", (text, d)
