"""Tests for jarvis.tools.registry's budget reporting (spec 4.1).

The registry is the one place that knows what the tool block costs, and
the cost the model actually pays is PROMPT TOKENS, not the tool count:
measured on this box 2026-08-31 against the resident gemma4:26b, the 28
tools the app registers are 2226 prompt tokens against a 900-token budget,
and even the spec's own eleven measure 1219.  These tests pin the two
things that follow: the report is one line, not one per tool, and the
schema bytes never vary (jarvis/brain.py's static-prefix rule caches on
them)."""
import json

from jarvis.tools.registry import (CHARS_PER_TOKEN, MAX_SCHEMA_TOKENS,
                                   MAX_TOOLS, ToolRegistry, ToolResult,
                                   ToolSpec)


def _fill(reg, n, prefix="t"):
    for i in range(n):
        reg.register(ToolSpec(f"{prefix}{i}", "Does a thing for Hunter."))
    return reg


def test_the_over_budget_report_is_one_line_not_one_per_tool(caplog):
    """A 28-tool boot wrote a 17-line WARNING ladder that said the same
    thing 17 times and buried the tool list between the rungs."""
    reg = ToolRegistry()
    with caplog.at_level("INFO"):
        _fill(reg, MAX_TOOLS + 17)
        assert [r.message for r in caplog.records] == [], caplog.records
        reg.schemas()
        reg.schemas()                       # every turn asks; it reports once
    lines = [r for r in caplog.records if "registered" in r.message]
    assert len(lines) == 1, [r.message for r in lines]
    assert lines[0].levelname == "WARNING"
    said = lines[0].getMessage()
    assert "28 registered (budget 11)" in said, said
    assert f"budget {MAX_SCHEMA_TOKENS}" in said, said


def test_a_changed_tool_set_is_reported_again(caplog):
    reg = ToolRegistry()
    with caplog.at_level("INFO"):
        _fill(reg, MAX_TOOLS + 1)
        assert not [r for r in caplog.records if "registered" in r.getMessage()]
        reg.schemas()
        reg.register(ToolSpec("late", "Registered after the first turn."))
        reg.schemas()
    said = [r.getMessage() for r in caplog.records if "registered" in r.getMessage()]
    assert len(said) == 2, said
    assert "12 registered" in said[0] and "13 registered" in said[1], said


def test_schema_budget_counts_tokens_not_only_tools():
    """A registry can sit inside MAX_TOOLS and still blow the token
    budget, which is the one that is paid in prefill on every turn."""
    reg = ToolRegistry()
    reg.register(ToolSpec("small", "Short."))
    state = reg.schema_budget()
    assert state["tools"] == 1 and state["ok"] is True
    assert state["max_schema_tokens"] == MAX_SCHEMA_TOKENS
    assert state["schema_tokens"] < MAX_SCHEMA_TOKENS

    reg.register(ToolSpec(
        "fat", "Short enough to pass the word cap.",
        {"type": "object", "properties": {
            "q": {"type": "string", "description": "x" * 4 * MAX_SCHEMA_TOKENS}}}))
    state = reg.schema_budget()
    assert state["tools"] == 2 <= MAX_TOOLS         # inside the COUNT budget
    assert state["over_word_cap"] == []             # and inside the word cap
    assert state["schema_tokens"] > MAX_SCHEMA_TOKENS
    assert state["ok"] is False                     # but not inside the bill
    # budget() keeps its shape: brain's report compares the dict whole
    assert set(reg.budget()) == {"tools", "max_tools", "over_word_cap", "ok"}


def test_the_token_estimate_tracks_the_schema_bytes():
    reg = _fill(ToolRegistry(), 12)
    chars = len(json.dumps(reg.schemas()))
    assert reg.schema_tokens() == int(chars / CHARS_PER_TOKEN)


def test_schemas_are_byte_stable_across_calls():
    """The static-prefix rule (jarvis/brain.py): the tool block is part of
    the cached prefix, so a set that renders differently between turns
    costs a full reprocess every turn."""
    reg = _fill(ToolRegistry(), 6)
    first = json.dumps(reg.schemas())
    assert all(json.dumps(reg.schemas()) == first for _ in range(3))


# ------------------------------------------------------- reserved args
# A handler may accept a keyword only trusted code (a commander force_args)
# is allowed to fill. spotify_liked's ``shuffle`` earned this: gemma4 sent
# shuffle=true unasked for "Play my like songs." (live, 2026-09-01 19:59)
# and the newest-first default was lost. registry.call carries the model's
# args and nothing else, so the utterance cannot be checked in the handler;
# the registry strips the reserved keys from model-originated calls instead.
def _reserved_reg(seen):
    reg = ToolRegistry()

    def handler(device="", shuffle=None, **_):
        seen.append({"device": device, "shuffle": shuffle})
        return ToolResult(text="ok")

    reg.register(ToolSpec("liked", "Play the liked songs.",
                          {"type": "object",
                           "properties": {"device": {"type": "string"}}},
                          handler, reserved=frozenset({"shuffle"})))
    return reg


def test_reserved_args_are_dropped_from_model_calls_only(caplog):
    seen = []
    reg = _reserved_reg(seen)
    with caplog.at_level("INFO"):
        res = reg.call("liked", {"shuffle": True, "device": "phone"}, from_model=True)
    assert res.ok and seen == [{"device": "phone", "shuffle": None}]
    said = [r.getMessage() for r in caplog.records if "reserved" in r.getMessage()]
    assert said == ['tool liked: dropped model-supplied reserved args {"shuffle": true}']
    # the commander's forced call is trusted: every key reaches the handler
    reg.call("liked", {"shuffle": True, "device": "phone"})
    assert seen[-1] == {"device": "phone", "shuffle": True}
    # ...and the default (from_model=False) is the trusted path, so no
    # existing caller changes behaviour
    reg.call("liked", {"shuffle": False})
    assert seen[-1] == {"device": "", "shuffle": False}


def test_reserved_args_are_dropped_from_json_string_model_calls(caplog):
    """Some models emit arguments as a JSON string; the filter runs after
    the parse, so the string form is not a way around it."""
    seen = []
    reg = _reserved_reg(seen)
    with caplog.at_level("INFO"):
        reg.call("liked", '{"shuffle": true}', from_model=True)
    assert seen == [{"device": "", "shuffle": None}]
    # nothing dropped -> nothing logged: the line is an audit trail, not noise
    caplog.clear()
    with caplog.at_level("INFO"):
        reg.call("liked", {"device": "phone"}, from_model=True)
    assert not [r for r in caplog.records if "reserved" in r.getMessage()]


# ------------------------------------------- args derived from the words
# Reserving a key closes the model's vote on it, which leaves a hole: the
# COMMANDER route is then the only thing that can say yes, and a phrasing
# its matcher misses silently gets the default (2026-09-02 review -- "put
# my liked songs on shuffle please" played in order and said "newest
# first"). ``derive`` fills that hole: on a model call the spec reads the
# utterance itself, after the reserved keys are stripped.
def _derive_reg(seen, derive):
    reg = ToolRegistry()

    def handler(device="", shuffle=None, **_):
        seen.append({"device": device, "shuffle": shuffle})
        return ToolResult(text="ok")

    reg.register(ToolSpec("liked", "Play the liked songs.",
                          {"type": "object",
                           "properties": {"device": {"type": "string"}}},
                          handler, reserved=frozenset({"shuffle"}),
                          derive=derive))
    return reg


def test_derived_args_replace_the_models_on_model_calls(caplog):
    seen = []
    reg = _derive_reg(seen, lambda t: {"shuffle": "shuffle" in (t or "")})
    with caplog.at_level("INFO"):
        reg.call("liked", {"shuffle": False}, from_model=True,
                 utterance="put my liked songs on shuffle please")
    assert seen == [{"device": "", "shuffle": True}]
    said = [r.getMessage() for r in caplog.records if "utterance" in r.getMessage()]
    assert said == ['tool liked: {"shuffle": true} from the utterance']
    # no utterance to read (a channel that has none): the handler's own
    # default stands, exactly as before
    seen.clear()
    reg.call("liked", {}, from_model=True)
    assert seen == [{"device": "", "shuffle": False}]


def test_derive_never_touches_a_forced_call():
    """force_args ARE the utterance's, decided by the commander: a second
    reading of the words must not overrule the first."""
    seen = []
    reg = _derive_reg(seen, lambda t: {"shuffle": True})
    reg.call("liked", {"shuffle": False}, utterance="shuffle my liked songs")
    assert seen == [{"device": "", "shuffle": False}]


def test_a_deriver_that_raises_is_not_a_failed_tool_call(caplog):
    seen = []

    def boom(_text):
        raise RuntimeError("kaboom")

    reg = _derive_reg(seen, boom)
    with caplog.at_level("ERROR"):
        res = reg.call("liked", {}, from_model=True, utterance="anything")
    assert res.ok and seen == [{"device": "", "shuffle": None}]
    assert any("derive" in r.getMessage() for r in caplog.records)


def test_a_spec_that_advertises_a_reserved_arg_is_a_warned_drift(caplog):
    """Reserved AND in the schema invites the model to send an argument
    call() will drop -- register() says so once, at boot."""
    reg = ToolRegistry()
    with caplog.at_level("WARNING"):
        reg.register(ToolSpec("drift", "Advertises what it drops.",
                              {"type": "object",
                               "properties": {"shuffle": {"type": "boolean"}}},
                              lambda **_: ToolResult(text="ok"),
                              reserved=frozenset({"shuffle"})))
    said = [r.getMessage() for r in caplog.records]
    assert said == ["tool drift advertises reserved argument(s) shuffle in its schema"]
    # the default spec reserves nothing and the schema is untouched
    assert ToolSpec("plain", "Plain.").reserved == frozenset()
