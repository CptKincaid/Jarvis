"""FOUND 2026-08-26 (adversarial routing re-verification): the cross-cutting
model / parallel modifiers are extracted with an UNANCHORED re.search() over
the whole utterance and the matched span is deleted from the text, so when the
phrase is part of the sentence's meaning rather than a trailing modifier, the
prompt actually sent to Claude is mutilated.

Router.route() runs, before any rule:

    m = _MODEL_RX.search(work);    work = _strip_span(work, m.span())
    m = _PARALLEL_RX.search(work); work = _strip_span(work, m.span())

The module docstring says these are pulled out "so that 'refactor the parser,
use fable, in parallel' still hits rule 4" — i.e. the design intends a trailing,
comma-delimited modifier.  Nothing requires a delimiter or a sentence-final
position, so any in-sentence occurrence is eaten.

Measured with the real Router on the real AssistantConfig:

    'refactor the parser so the workers run in parallel'
        -> prompt='refactor the parser so the workers run'  args={'parallel': True}
    'document how we use opus in the pipeline'
        -> prompt='document how we in the pipeline'         args={'model': 'opus'}
    'rewrite the docs to use sonnet as the example model'
        -> prompt='rewrite the docs to as the example model'
    'fix the bug where tasks run at the same time'
        -> prompt='fix the bug where tasks run'             args={'parallel': True}

Confirmed end to end against the real ClaudeSessionManager (spawn seam
blocked, no CLI invoked): the first one reaches the queue as

    Task.prompt = 'refactor the parser so the workers run'
    Task.model  = 'fable'   Task.parallel = True

so the requirement "in parallel" is deleted from the instruction and silently
reinterpreted as a task-scheduling flag — the opposite of what was asked.

Defect: jarvis/router.py, the _MODEL_RX / _PARALLEL_RX extraction at the top of
Router.route().

FIXED 2026-08-26: both are now taken through _modifier_match(), which accepts
an occurrence only when it opens the utterance, when a comma/semicolon/colon
delimits it, or when it trails a sentence with no subordinate clause -- so
"...so the workers run in parallel" and "...where tasks run at the same time"
stay in the prompt while "..., in parallel" is still a flag.  This test is no
longer xfail.

STILL BROKEN AT VERIFICATION ROUND 2, fixed 2026-08-27: an utterance-OPENING
occurrence was accepted unconditionally, so the two positions below were still
mangled -- and the first of them silently forced the expensive model:

    'use opus level reasoning and rewrite the parser'
        -> prompt='level reasoning and rewrite the parser'  args={'model': 'opus'}
    'in parallel with the release, write the changelog'
        -> prompt='write the changelog'  args={'parallel': True}   (then local)

An opening modifier now counts only when the phrase stands alone, is joined to
the task by a delimiter / "and" / "then" / an infinitive ("use opus to rewrite
the parser"), or is a bare directive with a courtesy tail ("use opus please").
Two further guards apply in every position: the alias may not be doing
adjectival work ("opus level reasoning", "opus-style", "fable grade") and
"in parallel" may not be prepositional ("in parallel WITH the release").
The full position table is pinned in tests/test_router.py::MODIFIER_TABLE.
"""
import pytest

from jarvis.router import Router

MANGLED = [
    ("refactor the parser so the workers run in parallel",
     "refactor the parser so the workers run in parallel"),
    ("document how we use opus in the pipeline",
     "document how we use opus in the pipeline"),
    ("rewrite the docs to use sonnet as the example model",
     "rewrite the docs to use sonnet as the example model"),
    ("fix the bug where tasks run at the same time",
     "fix the bug where tasks run at the same time"),
    # round 2: the opening position
    ("use opus level reasoning and rewrite the parser",
     "use opus level reasoning and rewrite the parser"),
    ("write opus-style docstrings for the parser",
     "write opus-style docstrings for the parser"),
    ("run the migration in parallel with the backfill",
     "run the migration in parallel with the backfill"),
]


@pytest.mark.parametrize("text,expected", MANGLED)
def test_the_prompt_sent_to_claude_keeps_the_whole_sentence(text, expected):
    d = Router(None, classify=None).route(text)
    assert d.kind == "claude", (text, d.kind, d.reason)
    assert d.prompt == expected, (text, d.prompt)


def test_a_trailing_comma_delimited_modifier_is_still_stripped_correctly():
    """Control: the intended form keeps working, isolating the cause to the
    unanchored search rather than to modifier extraction as such."""
    d = Router(None, classify=None).route("refactor the parser, use fable, in parallel")
    assert d.kind == "claude", (d.kind, d.reason)
    assert d.prompt == "refactor the parser", d.prompt
    assert d.args.get("model") == "fable"
    assert d.args.get("parallel") is True


def test_an_opening_modifier_is_still_a_directive_when_it_is_one():
    """Control for the round-2 fix: the position that was over-consumed is
    not disabled, only qualified."""
    for text, prompt, model in (
            ("use opus, rewrite the parser", "rewrite the parser", "opus"),
            ("use opus and rewrite the parser", "rewrite the parser", "opus"),
            ("use opus to rewrite the parser", "rewrite the parser", "opus"),
            ("in parallel, refactor the parser", "refactor the parser", None)):
        d = Router(None, classify=None).route(text)
        assert d.kind == "claude" and d.prompt == prompt, (text, d)
        assert d.args.get("model") == model, (text, d.args)


def test_the_round_two_failures_do_not_set_a_flag_either():
    """The prompt surviving is only half of it: an unintended 'use opus'
    also spends the user's credits on the expensive model."""
    d = Router(None, classify=None).route(
        "use opus level reasoning and rewrite the parser")
    assert "model" not in d.args, d.args
    d = Router(None, classify=None).route(
        "in parallel with the release, write the changelog")
    assert "parallel" not in d.args, d.args
