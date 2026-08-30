"""Questions for the internet go to Claude's CLI as a one-shot.

The local model cannot know today's result or price, and no local tool
covers it. Rather than a search package in the shared venv (12 packages for
ddgs + trafilatura) the router hands such questions to `claude -p` with web
search allowed -- ~12 s measured -- acknowledging at once and speaking the
answer through the same brain callback that closes the turn.
"""
import subprocess
from types import SimpleNamespace

import pytest

import jarvis.brain as brain_mod
from jarvis.brain import WEB_FAIL_LINE, WEB_SLOW_LINE, clean_web_answer
from jarvis.commander import WEB_LOOKUP_LINE, WEB_UNAVAILABLE_LINE
from jarvis.router import Router, _WEB_CUE_RX


@pytest.mark.parametrize("said", [
    "look up who won the last formula one race",
    "look it up",
    "search the web for the dutch grand prix result",
    "google the population of texas",
    "what's the latest news about nvidia",
    "who won the game last night",
    "what's the price of bitcoin",
    "what's the stock price of apple",
    "how much does a tesla model 3 cost",
    "what happened in london today",
    "is home depot open right now",
    "when does the new iphone come out",
    "ask claude to look up the weather in tokyo",       # a claude cue that is a lookup
])
def test_web_cues(said):
    assert _WEB_CUE_RX.search(said), said


@pytest.mark.parametrize("said", [
    "what's the weather", "what's the news", "set a timer for five minutes",
    "what's on my calendar tomorrow", "play some music", "what time is it",
    "refactor the parser", "how are you",
])
def test_local_questions_stay_local(said):
    assert not _WEB_CUE_RX.search(said), said


def test_the_router_returns_a_web_decision_with_the_question_as_prompt():
    r = Router(SimpleNamespace(get=lambda k, d=None: d, skill_phrases={}),
               classify=lambda text: ("local", 0.9))
    d = r.route("look up who won the dutch grand prix")
    assert d.kind == "web" and d.prompt == "look up who won the dutch grand prix"
    d = r.route("ask claude to look up who won the dutch grand prix")
    assert d.kind == "web", "a claude cue that is a lookup must not open a coding session"


# ------------------------------------------------------------ the answer
def test_clean_web_answer_strips_what_the_model_was_told_not_to_add():
    raw = ("Lando Norris won the **Dutch Grand Prix** at Zandvoort on August 23.\n"
           "He beat [Kimi Antonelli](https://example.com/a) and George Russell.\n\n"
           "Sources: [Formula1.com](https://www.formula1.com/en/latest/x) "
           "https://www.bbc.com/sport/formula1/123")
    line = clean_web_answer(raw)
    assert "http" not in line and "[" not in line and "**" not in line
    assert "Sources" not in line
    assert line.startswith("Lando Norris won the Dutch Grand Prix")
    assert "Kimi Antonelli" in line


def test_clean_web_answer_caps_the_length():
    raw = " ".join(f"Sentence number {i} is here." for i in range(12))
    assert clean_web_answer(raw).count(".") <= 3


def _brain(monkeypatch, run):
    monkeypatch.setattr(brain_mod.MACHINE, "claude_bin", "/usr/bin/true", raising=False)
    b = brain_mod.JarvisBrain(None, None, registry=None)
    monkeypatch.setattr(b, "_run_claude", run)
    monkeypatch.setattr(b, "_remember", lambda q, tags: None)
    return b


def test_web_answer_runs_the_cli_with_search_allowed_and_speaks_the_answer(monkeypatch):
    seen = {}

    def run(prompt, timeout, extra_args=None):
        seen["prompt"], seen["timeout"], seen["args"] = prompt, timeout, extra_args
        return "Lando Norris won at Zandvoort.\n\nSources: [x](https://x)"
    b = _brain(monkeypatch, run)
    got = []
    t = b.web_answer("who won the dutch grand prix", callback=got.append, model="haiku")
    assert t is not None
    t.join(5)
    assert got == [[("SPEAK", "Lando Norris won at Zandvoort.")]]
    assert "who won the dutch grand prix" in seen["prompt"]
    assert seen["args"] == ["--model", "haiku", "--allowedTools", "WebSearch,WebFetch"]
    assert seen["timeout"] == brain_mod.WEB_TIMEOUT_S
    assert not b._busy


def test_web_answer_timeout_and_failure_speak_their_own_lines(monkeypatch):
    def slow(prompt, timeout, extra_args=None):
        raise subprocess.TimeoutExpired("claude", timeout)
    b = _brain(monkeypatch, slow)
    got = []
    b.web_answer("q", callback=got.append).join(5)
    assert got == [[("SPEAK", WEB_SLOW_LINE)]]

    def boom(prompt, timeout, extra_args=None):
        raise RuntimeError("cli exploded")
    b = _brain(monkeypatch, boom)
    got = []
    b.web_answer("q", callback=got.append).join(5)
    assert got == [[("SPEAK", WEB_FAIL_LINE)]]

    def empty(prompt, timeout, extra_args=None):
        return "   "
    b = _brain(monkeypatch, empty)
    got = []
    b.web_answer("q", callback=got.append).join(5)
    assert got == [[("SPEAK", WEB_FAIL_LINE)]]


def test_web_answer_refuses_when_the_cli_is_missing_or_the_brain_is_busy(monkeypatch):
    b = _brain(monkeypatch, lambda *a, **k: "x")
    monkeypatch.setattr(brain_mod.MACHINE, "claude_bin", "", raising=False)
    assert b.web_answer("q", callback=lambda t: None) is None
    monkeypatch.setattr(brain_mod.MACHINE, "claude_bin", "/usr/bin/true", raising=False)
    import time
    b._busy, b._busy_since = True, time.monotonic()   # a live job, not a stale flag
    got = []
    assert b.web_answer("q", callback=got.append) is None
    assert got and "Still on the last one" in got[0][0][1]


# ------------------------------------------------------------ the commander
def test_the_commander_acknowledges_and_leaves_the_turn_open():
    from jarvis.commander import Commander
    from jarvis.router import RouteDecision
    calls = []
    c = object.__new__(Commander)
    c.services = SimpleNamespace(
        brain=SimpleNamespace(web_answer=lambda q, model="haiku": calls.append((q, model)) or object()),
        assistant=SimpleNamespace(get=lambda k, d=None: {"claude.web_model": "sonnet"}.get(k, d)))
    res = c._web_lookup(RouteDecision("web", "web-cue", prompt="who won"))
    assert calls == [("who won", "sonnet")]
    assert res.reply == WEB_LOOKUP_LINE and res.speak and res.ack and res.done is False


def test_the_commander_says_so_when_the_lookup_cannot_start():
    from jarvis.commander import Commander
    from jarvis.router import RouteDecision
    c = object.__new__(Commander)
    c.services = SimpleNamespace(brain=SimpleNamespace(web_answer=lambda q, model="haiku": None),
                                 assistant=SimpleNamespace(get=lambda k, d=None: d))
    res = c._web_lookup(RouteDecision("web", "web-cue", prompt="who won"))
    assert res.reply == WEB_UNAVAILABLE_LINE and res.done is not False
    c.services = SimpleNamespace(brain=None, assistant=SimpleNamespace(get=lambda k, d=None: d))
    assert c._web_lookup(RouteDecision("web", "web-cue", prompt="q")).reply == WEB_UNAVAILABLE_LINE
