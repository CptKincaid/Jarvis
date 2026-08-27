"""Local-vs-Claude router for Jarvis (spec 2026-08-26 section 5.1).

Pure rules, in a binding order; the local model is ONLY the tie-breaker:

  1. session actions   cancel / work on / resume / new project / model /
                       fast mode ("in parallel" inside a task -> args)
  2. skill phrases     cfg.claude.skill_phrases (regex -> template, $1) and
                       "run the <name> skill (on|with) <args>" -> "/<name> <args>"
  3. explicit cue      "claude", "have claude", "ask claude" (stripped)
  4. coding cues       verb x object, verb x path, question-about-code,
                       "step by step"/"then ... then" with a coding verb
  5. local cues        weather, clock, calendar, timekeeper, notes, mail,
                       briefing, location, greetings, jokes, question words
  6. neither / both    classify(text): a LOCAL answer routes silently at
                       >= 0.75 (answering locally is free); a CLAUDE answer
                       is a billed CLI task, so it routes silently only at
                       >= 0.95 AND with a coding cue in the utterance --
                       anything below that asks instead (remembered 90 s;
                       resolve_answer).  Asking is free; dispatching is not.
  7. length prior      a claude decision needs a cue; a bare <= 6-word
                       sentence with no cue is local without asking

Cue strength: a coding VERB+OBJECT pair (or verb+path) is strong, a bare
coding noun is weak; a local question/imperative form ("what's the
weather", "remind me to ...") is strong, a bare topic noun ("weather"
inside "fix the weather tool") is weak. Only strong cues count as "both".
Utterances that START with a wrapper imperative (remind me to ..., set a
timer, note that ...) are local whatever follows, because the tail is
free text.

The router never speaks, publishes or touches a service; the commander
does. `estimate_size(prompt)` feeds the fable escalation (7.1).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from jarvis.logs import get_logger

log = get_logger("router")

ROUTER_QUESTION = "Shall I hand that to Claude, sir, or is it a quick one for me?"
ASK_TTL_S = 90.0
# Rule 6 is asymmetric on purpose: a wrong "local" costs one cheap Ollama
# turn, a wrong "claude" spends the user's Claude credits with no
# confirmation.  Measured on an 80-utterance labelled set (40 coding, 40
# ordinary English) against gemma4:26b on 2026-08-26: every ordinary-English
# sentence the classifier called "claude" scored <= 0.90 (0.60-0.90), every
# genuine coding sentence scored >= 0.95 (0.95-1.00).  0.95 is the only
# value between the two, and it costs nothing: 0 of 12 coding utterances
# that reach rule 6 have to be asked about.
CLASSIFY_THRESHOLD = 0.75          # the local direction (free to be wrong)
CLASSIFY_CLAUDE_THRESHOLD = 0.95   # the billed direction
SHORT_WORDS = 6
MODEL_ALIASES = ("opus", "sonnet", "fable", "haiku")
BIG_MODEL = "fable"

DEFAULT_SKILL_PHRASES = {
    "^review (this|my|the) code$": "/code-review",
    "^commit (this|it|that)$": "/commit",
    "^simplify (this|it|that)$": "/simplify",
    "^security review$": "/security-review",
    "^run a ralph loop on (.+)$": "/ralph-loop $1",
    "^plan a feature (.+)$": "/feature-dev $1",
}


@dataclass
class RouteDecision:
    kind: str                 # "local" | "claude" | "ask" | "action"
    reason: str               # rule name, for the log
    prompt: str = ""          # the text handed to Claude (skill-expanded)
    project: str = ""         # slug when the utterance named one
    action: str = ""          # cancel | work_on | resume | new_project | set_model | fast_mode
    args: dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class PendingAsk:
    text: str
    project: str
    at: float
    # The modifiers the utterance carried ("use haiku", "in parallel").
    # Without these the answer "yes" dispatches on the DEFAULT model, so
    # asking for the cheap model and saying yes bought the expensive one.
    args: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers
_PREFIX_RX = re.compile(r"^(?:(?:hey|ok|okay|hi)[,\s]+)?jarvis[,!:]?\s*", re.I)
_TRAIL_RX = re.compile(r"[\s.!?]+$")


def normalise(text: str) -> str:
    """Strip the 'jarvis' prefix, surrounding space and trailing
    punctuation; casing is kept (file and class names matter to Claude)."""
    t = (text or "").strip()
    t = _PREFIX_RX.sub("", t, count=1)
    t = _TRAIL_RX.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def _strip_span(text: str, span: tuple) -> str:
    """Remove a span and tidy the join (", and", double spaces, commas)."""
    a, b = span
    left, right = text[:a], text[b:]
    # A modifier lifted out from between a delimiter and a conjunction
    # ("in the terminal, USE OPUS and rewrite the parser") leaves the
    # conjunction dangling on the seam.
    if left.rstrip()[-1:] in ",;:":
        right = re.sub(r"^\s*[,;:]?\s*(?:and|then)\s+", " ", right, flags=re.I)
    out = (left + " " + right)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"^\s*(?:,|and|then)\s+", "", out, flags=re.I)
    out = re.sub(r"\s+(?:,|and|then)\s*$", "", out, flags=re.I)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r",\s*,", ",", out)
    return out.strip(" ,;")


# A subordinate clause means the modifier phrase belongs to the sentence,
# not to the dispatch: "refactor the parser SO THE WORKERS run in parallel"
# is a requirement, "fix the failing test, in parallel" is a flag.
_SUBORD_RX = re.compile(
    r"\b(?:so|where|when|while|because|since|although|though|if|unless|"
    r"until|after|before|whether|how|why|which|who|that|as)\b", re.I)
# The alias is doing ADJECTIVAL work when a noun-modifier word follows it:
# "use opus level reasoning", "opus-style docstrings", "a fable grade
# summary".  The word names a standard, not a model to dispatch on, so the
# phrase belongs to the prompt.
_ADJECTIVAL_RX = re.compile(
    r"^[-\s]*(?:level|levels|grade|graded|class|tier|quality|calibre|caliber|"
    r"standard|standards|style|styled|esque|like|sized|size|mode|"
    r"reasoning|thinking|depth|effort|power|strength|territory|worthy|type|"
    r"flavou?r|vibes?)\b", re.I)
# "in parallel WITH the release" / "in parallel TO the rollout" is a
# comparison inside the sentence, not the parallel-dispatch flag.  (The
# flag's own tails -- "in parallel with the other task" -- are part of
# _PARALLEL_RX itself and are consumed by it, so they never reach here.)
_PREPOSITIONAL_RX = re.compile(r"^\s*(?:with|to|as|against|alongside)\b", re.I)
# What joins a leading directive to the task it applies to.
_JOIN_RX = re.compile(r"^\s*(?:[,;:]|and\b|then\b)", re.I)
_INFINITIVE_RX = re.compile(r"^\s*to\s+(?P<verb>[\w-]+)", re.I)
# Courtesy / scope fillers that can follow a bare model directive without
# making it part of a sentence ("use opus please", "use haiku for now").
# _action() recognises exactly this remainder as a set_model utterance.
_MODEL_FILLER = (r"(?:for (?:this|the next one|now)|please|from now on|"
                 r"on this one|for this one|for this task)")
_MODEL_FILLER_RX = re.compile(_MODEL_FILLER + r"?", re.I)


def _modifier_match(rx: re.Pattern, text: str, kind: str = "model"
                    ) -> Optional[tuple]:
    """The first occurrence of a cross-cutting modifier ("use fable", "in
    parallel") that is genuinely a modifier rather than part of what the
    user asked for.

    It qualifies when it stands alone, when a comma/semicolon/colon
    delimits it, when it opens the utterance and is joined to the task by
    a delimiter, "and"/"then" or an infinitive ("use opus to rewrite the
    parser"), or when it trails a sentence with no subordinate clause.

    It does NOT qualify while the token is doing grammatical work in the
    sentence -- adjectival ("opus level reasoning", "opus-style") or, for
    "in parallel", prepositional ("in parallel with the release").
    Anything else is left in the prompt: deleting it would send Claude an
    instruction the user did not give, and (for the model) would spend the
    user's money on a model he never asked for.
    """
    for m in rx.finditer(text):
        a, b = m.span()
        head, tail = text[:a], text[b:]
        if _ADJECTIVAL_RX.match(tail):
            continue                              # "use opus LEVEL reasoning"
        if kind == "parallel" and _PREPOSITIONAL_RX.match(tail):
            continue                              # "in parallel WITH the release"
        if not head.strip():                      # opens the utterance
            if not tail.strip():
                return m, (a, b)                  # stands alone
            if _JOIN_RX.match(tail):
                return m, (a, b)                  # "use opus, ..." / "... and ..."
            if kind == "model":
                if _MODEL_FILLER_RX.fullmatch(tail.strip(" ,.")):
                    return m, (a, b)              # "use opus please"
                mi = _INFINITIVE_RX.match(tail)
                if mi and _VERB_RX.fullmatch(mi.group("verb")):
                    # the "to" belongs to the directive, not to the task
                    return m, (a, b + mi.end() - len(mi.group("verb")))
            continue                              # runs on into the sentence
        if m.group(0)[:1] in ",;:" or head.rstrip()[-1:] in ",;:":
            return m, (a, b)                      # delimited on the left
        if not tail.strip() and not _SUBORD_RX.search(head):
            return m, (a, b)                      # trailing modifier
    return None


def _expand_template(template: str, m: re.Match) -> str:
    def repl(g):
        idx = int(g.group(1))
        try:
            return (m.group(idx) or "").strip()
        except (IndexError, re.error):
            return ""
    return re.sub(r"\$(\d+)", repl, template).strip()


# ------------------------------------------------------------ rule 1 tables
_CANCEL_RX = re.compile(
    r"^(?:cancel|abort|stop that)(?:\s+(?:that|it|this|the task|the job|"
    r"claude|the claude task|everything))?$", re.I)
_WORK_ON_RX = re.compile(
    r"^(?:let'?s\s+)?(?:work on|switch to|go to|open)\s+(?:the\s+)?"
    r"(?P<name>.+?)(?:\s+project)?$", re.I)
_WORK_ON_STRICT_RX = re.compile(
    r"^(?:let'?s\s+)?(?:work on\s+(?:the\s+)?(?P<n1>.+?)(?:\s+project)?|"
    r"(?:switch to|go to|open|change to)\s+(?:the\s+)?(?P<n2>.+?)\s+project)$",
    re.I)
_RESUME_RX = re.compile(
    r"^(?:(?:let'?s\s+)?(?:pick up|carry on|continue|resume)"
    r"(?:\s+(?:from\s+)?where we left off(?:\s+(?:on|with|in)\s+(?:the\s+)?"
    r"(?P<n1>.+?)(?:\s+project)?)?|"
    r"\s+(?:with\s+|on\s+)?(?:the\s+)?(?P<n2>.+?)\s+project(?:\s+where we left off)?)"
    r"|what (?:were|was|are) we working on(?:\s+(?P<w1>yesterday|today|last week|"
    r"last night|this morning|earlier))?"
    r"|what we were working on(?:\s+(?P<w2>yesterday|today|last week|last night|"
    r"this morning|earlier))?"
    r"|(?:pick up|continue|resume)\s+(?:the\s+)?(?P<n3>.+?)\s+"
    r"(?:project\s+)?(?:from\s+)?(?P<w3>yesterday|last week|last night|this morning|"
    r"earlier))(?:\s+(?P<w4>yesterday|today|last week|last night|this morning|"
    r"earlier))?$",
    re.I)
_NEW_PROJECT_RX = re.compile(
    r"^(?:start|create|make|set up|setup|begin)\s+(?:a\s+)?new project"
    r"(?:\s+(?:called|named|for))?\s+(?P<name>.+?)$|"
    r"^new project(?:\s+(?:called|named))?\s+(?P<name2>.+?)$", re.I)
_MODEL_RX = re.compile(
    r"\b(?:use|with|on|switch to|using|try)\s+(?:the\s+)?"
    r"(?P<alias>opus|sonnet|fable|haiku)"
    r"(?:\s+(?:model|for (?:this|it|that)(?:\s+(?:one|task))?))?\b"
    r"|\b(?P<hard>think hard(?:er)?|think (?:really )?deeply|this is a big one|"
    r"this one'?s big|big one)\b", re.I)
_FAST_RX = re.compile(
    r"^(?:(?:turn|switch|put|set)\s+)?fast mode\s+(?P<on>on|off)$|"
    r"^(?:(?P<en>enable|turn on|switch on)|(?P<dis>disable|turn off|switch off))"
    r"\s+fast mode$", re.I)
_PARALLEL_RX = re.compile(
    r"[,;]?\s*\b(?:in parallel|at the same time|simultaneously|concurrently)\b"
    r"(?:\s+(?:with|as)\s+(?:the\s+)?(?:other|last|current|running)\s+"
    r"(?:one|task|job))?", re.I)
# "in my repo" / "in this project" name no project at all: they mean the
# one already open.  A bare determiner captured as a <name> is slugified
# into "my" / "this", which ClaudeSessionManager cannot resolve -- and it
# OVERRIDES the active project, so the determiner breaks a path that works
# without it.  Determiners are consumed, never captured.
_DETERMINER = r"(?:my|our|your|its|their|this|that|these|those|the|current)"
_PROJECT_RX = re.compile(
    r"[,;]?\s*\b(?:in|on|for|inside|under|within)\s+"
    r"(?:" + _DETERMINER + r"\s+)*"
    r"(?P<name>(?!" + _DETERMINER + r"\b)(?:[\w-]+\s+){0,2}[\w-]+?)"
    r"\s+(?:project|repo|repository|codebase)\b",
    re.I)
_CLAUDE_CUE_RX = re.compile(
    r"^(?:(?:please\s+)?(?:have|ask|get|tell|let|make)\s+claude(?:\s+to)?\s+"
    r"|claude[,:]?\s+(?:please\s+)?|(?:please\s+)?hand (?:this|that|it) (?:to|over to) claude[,:]?\s*)"
    r"|[,;]?\s*\b(?:with|using|via|through)\s+claude\b|[,]?\s+claude$"
    r"|\bhave claude\b|\bask claude(?: to)?\b|\bget claude to\b|\btell claude to\b",
    re.I)
# ---------------------------------------------------------- terminal address
# A project whose tmux session has a client attached is a LIVE terminal the
# user can see; addressing it ("in the terminal, run the tests", "tell it to
# stop") means the words go to THAT Claude session rather than starting a
# task somewhere else or being answered locally.  Only ClaudeSessionManager
# knows whether a terminal is attached, so the router takes the answer as a
# callable (rule 1b); with no callable, these phrasings route as they always
# did.
_TERMINAL_NAME = r"(?!(?:terminal|claude|open|other|same|new)\b)(?P<name>[\w-]+)\s+"
_TERMINAL_LEAD_RX = re.compile(
    r"^(?:please\s+)?(?:in|on|over in|through|into)\s+"
    r"(?:the\s+|my\s+|that\s+)?(?:" + _TERMINAL_NAME + r")?"
    r"terminal(?:\s+(?:session|window))?\s*[,;:]?\s+"
    r"(?:please\s+)?(?P<rest>.+)$", re.I)
_TERMINAL_TAIL_RX = re.compile(
    r"^(?P<rest>.+?)\s*[,;]?\s+(?:in|on|over in|through|into)\s+"
    r"(?:the\s+|my\s+|that\s+)?(?:" + _TERMINAL_NAME + r")?"
    r"terminal(?:\s+(?:session|window))?$", re.I)
# "tell it to ...", "have it ..." -- the pronoun only has a referent when a
# terminal is open, so a coding verb is required to keep "have it ready by
# noon" out.
_TERMINAL_TELL_RX = re.compile(
    r"^(?:please\s+)?(?:(?:tell|ask|get)\s+it\s+to|have\s+it|"
    r"(?:tell|ask|get)\s+(?:the\s+)?(?:terminal|session)\s+to)\s+"
    r"(?P<rest>.+)$", re.I)


def terminal_address(text: str) -> Optional[tuple]:
    """(prompt, project name) when the utterance addresses an open terminal,
    else None.  Pure text: it does not know whether one IS open."""
    for rx in (_TERMINAL_LEAD_RX, _TERMINAL_TAIL_RX):
        m = rx.match(text)
        if m:
            rest = (m.group("rest") or "").strip(" ,;:")
            if rest:
                return rest, (m.group("name") or "").strip()
    m = _TERMINAL_TELL_RX.match(text)
    if m:
        rest = (m.group("rest") or "").strip(" ,;:")
        if rest and (_VERB_RX.match(rest) or _CANCEL_RX.match(rest)):
            return rest, ""
    return None


_SKILL_RUN_RX = re.compile(
    r"^(?:run|use|invoke|start|launch)\s+(?:the\s+|a\s+)?(?P<name>[\w-]+)\s+skill"
    r"(?:\s+(?:on|with|for|against)\s+(?P<args>.+))?$", re.I)
_SLASH_RX = re.compile(r"^/[\w-]+(?:\s|$)")

# ------------------------------------------------------------ rule 4 tables
def _phrase_rx(words) -> re.Pattern:
    alts = sorted(words, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in alts) + r")\b",
                      re.I)


CODE_VERBS = (
    "fix", "implement", "refactor", "debug", "write", "add", "remove", "rename",
    "migrate", "deploy", "build", "test", "lint", "review", "commit", "push",
    "merge", "rebase", "install", "upgrade", "scaffold", "generate", "optimise",
    "optimize", "profile", "document", "create", "update", "change", "patch",
    "delete", "run", "rerun", "re-run", "rewrite", "clean up", "cleanup",
    "format", "type-check", "typecheck", "bump", "release", "ship", "extract",
    "move", "split", "wire", "wire up", "hook up", "port", "convert",
    "integrate", "configure", "set up", "setup", "check", "investigate",
    "look at", "look into", "inspect", "trace", "find", "restart", "start",
    "stop", "kill", "diff", "revert", "roll back", "rollback", "squash",
    "tag", "publish", "vendor", "pin", "unpin", "compile", "benchmark",
    "measure", "make", "edit", "open", "read", "grep", "search", "replace",
    "annotate", "type", "mock", "stub", "cover", "harden", "secure", "audit",
    "simplify", "modularise", "modularize", "parallelise", "parallelize",
    "speed up", "cache", "log", "instrument", "work on", "sort out", "take care of",
    "handle", "tidy", "tidy up",
)
CODE_OBJECTS = (
    "code", "codebase", "function", "functions", "method", "methods", "class",
    "classes", "module", "modules", "file", "files", "test", "tests",
    "unit test", "unit tests", "integration test", "integration tests",
    "test suite", "bug", "bugs", "feature", "features", "branch", "branches",
    "repo", "repos", "repository", "pr", "prs", "pull request",
    "pull requests", "script", "scripts", "error", "errors", "exception",
    "exceptions", "traceback", "stack trace", "stacktrace", "package",
    "packages", "dependency", "dependencies", "deps", "config",
    "configuration", "pipeline", "pipelines", "dataset", "datasets",
    "model training", "training run", "training script", "docker",
    "dockerfile", "container", "containers", "service", "services",
    "endpoint", "endpoints", "api", "database", "db", "migration",
    "migrations", "schema", "linter", "ci", "logs", "log file", "readme",
    "docs", "documentation", "docstring", "docstrings", "type hints",
    "typing", "variable", "variables", "loop", "regex", "query", "sql",
    "cli", "widget", "widgets", "router", "parser", "handler", "handlers",
    "callback", "callbacks", "thread", "threads", "daemon", "systemd",
    "cron", "cronjob", "venv", "virtualenv", "environment", "makefile",
    "commit", "commits", "changes", "diff", "patch", "refactor", "build",
    "compiler", "interpreter", "kernel", "driver", "library", "libraries",
    "framework", "plugin", "plugins", "skill", "mcp server", "server",
    "backend", "frontend", "ui code", "stylesheet", "css", "html",
    "javascript", "typescript", "python", "rust", "bash script",
    "shell script", "notebook", "unit", "fixture", "fixtures", "coverage",
    "benchmark", "profiler", "memory leak", "race condition", "deadlock",
    "segfault", "crash", "regression", "flaky test", "lint errors",
    "type errors", "warnings", "import", "imports", "argument", "arguments",
    "parameter", "parameters", "return value", "interface", "abstraction",
    "helper", "helpers", "utility", "utilities", "wrapper", "decorator",
    "generator", "iterator", "template", "templates", "boilerplate",
    # The spec's own worked example of an ambiguous utterance ("check the
    # weather tool") turns on this word: without it the coding pair
    # collapses and the local topic noun beside it wins.
    "tool", "tools",
)
_PATH_RX = re.compile(
    r"(?<![\w/])(?:~|\.{1,2}|/home|/tmp|/etc|/usr|/opt|/var)?/?[\w.~-]*"
    r"(?:/[\w.~-]+)*\.(?:py|js|ts|tsx|jsx|md|json|yaml|yml|toml|sh|rs|go|c|cc|"
    r"cpp|h|hpp|css|html|txt|cfg|ini|sql|ipynb|lock|env|service|desktop)\b"
    r"|(?<![\w/])(?:~|/home|/tmp|/etc|/usr|/opt|/var|\./)[\w./~-]+", re.I)
_QUESTION_CODE_RX = re.compile(
    r"^(?:what|how|why|where|which|when|does|do|is|are|can|could|should|explain|"
    r"describe|walk me through|show me)\b", re.I)
# Objects that make a question a question ABOUT CODE ("classes" is left out:
# "should I use focal loss for the rare classes" is ML talk, not code).
_QUESTION_OBJECT_RX = _phrase_rx((
    "code", "codebase", "function", "functions", "method", "methods", "class",
    "module", "modules", "file", "files", "error", "errors", "exception",
    "traceback", "stack trace", "test", "tests", "test suite", "script",
    "scripts", "bug", "bugs", "repo", "repository", "branch", "pr",
    "pull request", "commit", "commits", "diff", "config", "pipeline",
    "docker", "dockerfile", "dependency", "dependencies", "package",
    "traceback", "regression", "linter", "ci", "venv", "makefile",
    "docstring", "type hints", "unit test", "unit tests", "memory leak",
    "race condition", "deadlock", "segfault",
))
_FAILING_RX = re.compile(
    r"\bwhy (?:is|are|does|do|did|was|were)\b.*\b(?:fail(?:s|ing|ed)?|broken|"
    r"crash(?:es|ing|ed)?|hang(?:s|ing)?|slow|erroring|throwing|red)\b", re.I)
_STEPWISE_RX = re.compile(r"\bstep by step\b|\bthen\b.*\bthen\b", re.I)


_VERB_RX = _phrase_rx(CODE_VERBS)
_OBJECT_RX = _phrase_rx(CODE_OBJECTS)
# "explain this code" style verb phrases count as a full coding cue on their own.
_VERB_PHRASE_RX = re.compile(
    r"\b(?:explain|walk me through|describe|summari[sz]e|understand|read)\s+"
    r"(?:this|that|the|my|these|those)\s+(?:code|function|class|module|file|"
    r"error|traceback|stack trace|test|tests|script|bug|change|changes|diff|"
    r"repo|codebase|pr|pull request)\b", re.I)

# ------------------------------------------------------------ rule 5 tables
# Strong local forms: a question / imperative that names a local skill.
_LOCAL_STRONG = [
    ("weather", re.compile(
        r"\b(?:what(?:'s| is| will|s)?|how(?:'s| is)?|is it|will it|should i|do i need)\b.*"
        r"\b(?:weather|forecast|temperature|rain(?:ing|y)?|snow(?:ing)?|sunny|cloudy|"
        r"windy|humid(?:ity)?|umbrella|jacket|coat|hot|cold|warm|chilly|freezing|"
        r"degrees)\b"
        r"|\b(?:weather|forecast)\s+(?:for|in|at|this|today|tomorrow|tonight|"
        r"this week|next week)\b|^(?:the\s+)?(?:weather|forecast)\b", re.I)),
    ("clock", re.compile(
        r"\bwhat(?:'s| is|s)?\s+(?:the\s+)?(?:current\s+)?(?:time|date|day|year)\b|"
        r"\bwhat time\b|\btime is it\b|\bwhat day\b|\bwhat date\b|\btoday'?s date\b|"
        r"\bday of the week\b|\b(?:have you|do you have|you) got the time\b|"
        r"\bwhat'?s? the time\b|\bcurrent time\b|\bwhat year\b|\btime in\s+\w+", re.I)),
    ("calendar", re.compile(
        r"\b(?:what(?:'s| is|s| do i have)?|anything|any|do i have|is there|"
        r"when(?:'s| is)?|show me|read me|check|list|tell me)\b.*"
        r"\b(?:calendar|schedule|agenda|meetings?|appointments?|events?|"
        r"plans? (?:today|tomorrow|tonight|this week|for))\b"
        r"|\bon (?:my )?(?:calendar|schedule|agenda)\b|\bnext (?:meeting|appointment|event)\b|"
        r"\b(?:schedule|book|put)\b.*\b(?:meeting|appointment|on my calendar)\b", re.I)),
    ("timekeeper", re.compile(
        r"\b(?:remind me|set (?:a |an |the )?(?:reminder|timer|alarm)|"
        r"(?:start|put on|run) (?:a |the )?timer|timer for|wake me|"
        r"(?:set|create|add|schedule) (?:a |an )?(?:reminder|timer|alarm)|"
        r"alarm (?:for|at)|snooze|"
        r"(?:what|which|any|list|show|read|cancel|clear|delete|stop|dismiss)\b.*"
        r"\b(?:reminders?|timers?|alarms?)\b|"
        r"(?:reminders?|timers?|alarms?)\b.*\b(?:do i have|have i got|are set|set\?)|"
        r"how (?:long|much time) (?:is )?left)\b", re.I)),
    ("notes", re.compile(
        r"\b(?:take a note|note that|note down|jot down|make a note|write (?:this |that |it )?down|"
        r"(?:add|put)\b.*\b(?:to|on) my (?:to-?do|todo|list|shopping list)|"
        r"(?:add|create|new) (?:a )?(?:to-?do|todo|task)\b|"
        r"(?:show|read|list|what(?:'s| is|s)? (?:on|in)|check|open)\b.*\b(?:my )?(?:notes?|to-?dos?|todos?|"
        r"to-?do list|todo list|task list|shopping list)|"
        r"\bmy (?:notes|to-?dos|todos|to-?do list|todo list|task list)\b|"
        r"(?:mark|tick|check off|cross off)\b.*\b(?:done|off|complete)|"
        r"(?:remove|delete|clear)\b.*\b(?:note|to-?do|todo|from my list)|"
        r"\b(?:shopping|grocery|groceries|to-?do|todo|reading|packing) list\b)",
        re.I)),
    ("mail", re.compile(
        r"\b(?:(?:check|read|any|new|show|what(?:'s| is|s)? in|open|summari[sz]e|"
        r"go through)\b.*\b(?:e-?mails?|mail|inbox|messages)\b|"
        r"\b(?:e-?mails?|mail|inbox)\b.*\b(?:today|new|unread|from|waiting)\b|"
        r"\bunread\b|\bwho (?:e-?mailed|wrote|messaged) me\b|^(?:e-?mail|mail|inbox)$)",
        re.I)),
    ("briefing", re.compile(
        r"\b(?:briefing|brief me|headlines|(?:the |any |what'?s? (?:in |on )?the )?news\b|"
        r"hacker news|what(?:'s| is|s)? (?:going on|happening) (?:in|with) (?:tech|ai))",
        re.I)),
    ("location", re.compile(
        r"\bwhere am i\b|\bwhere are we\b|\bmy location\b|\bwhat city\b|"
        r"\bwhat(?:'s| is) (?:my|our) (?:location|address|city|town)\b|"
        r"\bwhich (?:city|town|state|country) (?:am i|are we)\b", re.I)),
    ("greeting", re.compile(
        r"^(?:hi|hello|hey|yo|hiya|howdy|good (?:morning|afternoon|evening|night)|"
        r"morning|evening|afternoon|how are you(?: doing| today)?|how'?s it going|"
        r"how are things|what'?s up|sup|are you (?:there|awake|up|around)|you there|"
        r"still there|thanks?(?: you)?(?: very much| a lot)?|cheers|ta|bye|goodbye|"
        r"see you(?: later)?|nice one|well done|good job|appreciate it|"
        r"much appreciated|sleep well|night)\b(?:[, ]+jarvis)?[!.?]*$", re.I)),
    ("joke", re.compile(
        r"\b(?:joke|jokes|something funny|make me laugh|riddle|pun|limerick|"
        r"tell me a story|fun fact|entertain me|cheer me up)\b", re.I)),
    # Media is the one local skill whose vocabulary is ordinary English:
    # "put on some jazz from the seventies please" carried no cue at all
    # and fell through to the paid tie-breaker.  The imperative may be
    # wrapped in a polite request ("could you play something upbeat"), and
    # a genre name counts as the object.  "play back" is excluded: playing
    # a log back is a coding sense of the same verb.
    ("music", re.compile(
        r"^(?:(?:hey |ok(?:ay)? )?(?:can|could|would|will) you\s+)?"
        r"(?:please\s+)?(?:just\s+)?"
        r"(?:play(?!\s+back\b)|put on|throw on|stick on|pop on|queue(?: up)?|"
        r"pause|resume|unpause|skip|next|previous|back|"
        r"shuffle|repeat|turn (?:it |the (?:music|volume) )?(?:up|down)|"
        r"volume(?: \d{1,3})?)\b(?!\w)"
        r"|\b(?:spotify|liked songs|now playing)\b"
        r"|\b(?:play(?!\s+back\b)|put on|throw on|stick on|pause|resume|skip|"
        r"stop|queue|shuffle|listen to)\b.*"
        r"\b(?:song|songs|track|tracks|album|albums|artist|artists|playlist|"
        r"playlists|music|radio|podcast|jazz|blues|classical|rock|soul|funk|"
        r"hip.?hop|rap|reggae|country|folk|metal|punk|disco|techno|opera|"
        r"motown|r&b|symphony|concerto)\b"
        r"|\bwhat(?:'s| is|s)? (?:playing|this song|this track)\b"
        r"|\b(?:like|save) this (?:song|track|one)\b"
        r"|\bsomething like this\b|\bturn (?:it|the music|the volume) "
        r"(?:up|down)\b|\bon my phone\b", re.I)),
    ("remember", re.compile(
        r"^(?:remember (?:that )?|what did i (?:say|tell you) about|do you remember|"
        r"recall )", re.I)),
]
# Utterances that start with these wrap free text: the tail can say anything.
_LOCAL_WRAPPER_RX = re.compile(
    r"^(?:remind me\b|set (?:a |an |the )?(?:reminder|timer|alarm)\b|"
    r"(?:start|put on) (?:a |the )?timer\b|timer for\b|wake me\b|"
    r"(?:take a note|note that|note down|jot down|make a note|write (?:this |that )?down)\b|"
    r"add\b.*\bto my (?:to-?do|todo|list|shopping list)\b|"
    r"(?:add|create|new) (?:a )?(?:to-?do|todo)\b|"
    r"remember (?:that )?|tell me a joke\b|"
    r"(?:schedule|book)\b.*\b(?:meeting|appointment)\b)", re.I)
# Weak local topic nouns (only matter when nothing strong is present).
_LOCAL_WEAK_RX = re.compile(
    r"\b(?:weather|forecast|temperature|rain|calendar|schedule|meeting|meetings|"
    r"appointment|appointments|reminder|reminders|timer|timers|alarm|alarms|"
    r"snooze|note|notes|to-?do|todo|todos|to-?dos|task list|e-?mail|e-?mails|"
    r"mail|inbox|briefing|news|headlines|location|joke|jokes|birthday|"
    r"anniversary|holiday|sunrise|sunset|moon|traffic|commute|recipe|"
    r"grocery|groceries|shopping|dinner|lunch|breakfast|coffee|workout|gym|"
    r"sleep|nap|wake|bedtime|music|song|songs|track|album|artist|playlist|"
    r"spotify|volume|radio|podcast)\b", re.I)
_QUESTION_RX = re.compile(
    r"^(?!(?:can|could|would|will|should) you\b)"
    r"(?:who|whom|whose|what|when|where|why|how|which|is|are|was|were|do|does|"
    r"did|can|could|should|would|will|am|have|has|had|tell me|define|"
    r"what'?s|who'?s|how'?s|where'?s|when'?s|why'?s|any idea|do you know|"
    r"remind me what|explain(?! (?:this|that|the|my) (?:code|function|class|"
    r"module|file|error|traceback|test|tests|script|bug|change|changes|diff|"
    r"repo|codebase|pr))|recommend|suggest|convert \d|calculate|compute|"
    r"translate|spell|pronounce|i wonder|i'm curious|curious)\b", re.I)


@dataclass
class Cues:
    code_strong: bool = False
    code_weak: bool = False
    code_verb: bool = False
    code_at: int = 10 ** 6
    local_strong: bool = False
    local_weak: bool = False
    local_kind: str = ""
    local_at: int = 10 ** 6
    wrapper: bool = False
    question: bool = False
    words: int = 0

    @property
    def any(self) -> bool:
        return (self.code_strong or self.code_weak or self.local_strong or
                self.local_weak or self.question)


def _first_nonoverlapping(a_spans, b_spans):
    """(a, b) positions of the earliest verb/object pair whose spans don't
    overlap ('test the alarm' has one token doing both jobs -> no pair)."""
    for a0, a1 in a_spans:
        for b0, b1 in b_spans:
            if b1 <= a0 or b0 >= a1:
                return min(a0, b0)
    return None


def code_cues(text: str) -> tuple[bool, bool, int]:
    """(strong, weak, position) coding cues for a normalised utterance."""
    verbs = [m.span() for m in _VERB_RX.finditer(text)]
    objs = [m.span() for m in _OBJECT_RX.finditer(text)]
    paths = [m.span() for m in _PATH_RX.finditer(text)]
    weak = bool(objs or paths)
    pos = None
    vp = _VERB_PHRASE_RX.search(text)
    if vp:
        pos = vp.start()
    if verbs and objs:
        p = _first_nonoverlapping(verbs, objs)
        if p is not None:
            pos = p if pos is None else min(pos, p)
    if verbs and paths:
        p = _first_nonoverlapping(verbs, paths)
        if p is not None:
            pos = p if pos is None else min(pos, p)
    if pos is None and _FAILING_RX.search(text) and (objs or paths or verbs):
        pos = 0
    if pos is None and (paths or _QUESTION_OBJECT_RX.search(text)) and \
            _QUESTION_CODE_RX.match(text):
        pos = 0                      # "what does this function do", "what's in router.py"
    if pos is None and verbs and _STEPWISE_RX.search(text):
        pos = verbs[0][0]
    return pos is not None, weak, (pos if pos is not None else 10 ** 6)


# A local topic noun used as a MODIFIER of a code object -- "the spotify
# tool", "the calendar module" -- names a piece of software, not an errand.
# The strong local reading is demoted to a weak topic so the coding pair
# beside it can win (spec 5.1's ambiguity example).
_CODE_HEAD_RX = re.compile(
    r"^\s+(?:tool|tools|module|modules|handler|handlers|integration|client|"
    r"api|wrapper|plugin|plugins|parser|script|scripts|class|function|"
    r"endpoint|service|code|codebase)\b", re.I)


def local_cues(text: str) -> tuple[bool, bool, str, int]:
    """(strong, weak, kind, position) local cues for a normalised utterance."""
    best = None
    for kind, rx in _LOCAL_STRONG:
        m = rx.search(text)
        if m and (best is None or m.start() < best[1]):
            best = (kind, m.start(), m.end())
    if best is not None and not _CODE_HEAD_RX.match(text[best[2]:]):
        return True, True, best[0], best[1]
    m = _LOCAL_WEAK_RX.search(text)
    if m:
        return False, True, "topic", m.start()
    if best is not None:                      # demoted: still a topic noun
        return False, True, "topic", best[1]
    return False, False, "", 10 ** 6


def analyse(text: str) -> Cues:
    c = Cues()
    c.words = len(text.split())
    c.code_strong, c.code_weak, c.code_at = code_cues(text)
    c.code_verb = bool(_VERB_RX.search(text))
    c.local_strong, c.local_weak, c.local_kind, c.local_at = local_cues(text)
    c.wrapper = bool(_LOCAL_WRAPPER_RX.match(text))
    c.question = bool(_QUESTION_RX.match(text))
    return c


# Git and glue nouns that do not make a task bigger on their own.
_SIZE_IGNORE = {"changes", "commit", "commits", "branch", "branches", "diff",
                "patch", "build", "import", "imports", "warnings", "readme",
                "docs", "documentation", "unit", "coverage", "refactor"}


def estimate_size(prompt: str) -> str:
    """'small' | 'large' — the fable escalation (7.1): two or more distinct
    coding objects, or refactor / feature / across / all the / migrate /
    rewrite / overhaul / 'debug ... and ...', or a long brief."""
    t = (prompt or "").lower()
    if not t:
        return "small"
    paths = [m.group(0) for m in _PATH_RX.finditer(t)]
    bare = _PATH_RX.sub(" ", t)
    objs = {m.group(0) for m in _OBJECT_RX.finditer(bare)} - _SIZE_IGNORE
    objs |= set(paths)
    if len(objs) >= 3:
        return "large"
    if len(objs) >= 2 and re.search(
            r"\b(?:and|plus|as well as|along with|also)\b|,", bare):
        return "large"
    if re.search(r"\b(?:refactor|feature|across|all the|all of the|every|entire|"
                 r"whole|migrate|migration|rewrite|overhaul|redesign|"
                 r"re-?architect|end to end|end-to-end|multi-?file|throughout)\b", t):
        return "large"
    if re.search(r"\bdebug\b.*\band\b", t):
        return "large"
    if len(t.split()) >= 40:
        return "large"
    return "small"


# ------------------------------------------------------------------ router
class Router:
    """Pure decision maker; see the module docstring for the rule order."""

    def __init__(self, cfg=None, classify: Optional[Callable] = None,
                 now: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.classify = classify
        self._now = now
        self._pending: Optional[PendingAsk] = None
        self._skill_cache: tuple = ()

    # -- config -------------------------------------------------------
    def _skill_phrases(self) -> list[tuple[re.Pattern, str]]:
        phrases = None
        cfg = self.cfg
        if cfg is not None:
            try:
                phrases = getattr(cfg, "skill_phrases", None)
                if callable(phrases):
                    phrases = phrases()
                if not isinstance(phrases, dict) and hasattr(cfg, "get"):
                    phrases = cfg.get("claude.skill_phrases")
            except Exception:
                log.exception("skill_phrases unreadable; using defaults")
                phrases = None
        if not isinstance(phrases, dict) or not phrases:
            phrases = DEFAULT_SKILL_PHRASES
        key = tuple(sorted((str(k), str(v)) for k, v in phrases.items()))
        if self._skill_cache and self._skill_cache[0] == key:
            return self._skill_cache[1]
        compiled = []
        for pat, tpl in key:
            try:
                compiled.append((re.compile(pat, re.I), tpl))
            except re.error:
                log.warning("skill phrase %r is not a valid regex; skipped", pat)
        self._skill_cache = (key, compiled)
        return compiled

    # -- pending question --------------------------------------------
    def pending(self) -> Optional[PendingAsk]:
        p = self._pending
        if p is None:
            return None
        if self._now() - p.at > ASK_TTL_S:
            self._pending = None
            return None
        return p

    def clear_pending(self) -> None:
        self._pending = None

    def resolve_answer(self, text: str) -> Optional[str]:
        """'claude' | 'local' for a yes/no-style answer to a pending
        question (clears it); None when nothing is pending or the text is
        not an answer (the question is then dropped by the caller)."""
        if self.pending() is None:
            return None
        kind = answer_kind(text)
        if kind is None:
            return None
        self._pending = None
        return kind

    # -- routing ------------------------------------------------------
    def route(self, text: str, active_project: Optional[str] = None,
              terminal_open: Optional[Callable] = None) -> RouteDecision:
        raw = normalise(text)
        if not raw:
            return RouteDecision("local", "empty")
        args: dict = {}
        work = raw

        # Cross-cutting modifiers, extracted before any rule so that
        # "refactor the parser, use fable, in parallel" still hits rule 4.
        hit = _modifier_match(_MODEL_RX, work)
        if hit:
            m, span = hit
            alias = (m.group("alias") or BIG_MODEL).lower()
            args["model"] = alias
            work = _strip_span(work, span)
        hit = _modifier_match(_PARALLEL_RX, work, "parallel")
        if hit:
            args["parallel"] = True
            work = _strip_span(work, hit[1])
        # 1. session actions (before the project modifier is stripped:
        #    "work on the jarvis project" IS the project) ---------------
        d = self._action(work.strip(" ,;:"), args, "", raw)
        if d is not None:
            return d
        # 1b. an open terminal is a live session the user can address ----
        d = self._terminal(work.strip(" ,;:"), args, active_project,
                           terminal_open)
        if d is not None:
            return d
        project = ""
        m = _PROJECT_RX.search(work)
        if m:
            project = slugify(m.group("name"))
            work = _strip_span(work, m.span())
        explicit = False
        m = _CLAUDE_CUE_RX.search(work)
        if m:
            explicit = True
            work = _strip_span(work, m.span())
        work = work.strip(" ,;:")
        d = self._action(work, args, project, raw)      # "claude, cancel that"
        if d is not None:
            return d
        if not work:
            if "model" in args:
                return RouteDecision("action", "set_model", action="set_model",
                                     args={"alias": args["model"]})
            if explicit:
                return RouteDecision("local", "claude-cue-empty")
            return RouteDecision("local", "empty")

        # 2. skill phrases --------------------------------------------
        prompt = self._skill(work)
        if prompt is not None:
            args.setdefault("size", estimate_size(prompt))
            return RouteDecision("claude", "skill", prompt=prompt,
                                 project=project, args=args)

        # 3. explicit cue ---------------------------------------------
        if explicit:
            args.setdefault("size", estimate_size(work))
            return RouteDecision("claude", "explicit", prompt=work,
                                 project=project, args=args)

        cues = analyse(work)
        # 4/5. cue tables (strong beats weak; wrappers are local) -------
        if cues.wrapper:
            return RouteDecision("local", f"local:{cues.local_kind or 'wrapper'}")
        if cues.code_strong and not cues.local_strong:
            args.setdefault("size", estimate_size(work))
            return RouteDecision("claude", "code-cue", prompt=work,
                                 project=project, args=args)
        if cues.local_strong and not cues.code_strong:
            return RouteDecision("local", f"local:{cues.local_kind}")
        if not cues.code_strong and not cues.local_strong:
            if cues.question and not cues.code_weak:
                return RouteDecision("local", "local:question")
            if cues.local_weak and not cues.code_weak:
                return RouteDecision("local", "local:topic")
            if project and cues.code_weak:
                args.setdefault("size", estimate_size(work))
                return RouteDecision("claude", "project+object", prompt=work,
                                     project=project, args=args)
            # 7. length prior: nothing to go on and short -> local ------
            if not cues.any and cues.words <= SHORT_WORDS:
                return RouteDecision("local", "short")
        # 6. neither or both -> the model, else ask ---------------------
        return self._tie_break(work, project, args, active_project, cues)

    # -- rule 1 ---------------------------------------------------------
    def _action(self, work: str, args: dict, project: str, raw: str) -> Optional[RouteDecision]:
        t = work.strip()
        if not t:
            return None
        if _CANCEL_RX.match(t):
            return RouteDecision("action", "cancel", action="cancel", args={})
        m = _FAST_RX.match(t)
        if m:
            on = bool(m.group("on") and m.group("on").lower() == "on") or bool(m.group("en"))
            return RouteDecision("action", "fast_mode", action="set_fast_mode",
                                 args={"on": on})
        m = _NEW_PROJECT_RX.match(t)
        if m:
            name = (m.group("name") or m.group("name2") or "").strip(" .\"'")
            return RouteDecision("action", "new_project", action="new_project",
                                 args={"name": name}, project=slugify(name))
        m = _RESUME_RX.match(t)
        if m:
            name = (m.group("n1") or m.group("n2") or m.group("n3") or "").strip()
            when = (m.group("w1") or m.group("w2") or m.group("w3") or
                    m.group("w4") or "").lower()
            if name and re.match(r"^(?:where we left off|reading|the reading)$", name, re.I):
                name = ""
            a = {"utterance": raw, "when": when, "name": name}
            return RouteDecision("action", "resume", action="resume", args=a,
                                 project=slugify(name) if name else project)
        m = _WORK_ON_STRICT_RX.match(t)
        if m:
            name = (m.group("n1") or m.group("n2") or "").strip(" .\"'")
            # "work on the login bug" is a task, not a project switch.
            if name and not _OBJECT_RX.search(name) and not _PATH_RX.search(name) \
                    and len(name.split()) <= 4 and not _VERB_RX.match(name):
                return RouteDecision("action", "work_on", action="work_on",
                                     args={"name": name}, project=slugify(name))
        if "model" in args and _MODEL_FILLER_RX.fullmatch(t):
            return RouteDecision("action", "set_model", action="set_model",
                                 args={"alias": args["model"]})
        return None

    # -- rule 1b --------------------------------------------------------
    def _terminal(self, work: str, args: dict, active_project,
                  terminal_open) -> Optional[RouteDecision]:
        """"in the terminal, do X" / "tell it to X" -> that project's Claude
        session.  Only when a terminal really is attached: `terminal_open`
        is ClaudeSessionManager.terminal_open (cheap, non-blocking).  With
        no callable, or none open, the utterance routes as it always did."""
        if not callable(terminal_open) or not work:
            return None
        hit = terminal_address(work)
        if hit is None:
            return None
        prompt, name = hit
        project = slugify(name) if name else ""
        # "in the terminal, cancel that" is still Jarvis's cancel: queuing
        # those words as a new prompt would sit BEHIND the very task the
        # user is trying to stop.  A session action wins over the address.
        act = self._action(prompt, args, project, prompt)
        if act is not None:
            return act
        try:
            if not terminal_open(project or active_project or None):
                return None
        except Exception:
            log.exception("terminal_open failed; routing normally")
            return None
        a = dict(args)
        a["terminal"] = True
        a.setdefault("size", estimate_size(prompt))
        return RouteDecision("claude", "terminal", prompt=prompt,
                             project=project, args=a)

    # -- rule 2 ---------------------------------------------------------
    def _skill(self, work: str) -> Optional[str]:
        t = work.strip()
        if _SLASH_RX.match(t):
            return t                                  # typed slash command passthrough
        for rx, tpl in self._skill_phrases():
            m = rx.search(t)
            if m:
                return _expand_template(tpl, m)
        m = _SKILL_RUN_RX.match(t)
        if m:
            name = m.group("name").lower()
            a = (m.group("args") or "").strip()
            return f"/{name} {a}".strip()
        return None

    # -- rule 6 ---------------------------------------------------------
    def _tie_break(self, work, project, args, active_project, cues) -> RouteDecision:
        route, conf = "local", 0.0
        if self.classify is not None:
            try:
                out = self.classify(work)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    route, conf = str(out[0]).lower(), float(out[1])
                elif isinstance(out, str):
                    route, conf = out.lower(), 1.0
            except Exception:
                log.exception("classify failed; asking")
                route, conf = "local", 0.0
        if route not in ("local", "claude"):
            route, conf = "local", 0.0
        if route == "claude":
            # A silent dispatch here spends money, so it takes a much higher
            # bar AND a coding cue somewhere in the sentence.  Everything
            # else asks -- the question is free and resolves on the next
            # utterance.
            if conf >= CLASSIFY_CLAUDE_THRESHOLD and \
                    (cues.code_weak or cues.code_strong or cues.code_verb):
                args.setdefault("size", estimate_size(work))
                return RouteDecision("claude", "classify", prompt=work,
                                     project=project, args=args, confidence=conf)
        elif conf >= CLASSIFY_THRESHOLD:
            return RouteDecision("local", "classify", confidence=conf)
        self._pending = PendingAsk(text=work, project=project or (active_project or ""),
                                   at=self._now(), args=dict(args))
        if self.classify is None:
            reason = "no-classifier"
        elif route == "claude":
            reason = "classify-claude-unconfirmed"
        else:
            reason = "classify-low"
        return RouteDecision("ask", reason, prompt=work, project=project,
                             args=args, confidence=conf)


# ------------------------------------------------------------ answers
_ANSWER_CLAUDE_RX = re.compile(
    r"^(?:yes|yeah|yep|yup|aye|claude|have claude|hand it (?:over|to claude)|"
    r"give it to claude|go ahead|do it|sure|please do|send it(?: over)?|"
    r"claude please|the big guy|hand it off|pass it on|yes claude|yes please)"
    r"(?:[, ]+(?:please|sir|jarvis|claude|do it))*[.!]*$", re.I)
_ANSWER_LOCAL_RX = re.compile(
    r"^(?:(?:no|nope|nah)[, ]+)?"
    r"(?:no|nope|nah|you|just you|quick|quick one|it'?s a quick one|"
    r"answer it(?: yourself)?|you (?:do|answer|handle|take) it|you answer|"
    r"local|yourself|you can(?: do it)?|you've got it|you got it|no you|"
    r"no just you|keep it (?:local|here|in house)|do it yourself|"
    r"don'?t bother claude|no need for claude)"
    r"(?:[, ]+(?:please|jarvis|thanks|sir))*[.!]*$", re.I)


def answer_kind(text: str) -> Optional[str]:
    t = normalise(text).lower()
    if _ANSWER_CLAUDE_RX.match(t):
        return "claude"
    if _ANSWER_LOCAL_RX.match(t):
        return "local"
    return None
