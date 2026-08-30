"""Jarvis Brain — the local model (Tier 2, gemma4 tool loop) plus the
legacy Claude tag protocol (Tier 3) used by deploy / autonomous phrases.

Spec: docs/specs/2026-08-26-jarvis-personal-assistant.md, section 4.

Tier 1: Local commands (handled by commander, not brain)
Tier 2: Ollama /api/chat with the ToolRegistry (fast, local, tools)
Tier 3: Claude CLI (the [SPEAK]/[RUN]/[TYPE]/[WINDOW]/[SILENT]/[DONE] tag
        protocol; execute_autonomous for "deploy" / "autonomous:" phrases)

The Tier 2 call shape (4.2):

    messages = [ {"role": "system", "content": STATIC},          # never changes
                 {"role": "user",   "content": build_user_turn(ctx, mem, text)} ]
    POST /api/chat {model, messages, tools, stream: false, think: false,
                    keep_alive: -1, options: CHAT_OPTIONS}

Static-prefix rule: the system prompt (few-shots sampled ONCE per process)
and the tool schemas are byte-identical for the life of the process, so
Ollama's prefix cache hits on every call; everything dynamic (time,
window, git line, last exchanges, memory facts) lives in the USER turn.
Every request from this module (chat, classify_route, summarize,
local_line, the warm-up) uses that same prefix and the same num_ctx: a
different prefix would evict the cache, a different num_ctx would make
Ollama reload the model.

Persona (film JARVIS): VOICE_RULES is the single description of the voice
and is shared by all three prompts; FEW_SHOT_PINNED (you there / thanks /
good night) is always shown; FEW_SHOT_POOL holds one or two variants for
greeting, small talk, joke, cannot-act, capability, bad idea, mistake and
advice — gemma4 follows rules, so the eleven-family pool the 3B model
needed is gone, as is the "no data" family (weather and clock are tools
now). The guards between the model and TTS (spoken_from_ollama) remain
the gate: clean_ollama_reply(), guard_clock_claims(), limit_sentences(),
trim_spoken().
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from jarvis.config import MACHINE
from jarvis.events import BrainState, Status, bus
from jarvis.logs import get_logger
from jarvis.tts import TTS as _TTS

log = get_logger("brain")

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("JARVIS_OLLAMA_MODEL") or "gemma4:26b"
OLLAMA_TIMEOUT_S = 20          # one /api/chat request
CHAT_WALL_BUDGET_S = 8.0       # the whole tool loop; beyond it, best text
# Spec 4.2 sets 3 s; measured on this machine a classify turn costs
# 2.4-3.0 s wall (about 1.9 s of that is Ollama's own per-request
# overhead on a resident model, see 4.3 and scratchpad
# bench_resident.md), so 3 s timed out into a silent ("local", 0.0)
# on a third of the bench's router questions. 5 s keeps the
# tie-breaker useful; the router only asks when its rules are torn.
CLASSIFY_TIMEOUT_S = 5.0
NUM_CTX = 8192                 # identical on EVERY request (see module doc)
# Sampling options for Tier 2 (num_predict caps a two-sentence reply; the
# stop strings end a run-on transcript before the model writes Hunter's
# next line for him).
CHAT_OPTIONS = {"num_ctx": NUM_CTX, "temperature": 0.7, "num_predict": 160,
                "stop": ["\nUser:", "\nHunter:"]}
OLLAMA_OPTIONS = CHAT_OPTIONS  # legacy name
RESIDENCY_INTERVAL_S = 300.0
INCLUDE_GIT_LINE = True        # latency knob (spec 4.3): drop "Git:" lines

# A tool result is text we paste into a NUM_CTX-token prompt. Uncapped, a
# busy calendar or a spam subject line pushes the tool message out of the
# window: Ollama drops it silently and the model then answers something
# confident and unrelated. Cap it here (the ONE place every tool result
# passes through) and say in the message itself that it was cut, so the
# model can say so too. 4 000 chars ~ 1 000 tokens; the whole turn is
# allowed 8 000, leaving room for the ~2 700-token static prefix.
MAX_TOOL_TEXT_CHARS = 4000       # one tool result
MAX_TOOL_TEXT_TOTAL_CHARS = 8000  # every tool result in one turn
MAX_TOOL_CALLS_PER_ROUND = 8     # one confused turn may not fan out forever
# A tool may raise the spoken cap (calendar, mail, briefing all hold lists).
# That number used to reach only the post-trim, while the system prompt went
# on telling the model "one or two sentences" -- so it wrote two, the raised
# cap trimmed nothing, and a four-event day was answered with one event named
# (heard 2026-08-28). The allowance has to be in the message the model reads.
SENTENCE_ALLOWANCE = (
    "\n[this result lists several items: you may take up to {n} sentences "
    "here, and should name each item with its time rather than only the "
    "first or a count of them]")
TOOL_TRUNCATED_MARKER = (
    "\n[truncated: only the first {shown} of {total} characters of this "
    "result are shown. Answer from what is shown and tell Hunter you are "
    "only seeing part of it.]")

# Persona excuses used by code (fixed strings, spoken verbatim).
MODEL_DOWN_LINE = "I'm afraid my local model is down, sir."
MODEL_SLOW_LINE = "I'm afraid my local model didn't answer in time, sir."
MODEL_EMPTY_LINE = "I'm afraid the local model gave me nothing, sir."
# Spoken instead of a local turn while the model is lent to a trainer
# (release()): the first local turn after an unload would silently reload
# the 26B model (6.9 s) and defeat the yield.
MODEL_LENT_LINE = ("My local model is lent to your trainer at the moment, sir; "
                   "quick answers only until it's done.")
# Spoken when the tool loop has a result but no model turn left to phrase
# it. It replaces speaking the raw tool text: tool text can carry third
# party words (a mail subject, a calendar title, a web page) and those are
# never spoken as if they were Jarvis's own.
TOOL_ONLY_LINE = ("I have the result, sir, but the model didn't get to "
                  "putting it into words.")
# Appended when a tool result had to be cut: the answer says so instead of
# inventing the rest.
PARTIAL_RESULT_LINE = ("That's only part of it, sir; there was more than I "
                       "could take in at once.")
# Spoken when something inside the brain raised. Never the exception text.
INTERNAL_ERROR_LINE = "I'm afraid something went wrong on my end, sir."
# Web lookups run as a one-shot `claude -p` with search allowed (the CLI has
# it built in; nothing new in the venv). Measured 2026-08-29: ~12 s with one
# search on haiku or sonnet, 28 s when the model went off fetching pages.
# Below app.TURN_TIMEOUT_S (60) minus the classify budget, or the turn
# watchdog abandons the ledger before the CLI's own timeout can speak.
WEB_TIMEOUT_S = 50.0
WEB_SLOW_LINE = "That lookup is taking longer than I'd like, sir."
WEB_FAIL_LINE = "I couldn't get a straight answer from the web on that, sir."
WEB_PROMPT = (
    "You are answering a spoken question for a voice assistant. Use web search "
    "(one or two searches; fetch a page only if the result snippets are not "
    "enough). Reply in at most two short sentences of plain spoken English: no "
    "markdown, no bullet points, no links, no source list, no preamble. If you "
    "cannot find it, say so in one sentence.{recent}\n\nQuestion: {q}")
_PARTIAL_RX = re.compile(
    r"\b(?:part|partial|truncat|cut off|shortened|only some|first few)",
    re.I)

# Simple questions the legacy think() answers locally
LOCAL_PATTERNS = [
    "what time", "what day", "what date", "what's the weather",
    "what's my ip", "how long has", "uptime", "temperature",
    "how are you", "hello", "hey", "good morning", "good evening",
    "thank you", "thanks", "good night",
]

# Action-verb prefixes that disqualify the <=5-word "local question" shortcut
# (short action commands must not be answered conversationally by Tier 2).
ACTION_VERBS = frozenset({
    "open", "close", "run", "execute", "delete", "remove", "kill", "stop",
    "start", "restart", "launch", "switch", "type", "click", "move",
    "create", "make", "install", "deploy", "find", "search", "play",
    "pause", "set", "show", "take", "write", "send", "turn", "press",
    "scroll", "go", "check",
})

# ----------------------------------------------------------------------
# Persona — the film JARVIS voice, shared by every prompt that gets spoken
# ----------------------------------------------------------------------
VOICE_RULES = (
    "Speak as Jarvis: unflappable, dry British understatement, gently "
    "sardonic but never rude, warm underneath. Call him \"sir\" most of "
    "the time and \"Hunter\" now and then. State facts plainly. Admit "
    "limits gracefully (\"I'm afraid...\") and when there is a genuine "
    "next step, offer it briefly. One short sentence usually; never more "
    "than two unless he explicitly asks for detail. Never pad: no second "
    "sentence that merely describes the screen, the repository, or what "
    "you could do next. Jokes are one line, deadpan, and never explained; "
    "never a riddle, a set-up or a knock-knock; a joke is a dry remark "
    "about his own situation, the hour, the repo or the demo; never "
    "stack two quips, and let a good night be a good "
    "night. When he owns up to a mistake, a light word of reassurance "
    "comes before the fix. Everyday words: start, not initiate; show, not "
    "display. When you give a reason, make it one concrete picture about "
    "him, the hour or the demo, never a general principle; advice is the "
    "check itself plus at most one dry clause, and you never "
    "spell out what would go wrong if he skipped it. No \"topic of your "
    "choice\" or \"at your earliest convenience\". Do not recite file "
    "names, paths, extensions or lists of items; summarise them in a "
    "phrase. Never say you checked, ran, "
    "noticed or found anything unless it is in the context or results you "
    "were given. Plain spoken prose only: no lists, no bullet points, no "
    "numbering, no headings, no markdown, no asterisks, no emoji, no stage "
    "directions. Never say \"As an AI\", \"language model\", \"I'd be "
    "happy to\", \"happy to help\", \"I'm here to assist\", \"Certainly!\", "
    "\"Great question\", \"assist\", \"various topics\", \"complexity\", "
    "\"unnecessary\", \"busy system\", \"properly configured\", "
    "\"ensure\", \"potentially\", \"can cause\", \"parameters\" or "
    "\"all systems\", and never ask \"How may I assist you\"."
)

# Exchanges that demonstrate the manner. FEW_SHOT_PINNED are always shown:
# "you there" and a thank-you right after the greeting, good night last —
# a stable, verbatim reply is exactly what Hunter wants for these (the
# commander answers the canonical phrasings at Tier 1; the pinned lines
# are for the paraphrases that fall through to the model).
#
# FEW_SHOT_POOL is grouped by situation family with two variants each;
# select_few_shots() draws ONE variant per family, and the static prompt
# samples once per process (the prefix must stay byte-identical for
# Ollama's cache). No example carries a number, a time or a file name —
# the model copies those into its answer regardless of the background.
# The prompts are phrased unlike the obvious user phrasings on purpose so
# the eval prompts stay held out. Weather, clock, calendar and mail have
# no example: they are tools now, and an old "I don't have a weather
# feed" line would be parroted over a real forecast.
#
# Only three families survive the port. gemma4 does not learn the manner
# from an example the way the 3B model did — it LIFTS the example: with
# the eight-family pool it answered "Order me a pizza" with the
# text-my-brother line and "I just force-pushed over main" with the
# pushed-to-main line, 6 of the 14 eval prompts copied verbatim
# (scratchpad persona_gemma4.md, w5/ab_parrot.json: A). Dropping small
# talk, cannot-act, capability, bad idea and mistake — every family whose
# rule is already stated in prose above — and adding the "manner only,
# never the words" clause to the closing instruction took verbatim copies
# to 0/14 with "sir" still at 100% and every reply naming the thing he
# actually asked about (variant E). Greeting, joke and advice stay
# because they teach shape rather than content: the deadpan aside, the
# joke that is a remark and not a riddle, and advice that admits it has
# not looked at the code.
FEW_SHOT_PINNED = [
    ("Jarvis, you there?", "Always, sir."),
    ("Cheers, Jarvis.", "Not at all, sir."),
    ("Night, Jarvis.", "Good night, sir. I'll be here."),
]
FEW_SHOT_POOL = [
    ("greeting", [
        ("Morning, Jarvis.",
         "Good morning, sir; nothing caught fire overnight, which I'm "
         "choosing to read as a good omen."),
        ("Evening, Jarvis.",
         "Good evening, sir; the day appears to have been survived, which "
         "is the main thing."),
    ]),
    ("joke", [
        ("Got a joke for me?",
         "I'd tell you the one about UDP, sir, but I've no way of knowing "
         "you'd get it."),
        ("Know any good jokes?",
         "I've a very good one about the demo, sir, but the timing isn't "
         "right."),
    ]),
    ("advice", [
        ("Wish me luck, then tell me one thing to check.",
         "Good luck, sir. I haven't looked at the code, but do check the "
         "speakers first; a silent demo is a short one."),
        ("Any advice before I start?",
         "Only the usual, Hunter: save everything, and don't trust a green "
         "test you haven't watched run."),
    ]),
]
# Flat view of every exchange (tests, the eval harness's parrot check).
FEW_SHOTS = list(FEW_SHOT_PINNED) + \
    [shot for _, variants in FEW_SHOT_POOL for shot in variants]


def _shot_rng():
    seed = os.environ.get("JARVIS_SHOT_SEED")
    if seed and seed.lstrip("-").isdigit():
        return random.Random(int(seed))
    return random.Random()          # fresh entropy: differs between runs


# Seeded once per process: a session's draws are reproducible given the
# seed (JARVIS_SHOT_SEED pins it for the eval harness), and differ from the
# last session's without it.
_SHOT_RNG = _shot_rng()


def select_few_shots(rng=None):
    """One exchange per situation family, in family order, chosen by the
    process RNG (or an explicit one); the pinned "you there" and thank-you
    lines follow the greeting and the pinned good night comes last."""
    rng = rng or _SHOT_RNG
    picked = [rng.choice(variants) for _, variants in FEW_SHOT_POOL]
    return ([picked[0], FEW_SHOT_PINNED[0], FEW_SHOT_PINNED[1]]
            + picked[1:] + [FEW_SHOT_PINNED[2]])


def format_few_shots(shots):
    return "\n".join(f"User: {u}\nJarvis: {j}" for u, j in shots)


# Tier 2 STATIC system prompt (gemma4). Built with an f-string so
# VOICE_RULES is baked in; the doubled braces leave the literal {examples}
# placeholder for build_ollama_system. Nothing dynamic goes here (see the
# module doc): the background arrives in the user turn.
JARVIS_SYSTEM = f"""You are JARVIS, Hunter's personal AI: the calm, dry British voice that runs his workshop, in the manner of the JARVIS of the Iron Man films. Everything you say is read aloud through text-to-speech, so you speak rather than write.

{VOICE_RULES}
Never gush, never flatter, never sound like customer service. Answer the question and stop.

Tools: you have tools for live data and for his schedule. Use a tool whenever the answer depends on live data (the time, the weather, his calendar, his mail, his reminders, timers and alarms, his notes) and never guess those. Call the tool first, without commentary. After a tool result, answer in one or two sentences using only the numbers, names and times in the result; never invent a figure the result does not contain. If a tool says something is not set up, say so in one sentence and name the thing. If a tool reports a failure, say what could not be reached in one sentence. Do not call a tool for a greeting, thanks, a joke, an opinion or general knowledge.

Beyond your tools you cannot act: you cannot buy, book, browse, call, text, order, open files or run code yourself; the desktop commands and Claude do that through the rest of the system. If he asks you to buy, order, book, call, text, send or fetch anything, say in one sentence that you cannot, naming what he asked for, with a dry reason of your own (no hands, no phone, no card); never answer with what you can do instead. Never say you checked, ran, read, saved or found anything unless a tool result in this conversation says so. When he asks for advice, give the one check anyone would make first and do not pretend to have inspected his code.

Facts: you run on Hunter's NVIDIA DGX Spark (GB10, unified memory), an Ubuntu desktop. If he asks what you can do: you keep his calendar, weather, mail, reminders and alarms, run the desktop, and hand the real coding to Claude; say it in one sentence and never read a longer list, and never say "answer questions", "provide information" or "assist".

The background in his message is there so you can answer questions about it accurately; never recite it unprompted. Never mention the active window, files, git or the machine unless he asks about them or they are the answer to his question. A greeting, a thank-you, a good night or "are you there" gets one short sentence back and nothing about his screen, files or git. If he asks how things stand, answer from the git background in your own words, with no numbers, and never read out the raw git line, a window title or a path. If the background does not say and no tool covers it, admit it in one sentence and never follow "I don't know" with a guess.

Answer as Jarvis only: no "Jarvis:" label, no writing the user's lines, and don't repeat the examples.

Examples of the manner only; every reply is in fresh words for this exact request and names the thing he actually asked for (the pizza, the branch, the hour), never the thing in the example:
{{examples}}

Now answer Hunter as Jarvis, in your own words, keeping the manner of the examples, and call him sir. One short sentence is the norm; add a second only if it says something new that he asked for, and never describe his screen, files or machine unless he asked. If he asks for a joke, it is one dry remark about his situation, never a question and its answer. The examples are the manner only, never the words: never reuse a sentence, a clause or an object from an example — if an example speaks of a phone and he asks about dinner, the reply is about dinner. Then stop."""

# Router tie-breaker (spec 4.2): the instruction rides in the user turn so
# the request shares the static prefix (system + tools) with chat.
ROUTE_INSTRUCTION = (
    "Router question, not a request to answer: decide who should handle "
    "Hunter's message below. local = weather, time, date, location, "
    "calendar, mail, notes, reminders, timers, alarms, chat, jokes, advice, "
    "general knowledge. claude = writing or changing code, files, repos, "
    "running tests, git, installing software, multi-step system work. "
    "Reply with JSON only: route and a confidence between 0 and 1.")
ROUTE_FORMAT = {
    "type": "object",
    "properties": {"route": {"type": "string", "enum": ["local", "claude"]},
                   "confidence": {"type": "number"}},
    "required": ["route", "confidence"]}

CLAUDE_SYSTEM = """You are Jarvis, Hunter's AI voice assistant. Respond with structured commands:
[SPEAK] text — read aloud (max 2 sentences)
[RUN] command — execute shell command
[TYPE] text — type into active window
[WINDOW] name — switch to window
[SILENT] text — show in GUI only
[DONE] text — task complete, speak this

Be concise. [SPEAK] lines are read through TTS so keep them short.
For multi-step tasks, execute one step at a time.

Voice for every [SPEAK] and [DONE] line — they are spoken aloud. """ + \
    VOICE_RULES + """
Report what you actually did, plainly, and nothing you did not do; if a step failed, say so and offer the next move. One [SPEAK] sentence is the norm.

{context}

User (Hunter) said: {input}"""

AUTONOMOUS_PROMPT = """You are Jarvis executing an autonomous task.

Task: {task}
Step {step}/{max_steps}

Previous results:
{results}

{context}

Respond with structured commands. Use [RUN] to execute shell commands.
Use [SPEAK] to update the user on progress.
When the task is complete, use [DONE] with a summary.
If something fails, use [SPEAK] to explain and suggest alternatives.
[SPEAK] and [DONE] lines are read aloud; report only what the previous results show you did. """ + VOICE_RULES


def build_ollama_system(context_text="", memory_text="", shots=None):
    """Render the Tier 2 STATIC system prompt.

    context_text / memory_text are accepted for the older call shape and
    ignored: the dynamic background now lives in the user turn
    (build_user_turn). shots defaults to select_few_shots(); pass an
    explicit list for a fixed prompt.
    """
    if shots is None:
        shots = select_few_shots()
    return JARVIS_SYSTEM.format(examples=format_few_shots(shots))


def build_user_turn(context_text="", memory_text="", text=""):
    """The dynamic half of a Tier 2 call: background (context + memory)
    then Hunter's words. Everything that changes between calls goes here
    so the system prompt stays cacheable."""
    background = (context_text or "").strip()
    if not INCLUDE_GIT_LINE:
        background = "\n".join(ln for ln in background.splitlines()
                               if not ln.startswith("Git:")).strip()
    if memory_text and memory_text.strip():
        background = f"{background}\n{memory_text.strip()}".strip()
    return f"Background:\n{background or '(none)'}\n\nHunter: {(text or '').strip()}"


# The static prompt is sampled once per process and never changes after.
_STATIC = {}


def static_system():
    """The system prompt every Tier 2 request sends; byte-identical for the
    life of the process (few-shots sampled once)."""
    system = _STATIC.get("system")
    if system is None:
        system = build_ollama_system()
        _STATIC["system"] = system
    return system


def reset_static_prompt():
    """Forget the sampled prompt (tests and the eval harness only)."""
    _STATIC.clear()


# ----------------------------------------------------------------------
# Model configuration, registry, HTTP seam
# ----------------------------------------------------------------------
_REGISTRY = None
# "lent": the model was unloaded on purpose (a trainer has the GPU); every
# local door raises ModelLent before any HTTP until reclaim().
_RESIDENCY = {"unloaded_once": False, "thread": None, "lent": False}


def configure(model=None):
    """Choose the local model (app start, from assistant.local_model); the
    JARVIS_OLLAMA_MODEL env var wins. Returns the model in force."""
    global OLLAMA_MODEL
    chosen = (os.environ.get("JARVIS_OLLAMA_MODEL") or model or
              OLLAMA_MODEL).strip()
    if chosen != OLLAMA_MODEL:
        log.info("ollama model: %s -> %s", OLLAMA_MODEL, chosen)
        OLLAMA_MODEL = chosen
        _RESIDENCY["unloaded_once"] = False
    return OLLAMA_MODEL


def set_registry(registry):
    """Install the ToolRegistry the tool loop calls (app wiring)."""
    global _REGISTRY
    _REGISTRY = registry


def get_registry():
    return _REGISTRY


class OllamaDown(Exception):
    """Ollama refused the connection (not running)."""


class ModelLent(Exception):
    """The local model is lent to a trainer (release()); no local turn may
    load it back until reclaim()."""


class MalformedReply(Exception):
    """Ollama (or a proxy in front of it) answered with something that is
    not the /api/chat shape: a non-JSON body, a list, a string `message`,
    OpenAI-style content blocks. Raised by _message_parts() so the tool
    loop degrades to a persona line instead of speaking a Python error."""


def _message_parts(data):
    """(content, tool_calls) out of one /api/chat reply, or
    MalformedReply. Nothing here trusts the shape: this is the seam a
    LiteLLM/OpenAI-compatible proxy or a truncated body comes through."""
    if not isinstance(data, dict):
        raise MalformedReply(f"reply is {type(data).__name__}, not an object")
    msg = data.get("message")
    if msg is None:
        msg = {}
    if not isinstance(msg, dict):
        raise MalformedReply(f"message is {type(msg).__name__}, not an object")
    calls = msg.get("tool_calls") or []
    if not isinstance(calls, list):
        raise MalformedReply(
            f"tool_calls is {type(calls).__name__}, not a list")
    content = msg.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise MalformedReply(
            f"content is {type(content).__name__}, not a string")
    return content, calls


def cap_tool_text(text, cap=MAX_TOOL_TEXT_CHARS):
    """(text_for_the_model, was_truncated). Over `cap`, keep the head —
    cut at a line break when one is near the end — and append a marker
    that names the loss, so the model reads a short result AND knows it is
    short. Never returns more than cap + len(marker) characters."""
    text = "" if text is None else str(text)
    if cap <= 0:
        cap = 0
    if len(text) <= cap:
        return text, False
    head = text[:cap]
    nl = head.rfind("\n")
    if nl > cap // 2:
        head = head[:nl]
    marker = TOOL_TRUNCATED_MARKER.format(shown=len(head), total=len(text))
    return head.rstrip() + marker, True


def _http(path, payload=None, timeout=OLLAMA_TIMEOUT_S):
    """The ONE seam for every Ollama call (tests monkeypatch this).
    GET when payload is None, else POST JSON. Returns the decoded JSON.
    Raises OllamaDown on connection refused, urllib errors otherwise."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(f"{OLLAMA_URL}{path}", data=data,
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), ConnectionRefusedError):
            raise OllamaDown(str(exc)) from exc
        raise
    try:
        return json.loads(body or b"{}")
    except ValueError as exc:
        # a proxy's HTML error page, or a truncated body
        raise MalformedReply(f"body is not JSON: {str(exc)[:60]}") from exc


def _http_stream(path, payload, timeout=OLLAMA_TIMEOUT_S):
    """Streaming variant of _http: yields each NDJSON object as it arrives.
    Same error mapping. Tests monkeypatch this."""
    data = json.dumps(dict(payload, stream=True)).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), ConnectionRefusedError):
            raise OllamaDown(str(exc)) from exc
        raise
    with resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise MalformedReply(f"stream line is not JSON: {str(exc)[:60]}") from exc


_SENTENCE_END_RX = re.compile(r"(?<=[.!?])\s+(?=\S)")


def _split_complete_sentences(buf):
    """(complete sentences, remainder) -- a sentence is complete once text
    follows its terminator, so an abbreviation mid-stream is not cut."""
    parts = _SENTENCE_END_RX.split(buf)
    if len(parts) <= 1:
        return [], buf
    return parts[:-1], parts[-1]


def _options(**overrides):
    opts = dict(CHAT_OPTIONS)
    opts.update(overrides)
    opts["num_ctx"] = NUM_CTX          # never varies: a change reloads the model
    return opts


def _chat_payload(messages, tools=None, fmt=None, **opt_overrides):
    # keep_alive -1 pins the model; while it is lent out any request that
    # still slips through (a caller that skipped _check_lent) must not pin
    # it again, so the payload itself says "unload after this one".
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False,
               "think": False,
               "keep_alive": 0 if _RESIDENCY.get("lent") else -1,
               "options": _options(**opt_overrides)}
    if tools:
        payload["tools"] = tools
    if fmt is not None:
        payload["format"] = fmt
    return payload


def _registry_schemas(registry, text=None):
    """Every registered tool, every turn. A per-turn subset chosen from
    the text was tried (2026-08-30) and dropped: the chat template renders
    the tools into the prefix, so a subset that changes between turns
    evicts Ollama's prefix cache (module doc, "static-prefix rule") and
    costs more prefill than the schemas it saves. ``text`` is accepted for
    the older call shape and ignored."""
    if registry is None:
        return []
    try:
        return registry.schemas()
    except Exception:
        log.exception("tool registry schemas failed")
        return []


def _same_model(a, b):
    def norm(name):
        name = (name or "").strip()
        return name if ":" in name else f"{name}:latest"
    return norm(a) == norm(b)


# ----------------------------------------------------------------------
# Residency (spec 4.3)
# ----------------------------------------------------------------------
def ensure_resident(first=None):
    """Keep OLLAMA_MODEL loaded with keep_alive -1 and its prefix cached.

    At boot (first=True, or the first call of the process): every other
    loaded model is unloaded ONCE, then ours is warmed with the real static
    prompt + tool schemas. Later calls only re-warm ours if it fell out.
    Returns True when the model is resident afterwards; never raises.
    """
    if first is None:
        first = not _RESIDENCY["unloaded_once"]
    if _RESIDENCY.get("lent"):
        # The residency loop keeps running; it just must not re-warm the
        # model a trainer was given the room for.
        log.debug("ollama: %s is lent out; not re-warming", OLLAMA_MODEL)
        return False
    try:
        ps = _http("/api/ps", timeout=5)
    except OllamaDown:
        log.warning("ollama: not running; %s cannot be made resident",
                    OLLAMA_MODEL)
        return False
    except Exception as exc:
        log.warning("ollama: /api/ps failed: %s", exc)
        return False
    loaded = [m.get("name") or m.get("model") or ""
              for m in ps.get("models", []) or []]
    ours_loaded = any(_same_model(m, OLLAMA_MODEL) for m in loaded)
    if first:
        for name in loaded:
            if _same_model(name, OLLAMA_MODEL):
                continue
            try:
                _http("/api/generate", {"model": name, "keep_alive": 0},
                      timeout=30)
                log.info("ollama: unloaded %s at startup", name)
            except Exception as exc:
                log.warning("ollama: could not unload %s: %s", name, exc)
        _RESIDENCY["unloaded_once"] = True
    elif ours_loaded:
        return True
    messages = [{"role": "system", "content": static_system()},
                {"role": "user", "content": ""}]
    t0 = time.monotonic()
    payload = _chat_payload(messages, _registry_schemas(_REGISTRY),
                            num_predict=1)
    try:
        data = _http("/api/chat", payload, timeout=300)
    except Exception as exc:
        log.warning("ollama: warm-up of %s failed: %s", OLLAMA_MODEL, exc)
        return False
    finally:
        _unpin_if_lent(payload)            # a 300 s warm can race release()
    load_s = (data.get("load_duration") or 0) / 1e9 or \
        (time.monotonic() - t0)
    log.info("ollama: %s resident (load %.1f s)", OLLAMA_MODEL, load_s)
    return True


def start_residency(interval_s=RESIDENCY_INTERVAL_S):
    """Boot warm-up now, then a re-check every interval_s (daemon thread;
    idempotent)."""
    t = _RESIDENCY.get("thread")
    if t is not None and t.is_alive():
        return t

    def _loop():
        while True:
            try:
                ensure_resident()
            except Exception:
                log.exception("ensure_resident failed")
            time.sleep(interval_s)

    t = threading.Thread(target=_loop, daemon=True, name="ollama-resident")
    _RESIDENCY["thread"] = t
    t.start()
    return t


# ----------------------------------------------------------------------
# GPU yield: lend the model to a trainer, take it back when it is gone
# ----------------------------------------------------------------------
def is_lent():
    return bool(_RESIDENCY.get("lent"))


def _check_lent():
    if _RESIDENCY.get("lent"):
        raise ModelLent(OLLAMA_MODEL)


def release(reason=""):
    """Unload OLLAMA_MODEL now (keep_alive 0, the same call ensure_resident
    makes for foreign models) and mark it lent. The flag is set even when
    Ollama cannot be reached -- the point is that nothing reloads the model
    behind the trainer's back. Returns True when the unload was accepted;
    never raises."""
    if _RESIDENCY.get("lent"):
        return True
    _RESIDENCY["lent"] = True
    log.info("ollama: lending %s out%s", OLLAMA_MODEL,
             f" ({reason})" if reason else "")
    try:
        _http("/api/generate", {"model": OLLAMA_MODEL, "keep_alive": 0},
              timeout=30)
        return True
    except OllamaDown:
        log.warning("ollama: not running; %s marked lent anyway", OLLAMA_MODEL)
    except Exception as exc:
        log.warning("ollama: could not unload %s: %s", OLLAMA_MODEL, exc)
    return False


def _unpin_if_lent(payload):
    """A request that raced release(): its payload was built with
    keep_alive -1 (pinning) before the lend, and finished after -- so the
    model the trainer was given the room for is resident again. Compensate
    with a best-effort unload; never raises. Called in a finally after
    every pinning chat/warm request."""
    if payload.get("keep_alive") != -1 or not _RESIDENCY.get("lent"):
        return
    log.info("ollama: request raced the lend; unloading %s again", OLLAMA_MODEL)
    try:
        _http("/api/generate", {"model": OLLAMA_MODEL, "keep_alive": 0},
              timeout=30)
    except Exception as exc:               # noqa: BLE001 - best effort
        log.warning("ollama: could not unpin %s: %s", OLLAMA_MODEL, exc)


def reclaim():
    """Clear the lent flag and warm the model again. Returns
    ensure_resident()'s verdict; never raises."""
    if not _RESIDENCY.get("lent"):
        return ensure_resident()
    _RESIDENCY["lent"] = False
    log.info("ollama: reclaiming %s", OLLAMA_MODEL)
    try:
        return bool(ensure_resident())
    except Exception:
        log.exception("ollama: reclaim warm-up failed")
        return False


# ----------------------------------------------------------------------
# Guards between the model and TTS
# ----------------------------------------------------------------------
MAX_SPOKEN_SENTENCES = 2
MAX_SPOKEN_CHARS = 250                    # prefer a sentence end below this
HARD_SPOKEN_CHARS = _TTS.MAX_SPEAK_LENGTH  # the one hard limit, shared w/ TTS
NO_CLOCK_LINE = "I'm afraid I haven't a clock in front of me just now, sir."
# Said instead when a clock IS available and the model's reading contradicts
# it — claiming to have no clock would be the second false statement.
UNSURE_CLOCK_LINE = "Let me check the time again, sir; that didn't look right."

_LABEL_RX = re.compile(r"^\s*(?:jarvis|assistant)\s*:\s*", re.I)
_TURN_RX = re.compile(r"\n\s*(?:user|hunter)\s*:.*", re.I | re.S)
_STAGE_RX = re.compile(
    r"\s*[*(]\s*(?:chuckles|laughs|sighs|pauses|smiles|smirks|"
    r"clears throat|adjusts)[^*)]*[*)]", re.I)
_EMOJI_RX = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]")
_BULLET_RX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
# "tts.py" is spoken as "tts": TTS would otherwise voice "dot pee why".
_FILE_EXT_RX = re.compile(
    r"\b(\w+)\.(py|txt|md|json|yaml|yml|toml|cfg|ini|sh|log|csv)\b")
_ABBR_RX = re.compile(
    r"\b(?:e\.g|i\.e|etc|Mr|Mrs|Ms|Dr|St|vs|a\.m|p\.m|No)\.", re.I)
_SENT_END_RX = re.compile(r"(?<=[.!?])\s+")
_CLOCK_RX = re.compile(r"\b\d{1,2}:\d{2}\b")
# the same reading with its meridiem, for comparing times rather than strings
_CLOCK_MER_RX = re.compile(r"\b(\d{1,2}):(\d{2})\s*([ap]\.?\s?m\.?)?", re.I)
# a sentence that asserts what the time is NOW (as opposed to when
# something starts, ends or finishes) — the only kind that must match a
# known reading.
_NOW_CLAIM_RX = re.compile(
    r"\b(?:it'?s|it is|it appears to be|the time (?:is|reads)|"
    r"time is now|currently|right now|the clock (?:says|shows|reads))\b",
    re.I)
_DOT_HOLD = "\x00"
_ELLIPSIS_HOLD = "\x01"


def split_sentences(text):
    """Abbreviation-aware sentence split; '...' stays inside a sentence."""
    t = (text or "").replace("...", _ELLIPSIS_HOLD)
    t = _ABBR_RX.sub(lambda m: m.group(0).replace(".", _DOT_HOLD), t)
    parts = [p.strip() for p in _SENT_END_RX.split(t) if p.strip()]
    return [p.replace(_DOT_HOLD, ".").replace(_ELLIPSIS_HOLD, "...")
            for p in parts]


def limit_sentences(text, n=MAX_SPOKEN_SENTENCES):
    """Keep the first n sentences (the <=2 rule, enforced in code)."""
    return " ".join(split_sentences(text)[:max(1, int(n or 1))])


def clean_ollama_reply(text):
    """Strip artefacts before the reply reaches TTS: a leading "Jarvis:"
    label, a run-on "User:" turn, stage directions, bullet markers, emoji
    and file extensions. Markdown emphasis and headings are handled by
    strip_markdown."""
    text = _LABEL_RX.sub("", (text or "").strip())
    text = _TURN_RX.sub("", text)
    text = _STAGE_RX.sub("", text)
    text = _EMOJI_RX.sub("", text)
    text = _BULLET_RX.sub("", text)
    text = _FILE_EXT_RX.sub(r"\1", text)
    return text.strip()


_URL_RX = re.compile(r"\bhttps?://[^\s)\]]+|\bwww\.[^\s)\]]+", re.I)
_MD_LINK_RX = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# A source list at a line start OR after a sentence end -- and only when it
# actually carries links, so an answer that opens "Source: Reuters." keeps
# its text instead of being wiped to the failure line.
_SOURCES_RX = re.compile(
    r"(?:^|(?<=[.!?])\s+)(?:sources?|references?|citations?)\s*:"
    r"(?=.*(?:https?://|www\.|\[[^\]]+\]\())(?:.*)$",
    re.I | re.M | re.S)


def clean_web_answer(text, max_sentences=3):
    """A spoken line from a one-shot web answer: the model was told no links
    and no source list, and sonnet appended "Sources: [..](..)" anyway."""
    text = text or ""
    text = _SOURCES_RX.sub("", text)          # from "Sources:" to the end
    text = _MD_LINK_RX.sub(r"\1", text)       # [text](url) -> text
    text = _URL_RX.sub("", text)
    text = _BULLET_RX.sub("", text)           # while the line breaks still exist
    text = strip_markdown(text)
    text = re.sub(r"\s+", " ", text).strip(" -–—:;,")
    return trim_spoken(limit_sentences(text, max_sentences))


def strip_markdown(text):
    """Bold, code spans, headings, tables and URLs out; whitespace
    collapsed. Runs AFTER clean_ollama_reply (it destroys the line breaks
    that function's guards are anchored to)."""
    clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', text or "")
    clean = re.sub(r'`([^`]+)`', r'\1', clean)
    clean = re.sub(r'#{1,6}\s+', '', clean)
    clean = re.sub(r'https?://\S+', '', clean)
    # tables: the rule row goes, the cell pipes become clause breaks —
    # "| Wed | 85 |" is read as "Wed, 85", not "pipe Wed pipe".
    clean = re.sub(r'^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$', '', clean, flags=re.M)
    clean = re.sub(r'[ \t]*\|[ \t]*', ', ', clean)
    clean = re.sub(r'(?m)^[ \t]*,[ \t]*|[ \t]*,[ \t]*$', '', clean)
    clean = re.sub(r'(?:,\s*){2,}', ', ', clean)
    return re.sub(r'\s+', ' ', clean).strip()


def _clock_minutes(text):
    """Every hh:mm reading in `text` as minutes past midnight. A reading
    with no am/pm is ambiguous, so BOTH readings are returned for it:
    "10:30" is 10:30 and 22:30, "19:14" only 19:14. That is what makes a
    24-hour rendering of a tool's "7:14 pm" match the tool."""
    out = set()
    for hh, mm, mer in _CLOCK_MER_RX.findall(text or ""):
        h, m = int(hh), int(mm)
        if h > 23 or m > 59:
            continue
        mer = mer.lower().replace(".", "").replace(" ", "")
        if mer == "am":
            out.add((h % 12) * 60 + m)
        elif mer == "pm":
            out.add((h % 12) * 60 + m + 720)
        elif h >= 13:
            out.add(h * 60 + m)
        else:
            out.add(h * 60 + m)                 # as written
            out.add((h % 12) * 60 + m + 720)    # the other half of the day
    return out


def guard_clock_claims(text, context_text="", user_text=""):
    """Drop clock readings the model cannot know.

    A reading is grounded when it matches — as a time, not as a string —
    one the background, a tool result or Hunter himself supplied. So a
    24-hour rendering of the tool's own "7:14 pm" survives, and so does
    honest arithmetic on it ("your build finishes at 9:14 pm"): only a
    sentence that ASSERTS the present time ("it's 3:00 am") has to match
    something known. If no clock reading is available at all, any hh:mm is
    an invention and goes.

    Note this runs in production: it used to switch itself off whenever
    the context held the string "Current time", which ContextEngine always
    emits, so it never once fired on the live path."""
    text = text or ""
    if not _CLOCK_RX.search(text):
        return text
    allowed = _clock_minutes(context_text) | _clock_minutes(user_text)
    sentences = split_sentences(text)
    kept = []
    for s in sentences:
        toks = _CLOCK_RX.findall(s)
        if not toks:
            kept.append(s)
            continue
        grounded = all(_clock_minutes(tok) & allowed for tok in toks)
        if grounded or (allowed and not _NOW_CLAIM_RX.search(s)):
            kept.append(s)
    if len(kept) != len(sentences):
        log.info("dropped an ungrounded clock reading: %s", text[:80])
    if kept:
        return " ".join(kept)
    return UNSURE_CLOCK_LINE if allowed else NO_CLOCK_LINE


def spoken_from_ollama(raw, context_text="", user_text="",
                       n=MAX_SPOKEN_SENTENCES):
    """The full Tier 2 reply pipeline: clean, strip markdown, guard, cap
    at n sentences (two by default; a briefing tool may raise it).

    Order matters: clean_ollama_reply's bullet and run-on-turn regexes are
    anchored to line starts, and strip_markdown collapses every newline —
    running the markdown pass first (as _finish_spoken used to) left both
    of them dead on the production path."""
    text = clean_ollama_reply(raw)
    text = clean_ollama_reply(strip_markdown(text))
    text = guard_clock_claims(text, context_text, user_text)
    return limit_sentences(text, n)


def _cut_at_clause(text, cap):
    head = text[:cap]
    idx = max(head.rfind(";"), head.rfind(","), head.rfind("."))
    if idx < 100:
        idx = head.rfind(" ")
    if idx < 100:
        idx = cap
    return head[:idx].rstrip(" ,;.") + "."


def trim_spoken(text, cap=MAX_SPOKEN_CHARS, hard=HARD_SPOKEN_CHARS):
    """Shorten free text for speech without ever ending mid-word: whole
    sentences that fit in `cap`; failing that the whole first sentence if
    it fits the TTS limit; failing that a clause boundary (';', ',', '.')
    beyond index 100, else the last space, plus a period."""
    text = (text or "").strip()
    if len(text) <= cap:
        return text
    sentences = split_sentences(text)
    kept = ""
    for s in sentences:
        candidate = f"{kept} {s}".strip()
        if len(candidate) > cap:
            break
        kept = candidate
    if kept:
        return kept
    first = sentences[0] if sentences else text
    if len(first) <= hard:
        return first
    return _cut_at_clause(first, hard)


def _finish_spoken(raw, guard_context, user_text, n):
    """Model text -> the line TTS gets: guards, markdown, the n-sentence
    cap and a char budget that grows with n (briefings) but never passes
    the TTS hard limit."""
    text = spoken_from_ollama(raw, guard_context, user_text, n)
    cap = MAX_SPOKEN_CHARS if n <= MAX_SPOKEN_SENTENCES else HARD_SPOKEN_CHARS
    return trim_spoken(text, cap=cap)


# ----------------------------------------------------------------------
# Persona-voiced helpers (spec 4.2): never raise
# ----------------------------------------------------------------------
def _persona_turn(instruction, text, n):
    unit = "sentence" if n == 1 else "sentences"
    return (f"Background:\n(none)\n\nInstruction for Jarvis (not a question "
            f"from Hunter): {instruction} Reply with the spoken line only, "
            f"at most {n} {unit}, in your own words.\n\nText:\n"
            f"{(text or '').strip()}")


def _persona_request(instruction, text, n, timeout, num_predict):
    _check_lent()          # summarize/local_line fall back to their text
    messages = [{"role": "system", "content": static_system()},
                {"role": "user", "content": _persona_turn(instruction, text, n)}]
    payload = _chat_payload(messages, _registry_schemas(_REGISTRY),
                            num_predict=num_predict)
    try:
        data = _http("/api/chat", payload, timeout=timeout)
    finally:
        _unpin_if_lent(payload)
    msg = data.get("message") or {}
    if msg.get("tool_calls"):
        return ""
    return _finish_spoken(msg.get("content", ""), text, "", n)


def summarize(text, max_sentences=2, timeout=6.0):
    """A spoken, persona-voiced summary of free text (a Claude result, a
    mail digest). On any failure: the text itself, capped and trimmed."""
    text = (text or "").strip()
    if not text:
        return ""
    n = max(1, int(max_sentences or 1))
    fallback = trim_spoken(limit_sentences(strip_markdown(text), n),
                           cap=MAX_SPOKEN_CHARS if n <= 2
                           else HARD_SPOKEN_CHARS)
    try:
        out = _persona_request(
            "Tell Hunter what this says, as Jarvis would aloud, keeping "
            "every number and name that matters.",
            text, n, timeout, num_predict=60 * n)
    except Exception as exc:
        log.warning("summarize fell back: %s", exc)
        return fallback
    return out or fallback


def local_line(instruction, text, max_sentences=1, timeout=2.0, fallback=""):
    """One persona line to order (an acknowledgement, a rewording). Returns
    `fallback` on timeout, error or an empty reply."""
    n = max(1, int(max_sentences or 1))
    try:
        out = _persona_request(instruction, text, n, timeout,
                               num_predict=40 * n)
    except Exception as exc:
        log.warning("local_line fell back: %s", exc)
        return fallback
    return out or fallback


def classify_route(text, timeout=CLASSIFY_TIMEOUT_S):
    """Router tie-breaker: ("local"|"claude", confidence). One /api/chat
    call with a JSON schema; ("local", 0.0) on any failure."""
    messages = [{"role": "system", "content": static_system()},
                {"role": "user",
                 "content": f"{ROUTE_INSTRUCTION}\n\nHunter: "
                            f"{(text or '').strip()}"}]
    try:
        _check_lent()      # the router's rules decide alone while lent
        payload = _chat_payload(messages, _registry_schemas(_REGISTRY),
                                fmt=ROUTE_FORMAT, num_predict=40,
                                temperature=0.0)
        try:
            data = _http("/api/chat", payload, timeout=timeout)
        finally:
            _unpin_if_lent(payload)
        obj = json.loads((data.get("message") or {}).get("content") or "{}")
        route = str(obj.get("route", "")).strip().lower()
        confidence = float(obj.get("confidence", 0.0))
    except Exception as exc:
        log.warning("classify_route fell back to local: %s", exc)
        return ("local", 0.0)
    if route not in ("local", "claude"):
        return ("local", 0.0)
    return (route, max(0.0, min(1.0, confidence)))


# ----------------------------------------------------------------------
# Study helpers: explain a document, write a quiz, grade an answer.
# ----------------------------------------------------------------------
# These are NOT persona requests: _persona_request trims to n sentences and
# HARD_SPOKEN_CHARS (~500 chars, ~30 s of speech), and returns "" when the
# model emits a tool call -- a document summary needs a paragraph and a
# quiz needs five pairs. Each is one /api/chat with a JSON schema, the
# SAME static system prompt (so the persona and the "spoken" rules hold)
# and NO tool schemas: the document text takes the room the schemas would
# have (static_system is ~1.5k tokens; EXPLAIN_MAX_CHARS of text is ~3.5k;
# NUM_CTX is 8192). The one cost is a prefix-cache miss on the next tool
# loop call (~2 s of prefill, once), which a 20 s summary already dwarfs.
EXPLAIN_MAX_CHARS = 12_000       # the head of a document one call can read
EXPLAIN_TIMEOUT_S = 60.0
QUIZ_MAX_CHARS = 6_000           # study text per generation call
QUIZ_TIMEOUT_S = 45.0
GRADE_TIMEOUT_S = 8.0            # local_line's 2 s is too tight for gemma4:26b
EXPLAIN_FORMAT = {
    "type": "object",
    "properties": {"lead": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["lead", "summary"]}
QUIZ_FORMAT = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
        "required": ["question", "answer"]}}},
    "required": ["questions"]}
GRADE_FORMAT = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}, "note": {"type": "string"}},
    "required": ["correct", "note"]}


def _json_request(instruction, fmt, timeout, num_predict, temperature=0.2):
    """One tool-free /api/chat with a JSON schema; the parsed object, or
    None on any failure (callers speak a fixed excuse)."""
    messages = [{"role": "system", "content": static_system()},
                {"role": "user", "content": instruction}]
    payload = _chat_payload(messages, None, fmt=fmt, num_predict=num_predict,
                            temperature=temperature)
    try:
        try:
            data = _http("/api/chat", payload, timeout=timeout)
        finally:
            _unpin_if_lent(payload)
        content, _calls = _message_parts(data)
        obj = json.loads(content or "{}")
    except Exception as exc:               # noqa: BLE001 - one seam, one excuse
        log.warning("json request failed: %s: %s", type(exc).__name__, exc)
        return None
    return obj if isinstance(obj, dict) else None


def explain_text(text, name="the document", timeout=EXPLAIN_TIMEOUT_S):
    """(lead, summary) for a document: ``lead`` is the two-sentence spoken
    opening, ``summary`` the fuller paragraph shown as a card. ("", "")
    when the model is down or the text is empty."""
    text = (text or "").strip()
    if not text:
        return "", ""
    body = text[:EXPLAIN_MAX_CHARS]
    cut = " (the opening pages; it goes on)" if len(text) > len(body) else ""
    instruction = (
        f"Instruction for Jarvis (not a question from Hunter): Hunter asked you "
        f"to explain his document \"{name}\"{cut}. Reply with JSON only. "
        f"\"lead\": two spoken sentences, as Jarvis would say them aloud, giving "
        f"what the document is and the one thing that matters most in it. "
        f"\"summary\": one plain paragraph of five to eight sentences with the key "
        f"points, every number, date, deadline and anything due, in your own "
        f"words, no lists, no markdown.\n\nDocument:\n{body}")
    obj = _json_request(instruction, EXPLAIN_FORMAT, timeout, num_predict=420)
    if not obj:
        return "", ""
    lead = strip_markdown(clean_ollama_reply(str(obj.get("lead") or "")))
    summary = strip_markdown(clean_ollama_reply(str(obj.get("summary") or "")))
    lead = trim_spoken(limit_sentences(lead, 2), cap=HARD_SPOKEN_CHARS)
    if not lead and summary:
        lead = trim_spoken(limit_sentences(summary, 2), cap=HARD_SPOKEN_CHARS)
    return lead, summary


def make_quiz(text, n=5, topic="", timeout=QUIZ_TIMEOUT_S):
    """[{question, answer}] x up to n from study text; [] on failure. The
    answers are asked to be short (a phrase, a number, a name) so the
    string grader in tools.quiz has something to match."""
    text = (text or "").strip()
    if not text:
        return []
    n = max(1, min(int(n or 5), 10))
    about = f" about {topic}" if topic else ""
    instruction = (
        f"Instruction for Jarvis (not a question from Hunter): write {n} quiz "
        f"questions{about} from the study text below, to test whether Hunter has "
        f"learned it. Reply with JSON only. Each question is one spoken sentence "
        f"answerable from the text; each answer is short -- a phrase, a number, "
        f"a name or a definition of at most twelve words -- never a yes or no. "
        f"Cover different facts; no markdown.\n\nStudy text:\n{text[:QUIZ_MAX_CHARS]}")
    obj = _json_request(instruction, QUIZ_FORMAT, timeout, num_predict=110 * n,
                        temperature=0.4)
    out = []
    for item in (obj or {}).get("questions") or []:
        if not isinstance(item, dict):
            continue
        q = " ".join(str(item.get("question") or "").split())
        a = " ".join(str(item.get("answer") or "").split())
        if q and a:
            out.append({"question": q, "answer": a})
    return out[:n]


def grade_answer(question, expected, given, timeout=GRADE_TIMEOUT_S):
    """(correct, note) from the model, or None when it did not answer (the
    caller then falls back to the string match or a shrug)."""
    instruction = (
        f"Instruction for Jarvis (not a question from Hunter): grade Hunter's "
        f"spoken quiz answer. Reply with JSON only: \"correct\" is true when his "
        f"answer means the same as the expected answer (wording, order and "
        f"small transcription slips do not matter; a missing or wrong fact "
        f"does), and \"note\" is one short spoken sentence from Jarvis saying "
        f"what was right or what the answer was.\n\nQuestion: {question}\n"
        f"Expected answer: {expected}\nHunter's answer: {given}")
    obj = _json_request(instruction, GRADE_FORMAT, timeout, num_predict=60,
                        temperature=0.0)
    if not obj or "correct" not in obj:
        return None
    note = strip_markdown(clean_ollama_reply(str(obj.get("note") or "")))
    return bool(obj.get("correct")), trim_spoken(limit_sentences(note, 1))


class JarvisBrain:
    """Hybrid brain: Ollama (fast, tools) + Claude (smart), with context +
    memory.

    context and memory are injected (shared app-wide instances); either may
    be None in tests. registry defaults to the module registry installed
    by set_registry() (looked up per call, so wiring order is free).
    """

    BUSY_MAX_S = 180.0     # busy guard auto-expires after this long

    def __init__(self, context, memory, registry=None):
        self._context = context
        self._memory = memory
        self._registry = registry
        self._busy = False
        self._busy_since = 0.0
        self._busy_lock = threading.Lock()
        self._cancelled = False
        self._proc = None
        self._proc_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(self, text, callback=None, force_tool=None, force_args=None,
             max_rounds=3, on_sentence=None):
        """Tier 2 with tools on a worker thread; callback gets the tags
        ([("BRIEFING", json)] when a card was produced, then ("SPEAK",
        line))."""
        if not self._acquire_busy():
            if callback:
                callback([("SPEAK",
                           "Still on the last one, sir. One moment.")])
            return None
        gen = self._job_gen                   # this job's identity, for cancel()

        def _live():
            # cancel() then a new job: the new acquire resets _cancelled,
            # so the flag alone let the dead job's reply be remembered and
            # spoken over the live one (a "no, I said ..." correction
            # re-dispatches within milliseconds). The generation cannot be
            # reset back.
            return not self._cancelled and self._job_gen == gen

        def _process():
            bus.publish(BrainState(state="thinking"))
            try:
                tags = self._chat_sync(text, force_tool=force_tool,
                                       force_args=force_args,
                                       max_rounds=max_rounds,
                                       on_sentence=on_sentence, gen=gen)
                if not _live():
                    log.info("chat cancelled; dropping result")
                    return
                self._remember(text, tags)
                if callback:
                    callback(tags)
            except Exception:
                # The message of a Python exception is not a spoken line:
                # it is unread text (and sometimes third-party text) that
                # used to be read aloud verbatim. The log has the detail.
                log.exception("chat error")
                if callback and _live():
                    callback([("SPEAK", INTERNAL_ERROR_LINE)])
            finally:
                # cancel() (barge-in) already released the guard, and a
                # newer job may hold it by now: a blanket release here let
                # the dead job's tail free the live job's guard.
                if self._job_gen == gen:
                    self._busy = False
                bus.publish(BrainState(state="idle"))

        t = threading.Thread(target=_process, daemon=True, name="brain-chat")
        t.start()
        return t

    def classify_route(self, text, timeout=CLASSIFY_TIMEOUT_S):
        return classify_route(text, timeout=timeout)

    def web_answer(self, question, callback=None, model="haiku",
                   timeout=WEB_TIMEOUT_S):
        """Answer a question from the web through a one-shot `claude -p` on
        a worker thread; the callback gets [("SPEAK", line)] like chat().
        Returns the thread, or None when the CLI is missing or the brain is
        busy (the busy line is spoken through the callback)."""
        if not MACHINE.claude_bin:
            return None
        if not self._acquire_busy():
            if callback:
                callback([("SPEAK", "Still on the last one, sir. One moment.")])
            return False                  # busy: already spoken (None = no CLI)
        gen = self._job_gen               # this job's identity, for cancel()

        # "Look it up" needs a referent: the one-shot has no history unless
        # it is handed some.
        recent = ""
        try:
            # The last exchanges only. format_for_prompt() also carries the
            # git state, the active window title and the session log, and
            # this prompt leaves the machine.
            if self._context and hasattr(self._context, "recent_conversation_text"):
                recent = self._context.recent_conversation_text() or ""
        except Exception:
            log.debug("web answer: context unavailable", exc_info=True)
        recent = recent.strip()[-1200:]
        block = ("\n\nRecent conversation, only so pronouns like 'it' resolve "
                 "(do not answer it):\n" + recent) if recent else ""

        def _live():
            return not self._cancelled and self._job_gen == gen

        def _process():
            bus.publish(BrainState(state="thinking"))
            try:
                out = self._run_claude(
                    WEB_PROMPT.format(recent=block, q=question), timeout=timeout,
                    extra_args=["--model", str(model or "haiku"),
                                "--allowedTools", "WebSearch,WebFetch"],
                    output_format="json")
                if not _live():
                    log.info("web answer cancelled; dropping result")
                    return
                line = clean_web_answer(out) or WEB_FAIL_LINE
                log.info("web answer (%s): %.80s", model, line)
                tags = [("SPEAK", line)]
                try:
                    self._remember(question, tags)
                except Exception:
                    log.debug("web answer remember failed", exc_info=True)
                if callback:
                    callback(tags)
            except subprocess.TimeoutExpired:
                log.warning("web answer timed out after %.0fs", timeout)
                if callback and _live():
                    callback([("SPEAK", WEB_SLOW_LINE)])
            except Exception:
                log.exception("web answer error")
                if callback and _live():
                    callback([("SPEAK", WEB_FAIL_LINE)])
            finally:
                if self._job_gen == gen:  # a newer job owns the guard now
                    self._busy = False
                    bus.publish(BrainState(state="idle"))

        t = threading.Thread(target=_process, daemon=True, name="brain-web")
        t.start()
        return t

    def summarize(self, text, max_sentences=2, timeout=6.0):
        return summarize(text, max_sentences=max_sentences, timeout=timeout)

    def local_line(self, instruction, text, max_sentences=1, timeout=2.0,
                   fallback=""):
        return local_line(instruction, text, max_sentences=max_sentences,
                          timeout=timeout, fallback=fallback)

    def explain_text(self, text, name="the document", timeout=EXPLAIN_TIMEOUT_S):
        return explain_text(text, name=name, timeout=timeout)

    def make_quiz(self, text, n=5, topic="", timeout=QUIZ_TIMEOUT_S):
        return make_quiz(text, n=n, topic=topic, timeout=timeout)

    def grade_answer(self, question, expected, given, timeout=GRADE_TIMEOUT_S):
        return grade_answer(question, expected, given, timeout=timeout)

    def think(self, user_input, callback=None):
        """Legacy entry (deploy/autonomous era): a local question goes to
        the tool loop, anything else to Claude's tag protocol."""
        if not self._acquire_busy():
            if callback:
                callback([("SPEAK",
                           "Still on the last one, sir. One moment.")])
            return

        def _process():
            bus.publish(BrainState(state="thinking"))
            try:
                if self._is_local_question(user_input):
                    actions = self._query_ollama(user_input)
                else:
                    actions = self._query_claude(user_input)

                if self._cancelled:
                    log.info("think cancelled; dropping result")
                    return

                self._remember(user_input, actions)
                if callback:
                    callback(actions)
            except Exception:
                log.exception("brain error")
                if callback and not self._cancelled:
                    callback([("SPEAK", INTERNAL_ERROR_LINE)])
            finally:
                self._busy = False
                bus.publish(BrainState(state="idle"))

        threading.Thread(target=_process, daemon=True,
                         name="brain-think").start()

    def execute_autonomous(self, task_description, callback=None):
        """Execute a multi-step task autonomously.

        Queries Claude repeatedly, executing [RUN] commands and feeding
        results back until [DONE] or max steps reached. Wired to the
        commander registry ("deploy" / "autonomous:" phrases).
        """
        if not self._acquire_busy():
            if callback:
                callback([("SPEAK",
                           "I'm afraid I'm mid-task, sir. Give me a moment.")])
            return

        def _run():
            bus.publish(BrainState(state="thinking"))
            try:
                results = self._autonomous_loop(task_description, callback)
                log.info("autonomous task complete: %d steps", len(results))
            except Exception as e:
                log.exception("autonomous error")
                if callback and not self._cancelled:
                    callback([("SPEAK",
                               f"I'm afraid the task failed, sir. "
                               f"{str(e)[:40]}")])
            finally:
                self._busy = False
                bus.publish(BrainState(state="idle"))

        threading.Thread(target=_run, daemon=True,
                         name="brain-autonomous").start()

    def warmup(self):
        """Make the local model resident now and keep it so (spec 4.3):
        boot warm-up, then a check every five minutes."""
        return start_residency()

    def ensure_resident(self, first=None):
        return ensure_resident(first=first)

    # GPU yield (module-level state: the watchdog and the commander share it)
    def release(self, reason=""):
        return release(reason=reason)

    def reclaim(self):
        return reclaim()

    def is_lent(self):
        return is_lent()

    def cancel(self):
        """Kill any in-flight subprocess and clear the busy guard."""
        self._cancelled = True
        with self._proc_lock:
            proc = self._proc
            self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                log.info("cancelled in-flight brain subprocess")
            except Exception:
                log.exception("cancel: failed to kill subprocess")
        self._busy = False
        bus.publish(BrainState(state="idle"))

    @property
    def is_busy(self):
        if self._busy and \
                time.monotonic() - self._busy_since > self.BUSY_MAX_S:
            return False   # stale guard; treated as expired
        return self._busy

    @property
    def registry(self):
        return self._registry if self._registry is not None else _REGISTRY

    # ------------------------------------------------------------------
    # Busy guard (auto-expiring — no permanent deafness)
    # ------------------------------------------------------------------
    def _acquire_busy(self):
        # Check-then-set under a lock: two wake-words a millisecond apart
        # used to be able to both see an idle brain and both proceed.
        with self._busy_lock:
            if self._busy and \
                    time.monotonic() - self._busy_since > self.BUSY_MAX_S:
                log.warning("busy guard held > %.0fs — auto-expiring and "
                            "cancelling the stale call", self.BUSY_MAX_S)
                self.cancel()
            if self._busy:
                return False
            self._busy = True
            self._busy_since = time.monotonic()
            self._cancelled = False
            # Each job gets an identity: cancel() then a new job used to let
            # the dead job's tail speak and release the new job's guard.
            self._job_gen = getattr(self, "_job_gen", 0) + 1
            return True

    def _remember(self, user_input, tags):
        spoken = " ".join(d for t, d in tags if t == "SPEAK")
        if self._memory:
            try:
                self._memory.log_habit(user_input[:50])
            except Exception:
                log.exception("memory.log_habit failed")
        if self._context:
            try:
                self._context.add_exchange(user_input, spoken)
            except Exception:
                log.exception("context.add_exchange failed")

    # ------------------------------------------------------------------
    # Routing (legacy think())
    # ------------------------------------------------------------------
    def _is_local_question(self, text):
        lower = text.lower()
        for pattern in LOCAL_PATTERNS:
            if pattern in lower:
                return True
        words = lower.split()
        if len(words) <= 5 and words and \
                words[0].strip(",.!?") not in ACTION_VERBS:
            return True
        return False

    # ------------------------------------------------------------------
    # Tier 2: Ollama /api/chat with tools
    # ------------------------------------------------------------------
    def _dynamic_context(self, text=""):
        """The per-turn background: context + memory. ``text`` (Hunter's
        words) lets the memory pick the facts RELEVANT to this utterance;
        it stays in the user turn, never the static system prompt."""
        ctx_text = ""
        if self._context:
            ctx = self._context.get_context("standard")
            ctx_text = self._context.format_for_prompt(ctx, spoken=True)
        mem_text = ""
        if self._memory:
            mem_text = self._memory.format_for_context(text) if text else \
                self._memory.format_for_context()
        return ctx_text, mem_text

    def _journal_tool(self, name, args, result):
        """One journal row per tool call (jarvis.context journal); the
        recap reads it back. Never raises into the tool loop."""
        journal = getattr(self._context, "journal_tool", None)
        if journal is None:
            return
        try:
            journal(name, args, ok=bool(getattr(result, "ok", True)),
                    text=str(getattr(result, "text", "") or ""))
        except Exception:
            log.exception("journal_tool failed")

    def _query_ollama(self, user_input):
        """Legacy action-list shape for think(): the tool loop's tags."""
        return self._chat_sync(user_input)

    @staticmethod
    def _tool_call_parts(call):
        fn = (call or {}).get("function") or {}
        name = str(fn.get("name") or "").strip()
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                args = {"text": args}
        if not isinstance(args, dict):
            args = {}
        return name, args

    def _stale(self, gen=None) -> bool:
        """True once this job was cancelled OR superseded by a newer one
        (its generation moved on): the shared _cancelled flag is reset by
        the next acquire, the generation is not."""
        return self._cancelled or (gen is not None and self._job_gen != gen)

    def _chat_sync(self, text, force_tool=None, force_args=None, max_rounds=3,
                   on_sentence=None, gen=None):
        """The tool loop (spec 4.2), synchronous. Returns tags.

        ``on_sentence(sentence)``: when given, the final model turn is
        STREAMED and each complete sentence is handed over as it lands, so
        the first one is speaking while the rest generates. The returned
        tags then carry ("STREAMED", "<n>") ahead of SPEAK so the caller
        shows the full reply without speaking it a second time.
        """
        log.info("chat: %s", text[:60])
        if _RESIDENCY.get("lent"):
            # Before any HTTP, including a forced tool's render turn: the
            # gate has to sit here, not on the residency loop, because every
            # payload used to pin the model back with keep_alive -1.
            log.info("chat: model lent out; answering with the excuse")
            bus.publish(Status(text="Local model lent to the trainer",
                               kind="info"))
            return [("SPEAK", MODEL_LENT_LINE)]
        registry = self.registry
        ctx_text, mem_text = self._dynamic_context(text)
        messages = [{"role": "system", "content": static_system()},
                    {"role": "user",
                     "content": build_user_turn(ctx_text, mem_text, text)}]
        tools = _registry_schemas(registry, text)
        started = time.monotonic()
        deadline = started + CHAT_WALL_BUDGET_S
        cap = MAX_SPOKEN_SENTENCES
        card = None
        speak = None
        tool_texts = []
        tool_budget = MAX_TOOL_TEXT_TOTAL_CHARS
        truncated = False
        final = ""
        # Ollama reports load_duration even on a resident model: it is the
        # server's own per-request overhead before the runner sees the
        # prompt. Logging it separates "the model was slow" from "Ollama
        # was slow" when a reply misses the latency bar (spec 4.3).
        server_s = 0.0

        def note(result, name, args=None):
            nonlocal cap, card
            cap = max(cap, int(getattr(result, "max_sentences", 2) or 2))
            if getattr(result, "card", None):
                card = result.card
            tool_texts.append(result.text or "")
            log.info("tool %s -> ok=%s %s", name, result.ok,
                     (result.text or "")[:80])
            self._journal_tool(name, args, result)

        def tool_message(result, name):
            """The tool result as the MODEL sees it: capped against
            NUM_CTX (uncapped, an oversized result falls out of the
            context window and the model answers something unrelated with
            complete confidence) and, when cut, saying so in the message
            itself."""
            nonlocal tool_budget, truncated
            content, cut = cap_tool_text(
                result.text, min(MAX_TOOL_TEXT_CHARS, max(0, tool_budget)))
            tool_budget -= len(content)
            if cut:
                truncated = True
                log.warning("tool %s text truncated: %d chars -> %d",
                            name, len(result.text or ""), len(content))
            allowed = int(getattr(result, "max_sentences", 2) or 2)
            if allowed > MAX_SPOKEN_SENTENCES:
                # appended after the budget accounting: this is instruction,
                # not tool data, and must not squeeze the result itself out
                content += SENTENCE_ALLOWANCE.format(n=allowed)
            return {"role": "tool", "content": content, "tool_name": name}

        rounds_left = max(1, int(max_rounds or 1))
        if force_tool and registry is not None and registry.has(force_tool):
            # force_args pins arguments the model gets wrong on its own. It
            # chose unread_only=True for "what was my last email about?"
            # even with the parameter documented as the thing that answers
            # exactly that question, so the read mail newer than the answer
            # it gave was never searched.
            args = dict(force_args or {})
            result = registry.call(force_tool, args)
            note(result, force_tool, args)
            if result.speak:
                speak = result.speak
            else:
                messages.append({"role": "assistant", "content": "",
                                 "tool_calls": [{"function": {
                                     "name": force_tool, "arguments": args}}]})
                messages.append(tool_message(result, force_tool))
                rounds_left = 1        # one model turn renders the result

        streamed_sentences = []           # what on_sentence already received
        clock_line_said = []

        def guard(sentence):
            # The guards _finish_spoken applies to the whole reply, per
            # sentence: a streamed sentence is spoken before the reply
            # exists, so it must not carry an ungrounded clock claim or
            # a leaked context line the full reply would have lost.
            line = clean_ollama_reply(strip_markdown(clean_ollama_reply(sentence)))
            guarded = guard_clock_claims(
                line, "\n".join([ctx_text, mem_text] + tool_texts), text)
            if guarded != line and guarded in (NO_CLOCK_LINE, UNSURE_CLOCK_LINE):
                if clock_line_said:
                    return ""             # the honest line once, not per sentence
                clock_line_said.append(True)
            return trim_spoken(guarded.strip())

        try:
            while speak is None and rounds_left > 0:
                rounds_left -= 1
                if on_sentence is not None:
                    # Each round gets the full spoken cap: what the model
                    # said before a tool call must not eat the answer's.
                    round_sentences = []
                    try:
                        data, content, calls = self._stream_round(
                            messages, tools, cap, on_sentence, round_sentences, guard,
                            gen=gen)
                    finally:
                        # kept even when the stream dies: they were spoken
                        streamed_sentences.extend(round_sentences)
                    if self._stale(gen):
                        final = ""            # barged in: nothing more to say
                        break
                else:
                    payload = _chat_payload(messages, tools)
                    try:
                        data = _http("/api/chat", payload,
                                     timeout=OLLAMA_TIMEOUT_S)
                    finally:
                        _unpin_if_lent(payload)
                    content, calls = _message_parts(data)
                server_s += (data.get("load_duration") or 0) / 1e9
                if not calls or registry is None:
                    final = content
                    break
                if force_tool and tool_texts and rounds_left == 0 and \
                        messages[-1].get("role") == "tool":
                    # the single forced turn asked for more tools instead
                    # of rendering the result
                    final = TOOL_ONLY_LINE
                    break
                messages.append({"role": "assistant", "content": content,
                                 "tool_calls": calls})
                if len(calls) > MAX_TOOL_CALLS_PER_ROUND:
                    log.warning("chat: model asked for %d tools in one "
                                "round; running the first %d", len(calls),
                                MAX_TOOL_CALLS_PER_ROUND)
                    calls = calls[:MAX_TOOL_CALLS_PER_ROUND]
                for call in calls:
                    name, args = self._tool_call_parts(call)
                    result = registry.call(name, args)
                    note(result, name, args)
                    messages.append(tool_message(result, name))
                    if result.speak:
                        speak = result.speak
                        break
                    if time.monotonic() > deadline:
                        # the budget is checked INSIDE the round: a round
                        # of many calls must not run to the end first
                        log.warning("chat: over the %.0fs budget mid-round",
                                    CHAT_WALL_BUDGET_S)
                        rounds_left = 0
                        break
                if speak is None and rounds_left > 0 and \
                        time.monotonic() > deadline:
                    log.warning("chat: tool loop over %.0fs budget",
                                CHAT_WALL_BUDGET_S)
                    rounds_left = 0
                if speak is None and rounds_left == 0:
                    # Never the tool text itself: it can carry a stranger's
                    # words (a mail subject, a calendar title, a web page)
                    # and those are not spoken as if they were Jarvis's own.
                    final = TOOL_ONLY_LINE if tool_texts else ""
        except OllamaDown:
            log.warning("ollama connection refused")
            bus.publish(Status(text="Ollama isn't running", kind="warn"))
            return [("SPEAK", MODEL_DOWN_LINE)]
        except (MalformedReply, ValueError) as exc:
            # a proxy's HTML error page, OpenAI-style content blocks, a
            # truncated body: not the /api/chat shape, so there is nothing
            # to say but the honest line (the detail goes to the log).
            log.warning("ollama reply was malformed: %s", exc)
            bus.publish(Status(text="Local model reply was unreadable",
                               kind="warn"))
            return [("SPEAK", MODEL_EMPTY_LINE)]
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            log.warning("ollama request failed: %s", exc)
            if streamed_sentences:
                # the model died mid-reply after some of it was already
                # spoken: keep what was said rather than say it timed out
                final = " ".join(streamed_sentences)
            elif tool_texts:
                final = TOOL_ONLY_LINE
            else:
                bus.publish(Status(text="Local model timed out", kind="warn"))
                return [("SPEAK", MODEL_SLOW_LINE)]

        guard_ctx = "\n".join([ctx_text, mem_text] + tool_texts)
        if speak is not None:
            # An authored tool line still goes through the guards: notes
            # and to-dos put Hunter's own text on this path, emoji,
            # markdown, ten items and all.
            spoken = _finish_spoken(speak, guard_ctx, text, cap)
        else:
            spoken = _finish_spoken(final, guard_ctx, text, cap)
        if not spoken:
            spoken = MODEL_EMPTY_LINE
        elif truncated and not _PARTIAL_RX.search(spoken):
            # The model answered from a result it only half saw: say so
            # rather than let a confident half-answer stand. The notice
            # costs a sentence, so the answer gives one up.
            budget = (MAX_SPOKEN_CHARS if cap <= MAX_SPOKEN_SENTENCES
                      else HARD_SPOKEN_CHARS) - len(PARTIAL_RESULT_LINE) - 1
            budget = max(120, budget)
            head = trim_spoken(limit_sentences(spoken, max(1, cap - 1)),
                               cap=budget)
            if len(head) > budget:            # one very long sentence
                head = _cut_at_clause(head, budget)
            spoken = f"{head} {PARTIAL_RESULT_LINE}"
            if streamed_sentences and speak is None and on_sentence is not None \
                    and not _PARTIAL_RX.search(" ".join(streamed_sentences)):
                # the answer was spoken sentence by sentence: so is the notice
                streamed_sentences.append(PARTIAL_RESULT_LINE)
                try:
                    on_sentence(PARTIAL_RESULT_LINE)
                except Exception:
                    log.exception("on_sentence failed")
        log.info("chat reply (%.2fs wall, %.2fs ollama overhead): %s",
                 time.monotonic() - started, server_s, spoken[:80])
        tags = []
        if card:
            tags.append(("BRIEFING", json.dumps(card)))
        if streamed_sentences and speak is None:
            # A tool's speak= line (screen_qa's answer, a canvas excuse) was
            # never streamed: tagging it STREAMED silenced it.
            tags.append(("STREAMED", str(len(streamed_sentences))))
        tags.append(("SPEAK", spoken))
        return tags

    def _stream_round(self, messages, tools, cap, on_sentence, streamed,
                      guard=None, gen=None):
        """One /api/chat round, streamed. Complete sentences go to
        on_sentence as they land (guarded per sentence, capped at ``cap``);
        a round that turns out to be a tool call speaks nothing. Returns
        (data, content, calls) shaped like the non-streaming path."""
        content, calls, buf, last = "", [], "", {}
        # The socket timeout is per read, so a stream that keeps trickling
        # had no bound at all; the non-streamed request's wall bound stays.
        deadline = time.monotonic() + OLLAMA_TIMEOUT_S
        payload = _chat_payload(messages, tools)
        stream = _http_stream("/api/chat", payload, timeout=OLLAMA_TIMEOUT_S)
        try:
            for chunk in stream:
                if self._stale(gen):
                    log.info("chat: cancelled mid-stream")
                    break
                err = chunk.get("error")
                if err:
                    # A failure after the first token (runner died, out of
                    # memory) arrives as an NDJSON line, not an HTTP status:
                    # the headers had already gone out. Treat it as the
                    # request dying, which keeps what was already spoken.
                    raise OSError(f"ollama stream error: {str(err)[:80]}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"ollama stream over {OLLAMA_TIMEOUT_S}s")
                last = chunk
                msg = chunk.get("message") or {}
                piece = msg.get("content") or ""
                if piece:
                    content += piece
                    buf += piece
                for call in msg.get("tool_calls") or []:
                    calls.append(call)
                if calls:
                    continue                  # a tool round: never spoken
                done, buf = _split_complete_sentences(buf)
                for sent in done:
                    self._emit_sentence(sent, cap, on_sentence, streamed, guard)
                if chunk.get("done"):
                    break
        finally:
            stream.close()                    # a broken-off stream frees its socket
            _unpin_if_lent(payload)
        if not calls and buf.strip() and not self._stale(gen):
            self._emit_sentence(buf, cap, on_sentence, streamed, guard)
        data = dict(last)
        data["message"] = {"role": "assistant", "content": content,
                           "tool_calls": calls}
        return data, content, calls

    @staticmethod
    def _emit_sentence(sentence, cap, on_sentence, streamed, guard=None):
        if len(streamed) >= cap:
            return                            # the spoken cap still holds
        if guard is not None:
            line = guard(sentence)
        else:
            line = trim_spoken(strip_markdown(sentence).strip())
        if not line:
            return
        streamed.append(line)
        try:
            on_sentence(line)
        except Exception:
            log.exception("on_sentence failed")

    # ------------------------------------------------------------------
    # Tier 3: Claude CLI (deep reasoning)
    # ------------------------------------------------------------------
    def _query_claude(self, user_input):
        log.info("claude: %s", user_input[:60])

        if not MACHINE.claude_bin:
            log.error("claude binary not found (JARVIS_CLAUDE_BIN unset, "
                      "'claude' not on PATH)")
            bus.publish(Status(
                text="Claude CLI not found — install claude or set "
                     "JARVIS_CLAUDE_BIN.",
                kind="error"))
            return [("SPEAK",
                     "I'm afraid the Claude CLI isn't available, sir.")]

        ctx_text = ""
        if self._context:
            ctx = self._context.get_context("full")
            ctx_text = self._context.format_for_prompt(ctx)

        mem_text = ""
        if self._memory:
            mem_text = self._memory.format_for_context()
            sessions = self._memory.format_sessions_for_prompt()
            if sessions:
                mem_text += f"\n{sessions}"

        full_context = ctx_text
        if mem_text:
            full_context += f"\n\nMemory:\n{mem_text}"

        prompt = CLAUDE_SYSTEM.format(
            context=full_context,
            input=user_input,
        )

        try:
            response = self._run_claude(prompt, timeout=120)
            log.info("claude response: %s",
                     (response or "")[:80])
            if self._cancelled:
                return []
            if response:
                return self._parse_response(response)
        except subprocess.TimeoutExpired:
            return [("SPEAK",
                     "That one's taking longer than I'd like, sir. "
                     "Could you narrow it down?")]
        except Exception:
            log.exception("claude error")

        return [("SPEAK", "I'm afraid that one got away from me, sir.")]

    def _run_claude(self, prompt, timeout, extra_args=None, output_format="text"):
        """Run the Claude CLI with the prompt on stdin.

        Uses Popen (not run) so cancel() can kill it mid-flight. The
        binary's own dir is prepended to PATH for any helpers it spawns.
        ``extra_args`` go after the fixed flags (--model, --allowedTools).
        ``output_format="json"`` returns only the model's `result` field --
        a login prompt, a settings warning or an error on stdout is then
        never mistaken for an answer -- and "" when the CLI reports an error.
        """
        claude = MACHINE.claude_bin
        env = dict(os.environ)
        env["PATH"] = f"{Path(claude).parent}:{env.get('PATH', '')}"
        proc = subprocess.Popen(
            [claude, "-p", "--output-format", output_format, *(extra_args or [])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env,
        )
        with self._proc_lock:
            self._proc = proc
        try:
            out, err = proc.communicate(input=prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        finally:
            with self._proc_lock:
                if self._proc is proc:
                    self._proc = None
        if self._cancelled:
            return ""
        if proc.returncode not in (0, None):
            # Whatever a failing CLI printed is not an answer: a login
            # prompt or a bad --model message used to be read aloud.
            log.warning("claude exited rc=%s stdout=%.200s stderr=%.200s",
                        proc.returncode, (out or ""), (err or ""))
            return ""
        out = (out or "").strip()
        if output_format == "json":
            try:
                data = json.loads(out)
            except ValueError:
                log.warning("claude json output unreadable: %.200s", out)
                return ""
            if data.get("is_error"):
                log.warning("claude reported an error: %.200s", data.get("result"))
                return ""
            return str(data.get("result") or "").strip()
        return out

    # ------------------------------------------------------------------
    # Autonomous multi-step execution
    # ------------------------------------------------------------------
    def _autonomous_loop(self, task, callback, max_steps=10):
        results = []

        for step in range(max_steps):
            if self._cancelled:
                log.info("autonomous loop cancelled at step %d", step)
                return results

            ctx_text = ""
            if self._context:
                ctx = self._context.get_context("full")
                ctx_text = self._context.format_for_prompt(ctx)

            prompt = AUTONOMOUS_PROMPT.format(
                task=task, step=step + 1, max_steps=max_steps,
                results=(json.dumps(results[-3:], indent=2)
                         if results else "None yet"),
                context=ctx_text,
            )

            try:
                if not MACHINE.claude_bin:
                    raise RuntimeError("claude binary not found")
                response = self._run_claude(prompt, timeout=60)
                actions = self._parse_response(response)
            except Exception as e:
                log.exception("autonomous step %d error", step)
                if callback and not self._cancelled:
                    callback([("SPEAK",
                               f"Step {step + 1} failed, sir. "
                               f"{str(e)[:30]}")])
                break

            for action_type, action_data in actions:
                if self._cancelled:
                    return results
                if action_type == "DONE":
                    if callback:
                        callback([("SPEAK", action_data)])
                    return results
                elif action_type == "RUN":
                    log.info("auto-run: %s", action_data[:50])
                    try:
                        r = subprocess.run(
                            action_data, shell=True,
                            capture_output=True, text=True, timeout=30,
                        )
                        output = r.stdout.strip()[:500]
                        if r.returncode != 0:
                            output += f"\nSTDERR: {r.stderr.strip()[:200]}"
                        results.append({
                            "step": step, "command": action_data,
                            "output": output, "rc": r.returncode,
                        })
                    except Exception as e:
                        log.exception("auto-run failed")
                        results.append({
                            "step": step, "command": action_data,
                            "output": f"ERROR: {e}", "rc": -1,
                        })
                elif action_type == "SPEAK":
                    if callback:
                        callback([("SPEAK", action_data)])

            time.sleep(0.5)

        # Max steps reached
        if callback and not self._cancelled:
            callback([("SPEAK",
                       f"I've stopped after {len(results)} steps, sir; "
                       f"that was the limit.")])
        return results

    # ------------------------------------------------------------------
    # Response parsing (tag protocol — ported verbatim)
    # ------------------------------------------------------------------
    def _parse_response(self, response):
        actions = []
        has_structured = False

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            for tag in ("SPEAK", "RUN", "TYPE", "CLICK", "WINDOW",
                        "SILENT", "DONE"):
                if line.startswith(f"[{tag}]"):
                    content = line[len(tag) + 2:].strip()
                    if content:
                        actions.append((tag, content))
                        has_structured = True
                    break

        if not has_structured and response.strip():
            clean = trim_spoken(strip_markdown(response))  # never mid-word
            actions.append(("SPEAK", clean))

        return actions
