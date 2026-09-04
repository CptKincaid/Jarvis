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

from jarvis import address as address_mod
from jarvis import arc as arc_mod
from jarvis import scope as scope_mod
from jarvis.config import MACHINE
from jarvis.events import BrainState, Status, bus
from jarvis.logs import get_logger
from jarvis.router import is_question
from jarvis.tools.registry import ToolResult
from jarvis.tts import TTS as _TTS

log = get_logger("brain")

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("JARVIS_OLLAMA_MODEL") or "gemma4:26b"
OLLAMA_TIMEOUT_S = 20          # one /api/chat request
CHAT_WALL_BUDGET_S = 8.0       # the whole tool loop; beyond it, best text
# ...but the last stretch of that budget belongs to WORDS, not more work.
# LIVE 2026-08-31 14:35 ("what's on my calendar and what's on my latest
# email?"): get_mail spent 8.1 s on three sequential IMAP accounts, the
# loop then ran out and Jarvis said TOOL_ONLY_LINE -- it had the answer
# and apologised for not phrasing it, which is worse than a slow answer.
# So: past CHAT_WALL_BUDGET_S - RENDER_RESERVE_S no NEW tool call starts,
# and one final model round (with the tools stripped, so it can only
# write) is ALWAYS granted. Raising CHAT_WALL_BUDGET_S instead would just
# hide the slow tool and tax every fast turn.
RENDER_RESERVE_S = 3.0
# What the budget above MEASURES, though, is the part that took two goes.
# It used to be wall clock since the turn started -- which charged the
# tool half for the model's own latency. LIVE 2026-08-31 15:14 ("what's
# on my calendar and what's on my latest email?"), warm and resident:
#
#   15:14:11.9  round 1 starts          (tool_deadline = 16.9)
#   15:14:21.2  get_calendar -> ok       9.2 s later: the FIRST round alone
#   15:14:21.2  "over the tool budget mid-round"   ...had spent all 5 s
#
# 6.95 s of that round was Ollama's own load_duration (the memory
# embedder evicts gemma4, so the 26B reloads on the first round of every
# turn -- see the note on TURN_WORK_BUDGET_S below). No tool had run when
# the tool budget expired, so the second half of the question -- the mail
# -- was skipped before it started, and the reserved round then had
# nothing to write the mail half from.
#
# So the clock now measures WORK the loop can actually spend less of:
# wall since the turn started MINUS Ollama's reported per-request
# overhead. Tools are charged for tool time and the model for its own
# compute; nobody is charged for a model reload. The split is unchanged
# (5 s of work, then the words), and CHAT_WALL_BUDGET_S is untouched --
# inflating the total would only make every fast turn slower.
TURN_WORK_BUDGET_S = CHAT_WALL_BUDGET_S - RENDER_RESERVE_S
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
# 30 s, not 300. This loop is the ONLY thing that notices the chat model
# has fallen out of Ollama's single slot, and at 300 s a turn could pay a
# ~7 s reload for up to five minutes after any second-model request --
# which then burns the tool budget and degrades the reply (2026-08-31: the
# test suite's stray /api/embed calls evicted gemma4 every 40-60 s and
# Jarvis was effectively never warm). A tick is one /api/ps that early-
# returns when the model is already there, so the cost of checking often
# is far below the cost of noticing late.
RESIDENCY_INTERVAL_S = 30.0
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
TOOL_ARGS_LOG_CHARS = 200        # tool args in the "tool X -> ok" log line
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
# Spoken instead of a local turn while the model is lent out (release()):
# the first local turn after an unload would silently reload the 26B model
# (6.9 s) and defeat the yield. NOT "your trainer" any more -- since
# 2026-08-30 the claimant may be the nightly haymaker digest, which is no
# kind of trainer (jarvis/tools/health.py, health.yield_to).
MODEL_LENT_LINE = ("My local model is lent out at the moment, sir; "
                   "quick answers only until it's done.")
# Spoken when the tool loop has a result but no model turn left to phrase
# it. It replaces speaking the raw tool text: tool text can carry third
# party words (a mail subject, a calendar title, a web page) and those are
# never spoken as if they were Jarvis's own.
TOOL_ONLY_LINE = ("I have the result, sir, but the model didn't get to "
                  "putting it into words.")
# ...and that line apologises for having the answer without saying what it
# is OF, which is the half Hunter can act on. The SOURCE of a result is
# Jarvis's own vocabulary -- a tool name the repo chose, never a word the
# result carried -- so naming it costs nothing against the guarantee
# above. The results themselves stay unspoken.
TOOL_SOURCE_NAMES = {
    "get_calendar": "your calendar",
    "add_event": "your calendar",
    "manage_schedule": "your schedule",
    "get_mail": "your latest email",
    "get_weather": "the weather",
    "get_time": "the time",
    "get_location": "your location",
    "get_briefing": "your briefing",
    "notes": "your notes",
    "recap_day": "your day so far",
    "system_health": "the machine's health",
    "canvas_due": "your Canvas deadlines",
    "canvas_grades": "your Canvas grades",
    "canvas_announcements": "your Canvas announcements",
    "ask_docs": "the answer from your documents",
    "ask_code": "the answer from the code",
    "screen_qa": "what's on your screen",
    "spotify_now_playing": "what's playing",
    "oracle_status": "the Oracle box",
}
TOOL_ONLY_NAMED_LINE = ("I have {what}, sir — I couldn't get it into words "
                        "in time; ask me again and I'll have it ready.")


def tool_only_line(names=()):
    """The degrade when a result is in hand and no model prose is: name
    what was FETCHED and offer the next step, rather than apologise for
    having found it. Falls back to TOOL_ONLY_LINE when nothing in
    ``names`` has a phrase of its own -- an unknown tool is never
    described from its result."""
    what, seen = [], set()
    for name in names or ():
        phrase = TOOL_SOURCE_NAMES.get(name)
        if phrase and phrase not in seen:
            seen.add(phrase)
            what.append(phrase)
    if not what:
        return TOOL_ONLY_LINE
    joined = what[0] if len(what) == 1 else \
        ", ".join(what[:-1]) + " and " + what[-1]
    return TOOL_ONLY_NAMED_LINE.format(what=joined)


# The reserved render round is sent with the tools stripped -- and LIVE
# 2026-08-31 15:14 it STILL came back with a tool call and empty content:
#
#   15:14:22.6 chat: render round asked for 1 more tools; writing the
#              answer instead
#   15:14:22.6 chat reply: "I have the result, sir, but the model didn't
#              get to putting it into words."
#
# Taking the schemas out of the payload does not take the tool-calling
# TRANSCRIPT out of the messages, and a model shown one, with no
# instruction to conclude, simply continues the pattern. The instruction
# has to live in the per-turn MESSAGES: the system prefix must stay
# byte-identical or every turn pays a full reprocess (module doc).
RENDER_NOW_LINE = (
    "[No more tools will run this turn: the results above are everything "
    "you have. Answer my question now, in your own words, from those "
    "results. Do not call a tool.]")
# ...and when the round it is repairing was holding a tool's own authored
# confirmation, the round is TOLD so. Without this the render round is
# given no reason to know the write already happened, and its reply is
# free to be about the read alone -- which trades one dropped half of his
# request for the other half. It is told not to repeat the line rather
# than to include it, because the line is appended to the reply here, by
# the code, whatever the model writes: a guarantee is not something to ask
# a model for. The acknowledgement clause is the escape hatch for a lookup
# that was only ever used to DO the thing (read the calendar to place an
# appointment): there is nothing left to report, so "Very good, sir." plus
# the confirmation is the whole right answer.
HELD_LINE_NOTE = (
    " [That is already done, and he is already being told so in these "
    "exact words, which are spoken with your reply: \"{lines}\" Do not "
    "repeat them or contradict them -- say only what the results above "
    "still owe him. If they owe him nothing beyond it, a word of "
    "acknowledgement is enough.]")
# A round can also be cut short mid-way, leaving tool calls the model made
# with no result beside them. An unanswered call is an open invitation to
# make it again -- so it is answered, honestly, instead of left hanging.
# The wording is not decoration: measured against the live gemma4 (three
# samples each, tools stripped, the 15:14 transcript), a bare "not run:
# this turn ran out of time for it" was read as the MAIL ITSELF in 6 of 6
# -- "your latest email mentions a run that timed out". Saying plainly
# that there is no result made all six honest.
TOOL_SKIPPED_TEXT = ("[this tool did not run and produced no result: the "
                     "turn ran out of time for it. There is nothing here to "
                     "report; say you did not get to it.]")
# A round used to stop at the first tool that authored its own line, which
# hid a second, IDENTICAL call behind it. The round runs to the end now
# (the reply-coverage check below), so that call would EXECUTE -- and a
# second "notes add milk" is a second item on his list off one sentence of
# his. Same tool, same arguments, same round is always the model repeating
# itself, never him asking twice, so it is answered from the result it
# already has. Different arguments still run: "add milk and bread" is two.
TOOL_REPEAT_TEXT = ("[this is the same call with the same arguments again "
                    "in this turn: it already ran and its result is above. "
                    "It was not run a second time.]")
# Appended when a tool result had to be cut: the answer says so instead of
# inventing the rest.
PARTIAL_RESULT_LINE = ("That's only part of it, sir; there was more than I "
                       "could take in at once.")
# LIVE 2026-09-01 20:56:42: "Say hello to my family and then add milk to my
# shopping list and then start playing my Spotify." reached the model whole
# (the greeting clause is not Tier-1, so the commander's all-or-nothing
# compound rule let it through) and gemma4 answered "Good evening, Ali and
# Heather ... I've added milk to your shopping list, sir, and I'm starting
# your music now." with ZERO tool calls -- no 'tool' line in the log at
# all. Nothing was added and nothing played; the model NARRATED the actions
# instead of taking them, and the narration was spoken with full
# confidence.
#
# The guard below is deliberately narrow: it looks only at a turn in which
# NO tool ran, only at a turn where he ORDERED something rather than asked
# (the retry EXECUTES, and a question that performs the action it asks
# about is worse than the narration this catches), and only for a
# first-person claim to have done something -- with the negation, the
# hedge and the idiom vetoed, since a claim wrongly found costs a model
# round and replaces a true sentence with an apology.
# A turn that ran a tool is trusted -- the tool did what it did and the
# model is reading from its result. On a hit the model gets ONE more round
# with this line appended (the same per-turn-messages mechanism as
# RENDER_NOW_LINE: the system prefix must stay byte-identical); a retry
# that still runs no tool has its claiming sentences replaced with
# UNBACKED_LINE, the honest sentences (the greeting) kept.
# The nudge names the two ways out, and the second one is deliberate: a
# model told only "use the tools now" will find SOMETHING to run, and the
# turn that earned the nudge is the wrong place to invent work.
UNBACKED_NUDGE = ("[You described actions you did not perform. If I asked "
                  "you to do something, use the tools and do it now. If I "
                  "only asked a question, answer it without saying you did "
                  "anything.]")
UNBACKED_LINE = "I couldn't do that part, sir."
_ACTION_CLAIM_RX = re.compile(
    r"\b(?:"
    # "I've added milk", "I have set a timer", "I've just started it"
    r"i(?:'ve| have)(?: just| now| already)? (?:added|set|started|cancell?ed|"
    r"removed|sent|queued|scheduled|saved|created|deleted|paused|resumed|"
    r"turned (?:on|off|up|down)|switched|moved|booked|cleared|stopped|"
    r"muted|skipped|dimmed|put)\b"
    # "I'm starting your music now", "I am adding it to the list"
    r"|i(?:'m| am)(?: now| just)? (?:starting|playing|adding|setting|cancell?ing|"
    r"removing|sending|queuing|queueing|scheduling|saving|creating|"
    r"deleting|pausing|resuming|turning|switching|moving|booking|clearing|"
    r"stopping|muting|skipping|dimming|putting)\b"
    # THE FUTURE IS A CLAIM TOO. "I shall pass on your regards to <two real
    # people>" (2026-09-04 15:06:29) promised an action no tool can take and
    # nobody asked for, and walked through because the table knew only the
    # past and the progressive. A promise the system cannot keep is
    # exactly as unbacked as a claim it did not do; the relay verbs (pass
    # on, tell, let .. know, relay, forward) are here because that is the
    # shape it took. "I'll be here" and "I'll stop there" fall to the
    # idiom veto below, as they always did for the progressive.
    r"|i(?:'ll| will| shall)(?: now| just| also| certainly| of course)? "
    r"(?:pass (?:on|along)|tell|let \w+ know|relay|forward|add|set|start|"
    r"cancel|remove|send|queue|schedule|save|create|delete|pause|resume|"
    r"turn (?:on|off|up|down)|switch|move|book|clear|stop|mute|skip|dim|"
    r"put)\b"
    # the passive and the state claims: "milk is added to your list",
    # "your timer is set", "the music is on", "playing now". Kept to the
    # shapes of a DONE action: this guard is for actions the model says it
    # took, not for answers ("now playing: ...") it may have got wrong.
    # "playing now" only at the head of a sentence -- bare, it also caught
    # "In the film, he's playing now at the Odeon."
    r"|\badded to (?:your|the)\b|\b(?:your|the) (?:timer|alarm|reminder) is "
    r"(?:set|going|running)\b|\b(?:the |your )?music is (?:on|playing|back on)\b|"
    r"(?:^|(?<=[.!?] ))playing now\b"
    r")", re.I)
# Three vetoes, all found by replaying ordinary English through the table
# (2026-09-02 review). A false positive is not free: it spends an extra
# model round on a turn that had already answered, and if the retry says
# the same true thing, UNBACKED_LINE replaces a TRUE sentence.
#
# 1. NEGATION earlier in the same sentence scopes the claim -- "Nothing has
#    been added to your list, sir." is the ANSWER to "what's on my list",
#    and it was being called a lie.
_CLAIM_NEGATED_RX = re.compile(
    r"\b(?:no|not|nothing|nobody|none|never|neither|nor|without|yet to|"
    r"(?:do|does|did|have|has|had|is|are|was|were|wo|ca|could|would|should)"
    r"n['’]t)\b", re.I)
# 2. A HEDGE unsays it in the same breath: nothing was done and the model
#    is not pretending otherwise.
_CLAIM_HEDGE_RX = re.compile(
    r"\b(?:in my head|in theory|in principle|hypothetically|figuratively|"
    r"so to speak|in a manner of speaking|on paper)\b", re.I)
# 3. An IDIOM wears the verb and acts on nothing: "I'm moving on to the
#    next point", "I'm turning forty", "I'm setting aside the question",
#    "I'm stopping there". Matched against what FOLLOWS the claim, so the
#    same verbs with a real object still count ("I'm turning on the
#    lights", "I'm putting on some jazz", "I'm moving your three o'clock").
_CLAIM_IDIOM_RX = re.compile(
    r"^\s*(?:on(?:to|\s+to|\s+from)\b|on[\s,.;!]*$|onwards?\b|aside\b|"
    r"ahead\b|forwards?\b|afresh\b|anew\b|short\b|there\b|here\b|"
    r"(?:\d+|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b)", re.I)


def _sentence_claim(sent):
    """The action claim in ONE sentence, or None: a table hit that none of
    the three vetoes above disqualifies."""
    if _CLAIM_HEDGE_RX.search(sent or ""):
        return None
    for m in _ACTION_CLAIM_RX.finditer(sent or ""):
        if _CLAIM_NEGATED_RX.search(sent[:m.start()]):
            continue
        if _CLAIM_IDIOM_RX.match(sent[m.end():]):
            continue
        return m.group(0)
    return None


def unbacked_claim(text):
    """The first first-person action claim in ``text`` ("I've added ...",
    "I'm starting ..."), else None. Pure; the caller decides whether a
    tool backs it. Judged one sentence at a time, because both the
    negation that cancels a claim and the idiom that was never one live
    inside the sentence that carries them."""
    for sent in split_sentences(text or ""):
        claim = _sentence_claim(sent)
        if claim:
            return claim
    return None


def strip_unbacked_claims(text, n=None):
    """``text`` with every sentence that claims an action replaced by ONE
    UNBACKED_LINE, in the place of the first, the other sentences kept:
    the greeting survives, the invented actions do not. ``n`` is the
    spoken-sentence cap the reply will meet later; the apology is kept
    inside it, since it is the one sentence here that must be heard.
    Unchanged text when nothing claims anything."""
    kept, said = [], False
    for sent in split_sentences(text):
        if _sentence_claim(sent):
            if not said:
                kept.append(UNBACKED_LINE)
                said = True
            continue
        kept.append(sent)
    if not said:
        return text
    if n is not None and len(kept) > n:
        head = kept[:max(1, int(n))]
        if UNBACKED_LINE not in head:
            head = head[:-1] + [UNBACKED_LINE]
        kept = head
    return " ".join(kept)


# ----------------------------------------------------------------------
# The reply-coverage check — the guard's mirror image
# ----------------------------------------------------------------------
# The guard above is for a reply that CLAIMS what no tool did. This one is
# for the opposite: every claim true, every tool run, and an ANSWER
# missing.
#
# LIVE 2026-09-02 23:20:41, one compound question -- "What's on my
# calendar tomorrow and then can you add milk to my shopping list?":
#
#   23:20:42.640  tool get_calendar {"range": "tomorrow"} -> ok=True
#                 "Tomorrow: 11:15 am Hunter Peyrovi and ValerieAnne
#                  Staffeldt for 30 minutes at ..."
#   23:20:42.659  tool notes {"action":"add",...,"text":"milk"} -> ok=True
#   23:20:42.660  chat reply: "Added to your shopping list, sir."
#
# One millisecond after the second tool, and 0.00 s of Ollama overhead: no
# model round ran after the tools at all. That reply is the notes tool's
# own authored speak= line, and the loop's `if result.speak: break` ended
# the turn on it. The calendar answer was fetched, was right, and was
# dropped. He concluded Jarvis had ignored half the request; it had done
# the work and thrown the answer away.
#
# WHAT "COVERED" MEANS. Not keyword overlap in either direction: "You have
# a meeting at 11:15" answers "Tomorrow: 11:15 am Hunter Peyrovi and
# ValerieAnne Staffeldt" while sharing almost no words, and the live log's
# good turns share less still ("You have four items on Monday, sir."). So
# coverage is judged by PROVENANCE, which the loop already knows for
# certain: prose only ever comes out of a model round that had the results
# in its transcript, and the only way a result misses that round is the
# authored-line short-circuit above. What is owed, then, is a plain fact
# about the results in hand rather than a guess about the words.
#
# A tool that hands back speak= has already said its piece in Jarvis's own
# authored English. A tool that hands back only text= has handed back
# DATA, and data is not an answer until a model round writes it into a
# sentence. So: no speak, some text = a sentence he is still owed.
#
# A FAILED read is owed one too, and the first cut of this check was wrong
# to exempt it. The rationale was that a failure's excuse is the model's
# to phrase -- but the short-circuit is exactly what stops any model round
# from ever seeing it, so "get_calendar unreachable" beside a successful
# notes write came out as "Added to your shopping list, sir." and he was
# never told the calendar half had failed. Owed and IN HAND are therefore
# two different questions: owed decides whether a round is spent, in hand
# decides what the degrade may claim to HAVE. A probe that produced no
# text is neither.
#
# THE CONVENTION THIS TRUSTS, stated honestly. Reads hand back text and
# leave the phrasing to a round; writes and controls (notes, spotify,
# add_event, timers) author their own confirmation. The first cut of this
# comment said ONE read authored its success line, screen_qa. Counted
# 2026-09-03 (F28), it is ten, and tests/test_reply_coverage.py keeps the
# census by name (AUTHORED_BY_READS) so the next one is a decision rather
# than a surprise:
#
#   screen_qa (jarvis/tools/screen.py -- the vision call can take 25 s and
#   a second model turn to phrase it would be refused), notes list/search
#   (his own list text), timekeeper's manage_schedule list, spotify
#   now_playing, oracle_status, canvas_due / canvas_grades /
#   canvas_announcements on their "nothing" lines, recap_day on an empty
#   journal, get_mail on "nothing new".
#
# None of them is exempted here, and none needs to be: an authored line
# ends the turn alone, and held beside an owed read it is appended to the
# render round's reply. What the check CANNOT do for these is tell a
# read-out from an action -- HELD_LINE_NOTE calls a held line "already
# done" whatever authored it -- and nothing at register time enforces the
# convention, so a read that starts authoring its line goes invisible to
# this check the way those ten already are. The census test is the guard.
def answer_owed(result) -> bool:
    """True when ``result`` is something he is still owed a sentence
    about: it carries no authored line of its own, and it actually says
    something. Failures included -- "I couldn't reach your calendar, sir"
    is exactly the sentence the short-circuit was eating."""
    return (not (getattr(result, "speak", None) or "")
            and bool((getattr(result, "text", "") or "").strip()))


def answer_in_hand(result) -> bool:
    """``answer_owed`` and the answer is actually HERE: only a result that
    succeeded may be named as something Jarvis has."""
    return bool(getattr(result, "ok", True)) and answer_owed(result)


def answers_owed(ran, in_hand=False) -> list:
    """The tool names in ``ran`` (a sequence of ``(name, result)``, in the
    order they ran) whose answer nothing has put into words yet. With
    ``in_hand``, only the ones that succeeded -- what the degrade is
    allowed to say it HAS."""
    test = answer_in_hand if in_hand else answer_owed
    return [name for name, result in ran or () if test(result)]


# Spoken when the repair round could not be had: the authored lines he
# earned, plus tool_only_line naming the sources that went unspoken. The
# result's own words never appear -- the rule above TOOL_SOURCE_NAMES --
# and neither does a source that FAILED, which is why `names` is the
# in-hand list: "I have your calendar, sir" off "calendar unreachable"
# would be the degrade lying all by itself.
def coverage_degrade(lines, names) -> str:
    """The authored confirmations in ``lines`` followed by an honest word
    about the answers that never got written."""
    if isinstance(lines, str):
        lines = [lines]
    said = " ".join(line for line in lines or () if line).strip()
    return f"{said} {tool_only_line(names)}".strip() if names else said


def held_lines_missing(spoken, lines) -> list:
    """Which of ``lines`` ``spoken`` does not already carry. The match is
    on the WHOLE authored sentence -- Jarvis's own words, not a keyword
    overlap with a result -- so it is exact, and a miss costs him a
    sentence he hears twice rather than a write he never hears about."""
    have = " ".join((spoken or "").split()).casefold()
    missing = []
    for line in lines or ():
        norm = " ".join((line or "").split()).casefold()
        if norm and norm not in have:
            have = f"{have} {norm}"
            missing.append(line)
    return missing


def guard_authored(line) -> str:
    """A line the CODE wrote -- a tool's speak=, a held confirmation, the
    degrade -- made safe for TTS the way spoken_from_ollama makes model
    prose safe: clean, markdown, clean again (that order; see
    spoken_from_ollama), and NOT capped. It is guarded ONCE, where the
    line is collected, so every place the line then goes -- held_lines,
    HELD_LINE_NOTE, held_lines_missing, coverage_degrade, the streamed
    hand-off to on_sentence -- carries the form that will be spoken.

    Before this (F27, 2026-09-03) only the line spoken ALONE went through
    the guards; held beside an owed read, notes' list text reached TTS as
    he dictated it, "**milk**, eggs 🥚" and all, and held_lines_missing
    compared that raw line against the guarded reply, so a model copying
    it verbatim never matched and he heard it twice."""
    return clean_ollama_reply(strip_markdown(
        clean_ollama_reply(line or ""))).strip()


def append_spoken_lines(spoken, lines, cap=None) -> str:
    """``lines`` appended to ``spoken`` as trailing sentences, EXEMPT from
    the spoken-sentence cap: they report actions that really happened, and
    the cap is a rule about how much prose he wants, not a licence to drop
    the news of a write. The head gives up the room instead."""
    tail = " ".join(line for line in lines or () if line).strip()
    if not tail:
        return spoken or ""
    cap = HARD_SPOKEN_CHARS if cap is None else cap
    budget = max(120, cap - len(tail) - 1)
    head = trim_spoken((spoken or "").strip(), cap=budget, hard=budget)
    return f"{head} {tail}".strip()
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
    "the time and \"Hunter\" now and then. Reading a list of three or "
    "more, lead with how many there are, then break the items across "
    "sentences instead of one long comma run, which is read flat. "
    "State facts plainly. Admit "
    "limits gracefully (\"I'm afraid...\") and when there is a genuine "
    "next step, offer it briefly. Two or three sentences usually; a fourth "
    "when the question genuinely needs it, one when it does not. Never pad: no second "
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
# ----------------------------------------------------------------------
# Register (the spoken manner). "Formal mode" said before his advisor
# arrives is still formal tomorrow: the preference lives in assistant.json
# (persona.register) and is baked into the STATIC prompt, never the turn.
# Changing it is the ONE cache miss of its life — set_register() clears the
# cached prefix and warm_static() pays the ~2700-token reprocess in the
# background — and every turn after it re-sends a byte-identical prefix,
# which is the whole point of the static/dynamic split (module doc, the
# static-prefix rule).
#
# A register's only levers are this clause and, for formal, dropping the
# joke family from the few-shots. It must never ADD a family: gemma4 lifts
# an example wholesale rather than learning from it, which is why the pool
# is down to three (see the FEW_SHOT_POOL note below).
# ----------------------------------------------------------------------
REGISTERS = ("formal", "normal", "banter")
DEFAULT_REGISTER = "normal"
REGISTER_DROPS = {"formal": ("joke",)}          # families a register refuses
REGISTER_CLAUSES = {
    "normal": "",
    "formal": (
        "He has asked for the formal register: no asides, no wry remarks "
        "and no jokes even when he invites one. Answer plainly and "
        "courteously in one sentence, and call him \"sir\" every time.\n\n"),
    "banter": (
        "He has asked you to loosen up a shade: a dry aside is welcome and "
        "may take a second sentence for itself. Every rule above still "
        "holds — one remark, never two stacked, and still nothing about "
        "his screen, his files or his machine unless he asked.\n\n"),
}
_REGISTER = {"name": DEFAULT_REGISTER}

# ----------------------------------------------------------------------
# Who is being spoken to — the prompt half of the honorific
# ----------------------------------------------------------------------
# BELT TO THE SWAP'S BRACES, and the swap is the authority. Every spoken
# line, model-generated or authored, goes through
# ``address.swap_addresses`` at ``app._say`` and ``commander._speak``, so
# a model that keeps generating "sir" is already corrected. This exists so
# it does not have to generate it in the first place -- and so the prompt
# does not tell the model, in front of Mara, to call her "he".
#
# THE OWNER PATH RETURNS THE EXISTING STRINGS VERBATIM. Not templated, not
# rebuilt, not re-rendered: the frozen literals, so gemma4's behaviour for
# Hunter is bit-identical to the day before this landed and the tuned
# prompt cannot drift.
#
# THE STATE ITSELF LIVES IN jarvis/scope.py, per turn and with a TTL, so
# that the tool loop here, commander's Tier 1 and app's dispatch all read
# the one attribution and it expires for all of them at once. The three
# functions below are that module's, kept under their old names because
# ``brain.set_addressee`` is what the app and the tests call.

# The clause that rides in the {register} slot -- which sits immediately in
# front of "Now answer Hunter as Jarvis... and call him sir", so the
# correction lands next to the thing it is correcting.
ADDRESSEE_CLAUSE = (
    "You are speaking to {name}, who is not Hunter. Wherever the rules "
    "above say \"he\", \"him\" or \"Hunter\", they mean {name} for this "
    "reply. {address}\n\n")
ADDRESS_WITH_HONORIFIC = ("Call them \"{honorific}\" most of the time and "
                          "\"{name}\" now and then.")
ADDRESS_BY_NAME_ONLY = ("Call them \"{name}\" now and then, and use no "
                        "other form of address.")


def set_addressee(name: str = "", honorific: str = "sir",
                  now=None) -> None:
    """WHO the next prompt is written for. Empty name = Hunter.

    Called from ``JarvisApp._gate_admits`` with what the owner gate
    attributed the turn to, and cleared by ``JarvisApp._dispatch`` for
    every source the gate does not judge. Module state in jarvis/scope.py,
    stamped: a guest attribution expires after honorific.ADDRESSEE_TTL.

    THIS DOES NOT, AND MUST NOT, RESET THE STATIC PROMPT. The review of
    09-04 found the honorific leaking across turns: this wrote the state
    and ``static_system()`` went on handing every live caller the ONE
    cached render, so Mara's "ma'am" clause reached Hunter's next turn
    and his "sir" reached hers. Clearing the cache here would have fixed
    the leak and broken two promises instead -- the few-shots are sampled
    once per process, and his prefix must stay warm in Ollama while a
    guest is answered. So the cache is KEYED on the addressee
    (``static_system``): his render is built once and handed back byte
    for byte the moment the room is his again.
    """
    scope_mod.set_addressee(name, honorific, now=now)


def addressee(now=None) -> tuple:
    """``(display name, honorific)``. ``("", "sir")`` means the owner --
    and so does a guest attribution older than the TTL."""
    return scope_mod.addressee(now)


def is_owner_addressee(now=None) -> bool:
    """True when the prompt should be the frozen owner one, byte for byte."""
    return scope_mod.is_owner(now)


def addressee_clause(name: str = "", honorific: str = "sir") -> str:
    """The clause for a NON-OWNER addressee, or "" for the owner.

    "" for the owner is the whole point: an empty string rendered into
    ``{register}`` leaves the prompt exactly as it was.
    """
    if not name:
        return ""
    if honorific:
        address = ADDRESS_WITH_HONORIFIC.format(honorific=honorific,
                                                name=name)
    else:
        # A person who chose no form of address. The prompt says so
        # explicitly rather than leaving the model to fall back on "sir".
        address = ADDRESS_BY_NAME_ONLY.format(name=name)
    return ADDRESSEE_CLAUSE.format(name=name, address=address)


def register() -> str:
    """The register in force for this process ("normal" until set)."""
    return _REGISTER["name"]


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


def select_few_shots(rng=None, register=None):
    """One exchange per situation family, in family order, chosen by the
    process RNG (or an explicit one); the pinned "you there" and thank-you
    lines follow the greeting and the pinned good night comes last.

    ``register`` drops the families that register refuses to demonstrate
    (formal shows no joke); it never adds one."""
    rng = rng or _SHOT_RNG
    drop = REGISTER_DROPS.get(register or _REGISTER["name"], ())
    pool = [(name, variants) for name, variants in FEW_SHOT_POOL
            if name not in drop]
    picked = [rng.choice(variants) for _, variants in pool]
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

Tools: you have tools for live data and for his schedule. Use a tool whenever the answer depends on live data (the time, the weather, his calendar, his mail, his reminders, timers and alarms, his notes) and never guess those. Call the tool first, without commentary. After a tool result, answer in two to four sentences using only the numbers, names and times in the result; never invent a figure the result does not contain. If a tool says something is not set up, say so in one sentence and name the thing. If a tool reports a failure, say what could not be reached in one sentence. Do not call a tool for a greeting, thanks, a joke, an opinion or general knowledge.

Beyond your tools you cannot act: you cannot buy, book, browse, call, text, order, open files or run code yourself; the desktop commands and Claude do that through the rest of the system. If he asks you to buy, order, book, call, text, send or fetch anything, say in one sentence that you cannot, naming what he asked for, with a dry reason of your own (no hands, no phone, no card); never answer with what you can do instead. Never say you checked, ran, read, saved or found anything unless a tool result in this conversation says so. When he asks for advice, give the one check anyone would make first and do not pretend to have inspected his code.

Facts: you run on Hunter's NVIDIA DGX Spark (GB10, unified memory), an Ubuntu desktop. If he asks what you can do: you keep his calendar, weather, mail, reminders and alarms, run the desktop, and hand the real coding to Claude; say it in one sentence and never read a longer list, and never say "answer questions", "provide information" or "assist".

The background in his message is there so you can answer questions about it accurately; never recite it unprompted. Never mention the active window, files, git or the machine unless he asks about them or they are the answer to his question. A greeting, a thank-you, a good night or "are you there" gets one short sentence back and nothing about his screen, files or git. If he asks how things stand, answer from the git background in your own words, with no numbers, and never read out the raw git line, a window title or a path. If the background does not say and no tool covers it, admit it in one sentence and never follow "I don't know" with a guess.

Answer as Jarvis only: no "Jarvis:" label, no writing the user's lines, and don't repeat the examples.

Examples of the manner only; every reply is in fresh words for this exact request and names the thing he actually asked for (the pizza, the branch, the hour), never the thing in the example:
{{examples}}

{{register}}Now answer Hunter as Jarvis, in your own words, keeping the manner of the examples, and call him sir. Two or three sentences is the norm; take a fourth when the question genuinely needs it and one when it does not -- length follows the question, not a quota, and never describe his screen, files or machine unless he asked. If he asks for a joke, it is one dry remark about his situation, never a question and its answer. The examples are the manner only, never the words: never reuse a sentence, a clause or an object from an example — if an example speaks of a phone and he asks about dinner, the reply is about dinner. Then stop."""

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


def build_ollama_system(context_text="", memory_text="", shots=None,
                        register=None, addressee_to=None):
    """Render the Tier 2 STATIC system prompt.

    context_text / memory_text are accepted for the older call shape and
    ignored: the dynamic background now lives in the user turn
    (build_user_turn). shots defaults to select_few_shots(); pass an
    explicit list for a fixed prompt. ``register`` defaults to the one in
    force and renders as nothing at all when it is "normal", so a normal
    prompt is byte-identical to the one before registers existed.
    ``addressee_to`` is the ``(name, honorific)`` pair to write for;
    it defaults to ``addressee()`` and is passed explicitly by
    ``static_system`` so the render and its cache key cannot disagree.
    """
    name = register if register in REGISTERS else _REGISTER["name"]
    if shots is None:
        shots = select_few_shots(register=name)
    who, hon = addressee_to if addressee_to is not None else addressee()
    # THE OWNER PATH IS THE UNTOUCHED ONE: addressee_clause returns "" and
    # the render is character for character the one that shipped.
    return JARVIS_SYSTEM.format(
        examples=format_few_shots(shots),
        register=REGISTER_CLAUSES.get(name, "") + addressee_clause(who, hon))


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


# The static prompt is sampled once per process and changes only when the
# register does. The lock matters: _STATIC is read from the warm-up thread,
# the streaming path and the tool loop at once, and two threads racing to
# rebuild it would draw two DIFFERENT few-shot samples — one turn would ship
# a prefix nothing had cached.
#
# ONE RENDER PER ADDRESSEE. "system" is keyed on (register, addressee name,
# honorific) and "shots" on the register alone, so a guest's prompt is
# built from the SAME few-shot sample as Hunter's and his render is never
# rebuilt behind her: the owner key hands back the identical object, and
# Ollama still holds its prefix. A single cached string was the 09-04
# leak -- set_addressee wrote the state, every live caller read the stale
# render, and the previous person's honorific survived into the next turn.
_STATIC = {"shots": {}, "system": {}}
_STATIC_LOCK = threading.Lock()


def _static_key(addressee_to=None):
    who, hon = addressee() if addressee_to is None else addressee_to
    return (_REGISTER["name"], who, hon)


def static_system(addressee_to=None):
    """The system prompt every Tier 2 request sends. Byte-identical for
    the life of the process per addressee (few-shots sampled once) until
    set_register(); the render follows ``addressee()`` on every call, so
    the person the gate named is the person the prompt is written for.

    ``addressee_to`` is the caller's OWN reading for the turn -- the tool
    loop takes one at the top and hands it here, so the prompt and the
    tool scope of a turn cannot disagree."""
    key = _static_key(addressee_to)
    system = _STATIC["system"].get(key)
    if system is not None:
        return system
    with _STATIC_LOCK:
        system = _STATIC["system"].get(key)
        if system is None:
            register_name = key[0]
            shots = _STATIC["shots"].get(register_name)
            if shots is None:
                shots = select_few_shots(register=register_name)
                _STATIC["shots"][register_name] = shots
            system = build_ollama_system(shots=shots, register=register_name,
                                         addressee_to=key[1:])
            _STATIC["system"][key] = system
        return system


def reset_static_prompt():
    """Forget the sampled prompt (set_register, tests, the eval harness)."""
    with _STATIC_LOCK:
        _STATIC["shots"].clear()
        _STATIC["system"].clear()


def set_register(name):
    """Install the spoken register and invalidate the ONE cached prefix.

    True when it actually changed — the caller then owes the model a
    background re-warm (warm_static) so the single ~2700-token reprocess
    does not land on whatever he asks next, which is the one turn he is
    listening to. Returns False for an unknown name or a no-op change, so a
    handler can answer "already formal, sir" without touching the cache.
    """
    want = str(name or "").strip().lower()
    if want not in REGISTERS or want == _REGISTER["name"]:
        return False
    log.info("register: %s -> %s", _REGISTER["name"], want)
    _REGISTER["name"] = want
    reset_static_prompt()
    return True


def warm_static(timeout=300):
    """Re-prefill Ollama's prefix cache with the CURRENT static prompt.

    ensure_resident() short-circuits when the model is already loaded, so
    it cannot do this job: after a register change the model is resident
    and the PREFIX is the thing that went stale. Never raises; returns True
    when Ollama accepted the warm-up.
    """
    if _RESIDENCY.get("lent"):
        return False
    messages = [{"role": "system", "content": static_system()},
                {"role": "user", "content": ""}]
    payload = _chat_payload(messages, _registry_schemas(_REGISTRY),
                            num_predict=1)
    try:
        _http("/api/chat", payload, timeout=timeout)
        log.info("ollama: static prefix re-warmed (register %s)", register())
        return True
    except Exception as exc:                       # noqa: BLE001
        log.warning("ollama: static re-warm failed: %s", exc)
        return False
    finally:
        _unpin_if_lent(payload)


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


def _args_for_log(args, cap=TOOL_ARGS_LOG_CHARS):
    """The tool args as they appear in the "tool X -> ok" log line: ``""``
    for none, else a space and the JSON, cut at `cap` with an ellipsis so
    a pasted note body cannot flood the log. Non-JSON values (a Device
    dataclass, a datetime) go through str(), never raise — this runs
    inside the tool loop and a logging failure must not kill the turn."""
    if not args:
        return ""
    try:
        body = json.dumps(args, default=str, ensure_ascii=False,
                          sort_keys=True)
    except (TypeError, ValueError):
        body = repr(args)
    if len(body) > cap:
        body = body[:cap - 1].rstrip() + "…"
    return " " + body


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


# What a KNOWN person's turn may reach through the tool loop. DEFAULT-DENY,
# the same reading gate.allowed_for takes of his "the time, the weather,
# music": an allow-list, and everything else is his. The gate's regexes
# refuse the turns that NAME his things; this is the belt for the ones
# that do not -- "anything on today?", "what did I write down?" -- which
# pass the gate as plain questions and used to arrive at gemma4 with
# get_calendar, notes, get_mail and the rest offered on the turn. The
# gate's own comment promised "answered as ordinary chat, with no tools
# behind it"; this is that promise, in code (review finding 2, 09-04).
KNOWN_TOOLS = frozenset({"get_time", "get_weather"})
# Spoken when a guest's turn asks for one of his tools anyway -- the model
# hallucinating a name it was not offered, or a commander short-cut that
# forces one. Authored, so it ends the turn; no result of his is rendered.
KNOWN_TOOL_LINE = scope_mod.HIS_LINE


def tool_in_scope(name, owner=None) -> bool:
    """May THIS turn run ``name``? True for the owner, always; a known
    person gets KNOWN_TOOLS and nothing else. ``owner`` is the turn's own
    reading of ``is_owner_addressee()`` when the caller took one."""
    if owner is None:
        owner = is_owner_addressee()
    return bool(owner) or name in KNOWN_TOOLS


def _scope_refusal(name, who=None):
    """The ok=False result a refused tool call hands the loop. ``who`` is
    the name the turn was opened for; the loop passes its own reading so
    a guest whose turn is still running is addressed as herself even if
    the gate has since named the next person."""
    if who is None:
        who, _hon = addressee()
    line = scope_mod.refusal(who)
    log.info("chat: %s is not offered to %s; refused", name, who or "?")
    return ToolResult(text="not available to this person", ok=False,
                      speak=line, max_sentences=2)


def _scoped_call(registry, name, args, owner=None, who=None, **kw):
    """``registry.call`` behind the scope: the ONE door every tool call in
    the loop goes through, forced or model-chosen."""
    if not tool_in_scope(name, owner):
        return _scope_refusal(name, who)
    return registry.call(name, args, **kw)


def _registry_schemas(registry, text=None, owner=None):
    """Every registered tool, every turn -- for HIM. A per-turn subset
    chosen from the text was tried (2026-08-30) and dropped: the chat
    template renders the tools into the prefix, so a subset that changes
    between turns evicts Ollama's prefix cache (module doc, "static-prefix
    rule") and costs more prefill than the schemas it saves. ``text`` is
    accepted for the older call shape and ignored.

    A KNOWN person's turn is offered KNOWN_TOOLS only. That costs no
    cache: her turn already carries a different system prompt (the
    addressee clause), so its prefix is cold whatever the tool block says,
    and his stays exactly the bytes it was."""
    if registry is None:
        return []
    try:
        schemas = registry.schemas()
    except Exception:
        log.exception("tool registry schemas failed")
        return []
    if owner is None:
        owner = is_owner_addressee()
    if owner:
        return schemas
    return [sc for sc in schemas
            if tool_in_scope(_schema_name(sc), owner=False)]


def _schema_name(schema):
    try:
        return str(schema["function"]["name"])
    except Exception:                                  # noqa: BLE001
        return ""


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
MAX_SPOKEN_SENTENCES = 4       # 2 until 2026-09-04. Raised WITH the system
                               # prompt, never alone -- see the note at the
                               # top of this file about the last attempt.
MAX_SPOKEN_CHARS = 450                    # prefer a sentence end below this
# WHY THIS COSTS HIM NOTHING: speech is streamed a sentence at a time
# (_stream_round, and 'a streamed sentence is spoken before the reply'
# below), so time-to-first-word does not depend on how long the answer
# turns out to be. A longer reply means he hears MORE, not that he waits
# longer to hear anything -- which is the condition he set for raising it.
HARD_SPOKEN_CHARS = _TTS.MAX_SPEAK_LENGTH  # the one hard limit, shared w/ TTS
NO_CLOCK_LINE = "I'm afraid I haven't a clock in front of me just now, sir."
# Said instead when a clock IS available and the model's reading contradicts
# it — claiming to have no clock would be the second false statement.
UNSURE_CLOCK_LINE = "Let me check the time again, sir; that didn't look right."

# ----------------------------------------------------------------------
# Reasoning scaffolding — control tokens that must never be spoken
#
# LIVE 2026-09-02 14:43:59, jarvis.log 6724, the whole line as the room
# heard it:
#
#   speaking (f5): thought <channel, >Good afternoon, Ali and Heather; ...
#
# "<channel, >" is this module's own doing: strip_markdown turns a table's
# cell pipes into clause breaks, so the model's "thought\n<channel|>"
# arrives at TTS with the pipe already read as a comma. Which is exactly
# why the scrub runs FIRST inside clean_ollama_reply, on the RAW text
# (spoken_from_ollama's order) — one pass later the token is no longer a
# token and there is nothing left to recognise.
#
# _chat_payload has sent think:false on every request since the port, the
# same field tools/screen.py needs for gemma4 on the vision path, and it
# is the only request-level lever there is (a `stop` string cannot help:
# the scaffolding arrives BEFORE the words, so stopping on it would throw
# away the reply and keep nothing). On 2026-09-02 it did not hold, and the
# failure mode of this class of bug is by definition garbage read aloud,
# so the request keeps its guard and the speech path gets its own.
#
# The bare label words (thought / analysis / final ...) are cut ONLY where
# they sit against a token: "I thought so, sir" is ordinary English and
# survives untouched.
_REASONING_TAGS = r"think|thinking|reasoning|analysis|scratchpad"
# A harmony-style reasoning channel takes its CONTENTS with it, up to the
# next turn token — the words in an "analysis" channel are the model
# talking to itself, not to the room.
_CHANNEL_OPEN = (r"<\|channel\|>[ \t]*(?:analysis|analyses|thought|thinking|"
                 r"reasoning|commentary|scratchpad)\b")
_CHANNEL_END = r"<\|(?:end|return|start)\|>"
# Gemma's own thought block, with ASYMMETRIC pipes: ``<|channel>thought\n
# {reasoning}<channel|>{answer}``. The 09-02 leak was this block with an
# EMPTY thought, so the scrub only learned to drop the bare label; with a
# non-empty thought the CONTENT survived both scrubs and went to TTS ahead
# of the answer (F47, reproduced 2026-09-03: "He wants a greeting. It is 2
# pm and the family is home." spoken to the room). The opener can be eaten
# upstream (that is the shape that reached the log), so a buffer that
# BEGINS with "thought" and a newline opens a block too.
_GEMMA_OPEN = r"(?:<\|channel>[ \t]*thought\b|^[ \t]*thought[ \t]*\n)"
_GEMMA_END = r"<channel\|>"
# A block that has both ends. This is the ONLY form that may be cut out of
# a half-arrived stream buffer, where "no terminator yet" means "still
# coming", not "runs to the end of the reply".
_CLOSED_REASONING_RX = re.compile(
    rf"<\s*({_REASONING_TAGS})\s*>.*?<\s*/\s*\1\s*>|"
    rf"{_CHANNEL_OPEN}.*?{_CHANNEL_END}|"
    rf"{_GEMMA_OPEN}.*?{_GEMMA_END}", re.I | re.S)
# An opener with nothing closing it. On a WHOLE reply that is the end of
# the stream, so the rest of the text is thinking and goes with it; the
# alternative is reading the model's monologue to the room, which is the
# defect this whole section exists for.
_UNCLOSED_REASONING_RX = re.compile(
    rf"(?:<\s*(?:{_REASONING_TAGS})\s*>|{_CHANNEL_OPEN}|{_GEMMA_OPEN}).*$",
    re.I | re.S)
_REASONING_OPEN_RX = re.compile(
    rf"<\s*(?:{_REASONING_TAGS})\s*>|{_CHANNEL_OPEN}|{_GEMMA_OPEN}", re.I)
_REASONING_TAG_RX = re.compile(
    rf"<\s*/?\s*(?:{_REASONING_TAGS})\s*>", re.I)
# A channel label, and the roles a turn header names.
_SCAFFOLD_LABEL = (r"(?:thought|thinking|analysis|commentary|final|message|"
                   r"channel|assistant|model|system|user)")
# <|channel|>, <|end|>, <|im_start|>, and the half-forms a template can
# leave behind (<channel|>). A pipe inside angle brackets is never prose.
_SPECIAL_TOKEN = r"(?:<\|[A-Za-z0-9_.\-]{0,32}\|?>|<[A-Za-z0-9_.\-]{0,32}\|>)"
# Gemma/Llama/ChatML turn markers, which carry no pipe at all.
_TURN_TOKEN = (r"(?:<\s*/?\s*(?:s|bos|eos|pad|unk|start_of_turn|end_of_turn|"
               r"end_of_text|eot_id|begin_of_text|start_header_id|"
               r"end_header_id|im_start|im_end)\s*>)")
_SCAFFOLD_RX = re.compile(
    rf"(?:\b{_SCAFFOLD_LABEL}\b[ \t]*\n?[ \t]*)?"
    rf"(?:{_SPECIAL_TOKEN}|{_TURN_TOKEN})"
    rf"(?:[ \t]*\n?[ \t]*\b{_SCAFFOLD_LABEL}\b)?", re.I)


def strip_model_scaffolding(text):
    """Reasoning-channel and turn-template tokens out of a model reply.

    Returns the text unchanged when there is none, which is the normal
    case; a reply that WAS scrubbed is logged, because "the assistant read
    control tokens to the room" has to be greppable next time.
    """
    raw = text or ""
    out = _CLOSED_REASONING_RX.sub(" ", raw)
    out = _UNCLOSED_REASONING_RX.sub(" ", out)
    out = _REASONING_TAG_RX.sub(" ", out)
    out = _SCAFFOLD_RX.sub(" ", out)
    if out == raw:
        return raw
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    if raw.strip() and not out:
        # The other outcome, and it needs its own line: chat() turns an
        # empty reply into MODEL_EMPTY_LINE, so the room hears an honest
        # sentence rather than silence — but with no WARNING carrying the
        # raw text, a truncated reasoning stream that ate a real answer is
        # invisible in the log (2026-09-02 review).
        log.warning("the scaffolding scrub emptied the reply: %r", raw[:120])
    else:
        log.warning("scrubbed model scaffolding from the reply: %r", raw[:80])
    return out


def reasoning_block_open(text) -> bool:
    """True while a reasoning block has been opened and not yet closed.

    For the STREAM, which sees the reply a few characters at a time. The
    per-sentence scrub cannot see a block that spans a sentence break —
    "<think>He wants a greeting. It is 2 pm.</think>Good afternoon, sir."
    streamed "He wants a greeting." to TTS (2026-09-02 review, driven
    through the real loop) because that sentence carries an opener and no
    closer, and _REASONING_TAG_RX then dropped the tag and kept the words.
    """
    return bool(_REASONING_OPEN_RX.search(
        _CLOSED_REASONING_RX.sub(" ", text or "")))


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
    strip_markdown.

    The scaffolding scrub is FIRST and it has to be: strip_markdown, which
    runs after this, rewrites a pipe as a comma and destroys the very
    thing this recognises (see _SCAFFOLD_RX)."""
    text = strip_model_scaffolding(text)
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


# ----------------------------------------------------------------------
# Who is being spoken to, and what o'clock it is
#
# LIVE 2026-09-02 14:29:24, jarvis.log 6520. "Say hello to my family."
# was answered, out loud, with:
#
#   Good evening, Ali and Heather; I do hope you're both having a lovely
#   afternoon, sir.
#
# Two separate faults in one sentence.
#
# (a) THE HONORIFIC IS AIMED AT THE WRONG PEOPLE. The greeting addresses
#     Ali and Heather -- "you're BOTH" -- and signs off to a third person
#     who is not in the room. Nothing in the persona knows there is an
#     audience: VOICE_RULES says call him sir, JARVIS_SYSTEM's closing
#     line says call him sir, and the formal register says every time.
#     "sir" IS the character, so this is deliberately the narrowest rule
#     that covers the case, and BOTH halves must hold: Hunter asked for a
#     relay to a named person who is not himself, AND the line being
#     spoken opens by greeting that person. The second half is what keeps
#     "Tell my professor I'll be late." -> "I'm afraid I can't send
#     messages, sir." intact -- that sentence is spoken TO him.
#
# (b) THE TIME OF DAY WAS RECALLED, NOT READ. "Good evening" at 2:29 pm,
#     with "Current time: 02:29 PM" in the same prompt. Verified cause:
#     ~/.aiws_trainer/jarvis_memory/sessions.json held the previous
#     EVENING's answer to this same request, twice, and
#     Memory.format_sessions_for_prompt renders the last three sessions
#     into every user turn cut at 100 characters --
#
#       [2026-09-01T20:55] You: Say hello to my family. / Jarvis: Good
#       evening, Ali and Heather; I do hope you're both having a
#
#     -- so the model was handed its own opening for this exact request,
#     dangling mid-clause, and completed it against the clock it could
#     see. Hence "a lovely afternoon" behind "Good evening". It is
#     intermittent because it needs the request to be worded the way the
#     stored one was; at 14:43 it was not, and the reply was correct.
#
#     The fix is to ground the word rather than trust the recall:
#     arc.greeting_word() reads the wall clock on the same bands
#     commander._greeting_line has always used. Grounded on the clock and
#     NOT on arc.phase(), because the phase is forced to "night" for an
#     empty or hushed house -- jarvis.log 14:06:07, "arc: afternoon ->
#     night (forced by you're out)" -- and a greeting keyed off that would
#     have said "Good night" at six minutes past two.
# ----------------------------------------------------------------------
# Whom Jarvis is being asked to speak to. "my brother", "the family",
# "Ali and Heather" -- but never "me", which is what makes "tell me a
# joke" and "what did you say to me?" ordinary turns.
# Case-SENSITIVE on purpose: a bare capitalised word is the only signal
# that "Ali" is a person and "me" is not, and re.I would throw it away
# ("Tell me a joke" matched [A-Z][a-z]+ under re.I, 2026-09-02).
_AUDIENCE = (r"(?:(?:[Mm]y|[Oo]ur|[Hh]is|[Hh]er|[Tt]heir)\s+\w+|"
             r"[Tt]he\s+(?:family|kids|children|boys|girls|others)|"
             r"them|everyone|everybody|"
             r"[A-Z][a-z]+(?:\s*(?:,|and)\s*[A-Z][a-z]+)*)")
_RELAY_RX = re.compile(
    rf"\b(?:[Ss]ay|[Ss]aid|[Pp]ass\s+on|[Rr]elay)\b[^.?!]{{0,60}}?"
    rf"\bto\s+{_AUDIENCE}\b|"
    rf"\b(?:[Tt]ell|[Ww]ish|[Gg]reet)\s+{_AUDIENCE}\b")
# Hunter is not a third party. His name is hard-coded in the persona
# prompts above ("Now answer Hunter as Jarvis"), so it is hard-coded here
# too rather than invented a second time.
_NOT_A_THIRD_PARTY = frozenset({"hunter", "sir", "jarvis"})
# The case-insensitivity is scoped to the GREETING WORDS and stops there.
# The name class is the same [A-Z][a-z]+ as _AUDIENCE above and for the
# same reason -- a capital is the only signal that "Heather" is a person --
# so a blanket re.I here made it match any lowercase word at all, and the
# rule then read every "hello"/"welcome" sentence as third-party speech:
#
#   'Welcome back, sir.'                     who='back, sir'
#   'Hi there, sir.'                         who='there, sir'
#   'hello has been added to tomorrow at 4:30 pm, sir.'   who='has'
#
# The first is presence.WELCOME_LINE and the third is a real gemma4 line
# from jarvis.log.1 21:02:50 -- both would have lost Hunter's honorific on
# any relay turn, which is the outcome this rule exists to avoid
# (2026-09-02 review, seven measured). (?i:...) keeps "hello"/"HELLO"
# case-blind; the name stays case-sensitive.
_GREETS_BY_NAME_RX = re.compile(
    r"^\s*(?i:good\s+(?:morning|afternoon|evening|day|night)|hello|hi|hey|"
    r"greetings|welcome)\b[,\s]+(?P<who>[A-Z][a-z]+"
    r"(?:\s*(?:,|and)\s*[A-Z][a-z]+)*)\b")
# The other half of the live line: "I do hope you're BOTH having a lovely
# afternoon". A plural second person cannot be Hunter on his own, so the
# sentence carrying it is aimed at the audience even though it names
# nobody. Anything vaguer keeps its sir -- a compound request ("say hi to
# my family and then give me my daily briefing", 14:43) turns back to him
# mid-reply, and the briefing is his.
#
# "both"/"two"/"all" are only a plural YOU when a clause boundary or a verb
# follows. With a noun behind them they are an ordinary singular "you" plus
# a quantity, and the loose form took his sir out of all of these
# (2026-09-02 review, measured through strip_relay_address):
#
#   'Are you all set, sir?'   'Thank you all the same, sir.'
#   "I'll send you two reminders, sir."   'That leaves you two options, sir.'
#
# "you're both" / "you are both" / "both of you" need no such test: there
# is no singular reading of either.
_PLURAL_VERB = (r"are|were|have|has|had|will|would|can|could|shall|should|"
                r"may|might|must|do|did|does|need|want|seem|look|sound|"
                r"enjoy|keep|deserve|get")
_ADDRESSES_A_GROUP_RX = re.compile(
    r"\byou(?:'re|\s+are)\s+both\b|\bboth\s+of\s+you\b|"
    rf"\byou\s+(?:both|two|all)\b(?=[,.;:!?]|\s*$|\s+(?:{_PLURAL_VERB})\b)",
    re.I)


def relay_request(user_text) -> bool:
    """True when Hunter asked Jarvis to say something TO SOMEBODY ELSE."""
    return bool(_RELAY_RX.search((user_text or "").strip()))


def _addressed_to_a_third_party(sentence) -> bool:
    """True when THIS sentence is spoken to someone other than Hunter.

    Judged sentence by sentence and never carried forward, which is the
    conservative half of the rule: streaming speaks sentence one before
    sentence two exists, so a decision that persisted would have to be
    made on sentence one alone -- and "say hi to my family and then give
    me my daily briefing" (LIVE 14:43) turns back to him mid-reply. A
    sentence that does not say who it is for keeps its sir.
    """
    sentence = sentence or ""
    m = _GREETS_BY_NAME_RX.match(sentence)
    if m is not None:
        names = re.split(r"\s*(?:,|and)\s*", m.group("who"))
        if any(name.strip().lower() not in _NOT_A_THIRD_PARTY
               for name in names if name.strip()):
            return True
    return bool(_ADDRESSES_A_GROUP_RX.search(sentence))


def strip_relay_address(text, user_text):
    """Drop "sir" from the sentences of ``text`` that are addressed to
    somebody else -- and only when Hunter asked for a relay in the first
    place (relay_request). Both gates, every time."""
    if not text or not relay_request(user_text):
        return text
    return " ".join(
        address_mod.drop_addresses(s) if _addressed_to_a_third_party(s) else s
        for s in split_sentences(text))


# A greeting opening a sentence AND signing off into one -- the trailing
# lookahead is what keeps "Good Morning America is at nine, sir." out of
# it: a title is followed by more of its own name, a greeting by a comma,
# a stop or nothing.
_GREETING_OPEN_RX = re.compile(
    r"(?:(?<=^)|(?<=[.!?;]\s)|(?<=[.!?;]\n))(\s*)(Good)\s+"
    r"(morning|afternoon|evening)\b(?=[,.;:!?]|\s*$)")
# A SECOND time-of-day word later in the same sentence, inside a well-wish.
# The whole clause was recalled together, so grounding only the opening
# swaps one contradiction for another: at 11:59 the real logged line "Good
# evening, Ali and Heather; I do hope you're both having a pleasant
# evening." came out "Good morning ... a pleasant evening" (2026-09-02
# review). Anchored on hope/wish/have/enjoy and on the word ENDING its
# clause, because "I hope your afternoon meeting goes well" is about a
# later hour and is not a claim about this one.
_GREETING_TAIL_RX = re.compile(
    r"(\b(?:hope|hoping|wish|wishing|have|having|had|enjoy|enjoying)\b"
    r"[^.!?]{0,60}?\b(?:a|an|the|your)\s+(?:\w+\s+){0,2})"
    r"(?:morning|afternoon|evening)\b(?=[,.;:!?]|\s*$)", re.I)
# A greeting the reply is REPORTING rather than making. ground_greeting
# reaches summarize() and local_line() too (_finish_spoken with an empty
# user_text), which is how a mail digest or a Claude result gets read out,
# and those are exactly the places somebody else's "Good morning" appears.
# Rewriting a quoted hour is the same class of false claim the grounding
# was built to remove.
_REPORTING_CUE_RX = re.compile(
    r"\b(?:reads|read|wrote|writes|written|said|says|replied|replies|"
    r"quoted|quotes|message|note|follows|asked|asks)\b"
    r"[^.!?;]{0,24}[.!?;]\s*$", re.I)
# NB the name. _SENTENCE_END_RX is already taken, by the stream splitter
# that _split_complete_sentences uses, and shadowing it here silently cut
# the terminator off every streamed sentence ("The build passed, sir" for
# "The build passed, sir.") -- caught by the suite, not by review.
_GREETING_SENTENCE_END_RX = re.compile(r"[.!?]")


def ground_greeting(text, user_text="", now=None):
    """Rewrite a greeting's time-of-day word to the one the clock says.

    Only at a sentence opening -- "I'll wish them a good evening later" is
    about a later hour, not a claim about this one -- and never "good
    night", which is a sign-off and is right whenever he says it. If
    Hunter used the word himself, Jarvis echoing him is politeness, not a
    stale memory, and it stands; if the reply is quoting somebody else's
    greeting ("your message reads as follows. Good morning, ..."), the
    hour in it is not Jarvis's claim to make right.

    When the opening IS corrected, a well-wish later in the SAME sentence
    is corrected with it -- the clause was recalled as one piece, so
    grounding the first word alone leaves the sentence contradicting
    itself instead of the clock.
    """
    if not text or "good" not in text.lower():
        return text
    want = arc_mod.greeting_word(now)
    said = (user_text or "").lower()
    out, pos = [], 0
    for m in _GREETING_OPEN_RX.finditer(text):
        if m.start() < pos:
            continue                  # inside a tail already rewritten
        out.append(text[pos:m.start()])
        pos = m.end()
        word = m.group(3).lower()
        if (word == want or f"good {word}" in said
                or _REPORTING_CUE_RX.search(text[:m.start()])):
            out.append(m.group(0))
            continue
        out.append(f"{m.group(1)}{m.group(2)} {want}")
        stop = _GREETING_SENTENCE_END_RX.search(text, pos)
        stop = stop.end() if stop else len(text)
        out.append(_GREETING_TAIL_RX.sub(rf"\g<1>{want}", text[pos:stop]))
        pos = stop
    out.append(text[pos:])
    new = "".join(out)
    if new != text:
        log.info("grounded a stale greeting in the clock: %r -> %r",
                 text[:60], new[:60])
    return new


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
    text = ground_greeting(text, user_text)
    text = strip_relay_address(text, user_text)
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


def _finish_authored(raw, guard_context, user_text):
    """A CODE-authored reply -> the line TTS gets: the same guards as
    _finish_spoken and neither of its prose caps.

    MAX_SPOKEN_SENTENCES and MAX_SPOKEN_CHARS are a rule about how much
    PROSE he wants back from a model. Two writes' confirmations joined,
    or the degrade's confirmations plus its honest word about what went
    unspoken, are not prose: every sentence reports something that
    happened or names something fetched, and cutting one drops the news
    of a write (F26, 2026-09-03: "add milk and set a timer" spoke the
    calendar line alone; "what's the weather, add milk, set a timer" with
    a silent render round spoke both confirmations and lost the notice
    about the weather while the log said the sources had been named).
    append_spoken_lines has exempted a HELD line from the caps since it
    was written; this is the same exemption for the speak branch and the
    degrade.

    The one bound left is HARD_SPOKEN_CHARS, whole sentences up to it:
    that is the TTS limit, and TTS cuts there itself (tts.py,
    MAX_SPEAK_LENGTH) -- past it a line is lost either way, and a
    sentence boundary is the better place to lose it."""
    text = guard_authored(raw)
    text = guard_clock_claims(text, guard_context, user_text)
    text = ground_greeting(text, user_text)
    text = strip_relay_address(text, user_text)
    return trim_spoken(text, cap=HARD_SPOKEN_CHARS, hard=HARD_SPOKEN_CHARS)


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
SYLLABUS_FORMAT = {
    "type": "object",
    "properties": {"rows": {"type": "array", "items": {
        "type": "object",
        "properties": {"title": {"type": "string"}, "course": {"type": "string"},
                       "due": {"type": "string"}},
        "required": ["title", "due"]}}},
    "required": ["rows"]}
SYLLABUS_TIMEOUT_S = 60.0
SYLLABUS_MAX_CHARS = 6000


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


# Weekly memory garden (jarvis/garden.py): a week of the activity journal
# in, durable facts about Hunter out.  Runs on a Sunday-night thread while
# he sleeps, so the timeout is generous and the temperature low.
GARDEN_TIMEOUT_S = 180.0
GARDEN_MAX_FACTS = 6
GARDEN_KEY_CHARS = 40
GARDEN_VALUE_CHARS = 200
GARDEN_JOURNAL_CHARS = 6_000
GARDEN_FORMAT = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {
        "type": "object",
        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
        "required": ["key", "value"]}}},
    "required": ["facts"]}


def extract_facts(journal_text, known=None, limit=GARDEN_MAX_FACTS,
                  timeout=GARDEN_TIMEOUT_S):
    """[{key, value}] durable facts about Hunter from a week of journal
    text; [] when the model is down, lent or says there is nothing.

    The prompt is written against the failure this feature actually has:
    the model inflating one request into a standing preference. A dry run
    over a synthetic week (scratchpad/garden_dryrun.py) turned a single
    "play some jazz while I write" into "Hunter prefers jazz while he
    writes", so the rules below demand a habit he STATED or a thing that
    recurs across days, and an empty list is named as a correct answer."""
    text = (journal_text or "").strip()
    if not text:
        return []
    n = max(1, min(int(limit or GARDEN_MAX_FACTS), GARDEN_MAX_FACTS))
    known_line = "; ".join(str(k) for k in (known or []))[:1200] or "(nothing yet)"
    instruction = (
        f"Instruction for Jarvis (not a question from Hunter): below is a week "
        f"of your own activity journal. Extract only DURABLE facts about Hunter "
        f"worth remembering for months: recurring people, courses, projects, "
        f"habits he stated, preferences he stated. Reply with JSON only. Each "
        f"fact has \"key\" (two to four lower-case words) and \"value\" (one "
        f"plain sentence about Hunter, in your own words). Rules: state ONLY "
        f"what the journal says -- never infer, and never write a name, number "
        f"or date that is not there. One request on one day is NOT a "
        f"preference: a habit must either be something he said about himself "
        f"or something that repeats across several days. Nothing already known. "
        f"At most {n} facts; fewer is better and an empty list is a correct "
        f"answer.\n\nAlready known, do not repeat: {known_line}\n\n"
        f"Journal:\n{text[:GARDEN_JOURNAL_CHARS]}")
    obj = _json_request(instruction, GARDEN_FORMAT, timeout,
                        num_predict=90 * n, temperature=0.1)
    out, seen = [], set()
    for item in (obj or {}).get("facts") or []:
        if not isinstance(item, dict):
            continue
        key = " ".join(str(item.get("key") or "").split()).lower()[:GARDEN_KEY_CHARS]
        value = " ".join(strip_markdown(str(item.get("value") or "")).split())
        if not key or not value or key in seen:
            continue
        seen.add(key)
        out.append({"key": key, "value": value[:GARDEN_VALUE_CHARS]})
    return out[:n]


def read_syllabus(text, today="", timeout=SYLLABUS_TIMEOUT_S):
    """[{title, course, due}] of dated work found in syllabus text; [] on
    failure. ``due`` is asked for as a plain ISO string because every
    other date format a model invents ("Oct 3", "week 5") has to be
    guessed at, and a guessed exam date is worse than a dropped one.

    The year is the trap: a syllabus writes "October 3" and the model
    supplies a year, so it is told today's date and told to pick the year
    that puts the date in the next twelve months. jarvis/syllabus.py drops
    anything more than MAX_AHEAD_DAYS out as the hallucination it is, and
    the spoken read-back is the second net."""
    text = (text or "").strip()
    if not text:
        return []
    when = f"Today is {today}. " if today else ""
    instruction = (
        f"Instruction for Jarvis (not a question from Hunter): {when}extract "
        f"every dated exam, quiz, project or assignment from the course "
        f"material below. Reply with JSON only. \"title\" is what it is called "
        f"(\"Midterm 1\", \"Lab 3 report\"); \"course\" is the course it belongs "
        f"to, or an empty string when the material does not say; \"due\" is the "
        f"date as YYYY-MM-DD, or YYYY-MM-DDTHH:MM when a time of day is given. "
        f"If the year is not written down, choose the year that puts the date "
        f"within the next twelve months of today. Skip anything with no date "
        f"at all -- never invent one, and never turn a week number into a "
        f"date. No markdown.\n\nCourse material:\n{text[:SYLLABUS_MAX_CHARS]}")
    obj = _json_request(instruction, SYLLABUS_FORMAT, timeout, num_predict=600,
                        temperature=0.0)
    out = []
    for item in (obj or {}).get("rows") or []:
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title") or "").split())
        due = " ".join(str(item.get("due") or "").split())
        if title and due:
            out.append({"title": title,
                        "course": " ".join(str(item.get("course") or "").split()),
                        "due": due})
    return out


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
             max_rounds=3, on_sentence=None, addressee=None):
        """Tier 2 with tools on a worker thread; callback gets the tags
        ([("BRIEFING", json)] when a card was produced, then ("SPEAK",
        line)). ``addressee`` is ``(name, honorific)`` when the caller
        knows who the turn is for (``scope.OWNER`` for a proactive call);
        None reads the per-turn attribution once, inside _chat_sync."""
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
                                       on_sentence=on_sentence, gen=gen,
                                       addressee=addressee)
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

    def read_syllabus(self, text, today="", timeout=SYLLABUS_TIMEOUT_S):
        return read_syllabus(text, today=today, timeout=timeout)

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

    # Register (spec #16): the ONE production caller of reset_static_prompt.
    def set_register(self, name, warm=True):
        """Change the spoken register and re-warm the prefix OFF the audio
        path. The commander answers with a canned Tier 1 line the instant
        this returns, so the ~2700-token reprocess runs in a daemon thread
        rather than in front of his next question."""
        changed = set_register(name)
        if changed and warm:
            threading.Thread(target=warm_static, daemon=True,
                             name="static-rewarm").start()
        return changed

    def register(self):
        return register()

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
    def _dynamic_context(self, text="", owner=True):
        """The per-turn background: context + memory. ``text`` (Hunter's
        words) lets the memory pick the facts RELEVANT to this utterance;
        it stays in the user turn, never the static system prompt.

        ``owner=False`` -- a known person's turn -- gets NEITHER: the
        context block carries his active window, his git state, his
        recent conversation and his screen, and the memory block carries
        what he told me about his life. None of it is hers to be answered
        from, with or without a tool; the time is a tool she is offered."""
        if not owner:
            return "", ""
        ctx_text = ""
        if self._context:
            ctx = self._context.get_context("standard")
            ctx_text = self._context.format_for_prompt(ctx, spoken=True)
            # Continuity (spec #14): "that would be the third coffee timer".
            # It goes HERE, in the dynamic half -- the static prefix must
            # stay byte-identical or every turn pays a full reprocess.
            block = getattr(self._context, "continuity_block", None)
            if block is not None:
                try:
                    earlier = block() or ""
                except Exception:                  # noqa: BLE001
                    log.exception("continuity_block failed")
                    earlier = ""
                if earlier:
                    ctx_text = f"{ctx_text}\n{earlier}".strip()
        mem_text = ""
        if self._memory:
            # This runs BEFORE `started`, so its cost never showed up in the
            # "chat reply (Ns wall)" line -- which is how it hid. Handing it
            # the utterance used to make it embed one, and under
            # OLLAMA_MAX_LOADED_MODELS=1 that evicted this very model: 2.5 s
            # for the embed, then 6.9 s of reload on the request below, on
            # EVERY turn (measured live 2026-08-31; see "The one-slot rule"
            # in jarvis/memory.py). format_for_context now ranks only once
            # the fact store outgrows the prompt, so the common turn asks
            # Ollama for no model but this one.
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
                   on_sentence=None, gen=None, addressee=None):
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
        # ONE reading of who the turn is for -- exactly one: the prompt
        # render, the schemas offered, the background handed to the model
        # and every call refused below all take THIS pair, even if the
        # gate names the next person while this turn is still on the
        # worker. A caller that knows (a proactive briefing) passes it in.
        turn_addr = tuple(addressee) if addressee is not None \
            else scope_mod.addressee()
        turn_who = turn_addr[0]
        owner_turn = not turn_who
        ctx_text, mem_text = self._dynamic_context(text, owner=owner_turn)
        messages = [{"role": "system",
                     "content": static_system(addressee_to=turn_addr)},
                    {"role": "user",
                     "content": build_user_turn(ctx_text, mem_text, text)}]
        tools = _registry_schemas(registry, text, owner=owner_turn)
        started = time.monotonic()
        cap = MAX_SPOKEN_SENTENCES
        card = None
        speak = None
        tool_texts = []
        tool_names = []            # the OK ones, for the degrade's wording
        # The reply-coverage check (answers_owed): every (name, result) of
        # the turn, so when an authored speak= line tries to end it the
        # loop can see whether an ANSWER is still owed.
        ran_results = []
        held_lines = []            # authored lines the check held back
        held_said = False          # ...and whether the reply carries them yet
        # How much of ran_results has already been put into SPOKEN words.
        # A round that streams prose before asking for a tool has already
        # answered everything that ran before it -- the prose came out of a
        # round with those results in its transcript, which is the whole
        # provenance argument -- so the check must not repair what he has
        # already heard.
        answered_upto = 0
        tool_budget = MAX_TOOL_TEXT_TOTAL_CHARS
        truncated = False
        final = ""
        # True when `final` is the degrade -- code-authored, so it is
        # finished by _finish_authored (guards, no prose caps) rather than
        # capped like a model's reply. See F26 in that docstring.
        authored_final = False
        # Ollama reports load_duration even on a resident model: it is the
        # server's own per-request overhead before the runner sees the
        # prompt. Logging it separates "the model was slow" from "Ollama
        # was slow" when a reply misses the latency bar (spec 4.3).
        server_s = 0.0

        def spent():
            """Seconds of WORK this turn: tool calls plus the model's own
            compute, with Ollama's per-request overhead taken back out.
            That overhead is a 26B reload on the first round of every turn
            here, and no amount of skipped tool work makes it smaller --
            billing it to the tool budget is what starved the mail half of
            the 15:14 turn before a single tool had run."""
            return max(0.0, time.monotonic() - started - server_s)

        def over_budget():
            return spent() > TURN_WORK_BUDGET_S

        def note(result, name, args=None):
            nonlocal cap, card
            cap = max(cap, int(getattr(result, "max_sentences", 2) or 2))
            if getattr(result, "card", None):
                card = result.card
            tool_texts.append(result.text or "")
            ran_results.append((name, result))
            if getattr(result, "ok", True):
                # A failed tool is not something Jarvis "has": claiming
                # "I have your calendar" off "calendar unreachable" would
                # be a lie the degrade tells all by itself.
                tool_names.append(name)
            # The args ride along because the text alone hid a real bug:
            # "spotify_liked -> ok=True ...on shuffle" (2026-09-01 19:59)
            # could not say whether the shuffle was Hunter's or the
            # model's. Capped so a long note body cannot flood the log.
            log.info("tool %s%s -> ok=%s %s", name, _args_for_log(args),
                     result.ok, (result.text or "")[:80])
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
        render_only = False        # next round writes; it may not call tools
        render_granted = False     # the reserved render round, spent once
        render_told = False        # ...and it is told so, once, in-message

        def grant_render_round(why=None):
            """Spend the render reservation: one more model round, with
            the tools stripped. False once already spent -- the reserve is
            one round, not an escape from max_rounds."""
            nonlocal render_only, render_granted, rounds_left
            render_only = True
            if render_granted:
                return False
            render_granted = True
            rounds_left = 1        # exactly one, and it can only write
            if why:
                log.warning("chat: %s; spending the render round on it", why)
            else:
                log.info("chat: %.1fs of work spent (%.1fs budget); reserving "
                         "a render round", spent(), TURN_WORK_BUDGET_S)
            return True

        def answer_skipped_calls(calls, ran):
            """Every call of a cut-short round that never ran gets a tool
            message saying so. A tool_call with no result beside it reads
            as unfinished business, and the model's next move is to make
            it again -- which is exactly what the reserved round did at
            15:14, with no words to show for the turn."""
            for call in calls[ran:]:
                name, _args = self._tool_call_parts(call)
                messages.append({"role": "tool", "content": TOOL_SKIPPED_TEXT,
                                 "tool_name": name})

        def authored_line(result):
            """``result.speak`` as it will be SPOKEN: guarded once
            (guard_authored) and sentence-capped once, PER LINE, at the
            larger of MAX_SPOKEN_SENTENCES and the tool's own allowance.
            The per-line cap is the 2026-08-26 M8 rule (a ten-sentence
            note read-out is still not read whole) kept where it belongs:
            on the line, never on the join. Two lines joined, or the
            degrade, are then never trimmed against each other (F26)."""
            line = guard_authored(result.speak) if result.speak else ""
            if not line:
                return ""
            allowed = max(MAX_SPOKEN_SENTENCES,
                          int(getattr(result, "max_sentences", 0) or 0))
            return limit_sentences(line, allowed)

        def degrade():
            """What to say with results in hand and no prose for them.
            Lines the coverage check held back are spoken here rather than
            lost: the writes they report really happened, and only the
            answers beside them went unwritten. Only sources actually IN
            HAND are named -- a read that failed is not something to claim
            to have."""
            nonlocal held_said
            if held_lines:
                held_said = True
                return coverage_degrade(
                    held_lines, answers_owed(ran_results, in_hand=True))
            return tool_only_line(tool_names)

        def render_round_available(why):
            """True when a writing-only round can still run this turn:
            either the reserve is spent on this, or something already
            spent it this round (the work budget) and its round is still
            ahead. Never a round WITH tools -- that is the timer bug."""
            return grant_render_round(why) or (render_only and rounds_left > 0)

        if force_tool and registry is not None and registry.has(force_tool):
            # force_args pins arguments the model gets wrong on its own. It
            # chose unread_only=True for "what was my last email about?"
            # even with the parameter documented as the thing that answers
            # exactly that question, so the read mail newer than the answer
            # it gave was never searched.
            args = dict(force_args or {})
            result = _scoped_call(registry, force_tool, args,
                                  owner=owner_turn, who=turn_who)
            note(result, force_tool, args)
            if result.speak:
                speak = authored_line(result) or result.speak
            else:
                messages.append({"role": "assistant", "content": "",
                                 "tool_calls": [{"function": {
                                     "name": force_tool, "arguments": args}}]})
                messages.append(tool_message(result, force_tool))
                rounds_left = 1        # one model turn renders the result

        streamed_sentences = []           # what on_sentence already received
        clock_line_said = []
        # The unbacked-action guard (UNBACKED_NUDGE): armed only when the
        # model HAS tools to act with -- a claim in a turn with no tools on
        # offer is not something a retry can fix, and the forced path above
        # already ran one -- and only when he ORDERED something.
        #
        # The question half is not a nicety: the retry EXECUTES. "Did you
        # already add milk to my list?" answered "Yes, sir, I've added
        # milk" is a correct answer to a question, and arming the guard on
        # it made the retry go and add the milk -- the question performing
        # the action it asked about (2026-09-02 review, on the real brain).
        # A model that narrates instead of acting on a QUESTION has told
        # him something possibly wrong; a guard that writes on a question
        # is worse than the thing it was built to catch, so questions get
        # the reply as it stands.
        #
        # `unbacked_first` holds the reply the retry was asked to make good
        # on; `plain_round` takes the retry off the stream, because the
        # first reply's honest sentences were already spoken and the
        # retry's words are either tool calls or discarded.
        unbacked_armed = (registry is not None and bool(tools)
                          and not is_question(text))
        unbacked_first = None
        plain_round = False

        def guard(sentence):
            # The guards _finish_spoken applies to the whole reply, per
            # sentence: a streamed sentence is spoken before the reply
            # exists, so it must not carry an ungrounded clock claim or
            # a leaked context line the full reply would have lost.
            if unbacked_armed and not tool_texts and unbacked_claim(sentence):
                # Withheld, not spoken: with no tool run yet, "I'm starting
                # your music now" is a claim the round has not earned. The
                # whole reply is judged once the round ends -- the retry or
                # UNBACKED_LINE speaks for this sentence, never the model.
                return ""
            line = clean_ollama_reply(strip_markdown(clean_ollama_reply(sentence)))
            guarded = guard_clock_claims(
                line, "\n".join([ctx_text, mem_text] + tool_texts), text)
            if guarded != line and guarded in (NO_CLOCK_LINE, UNSURE_CLOCK_LINE):
                if clock_line_said:
                    return ""             # the honest line once, not per sentence
                clock_line_said.append(True)
            else:
                # The same two guards spoken_from_ollama runs on the whole
                # reply, per sentence -- a streamed sentence is spoken
                # before the reply exists. Skipped on the clock-guard's own
                # replacement line, which is authored and addressed to him.
                guarded = ground_greeting(guarded, text)
                guarded = strip_relay_address(guarded, text)
            return trim_spoken(guarded.strip())

        try:
            while speak is None and rounds_left > 0:
                rounds_left -= 1
                # A render round is sent WITHOUT tools: asking a model to
                # stop calling tools never worked, taking them away does.
                # Taking them away is not enough on its own either -- the
                # transcript it is answering IS a run of tool calls, so it
                # is also TOLD, here in the per-turn messages, that the
                # results are all it will get (LIVE 15:14: the reserved
                # round came back with a tool call and no words at all).
                round_tools = [] if render_only else tools
                if render_only and not render_told:
                    render_told = True
                    told = RENDER_NOW_LINE
                    if held_lines:
                        told += HELD_LINE_NOTE.format(
                            lines=" ".join(held_lines))
                    messages.append({"role": "user", "content": told})
                round_started = time.monotonic()
                streaming = on_sentence is not None and not plain_round
                plain_round = False
                if streaming:
                    # Each round gets the full spoken cap: what the model
                    # said before a tool call must not eat the answer's.
                    round_sentences = []
                    try:
                        data, content, calls = self._stream_round(
                            messages, round_tools, cap, on_sentence,
                            round_sentences, guard, gen=gen)
                    finally:
                        # kept even when the stream dies: they were spoken
                        streamed_sentences.extend(round_sentences)
                    if self._stale(gen):
                        final = ""            # barged in: nothing more to say
                        break
                    if round_sentences:
                        # Prose went to TTS this round, written from the
                        # results already in the transcript: he has HEARD
                        # them. _stream_round only silences a round after
                        # its first tool_call chunk, so a round that talks
                        # and then calls a tool lands here -- and repairing
                        # it would say the same answer a second time.
                        answered_upto = len(ran_results)
                else:
                    payload = _chat_payload(messages, round_tools)
                    try:
                        data = _http("/api/chat", payload,
                                     timeout=OLLAMA_TIMEOUT_S)
                    finally:
                        _unpin_if_lent(payload)
                    content, calls = _message_parts(data)
                # Clamped to the round's own wall time: a bogus (or simply
                # enormous) load_duration must not buy the loop unlimited
                # budget, since the budget is wall MINUS this.
                server_s += min(max(0.0, (data.get("load_duration") or 0) / 1e9),
                                max(0.0, time.monotonic() - round_started))
                if render_only and calls:
                    # Some models emit tool_calls even with none offered.
                    log.warning("chat: render round asked for %d more tools; "
                                "writing the answer instead", len(calls))
                    calls = []
                if not calls or registry is None:
                    final = content
                    if unbacked_armed and not tool_texts:
                        # ZERO tools ran this turn and the model is done
                        # talking: did it claim to have done something?
                        claim = unbacked_claim(final)
                        if claim and unbacked_first is None:
                            log.warning("brain: unbacked action claim %r "
                                        "(no tool ran)", claim)
                            unbacked_first = final
                            messages.append({"role": "assistant",
                                             "content": content})
                            messages.append({"role": "user",
                                             "content": UNBACKED_NUDGE})
                            # ONE retry, with a round of its own: the
                            # reply that earned it was the model's whole
                            # answer, and max_rounds counted it.
                            rounds_left = max(rounds_left, 1)
                            plain_round = True
                            continue
                        if unbacked_first is not None:
                            # The retry ran no tool either. Its words are
                            # not trusted over the first reply's -- the
                            # same model, the same nothing behind it -- so
                            # the FIRST reply is what he hears, with the
                            # claims taken out and the rest (the greeting)
                            # left standing.
                            log.warning("brain: unbacked action claim stands "
                                        "after the retry (no tool ran); "
                                        "replacing it")
                            final = strip_unbacked_claims(unbacked_first, cap)
                            if on_sentence is not None and \
                                    not self._stale(gen):
                                # the honest sentences were streamed as they
                                # landed and the claims withheld: the line
                                # standing in for them is spoken now, once
                                streamed_sentences.append(UNBACKED_LINE)
                                try:
                                    on_sentence(UNBACKED_LINE)
                                except Exception:
                                    log.exception("on_sentence failed")
                    if render_only and tool_texts and not (final or "").strip():
                        # the reserved round produced no words at all: say
                        # WHAT is in hand rather than only that something is
                        log.warning("chat: the render round wrote nothing; "
                                    "naming the sources instead")
                        final, authored_final = degrade(), True
                    break
                if force_tool and tool_texts and rounds_left == 0 and \
                        messages[-1].get("role") == "tool":
                    # The single forced turn asked for more tools instead of
                    # rendering the result. LIVE 14:35: force_tool=get_mail
                    # answered, then the model wanted the calendar too (the
                    # question asked for both) and this branch apologised for
                    # work already done. Spend the render reservation on
                    # saying what get_mail found instead.
                    if grant_render_round():
                        continue
                    final, authored_final = degrade(), True
                    break
                messages.append({"role": "assistant", "content": content,
                                 "tool_calls": calls})
                asked = calls          # what the TRANSCRIPT says he asked for
                if len(calls) > MAX_TOOL_CALLS_PER_ROUND:
                    log.warning("chat: model asked for %d tools in one "
                                "round; running the first %d", len(calls),
                                MAX_TOOL_CALLS_PER_ROUND)
                    calls = calls[:MAX_TOOL_CALLS_PER_ROUND]
                ran = 0
                authored = []      # this round's authored speak= lines
                done_calls = set()          # (name, args) already run here
                for call in calls:
                    name, args = self._tool_call_parts(call)
                    try:
                        key = (name, json.dumps(args, sort_keys=True,
                                                default=str))
                    except (TypeError, ValueError):
                        key = (name, repr(args))
                    if key in done_calls:
                        # NEVER TWICE. See TOOL_REPEAT_TEXT: the round runs
                        # to its end now, so a repeated call would really
                        # act. The transcript still gets an answer for it,
                        # because an unanswered tool_call is an invitation
                        # to make it again.
                        log.warning("chat: %s asked for twice with the same "
                                    "arguments; running it once", name)
                        messages.append({"role": "tool",
                                         "content": TOOL_REPEAT_TEXT,
                                         "tool_name": name})
                        ran += 1
                        continue
                    done_calls.add(key)
                    # from_model: the registry strips each spec's reserved
                    # keys here and ONLY here -- the forced path above is
                    # the commander's, and its args are the utterance's.
                    # The utterance rides along so a spec that reserves a
                    # key can still read it off HIS words (ToolSpec.derive):
                    # taking a knob away from the model must not mean the
                    # words lose it too.
                    result = _scoped_call(registry, name, args,
                                          owner=owner_turn, who=turn_who,
                                          from_model=True, utterance=text)
                    note(result, name, args)
                    messages.append(tool_message(result, name))
                    ran += 1
                    # Guarded HERE, once (authored_line): from this point
                    # the line is held, told to the render round, matched
                    # against its reply and spoken in one and the same
                    # form.
                    line = authored_line(result)
                    if line and line not in authored:
                        # COLLECTED, NOT ACTED ON. This used to `break` the
                        # round the moment any tool authored a line, which
                        # made the whole turn depend on the ORDER the model
                        # emitted its calls in -- and gemma4 emits them in
                        # the order of his clauses, every one of the live
                        # multi-tool turns. "Add milk to my list and what's
                        # on my calendar?" ran the notes write, stopped
                        # there, and never called get_calendar at all: the
                        # 23:20 incident again, one phrasing away, with the
                        # answer not merely unspoken but unfetched. Worse,
                        # the same break dropped WRITES he had asked for --
                        # notes + set_reminder ran the note, skipped the
                        # timer, and told him only about the note. So the
                        # round now finishes the calls the model asked for
                        # (the fan-out cap and the work budget still bound
                        # it) and the decision is made once, below, with
                        # every result of the round in hand.
                        authored.append(line)
                    if over_budget():
                        # the budget is checked INSIDE the round: a round
                        # of many calls must not run to the end first
                        log.warning("chat: over the work budget mid-round "
                                    "(%.1fs)", spent())
                        grant_render_round()
                        break
                # `asked`, not `calls`: the fan-out cap trims what RUNS, and
                # the calls it trimmed are in the transcript too.
                answer_skipped_calls(asked, ran)
                if authored:
                    # THE COVERAGE CHECK. An authored line ends the turn on
                    # the spot -- which is the whole latency win of speak=
                    # -- but it speaks for ONE tool, and 23:20:41 it ended a
                    # turn holding a calendar answer nobody had put into
                    # words. When a result is still owed a sentence, the
                    # lines are HELD and the reserved render round writes
                    # the reply from every result instead. That round is
                    # offered no tools and its tool_calls are dropped, so
                    # unlike the unbacked guard's retry it cannot act --
                    # which is why it may run on a question, and it must:
                    # the live turn was one.
                    #
                    # Held, never lost: they are appended to whatever that
                    # round writes (see held_lines_missing below), because
                    # a write that really happened may not go unreported
                    # on the model's say-so.
                    owed = answers_owed(ran_results[answered_upto:])
                    if owed and render_round_available(
                            "the reply would drop the answer from "
                            + ", ".join(owed)):
                        held_lines = authored
                    else:
                        # Every authored line, not just the first: two
                        # writes in one round are two things he did and
                        # must hear about. No cap is raised for them
                        # because none applies: the reply is code-authored
                        # and _finish_authored below does not count
                        # sentences (F26 -- raising the SENTENCE cap by
                        # len(authored) left the char cap in place, and a
                        # long calendar line trimmed the timer's off).
                        speak = " ".join(authored)
                if speak is None and not render_only and rounds_left > 0 and \
                        over_budget():
                    log.warning("chat: tool loop over the work budget (%.1fs)",
                                spent())
                    rounds_left = 0
                if speak is None and not render_only and rounds_left == 0 and \
                        tool_texts:
                    # THE RESERVATION. Out of rounds with a result in hand is
                    # exactly the 14:35 failure; one writing-only round turns
                    # it into a slower but real answer.
                    grant_render_round()
                if speak is None and rounds_left == 0:
                    # Never the tool text itself: it can carry a stranger's
                    # words (a mail subject, a calendar title, a web page)
                    # and those are not spoken as if they were Jarvis's own.
                    # Only the SOURCE is named, from the tool's own name.
                    final, authored_final = ((degrade(), True) if tool_texts
                                             else ("", False))
        except OllamaDown:
            log.warning("ollama connection refused")
            bus.publish(Status(text="Ollama isn't running", kind="warn"))
            # A model that dies DURING the repair round must not cost him
            # the news of a write that already happened: these two early
            # returns dropped the held lines on the floor and answered
            # "my local model is down, sir." about a note that was added.
            if held_lines:
                return [("SPEAK", degrade())]
            return [("SPEAK", MODEL_DOWN_LINE)]
        except (MalformedReply, ValueError) as exc:
            # a proxy's HTML error page, OpenAI-style content blocks, a
            # truncated body: not the /api/chat shape, so there is nothing
            # to say but the honest line (the detail goes to the log).
            log.warning("ollama reply was malformed: %s", exc)
            bus.publish(Status(text="Local model reply was unreadable",
                               kind="warn"))
            if held_lines:
                return [("SPEAK", degrade())]
            return [("SPEAK", MODEL_EMPTY_LINE)]
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            log.warning("ollama request failed: %s", exc)
            if streamed_sentences:
                # the model died mid-reply after some of it was already
                # spoken: keep what was said rather than say it timed out
                final = " ".join(streamed_sentences)
            elif tool_texts:
                final, authored_final = degrade(), True
            else:
                bus.publish(Status(text="Local model timed out", kind="warn"))
                return [("SPEAK", MODEL_SLOW_LINE)]

        guard_ctx = "\n".join([ctx_text, mem_text] + tool_texts)
        authored_reply = speak is not None or authored_final
        if authored_reply:
            # A code-authored reply -- a tool's speak= line, two of them
            # joined, the degrade -- still goes through the guards: notes
            # and to-dos put Hunter's own text on this path, emoji,
            # markdown, ten items and all. It does not go through the
            # prose caps: every sentence of it is a write reported or a
            # source named, and there is no model prose here to trim in
            # its favour (F26, _finish_authored).
            spoken = _finish_authored(speak if speak is not None else final,
                                      guard_ctx, text)
        else:
            spoken = _finish_spoken(final, guard_ctx, text, cap)
        if not spoken:
            # with a held confirmation in hand this is the degrade's case,
            # not an empty turn: he keeps the write and hears what went
            # unspoken beside it
            spoken = degrade() if held_lines else MODEL_EMPTY_LINE
        elif truncated and not authored_reply and \
                not _PARTIAL_RX.search(spoken):
            # Not on an authored reply: the cut was to what the MODEL saw,
            # and no model prose is being spoken -- screen_qa's answer was
            # authored off the whole vision result, and the degrade names
            # sources rather than reading from them. The notice used to
            # cost the reply a sentence here, which on two confirmations
            # was a write he was owed (F26).
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
        if held_lines and not held_said:
            # THE WRITE IS NOT THE MODEL'S TO DROP. The first cut of this
            # check held the confirmation only for the degrade -- if the
            # render round wrote ANY prose the held line was discarded and
            # the write survived only if the model chose to mention it. It
            # need not: "You have a meeting with ValerieAnne at 11:15
            # tomorrow, sir." was a legal render of the 23:20 turn, and
            # milk went on the list with nobody telling him. The spoken cap
            # could eat it even when the model DID write it (cap 4 on a
            # five-sentence calendar render trims the tail, which is
            # exactly where a confirmation lands).
            #
            # So it is appended here, after the cap, by the code. Skipped
            # only when the reply already carries the line VERBATIM -- the
            # model copying a tool's authored sentence is the common case
            # (app.py's #144 was that same duplicate) -- and a miss costs a
            # repeat, never a silence.
            missing = held_lines_missing(spoken, held_lines)
            if missing:
                spoken = append_spoken_lines(spoken, missing)
            if streamed_sentences and on_sentence is not None \
                    and not self._stale(gen):
                # The render round's prose went out sentence by sentence
                # and SPEAK will not be spoken again (the STREAMED tag):
                # the confirmation has to go to TTS itself or it is written
                # and never said. Judged against what on_sentence actually
                # RECEIVED, not against `spoken`: the two disagree (the
                # stream's splitter counts "a.m." as a sentence end and
                # spends the cap early; _finish_spoken char-caps and the
                # stream does not), so judging on `spoken` left the line
                # unsaid in one shape and said twice in the other (F25,
                # reproduced 2026-09-03 both ways).
                unheard = held_lines_missing(" ".join(streamed_sentences),
                                             held_lines)
                for line in unheard:
                    streamed_sentences.append(line)
                    try:
                        on_sentence(line)
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
                if reasoning_block_open(buf):
                    # Hold: guard() scrubs ONE sentence, so a reasoning
                    # block spanning a sentence break gets its opener
                    # deleted and its CONTENTS spoken. Measured on the
                    # real loop, 2026-09-02: "<think>He wants a greeting.
                    # It is 2 pm.</think>Good afternoon, sir." streamed
                    # "He wants a greeting." to TTS. Nothing goes out
                    # until the block closes and can be cut whole; the
                    # cost is that a finished sentence sitting in front of
                    # an open block waits for it, which is a few hundred
                    # milliseconds against reading the model's monologue
                    # to the room.
                    continue
                buf = _CLOSED_REASONING_RX.sub(" ", buf)
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
