"""Command routing for Jarvis V3 — IntentClassifier, command tables, registry.

Replaces the monolith's fused dispatcher:
  - IntentClassifier              ported verbatim from voice_input_gui.py 163-345
  - command tables                ported from voice_input_gui.py 377-510
  - _check_quick_command branches (3036-3485) as an ordered REGISTRY
  - routing pipeline              from _transcribe_worker 2559-2680 +
                                  _on_transcription 2820-2940

Handlers never touch widgets. They call injected services and return a
CommandResult; asynchronous work publishes JarvisReply/Status events on the
bus. Services namespace (constructor arg) provides, lazily and optionally:

    desktop    parse_action(part)->tuple|None, execute_actions(actions),
               get_window_list()->[(id,name)], type_text(text),
               screenshot(text=None), target_window(query), reset_target(),
               handle_action(action), move_window_to_monitor(direction)
    workflows  get(name)->steps|None, run(name), set_reminder(seconds, task),
               set_trigger(condition, message)
    brain      think(text), execute_autonomous(task)
    memory     remember(key, value), recall(query)->[{key,value,time}],
               save_note(text), get_notes()->[{file,content}],
               suggest_by_habit()->str|None, log_habit(command)
    context    get_last_window(), click_on_text(target), analyze_screen(),
               list_heavy_processes(), git_summary(), check_connectivity(),
               find_file(name), recent_files(), get_clipboard_history(),
               paste_from_history(idx), answer_question(text),
               run_shell(cmd), interpret_intent(text)
    tts        speak(text), interrupt()->bool, last_text, repeat_last()
    reader     read_clipboard(), read_selection(), read_file(name),
               read_text(text), continue_reading() -> ReadResult,
               stop(), pending_chunks, resolve_document(name),
               document_text(path), read_document(path)
    docs       DocsIndex parked by tools.docs.make_tools (quiz mode reads
               chunks straight from the store)
    flashcards FlashcardStore (optional; built lazily under MEMORY_DIR)
    reply      reply(text, speak=True): a worker-thread answer's door to
               the UI, the TTS, the follow-up window and the turn ledger

Personal-assistant services (spec 2026-08-26, sections 2.2 and 5.2; every
one optional — a missing member falls back to the legacy path):

    assistant  AssistantConfig: get(dotted, default), setup_line(section)
    router     Router: route(text, active_project) -> RouteDecision,
               pending(), resolve_answer(text), clear_pending()
    brain      + chat(text, force_tool=None, force_args=None), web_answer(question, model),
               local_line(instruction, text,
               fallback=""), classify_route(text)
    timekeeper add_timer(seconds, label), add_reminder(due, text),
               add_alarm(due, label, repeat), list_text(kind), cancel(which,
               kind), ringing, stop_ringing(action), snooze(minutes),
               parse_when(text, now), describe_due(due, now)
    notes      add(kind, text), list_text(kind), complete(which)
    claude     submit(prompt, project, parallel, model), cancel(), work_on(name),
               resume(utterance, when, name), new_project(name),
               set_model(alias), set_fast_mode(on), active_project
    approvals  pending() -> list, answer(allowed, request_id=None, source)

handle() order: dictation -> ringing-alarm words -> pending approval yes/no
-> open quiz question (the answer) -> terminal offer -> event confirm
-> pending router question -> desktop chains -> registry -> voice intent
gate -> jarvis-mode Tier 1 -> Router -> (local: brain.chat | claude:
claude.submit | ask: one question | action: claude.<action>).
"""
from __future__ import annotations

import inspect
import json
import os
import random
import re
from datetime import datetime, timedelta
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from jarvis import lecture as lecture_mod
from jarvis import pronounce, standup
from jarvis import reader as reader_mod
from jarvis.config import CONFIG, PATHS
from jarvis.events import JarvisReply, Status, bus
from jarvis.logs import get_logger
from jarvis.tools.briefing import OFFER_TTL_S
from jarvis.memory import parse_person_statement, parse_since
from jarvis.tools import notes as notes_mod
from jarvis.tools import quiz as quiz_mod
from jarvis.tools.calendar import write_event
from jarvis.tools.docs import EmbedError, INDEXING_LINE, topic_chunks
from jarvis.tools.notes import number_word
from jarvis.router import ROUTER_QUESTION, WEB_CUE_RX, RouteDecision, estimate_size

log = get_logger("commander")

# Fixed persona lines (spec 3.4) — prewarmed in the speech cache by the app.
ALLOWED_LINE = "Allowed, sir."
DECLINED_LINE = "Declined, sir."
STOPPED_LINE = "Stopped, sir."
NOTHING_RUNNING_LINE = "Nothing's running, sir."
CLAUDE_ACK_FALLBACK = "Right away, sir."
CLAUDE_SETUP_LINE = "I'll need Claude set up first, sir."
WEB_LOOKUP_LINE = "Looking that up, sir."
WEB_UNAVAILABLE_LINE = "I can't reach the web for that just now, sir."
TIMEKEEPER_SETUP_LINE = "I'll need my timekeeper set up first, sir."
NO_WHEN_LINE = "I didn't catch when, sir."


# ------------------------------------------------------------------
# Intent classifier — ported verbatim from voice_input_gui.py 163-345
# ------------------------------------------------------------------
class IntentClassifier:
    """Learns whether speech is directed at the assistant or is background chat.

    Three-tier system:
    - CONFIDENT_YES: clearly a command/question → type immediately
    - CONFIDENT_NO: clearly side conversation → discard silently
    - UNCERTAIN: ask user "Was this for me?" → log answer to improve

    Logged examples are stored in ~/.aiws_trainer/intent_log.json and used
    to train a simple text classifier that improves over time.
    """

    # JARVIS_INTENT_LOG: the test suite (tests/conftest.py) redirects the
    # feedback log. Tests that answered "Was that for me?" used to write
    # label noise straight into the user's real classifier data.
    INTENT_LOG = Path(os.environ.get("JARVIS_INTENT_LOG") or
                      (Path.home() / ".aiws_trainer" / "intent_log.json"))
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"

    # Patterns that strongly suggest assistant-directed speech
    _POSITIVE_PATTERNS = [
        "how do", "how can", "can you", "could you", "would you", "what is",
        "what are", "what's", "where is", "where's", "why is", "why does",
        "is there", "are there", "do you", "tell me", "show me", "explain",
        "implement", "fix", "create", "make", "build", "add", "remove",
        "delete", "update", "change", "modify", "run", "check", "test",
        "start", "stop", "open", "close", "save", "commit", "push",
        "install", "deploy", "debug", "refactor", "write",
        "jarvis", "claude", "hey claude",
        "the code", "the file", "the bug", "the error", "the gui",
        "the config", "the model", "the script", "the function",
        "this file", "this code", "this bug",
        "let's", "let me", "i want", "i need", "i'd like",
        # --- assistant vocabulary (timers, media, calendar, mail) ---
        # Absent until 2026-08-27, so "play some jazz", "volume up" and
        # "remind me to call the supplier" were classified NO and dropped
        # silently -- and only UNCERTAIN ever prompts, so the feedback loop
        # could never learn its way out of it.
        "remind", "timer", "alarm", "wake me", "snooze",
        "play", "pause", "resume", "skip", "next track", "previous track",
        "volume", "turn up", "turn down", "mute", "louder", "quieter",
        "read me", "read the", "what time", "the time",
        "weather", "forecast", "temperature",
        "calendar", "schedule", "agenda", "my day", "briefing", "brief me",
        "note", "email", "inbox", "message",
        "set a", "set an", "set the", "cancel",
        "how long", "how many", "when is", "when's",
        "summarize", "summarise", "look up", "search for", "give me",
        "terminal", "spotify", "music", "song", "track", "playlist",
        "go ahead", "please", "take a look", "look at",
        "check the screen", "screenshot", "take a screenshot",
        "take screenshot",
        # --- study (2026-08-30): focus sessions and lecture notes ---
        # "end the session" and "pomodoro" were NO, "focus session on the
        # thesis" UNCERTAIN -- the same silent drop as the media words above.
        "session", "pomodoro", "study", "focus", "how much time", "time left",
        "notes for", "notes on", "end notes", "lecture",
        # --- review round (2026-08-30): quiz, standup, logs, day review ---
        # The THIRD recurrence of the same silent drop (media words 08-27,
        # study words earlier today). The Tier-1 probe in _handle_inner now
        # bypasses the gate for exact command matches, and this vocabulary
        # covers the looser phrasings that reach the classifier anyway.
        "quiz", "flashcard", "flash card", "standup", "stand-up", "drill",
        "review", "cards", "yesterday go", "your logs", "the logs", "triage",
        "what did i do", "what did i miss",
        # --- named lists (2026-08-30) ---
        # The Tier-1 probe covers the exact phrasings; these carry the
        # looser ones ("anything else on the shopping list?") past the gate.
        "shopping list", "grocery list", "packing list", "reading list",
        "on the list", "off the list", "on my list", "my lists",
    ]

    # Patterns that suggest casual/side conversation
    # Verbs that make a 1-3 word utterance a command outright.
    _SHORT_COMMAND_VERBS = (
        "run", "fix", "check", "stop", "test", "take",
        "commit", "push", "show", "open", "screenshot",
        "save", "deploy", "start", "build", "install",
        # assistant-era, added 2026-08-27
        "play", "pause", "resume", "skip", "next", "back",
        "mute", "unmute", "snooze", "cancel", "read", "set",
        "call", "remind", "close", "quit", "louder", "quieter",
        "volume", "repeat", "again",
        # review round 2026-08-30: 1-3-word Tier-1 phrases
        "quiz", "review", "standup", "drill", "recap", "triage", "explain",
    )

    _NEGATIVE_PATTERNS = [
        "bless her", "bless him", "oh my god", "that's crazy",
        "no way", "for real", "i know right", "lol", "haha",
        "she said", "he said", "they said", "she's", "he's",
        "dude", "bro", "man ", "yo ",
    ]

    def __init__(self):
        self._log_data = []  # List of {"text": ..., "label": "yes"/"no"}
        self._learned_positive = set()  # Phrases learned as positive
        self._learned_negative = set()  # Phrases learned as negative
        self._load_log()

    def _load_log(self):
        """Load logged intent examples from disk."""
        if not self.INTENT_LOG.exists():
            return
        try:
            self._log_data = json.loads(self.INTENT_LOG.read_text())
            # Build learned pattern sets from logged examples
            for entry in self._log_data:
                text_lower = entry["text"].lower()
                words = text_lower.split()
                label = entry["label"]
                # Extract 2-3 word ngrams as learned patterns
                for n in (2, 3):
                    for i in range(len(words) - n + 1):
                        ngram = " ".join(words[i:i + n])
                        if label == self.YES:
                            self._learned_positive.add(ngram)
                            self._learned_negative.discard(ngram)
                        else:
                            self._learned_negative.add(ngram)
                            self._learned_positive.discard(ngram)
        except Exception:
            log.exception("intent log load failed; starting empty")
            self._log_data = []

    def _save_log(self):
        """Save intent log to disk."""
        try:
            self.INTENT_LOG.parent.mkdir(parents=True, exist_ok=True)
            self.INTENT_LOG.write_text(json.dumps(self._log_data, indent=2))
        except Exception:
            log.exception("intent log save failed")

    def log_feedback(self, text, is_for_assistant):
        """Record user feedback on whether text was directed at assistant."""
        label = self.YES if is_for_assistant else self.NO
        self._log_data.append({"text": text, "label": label})

        # Keep log manageable (last 500 entries)
        if len(self._log_data) > 500:
            self._log_data = self._log_data[-500:]

        # Update learned patterns
        words = text.lower().split()
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                if is_for_assistant:
                    self._learned_positive.add(ngram)
                    self._learned_negative.discard(ngram)
                else:
                    self._learned_negative.add(ngram)
                    self._learned_positive.discard(ngram)

        self._save_log()

    def classify(self, text):
        """Classify text as YES, NO, or UNCERTAIN.

        Returns (classification, confidence) where confidence is 0-1.
        """
        if not text or len(text.strip()) < 3:
            return self.NO, 1.0

        lower = text.strip().lower()
        words = lower.split()
        pos_score = 0
        neg_score = 0

        # --- Rule-based signals ---

        # Very short reactions (1-3 words)
        if len(words) <= 3:
            if any(lower.startswith(p) for p in self._SHORT_COMMAND_VERBS):
                return self.YES, 0.9
            if "screenshot" in lower:
                return self.YES, 0.9
            # A short command can carry no verb from that list at all --
            # "volume up", "next track", "snooze" are vocabulary, not syntax.
            if any(pattern in lower for pattern in self._POSITIVE_PATTERNS):
                return self.YES, 0.85
            if lower.endswith("?"):
                return self.YES, 0.8
            return self.NO, 0.8

        # Strong positive patterns
        for pattern in self._POSITIVE_PATTERNS:
            if pattern in lower:
                pos_score += 2

        # Strong negative patterns
        for pattern in self._NEGATIVE_PATTERNS:
            if pattern in lower:
                neg_score += 2

        # Questions
        if lower.rstrip().endswith("?"):
            pos_score += 1.5

        # 3rd person pronouns (talking about others)
        other_pronouns = {"she", "he", "they", "her", "him", "them", "his"}
        pronoun_count = sum(1 for w in words if w in other_pronouns)
        neg_score += pronoun_count * 0.5

        # (Removed: "len(words) >= 8 and pos_score == 0 -> neg_score += 1".
        # Length says nothing about who you are talking to, and it was what
        # pushed "remind me to call the supplier at four" to NO 1.00 -- a real
        # command, discarded with no prompt.)

        # --- Learned patterns from feedback ---
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                if ngram in self._learned_positive:
                    pos_score += 1.5
                if ngram in self._learned_negative:
                    neg_score += 1.5

        # --- Decision ---
        total = pos_score + neg_score
        if total == 0:
            # No signals either way — uncertain
            return self.UNCERTAIN, 0.5

        pos_ratio = pos_score / total

        if pos_ratio >= 0.7:
            return self.YES, pos_ratio
        elif pos_ratio <= 0.3:
            return self.NO, 1 - pos_ratio
        else:
            return self.UNCERTAIN, 0.5

    @property
    def num_examples(self):
        return len(self._log_data)


# ------------------------------------------------------------------
# Command tables — ported from voice_input_gui.py 125-128 + 377-480
# ------------------------------------------------------------------

# Filler sounds that indicate thinking — reset silence timer when detected
# Only pure filler sounds, NOT common words like "like", "so", "well"
FILLER_WORDS = {"uh", "um", "uhh", "umm", "hmm", "hm", "er", "ah", "ehh", "eh",
                "erm", "uhhh", "ummm"}

# Voice commands — spoken phrase → replacement
VOICE_COMMANDS = [
    # Punctuation
    (r"\b(?:period|full stop)\b", "."),
    (r"\bcomma\b", ","),
    (r"\b(?:question mark)\b", "?"),
    (r"\b(?:exclamation mark|exclamation point)\b", "!"),
    (r"\bcolon\b", ":"),
    (r"\bsemicolon\b", ";"),
    (r"\b(?:dash|hyphen)\b", "-"),
    (r"\bellipsis\b", "..."),
    (r"\b(?:open paren|open parenthesis|left paren)\b", "("),
    (r"\b(?:close paren|close parenthesis|right paren)\b", ")"),
    (r"\b(?:open quote|open quotes|begin quote)\b", '"'),
    (r"\b(?:close quote|close quotes|end quote)\b", '"'),
    (r"\b(?:single quote|apostrophe)\b", "'"),
    # Whitespace / structure
    (r"\b(?:new line|newline|line break)\b", "\n"),
    (r"\btab\b(?:\s+(?:key|character))?", "\t"),
    # Editing (special actions handled separately)
    (r"\b(?:backspace|back space)\b", "\x08"),
]

# Special action commands (not simple replacements)
ACTION_COMMANDS = {
    "delete that": "delete_last_sentence",
    "scratch that": "delete_last_sentence",
    "undo that": "delete_last_sentence",
    "clear all": "clear_all",
    "select all": "select_all",
}

# Screenshot trigger phrases — stripped from text, triggers capture after typing
SCREENSHOT_PHRASES = [
    "and take a screenshot", "and take screenshot", "and screenshot",
    "take a screenshot", "take screenshot", "capture screen",
    "screen capture", "screenshot",
]

# Voice targeting patterns — "target X", "switch to X", "type in X", "go to X"
TARGET_PATTERN = re.compile(
    r"^(?:target|switch to|type in|go to|focus|open)\s+(.+)$",
    re.IGNORECASE,
)
# Reset target back to auto
TARGET_RESET_PHRASES = {"target auto", "target claude", "reset target",
                        "target default"}

# Voice phrases that stop recording (stripped from final transcription)
STOP_RECORDING_PHRASES = {
    "end recording", "stop recording", "stop listening",
    "done recording", "finish recording",
}

# Quick voice commands — "Jarvis, commit" etc.
# (hardcoded /home/hunterp/vss_env replaced with PATHS.VSS_ENV)
QUICK_COMMANDS = {
    "commit": f"cd {PATHS.VSS_ENV} && git add -A && git status -s",
    "run tests": f"cd {PATHS.VSS_ENV} && python scripts/agents/run_all.py --quick",
    "check gpu": "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv",
    "check disk": "df -h / /storage 2>/dev/null",
    "check logs": "tail -20 /tmp/vss_voice/gui_debug.log",
    "system status": "uptime && free -h && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
}

# Desktop control commands — parsed from natural language (data consumed by
# jarvis.desktop's parser)
DESKTOP_ACTIONS = {
    # Window management
    "switch to": "window",
    "go to": "window",
    "open": "window",
    "focus": "window",
    "close window": "close",
    "minimize": "minimize",
    "maximize": "maximize",
    "full screen": "fullscreen",
    # Mouse/scroll
    "scroll up": "scroll_up",
    "scroll down": "scroll_down",
    "scroll left": "scroll_left",
    "scroll right": "scroll_right",
    "click": "click",
    "double click": "double_click",
    "right click": "right_click",
    # Tabs
    "next tab": "next_tab",
    "previous tab": "prev_tab",
    "new tab": "new_tab",
    "close tab": "close_tab",
    # System
    "volume up": "vol_up",
    "volume down": "vol_down",
    "mute": "mute",
    "play": "media_play",
    "pause": "media_pause",
    # Keyboard shortcuts
    "copy": "copy",
    "paste": "paste",
    "undo": "undo",
    "redo": "redo",
    "save": "save",
    "select all": "select_all",
    "find": "find",
}

# "jarvis, <command>" prefixes (voice_input_gui.py 3041 / 3553)
JARVIS_PREFIXES = ("jarvis ", "jarvis, ", "hey jarvis ", "hey jarvis, ")


def _apply_voice_commands(text):
    """Apply voice command replacements to transcribed text.

    Ported verbatim from voice_input_gui.py 483-510.
    """
    result = text

    # Check for action commands first (full phrase match)
    text_lower = result.strip().lower()
    for phrase, action in ACTION_COMMANDS.items():
        if text_lower == phrase or text_lower.endswith(phrase):
            return f"__ACTION__{action}"

    # Apply punctuation/whitespace replacements
    for pattern, replacement in VOICE_COMMANDS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Clean up spaces around punctuation and newlines
    result = re.sub(r"\s+([.,!?;:)\]])", r"\1", result)
    result = re.sub(r"([([\[])\s+", r"\1", result)
    result = re.sub(r"\s*\n\s*", "\n", result)

    # Handle backspace markers
    while "\x08" in result:
        idx = result.index("\x08")
        if idx > 0:
            result = result[:idx - 1] + result[idx + 1:]
        else:
            result = result[1:]

    return result.strip()


_ADDRESS_RX = re.compile(r"^(?:(?:hey|ok|okay|hi)[,\s]+)?jarvis[,!:]?\s+", re.I)


def strip_address(text: str) -> str:
    """"Hey Jarvis, what's the weather?" -> "what's the weather?".

    Unlike strip_jarvis_prefix this keeps the casing and the punctuation
    (the local model must see a normal sentence, question mark and all)
    and returns the text unchanged when it carries no address.
    """
    t = (text or "").strip()
    return _ADDRESS_RX.sub("", t, count=1).strip() or t


def strip_jarvis_prefix(text: str) -> Optional[str]:
    """Return the command text after a 'jarvis' prefix, or None.

    Ported from voice_input_gui.py 3038-3046 (same logic at 3550-3558).
    """
    lower = text.strip().lower().rstrip(".")
    for prefix in JARVIS_PREFIXES:
        if lower.startswith(prefix):
            return lower[len(prefix):].strip()
    return None


# ------------------------------------------------------------------
# Command result + registry types
# ------------------------------------------------------------------
@dataclass
class CommandResult:
    handled: bool
    reply: Optional[str] = None       # text to show (and speak when speak=True)
    speak: bool = False               # speak `reply` via TTS
    status: Optional[str] = None      # short status-strip text
    done: bool = True                 # False → async work still in flight
    ack: bool = False                 # `reply` acknowledges; the answer follows
    # Set when the result answers a DIFFERENT utterance than the one
    # handled: "no, I said X" re-dispatches X, "that was for you" re-runs
    # the dropped command. The app records the exchange under this text.
    corrected: Optional[str] = None
    # How to take this turn back (spec 5): a handler that CREATED something
    # -- a timer, an alarm, a reminder, a note, a list item -- hands back a
    # closure that removes exactly that thing and returns the line to speak.
    # Commander.handle stashes it for UNDO_WINDOW_S so "scratch that" can
    # run it. None means the turn cannot be undone, and "scratch that"
    # keeps its old dictation meaning.
    undo: Optional[Callable[[], str]] = None


@dataclass
class Command:
    name: str
    matcher: Callable[[str], Any]     # cmd_text -> truthy match or falsy
    handler: Callable                 # (commander, cmd_text, match) -> CommandResult|None
    needs: tuple = ()                 # required service names


# ---- matcher helpers ----------------------------------------------------
def _m_exact(*phrases):
    return lambda t: t in phrases


def _m_contains(*phrases):
    return lambda t: any(p in t for p in phrases)


def _m_re(pattern):
    rx = re.compile(pattern)
    return rx.match


def _talkback() -> bool:
    return bool(CONFIG.talkback)


# ---- Tier 1 clock ---------------------------------------------------
# "What time is it" is answered from datetime.now() here, in Jarvis's
# voice, so the Tier 2 model never has to (it invented "23:47" when the
# context lacked a clock). Matched both with the jarvis prefix (registry)
# and without it in jarvis mode (_route_text), right before the brain.
_CLOCK_KINDS = (
    ("time", re.compile(
        r"\b(?:what(?:'s| is|s)?\s+(?:the\s+)?(?:current\s+)?time\b|"
        r"what time is it|(?:have you|do you have|you) got the time|"
        r"do you have the time|time is it\b|current time\b)", re.I)),
    ("date", re.compile(
        r"\b(?:what(?:'s| is|s)?\s+(?:the\s+|today's\s+)?date\b|"
        r"what date is it|date today\b|today's date)", re.I)),
    ("day", re.compile(
        r"\b(?:what day (?:is it|is today|of the week)|which day is it)\b",
        re.I)),
)


# "what's the time in London" names a place: that is the get_time tool's job
# (it geocodes the city for a timezone), not the local clock's. The Tier-1
# match answered it with the home time (live, 2026-08-29 22:15).
_CLOCK_PLACE_RX = re.compile(
    r"\b(?:time|date|day)\b.*?\b(?:in|at|over in|for)\s+(?!the\b|a\b|an\b)[a-z]", re.I)


def clock_kind(text: str) -> Optional[str]:
    """'time' / 'date' / 'day' when the text asks for the clock, else None.
    A question that names a place is left to the router and get_time."""
    text = text or ""
    for kind, rx in _CLOCK_KINDS:
        if rx.search(text):
            return None if _CLOCK_PLACE_RX.search(text) else kind
    return None


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else \
        {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def clock_reply(now: datetime, kind: str = "time") -> str:
    """The clock in Jarvis's voice: 12-hour, spoken, one sentence."""
    weekday = now.strftime("%A")
    if kind == "date":
        return (f"It's {weekday} the {_ordinal(now.day)} of "
                f"{now.strftime('%B')}, sir.")
    if kind == "day":
        return f"It's {weekday}, sir."
    hour = now.hour
    if 5 <= hour < 12:
        part = "in the morning"
    elif 12 <= hour < 17:
        part = "in the afternoon"
    elif 17 <= hour < 21:
        part = "in the evening"
    else:
        part = "at night"
    return f"It's {hour % 12 or 12}:{now.minute:02d} {part}, sir."


def _h_clock(c, t, m):
    # Spoken regardless of talk-back: a clock question asked aloud wants
    # the answer aloud, exactly as the brain's replies are always spoken.
    return CommandResult(handled=True, reply=clock_reply(datetime.now(), m),
                         speak=True, status="Clock")


# ---- Tier 1 courtesy --------------------------------------------------
# "Are you there", "thank you" and "good night" have one right answer
# each and the 3B model muddles short "..., Jarvis" phrases (it has
# answered "good night" with the thank-you line, and "nice one, Jarvis"
# as sarcasm). The canonical phrasings and the common thank-you
# paraphrases are answered here, instantly and in voice, with a little
# variation; other paraphrases still reach the brain.
_JV = r"(?:jarvis[,!]?\s*)?"
_COURTESY_KINDS = (
    ("presence", re.compile(
        r"^" + _JV + r"(?:are you (?:there|awake|listening|around|still "
        r"there)|you there|still there|can you hear me|you awake)"
        r"(?:[,]?\s*jarvis)?[?.!\s]*$", re.I)),
    ("thanks", re.compile(
        r"^" + _JV + r"(?:thank you|thanks|cheers|much obliged|ta|"
        r"(?:much )?appreciated|appreciate it|nice one|good job|"
        r"well done|nicely done|spot on|brilliant)"
        r"(?:\s+(?:very much|a lot|so much|for that))?"
        r"(?:[,]?\s*jarvis)?[.!\s]*$", re.I)),
    ("goodnight", re.compile(
        r"^" + _JV + r"(?:good ?night|night night|night|sleep well|"
        r"(?:i'm )?off to bed|going to bed)"
        r"(?:[,]?\s*jarvis)?[.!\s]*$", re.I)),
)

# ---- Tier 1 greetings -------------------------------------------------
# "hello", "good morning", "how are you" and "are you busy" have one right
# answer each and were the only courtesy shapes still costing a full model
# turn.  They are matched LATER than the courtesies above -- after the
# briefing command in the registry -- because "good morning" is the phrase
# that triggers the morning briefing when the user has it switched on.
_GREETING_KINDS = (
    ("greeting", re.compile(
        r"^" + _JV + r"(?:hello|hi|hey|hiya|howdy|yo|greetings|"
        r"good (?:morning|afternoon|evening)|morning|afternoon|evening)"
        r"(?:\s+there)?(?:[,]?\s*jarvis)?[?.!\s]*$", re.I)),
    ("wellbeing", re.compile(
        r"^" + _JV + r"(?:how are you(?: doing| today| feeling)?|"
        r"how'?s it going|how are things|how do you do|how goes it|"
        r"what'?s up|you all right|are you (?:ok|okay|well))"
        r"(?:[,]?\s*jarvis)?[?.!\s]*$", re.I)),
    ("availability", re.compile(
        r"^" + _JV + r"(?:are you (?:busy|free|available|occupied)|"
        r"you (?:busy|free)|(?:have you )?got a (?:minute|moment|second))"
        r"(?:[,]?\s*jarvis)?[?.!\s]*$", re.I)),
)
COURTESY_REPLIES = {
    "presence": ["Always, sir.", "Right here, sir.", "Yes, sir."],
    "thanks": ["Not at all, sir.", "It's rather what I'm for, sir.",
               "Any time, sir."],
    "goodnight": ["Good night, sir. I'll be here.",
                  "Good night, sir; I'll be here in the morning.",
                  "Sleep well, sir."],
    # Picked by the clock rather than at random -- see courtesy_reply().
    "greeting": ["Good morning, sir.", "Good afternoon, sir.",
                 "Good evening, sir."],
    "wellbeing": ["All systems nominal, sir.", "Very well, sir.",
                  "Never better, sir."],
    "availability": ["Never too busy for you, sir.", "Quite free, sir.",
                     "Nothing pressing, sir."],
}


def _greeting_line(now: Optional[datetime] = None) -> str:
    """The time-appropriate line from COURTESY_REPLIES["greeting"]."""
    hour = (now or datetime.now()).hour
    idx = 0 if 5 <= hour < 12 else 1 if 12 <= hour < 17 else 2
    return COURTESY_REPLIES["greeting"][idx]


def courtesy_kind(text: str) -> Optional[str]:
    """'presence' / 'thanks' / 'goodnight' for a whole-utterance courtesy,
    else None (so "thanks, now open the terminal" is not swallowed)."""
    for kind, rx in _COURTESY_KINDS:
        if rx.match((text or "").strip()):
            return kind
    return None


def courtesy_reply(kind: str, rng=None) -> str:
    # "Good afternoon" answered with "Good morning" is worse than no
    # variation at all, so the greeting is chosen by the clock.
    if kind == "greeting":
        return _greeting_line()
    return (rng or random).choice(COURTESY_REPLIES[kind])


def _h_courtesy(c, t, m):
    if m == "goodnight":
        # "Good night" is a courtesy first (this runs ahead of the registry's
        # own good-night entry and again in _route_text): the wind-down
        # preview hangs off it here, and falls back to the plain line.
        res = _goodnight_preview(c, t)
        if res is not None:
            return res
    return CommandResult(handled=True, reply=courtesy_reply(m), speak=True,
                         status="Courtesy")


def greeting_kind(text: str) -> Optional[str]:
    """'greeting' / 'wellbeing' / 'availability' for a whole-utterance
    greeting, else None."""
    for kind, rx in _GREETING_KINDS:
        if rx.match((text or "").strip()):
            return kind
    return None


def _h_greeting(c, t, m):
    return CommandResult(handled=True, reply=courtesy_reply(m), speak=True,
                         status="Greeting")


# ---- Tier 1 voice I/O: quiet / say again / pronounce / read aloud -------
# Answered locally, never by Tier 2: "quiet" has to cut the speech NOW,
# "say again" replays the last line verbatim, "pronounce X as Y" edits the
# TTS dictionary and "read the clipboard" streams text through the TTS in
# chunks — none of which a language model can do for him. A bare "stop"
# is quiet too ("stop recording/listening" stay STOP_RECORDING_PHRASES,
# stripped from the end of a voice transcript before this runs).
_QUIET_RX = re.compile(
    r"^" + _JV + r"(?:stop(?: it| now| there| talking| speaking| reading)?|"
    r"be quiet|quiet|hush|shush|shh+|shut up|shut it|silence|that's enough|"
    r"that'll do|enough|never ?mind|cancel that|stop that|zip it|pipe down)"
    r"(?:[,]?\s*jarvis)?[.!\s]*$", re.I)
_REPEAT_RX = re.compile(
    r"^" + _JV + r"(?:say (?:that |it )?again|repeat (?:that|it)|come again|"
    r"pardon(?: me)?|what was that|sorry,? what|once more|repeat)"
    r"(?:[,]?\s*jarvis)?[?.!\s]*$", re.I)
_PRONOUNCE_RX = re.compile(
    r"^" + _JV + r"pronounce\s+[\"']?(.+?)[\"']?\s+(?:as|like)\s+[\"']?(.+?)"
    r"[\"']?[.!\s]*$", re.I)
_READ_RX = re.compile(
    r"^" + _JV + r"read\s+(?:me\s+|out\s+|back\s+)?(?:"
    r"(?P<clip>(?:the\s+|my\s+)?clipboard|what i copied|what i just copied)"
    r"|(?P<sel>(?:the\s+|my\s+)?(?:selection|selected text|highlighted text|"
    r"highlight)|this|that|it)"
    r"|(?P<doc>(?:the\s+|that\s+|this\s+|my\s+)?(?:whole\s+|full\s+|entire\s+)?"
    r"(?:document|doc|handout|pdf|paper|summary(?:'s|\s+document)?)(?:\s+in full)?)"
    r"|file\s+(?P<file>\S.*?)"
    r"|(?:aloud|out loud)[:,]?\s+(?P<inline>.+?)"
    r"|(?P<inline2>.+?)\s+(?:aloud|out loud)"
    r")(?:\s+(?:aloud|out loud|to me|back to me|back))?[.!\s]*$", re.I)
_CONTINUE_RX = re.compile(
    r"^" + _JV + r"(?:(?:continue|keep|carry on|resume)\s+reading|"
    r"go on|next part|carry on|continue|keep going|more)"
    r"(?:[,]?\s*jarvis)?[.!\s]*$", re.I)


def quiet_kind(text: str) -> bool:
    return bool(_QUIET_RX.match((text or "").strip()))


def repeat_kind(text: str) -> bool:
    return bool(_REPEAT_RX.match((text or "").strip()))


def read_kind(text: str) -> Optional[tuple]:
    """('clipboard'|'selection'|'file'|'text', arg) or None."""
    m = _READ_RX.match((text or "").strip())
    if not m:
        return None
    if m.group("clip"):
        return ("clipboard", None)
    if m.group("sel"):
        return ("selection", None)
    if m.group("doc"):
        return ("document", None)
    if m.group("file"):
        return ("file", m.group("file").strip())
    inline = m.group("inline") or m.group("inline2") or ""
    return ("text", inline.strip()) if inline.strip() else None


def continue_kind(text: str) -> bool:
    return bool(_CONTINUE_RX.match((text or "").strip()))


# "cancel that" / "stop that" / "abort" also abort a running Claude task
# (user rule: 'cancel' / 'stop that' aborts the active task). A bare
# "stop" only cuts speech: Claude may keep working while Jarvis hushes.
_CANCEL_TASK_RX = re.compile(
    r"^" + _JV + r"(?:cancel(?: that| it| this| the task| the job| claude|"
    r" everything)?|abort(?: that| it| the task)?|stop (?:that|the task|the job|"
    r"claude|working|the claude task))(?:[,]?\s*jarvis)?[.!\s]*$", re.I)


def cancel_kind(text: str) -> bool:
    return bool(_CANCEL_TASK_RX.match((text or "").strip()))


def _cut_speech(c):
    reader = c._svc("reader")
    if reader is not None:
        try:
            reader.stop()
        except Exception:
            log.exception("reader stop failed")
    tts = c._svc("tts")
    if tts is not None:
        try:
            if hasattr(tts, "interrupt"):
                tts.interrupt()
            else:
                tts.stop()
        except Exception:
            log.exception("tts interrupt failed")


def _h_quiet(c, t, m):
    _cut_speech(c)
    claude = c._svc("claude")
    if claude is not None and cancel_kind(t):
        try:
            if claude.cancel():
                return CommandResult(handled=True, reply=STOPPED_LINE,
                                     speak=True, status="Cancelled")
        except Exception:
            log.exception("claude cancel failed")
    # Shown, never spoken: he was just told to be quiet.
    return CommandResult(handled=True, reply="Very good, sir.", speak=False,
                         status="Quiet")


def _h_repeat(c, t, m):
    tts = c._svc("tts")
    last = getattr(tts, "last_text", "") if tts is not None else ""
    if not last:
        return CommandResult(handled=True,
                             reply="I haven't said anything yet, sir.",
                             speak=True, status="Nothing to repeat")
    if _talkback():
        try:
            tts.repeat_last()
        except Exception:
            log.exception("repeat_last failed")
    return CommandResult(handled=True, reply=last, speak=False,
                         status="Repeating")


def _h_pronounce(c, t, m):
    # The registry path lower-cases the command text; the user's casing
    # ("Peyrovi", "GB10") matters for a dictionary entry, so re-match the
    # raw utterance the commander stashed in handle().
    raw = getattr(c, "_raw_text", "") or ""
    m2 = _PRONOUNCE_RX.match(raw.strip()) if raw else None
    if m2 is not None:
        m = m2
    word, spoken = m.group(1).strip(), m.group(2).strip()
    try:
        pronounce.get().add(word, spoken)
    except Exception:
        log.exception("pronunciation add failed")
        return CommandResult(handled=True,
                             reply="I couldn't save that, sir.", speak=True,
                             status="Pronunciation failed")
    # The reply carries the word, so he confirms it in the new pronunciation.
    return CommandResult(handled=True, reply=f"Noted, sir. {word} it is.",
                         speak=True, status=f"Pronounce {word} as {spoken}")


# ---- Tier 1 name spelling: Whisper and the TTS in one utterance ---------
# "my advisor's name is spelled P-E-Y-R-O-V-I, say it pay-ROH-vee" fixes
# both directions at once: the letters go to the names file the prompt
# builder (jarvis/vocab.py) feeds Whisper, and the "say it" clause goes to
# the same TTS dictionary "pronounce X as Y" edits. Whisper renders spelled
# letters inconsistently ("P E Y R O V I", "P-E-Y-R-O-V-I", "P., E., Y.",
# sometimes glued to "PEYROVI"), so the run is normalised and the reply
# echoes the reconstructed spelling for him to confirm by ear.
_SPELL_NAME_RX = re.compile(
    r"^" + _JV + r"(?:my\s+)?(?:(?P<who>[\w' ]+?)\s+)?(?:name\s+)?"
    r"is\s+spel(?:led|t)(?:\s+(?:as|like))?[:,]?\s+"
    r"(?P<letters>[a-z](?:[\s.,-]+[a-z])+|[a-z]{2,})[.!]?\s*"
    r"(?:[,;]?\s*(?:and\s+)?(?:say|pronounce)\s+(?:it|that|him|her)"
    r"(?:\s+(?:as|like))?\s+[\"']?(?P<spoken>.+?)[\"']?)?[.!\s]*$",
    re.I)
_ADD_VOCAB_RX = re.compile(
    r"^" + _JV + r"add\s+(?:the\s+(?:word|name)\s+)?[\"']?(?P<word>.+?)[\"']?"
    r"\s+to\s+(?:your|the|my)\s+(?:vocab(?:ulary)?|dictionary|word\s?list)"
    r"[.!\s]*$", re.I)


def spelled_word(run: str) -> str:
    """Normalise a spelled letter run to a word: "P-E-Y-R-O-V-I",
    "p e y r o v i" and "P., E., Y." all -> "Peyrovi". Whisper sometimes
    glues the letters into one token ("PEYROVI"), so a single alphabetic
    word passes through title-cased. Anything else (mixed multi-letter
    tokens: "pey rovi") returns "" -- guessing at the word boundary would
    store a wrong name forever."""
    tokens = [t for t in re.split(r"[\s.,-]+", (run or "").strip()) if t]
    if not tokens:
        return ""
    if all(len(t) == 1 and t.isalpha() for t in tokens):
        word = "".join(tokens)
    elif len(tokens) == 1 and tokens[0].isalpha():
        word = tokens[0]
    else:
        return ""
    if len(word) < 2:
        return ""
    return word[0].upper() + word[1:].lower()


def _h_spell_name(c, t, m):
    # The registry path lower-cases the command text; the "say it" clause's
    # casing is his ("pay-ROH-vee"), so re-match the raw utterance the
    # commander stashed in handle(), exactly as _h_pronounce does.
    raw = getattr(c, "_raw_text", "") or ""
    m2 = _SPELL_NAME_RX.match(raw.strip()) if raw else None
    if m2 is not None:
        m = m2
    word = spelled_word(m.group("letters"))
    if not word:
        return CommandResult(
            handled=True, speak=True, status="Spelling unclear",
            reply="I didn't quite catch the letters, sir — once more, "
                  "one at a time?")
    try:
        from jarvis import vocab as vocab_mod
        vocab_mod.add_name(word)
    except Exception:
        log.exception("name save failed")
        return CommandResult(handled=True, speak=True,
                             reply="I couldn't save that, sir.",
                             status="Name save failed")
    spoken = (m.group("spoken") or "").strip()
    if spoken:
        try:
            pronounce.get().add(word, spoken)
        except Exception:
            log.exception("pronunciation add failed")
    # The echo IS the confirmation loop: the spelling proves the letters
    # arrived intact, and the trailing word is spoken through the new
    # pronunciation when one was given.
    spelling = "-".join(word.upper())
    return CommandResult(handled=True, speak=True,
                         reply=f"Noted, sir: {spelling}. {word}.",
                         status=f"Spelled {word}")


def _h_add_vocab(c, t, m):
    # Raw casing matters: "add Librespot ..." must store "Librespot".
    word = " ".join(_raw_group(c, _ADD_VOCAB_RX, m, group="word").split())
    word = word.strip("\"'")
    if not word:
        return CommandResult(handled=True, speak=True,
                             reply="Add what, sir?", status="No word")
    try:
        from jarvis import vocab as vocab_mod
        added = vocab_mod.add_name(word)
    except Exception:
        log.exception("vocabulary add failed")
        return CommandResult(handled=True, speak=True,
                             reply="I couldn't save that, sir.",
                             status="Vocabulary save failed")
    reply = (f"Noted, sir — I'll listen for {word}." if added
             else f"I already have {word}, sir.")
    return CommandResult(handled=True, reply=reply, speak=True,
                         status=f"Vocabulary: {word}")


# "Read it to me" after an explain means the document just explained, not
# the X selection. The pronoun form is only diverted while a document is
# fresh (LAST_DOCUMENT_S); "read this" / "read the selection" never are.
_READ_PRONOUN_RX = re.compile(
    r"^" + _JV + r"read\s+(?:me\s+)?(?:it|that)(?:\s+(?:to me|back|aloud|"
    r"out loud|back to me))?[.!\s]*$", re.I)
LAST_DOCUMENT_S = 900.0
NO_DOCUMENT_LINE = "I haven't a document on hand to read, sir."
NO_SUCH_DOCUMENT_LINE = "I can't find a document called {name}, sir."
EXPLAIN_ACK_LINE = "Let me have a look at {name}, sir."
EXPLAIN_FAIL_LINE = "I'm afraid I couldn't make sense of that document just now, sir."
EXPLAIN_NO_MODEL_LINE = ("I can't summarise just now, sir; say 'read it to me' "
                         "and I'll read it instead.")
READ_OFFER_LINE = "Say 'read it to me' for the whole thing."

# Narrow on purpose (the tool loop handles "summarize my inbox"): a verb of
# explanation plus a name that carries a document word or an extension.
# STRONG document words speak an excuse when nothing resolves; WEAK ones
# ("the report", "that paper") fall through to the router instead, since
# they are as likely to mean a Claude result or a question about the world.
_EXPLAIN_RX = re.compile(
    r"^" + _JV + r"(?:explain|summari[sz]e|sum up|walk me through|brief me on|"
    r"give me (?:a |the )?(?:summary|rundown|gist) of)\s+"
    r"(?:the\s+|my\s+|this\s+|that\s+)?(?P<name>.+?)"
    r"(?:\s+(?:to me|for me|please))?[.!?\s]*$", re.I)
_STRONG_DOC_RX = re.compile(
    r"\b(?:handout|pdf|docx?|document|syllabus|textbook|worksheet|slides|"
    r"lecture notes|rubric|reading)\b|\.(?:pdf|docx|txt|md)\b", re.I)
_WEAK_DOC_RX = re.compile(
    r"\b(?:file|paper|report|article|essay|manual|assignment|spec|"
    r"specification|chapter)\b", re.I)


def explain_kind(text: str) -> Optional[tuple]:
    """('strong'|'weak', name) for "explain the biosensors handout"; None
    for anything without a document word in the name."""
    m = _EXPLAIN_RX.match((text or "").strip())
    if not m:
        return None
    name = m.group("name").strip()
    if _STRONG_DOC_RX.search(name):
        return ("strong", name)
    if _WEAK_DOC_RX.search(name):
        return ("weak", name)
    return None


def _spoken_name(path) -> str:
    stem = re.sub(r"[_\-]+", " ", getattr(path, "stem", str(path)))
    return " ".join(stem.split()) or str(path)


def _deliver(c, text: str, speak: bool = True):
    """A worker thread's answer: through services.reply when the app
    provides it (show, speak, follow-up, close the turn), else the bus and
    the commander's own TTS door."""
    if not text:
        return
    fn = c._svc("reply")
    if callable(fn):
        try:
            fn(text, speak=speak)
            return
        except Exception:
            log.exception("services.reply failed")
    bus.publish(JarvisReply(text=text, speak=speak))
    if speak:
        c._speak(text)


def _h_explain_doc(c, t, m):
    strength, name = m
    reader = c._svc("reader")
    if reader is None or not hasattr(reader, "resolve_document"):
        return None
    path = reader.resolve_document(name)
    if path is None:
        if strength == "weak":
            return None                  # not a file of his: let the router have it
        return CommandResult(handled=True, reply=NO_SUCH_DOCUMENT_LINE.format(name=name),
                             speak=True, status="No such document")
    text = reader.document_text(path)
    if not text.strip():
        return CommandResult(handled=True,
                             reply=f"I couldn't get any text out of {path.name}, sir.",
                             speak=True, status="No text")
    c._last_document = (path, datetime.now().timestamp())
    brain = c._svc("brain")
    if brain is None or not hasattr(brain, "explain_text"):
        return CommandResult(handled=True, reply=EXPLAIN_NO_MODEL_LINE, speak=True,
                             status="No model")
    spoken = _spoken_name(path)

    def _work():
        try:
            lead, summary = brain.explain_text(text, name=spoken)
        except Exception:
            log.exception("explain_text failed")
            lead, summary = "", ""
        if not lead:
            _deliver(c, EXPLAIN_FAIL_LINE)
            return
        # The fuller paragraph is a card (display only, never read aloud);
        # the lead plus the offer is the spoken reply and arms the follow-up
        # window so "read it to me" needs no wake word.
        if summary and summary != lead:
            bus.publish(JarvisReply(text=f"{path.name}\n{summary}", speak=False))
        _deliver(c, f"{lead} {READ_OFFER_LINE}")

    c._bg(_work)
    return CommandResult(handled=True, reply=EXPLAIN_ACK_LINE.format(name=spoken),
                         speak=True, ack=True, done=False,
                         status=f"Explaining {path.name}…")


def _fresh_document(c):
    last = getattr(c, "_last_document", None)
    if not last:
        return None
    path, when = last
    if datetime.now().timestamp() - float(when) > LAST_DOCUMENT_S:
        return None
    return path


def _h_read_aloud(c, t, m):
    reader = c._svc("reader")
    kind, arg = m
    raw = getattr(c, "_raw_text", "") or t
    if kind == "selection" and _READ_PRONOUN_RX.match(raw.strip()) \
            and _fresh_document(c) is not None:
        kind = "document"
    if kind == "document":
        path = _fresh_document(c)
        if path is None:
            return CommandResult(handled=True, reply=NO_DOCUMENT_LINE, speak=True,
                                 status="No document")
        res = reader.read_document(path)
    elif kind == "clipboard":
        res = reader.read_clipboard()
    elif kind == "selection":
        res = reader.read_selection()
    elif kind == "file":
        res = reader.read_file(arg)
    else:
        res = reader.read_text(arg, label="that")
    if res.ok:
        return CommandResult(handled=True, status=res.message)
    return CommandResult(handled=True, reply=res.message, speak=True,
                         status="Nothing to read")


def _h_continue(c, t, m):
    reader = c._svc("reader")
    if reader is None:
        return None
    if not reader.pending_chunks and getattr(reader, "paused", False) is not True:
        return None                     # not a reading session: fall through
    res = reader.continue_reading()
    if res.ok:
        return CommandResult(handled=True, status=res.message)
    return CommandResult(handled=True, reply=res.message, speak=True,
                         status="Reading finished")


# ---- Tier 1 read-aloud steering: skip / back / pause / go on -------------
# "pause", "skip" and "back" are ALSO Spotify transport words (the router's
# spotify_control tool) and "pause"/"play" are desktop media keys in
# ACTION_COMMANDS. Ownership is decided by context, not by phrasing: while
# the reader is active (a part is being spoken, or it is paused) the verbs
# steer the reading; otherwise they fall through untouched and mean what
# they always meant. "quiet" ends the reading and hands the words back.
_READ_CTL_RX = re.compile(
    r"^" + _JV + r"(?:"
    r"(?P<skip>skip(?:\s+(?:that|this|it|ahead|forward|on|that bit|this bit|"
    r"that part|this part))?)"
    r"|(?P<back>(?:go\s+)?back(?:\s+(?:one|up|a bit|a little))?|"
    r"(?:say|read)\s+(?:that|the last (?:bit|part|one))\s+again|"
    r"previous(?:\s+(?:bit|part|one))?|what was that)"
    r"|(?P<pause>pause(?:\s+(?:that|it|there|reading|the reading))?|"
    r"hold(?:\s+(?:on|it|there|that thought))?|hang on|wait(?:\s+(?:a (?:sec|second|moment|minute)))?|"
    r"one (?:sec|second|moment|minute))"
    r"|(?P<resume>go on|carry on|continue(?:\s+reading)?|resume(?:\s+reading)?|"
    r"keep going|unpause|where were we|as you were)"
    r")(?:[,]?\s*jarvis)?[?.!\s]*$", re.I)


def read_control_kind(text: str) -> Optional[str]:
    """'skip' | 'back' | 'pause' | 'resume' | None."""
    m = _READ_CTL_RX.match((text or "").strip())
    if not m:
        return None
    return next(k for k in ("skip", "back", "pause", "resume") if m.group(k))


def _h_read_control(c, t, m):
    """Steer an active reading. Returns None (fall through) when nothing is
    being read, so Spotify / the media keys / the brain get the word."""
    reader = c._svc("reader")
    # `is True`, not truthiness: the reader must SAY it is active
    if reader is None or getattr(reader, "active", False) is not True:
        return None
    kind = m if isinstance(m, str) else read_control_kind(t)
    try:
        if kind == "skip":
            res = reader.skip()
        elif kind == "back":
            res = reader.back()
        elif kind == "pause":
            res = reader.pause()
        else:
            res = reader.resume()
            if res is None:             # not paused: "go on" = the next part
                return _h_continue(c, t, m)
    except Exception:
        log.exception("reader %s failed", kind)
        return CommandResult(handled=True, reply="I couldn't manage that, sir.",
                             speak=True, status=f"Read {kind} failed")
    if not res.ok:
        return CommandResult(handled=True, reply=res.message, speak=True,
                             status=f"Read {kind}: nothing")
    # "Paused, sir." is spoken (the silence needs an owner); skip, back and
    # go on are confirmed by the reading itself carrying on.
    spoken = kind == "pause"
    return CommandResult(handled=True, reply=res.message, speak=spoken,
                         status=res.message if not spoken else "Reading paused")


# ------------------------------------------------------------------
# Registry handlers — each ports one branch of _check_quick_command
# (voice_input_gui.py 3036-3485). Order in REGISTRY preserves the
# monolith's branch order exactly (precedence is load-bearing).
# Handlers return None to fall through, exactly where the monolith's
# branch could fall through.
# ------------------------------------------------------------------

def _h_go_back(c, t, m):                                   # 3048-3057
    prev = c._svc("context").get_last_window()
    if not prev:
        return None
    log.info("Go back to: %s", prev)
    desktop = c._svc("desktop")
    c._bg(lambda: desktop.execute_actions([("window", prev)]))
    return CommandResult(handled=True, status=f"Back to {prev}")


def _h_click_on(c, t, m):                                  # 3059-3069
    target = m.group(1).strip()
    log.info("Click on text: %s", target)
    bus.publish(Status(text=f"Finding '{target}'", kind="busy"))
    ctx = c._svc("context")
    c._bg(lambda: ctx.click_on_text(target))
    return CommandResult(handled=True, status=f"Finding '{target}'", done=False)


def _h_describe_screen(c, t, m):                           # 3071-3080
    info = c._svc("context").analyze_screen()
    if info:
        c._speak(f"You are currently in {info['active_window']}.")
        return CommandResult(handled=True,
                             reply=f"Active: {info['active_window']}")
    return CommandResult(handled=True, status="Screen analysis unavailable")


def _m_autonomous(t):
    # V3 spec wiring: "deploy" / "autonomous:" phrases → brain.execute_autonomous
    if t == "deploy" or t.startswith("autonomous:") or \
            t.startswith("autonomously "):
        return t
    return None


def _h_autonomous(c, t, m):
    task = t
    if task.startswith("autonomous:"):
        task = task[len("autonomous:"):].strip()
    elif task.startswith("autonomously "):
        task = task[len("autonomously "):].strip()
    log.info("Autonomous task: %s", task)
    c._svc("brain").execute_autonomous(task)
    return CommandResult(handled=True, status=f"Autonomous: {task[:40]}",
                         done=False)


def _h_workflow(c, t, m):                                  # 3082-3092
    workflows = c._svc("workflows")
    workflow = workflows.get(t)
    if not workflow:
        return None
    log.info("Workflow: %s (%d steps)", t, len(workflow))
    bus.publish(Status(text=t, kind="busy"))
    c._bg(lambda: workflows.run(t))
    return CommandResult(handled=True, status=f"Workflow: {t}", done=False)


def _h_suggest(c, t, m):                                   # 3094-3107
    suggestion = c._svc("memory").suggest_by_habit()
    if suggestion:
        msg = (f"Based on your habits, you usually run '{suggestion}' "
               f"around this time.")
        return CommandResult(handled=True, reply=msg, speak=_talkback())
    msg = ("I don't have enough data yet to make suggestions. Keep using "
           "voice commands and I'll learn your patterns.")
    return CommandResult(handled=True, reply=msg)


def _raw_command(c, t):
    """The utterance with its original casing (Whisper capitalises names;
    the matchers only ever see the lowercased form), minus any "jarvis"
    prefix. Falls back to the lowercased text."""
    raw = (getattr(c, "_raw_text", "") or "").strip()
    if not raw:
        return t
    lower = raw.lower().rstrip(".")
    for prefix in JARVIS_PREFIXES:
        if lower.startswith(prefix):
            return raw[len(prefix):].strip()
    return raw


def _person_line(person) -> str:
    """'Your advisor is Dr Peyrovi, sir; hp@tamu.edu.'"""
    who = person.get("name", "")
    alias = person.get("alias") or person.get("relation") or "contact"
    line = f"Your {alias} is {who}, sir"
    if person.get("email"):
        line += f"; {person['email']}"
    return line + "."


def _h_add_person(c, t, m):
    """'my advisor is Dr Peyrovi, email hp@tamu.edu' -> the people book.
    None (fall through) when the sentence is not about a person, so "my
    favourite colour is blue" still reaches the model."""
    person = parse_person_statement(_raw_command(c, t))
    if person is None:
        return None
    memory = c._svc("memory")
    if not hasattr(memory, "add_person"):
        return None
    stored = memory.add_person(person["alias"], person["name"],
                               email=person["email"])
    if not stored:
        return None
    line = (f"Noted, sir: your {stored['alias']} is {stored['name']}"
            + (f", {stored['email']}" if stored.get("email") else "") + ".")
    c._speak(line)
    return CommandResult(handled=True, reply=line)


def _h_who_is(c, t, m):
    """'who's my advisor' from the people book; None when unknown so the
    model can answer from the facts it was shown."""
    memory = c._svc("memory")
    resolve = getattr(memory, "resolve_person", None)
    if not callable(resolve):
        return None
    person = resolve(m.group(1).strip())
    if not person:
        return None
    line = _person_line(person)
    c._speak(line)
    return CommandResult(handled=True, reply=line)


def _h_remember(c, t, m):                                  # 3109-3118
    note = m.group(1).strip()
    # PERSISTENT store (jarvis.memory) — fixes the monolith's data loss.
    # Key on the note text (monolith reused one "user_note" key, which
    # silently overwrote every previous note). remember() also writes the
    # semantic index, so a paraphrase finds it later.
    memory = c._svc("memory")
    memory.remember(note[:60], note)
    # "remember that my advisor is Dr X" is a fact AND a contact (parsed
    # from the raw casing: the title and the capital are the evidence).
    rm = re.match(r"^remember (?:that )?(.+)$", _raw_command(c, t), re.I)
    person = parse_person_statement(rm.group(1) if rm else note)
    if person is not None and hasattr(memory, "add_person"):
        memory.add_person(person["alias"], person["name"], email=person["email"])
    c._speak("Noted. I'll remember that.")
    return CommandResult(handled=True, reply=f"Remembered: {note}")


def _h_recall(c, t, m):                                    # 3120-3133
    query, since = parse_since(m.group(1).strip())
    memory = c._svc("memory")
    results = memory.recall(query, since=since) if since is not None \
        else memory.recall(query)
    if results:
        text = "\n".join(f"- {r['value']}" for r in results[:3])
        c._speak(f"I recall: {results[0]['value']}")
        return CommandResult(handled=True, reply=f"I recall:\n{text}")
    when = " in that time" if since is not None else ""
    line = f"I don't have anything stored about that{when}, sir."
    c._speak(line)
    return CommandResult(handled=True, reply=line)


# "recap my day", "what was I doing before lunch", "what did I get done this
# morning": the journal tool, pinned so the router never sends a recap to
# Claude or the classifier calls it background chat.
_RECAP_RX = re.compile(
    r"^(?:(?:give me a |a )?(?:recap|summary|rundown|review) (?:of )?(?:my |the )?"
    r"(?:day|morning|afternoon|evening|week|last \w+ hours?)\b|"
    r"recap (?:my |the )?(?:day|morning|afternoon|evening|week)\b|"
    r"what (?:was|have|had) i (?:been )?(?:doing|working on|up to|done)\b|"
    # "what did I do today/yesterday" belongs to the git standup (repos,
    # commits, sessions); the journal answers the within-day windows.
    r"what did i (?:do|get done|work on|accomplish)\b(?=.*\b(?:morning|afternoon|evening|before|last|lunch)\b))", re.I)


def _h_recap(c, t, m):
    brain = c._svc("brain")
    if brain is None or not hasattr(brain, "chat"):
        return None
    brain.chat(t, force_tool="recap_day", force_args={"when": t})
    return CommandResult(handled=True, status="Looking back…", done=False)


def _h_windows(c, t, m):                                   # 3135-3145
    windows = c._svc("desktop").get_window_list()
    names = [n for _, n in windows[:10]]
    text = "\n".join(f"- {n}" for n in names)
    c._speak(f"You have {len(names)} windows open. {', '.join(names[:4])}")
    return CommandResult(handled=True, reply=f"Open windows:\n{text}")


def _h_launch(c, t, m):                                    # 3147-3170
    app = m.group(1).strip()
    log.info("Launching: %s", app)
    try:
        subprocess.Popen(
            [app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        c._speak(f"Launching {app}.")
        return CommandResult(handled=True, reply=f"Launched {app}")
    except FileNotFoundError:
        # Try xdg-open for .desktop apps
        try:
            subprocess.Popen(
                ["gtk-launch", app],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return CommandResult(handled=True, reply=f"Launched {app}")
        except Exception:
            log.exception("gtk-launch failed for %r", app)
            return CommandResult(handled=True, reply=f"Could not find '{app}'")


def _h_type(c, t, m):                                      # 3172-3183
    to_type = m.group(1).strip()
    c._type_raw(to_type)
    return CommandResult(handled=True, reply=f"Typed: {to_type}")


def _h_clipboard(c, t, m):                                 # 3185-3200
    try:
        r = subprocess.run(
            ["xclip", "-selection", "clipboard", "-o"],
            capture_output=True, text=True, timeout=2,
        )
        clip = r.stdout.strip()[:200]
        if clip:
            c._speak(f"Your clipboard contains: {clip[:100]}")
        return CommandResult(handled=True, reply=f"Clipboard: {clip}")
    except Exception:
        log.exception("clipboard read failed")
        return CommandResult(handled=True, reply="Could not read clipboard")


# ---- Tier 1 clipboard-to-Claude -------------------------------------
# "have Claude fix what I copied" / "send the clipboard to Claude" / "ask
# Claude about the selection": the clip (X clipboard or primary selection)
# plus the spoken instruction becomes the next task in the active project.
# The clip goes by FILE, never inline: sanitize_keys collapses a prompt to one
# line before `send-keys -l` (a traceback would be mangled), and the prompt
# is echoed to task_dir/<id>.prompt, the bus and Discord -- a pasted secret
# would leak through every one of those.  The file sits under the project
# (<proj>/.jarvis/clips/, self-ignored) so the interactive Claude can read it
# without a permission prompt: task_dir is outside claude.allowed_dirs.
_CLIP_PHRASE = (r"what i (?:just )?copied|(?:the |my )?clipboard(?: contents?| text)?|"
                r"what(?:'s| is) (?:on|in) (?:the |my )?clipboard|"
                r"the copied (?:text|bit|error|traceback|snippet|thing)")
# The noun forms want an article: a bare "highlight" / "selection" is more
# often a verb or a code noun ("change the selection logic") than the X
# primary selection, hence the article and the lookahead below.
_SEL_PHRASE = (r"(?:the |my )(?:selection|selected text|highlighted(?: text| bit| part| error)?|"
               r"highlight)|what i (?:just )?(?:selected|highlighted)|this selection")
_NOT_A_CLIP = (r"(?!\s+(?:logic|code|handler|function|class|module|widget|menu|list|"
               r"history|manager|api|tool|box))")
_CLIP_TO_CLAUDE_RX = re.compile(
    r"^(?:please\s+)?(?:"
    r"(?:have|ask|tell|get|let|make)\s+claude(?:\s+to)?\s+(?P<instr>.*?)\s*"
    r"(?:(?P<clip>" + _CLIP_PHRASE + r")|(?P<sel>" + _SEL_PHRASE + r"))\b" + _NOT_A_CLIP +
    r"(?P<tail>.*?)"
    r"|(?:send|give|hand|pass|paste|forward|show)\s+(?:claude\s+)?"
    r"(?:(?P<clip2>" + _CLIP_PHRASE + r")|(?P<sel2>" + _SEL_PHRASE + r"))\b" + _NOT_A_CLIP +
    r"(?:\s+(?:over\s+)?to\s+claude)?(?:\s*[,;]?\s*(?:and\s+)?(?P<instr2>.+?))?"
    r")[.!?\s]*$", re.I)
# "ask claude ABOUT the clipboard": a lone preposition is no instruction.  A
# trailing one is kept -- "look at" / "deal with" are verb phrases.
_CLIP_PREP_RX = re.compile(r"^(?:about|at|with|on|to|over|through|into|regarding)$", re.I)
CLIP_EMPTY_LINE = "The clipboard is empty, sir."
SELECTION_EMPTY_LINE = "Nothing is highlighted, sir."
CLIP_DEFAULT_INSTRUCTION = "Have a look at"


def _clip_instruction(m) -> tuple[str, str, str]:
    """(instruction, selection, tail) from a _CLIP_TO_CLAUDE_RX match."""
    if m.group("clip") or m.group("sel"):
        instr, tail = m.group("instr") or "", m.group("tail") or ""
        selection = "clipboard" if m.group("clip") else "primary"
    else:
        instr, tail = m.group("instr2") or "", ""
        selection = "clipboard" if m.group("clip2") else "primary"
    instr = instr.strip(" ,;")
    if _CLIP_PREP_RX.match(instr):
        instr = ""
    tail = tail.strip(" ,;.")            # "and explain it" keeps its "and"
    return instr, selection, tail


def _write_clip(project_path: str, text: str) -> Path:
    """Write the clip 0600 under <project>/.jarvis/clips/ (dir 0700, with a
    .gitignore so it never shows up in the user's `git status`)."""
    clips = Path(project_path) / ".jarvis" / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(clips, 0o700)
        ignore = clips.parent / ".gitignore"
        if not ignore.exists():
            ignore.write_text("*\n", encoding="utf-8")
    except OSError:
        log.debug("clip dir hygiene failed", exc_info=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = clips / f"{stamp}.txt"
    n = 0
    while True:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            n += 1
            path = clips / f"{stamp}-{n}.txt"
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    return path


def _h_clip_to_claude(c, t, m):
    claude = c._svc("claude")
    if claude is None:
        return None
    # Re-match the RAW text: the Tier-1 text is lowercased, and Claude should
    # see the instruction as spoken (class and file names keep their case).
    raw = strip_address(getattr(c, "_raw_text", "") or t)
    mr = _CLIP_TO_CLAUDE_RX.match(raw.strip())
    instr, selection, tail = _clip_instruction(mr or m)
    text = reader_mod._xclip(selection)
    if not text.strip():
        line = CLIP_EMPTY_LINE if selection == "clipboard" else SELECTION_EMPTY_LINE
        return CommandResult(handled=True, reply=line, speak=True, status="Clipboard empty")
    active = None
    try:
        active = getattr(claude, "active_project", None)
        proj = claude.project_for(active) if active else None
    except Exception:
        log.exception("claude.project_for(%r) failed", active)
        proj = None
    if proj is None or not getattr(proj, "path", ""):
        cs_mod = sys.modules.get("jarvis.claude_session")
        line = getattr(cs_mod, "NO_PROJECT_LINE", None) or \
            "I don't have a project to work in, sir; name one first."
        return CommandResult(handled=True, reply=line, speak=True, status="No project")
    try:
        path = _write_clip(proj.path, text)
    except OSError:
        log.exception("clip write failed under %s", proj.path)
        return CommandResult(handled=True, speak=True, status="Clip write failed",
                             reply="I couldn't put the clip where Claude can read it, sir.")
    ref = f"the text in {path}"
    body = f"{instr} {ref}" if instr else f"{CLIP_DEFAULT_INSTRUCTION} {ref}"
    if tail:
        body += f" {tail}"
    what = "copied" if selection == "clipboard" else "highlighted"
    prompt = f"{body}. That file holds what I just {what} on my screen; read it first."
    log.info("clip -> Claude: %d chars in %s", len(text), path)
    d = RouteDecision("claude", "clip-to-claude", prompt=prompt, project=active,
                      args={"size": estimate_size(prompt)})
    return c._dispatch_route(d, t)


def _h_search(c, t, m):                                    # 3202-3215
    # With a brain that can answer from the web, "jarvis, look up X" is a
    # question, not a request to open a browser tab: yield to the router.
    brain = c._svc("brain")
    if brain is not None and hasattr(brain, "web_answer") and c._svc("router") is not None:
        return None
    query = m.group(1).strip()
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    subprocess.Popen(
        ["xdg-open", url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    c._speak(f"Searching for {query}.")
    return CommandResult(handled=True, reply=f"Searching: {query}")


# ---- Tier 1 assistant: timekeeper, notes, briefing -----------------------
# Instant, regex-matched forms of the assistant tools (spec 5.2 "registry
# re-points"). Anything these cannot parse falls through to the router, where
# the local model reaches the same tools with its own parser.
_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "twenty five": 25,
    "twenty-five": 25, "thirty": 30, "forty": 40, "forty five": 45,
    "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90,
    "a couple of": 2, "a few": 3,
}
_NUM_ALT = r"\d+|" + "|".join(re.escape(w) for w in
                              sorted(_NUM_WORDS, key=len, reverse=True))
_UNIT_ALT = r"minutes?|mins?|seconds?|secs?|hours?|hrs?"
_TIMER_RX = re.compile(
    r"^(?:(?:set|start|put on|run|create|make|give me)\s+(?:a\s+|an\s+|the\s+)?)?"
    r"(?:(?P<n1>" + _NUM_ALT + r")\s*[- ]?(?P<u1>" + _UNIT_ALT + r")\s+timer"
    r"|timer\s+(?:for\s+)?(?P<n2>" + _NUM_ALT + r")\s*(?P<u2>" + _UNIT_ALT + r"))"
    r"(?:\s+(?:for|to|called|named|labell?ed)\s+(?P<label>.+?))?[.!]*$", re.I)
_DAYS = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_CLOCK_T = r"\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|o'?clock)?"
_WHEN_RX = re.compile(
    r"(?:^|(?<=\s))(?P<when>"
    r"in\s+(?:" + _NUM_ALT + r"|half an?)\s*(?:and a half\s+)?(?:" + _UNIT_ALT +
    r"|days?|weeks?)(?:\s+and\s+(?:a\s+)?half)?"
    r"|in an hour(?: and a half)?|in half an hour"
    r"|(?:tomorrow|tonight|today|this (?:morning|afternoon|evening)|"
    r"tomorrow (?:morning|afternoon|evening|night)|"
    r"(?:on|next|this)\s+(?:" + _DAYS + r"|week|month)(?:\s+(?:morning|afternoon|evening|night))?|"
    r"every (?:day|weekday|morning|evening|" + _DAYS + r"))"
    r"(?:\s+at\s+(?:" + _CLOCK_T + r"|noon|midnight|midday))?"
    r"|at\s+(?:" + _CLOCK_T + r"|noon|midnight|midday)"
    r"(?:\s+(?:tomorrow|tonight|today|tomorrow (?:morning|evening)|on\s+(?:" + _DAYS + r")))?"
    r")(?=\s|$|[,.!])", re.I)
_REMIND_RX = re.compile(r"^remind me\b\s*(?P<body>.+)$", re.I)
_LEGACY_IN_RX = re.compile(
    r"^in\s+(?P<n>" + _NUM_ALT + r")\s*(?:and a half\s+)?(?P<u>" + _UNIT_ALT +
    r")(?:\s+and\s+(?:a\s+)?half)?$", re.I)
_ALARM_RX = re.compile(
    r"^(?:(?:please\s+)?wake me(?:\s+up)?(?:\s+(?P<w1>.+?))?"
    r"|(?:set|create|make|put on|add)\s+(?:an?\s+|the\s+)?alarm(?:\s+(?:for|at)\s+(?P<w2>.+?))?"
    r"|alarm\s+(?:for|at)\s+(?P<w3>.+?))"
    r"(?:\s+(?:called|named|labell?ed|so (?:that )?i can|to)\s+(?P<label>.+?))?[.!]*$",
    re.I)
_ALARM_REPEAT_RX = re.compile(r"\b(?:every\s+(?P<r>day|morning|weekday|week ?day)|daily|"
                        r"(?:on\s+)?weekdays)\b", re.I)
_LIST_SCHED_RX = re.compile(
    r"^(?:(?:what|which|any|list|show|read|tell me|check|do i have any|"
    r"have i got any|are there any|is there an?)\b.*?\b(?P<k1>reminders?|timers?|alarms?)\b.*"
    r"|(?:my\s+)?(?P<k2>reminders?|timers?|alarms?)(?:\s+(?:list|status))?"
    r"|what(?:'s| is|s)? (?:set|scheduled|pending))[?.!]*$", re.I)
_CANCEL_SCHED_RX = re.compile(
    r"^(?:cancel|clear|delete|remove|kill|drop|scrap|stop)\s+"
    r"(?P<all>all\s+(?:of\s+)?(?:my\s+|the\s+)?|every\s+)?(?:the\s+|my\s+|that\s+|this\s+)?"
    r"(?P<kind>reminders?|timers?|alarms?)"
    r"(?:\s+(?:for|about|to|called|named)\s+(?P<which>.+?))?[.!]*$", re.I)
_NOTE_RX = re.compile(
    r"^(?:take a note|make a note|note that|note down|jot down|write (?:this |that )?down|"
    r"note)[:,]?\s+(?:that\s+)?(?P<text>.+?)[.!]*$", re.I)
_TODO_ADD_RX = re.compile(
    r"^(?:(?:add|put)\s+(?P<t1>.+?)\s+(?:to|on)\s+(?:my\s+)?(?:to-?do|todo|task|shopping)?\s*list"
    r"|(?:add|create|new)\s+(?:a\s+)?(?:to-?do|todo|task)(?:\s*:|\s+to|\s+for)?\s+(?P<t2>.+?)"
    r"|to-?do:?\s+(?P<t3>.+?)|i need to\s+(?P<t4>.+?)(?:,? add (?:it|that) to (?:my|the) list))"
    r"[.!]*$", re.I)
_TODO_LIST_RX = re.compile(
    r"^(?:(?:show|read|list|what(?:'s| is|s)? (?:on|in)|open|check)\s+(?:me\s+)?(?:my\s+)?"
    r"(?:to-?do|todo|task)s?(?:\s+list)?|my (?:to-?do|todo)s?(?: list)?|"
    r"what do i (?:have|need) to do(?: today)?|what'?s? (?:left|outstanding))[?.!]*$", re.I)
_TODO_DONE_RX = re.compile(
    r"^(?:(?:mark|tick|check|cross)\s+(?:off\s+)?(?P<w1>.+?)\s+(?:as\s+)?(?:done|off|complete|completed|finished)"
    r"|(?:tick|check|cross)\s+off\s+(?P<w2>.+?)|(?:done with|finished|i did|i've done)\s+(?P<w3>.+?)"
    r"|(?:that's|thats|it's|its)\s+done)[.!]*$", re.I)
# ---- Named lists (spec 14) ---------------------------------------------
# "add milk to the shopping list" used to land in the generic to-do list:
# _TODO_ADD_RX swallows every "... to my <word> list" phrasing (its
# alternation even names "shopping"). These run BEFORE the to-do commands
# and hand back None when the name IS a to-do word, so "add buy milk to my
# task list" still reaches _h_todo_add. The name is required: "add milk to
# my list" has no name and stays the to-do list.
_LIST_NAME = r"(?P<name>[a-z0-9][a-z0-9'\- ]{0,39}?)"
_LIST_ADD_RX = re.compile(
    r"^(?:add|put|stick|throw|chuck)\s+(?P<item>.+?)\s+(?:to|on|onto|in)\s+"
    r"(?:my|the|our)\s+" + _LIST_NAME + r"\s+list[.!]*$", re.I)
_LIST_READ_RX = re.compile(
    r"^(?:(?:read|show|list|check|open|give|tell)\s+(?:me\s+)?(?:out\s+)?"
    r"|what(?:'s|s| is| are)\s+(?:on|in)\s+|what(?:'s|s| is)\s+)"
    r"(?:my|the|our)\s+" + _LIST_NAME + r"\s+list[?.!]*$", re.I)
_LIST_STRIKE_RX = re.compile(
    r"^(?:take|cross|scratch|strike|knock|tick|check|rub)\s+(?:off\s+)?"
    r"(?P<item>.+?)\s+(?:off(?:\s+of)?|from|out of)\s+(?:my|the|our)\s+"
    + _LIST_NAME + r"\s+list[.!]*$", re.I)
_LIST_CLEAR_RX = re.compile(
    r"^(?:clear|empty|wipe|reset|erase|bin|delete)\s+(?:out\s+)?"
    r"(?:everything\s+(?:off|from)\s+)?(?:my|the|our)\s+"
    + _LIST_NAME + r"\s+list[.!]*$", re.I)
_LISTS_RX = re.compile(
    r"^(?:(?:what|which)\s+lists\s+(?:do i have|have i got|are there|"
    r"do you have|do i keep)|(?:show|read|list|name)\s+(?:me\s+)?(?:my|the)\s+lists"
    r"|what are (?:my|the) lists)[?.!]*$", re.I)
# "milk, eggs and bread" is three items; "pick up the dry cleaning and post
# the forms" is ONE errand. Only short, list-shaped text is split.
_ITEM_SPLIT_RX = re.compile(r"\s*,\s*|\s+and\s+", re.I)
LIST_SPLIT_CHARS = 60
LIST_SPOKEN_LIMIT = 10          # == NotesStore.resolve's ordinal window


def _split_items(text: str) -> list:
    text = " ".join(str(text or "").split()).strip(" .,")
    if not text:
        return []
    if len(text) > LIST_SPLIT_CHARS:
        return [text]
    parts = [p.strip(" .,") for p in _ITEM_SPLIT_RX.split(text)]
    parts = [p for p in parts if p]
    if len(parts) > 1 and all(len(p.split()) <= 3 for p in parts):
        return parts
    return [text]


def _list_target(c, m, create: bool = False):
    """``(store, name, kind)`` for a named-list command.

    ``kind`` is None when the spoken name belongs to the built-in notes or
    to-dos (the handler returns None so the to-do commands get their turn)
    and "" when the list simply does not exist yet.
    """
    store = c._svc("notes")
    spoken = (m.group("name") or "").strip()
    if store is None or not notes_mod.list_kind(spoken):
        return None, notes_mod.canon_list(spoken), None
    name = store.make_list(spoken) if create else store.find_list(spoken)
    if not isinstance(name, str) or not name:      # duck-typed / stub store
        return store, notes_mod.canon_list(spoken), ""
    return store, name, notes_mod.list_kind(name)


def _no_such_list(name: str) -> CommandResult:
    return CommandResult(handled=True, reply=f"You haven't a {name} list, sir.",
                         speak=True, status="No such list")


def _h_list_add(c, t, m):
    # Original casing from the raw utterance ("add Waitrose coffee ...").
    raw = getattr(c, "_raw_text", "") or ""
    m2 = _LIST_ADD_RX.match(strip_jarvis_prefix(raw) or raw.strip()) if raw else None
    mm = m2 or m
    store, name, kind = _list_target(c, mm, create=True)
    if kind is None:
        return None
    items = _split_items(mm.group("item"))
    if not items:
        return None
    ids = [store.add(kind, i) for i in items]
    line = f"Added to your {name} list, sir." if len(items) == 1 else \
        f"{number_word(len(items)).capitalize()} added to your {name} list, sir."
    return CommandResult(handled=True, reply=line, speak=True,
                         status=f"{name}: {', '.join(items)[:40]}",
                         undo=_undo_notes(store, kind, ids,
                                          f"Off the {name} list again, sir."))


def _h_list_read(c, t, m):
    store, name, kind = _list_target(c, m)
    if kind is None:
        return None
    if not kind:
        return _no_such_list(name)
    return CommandResult(handled=True,
                         reply=store.list_text(kind, LIST_SPOKEN_LIMIT),
                         speak=True, status=f"{name} list")


def _h_list_strike(c, t, m):
    store, name, kind = _list_target(c, m)
    if kind is None:
        return None
    if not kind:
        return _no_such_list(name)
    which = (m.group("item") or "").strip(" .")
    removed = store.remove(kind, which)
    if not removed:
        return CommandResult(
            handled=True, status="Not on the list", speak=True,
            reply=f"I couldn't find that on your {name} list, sir.")
    left = store.count(kind)
    line = f"Off the {name} list, sir; {number_word(left)} left." if left else \
        f"Off the {name} list, sir; that's it clear."
    return CommandResult(handled=True, reply=line, speak=True,
                         status=f"{name}: struck {removed[0]['text'][:30]}",
                         undo=_undo_restore(store, kind, removed,
                                            f"Back on the {name} list, sir."))


def _do_list_clear(store, kind: str, name: str) -> CommandResult:
    rows = store.remove(kind, "all")
    n = len(rows)
    back = f"All {number_word(n)} back on the {name} list, sir." if n != 1 else \
        f"Back on the {name} list, sir."
    return CommandResult(handled=True, speak=True,
                         reply=f"The {name} list is clear, sir.",
                         status=f"Cleared {n} from the {name} list",
                         undo=_undo_restore(store, kind, rows, back))


def _h_list_clear(c, t, m):
    store, name, kind = _list_target(c, m)
    if kind is None:
        return None
    if not kind:
        return _no_such_list(name)
    n = store.count(kind)
    if not n:
        return CommandResult(handled=True, status="Empty", speak=True,
                             reply=f"Your {name} list is empty already, sir.")
    # The same read-back a bulk cancel gets: a wipe is worth one second of
    # "yes", and the undo below is only as good as the words he heard.
    if _wants_read_back(c, n):
        line = f"Clear all {number_word(n)} off your {name} list, sir?" \
            if n > 1 else f"Clear the one thing off your {name} list, sir?"
        c.stash_destructive(lambda: _do_list_clear(store, kind, name), line)
        return CommandResult(handled=True, reply=line, speak=True,
                             status="Confirm?")
    return _do_list_clear(store, kind, name)


def _h_lists(c, t, m):
    store = c._svc("notes")
    if store is None or not hasattr(store, "lists_text"):
        return None
    return CommandResult(handled=True, reply=store.lists_text(), speak=True,
                         status="Lists")


_BRIEFING_RX = re.compile(
    r"^(?:good morning(?:[, ]+jarvis)?|(?:morning|daily|my|the) briefing|briefing|"
    r"what'?s my briefing|(?:give me|read me|run|do) (?:the|my) (?:morning |daily )?briefing|"
    r"brief me)[.!?]*$", re.I)
# The evening preview (get_briefing when=tomorrow). "what's on tomorrow" is
# deliberately absent: that is a calendar question and stays with
# get_calendar; the preview is the whole of tomorrow in one breath.
_PREVIEW_RX = re.compile(
    r"^(?:(?:what(?:'s| does| is)|how(?:'s| does| is)) tomorrow (?:look(?:ing)?|shaping up)"
    r"(?: like)?|(?:preview|brief me on|brief me for) tomorrow|tomorrow'?s? (?:briefing|preview)|"
    r"(?:give me|read me|run|do) (?:the |my )?(?:tomorrow|evening|night(?:ly)?) (?:briefing|preview)|"
    r"(?:the |my )?(?:evening|night(?:ly)?) (?:briefing|preview))"
    r"(?:[, ]+(?:please|jarvis|sir))*[?.!]*$", re.I)
# The week forecast (get_briefing when=week). "what does my week look like"
# is left to get_calendar (docs, Calendars row); these are the workload
# phrasings a student actually uses on a Sunday evening.
_WEEK_RX = re.compile(
    r"^(?:how(?:'s| is| does| are) (?:my|the) (?:week|next seven days|next 7 days) "
    r"(?:look(?:ing)?|shaping up)(?: like)?|what(?:'s| is) (?:my|the) week looking like|"
    r"what(?:'s| is) (?:the |my )?(?:week|workload|week's workload) (?:ahead|looking like)|"
    r"(?:the |my )?week(?:ly)? (?:briefing|forecast|preview|overview|workload)|"
    r"(?:brief me on|preview|forecast) (?:the|my) week|how (?:heavy|busy|bad) is my week|"
    r"what(?:'s| is) (?:the |my )?week ahead like)"
    r"(?:[, ]+(?:please|jarvis|sir))*[?.!]*$", re.I)
# Explicit preferences (33): "no news in the morning", "put the sports back
# in my briefing". Persisted to assistant.json (briefing.sections.<name>)
# AND memory.set_preference, so the file is the store consumers read and
# the memory keeps the audit trail of what he asked for.
_PREF_SECTIONS = {
    "news": "news", "sport": "sports", "sports": "sports", "stock": "stocks",
    "stocks": "stocks", "weather": "weather", "calendar": "calendar",
    "canvas": "canvas", "coursework": "canvas", "todo": "todos", "todos": "todos",
    "to-do": "todos", "to-dos": "todos", "tasks": "todos", "alarm": "alarms",
    "alarms": "alarms", "reminder": "reminders", "reminders": "reminders",
}
_PREF_SECTION_WORDS = {"todos": "the to-dos", "alarms": "the alarms",
                       "reminders": "the reminders", "canvas": "Canvas",
                       "weather": "the weather", "calendar": "the calendar"}
_PREF_SECTION_RX = re.compile(
    r"^(?:(?P<off>no|skip|drop|leave out|lose|without|i don'?t want|i do not want|"
    r"don'?t (?:read|include|give me|do|mention)|stop (?:reading|including|giving me))"
    r"|(?P<on>include|add|put|bring back|read|give me|i want|i'?d like|"
    r"start (?:reading|including)|mention))"
    r"\s+(?:the\s+|my\s+|any\s+)?"
    r"(?P<section>news|sports?|stocks?|weather|calendar|canvas|coursework|to-?dos?|tasks|"
    r"alarms?|reminders?)"
    r"(?:\s+(?:back|again))?"
    r"\s+(?:(?:in|from|with|on)\s+(?:the|my)\s+(?:morning\s+|daily\s+|evening\s+|nightly\s+|"
    r"weekly\s+)?(?:briefings?|previews?|forecast)|in the mornings?)"
    r"(?:\s+(?:back|again))?"
    r"(?:[, ]+(?:please|jarvis|sir|from now on))*[.!]*$", re.I)
_PREF_VERBOSITY_RX = re.compile(
    r"^(?:(?P<brief>(?:be|keep it|make it|make them|keep them) (?:briefer|shorter|"
    r"more concise|more brief|snappier|less wordy)|"
    r"(?:shorter|briefer|snappier) (?:briefings?|answers|replies|previews?)|(?:less|fewer) words|"
    r"keep (?:the |your |my )?(?:briefings?|answers|replies) short(?:er)?|"
    r"(?:brief|short) (?:briefings?|answers|replies)(?: only)?|too (?:long|wordy)|less detail)"
    r"|(?P<full>(?:be|make it|make them) (?:more detailed|more thorough|less brief|longer|wordier)|"
    r"(?:longer|fuller|full|normal|regular|detailed) (?:briefings?|answers|replies|previews?)|"
    r"more detail(?:s)?|the full briefing))"
    r"(?:[, ]+(?:please|jarvis|sir|from now on))*[.!]*$", re.I)
PREVIEW_ASK = "good night; what does tomorrow look like?"
GOODNIGHT_PREVIEW_LINE = "Good night, sir. Tomorrow, briefly."
BRIEFER_LINE = "Briefer it is, sir."
FULL_LENGTH_LINE = "Very good, sir; the full briefing again."
NO_ALARM_LINE = "Very good, sir; no alarm."
ALARM_FAILED_LINE = "I couldn't set that alarm, sir."
# While an alarm rings (spec 5.2 a): these words stop it, "snooze [N]" snoozes.
_RING_STOP_RX = re.compile(
    r"^(?:stop|dismiss|okay|ok|i'?m up|i am up|shut it off|shut up|enough|"
    r"turn it off|alright|all right|got it|thank you|thanks|quiet|silence|"
    r"stop it|that'?s enough|cancel|off)(?:[, ]+(?:jarvis|thanks|please))*[.!]*$", re.I)
_SNOOZE_RX = re.compile(
    r"^(?:snooze|(?:five|ten|\d+) more minutes|(?:a )?(?:bit|few minutes) more)"
    r"(?:\s+(?:for\s+)?(?:(?P<n>" + _NUM_ALT + r")\s*(?:minutes?|mins?)?))?"
    r"(?:[, ]+(?:jarvis|please))*[.!]*$", re.I)
# Pending permission question (spec 5.2 b).
_YES_RX = re.compile(
    r"^(?:yes|yeah|yep|yup|aye|allow(?: it| that)?|approve(?:d| it)?|go ahead|"
    r"do it|sure|ok(?:ay)?|fine|permitted|allowed|proceed|let it|go on|"
    r"yes please|yes allow it|carry on|affirmative|by all means)"
    r"(?:[, ]+(?:sir|jarvis|please|claude))*[.!]*$", re.I)
_NO_RX = re.compile(
    r"^(?:no|nope|nah|deny|denied|decline|declined|don'?t|do not|refuse|block|"
    r"no way|absolutely not|negative|not that|don'?t allow it|deny it|"
    r"no thanks|no thank you)(?:[, ]+(?:jarvis|please|thanks))*[.!]*$", re.I)
# Claude refused an out-of-project task and offered the terminal instead
# (claude_session.OUTSIDE_LINE); "yes" / "open it" then opens the pop-out.
_OPEN_IT_RX = re.compile(
    r"^(?:open it|open the terminal|open a terminal|please do|do that|"
    r"terminal(?: then)?|yes open it)"
    r"(?:[, ]+(?:sir|jarvis|please))*[.!]*$", re.I)
TERMINAL_OPEN_LINE = "Up on screen, sir."
TERMINAL_FAIL_LINE = "I couldn't open the terminal, sir."
# Routed INTO an already-open terminal (router rule 1b): say which session
# took the words, so "in the terminal, run the tests" is never ambiguous.
TERMINAL_ROUTE_LINE = "Through to the {name} session, sir."
TERMINAL_ROUTE_LINE_ANON = "Through to the open session, sir."

# "switch to the X project" is a router action, not a window target.
_PROJECT_SWITCH_RX = re.compile(
    r"^(?:switch to|go to|open|focus)\s+(?:the\s+)?.+?\s+(?:project|repo|repository)$",
    re.I)


def _num(word) -> Optional[int]:
    w = (word or "").strip().lower()
    if w.isdigit():
        return int(w)
    return _NUM_WORDS.get(w)


def _seconds(n, unit, half=False) -> int:
    unit = (unit or "").lower()
    mult = 3600 if unit.startswith(("hour", "hr")) else \
        60 if unit.startswith("min") else 1
    return int((n + (0.5 if half else 0)) * mult)


def _unit_word(unit: str, n) -> str:
    unit = (unit or "").lower()
    base = "hour" if unit.startswith(("hour", "hr")) else \
        "minute" if unit.startswith("min") else "second"
    return base if n == 1 else base + "s"


def split_when(body: str) -> tuple:
    """('in 10 minutes', 'stretch') for 'in 10 minutes to stretch' /
    'to stretch in 10 minutes' / 'to stretch at 3 pm tomorrow'; (None,
    body) when no time phrase is found."""
    body = (body or "").strip()
    hits = list(_WHEN_RX.finditer(body))
    if not hits:
        return None, body
    # Prefer a phrase at the start or the end; merge "tomorrow ... at 8"
    # style pairs that the regex catches as one span already.
    m = hits[0] if hits[0].start() == 0 else hits[-1]
    when = m.group("when").strip()
    text = (body[:m.start()] + " " + body[m.end():]).strip(" ,.")
    text = re.sub(r"^(?:to|that|about)\s+", "", text, count=1, flags=re.I)
    text = re.sub(r"\s+(?:to|that|about)$", "", text, flags=re.I)
    return when, re.sub(r"\s+", " ", text).strip()


def _norm_when(when: str) -> str:
    """'at 7 tomorrow' -> 'tomorrow at 7'; bare '6:30' -> 'at 6:30'."""
    w = (when or "").strip()
    m = re.match(r"^at\s+(?P<t>.+?)\s+(?P<d>tomorrow(?: morning| evening)?|tonight|today|on \w+day)$",
                 w, re.I)
    if m:
        return f"{m.group('d')} at {m.group('t')}"
    if re.match(r"^\d", w):
        return "at " + w
    return w


def _epoch(value) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except Exception:
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _due_from(c, tk, when: str, now: datetime) -> Optional[float]:
    """Epoch seconds for a spoken time: 'in N units' is arithmetic here
    (no parser needed); everything else goes through timekeeper.parse_when."""
    m = _LEGACY_IN_RX.match(when or "")
    if m:
        n = _num(m.group("n"))
        if n is not None:
            half = "half" in when.lower()
            return now.timestamp() + _seconds(n, m.group("u"), half)
    if when.lower() in ("in half an hour",):
        return now.timestamp() + 1800
    if when.lower().startswith("in an hour"):
        return now.timestamp() + (5400 if "half" in when.lower() else 3600)
    parse = getattr(tk, "parse_when", None)
    if parse is None:
        try:
            from jarvis.tools.timekeeper import parse_when as parse
        except Exception:
            return None
    try:
        return _epoch(parse(_norm_when(when), now))
    except Exception:
        log.exception("parse_when failed for %r", when)
        return None


def _describe(tk, due: float, now: datetime, fallback: str) -> str:
    fn = getattr(tk, "describe_due", None)
    if fn is None:
        try:
            from jarvis.tools.timekeeper import describe_due as fn
        except Exception:
            return fallback
    try:
        out = fn(due, now)
        return str(out) if out else fallback
    except Exception:
        log.exception("describe_due failed")
        return fallback


def _assistant_get(c, key: str, default=None):
    cfg = c._svc("assistant")
    if cfg is None:
        return default
    try:
        val = cfg.get(key, default)
    except Exception:
        log.exception("assistant config read failed for %s", key)
        return default
    return default if val is None else val


# ------------------------------------------------------------------
# Spoken undo: closures for "scratch that" (2026-08-30)
# ------------------------------------------------------------------
# Each targets the ONE item the turn created, by id. "cancel the last
# timer" would be wrong: a timer set in between must not be the casualty,
# and neither must a note whose words happen to match.
def _undo_timekeeper(tk, item, kind: str, line: str):
    item_id = getattr(item, "id", None)
    if tk is None or not item_id:
        return None

    def _undo() -> str:
        n = tk.cancel(which=item_id, kind=kind)
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = 1 if n else 0
        return line if n else "That one had gone already, sir."
    return _undo


def _undo_notes(store, kind: str, ids, line: str):
    ids = [i for i in (ids if isinstance(ids, (list, tuple)) else [ids])
           if i is not None]
    if store is None or not ids or not hasattr(store, "delete"):
        return None

    def _undo() -> str:
        gone = 0
        for item_id in ids:
            try:
                if store.delete(kind, item_id):
                    gone += 1
            except Exception:
                log.exception("undo delete failed")
        return line if gone else "That one had gone already, sir."
    return _undo


def _undo_restore(store, kind: str, rows, line: str):
    """Put removed rows back, original timestamps and all -- the undo of a
    strike-off or a whole-list wipe."""
    rows = [dict(r) for r in (rows or []) if r and r.get("text")]
    if store is None or not rows:
        return None

    def _undo() -> str:
        for row in rows:
            try:
                store.add(kind, row["text"], created=row.get("created"))
            except Exception:
                log.exception("undo restore failed")
        return line
    return _undo


def _h_timer(c, t, m):                                     # 3217-3231
    n = _num(m.group("n1") or m.group("n2"))
    unit = (m.group("u1") or m.group("u2") or "minutes").lower()
    if n is None:
        return None
    seconds = _seconds(n, unit)
    label = (m.group("label") or "").strip()
    words = f"{n} {_unit_word(unit, n)}"
    tk = c._svc("timekeeper")
    if tk is not None:
        item = tk.add_timer(seconds, label or f"{words} timer")
        line = f"{words}, sir; I'll let you know."
        if label:
            what = label if re.match(r"^(?:the|my|a|an|your)\b", label, re.I) \
                else f"the {label}"
            line = f"{words} for {what}, sir; I'll let you know."
        return CommandResult(handled=True, reply=line, speak=True,
                             status=f"Timer set: {words}",
                             undo=_undo_timekeeper(tk, item, "timer",
                                                   "Timer scrapped, sir."))
    workflows = c._svc("workflows")
    if workflows is None:
        return None
    workflows.set_reminder(seconds, f"Timer for {n} {unit}")
    return CommandResult(handled=True, status=f"Timer set: {n} {unit}")


def _h_alarm(c, t, m):
    tk = c._svc("timekeeper")
    if tk is None:
        return CommandResult(handled=True, reply=TIMEKEEPER_SETUP_LINE,
                             speak=True, status="No timekeeper")
    when = (m.group("w1") or m.group("w2") or m.group("w3") or "").strip(" ,.")
    label = (m.group("label") or "").strip()
    if not when:
        return CommandResult(handled=True, reply=NO_WHEN_LINE, speak=True,
                             status="Alarm: when?")
    repeat = "once"
    rm = _ALARM_REPEAT_RX.search(when)
    if rm:
        r = (rm.group("r") or rm.group(0)).lower().replace(" ", "")
        repeat = "weekdays" if "weekday" in r else "daily"
        when = (when[:rm.start()] + " " + when[rm.end():]).strip(" ,.")
        when = re.sub(r"\s+", " ", when)
    now = datetime.now()
    due = _due_from(c, tk, when, now)
    if due is None:
        return CommandResult(handled=True, reply=NO_WHEN_LINE, speak=True,
                             status="Alarm: when?")
    item = tk.add_alarm(due, label, repeat)
    desc = _describe(tk, due, now, when)
    tail = {"daily": " Every day.", "weekdays": " Weekdays."}.get(repeat, "")
    return CommandResult(handled=True, reply=f"Alarm {desc}, sir.{tail}",
                         speak=True, status=f"Alarm {desc}",
                         undo=_undo_timekeeper(tk, item, "alarm",
                                               "Alarm cancelled, sir."))


def _h_list_schedule(c, t, m):
    tk = c._svc("timekeeper")
    if tk is None:
        return None
    word = (m.group("k1") or m.group("k2") or "all").lower()
    kind = "reminder" if word.startswith("remind") else \
        "timer" if word.startswith("timer") else \
        "alarm" if word.startswith("alarm") else "all"
    text = tk.list_text(kind)
    return CommandResult(handled=True, reply=text, speak=True,
                         status="Schedule")


# Destructive read-back (2026-08-30). A bulk cancel used to run on the first
# transcript, even one that scraped past the confidence gate; "cancel all
# alarms" now reads back "Cancel all three alarms, sir?" and waits for a
# yes. The follow-up window the app opens after any spoken reply is the
# VAD-timed yes/no: a one-second "yes" costs one second. Only whole-list
# actions with more than one item (or a shaky transcript) are read back --
# a single timer is never worth the question, and a mis-cancelled one
# costs nothing to set again.
DESTRUCTIVE_TTL_S = 60.0
_COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve")


def _count_word(n: int) -> str:
    return _COUNT_WORDS[n] if 0 <= n < len(_COUNT_WORDS) else str(n)


def _pending_count(tk, kind: str) -> int:
    """How many live items a whole-list cancel would take. 0 when the
    timekeeper cannot say (an older stand-in without list())."""
    try:
        items = tk.list(kind)
        return len(items)
    except Exception:
        return 0


def _wants_read_back(c, n: int) -> bool:
    if not _assistant_get(c, "confirm.read_back", True):
        return False
    if n > 1:
        return True
    return n == 1 and c.shaky_transcript()


def _do_cancel_schedule(tk, which: str, kind: str) -> CommandResult:
    n = tk.cancel(which or "last", kind)
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 1 if n else 0
    if n <= 0:
        return CommandResult(handled=True, reply="Nothing to cancel, sir.",
                             speak=True, status="Nothing to cancel")
    line = "Cancelled, sir." if n == 1 else f"Cancelled {n}, sir."
    return CommandResult(handled=True, reply=line, speak=True,
                         status=f"Cancelled {n} {kind}{'s' if n > 1 else ''}")


def _h_cancel_schedule(c, t, m):
    tk = c._svc("timekeeper")
    if tk is None:
        return None
    word = m.group("kind").lower()
    kind = "reminder" if word.startswith("remind") else \
        "timer" if word.startswith("timer") else "alarm"
    which = (m.group("which") or "").strip()
    if m.group("all") or (not which and word.endswith("s")):
        which = "all"
    if which == "all":
        n = _pending_count(tk, kind)
        if _wants_read_back(c, n):
            # n == 1 only on a shaky transcript: "all one alarm" reads badly
            line = f"Cancel the {kind}, sir?" if n == 1 else \
                f"Cancel all {_count_word(n)} {kind}s, sir?"
            c.stash_destructive(lambda: _do_cancel_schedule(tk, "all", kind), line)
            return CommandResult(handled=True, reply=line, speak=True,
                                 status="Confirm?")
    return _do_cancel_schedule(tk, which, kind)


def _h_briefing(c, t, m):
    enabled = bool(_assistant_get(c, "briefing.enabled", False))
    explicit = not re.match(r"^good morning", t, re.I)
    if not enabled and not explicit:
        return None            # a plain greeting: the local model answers it
    brain = c._svc("brain")
    if brain is None or not hasattr(brain, "chat"):
        return None
    brain.chat(t, force_tool="get_briefing")
    return CommandResult(handled=True, status="Briefing…", done=False)


def _force_briefing(c, text: str, when: str, status: str) -> Optional[CommandResult]:
    """One get_briefing view through the brain (when = tomorrow | week)."""
    brain = c._svc("brain")
    if brain is None or not hasattr(brain, "chat"):
        return None
    brain.chat(text, force_tool="get_briefing", force_args={"when": when})
    return CommandResult(handled=True, status=status, done=False)


def _h_preview(c, t, m):
    """"What does tomorrow look like?": the evening preview, on request,
    whatever briefing.enabled says (the flag governs only the plain
    good-night trigger, as it governs only "good morning")."""
    return _force_briefing(c, t, "tomorrow", "Preview…")


def _h_week(c, t, m):
    """"How's my week looking?": calendar, Canvas, reminders and to-dos day
    by day, with the heavy and the clear days named."""
    return _force_briefing(c, t, "week", "Week ahead…")


def _goodnight_preview(c, t) -> Optional[CommandResult]:
    """The "good night" wind-down: the night line now, tomorrow in one
    breath when the tool lands (and "Shall I wake you at seven?" when the
    first event is early and no alarm covers it -- see
    Commander._try_alarm_offer). Only while briefing.enabled is on, the
    same flag that turns a plain "good morning" into a briefing: on a box
    that never asked for briefings a good night stays a good night."""
    if not bool(_assistant_get(c, "briefing.enabled", False)):
        return None
    brain = c._svc("brain")
    if brain is None or not hasattr(brain, "chat"):
        return None
    try:
        brain.chat(PREVIEW_ASK, force_tool="get_briefing", force_args={"when": "tomorrow"})
    except Exception:
        log.exception("good-night preview failed to start")
        return None
    # ack=True: the night line is not the answer, so it neither closes the
    # turn nor arms a follow-up window of its own; the preview does both.
    return CommandResult(handled=True, reply=GOODNIGHT_PREVIEW_LINE, speak=True,
                         ack=True, status="Preview…", done=False)


def _persist_preference(c, key: str, value) -> bool:
    """A voice-set preference goes to BOTH stores: assistant.json is what
    the briefing reads (cfg.get), the memory's preferences.json is the
    record of what he asked for. Returns True when the config took it."""
    cfg = c._svc("assistant")
    wrote = False
    if cfg is not None and hasattr(cfg, "set"):
        try:
            cfg.set(key, value)
            wrote = True
        except Exception:
            log.exception("preference %s could not be saved to the config", key)
    memory = c._svc("memory")
    fn = getattr(memory, "set_preference", None) if memory is not None else None
    if callable(fn):
        try:
            fn(key, value)
        except Exception:
            log.exception("preference %s could not be saved to memory", key)
    return wrote


def _h_pref_section(c, t, m):
    """"No news in the morning" / "put the sports back in my briefing"."""
    cfg = c._svc("assistant")
    if cfg is None or not hasattr(cfg, "set"):
        return None
    name = _PREF_SECTIONS.get(m.group("section").lower().replace(" ", ""))
    if name is None:
        return None
    enable = bool(m.group("on"))
    if not _persist_preference(c, f"briefing.sections.{name}", enable):
        return None
    words = _PREF_SECTION_WORDS.get(name, name)
    if enable:
        line = f"Very good, sir; {words} {'are' if name in ('todos', 'alarms', 'reminders') else 'is'} back in the briefing."
    else:
        line = f"Very good, sir; no {name.replace('todos', 'to-dos')} in the briefing from now on."
    return CommandResult(handled=True, reply=line, speak=True,
                         status=f"Briefing: {name} {'on' if enable else 'off'}")


def _h_verbosity(c, t, m):
    """"Be briefer" / "shorter briefings" -> briefing.verbosity=brief (the
    briefing views halve their sentence allowance); "the full briefing" /
    "more detail" restores it. Ordinary replies are already capped at two
    sentences, so the briefings are the only place a knob can bite."""
    cfg = c._svc("assistant")
    if cfg is None or not hasattr(cfg, "set"):
        return None
    brief = bool(m.group("brief"))
    if not _persist_preference(c, "briefing.verbosity", "brief" if brief else "normal"):
        return None
    return CommandResult(handled=True, reply=BRIEFER_LINE if brief else FULL_LENGTH_LINE,
                         speak=True, status="Briefings: brief" if brief else "Briefings: full")


# "What was my last email about?" asks for the most recent message, read or
# not. get_mail documents unread_only=false as the flag that answers exactly
# that question, in the parameter description the model is shown -- and the
# model still chose true, so the answer skipped every read message and named
# an older one as the latest. Pin it here rather than ask again more loudly.
# Mail words only: "message" also means Discord, notes and Claude sessions
# in this app, so "the last message you sent" must not be hijacked into mail.
# ---- Tier 1 focus / study sessions (jarvis/focus.py) ------------------
# "study session biosensors", "start a fifty-minute focus session",
# "pomodoro", "25 minute study session for signals with a ten minute break".
_FOCUS_KIND = r"(?:study|studying|focus|focused|pomodoro|revision|work|deep[- ]work)"
_FOCUS_START_RX = re.compile(
    r"^(?:(?:start|begin|run|do|open|kick off)\s+(?:a\s+|an\s+|the\s+|my\s+)?)?"
    r"(?:(?P<n>" + _NUM_ALT + r")\s*[- ]?(?:minutes?|mins?)\s+)?"
    r"(?:(?:(?P<kind>" + _FOCUS_KIND + r")\s+)?session|(?P<pomo>pomodoro))"
    r"(?:\s+(?:for|on|of|about)?\s*(?P<label>(?!with\b)[^,]+?))?"
    r"(?:,?\s+with\s+(?:a\s+|an\s+)?(?P<b>" + _NUM_ALT + r")\s*[- ]?(?:minutes?|mins?)\s+breaks?)?"
    r"[.!]*$", re.I)
_FOCUS_LEFT_RX = re.compile(
    r"^(?:how (?:long|much time|much longer)(?: is| do i have| have i got|'s| is there)? left"
    r"|how long (?:until|till|to) (?:the |my )?(?:next )?break|time left"
    r"|how long (?:have i got|do i have|is there)(?: to go)?)"
    r"(?:\s+(?:in|on|of)\s+(?:the|this|my)\s+(?:block|session|break|timer))?[?.!]*$", re.I)
_FOCUS_END_RX = re.compile(
    r"^(?:(?:end|stop|finish|close|quit|wrap up)\s+(?:the\s+|this\s+|my\s+)?"
    r"(?:" + _FOCUS_KIND + r"\s+)?session|end session"
    r"|(?:i'?m|i am) done (?:studying|working|for today|for now)"
    r"|(?:that'?s|thats) enough (?:studying|for today|for now))[.!]*$", re.I)


def _m_focus_start(t):
    """A bare "session" is too vague (Claude sessions exist too): a kind
    word, a length or "pomodoro" must be present."""
    m = _FOCUS_START_RX.match(t)
    if m and (m.group("kind") or m.group("n") or m.group("pomo")):
        return m
    return None


def _h_focus_start(c, t, m):
    focus = c._svc("focus")
    if focus is None:
        return None
    n = _num(m.group("n")) if m.group("n") else None
    brk = _num(m.group("b")) if m.group("b") else None
    label = _raw_group(c, _FOCUS_START_RX, m, "label") if m.group("label") else ""
    line = focus.start(label, n, brk)
    status = f"Focus: {label}" if label else "Focus session"
    return CommandResult(handled=True, reply=line, speak=True, status=status)


def _h_focus_left(c, t, m):
    focus = c._svc("focus")
    if focus is not None and focus.active:
        return CommandResult(handled=True, reply=focus.time_left(), speak=True,
                             status="Focus: time left")
    tk = c._svc("timekeeper")
    if tk is None:
        return None
    # No session: the honest answer is whatever timer is running.
    return CommandResult(handled=True, reply=tk.list_text("timer"), speak=True,
                         status="Timers")


def _h_focus_end(c, t, m):
    focus = c._svc("focus")
    if focus is None or not focus.active:
        return None            # not ours: "stop the session" may mean Claude
    return CommandResult(handled=True, reply=focus.end(), speak=True,
                         status="Focus: ended")


# ---- Tier 1 lecture notes (jarvis/lecture.py) ---------------------------
# "notes for biosensors", "take notes for signals", "lecture notes on
# physiology". A mode like dictation: Commander.handle files every later
# utterance until "end notes".
_LECTURE_RX = re.compile(
    r"^(?:(?:start|open|begin|take|taking|start taking)\s+)?"
    r"(?:(?:the\s+|my\s+)?(?:lecture|class|course)\s+)?notes"
    r"\s+(?:for|on|in)\s+(?P<course>.+?)[.!]*$", re.I)
_LECTURE_END_RX = re.compile(
    r"^(?:(?:end|stop|close|finish|save)\s+(?:the\s+|my\s+)?(?:lecture\s+|class\s+)?notes"
    r"|(?:end|stop)\s+(?:the\s+)?note[- ]taking)(?:[, ]+(?:please|now|jarvis))*[.!?]*$", re.I)


def _h_lecture_start(c, t, m):
    spoken = _raw_group(c, _LECTURE_RX, m, "course")
    cfg = c._svc("assistant")
    course = lecture_mod.resolve_course(cfg, spoken)
    try:
        c._lecture = lecture_mod.LectureNotes(cfg, course, notes=c._svc("notes"))
    except OSError:
        log.exception("lecture notes: cannot open the notes file")
        return CommandResult(handled=True, reply=lecture_mod.FAIL_LINE, speak=True,
                             status="Notes: folder unwritable")
    c.lecture_course = c._lecture.course
    log.info("lecture notes: open for %r -> %s", c.lecture_course, c._lecture.path)
    return CommandResult(handled=True,
                         reply=lecture_mod.START_LINE.format(course=c.lecture_course),
                         speak=True, status=f"Lecture notes: {c.lecture_course}")


_LAST_MAIL_RX = re.compile(
    r"\b(?:last|latest|most recent|newest)\s+(?:e-?mails?|mails?)\b", re.I)
_LAST_MAIL_HOURS = 168        # a week: "my last email" is not "since midnight"
# A read, never a write: "reply to my latest email" / "delete the last mail"
# name the same message but want something get_mail cannot do. Those fall
# through to the router, which can at least say so.
# Anchored to the leading verb: "what did mark say in his last email" and
# "my last email about the move" are reads that happen to contain a name or
# a noun from this list, and a bare word-list guard blocked them.
# Unambiguous write verbs count anywhere ("go ahead and delete my last email"
# defeated a start-anchored form); the two that double as a name or a noun,
# mark and move, only when followed by an object ("mark my last email").
_MAIL_WRITE_RX = re.compile(
    r"\b(?:reply|respond|answer|delete|trash|forward|archive|send|compose|"
    r"write|draft|flag|star|unsubscribe)\b"
    r"|\b(?:mark|move)\s+(?:my|the|this|that|it|his|her|their|last|latest|"
    r"newest|most recent)\b", re.I)


_DIAG_RX = re.compile(
    r"^(?:run (?:a |the )?)?(?:diagnostics?|self[- ]test|system (?:report|check|status)|"
    r"status report|how are your systems)\W*$", re.I)


# "When's my next exam?" / "how long until the biosensors midterm?" --
# answered without a model turn from Canvas plus the calendar cache. The
# whole question after the opener is the query: tools/canvas.next_exam reads
# the kind word (exam/midterm/final vs quiz) and the course words from it.
_NEXT_EXAM_RX = re.compile(
    r"^(?:when(?:'s| is| do i have)\s+(?:my |the )?(?:next\s+)?"
    r"(?P<q1>(?:\w+\s+){0,4}?(?:exams?|midterms?|finals?|quiz(?:zes)?))"
    r"|how (?:long|many days) (?:is it |do i have )?(?:until|till|before|to)\s+"
    r"(?:my |the )?(?P<q2>(?:\w+\s+){0,4}?(?:exams?|midterms?|finals?|quiz(?:zes)?)))"
    r"\W*$", re.I)


def _h_next_exam(c, t, m):
    from jarvis.tools.canvas import NO_EXAM_LINE, NO_QUIZ_LINE, exam_words, find_next_exam
    cfg = c._svc("assistant")
    cal = c._svc("calendar")
    if cfg is None and cal is None:
        return None
    query = (m.group("q1") or m.group("q2") or "").strip()
    now = datetime.now().astimezone()
    try:
        exam, checked = find_next_exam(cfg, cal, query=query, now=now)
    except Exception:
        log.exception("next exam lookup failed")
        return None
    if exam is None:
        if not checked:
            # No token: let the router reach canvas_due, whose setup line
            # says what is missing -- a bare "nothing on the books" would
            # be a lie about a source that was never read.
            return None
        line = NO_QUIZ_LINE if re.search(r"quiz", query, re.I) else NO_EXAM_LINE
        return CommandResult(handled=True, reply=line, speak=True, status="No exam found")
    words = exam_words(exam, now)
    if re.search(r"\bnext\b", t, re.I):
        line = f"Your next {exam['kind']} is the {words}, sir."
    else:
        line = f"The {words}, sir."
    return CommandResult(handled=True, reply=line, speak=True,
                         status=f"{exam['title'][:24]} {words.split(', ', 1)[-1][:24]}")


def _h_diagnostics(c, t, m):
    """The film's "run diagnostics": uptime, models, today's turns, the box."""
    fn = c._svc("diagnostics")
    if fn is None:
        return None
    try:
        line = fn()
    except Exception:
        log.exception("diagnostics failed")
        line = "I'm afraid the diagnostics didn't complete, sir."
    return CommandResult(handled=True, reply=line, speak=True, status="Diagnostics")


# "How did yesterday go" -- the nightly self-review (jarvis/dayreview.py),
# spoken on demand. "Today" reads the open day so far. Whole-utterance only:
# "how did the meeting go yesterday" is a question for the model.
_DAYREVIEW_RX = re.compile(
    r"^(?:how(?: did| was| were| has| is| are|'s|'d) (?:things |it |your day |the day )?"
    r"(?:go(?:ing)? |been )?(?:for you )?(yesterday|today)(?: go(?:ing)?| been)?"
    r"|(?:(yesterday|today)'?s? (?:self[- ])?(?:review|report|digest|summary))"
    r"|(?:review (yesterday|today))"
    r"|(?:what went wrong (yesterday|today)))\W*$", re.I)


def _h_dayreview(c, t, m):
    fn = c._svc("dayreview")
    if fn is None:
        return None
    which = next((g for g in m.groups() if g), "yesterday").lower()
    try:
        line = fn(which)
    except Exception:
        log.exception("day review failed")
        line = "I'm afraid the review didn't complete, sir."
    return CommandResult(handled=True, reply=line, speak=True, status="Day review")

# Jarvis reading his own log (jarvis/logtriage.py): the developer's fastest
# bug report.  Log-specific words only -- "what went wrong" alone is the
# persona's, and "any errors" without "log" could be about a build.
_LOGTRIAGE_RX = re.compile(
    r"^(?:(?:is|was) there |is )?anything (?:wrong|bad|broken|amiss|off|new) "
    r"(?:in|with) (?:your|the) logs?(?: today| lately| recently)?\W*$"
    r"|^(?:check|read|look at|scan|triage|go through|review) (?:your|the) logs?"
    r"(?: for (?:errors|problems|trouble|warnings))?\W*$"
    r"|^(?:any|what|which) (?:errors|warnings|problems|trouble|failures) "
    r"(?:in|from|on) (?:your|the) logs?(?: today| lately| recently)?\W*$"
    r"|^what(?:'s| is) (?:in|wrong in|wrong with) (?:your|the) logs?\W*$"
    r"|^(?:log|logs) triage\W*$|^triage (?:your|the) logs?\W*$", re.I)
_SLOW_RX = re.compile(
    r"^why (?:was|is|did) (?:that|it|this|the last (?:one|turn|answer|reply))"
    r"(?: so| take so)? (?:slow|long)\W*$"
    r"|^why did (?:that|it|this) take (?:so|that) long\W*$"
    r"|^what took (?:you )?so long\W*$|^where did the time go\W*$"
    r"|^why (?:so|the) slow\W*$|^what was slow(?: about (?:that|it))?\W*$"
    r"|^(?:break down|explain) (?:that|the last) turn\W*$", re.I)


def _h_log_triage(c, t, m):
    """"Anything wrong in your log?": two spoken sentences; the clusters
    and their tracebacks go to the text card (a display-only JarvisReply)."""
    fn = c._svc("log_triage")
    if fn is None:
        return None
    card = ""
    try:
        out = fn()
        spoken, card = (out if isinstance(out, tuple) else (str(out), ""))
    except Exception:
        log.exception("log triage failed")
        spoken = "I'm afraid I couldn't read my own log, sir."
    if card:
        bus.publish(JarvisReply(text=card, speak=False))
    return CommandResult(handled=True, reply=spoken, speak=True, status="Log triage")


def _h_slow_turn(c, t, m):
    """"Why was that slow?": the last real turn on the ledger, by stage."""
    fn = c._svc("slow_turn")
    if fn is None:
        return None
    try:
        line = fn()
    except Exception:
        log.exception("slow-turn lookup failed")
        line = "I'm afraid I couldn't read the turn ledger, sir."
    return CommandResult(handled=True, reply=line, speak=True, status="Turn breakdown")


def _h_last_mail(c, t, m):
    if _MAIL_WRITE_RX.search(t):
        return None
    brain = c._svc("brain")
    if brain is None or not hasattr(brain, "chat"):
        return None
    brain.chat(t, force_tool="get_mail",
               force_args={"limit": 1, "since_hours": _LAST_MAIL_HOURS,
                           "unread_only": False})
    return CommandResult(handled=True, status="Checking mail…", done=False)


def _h_goodnight(c, t, m):                                 # 3233-3238
    if t in ("good night", "goodnight"):
        res = _goodnight_preview(c, t)
        if res is not None:
            return res
    return CommandResult(
        handled=True,
        reply="Good night sir. I'll be here when you need me.",
        speak=_talkback())


def _h_processes(c, t, m):                                 # 3240-3250
    procs = c._svc("context").list_heavy_processes()
    text = "\n".join(f"- {p['cmd']} (CPU:{p['cpu']}% MEM:{p['mem']}%)"
                     for p in procs)
    c._speak(f"Top process is {procs[0]['cmd']} using {procs[0]['cpu']} "
             f"percent CPU." if procs else "No heavy processes.")
    return CommandResult(handled=True, reply=f"Top processes:\n{text}")


def _h_git_status(c, t, m):                                # 3252-3265
    info = c._svc("context").git_summary()
    if info:
        text = (f"Branch: {info['branch']}\n"
                f"Changed: {info['changed_files']} files\n"
                f"Last: {info['last_commit']}\n"
                f"Ahead: {info['commits_ahead']} commits")
        c._speak(f"On branch {info['branch']}. {info['changed_files']} "
                 f"changed files. {info['commits_ahead']} commits ahead "
                 f"of remote.")
        return CommandResult(handled=True, reply=text)
    return CommandResult(handled=True, status="No git info")


def _h_standup(c, t, m):
    """"What did I do today?" / "standup": every cleared repo's commits for
    the day, the uncommitted diff stat, and the Claude sessions touched that
    day -- a card of subjects plus two spoken sentences composed from the
    data (no model turn, so it still answers while the GPU is lent out).
    Synchronous: three repos and a few capped transcript reads are well
    under a second, and a done=False turn would sit open until the 60 s
    watchdog with nothing to close it."""
    ctx = c._svc("context")
    git_repos = getattr(ctx, "git_repos", None)
    if not callable(git_repos):
        return None
    word = standup.day_word(m)
    try:
        line, card = standup.build(git_repos, word)
    except Exception:
        log.exception("standup failed")
        return CommandResult(handled=True, reply=standup.FAILED_LINE, speak=True,
                             status="Standup failed")
    # The card is display-only; the spoken line is the reply so a voice
    # question gets a voice answer (clock rule).
    bus.publish(JarvisReply(text=card, speak=False))
    return CommandResult(handled=True, reply=line, speak=True, status="Standup")


# ---- GPU yield (jarvis/brain.py release/reclaim) -------------------------
# The watchdog lends the model to a trainer on its own (health.yield_to_trainer);
# these two are the manual doors. "Take the GPU back" while the trainer still
# runs is an override: the watchdog remembers the trainer and does not lend to
# it again (Watchdog.manual_reclaim), or the next tick would undo the order.
_GPU_RECLAIM_RX = re.compile(
    r"^(?:take (?:the |your )?gpu back|take back (?:the |your )?gpu|"
    r"reclaim (?:the |your )?gpu|(?:re)?load your (?:model|brain)(?: back| again)?|"
    r"get your (?:model|brain|gpu) back|bring your (?:model|brain) back)\W*$", re.I)
_GPU_LEND_RX = re.compile(
    r"^(?:lend (?:the |your )?gpu(?: to (?:the |my )?trainer)?|"
    r"(?:release|free up|free|give up|yield) (?:the |your )?gpu|"
    r"unload your (?:model|brain))\W*$", re.I)
GPU_RECLAIMED_LINE = "The GPU is mine again, sir; my model is loading."
GPU_RECLAIM_FAILED_LINE = "I have the GPU back, sir, but my model would not load."
GPU_NOT_LENT_LINE = "I never lent it out, sir; my model is where it should be."
GPU_LENT_LINE = "I've lent the GPU out, sir; quick answers only until you take it back."
GPU_ALREADY_LENT_LINE = "It's already lent out, sir."


def _gpu_doors(c):
    brain = c._svc("brain")
    release = getattr(brain, "release", None)
    reclaim = getattr(brain, "reclaim", None)
    if not callable(release) or not callable(reclaim):
        return None
    return brain


def _h_gpu_reclaim(c, t, m):
    brain = _gpu_doors(c)
    if brain is None:
        return None
    is_lent = getattr(brain, "is_lent", None)
    if callable(is_lent) and not is_lent():
        return CommandResult(handled=True, reply=GPU_NOT_LENT_LINE, speak=True,
                             status="GPU not lent")
    wd = c._svc("health_watchdog")

    def _run():
        try:
            hold = getattr(wd, "manual_reclaim", None)
            ok = hold() if callable(hold) else brain.reclaim()
        except Exception:
            log.exception("gpu reclaim failed")
            ok = False
        bus.publish(Status(text="GPU reclaimed" if ok else "Model failed to load",
                           kind="ok" if ok else "warn"))
        if not ok:
            c._speak(GPU_RECLAIM_FAILED_LINE)

    # reclaim() warms the model (seconds): the acknowledgement goes first
    c._bg(_run)
    return CommandResult(handled=True, reply=GPU_RECLAIMED_LINE, speak=True,
                         status="Reclaiming GPU…")


def _h_gpu_lend(c, t, m):
    brain = _gpu_doors(c)
    if brain is None:
        return None
    is_lent = getattr(brain, "is_lent", None)
    if callable(is_lent) and is_lent():
        return CommandResult(handled=True, reply=GPU_ALREADY_LENT_LINE, speak=True,
                             status="GPU lent")
    try:
        brain.release()
    except Exception:
        log.exception("gpu release failed")
    return CommandResult(handled=True, reply=GPU_LENT_LINE, speak=True,
                         status="GPU lent")


def _h_network(c, t, m):                                   # 3267-3279
    net = c._svc("context").check_connectivity()
    status = "Online" if net.get("internet") else "Offline"
    latency = f", {net.get('latency_ms', '?')}ms" if net.get("internet") else ""
    ollama = "running" if net.get("ollama") else "not running"
    text = f"Internet: {status}{latency}\nOllama: {ollama}"
    c._speak(f"You are {status.lower()}{latency}. Ollama is {ollama}.")
    return CommandResult(handled=True, reply=text)


def _h_find_file(c, t, m):                                 # 3281-3294
    name = m.group(1).strip()
    files = c._svc("context").find_file(name)
    if files:
        text = "\n".join(f"- {f}" for f in files)
        c._speak(f"Found {len(files)} files matching {name}.")
        return CommandResult(handled=True, reply=f"Found:\n{text}")
    return CommandResult(handled=True, reply=f"No files matching '{name}'")


def _h_recent_files(c, t, m):                              # 3296-3306
    files = c._svc("context").recent_files()
    text = "\n".join(f"- {Path(f).name}" for f in files)
    names = ", ".join(Path(f).name for f in files[:3])
    c._speak(f"Most recently modified: {names}.")
    return CommandResult(handled=True, reply=f"Recently modified:\n{text}")


def _h_clip_history(c, t, m):                              # 3308-3321
    items = c._svc("context").get_clipboard_history()
    if items:
        text = "\n".join(f"{i + 1}. {clip['text'][:60]}"
                         for i, clip in enumerate(items))
        c._speak(f"You have {len(items)} items in clipboard history.")
        return CommandResult(handled=True, reply=f"Clipboard history:\n{text}")
    return CommandResult(handled=True, reply="Clipboard history is empty.")


def _h_paste_item(c, t, m):                                # 3323-3331
    idx_str = m.group(1)
    idx = 1 if idx_str in ("before last", "previous") else int(idx_str) - 1
    result = c._svc("context").paste_from_history(idx)
    if result:
        return CommandResult(handled=True, reply=f"Pasted: {result}")
    return CommandResult(handled=True, status="Nothing to paste")


def _h_take_note(c, t, m):                                 # 3333-3342
    # Original casing from the raw utterance when we have it (names).
    raw = getattr(c, "_raw_text", "") or ""
    m2 = _NOTE_RX.match(strip_jarvis_prefix(raw) or raw.strip()) if raw else None
    note = ((m2 or m).group("text") or "").strip()
    if not note:
        return None
    notes = c._svc("notes")
    if notes is not None:
        note_id = notes.add("note", note)
        return CommandResult(handled=True, reply="Noted, sir.", speak=True,
                             status=f"Note: {note[:40]}",
                             undo=_undo_notes(notes, "note", note_id,
                                              "Note struck out, sir."))
    memory = c._svc("memory")
    if memory is None:
        return None
    memory.save_note(note)
    c._speak("Note saved.")
    return CommandResult(handled=True, reply=f"Note saved: {note}")


def _h_show_notes(c, t, m):                                # 3344-3353
    notes = c._svc("notes")
    if notes is not None:
        return CommandResult(handled=True, reply=notes.list_text("note"),
                             speak=True, status="Notes")
    memory = c._svc("memory")
    if memory is None:
        return None
    items = memory.get_notes()
    if items:
        text = "\n".join(f"- {n['content']}" for n in items)
        return CommandResult(handled=True, reply=f"Recent notes:\n{text}")
    return CommandResult(handled=True, reply="No voice notes yet.")


def _h_todo_add(c, t, m):
    notes = c._svc("notes")
    if notes is None:
        return None
    raw = getattr(c, "_raw_text", "") or ""
    m2 = _TODO_ADD_RX.match(strip_jarvis_prefix(raw) or raw.strip()) if raw else None
    mm = m2 or m
    text = (mm.group("t1") or mm.group("t2") or mm.group("t3") or
            mm.group("t4") or "").strip(" .")
    if not text:
        return None
    todo_id = notes.add("todo", text)
    return CommandResult(handled=True, reply="Added to your list, sir.",
                         speak=True, status=f"To-do: {text[:40]}",
                         undo=_undo_notes(notes, "todo", todo_id,
                                          "Off your list again, sir."))


def _h_todo_list(c, t, m):
    notes = c._svc("notes")
    if notes is None:
        return None
    return CommandResult(handled=True, reply=notes.list_text("todo"),
                         speak=True, status="To-dos")


def _h_todo_done(c, t, m):
    notes = c._svc("notes")
    if notes is None or not hasattr(notes, "complete"):
        return None
    which = (m.group("w1") or m.group("w2") or m.group("w3") or "last").strip()
    done = notes.complete(which)
    if not done:
        return CommandResult(handled=True,
                             reply="I couldn't find that on your list, sir.",
                             speak=True, status="Not on the list")
    left = ""
    try:
        n = notes.count("todo")
        left = f" {n} left." if isinstance(n, int) and n > 0 else \
            " That's the list cleared." if n == 0 else ""
    except Exception:
        left = ""
    return CommandResult(handled=True, reply=f"Done, sir.{left}", speak=True,
                         status="To-do done")


_RUN_SHELL_RX = re.compile(r"(?:run|execute|shell)\s+(.+)")
_COUNT_LINES_RX = re.compile(r"count (?:the )?lines? in (.+)")


def _raw_cmd_text(c) -> str:
    """The utterance as the user actually said it, with only the "jarvis"
    prefix removed -- the same slice strip_jarvis_prefix() takes, minus the
    lower().  The registry matches on lower-cased text, but a shell command
    is case-sensitive: `echo $HOME/Jarvis` lower-cased becomes `echo
    $home/jarvis`, and an unset $home expands to nothing, so the command
    silently runs against the wrong path instead of failing."""
    raw = (getattr(c, "_raw_text", "") or "").strip().rstrip(".")
    lower = raw.lower()
    for prefix in JARVIS_PREFIXES:
        if lower.startswith(prefix):
            return raw[len(prefix):].strip()
    return raw


def _raw_group(c, rx, m, group: int = 1) -> str:
    """`m.group(n)` re-taken from the original casing when the same pattern
    still matches there; the lower-cased group otherwise."""
    m2 = rx.match(_raw_cmd_text(c))
    return (m2 or m).group(group).strip()


def _h_run_shell(c, t, m):                                 # 3355-3371
    shell_cmd = _raw_group(c, _RUN_SHELL_RX, m)
    log.info("Shell command: %s", shell_cmd)
    bus.publish(Status(text=shell_cmd[:30], kind="busy"))
    ctx = c._svc("context")

    def _run():
        output = ctx.run_shell(shell_cmd)
        bus.publish(JarvisReply(text=f"$ {shell_cmd}\n{output}"))
        bus.publish(Status(text="Command done", kind="ok"))
        if len(output) < 200:
            c._speak(f"Result: {output[:100]}")

    c._bg(_run)
    return CommandResult(handled=True, status=f"Running: {shell_cmd[:30]}",
                         done=False)


def _h_count_lines(c, t, m):                               # 3373-3382
    filename = _raw_group(c, _COUNT_LINES_RX, m)
    output = c._svc("context").run_shell(
        f"wc -l {filename} 2>/dev/null || find /home/hunterp -name "
        f"'{filename}' -exec wc -l {{}} + 2>/dev/null | tail -1")
    return CommandResult(handled=True, reply=output, speak=_talkback())


def _h_other_monitor(c, t, m):                             # 3384-3396
    success = c._svc("desktop").move_window_to_monitor("next")
    if success:
        c._speak("Done.")
        return CommandResult(handled=True,
                             reply="Window moved to other monitor.")
    return CommandResult(handled=True,
                         reply="Could not move window. Single monitor?")


def _h_dictate(c, t, m):                                   # 3398-3405
    c.dictation = True
    c._speak("Dictation mode active. I'll type everything you say directly. "
             "Say end dictation to stop.")
    return CommandResult(
        handled=True,
        reply="Dictation mode: ON\nSay 'end dictation' to stop.")


def _h_trigger(c, t, m):                                   # 3407-3416
    condition = m.group(1).strip()
    c._svc("workflows").set_trigger(condition, f"Alert: {condition}")
    c._speak(f"I'll notify you when {condition}.")
    return CommandResult(handled=True, reply=f"Trigger set: when {condition}")


def _h_transform_case(c, t, m):                            # 3418-3443
    try:
        r = subprocess.run(
            ["xclip", "-selection", "clipboard", "-o"],
            capture_output=True, text=True, timeout=2,
        )
        clip = r.stdout.strip()
        if "upper" in t:
            result = clip.upper()
        else:
            result = clip.lower()
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE,
        )
        proc.communicate(input=result.encode(), timeout=2)
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
            timeout=2, capture_output=True,
        )
        return CommandResult(handled=True, reply=f"Transformed: {result[:50]}")
    except Exception:
        log.exception("text transform failed")
        return CommandResult(handled=True, status="Transform failed")


# Only the system facts the legacy agent still owns: ip / uptime / battery.
# Weather and the clock are tools of the local model now (spec 5.2) and
# never reach jarvis_agent.answer_question again.
_ANSWER_Q_RX = re.compile(
    r"\b(?:ip(?: address)?|uptime|battery|how long has (?:the|this) (?:machine|"
    r"system|box|computer) been (?:up|running))\b", re.I)


def _m_answer_question(t):
    return bool(_ANSWER_Q_RX.search(t))


def _h_answer_question(c, t, m):                           # 3445-3452
    answer = c._svc("context").answer_question(t)
    if not answer:
        return None                    # fall through to QUICK_COMMANDS
    return CommandResult(handled=True, reply=answer, speak=_talkback())


def _m_quick_command(t):                                   # 3454-3463
    for phrase, shell_cmd in QUICK_COMMANDS.items():
        if phrase in t:
            return (phrase, shell_cmd)
    return None


def _h_quick_command(c, t, m):
    phrase, shell_cmd = m
    log.info("Quick command: %s -> %s", phrase, shell_cmd[:50])
    bus.publish(Status(text=phrase, kind="busy"))

    def _run():                                            # 3487-3512
        try:
            result = subprocess.run(
                shell_cmd, shell=True, capture_output=True,
                text=True, timeout=30,
            )
            output = result.stdout.strip()[:300]
            log.info("Quick command result: %s", output[:60])
            bus.publish(JarvisReply(text=f"[{phrase}]\n{output}"))
            c._speak(f"Command {phrase} complete.")
            bus.publish(Status(text=f"{phrase} done", kind="ok"))
        except Exception as e:
            log.exception("Quick command error: %s", phrase)
            bus.publish(Status(text=str(e)[:40], kind="error"))

    c._bg(_run)
    return CommandResult(handled=True, status=f"Running: {phrase}", done=False)


def _h_remind_me(c, t, m):                                 # 3465-3483
    raw = getattr(c, "_raw_text", "") or ""
    m2 = _REMIND_RX.match(strip_jarvis_prefix(raw) or raw.strip()) if raw else None
    body = ((m2 or m).group("body") or "").strip()
    when, task = split_when(body)
    if not when or not task:
        return None                    # the local model's parser has a go
    now = datetime.now()
    tk = c._svc("timekeeper")
    if tk is None:
        lm = _LEGACY_IN_RX.match(when)
        workflows = c._svc("workflows")
        if lm is None or workflows is None or _num(lm.group("n")) is None:
            return None
        seconds = _seconds(_num(lm.group("n")), lm.group("u"),
                           "half" in when.lower())
        workflows.set_reminder(seconds, task)
        return CommandResult(handled=True,
                             status=f"Reminder set: {seconds // 60}m — {task[:30]}")
    due = _due_from(c, tk, when, now)
    if due is None:
        return CommandResult(handled=True, reply=NO_WHEN_LINE, speak=True,
                             status="Reminder: when?")
    item = tk.add_reminder(due, task)
    desc = _describe(tk, due, now, when)
    what = task if re.match(r"^(?:that|about)\b", task, re.I) else f"to {task}"
    return CommandResult(handled=True,
                         reply=f"Very good, sir; I'll remind you {what} {desc}.",
                         speak=True, status=f"Reminder {desc}: {task[:30]}",
                         undo=_undo_timekeeper(tk, item, "reminder",
                                               "Reminder cancelled, sir."))


# ---- Tier 1 ambient: do not disturb / quiet hours / I am free ------------
# Answered locally: "do not disturb for an hour" must take effect NOW, not
# after a model round trip that might itself be the interruption. The bare
# "quiet" stays the barge-in above (_QUIET_RX is anchored, so "quiet for an
# hour" and "quiet hours from ..." never reach it).
_DND_RX = re.compile(
    r"^(?:(?:please\s+)?(?:do not|don't|dont)\s+disturb(?:\s+me)?"
    r"|(?:hold|mute|pause)\s+(?:my\s+|the\s+|all\s+|your\s+)?"
    r"(?:notifications|alerts|interruptions|announcements|reminders)"
    r"|(?:i'm|i am|im)\s+busy|(?:go\s+|be\s+|stay\s+)?quiet(?:\s+mode)?(?=\s+(?:for|until|till))"
    r"|dnd)"
    r"(?:\s+(?P<mode>for|until|till)\s+(?P<when>.+?))?[.!\s]*$", re.I)
_QUIET_HOURS_RX = re.compile(
    r"^(?:set\s+|make\s+|my\s+)?quiet\s+hours(?:\s+(?:are|to|from|between|run))?"
    r"\s+(?:from\s+)?(?P<a>.+?)\s+(?:to|until|till|and)\s+(?P<b>.+?)[.!\s]*$", re.I)
_QUIET_HOURS_OFF_RX = re.compile(
    r"^(?:(?:turn|switch)\s+off|disable|clear|cancel|remove|no|stop)\s+"
    r"(?:the\s+|my\s+)?quiet\s+hours[.!\s]*$", re.I)
_FREE_RX = re.compile(
    r"^(?:(?:i am|i'm|im)\s+(?:free|back|done|available|not busy|no longer busy|"
    r"out of (?:class|the meeting|my meeting|the exam|my exam))(?:\s+now)?"
    r"|(?:end|stop|cancel|lift|clear|turn off|switch off)\s+(?:the\s+)?"
    r"(?:do not disturb|dnd|quiet mode|quiet time)"
    r"|(?:resume|unmute)\s+(?:my\s+|the\s+|your\s+)?(?:alerts|notifications|announcements)"
    r"|what did i miss|anything (?:i missed|held(?: back)?|while i was (?:busy|out|away)))"
    r"[?.!\s]*$", re.I)
_QUIET_STATUS_RX = re.compile(
    r"^(?:(?:are you|am i)\s+(?:on|in)\s+(?:do not disturb|dnd|quiet hours|quiet mode)"
    r"|(?:what|when) are (?:my|the) quiet hours|quiet status|is (?:do not disturb|dnd) on)"
    r"[?.!\s]*$", re.I)


def _dnd_seconds(c, mode: str, when: str, now: datetime) -> Optional[float]:
    """'for an hour' / 'until seven' -> seconds from now; None = unparseable."""
    when = (when or "").strip()
    if not when:
        return 3600.0
    if mode in ("until", "till"):
        from jarvis.quiet import parse_clock
        # "until seven" with no am/pm: the NEXT seven, whichever half of the
        # day that is (at 2 pm it is 7 pm; at 9 pm it is 7 am). Explicit
        # am/pm parses the same under both defaults.
        ends = []
        for default in ("am", "pm"):
            hm = parse_clock(when, default=default)
            if hm is None:
                continue
            end = now.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
            if end <= now:
                end += timedelta(days=1)
            ends.append(end)
        if ends:
            return (min(ends) - now).total_seconds()
        due = _due_from(c, c._svc("timekeeper"), when, now)
        return None if due is None or due <= now.timestamp() else due - now.timestamp()
    try:
        from jarvis.tools.timekeeper import parse_duration
        secs = parse_duration(when)
    except Exception:
        secs = None
    if secs is None and when.lower() in ("the rest of the day", "the day", "today"):
        end = now.replace(hour=23, minute=59, second=0, microsecond=0)
        secs = (end - now).total_seconds()
    return secs


def _h_dnd(c, t, m):
    q = c._svc("quiet")
    if q is None:
        return None
    now = datetime.fromtimestamp(q.now())
    secs = _dnd_seconds(c, (m.group("mode") or "for").lower(), m.group("when") or "", now)
    if secs is None:
        from jarvis.tools.timekeeper import CANT_PARSE_LINE
        return CommandResult(handled=True, reply=CANT_PARSE_LINE, speak=True,
                             status="Do not disturb: when?")
    from jarvis.quiet import DND_SET_LINE, fmt_clock
    until = q.set_dnd(secs)
    end = datetime.fromtimestamp(until)
    words = fmt_clock(end.hour, end.minute)
    bus.publish(Status(text=f"Do not disturb until {words}", kind="info"))
    return CommandResult(handled=True, reply=DND_SET_LINE.format(until=words),
                         speak=True, status=f"Do not disturb until {words}")


def _h_quiet_hours(c, t, m):
    q = c._svc("quiet")
    if q is None:
        return None
    from jarvis.quiet import QUIET_HOURS_SET_LINE, fmt_clock, parse_clock
    # Quiet hours run overnight by convention: a bare start hour is pm, a
    # bare end hour is am ("from eleven to seven" = 23:00-07:00).
    start, end = parse_clock(m.group("a"), default="pm"), parse_clock(m.group("b"), default="am")
    if start is None or end is None or start == end:
        from jarvis.tools.timekeeper import CANT_PARSE_LINE
        return CommandResult(handled=True, reply=CANT_PARSE_LINE, speak=True,
                             status="Quiet hours: when?")
    q.set_hours(start, end)
    a, b = fmt_clock(*start), fmt_clock(*end)
    return CommandResult(handled=True, reply=QUIET_HOURS_SET_LINE.format(start=a, end=b),
                         speak=True, status=f"Quiet hours {a}–{b}")


def _h_quiet_hours_off(c, t, m):
    q = c._svc("quiet")
    if q is None:
        return None
    from jarvis.quiet import QUIET_HOURS_OFF_LINE
    q.set_hours(None, None)
    return CommandResult(handled=True, reply=QUIET_HOURS_OFF_LINE, speak=True,
                         status="Quiet hours off")


def _h_free(c, t, m):
    q = c._svc("quiet")
    if q is None:
        return None
    from jarvis.quiet import BUSY_PREFIX, NOTHING_HELD_LINE
    if t.startswith(("what did", "anything")):
        # A question, not an order: read what was held WITHOUT ending the
        # window. "What did I miss?" used to call free(), which cancelled
        # an explicit do-not-disturb for good as a side effect.
        held = q.release(prefix=BUSY_PREFIX)
        return CommandResult(handled=True, reply=held or NOTHING_HELD_LINE,
                             speak=True,
                             status="Held lines" if held else "Nothing held")
    line = q.free()
    bus.publish(Status(text="Free", kind="info"))
    return CommandResult(handled=True, reply=line, speak=True, status="Free")


def _h_quiet_status(c, t, m):
    q = c._svc("quiet")
    if q is None:
        return None
    from jarvis.quiet import (QUIET_HOURS_OFF_LINE, QUIET_HOURS_SET_LINE,
                              QUIET_STATUS_FREE_LINE, QUIET_STATUS_QUIET_LINE, fmt_clock)
    if "quiet hours" in t and t.startswith(("what", "when")):
        win = q.quiet_hours()
        if win is None:
            return CommandResult(handled=True, reply=QUIET_HOURS_OFF_LINE, speak=True,
                                 status="Quiet hours off")
        a, b = fmt_clock(*win[0]), fmt_clock(*win[1])
        return CommandResult(handled=True,
                             reply=QUIET_HOURS_SET_LINE.format(start=a, end=b),
                             speak=True, status=f"Quiet hours {a}–{b}")
    reason = q.reason()
    line = QUIET_STATUS_QUIET_LINE.format(reason=reason) if reason else QUIET_STATUS_FREE_LINE
    return CommandResult(handled=True, reply=line, speak=True,
                         status="Quiet" if reason else "Not quiet")


# Ordered registry — mirrors _check_quick_command branch order (3036-3485).
# The single insertion is "autonomous" before "workflow" (V3 spec: "deploy"
# and "autonomous:" phrases route to brain.execute_autonomous).
# ---- Tier 1 quiz mode -------------------------------------------------
# "Quiz me on chapter three" pulls the chunks (docs.topic_chunks), has the
# model write the questions (brain.make_quiz), files them as Leitner cards
# and asks the first one; the NEXT utterance is the answer (_try_quiz_answer
# in handle(), ahead of the intent gate, which would call "forty percent"
# background chat). All state lives here in the commander, never in the
# app: app._pending_uncertain blocks the follow-up window.
_QUIZ_RX = re.compile(
    r"^" + _JV + r"(?:quiz|test|drill|grill|examine) me (?:on|about|over|from|with) "
    r"(?:the\s+|my\s+)?(?P<topic>.+?)(?:\s+please)?[.!?\s]*$", re.I)
_REVIEW_RX = re.compile(
    r"^" + _JV + r"(?:(?:let's |let us )?(?:review|practice|practise|go through|"
    r"run through|drill|study) (?:my |the |some |today's )?(?:flash ?cards|cards|"
    r"due cards|deck|flashcard deck)|(?:start |begin )?(?:a |the |my )?"
    r"(?:flash ?card review|flash ?cards|review session|card review))"
    r"(?:\s+(?:please|now|again))?[.!?\s]*$", re.I)
_QUIZ_STOP_RX = re.compile(
    r"^" + _JV + r"(?:(?:stop|end|quit|finish|pause|cancel|enough(?: of| with)?|"
    r"that's enough(?: of)?) (?:the |this |my )?(?:quiz|quizzing|flash ?cards|"
    r"review|questions|test|quizzes)(?: me)?|stop quizzing me|no more questions|"
    r"that's enough questions)[.!?\s]*$", re.I)
_QUIZ_SKIP_RX = re.compile(
    r"^(?:skip(?: it| that| this one)?|pass|next(?: one| question)?|"
    r"i (?:don't|do not) know(?: that one| this one| it)?|no idea|not sure|"
    r"dunno|i give up|tell me(?: the answer)?|what's the answer|"
    r"what is the answer)[.!?\s]*$", re.I)


def quiz_kind(text: str) -> Optional[str]:
    m = _QUIZ_RX.match((text or "").strip())
    return m.group("topic").strip() if m else None


def review_kind(text: str) -> bool:
    return bool(_REVIEW_RX.match((text or "").strip()))


def quiz_stop_kind(text: str) -> bool:
    return bool(_QUIZ_STOP_RX.match((text or "").strip()))


def _quiz_store(c):
    store = c._svc("flashcards")
    if store is not None:
        return store
    if getattr(c, "_flashcards", None) is None:
        c._flashcards = quiz_mod.FlashcardStore()
    return c._flashcards


def _cards_line(n: int) -> str:
    return "One card due, sir." if n == 1 else f"{n} cards due, sir."


def _h_quiz(c, t, m):
    topic = m
    index = c._svc("docs")
    if index is None:
        return CommandResult(handled=True, reply=quiz_mod.NO_DOCS_LINE, speak=True,
                             status="No documents")
    brain = c._svc("brain")
    if brain is None or not hasattr(brain, "make_quiz"):
        return CommandResult(handled=True, reply=quiz_mod.NO_QUESTIONS_LINE.format(topic=topic),
                             speak=True, status="No model")
    try:
        store = _quiz_store(c)
    except Exception:
        log.exception("flashcard store unavailable")
        return CommandResult(handled=True, reply=quiz_mod.NO_QUESTIONS_LINE.format(topic=topic),
                             speak=True, status="No store")
    n = _int_setting(c, "quiz.questions", quiz_mod.DEFAULT_QUESTIONS)
    k = _int_setting(c, "quiz.chunks", quiz_mod.DEFAULT_CHUNKS)
    c._pending_quiz = None                       # a new quiz replaces the old

    def _work():
        try:
            chunks = topic_chunks(index, topic, k=k)
        except EmbedError as exc:
            log.warning("quiz: embed failed: %s", exc)
            _deliver(c, quiz_mod.INDEX_DOWN_LINE)
            return
        except Exception:                        # noqa: BLE001 - store boundary
            log.exception("quiz: index failed")
            _deliver(c, quiz_mod.INDEX_DOWN_LINE)
            return
        if not chunks:
            if index.document_count() > 0:
                _deliver(c, quiz_mod.NO_TOPIC_LINE.format(topic=topic))
            elif index.scan():
                index.start_background()         # files present, nothing stored yet
                _deliver(c, INDEXING_LINE)
            else:
                _deliver(c, quiz_mod.NO_DOCS_LINE)
            return
        try:
            pairs = brain.make_quiz(quiz_mod.study_text(chunks), n=n, topic=topic)
        except Exception:
            log.exception("make_quiz failed")
            pairs = []
        if not pairs:
            _deliver(c, quiz_mod.NO_QUESTIONS_LINE.format(topic=topic))
            return
        cards = store.add_cards(pairs, source=chunks[0].get("name", ""), topic=topic)
        session = quiz_mod.QuizSession(cards, topic=topic)
        c._pending_quiz = session
        _deliver(c, session.ask())

    c._bg(_work)
    return CommandResult(handled=True, reply=quiz_mod.PREPARING_LINE.format(topic=topic),
                         speak=True, ack=True, done=False, status=f"Quiz: {topic}")


def _h_review(c, t, m):
    try:
        store = _quiz_store(c)
    except Exception:
        log.exception("flashcard store unavailable")
        return CommandResult(handled=True, reply=quiz_mod.NO_CARDS_LINE, speak=True,
                             status="No store")
    n = _int_setting(c, "quiz.questions", quiz_mod.DEFAULT_QUESTIONS)
    cards = store.due(limit=n)
    if not cards:
        line = quiz_mod.NO_CARDS_LINE if store.count() == 0 else quiz_mod.NOTHING_DUE_LINE
        return CommandResult(handled=True, reply=line, speak=True, status="No cards due")
    session = quiz_mod.QuizSession(cards, topic="review")
    c._pending_quiz = session
    return CommandResult(handled=True, reply=f"{_cards_line(len(cards))} {session.ask()}",
                         speak=True, status=f"Flashcards 1/{len(cards)}")


def _h_quiz_stop(c, t, m):
    session = getattr(c, "_pending_quiz", None)
    if session is None:
        return None                              # no quiz: "stop the test" is the model's
    c._pending_quiz = None
    return CommandResult(handled=True, reply=session.score_line(), speak=True,
                         status="Quiz stopped")


def _int_setting(c, key: str, default: int) -> int:
    try:
        return max(1, int(_assistant_get(c, key, default) or default))
    except (TypeError, ValueError):
        return default


REGISTRY: list[Command] = [
    Command("go back",
            _m_exact("go back", "previous window", "last window"),
            _h_go_back, needs=("context", "desktop")),
    Command("click on",
            _m_re(r"click (?:on |the )?(.+)"),
            _h_click_on, needs=("context",)),
    Command("describe screen",
            _m_contains("what's on screen", "describe screen",
                        "what do you see", "look at screen"),
            _h_describe_screen, needs=("context",)),
    Command("autonomous", _m_autonomous, _h_autonomous, needs=("brain",)),
    Command("clock", clock_kind, _h_clock),              # Tier 1 clock
    Command("courtesy", courtesy_kind, _h_courtesy),     # Tier 1 courtesy
    Command("quiet", quiet_kind, _h_quiet),              # Tier 1 barge-in
    Command("repeat", repeat_kind, _h_repeat),           # Tier 1 say again
    Command("pronounce", _PRONOUNCE_RX.match, _h_pronounce),
    Command("spell name", _SPELL_NAME_RX.match, _h_spell_name),
    Command("add vocabulary", _ADD_VOCAB_RX.match, _h_add_vocab),
    Command("read aloud", read_kind, _h_read_aloud, needs=("reader",)),
    Command("continue reading", continue_kind, _h_continue,
            needs=("reader",)),
    Command("explain document", explain_kind, _h_explain_doc,
            needs=("reader",)),
    Command("quiz", quiz_kind, _h_quiz),
    Command("review flashcards", review_kind, _h_review),
    Command("stop quiz", quiz_stop_kind, _h_quiz_stop),
    # Reached only when the reader is idle (handle() gives an active reading
    # first claim on these words before the desktop chains and "go back");
    # the handler then falls through, so the entry documents Tier 1
    # membership without ever shadowing Spotify or the media keys.
    Command("read control", read_control_kind, _h_read_control,
            needs=("reader",)),
    Command("workflow", lambda t: True, _h_workflow, needs=("workflows",)),
    Command("suggest",
            _m_contains("suggest", "what should i do", "any suggestions"),
            _h_suggest, needs=("memory",)),
    # People book ahead of the generic "remember": "my advisor is Dr X"
    # is a contact, not a free-text fact. The handler returns None (falls
    # through) unless the sentence is plainly about a person.
    Command("person", _m_re(r"(?:my|our) [a-z][a-z' -]{0,30}? is .+"),
            _h_add_person, needs=("memory",)),
    # "remember to buy milk" is a to-do (the notes tool); only "remember
    # (that) <fact>" lands in long-term memory.
    Command("remember",
            _m_re(r"remember (?:that )?(?!to\b)(.+)"),
            _h_remember, needs=("memory",)),
    Command("recall",
            _m_re(r"(?:recall|what did i (?:say|tell you) about|remember about)\s+(.+)"),
            _h_recall, needs=("memory",)),
    Command("who is", _m_re(r"who(?:'s| is) (my .+?)(?:'s)?$"),
            _h_who_is, needs=("memory",)),
    Command("recap", _RECAP_RX.match, _h_recap, needs=("brain",)),
    Command("windows",
            _m_exact("what's open", "whats open", "list windows",
                     "show windows", "what windows are open"),
            _h_windows, needs=("desktop",)),
    Command("launch", _m_re(r"launch\s+(.+)"), _h_launch),
    Command("type", _m_re(r"type\s+(.+)"), _h_type),
    # BEFORE "clipboard": that matcher is a bare substring test, so "have
    # Claude fix the clipboard" would otherwise be read aloud instead.
    Command("clip to claude", _CLIP_TO_CLAUDE_RX.match, _h_clip_to_claude,
            needs=("claude",)),
    Command("clipboard",
            _m_contains("clipboard", "what did i copy", "read clipboard"),
            _h_clipboard),
    Command("web search",
            _m_re(r"(?:search|google|look up)\s+(?:for\s+)?(.+)"),
            _h_search),
    Command("focus start", _m_focus_start, _h_focus_start, needs=("focus",)),
    Command("focus left", _FOCUS_LEFT_RX.match, _h_focus_left),
    Command("focus end", _FOCUS_END_RX.match, _h_focus_end, needs=("focus",)),
    Command("timer", _TIMER_RX.match, _h_timer),
    Command("alarm", _ALARM_RX.match, _h_alarm),
    Command("list schedule", _LIST_SCHED_RX.match, _h_list_schedule,
            needs=("timekeeper",)),
    Command("cancel schedule", _CANCEL_SCHED_RX.match, _h_cancel_schedule,
            needs=("timekeeper",)),
    Command("briefing", _BRIEFING_RX.match, _h_briefing, needs=("brain",)),
    Command("preview", _PREVIEW_RX.match, _h_preview, needs=("brain",)),
    Command("week", _WEEK_RX.match, _h_week, needs=("brain",)),
    Command("briefing section", _PREF_SECTION_RX.match, _h_pref_section,
            needs=("assistant",)),
    Command("verbosity", _PREF_VERBOSITY_RX.match, _h_verbosity,
            needs=("assistant",)),
    Command("last mail", _LAST_MAIL_RX.search, _h_last_mail,
            needs=("brain",)),
    Command("diagnostics", _DIAG_RX.match, _h_diagnostics),
    Command("next exam", _NEXT_EXAM_RX.match, _h_next_exam),
    Command("day review", _DAYREVIEW_RX.match, _h_dayreview),

    Command("log triage", _LOGTRIAGE_RX.match, _h_log_triage),
    Command("slow turn", _SLOW_RX.match, _h_slow_turn),
    # After the briefing: "good morning" is a briefing trigger first.
    Command("greeting", greeting_kind, _h_greeting),
    Command("good night",
            _m_exact("good night", "goodnight", "go to sleep",
                     "shut down jarvis"),
            _h_goodnight),
    Command("processes",
            _m_contains("what's running", "whats running",
                        "heavy processes", "top processes"),
            _h_processes, needs=("context",)),
    # Before "git status": "what did I change today" contains no git-status
    # phrase, but the standup regex is anchored and the safer first match.
    Command("standup", standup.STANDUP_RX.match, _h_standup,
            needs=("context",)),
    Command("gpu reclaim", _GPU_RECLAIM_RX.match, _h_gpu_reclaim,
            needs=("brain",)),
    Command("gpu lend", _GPU_LEND_RX.match, _h_gpu_lend, needs=("brain",)),
    Command("git status",
            _m_contains("git status", "what's changed", "whats changed",
                        "repo status"),
            _h_git_status, needs=("context",)),
    Command("network",
            _m_contains("check network", "am i online", "internet",
                        "connectivity"),
            _h_network, needs=("context",)),
    Command("find file",
            _m_re(r"find (?:file |files? )?(.+)"),
            _h_find_file, needs=("context",)),
    Command("recent files",
            _m_contains("recent files", "what was i working on",
                        "last edited", "recently modified"),
            _h_recent_files, needs=("context",)),
    Command("clipboard history",
            _m_contains("show clipboard", "clipboard history",
                        "last copies", "paste history"),
            _h_clip_history, needs=("context",)),
    Command("paste item",
            _m_re(r"paste (?:item |number )?(\d+|before last|previous)"),
            _h_paste_item, needs=("context",)),
    # Named lists BEFORE the to-do commands: _TODO_ADD_RX would otherwise
    # swallow "add milk to my shopping list" into the generic to-do list.
    Command("list add", _LIST_ADD_RX.match, _h_list_add, needs=("notes",)),
    Command("list read", _LIST_READ_RX.match, _h_list_read, needs=("notes",)),
    Command("list strike", _LIST_STRIKE_RX.match, _h_list_strike,
            needs=("notes",)),
    Command("list clear", _LIST_CLEAR_RX.match, _h_list_clear, needs=("notes",)),
    Command("lists", _LISTS_RX.match, _h_lists, needs=("notes",)),
    Command("todo done", _TODO_DONE_RX.match, _h_todo_done, needs=("notes",)),
    Command("todo add", _TODO_ADD_RX.match, _h_todo_add, needs=("notes",)),
    Command("todo list", _TODO_LIST_RX.match, _h_todo_list, needs=("notes",)),
    Command("lecture notes", _LECTURE_RX.match, _h_lecture_start),
    Command("take note", _NOTE_RX.match, _h_take_note),
    Command("show notes",
            _m_contains("show notes", "read notes", "my notes",
                        "list notes", "voice notes"),
            _h_show_notes),
    Command("run shell", _RUN_SHELL_RX.match,
            _h_run_shell, needs=("context",)),
    Command("count lines", _COUNT_LINES_RX.match,
            _h_count_lines, needs=("context",)),
    Command("other monitor",
            _m_contains("other screen", "other monitor", "move to monitor",
                        "next screen", "next monitor"),
            _h_other_monitor, needs=("desktop",)),
    Command("dictate",
            _m_exact("dictate", "start dictation", "dictation mode"),
            _h_dictate),
    Command("trigger",
            _m_re(r"when (.+?)(?:,?\s*(?:notify me|tell me|alert me|"
                  r"let me know))"),
            _h_trigger, needs=("workflows",)),
    Command("transform case",
            _m_contains("make that uppercase", "uppercase that",
                        "make that lowercase", "lowercase that"),
            _h_transform_case),
    # answer-question fallback runs BEFORE the QUICK_COMMANDS table (3446);
    # it now answers ip / uptime / battery only (spec 5.2).
    Command("answer question", _m_answer_question, _h_answer_question,
            needs=("context",)),
    Command("quick command", _m_quick_command, _h_quick_command),
    Command("remind me", _REMIND_RX.match, _h_remind_me),
    # ambient (jarvis/quiet.py): before "free" so "I'm free until seven"
    # does not read as a DND request, and status before the hours setter.
    Command("quiet status", _QUIET_STATUS_RX.match, _h_quiet_status, needs=("quiet",)),
    Command("quiet hours off", _QUIET_HOURS_OFF_RX.match, _h_quiet_hours_off,
            needs=("quiet",)),
    Command("quiet hours", _QUIET_HOURS_RX.match, _h_quiet_hours, needs=("quiet",)),
    Command("do not disturb", _DND_RX.match, _h_dnd, needs=("quiet",)),
    Command("free", _FREE_RX.match, _h_free, needs=("quiet",)),
]

# The assistant's Tier 1 without the "jarvis" prefix (jarvis mode): the
# same handlers, in the same order, run right before the router so that a
# typed "timer for 5 minutes" is instant and never a model round trip.
ASSISTANT_TIER1: list[Command] = [
    cmd for cmd in REGISTRY
    if cmd.name in ("explain document", "quiz", "review flashcards", "stop quiz",
                    "focus start", "focus left", "focus end", "lecture notes",
                    "timer", "alarm", "list schedule", "cancel schedule",
                    "briefing", "preview", "week", "briefing section", "verbosity",
                    "last mail", "diagnostics", "next exam", "greeting", "day review",
                    "todo done", "todo add",
                    "todo list",
                    # named lists: spoken in the aisle and read back over
                    # SSH from the phone, neither with a wake-word prefix
                    "list add", "list read", "list strike", "list clear",
                    "lists",
                    "take note", "show notes", "answer question", "remind me",
                    # long-term memory, the people book and the day recap
                    # answer without the wake-word prefix too: unprefixed
                    # "remember that ..." used to reach the router and the
                    # notes tool instead of the memory it was pitched for
                    "person", "remember", "recall", "who is", "recap",
                    "quiet status", "quiet hours off", "quiet hours", "do not disturb",
                    "free",
                    "standup", "gpu reclaim", "gpu lend",
                    "log triage", "slow turn",
                    # the hotword consumes the wake word, so spoken text never
                    # reaches the prefixed registry: without this the router
                    # would hand Claude the bare words "fix what i copied".
                    "clip to claude",
                    "read control")
]


def _call_manager(fn, args: dict):
    """Call a session-manager method with the arguments it accepts.

    The router hands over an argument bag (`resume` carries `utterance`,
    `when` and `name`); the spec's signatures are narrower
    (`resume(utterance)`). Filtering by signature keeps the call to ONE
    attempt — a retry would risk submitting the same task twice when the
    method itself raised the TypeError.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(**args)
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return fn(**args)
    return fn(**{k: v for k, v in args.items() if k in params})


# ------------------------------------------------------------------
# Commander — routing pipeline
# ------------------------------------------------------------------
# Spoken replies to "Was that for me?" are read ONLY by this function and are
# never routed as a command. An ambiguous reply that went through the normal
# path could classify as uncertain again and the prompts would ping-pong, so
# anything not clearly recognised returns None and the on-screen card waits for
# a click instead. Erring toward "didn't catch that" costs a click; erring the
# other way runs something the user never asked for.
_YES_WORDS = ("yes", "yeah", "yep", "yup", "sure", "correct", "affirmative",
              "aye", "certainly")
# "for you" / "that was" are deliberately absent: they are substrings of
# "not for you" and "that wasn't", so they would fight the negatives.
# "was for you" is safe: "wasn't for you" / "was not for you" do not
# contain it, and the negatives still win a contradictory reply.
_YES_PHRASES = ("go ahead", "please do", "do it", "was for you", "meant for you",
                "asking you")
_NO_WORDS = ("no", "nope", "nah", "negative", "wasnt", "wasn't")
_NO_PHRASES = ("never mind", "nevermind", "ignore that", "forget it",
               "not for you", "not you", "talking to")
_UNSURE = ("not sure", "unsure", "dont know", "don't know", "dunno", "maybe",
           "no idea")


def parse_yes_no(text):
    """Read a spoken yes/no. None when the reply is neither.

    Whole-word matching only: substring matching would make "nothing",
    "north" and "you know" all mean no.
    """
    if not text:
        return None
    lowered = re.sub(r"[^a-z0-9\s']+", " ", str(text).lower())
    words = lowered.split()
    if not words:
        return None
    padded = " " + " ".join(words) + " "
    if any(p in padded for p in _UNSURE):
        return None
    # A reply to a yes/no question is short. A long sentence that merely
    # contains "no" ("she said no way lol haha dude") is overheard speech,
    # not an answer -- unless it actually opens with yes or no.
    if len(words) > 6 and words[0] not in _YES_WORDS \
            and words[0] not in _NO_WORDS:
        return None
    yes = any(w in _YES_WORDS for w in words) or \
        any(p in padded for p in _YES_PHRASES)
    no = any(w in _NO_WORDS for w in words) or \
        any(p in padded for p in _NO_PHRASES)
    if yes == no:                      # neither, or a contradictory reply
        return None
    return yes


# ------------------------------------------------------------------
# Route short-cuts: the router already named the tool (2026-08-30)
# ------------------------------------------------------------------
# For "what's on my calendar tomorrow?" the router says local:calendar and
# the brain then spends a model turn choosing get_calendar (0.98-1.45 s
# live, 2026-08-29) before a second turn renders the result. When the cue
# class names the tool AND the arguments are in the sentence, the call is
# forced (brain.chat force_tool/force_args, the path get_briefing and
# get_mail already use) and only the render turn runs. The render turn is
# kept on purpose: tool text can carry a stranger's words (calendar titles)
# and brain.py never speaks it raw. Deliberately narrow -- a write verb, a
# second clause or a city the regex is not sure of falls back to the full
# loop, which is slower but cannot be wrong in a new way.
_CAL_READ_RX = re.compile(
    r"\b(?:what(?:'s| is|s| do i have| have i got)?|anything|any|do i have|"
    r"have i got|is there|are there|show me|read me|check|list|tell me|"
    r"when(?:'s| is)?|how many)\b.*"
    r"\b(?:calendar|schedule|agenda|meetings?|appointments?|events?|plans?)\b"
    r"|\bon (?:my |the )?(?:calendar|schedule|agenda)\b"
    r"|\bnext (?:meeting|appointment|event)\b", re.I)
# A write: the verb leads the clause ("schedule a meeting", "can you add
# ...") or is unambiguous anywhere. "schedule" the noun ("what's on my
# schedule") must not count, so the noun-like verbs are start-anchored.
_CAL_WRITE_RX = re.compile(
    r"^(?:(?:please|jarvis|can you|could you|would you|will you|go ahead and)"
    r"[,\s]+)*(?:schedule|book|put|add|create|set up|make|move|reschedule|"
    r"change|edit|update|block|pencil)\b"
    r"|\b(?:book|reschedule|cancel|delete|remove|rename|invite|postpone|clear)\b",
    re.I)
# A second clause means a second intent: the full loop handles both.
_CLAUSE_RX = re.compile(r",\s*and\b|\band\b|\balso\b|\bthen\b|\bplus\b|"
                        r"\bas well as\b|;", re.I)
# "in London", "at Salt Lake City", "for Paris": a capitalised run after a
# place preposition. Whisper capitalises the places it knows; a lowercase
# candidate ("in london", typed) is NOT trusted as a city and is left to
# the model rather than geocoded blind.
_PLACE_RX = re.compile(
    r"\b(?:in|at|for|over in|out in)\s+"
    r"(?P<place>[A-Za-z][\w'.-]*(?:\s+[A-Z][\w'.-]*)*)")
_NOT_A_PLACE = {
    "the", "a", "an", "my", "our", "your", "his", "her", "their", "this", "that",
    "these", "those", "here", "there", "home", "town", "work", "school", "bed",
    "today", "tomorrow", "tonight", "now", "noon", "midnight", "morning",
    "afternoon", "evening", "night", "lunch", "dinner", "breakfast", "next",
    "last", "least", "all", "me", "us", "him", "them", "it", "present",
    "general", "celsius", "fahrenheit", "degrees", "case", "once", "about",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "christmas", "easter",
    "weekend", "week", "month", "year", "hour", "minute", "moment", "while",
    "sir", "jarvis", "hunter",
}


def place_in(text: str) -> Optional[str]:
    """The city named after in/at/for, as the tool wants it.

    "" when the sentence names no place (home), the city when it is
    capitalised, None when a lowercase candidate makes the answer unsure
    ("in london" typed, "in a while") -- the caller then does not force."""
    for m in _PLACE_RX.finditer(text or ""):
        place = m.group("place").strip(" .,!?")
        head = place.split()[0].lower().strip(".") if place else ""
        if not head or head in _NOT_A_PLACE:
            continue
        if place[0].isupper():
            return place
        return None
    return ""


def calendar_range(text: str) -> str:
    """coerce_range over the whole utterance, with one repair: "when's my
    meeting" carries no day, and today is the wrong default for a "when"
    question -- the first upcoming event is."""
    from jarvis.tools.calendar import coerce_range
    rng = coerce_range(text)
    t = (text or "").lower()
    if rng == "today" and re.match(r"^\s*when", t) and not re.search(
            r"\btoday\b|tonight|this (?:morning|afternoon|evening)|later", t):
        return "next"
    return rng


def weather_when(text: str) -> str:
    """The weather tool's when= from the utterance. Explicit day words win;
    a bare "forecast" is tomorrow-and-on, not this minute."""
    t = (text or "").lower()
    if "tomorrow" in t:
        return "tomorrow"
    if re.search(r"\bweek\b|7 day|seven day|weekend|next few days", t):
        return "week"
    if re.search(r"\btoday\b|tonight|this (?:evening|afternoon|morning)|"
                 r"\blater\b|rest of the day", t):
        return "today"
    if "forecast" in t:
        return "today"
    return "now"


def forced_call(reason: str, text: str) -> Optional[tuple]:
    """(tool, args) when the router's reason names the tool and the
    utterance carries its arguments; None to run the full tool loop."""
    t = (text or "").strip()
    if not t or _CLAUSE_RX.search(t):
        return None
    if reason == "local:calendar":
        if not _CAL_READ_RX.search(t) or _CAL_WRITE_RX.search(t):
            return None
        return "get_calendar", {"range": calendar_range(t)}
    if reason == "local:weather":
        place = place_in(t)
        if place is None:
            return None
        return "get_weather", {"when": weather_when(t), "location": place}
    if reason == "local:clock":
        # The home clock never gets here (Tier-1 _h_clock); what does is
        # "the time in <city>" -- and "what year is it", which has no
        # city and is left to the model.
        place = place_in(t)
        if not place:
            return None
        return "get_time", {"location": place}
    return None


# ------------------------------------------------------------------
# Corrections: "no, I said ..." (2026-08-30)
# ------------------------------------------------------------------
# A misheard transcript used to stay in the model's window paired with
# the reply it earned, and the only recourse was to wake Jarvis and say
# the whole thing again. The correction re-dispatches the meant text with
# the classifier bypassed (a correction is addressed to Jarvis by
# construction), cuts the reply in flight, forgets the misheard exchange
# and logs the (heard, meant) pair for a later vocab tune. Matched on the
# raw text ahead of the yes/no stages, which would eat "no, I said X" as
# a plain decline.
CORRECTION_WINDOW_S = 60.0
_CORRECTION_RX = re.compile(
    r"^(?:(?:no|nope|nah)[,.!]?\s+)?"
    r"(?:i\s+(?:said|meant|mean|actually said|was saying)|"
    r"what i (?:said|meant) was|that'?s not what i said[,.]?\s*(?:i said)?)"
    r"(?!\s+(?:to|it|that|you|nothing|so|this)\b)"
    r"[,:]?\s+(?P<meant>.+?)[.!?]*$", re.I)
# "not the terminal, the calendar": the comma is load-bearing; without it
# "not now" and "not really" are plain sentences.
_CORRECTION_NOT_RX = re.compile(
    r"^(?:no[,.!]?\s+)?not\s+(?P<heard>[^,]{1,60}),\s*(?P<meant>.+?)[.!?]*$", re.I)


def correction_kind(text: str) -> Optional[str]:
    """The meant text when the utterance is a correction, else None."""
    t = (text or "").strip()
    m = _CORRECTION_RX.match(t) or _CORRECTION_NOT_RX.match(t)
    if not m:
        return None
    meant = m.group("meant").strip(" ,")
    return meant or None


# Words that are capitalised for reasons other than being a name.
_VOCAB_STOP = {
    "i", "i'm", "i'll", "i've", "i'd", "jarvis", "the", "what", "what's", "how",
    "when", "where", "who", "why", "which", "can", "could", "would", "will",
    "please", "yes", "no", "okay", "ok", "set", "play", "remind", "open",
    "close", "call", "tell", "show", "read", "check", "turn", "start", "stop",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "sir", "hunter",
}
VOCAB_CHAR_CAP = 900      # Whisper's initial_prompt budget is ~224 tokens


def new_vocab_words(heard: str, meant: str) -> list:
    """Capitalised words in the correction that the transcript lacked --
    the names Whisper got wrong. Conservative by design: casing on a
    misheard name is itself a guess, so this is opt-in."""
    heard_l = {w.lower() for w in re.findall(r"[\w'-]+", heard or "")}
    out, seen = [], set()
    for w in re.findall(r"[A-Za-z][\w'-]+", meant or ""):
        key = w.lower()
        if not w[0].isupper() or len(w) < 3:
            continue
        if key in heard_l or key in _VOCAB_STOP or key in seen:
            continue
        seen.add(key)
        out.append(w)
    return out


# ------------------------------------------------------------------
# Voice feedback: "that was for you" / "that wasn't for you" (2026-08-30)
# ------------------------------------------------------------------
# The only learning signal used to be the YES/NO card on UNCERTAIN
# utterances; a command the classifier dropped as background chat (NO)
# was silent and unrecoverable, and a confident misroute was never
# recorded. These phrases label the LAST turn in the classifier's own log
# (IntentClassifier.log_feedback -- the log classify() actually reads) and
# re-run a dropped command. Matched ahead of the classifier: three-word
# feedback would itself be classified NO and dropped.
FEEDBACK_YES_WINDOW_S = 20.0    # after a dropped command: no reply marks the time
FEEDBACK_NO_WINDOW_S = 60.0     # after a reply he did not ask for
_FB_TAIL = r"(?:[,]?\s*(?:jarvis|sir|please|thanks))*[.!\s]*$"
_FEEDBACK_YES_RX = re.compile(
    r"^(?:(?:yes|yeah|jarvis)[,.!]?\s+)*"
    r"(?:that (?:was|is|one was) (?:for|to|meant for|aimed at|directed at) you|"
    r"i (?:was|am) (?:talking|speaking) to you|i (?:was|am) asking you|"
    r"that was (?:a|an) (?:command|question|request)(?: for you)?|"
    r"i meant you|that was you|it was for you)" + _FB_TAIL, re.I)
_FEEDBACK_NO_RX = re.compile(
    r"^(?:(?:no|nope|jarvis)[,.!]?\s+)*"
    r"(?:that (?:wasn't|was not|isn't|is not|one wasn't) (?:for|to|meant for|"
    r"aimed at|directed at) you|i (?:wasn't|was not) (?:talking|speaking) to you|"
    r"(?:i was )?(?:talking|speaking) to (?:someone|somebody) else|"
    r"not (?:for )?you|that wasn't you|i wasn't asking you|"
    r"that (?:wasn't|was not) (?:a|an) (?:command|question|request))" + _FB_TAIL,
    re.I)


def feedback_kind(text: str) -> Optional[bool]:
    """True = "that was for you", False = "that wasn't for you", else None."""
    t = (text or "").strip()
    if _FEEDBACK_NO_RX.match(t):
        return False
    if _FEEDBACK_YES_RX.match(t):
        return True
    return None


# ------------------------------------------------------------------
# Spoken undo: "scratch that" (2026-08-30)
# ------------------------------------------------------------------
# The phrase mapped to delete_last_sentence, a DICTATION action, so after
# "set an alarm for six" a spoken "scratch that" typed editing keys at
# whatever window had focus and the alarm stood. It now runs the last
# turn's undo closure (CommandResult.undo) when there is one within the
# window; with nothing to undo it returns None and the old typing meaning
# survives untouched -- including "delete that", which is left alone.
# The window is the correction window: an undo older than that is more
# likely a stray transcript than a change of mind.
UNDO_WINDOW_S = 60.0
_UNDO_RX = re.compile(
    r"^(?:(?:no|nope)[,.!]?\s+)?"
    r"(?:scratch|undo|cancel|forget|belay|take back)\s+"
    r"(?:that last one|the last(?: one| thing)?|that|it|this)"
    r"|^(?:scratch|undo|belay) that"
    r"|^undo(?: the)?(?: last)?(?: one| thing| action)?"
    r"|^take that back"
    r"|^(?:on second thought[s]?|actually)[,.]?\s+(?:scratch|undo|cancel) that",
    re.I)
_UNDO_TAIL_RX = re.compile(r"^[\s,.!]*(?:please|jarvis|sir|instead)?[\s,.!]*$", re.I)


def undo_kind(text: str) -> bool:
    """True when the utterance asks for the last action to be taken back."""
    t = (text or "").strip()
    m = _UNDO_RX.match(t)
    return bool(m) and bool(_UNDO_TAIL_RX.match(t[m.end():]))


@dataclass
class LastTurn:
    text: str
    status: str
    ts: float


class Commander:
    """Routes a user utterance (voice or typed) through the V3 pipeline:

    dictation → desktop (jarvis-prefixed) → registry (jarvis-prefixed) →
    intent classification (voice only) → voice-command substitution /
    targeting / actions → jarvis-mode brain → fallback type-to-window.

    Ported from _transcribe_worker (2559-2680) + _on_transcription
    (2820-2940). Speaker verification and the confidence gate stay in the
    transcription pipeline; handle() receives accepted text only.
    """

    # Class-level defaults for the per-turn state __init__ sets: a test
    # (tests/test_custom_phrases.py) builds a Commander with __new__ and
    # fills in only what it needs.
    claim_uncertain: Optional[Callable[[bool], bool]] = None
    _last_turn: Optional[LastTurn] = None
    _last_undo: Optional[tuple] = None
    _confidence: Optional[float] = None
    _pending_destructive: Optional[tuple] = None

    # One turn at a time: handle() mutates per-turn fields (_confidence,
    # _raw_text, _last_turn, _pending_*) and is entered from the voice
    # worker thread, cmdsock client threads and Discord concurrently.
    # Class-level default so a test commander built via object.__new__
    # still has one; RLock because a handler may re-enter handle()
    # (lecture recovery, corrections re-dispatch).
    _turn_lock = threading.RLock()

    def __init__(self, services):
        self.services = services
        self._turn_lock = threading.RLock()
        # (closure, monotonic stamp) from the last turn that created
        # something; consumed by "scratch that".
        self._last_undo: Optional[tuple] = None
        self.intent = IntentClassifier()
        self.dictation = False
        # Lecture-note capture (jarvis/lecture.py): the course name while
        # notes are open, else None. Checked right after dictation.
        self.lecture_course: Optional[str] = None
        self._lecture = None
        self._raw_text = ""
        # Set when the Claude manager refuses an out-of-project task and
        # offers the terminal; the next "yes" opens it (spec 7 / OUTSIDE_LINE).
        self._pending_terminal_slug = ""
        # Quiz mode: the QuizSession whose open question the next utterance
        # answers (jarvis.tools.quiz); the flashcard store behind it is
        # built on first use. The document last explained, for "read it to
        # me": (Path, epoch seconds).
        self._pending_quiz = None
        self._flashcards = None
        self._last_document = None
        # UI hook for uncertain intent ("Was this for me?"); wired by the
        # main window. Falls back to a warn Status event.
        self.on_uncertain: Optional[Callable[[str], None]] = None
        # App hook: a spoken "that was for you" answers the open card too
        # (claim_uncertain(yes) -> bool, whether a card was waiting).
        self.claim_uncertain: Optional[Callable[[bool], bool]] = None
        # The last utterance handled, for "no, I said ..." and "that was
        # for you"; the app's _last_user_text is not visible from here.
        self._last_turn: Optional[LastTurn] = None
        # Whisper avg_logprob of the utterance being handled (None when
        # typed / unknown); read by the destructive read-back.
        self._confidence: Optional[float] = None
        # A read-back waiting for a yes: (run, spoken line, stamp).
        self._pending_destructive: Optional[tuple] = None

    # -- service access ------------------------------------------------
    def _svc(self, name: str):
        return getattr(self.services, name, None)

    def _bg(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def _speak(self, text: str):
        """Speak via the TTS service when talk-back is enabled."""
        if not CONFIG.talkback or not text:
            return
        tts = self._svc("tts")
        if tts is None:
            return
        try:
            tts.speak(text)
        except Exception:
            log.exception("tts speak failed")

    def _type_raw(self, text: str):
        """Type raw text into the active window (monolith 3176-3181)."""
        self._bg(lambda: subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "5", text],
            timeout=10, capture_output=True,
        ))

    # -- public entry --------------------------------------------------
    def handle(self, text: str, source: str = "voice",
               confidence: Optional[float] = None) -> CommandResult:
        """Route one utterance. ``confidence`` is the transcript's Whisper
        avg_logprob when the app has one (voice); every other caller
        leaves it unset."""
        text = (text or "").strip()
        if not text:
            return CommandResult(handled=False, status="No speech detected")
        with self._turn_lock:
            self._confidence = confidence
            result = self._handle_inner(text, source)
            # A correction / re-run answers a different utterance: THAT is
            # the last turn, so a second "no, I said ..." corrects the
            # right text.
            self._last_turn = LastTurn(
                getattr(result, "corrected", None) or text,
                result.status or "", time.monotonic())
            undo = getattr(result, "undo", None)
            if undo is not None:
                self._last_undo = (undo, time.monotonic())
        return result

    def shaky_transcript(self) -> bool:
        """The utterance being handled scraped in under confirm.shaky_logprob."""
        conf = self._confidence
        if conf is None:
            return False
        try:
            floor = float(_assistant_get(self, "confirm.shaky_logprob", -0.7))
        except (TypeError, ValueError):
            floor = -0.7
        return conf < floor

    def stash_destructive(self, run: Callable[[], CommandResult], line: str):
        """A handler read an action back instead of doing it; the next yes
        runs it (``_try_destructive_confirm``)."""
        self._pending_destructive = (run, line, time.monotonic())

    def _handle_inner(self, text: str, source: str, gate: bool = True) -> CommandResult:
        log.info("handle %r source=%s", text, source)
        self._raw_text = text          # original casing for handlers that need it

        # 1. A ringing alarm owns the next words (spec 5.2 a) -- checked
        #    BEFORE the sticky modes: dictation and lecture notes swallow
        #    every utterance, so a ringing alarm could not be silenced by
        #    voice until "end notes". _try_ringing returns None unless an
        #    alarm is actually ringing AND the words are stop/snooze, so
        #    note lines and dictated text still reach their handlers.
        res = self._try_ringing(text)
        if res is not None:
            return res
        # 1a. Dictation mode — type directly, don't route (2611-2630)
        if self.dictation:
            return self._handle_dictation(text)
        # 1b. Lecture notes open: file it, unless it is "end notes".
        if getattr(self, "lecture_course", None):
            return self._handle_lecture(text, source)
        # 2b. "No, I said X": ahead of every yes/no stage, which would read
        #     it as a bare decline (parse_yes_no: any sentence opening with
        #     "no" is a no).
        res = self._try_correction(text, source)
        if res is not None:
            return res
        # 3. A pending permission question owns yes / no (spec 5.2 b).
        res = self._try_approval(text, source)
        if res is not None:
            return res
        # 3a'. A quiz question is on the table: this is the answer (or
        #      "skip" / "stop the quiz").
        res = self._try_quiz_answer(text)
        if res is not None:
            return res
        # 3b. Claude offered the terminal after refusing an outside-dir
        #     task: "yes" / "open it" opens it.
        res = self._try_terminal_offer(text)
        if res is not None:
            return res
        # 3c. add_event read an interpretation back; a plain yes means THAT
        #     event, not a new command.
        res = self._try_event_confirm(text)
        if res is not None:
            return res
        # 3d. The evening preview asked "Shall I wake you at seven?"; a
        #     plain yes sets THAT alarm rather than becoming a new command.
        res = self._try_alarm_offer(text)
        if res is not None:
            return res
        # 3e. A destructive action was read back ("Cancel all three alarms,
        #     sir?"): a plain yes runs it, anything else drops it.
        res = self._try_destructive_confirm(text)
        if res is not None:
            return res
        # 4. A pending router question: resolve it and dispatch the
        #    remembered utterance (spec 5.2 c).
        res = self._try_router_answer(text)
        if res is not None:
            return res

        cmd_text = strip_jarvis_prefix(text)

        # 3e. "That was for you" / "that wasn't for you": label the last
        #     turn for the classifier and re-run a dropped command. Before
        #     the classifier, which would drop the feedback itself.
        res = self._try_feedback(text, cmd_text, source)
        if res is not None:
            return res
        # 4b. An active read-aloud owns the transport words, the way a
        #     ringing alarm owns the next words. This runs BEFORE the
        #     desktop chains ("go back" = previous window), the media-key
        #     substitution ("pause" -> __ACTION__media_pause) and the intent
        #     gate (which calls one-word phrases background chat) -- each of
        #     which would otherwise eat "skip" / "back" / "pause" / "go on"
        #     before the reader saw them. When nothing is being read the
        #     handler returns None and every one of those keeps its meaning.
        rc = read_control_kind(cmd_text if cmd_text is not None else text)
        if rc and self._svc("reader") is not None:
            res = _h_read_control(self, text, rc)
            if res is not None:
                return res

        # 3a. User-defined phrases from assistant.json, ahead of the built-ins
        #     so a personal shortcut can shadow one -- and checked on the RAW
        #     text as well as the addressed form. The hotword consumes the wake
        #     word, so "Jarvis, drop my needle" arrives here as "drop my
        #     needle" with no prefix left to strip; that sends it to the
        #     classifier below, which calls short phrases background chat and
        #     drops them silently. A phrase the user configured by hand is by
        #     definition addressed to Jarvis and must not be subject to a guess.
        res = self._try_custom_phrase(cmd_text if cmd_text is not None else text)
        if res is not None:
            return res
        # 3a'. "Scratch that": take back the last thing he had me create.
        #      After the yes/no stages (a pending read-back owns "cancel
        #      that") and after custom phrases, which shadow built-ins;
        #      before the intent gate, which calls a two-word phrase
        #      background chat and drops it silently.
        res = self._try_undo(cmd_text if cmd_text is not None else text)
        if res is not None:
            return res

        if cmd_text is not None:
            # 2. Desktop control chains (2632-2635 → 3548-3584)
            if self._try_desktop(cmd_text):
                return CommandResult(handled=True, status="Desktop command",
                                     done=False)
            # 3. Quick/registry commands (2637-2640 → 3036-3485)
            res = self._try_registry(cmd_text)
            if res is not None:
                return res

        # 4. Intent classification — voice only; typed text is deliberate
        #    (2642-2653). "discord" and any other channel count as typed,
        #    and so does anything spoken with the "jarvis" address: the
        #    gate exists to drop background chat, and "Jarvis, cancel" is
        #    not background chat (the classifier calls every one- or
        #    two-word phrase NO, which used to eat the router actions).
        # A web cue is addressed to Jarvis by construction, like a custom
        # phrase: the classifier called "look up who won the last race"
        # uncertain and asked "Was that for me?" (live, 2026-08-29 23:45).
        # 4-pre. A Tier-1 assistant command that matches outright is by
        #    definition addressed to Jarvis -- the same rule as a custom
        #    phrase. Without this the classifier guessed on "standup" and
        #    "review my flashcards" and dropped them as background chat:
        #    the third recurrence of the silent-drop bug that the
        #    vocabulary patches of 08-27 and 08-30 each fixed once.
        if gate and source == "voice" and cmd_text is None:
            name = self._match_assistant(text)
            if name:
                log.info("tier-1 match %r bypasses the intent gate: %r",
                         name, text)
                gate = False
        if gate and source == "voice" and cmd_text is None \
                and not WEB_CUE_RX.search(text):
            intent, conf = self.intent.classify(text)
            if intent == IntentClassifier.NO:
                log.info("Ignored (background chat, conf=%.2f): %r",
                         conf, text)
                return CommandResult(handled=True,
                                     status="Ignored (background chat)")
            if intent == IntentClassifier.UNCERTAIN:
                log.info("Uncertain intent (conf=%.2f): %r", conf, text)
                self._prompt_uncertain(text)
                return CommandResult(handled=True, status="Was that for me?",
                                     done=False)

        return self._route_text(text)

    def resolve_uncertain(self, text: str, yes: bool) -> CommandResult:
        """UI feedback for the 'Was this for me?' prompt."""
        with self._turn_lock:
            return self._resolve_uncertain_locked(text, yes)

    def _resolve_uncertain_locked(self, text: str, yes: bool) -> CommandResult:
        self.intent.log_feedback(text, yes)
        self._feedback_line(text, "Was that for me?", yes, "card")
        if yes:
            res = self._route_text(text)
            self._last_turn = LastTurn(text, res.status or "", time.monotonic())
            if getattr(res, "undo", None) is not None:
                self._last_undo = (res.undo, time.monotonic())
            return res
        return CommandResult(handled=True, status="Discarded")

    # -- corrections, feedback, read-back ---------------------------------
    def _try_correction(self, text: str, source: str) -> Optional[CommandResult]:
        meant = correction_kind(strip_address(text))
        if not meant:
            return None
        prev = self._last_turn
        if source == "voice":
            # By voice the wake word is consumed before the text arrives,
            # so "addressed" cannot be read off a prefix. A correction
            # presupposes a turn to correct: without a recent one, "not
            # that one, the other one" is the room talking.
            if prev is None or time.monotonic() - prev.ts > CORRECTION_WINDOW_S:
                return None
        heard = prev.text if prev is not None else ""
        log.info("correction: heard %r -> meant %r", heard, meant)
        # Cut the reply in flight: speech now, the model job too (its
        # generation goes stale, so its result is dropped, not remembered).
        _cut_speech(self)
        brain = self._svc("brain")
        cancel = getattr(brain, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                log.exception("brain cancel failed")
        conv = self._svc("conversation")
        if conv is not None and heard:
            try:
                conv.forget_exchange(heard)
            except Exception:
                log.exception("forget_exchange failed")
        memory = self._svc("memory")
        if memory is not None and hasattr(memory, "log_correction"):
            try:
                memory.log_correction(heard, meant)
            except Exception:
                log.exception("log_correction failed")
        self._learn_vocab(heard, meant)
        res = self._handle_inner(meant, source, gate=False)
        res.corrected = meant
        return res

    def _learn_vocab(self, heard: str, meant: str):
        """Opt-in (corrections.learn_vocab): new capitalised words from the
        correction join the Whisper vocabulary prompt, so the next attempt
        decodes the name right."""
        if not _assistant_get(self, "corrections.learn_vocab", False):
            return
        words = new_vocab_words(heard, meant)
        if not words:
            return
        try:
            from jarvis.transcriber import load_vocab, save_vocab
            vocab = load_vocab() or ""
            have = {w.strip().lower() for w in re.split(r"[,\n]", vocab)}
            add = [w for w in words if w.lower() not in have]
            if not add:
                return
            text = (vocab.rstrip(", \n") + ", " if vocab.strip() else "") + ", ".join(add)
            if len(text) > VOCAB_CHAR_CAP:
                log.info("vocab learn skipped: prompt would exceed %d chars", VOCAB_CHAR_CAP)
                return
            save_vocab(text)
            log.info("vocab learned: %s", add)
        except Exception:
            log.exception("vocab learn failed")

    FEEDBACK_LOG = PATHS.MEMORY_DIR / "feedback.jsonl"

    def _feedback_line(self, prev_text: str, prev_status: str, label: bool, how: str):
        """One JSON line per label: the audit trail the classifier's own
        log (text + label only) cannot carry."""
        try:
            self.FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(self.FEEDBACK_LOG, "a") as fh:
                fh.write(json.dumps({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "text": prev_text[:200], "prev_status": prev_status[:60],
                    "label": "yes" if label else "no", "how": how}) + "\n")
            # Bounded: an append-only audit file on a box that runs for
            # months. Past ~256 KB the newest thousand lines are kept.
            if self.FEEDBACK_LOG.stat().st_size > 262144:
                lines = self.FEEDBACK_LOG.read_text().splitlines()[-1000:]
                self.FEEDBACK_LOG.write_text("\n".join(lines) + "\n")
        except Exception:
            log.exception("feedback log write failed")

    def _try_feedback(self, text: str, cmd_text: Optional[str],
                      source: str) -> Optional[CommandResult]:
        said = cmd_text if cmd_text is not None else text
        label = feedback_kind(said)
        if label is None:
            return None
        prev = self._last_turn
        if prev is None:
            return CommandResult(handled=True, reply="I have nothing recent to go on, sir.",
                                 speak=True, status="Feedback: no last turn")
        status = prev.status or ""
        dropped = status.startswith("Ignored")
        asked = status.startswith("Was that for me")
        window = FEEDBACK_YES_WINDOW_S if (dropped or asked) else FEEDBACK_NO_WINDOW_S
        if time.monotonic() - prev.ts > window:
            return CommandResult(handled=True,
                                 reply="I'm not sure which one you mean, sir.",
                                 speak=True, status="Feedback: stale")
        if asked:
            # The card is still up: settle it the same way a click would.
            claim = self.claim_uncertain
            if claim is not None:
                try:
                    claim(label)
                except Exception:
                    log.exception("claim_uncertain failed")
            res = self.resolve_uncertain(prev.text, label)
            if label:
                res.corrected = prev.text
            elif not res.reply:
                res.reply, res.speak = "Very good, sir.", True
            return res
        self.intent.log_feedback(prev.text, label)
        self._feedback_line(prev.text, status, label, "spoken")
        if label and dropped:
            log.info("feedback: re-running dropped %r", prev.text)
            res = self._handle_inner(prev.text, source, gate=False)
            res.corrected = prev.text
            return res
        if not label and not dropped:
            # He did not ask: stop talking and forget the exchange.
            _cut_speech(self)
            conv = self._svc("conversation")
            if conv is not None:
                try:
                    conv.forget_exchange(prev.text)
                except Exception:
                    log.exception("forget_exchange failed")
            return CommandResult(handled=True, reply="My mistake, sir.", speak=True,
                                 status="Feedback: not for me")
        return CommandResult(handled=True, reply="Very good, sir.", speak=True,
                             status="Feedback: noted")

    def _try_undo(self, text: str) -> Optional[CommandResult]:
        """Run the last turn's undo closure. None -- so the phrase keeps
        its dictation meaning -- when it is not an undo, when nothing
        undoable happened, or when what did happen has gone stale."""
        if not undo_kind(text):
            return None
        pend, self._last_undo = self._last_undo, None
        if pend is None:
            log.info("undo asked for with nothing to undo: %r", text)
            return None
        if time.monotonic() - pend[1] > UNDO_WINDOW_S:
            log.info("undo expired (%.0fs)", time.monotonic() - pend[1])
            return None
        try:
            line = pend[0]()
        except Exception:
            log.exception("undo failed")
            return CommandResult(handled=True, speak=True, status="Undo failed",
                                 reply="I couldn't take that back, sir.")
        if not line:
            return None
        return CommandResult(handled=True, reply=str(line), speak=True,
                             status="Undone")

    def _try_destructive_confirm(self, text: str) -> Optional[CommandResult]:
        """Resolve a read-back ("Cancel all three alarms, sir?").

        As with a calendar add, anything that is not a clear yes or no
        DROPS the offer: changing the subject is not consent, and a stale
        offer would attach the next stray "yes" to an old cancel. An offer
        older than DESTRUCTIVE_TTL_S is dropped even on a yes."""
        pend, self._pending_destructive = self._pending_destructive, None
        notes = self._svc("notes")
        npend = getattr(notes, "pending_clear", None) if notes is not None else None
        if not isinstance(npend, dict):
            npend = None
        elif notes is not None:
            try:
                notes.pending_clear = None
            except Exception:
                log.debug("notes.pending_clear reset failed", exc_info=True)
        now = time.monotonic()
        if pend is not None and now - pend[2] > DESTRUCTIVE_TTL_S:
            log.info("read-back expired: %r", pend[1])
            pend = None
        if npend is not None and time.time() - float(npend.get("ts") or 0) > DESTRUCTIVE_TTL_S:
            npend = None
        if pend is None and npend is None:
            return None
        answer = parse_yes_no(text)
        if answer is None:
            return None
        if not answer:
            return CommandResult(handled=True, reply="Very good, sir.", speak=True,
                                 status="Dropped")
        if pend is not None:
            try:
                return pend[0]()
            except Exception:
                log.exception("confirmed action failed")
                return CommandResult(handled=True, reply="I couldn't manage that, sir.",
                                     speak=True, status="error")
        kind = str(npend.get("kind") or "todo")
        try:
            removed = notes.remove(kind, "all")
        except Exception:
            log.exception("notes clear failed")
            return CommandResult(handled=True, reply="I couldn't clear that, sir.",
                                 speak=True, status="error")
        n = len(removed) if removed else 0
        lname = notes_mod.list_name(kind)
        line = "All cleared, sir." if n else "Nothing to clear, sir."
        if lname:
            line = f"The {lname} list is clear, sir." if n else \
                f"Your {lname} list was empty already, sir."
        where = f"from the {lname} list" if lname else \
            f"{kind}{'s' if n != 1 else ''}"
        return CommandResult(handled=True, reply=line, speak=True,
                             status=f"Cleared {n} {where}")

    # -- pipeline stages -----------------------------------------------
    def _handle_dictation(self, text: str) -> CommandResult:
        if "end dictation" in text.lower():
            self.dictation = False
            log.info("Dictation mode ended")
            return CommandResult(handled=True, reply="Dictation mode: OFF")
        if CONFIG.auto_type:
            self._type_raw(text + " ")
        return CommandResult(handled=True, reply=text, status="Dictating")

    def _handle_lecture(self, text: str, source: str = "voice") -> CommandResult:
        body = strip_address(text).strip()
        if _LECTURE_END_RX.match(body.lower()):
            capture, self._lecture = self._lecture, None
            course, self.lecture_course = self.lecture_course, None
            line = capture.close() if capture is not None else lecture_mod.END_NONE_LINE
            log.info("lecture notes: closed for %r", course)
            # The file just grew: queue a reindex so ask_docs sees it. The
            # docs tool parks its index on services.docs_index.
            index = self._svc("docs_index")
            kick = getattr(index, "kick", None)
            if callable(kick):
                try:
                    kick()
                except Exception:
                    log.exception("lecture notes: docs reindex kick failed")
            return CommandResult(handled=True, reply=line, speak=True,
                                 status="Notes closed")
        if self._lecture is None:          # flag without a file: recover
            self.lecture_course = None
            # _handle_inner, not handle(): the outer handle already set
            # _confidence, and re-entering handle() would reset the source
            # to "voice" and put a typed/cli utterance through the gate.
            return self._handle_inner(text, source)
        try:
            n = self._lecture.add(body)
        except OSError:
            log.exception("lecture notes: append failed")
            return CommandResult(handled=True, reply=lecture_mod.FAIL_LINE, speak=True,
                                 status="Notes: write failed")
        # speak=False: reading every line back would talk over the lecture.
        return CommandResult(handled=True, reply=body, speak=False,
                             status=f"Noting: {self.lecture_course} ({n})")

    def _try_desktop(self, cmd_text: str) -> bool:
        """Port of _check_desktop_command 3548-3584 (parse via services)."""
        desktop = self._svc("desktop")
        if desktop is None:
            return False

        # Split on "and then", "then", "and", commas for chained commands
        parts = re.split(r"\s+and then\s+|\s+then\s+|\s+and\s+|,\s*", cmd_text)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return False

        actions = []
        for part in parts:
            try:
                action = desktop.parse_action(part)
            except Exception:
                log.exception("desktop parse_action failed for %r", part)
                return False
            if action:
                actions.append(action)
        if not actions:
            return False

        log.info("Desktop commands: %s", actions)
        bus.publish(Status(
            text=f"{len(actions)} command{'s' if len(actions) > 1 else ''}",
            kind="busy"))
        self._bg(lambda: desktop.execute_actions(actions))
        return True

    @staticmethod
    def _phrase_key(text: str) -> str:
        """Lowercase, punctuation-free, single-spaced -- so "Drop my
        needle!" and "drop my needle" are the same phrase."""
        return " ".join(
            "".join(c for c in str(text).lower() if c.isalnum() or c.isspace())
            .split())

    def _try_custom_phrase(self, cmd_text: str) -> Optional[CommandResult]:
        """User-defined phrase -> tool call, from assistant.json.

        Deliberately ahead of the intent classifier and the local model: a
        personal shortcut should not depend on an LLM parsing it, should not
        cost a model round-trip, and should behave identically every time.

            "phrases": [
              {"say": "drop my needle",
               "tool": "spotify_play",
               "args": {"query": "..."},
               "reply": "Dropping the needle, sir."}
            ]

        `say` may be a string or a list of alternatives. Longest phrase wins,
        so a more specific shortcut beats a shorter one that is a prefix of
        it. Unknown tools and handler failures degrade to a spoken apology
        rather than silence -- the user said something they expect to work.
        """
        assistant = self._svc("assistant")
        tools = self._svc("tools")
        if assistant is None or tools is None:
            return None
        try:
            entries = assistant.get("phrases") or []
        except Exception:
            log.debug("phrases lookup failed", exc_info=True)
            return None
        if not isinstance(entries, (list, tuple)):
            return None

        said = self._phrase_key(cmd_text)
        if not said:
            return None

        best = None
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("tool"):
                continue
            says = entry.get("say")
            says = [says] if isinstance(says, str) else (says or [])
            for phrase in says:
                key = self._phrase_key(phrase)
                # Whole words only: raw containment let a short phrase fire
                # inside an unrelated word ("play" in "display the time").
                if key and re.search(r"(?<![a-z0-9])" + re.escape(key)
                                     + r"(?![a-z0-9])", said):
                    if best is None or len(key) > best[0]:
                        best = (len(key), entry)
        if best is None:
            return None

        entry = best[1]
        name = str(entry["tool"])
        args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        log.info("custom phrase -> %s(%s)", name, args)
        try:
            res = tools.call(name, dict(args))
        except Exception:
            log.exception("custom phrase tool %s failed", name)
            return CommandResult(handled=True, speak=True,
                                 reply="That shortcut failed, sir.",
                                 status="phrase failed")
        reply = entry.get("reply") or getattr(res, "speak", None) \
            or getattr(res, "text", None) or ""
        ok = getattr(res, "ok", True)
        return CommandResult(handled=True, reply=str(reply), speak=bool(reply),
                             status=name if ok else f"{name} failed")

    def _try_registry(self, cmd_text: str) -> Optional[CommandResult]:
        for cmd in REGISTRY:
            try:
                m = cmd.matcher(cmd_text)
            except Exception:
                log.exception("matcher %s failed", cmd.name)
                continue
            if not m:
                continue
            missing = [n for n in cmd.needs if self._svc(n) is None]
            if missing:
                log.warning("command %r matched but services missing: %s",
                            cmd.name, missing)
                continue
            try:
                res = cmd.handler(self, cmd_text, m)
            except Exception:
                log.exception("handler %s failed", cmd.name)
                return CommandResult(handled=True,
                                     reply=f"Command failed: {cmd.name}",
                                     status="error")
            if res is not None:
                return res
        return None

    def _prompt_uncertain(self, text: str):
        if self.on_uncertain is not None:
            try:
                self.on_uncertain(text)
                return
            except Exception:
                log.exception("on_uncertain hook failed")
        bus.publish(Status(text=f'Was that for me? — "{text[:60]}"',
                           kind="warn"))

    # -- assistant pre-stages (spec 5.2 a-c) -----------------------------
    def _try_ringing(self, text: str) -> Optional[CommandResult]:
        tk = self._svc("timekeeper")
        if tk is None:
            return None
        try:
            ringing = getattr(tk, "ringing", None)
        except Exception:
            log.exception("timekeeper.ringing failed")
            return None
        if not ringing:
            return None
        t = (strip_jarvis_prefix(text) or text).strip().lower().rstrip(".!")
        m = _SNOOZE_RX.match(t)
        if m:
            n = _num(m.group("n")) if m.group("n") else None
            if n is None:
                n = _assistant_get(self, "alarms.snooze_min", 10)
                try:
                    n = int(n)
                except (TypeError, ValueError):
                    n = 10
            tk.snooze(n)
            return CommandResult(handled=True, reply=f"{n} minutes then, sir.",
                                 speak=True, status=f"Snoozed {n} min")
        if _RING_STOP_RX.match(t):
            tk.stop_ringing("dismiss")
            # Not spoken: he has just been told to stop making noise.
            return CommandResult(handled=True, reply="Very good, sir.",
                                 speak=False, status="Alarm dismissed")
        return None

    def _try_approval(self, text: str, source: str) -> Optional[CommandResult]:
        ap = self._svc("approvals")
        if ap is None:
            return None
        try:
            if not ap.pending():
                return None
        except Exception:
            log.exception("approvals.pending failed")
            return None
        t = (strip_jarvis_prefix(text) or text).strip().lower()
        if _YES_RX.match(t):
            ap.answer(True, source=source)
            return CommandResult(handled=True, reply=ALLOWED_LINE, speak=True,
                                 status="Allowed")
        if _NO_RX.match(t):
            ap.answer(False, source=source)
            return CommandResult(handled=True, reply=DECLINED_LINE, speak=True,
                                 status="Declined")
        return None

    def _try_router_answer(self, text: str) -> Optional[CommandResult]:
        router = self._svc("router")
        if router is None:
            return None
        try:
            pend = router.pending()
        except Exception:
            log.exception("router.pending failed")
            return None
        if pend is None:
            return None
        kind = router.resolve_answer(text)
        if kind is None:
            # A new subject: the question is dropped, the new text routes.
            router.clear_pending()
            return None
        # The modifiers the utterance carried ("use haiku", "in parallel")
        # travel with the remembered question: without them "yes" runs the
        # task on the default model, which is the expensive one.
        d = RouteDecision(kind=kind, reason="answer", prompt=pend.text,
                          project=pend.project,
                          args=dict(getattr(pend, "args", None) or {}))
        return self._dispatch_route(d, pend.text)

    def _try_alarm_offer(self, text: str) -> Optional[CommandResult]:
        """Resolve "Shall I wake you at 7:00 am, sir?" from the evening
        preview (briefing.make_tools parks the offer on services.alarm_offer).

        Same rule as _try_event_confirm: anything but a clear yes or no
        DROPS the offer, and so does an offer older than OFFER_TTL_S -- a
        "yes" to something else the next morning must not set an alarm.
        """
        offer = getattr(self.services, "alarm_offer", None)
        if not isinstance(offer, dict) or not offer:
            return None
        try:
            self.services.alarm_offer = None
        except Exception:
            log.debug("could not clear the alarm offer", exc_info=True)
        try:
            made = float(offer.get("made_at") or 0.0)
        except (TypeError, ValueError):
            made = 0.0
        if made and time.time() - made > OFFER_TTL_S:
            log.info("alarm offer expired; %r is a new subject", text[:40])
            return None
        answer = parse_yes_no(text)
        if answer is None:
            return None
        if not answer:
            return CommandResult(handled=True, reply=NO_ALARM_LINE, speak=True,
                                 status="No alarm")
        tk = self._svc("timekeeper")
        if tk is None:
            return CommandResult(handled=True, reply=TIMEKEEPER_SETUP_LINE,
                                 speak=True, status="No timekeeper")
        try:
            tk.add_alarm(float(offer["due"]), offer.get("label") or "wake up", "once")
        except Exception:
            log.exception("alarm offer: add_alarm failed")
            return CommandResult(handled=True, reply=ALARM_FAILED_LINE, speak=True,
                                 status="Alarm failed")
        when = offer.get("time") or "then"
        return CommandResult(handled=True, reply=f"Alarm at {when}, sir.", speak=True,
                             status=f"Alarm {when}")

    def _try_event_confirm(self, text: str) -> Optional[CommandResult]:
        """Resolve a calendar add that was read back for confirmation.

        Anything that is not a clear yes or no DROPS the offer rather than
        writing: changing the subject must never be read as consent, and a
        lingering offer would attach the next stray "yes" to a stale event.
        """
        source = self._svc("calendar")
        pending = getattr(source, "pending_event", None) if source else None
        if not pending:
            return None
        answer = parse_yes_no(text)
        source.pending_event = None
        if answer is None:
            return None
        if not answer:
            return CommandResult(handled=True, reply="Very good, sir.",
                                 speak=True, status="Dropped")
        try:
            line = write_event(source.icloud_calendars(), pending["title"],
                               pending["start"], pending["end"],
                               calendar_name=pending.get("calendar"))
        except Exception as exc:             # noqa: BLE001 - refusal or server
            log.exception("calendar write failed")
            line = f"I couldn't add that, sir — {type(exc).__name__}."
            return CommandResult(handled=True, reply=line, speak=True,
                                 status="Add failed")
        return CommandResult(handled=True, reply=line, speak=True, status="Added")

    def _try_quiz_answer(self, text: str) -> Optional[CommandResult]:
        """While a quiz question is open, the utterance is the answer.

        Stop words end the quiz with the tally; skip words reveal the
        answer and move on; a fresh "quiz me" / "review my flashcards"
        drops the session for the new one; "quiet" ends it silently. A
        question older than ANSWER_WINDOW_S is not what he is answering:
        the session is dropped and the text routes as a new subject.
        Grading: the string match first (no model round trip for "forty
        percent"), the model only for the unclear ones, and a shrug that
        names the answer when neither can tell."""
        session = getattr(self, "_pending_quiz", None)   # a slim test commander has no quiz
        if session is None:
            return None
        t = (strip_jarvis_prefix(text) or text).strip()
        tl = t.lower().rstrip(".!?")
        if session.stale() or session.finished or quiz_kind(tl) or review_kind(tl):
            self._pending_quiz = None
            return None
        if quiet_kind(t) or cancel_kind(t):
            self._pending_quiz = None
            _cut_speech(self)
            return CommandResult(handled=True, reply="Very good, sir.", speak=False,
                                 status="Quiz stopped")
        if _QUIZ_STOP_RX.match(t):
            self._pending_quiz = None
            return CommandResult(handled=True, reply=session.score_line(), speak=True,
                                 status="Quiz stopped")
        card = session.current
        if _QUIZ_SKIP_RX.match(tl):
            session.settle(None)
            line = quiz_mod.SKIP_LINE.format(answer=card["answer"])
        else:
            verdict = quiz_mod.grade_by_string(card["answer"], t)
            note = ""
            if verdict is None:
                brain = self._svc("brain")
                if brain is not None and hasattr(brain, "grade_answer"):
                    try:
                        graded = brain.grade_answer(card["question"], card["answer"], t)
                    except Exception:
                        log.exception("grade_answer failed")
                        graded = None
                    if graded:
                        verdict, note = bool(graded[0]), str(graded[1] or "")
            if verdict is None:
                session.settle(None)
                line = quiz_mod.UNSURE_LINE.format(answer=card["answer"])
            else:
                try:
                    _quiz_store(self).record(card["id"], verdict)
                except Exception:
                    log.exception("flashcard record failed")
                session.settle(verdict)
                if verdict:
                    line = random.choice(quiz_mod.CORRECT_LINES)
                else:
                    line = note or quiz_mod.WRONG_LINE.format(answer=card["answer"])
                    if "answer" not in line.lower() and card["answer"].lower() not in line.lower():
                        line = quiz_mod.WRONG_LINE.format(answer=card["answer"])
        if session.finished:
            self._pending_quiz = None
            return CommandResult(handled=True, reply=f"{line} {session.score_line()}",
                                 speak=True, status="Quiz finished")
        return CommandResult(handled=True, reply=f"{line} {session.ask()}", speak=True,
                             status=f"Quiz {session.index + 1}/{session.total}")

    def _try_terminal_offer(self, text: str) -> Optional[CommandResult]:
        """After OUTSIDE_LINE ("...say the word and I'll open the terminal
        there instead") a plain yes means the terminal, not a new task."""
        slug = self._pending_terminal_slug
        if not slug:
            return None
        t = text.strip()
        if _YES_RX.match(t) or _OPEN_IT_RX.match(t):
            self._pending_terminal_slug = ""
            claude = self._svc("claude")
            opened = False
            if claude is not None:
                try:
                    opened = bool(claude.open_terminal(
                        None if slug == "*" else slug))
                except Exception:
                    log.exception("open_terminal(%s) failed", slug)
            line = TERMINAL_OPEN_LINE if opened else TERMINAL_FAIL_LINE
            return CommandResult(handled=True, reply=line, speak=True,
                                 status="Terminal" if opened else "No terminal")
        if _NO_RX.match(t):
            self._pending_terminal_slug = ""
            return CommandResult(handled=True, reply="Very good, sir.",
                                 speak=True, status="Dropped")
        self._pending_terminal_slug = ""       # a new subject drops the offer
        return None

    def _web_lookup(self, d) -> CommandResult:
        """A question for the internet: acknowledge now, answer when the
        one-shot lands (brain.web_answer speaks it through the brain
        callback, which also closes the turn)."""
        brain = self._svc("brain")
        if brain is None or not hasattr(brain, "web_answer"):
            return CommandResult(handled=True, reply=WEB_UNAVAILABLE_LINE,
                                 speak=True, status="Web unavailable")
        # a spoken "use sonnet" beats the config default
        model = ((getattr(d, "args", None) or {}).get("model")
                 or _assistant_get(self, "claude.web_model", "haiku"))
        try:
            started = brain.web_answer(d.prompt, model=model)
        except Exception:
            log.exception("web lookup failed to start")
            started = None
        if started is False:
            # busy: the brain already said "Still on the last one" itself
            return CommandResult(handled=True, status="Brain busy")
        if not started:
            return CommandResult(handled=True, reply=WEB_UNAVAILABLE_LINE,
                                 speak=True, status="Web unavailable")
        log.info("web lookup (%s): %r", model, d.prompt)
        return CommandResult(handled=True, reply=WEB_LOOKUP_LINE, speak=True,
                             ack=True, status="Looking it up…", done=False)

    def _match_assistant(self, text: str) -> Optional[str]:
        """The name of the ASSISTANT_TIER1 command whose matcher accepts
        the bare utterance, else None. A probe only -- no handler runs, no
        service is consulted -- used to spare exact command matches the
        intent classifier's guess."""
        t = text.strip().lower().rstrip(".!?")
        for cmd in ASSISTANT_TIER1:
            try:
                if cmd.matcher(t):
                    return cmd.name
            except Exception:
                log.exception("matcher %s failed", cmd.name)
        return None

    def _try_assistant(self, text: str) -> Optional[CommandResult]:
        """Unprefixed Tier 1 for the assistant tools (jarvis mode)."""
        t = text.strip().lower().rstrip(".!?")
        for cmd in ASSISTANT_TIER1:
            try:
                m = cmd.matcher(t)
            except Exception:
                log.exception("matcher %s failed", cmd.name)
                continue
            if not m:
                continue
            if any(self._svc(n) is None for n in cmd.needs):
                continue
            try:
                res = cmd.handler(self, t, m)
            except Exception:
                log.exception("handler %s failed", cmd.name)
                return CommandResult(handled=True,
                                     reply=f"Command failed: {cmd.name}",
                                     status="error")
            if res is not None:
                return res
        return None

    # -- router dispatch (spec 5.2) ---------------------------------------
    def _dispatch_router(self, text: str) -> CommandResult:
        router = self._svc("router")
        brain = self._svc("brain")
        claude = self._svc("claude")
        if router is None or brain is None or not hasattr(brain, "chat"):
            # Legacy wiring (no router / old brain): Tier 2 as before.
            if brain is not None and hasattr(brain, "think"):
                brain.think(text)
                return CommandResult(handled=True, status="Thinking...",
                                     done=False)
            log.warning("jarvis_mode enabled but brain service missing")
            return CommandResult(handled=False, reply=text,
                                 status="No route (no brain)")
        active = None
        if claude is not None:
            try:
                active = getattr(claude, "active_project", None)
            except Exception:
                log.exception("claude.active_project failed")
        try:
            d = router.route(text, active,
                             terminal_open=self._terminal_open_probe(claude))
        except Exception:
            log.exception("router.route failed; local")
            d = RouteDecision(kind="local", reason="router-error")
        return self._dispatch_route(d, text)

    @staticmethod
    def _terminal_open_probe(claude):
        """ClaudeSessionManager.terminal_open as a safe callable, or None.

        getattr, so an older / partly wired session manager without the
        method simply means "no terminal is open" and the router keeps its
        previous behaviour instead of raising."""
        fn = getattr(claude, "terminal_open", None)
        if not callable(fn):
            return None

        def probe(project=None):
            try:
                return bool(fn(project))
            except Exception:
                log.exception("claude.terminal_open(%r) failed", project)
                return False
        return probe

    def _dispatch_route(self, d: RouteDecision, text: str) -> CommandResult:
        log.info("route %s (%s) %r", d.kind, d.reason, (d.prompt or text)[:60])
        brain = self._svc("brain")
        claude = self._svc("claude")
        if d.kind == "action":
            return self._dispatch_action(d)
        if d.kind == "ask":
            return CommandResult(handled=True, reply=ROUTER_QUESTION,
                                 speak=True, status="Which way?")
        if d.kind == "web":
            return self._web_lookup(d)
        if d.kind == "claude":
            if claude is None:
                line = self._setup_line("claude", CLAUDE_SETUP_LINE)
                return CommandResult(handled=True, reply=line, speak=True,
                                     status="Claude unavailable")
            active = None
            try:
                active = getattr(claude, "active_project", None)
            except Exception:
                log.exception("claude.active_project failed")
            model = d.args.get("model")
            if model:
                try:
                    claude.set_model(model)          # sticks for the session
                except Exception:
                    log.exception("set_model(%s) failed", model)
            elif (d.args.get("size") or estimate_size(d.prompt)) == "large":
                model = _assistant_get(self, "claude.big_model", "fable")
            out = claude.submit(d.prompt, project=d.project or active,
                                parallel=bool(d.args.get("parallel")),
                                model=model)
            if isinstance(out, str):
                # A spoken refusal / queue line from the manager. The
                # outside-dir and unsafe-dir refusals both offer the
                # terminal: remember which project a following "yes" means.
                cs_mod = sys.modules.get("jarvis.claude_session")
                offers = {ln for ln in (getattr(cs_mod, "OUTSIDE_LINE", None),
                                        getattr(cs_mod, "UNSAFE_DIR_LINE", None))
                          if ln}
                if out.strip() in offers:
                    # "*" = whatever project is active when he answers
                    self._pending_terminal_slug = d.project or active or "*"
                return CommandResult(handled=True, reply=out, speak=True,
                                     status="Claude: queued", done=False)
            if d.args.get("terminal"):
                # Routed into a terminal the user is looking at: name the
                # session rather than paraphrasing the task (spec 5.1 1b).
                name = d.project or active
                return CommandResult(
                    handled=True, speak=True, done=False,
                    reply=(TERMINAL_ROUTE_LINE.format(name=name) if name
                           else TERMINAL_ROUTE_LINE_ANON),
                    status=f"Terminal: {name or 'open'}")
            ack = CLAUDE_ACK_FALLBACK
            if brain is not None and hasattr(brain, "local_line"):
                try:
                    ack = brain.local_line(
                        "Acknowledge in one short sentence that you are "
                        "starting this, naming the task", d.prompt,
                        fallback=CLAUDE_ACK_FALLBACK) or CLAUDE_ACK_FALLBACK
                except Exception:
                    log.exception("local_line failed; fallback ack")
                    ack = CLAUDE_ACK_FALLBACK
            return CommandResult(handled=True, reply=ack, speak=True,
                                 status=f"Claude: {d.prompt[:40]}", done=False)
        # local (and anything unknown)
        if brain is None or not hasattr(brain, "chat"):
            if brain is not None and hasattr(brain, "think"):
                brain.think(text)
                return CommandResult(handled=True, status="Thinking...",
                                     done=False)
            return CommandResult(handled=False, reply=text,
                                 status="No route (no brain)")
        stripped = strip_address(text)
        forced = forced_call(d.reason, stripped)
        if forced is not None:
            name, args = forced
            log.info("route short-cut: %s(%s)", name, args)
            brain.chat(stripped, force_tool=name, force_args=args)
        else:
            brain.chat(stripped)
        return CommandResult(handled=True, status="Thinking…", done=False)

    def _dispatch_action(self, d: RouteDecision) -> CommandResult:
        claude = self._svc("claude")
        if claude is None:
            line = self._setup_line("claude", CLAUDE_SETUP_LINE)
            return CommandResult(handled=True, reply=line, speak=True,
                                 status="Claude unavailable")
        fn = getattr(claude, d.action, None)
        if fn is None:
            log.warning("claude manager has no %s", d.action)
            return CommandResult(handled=True,
                                 reply="I can't do that one yet, sir.",
                                 speak=True, status=f"No action {d.action}")
        try:
            out = _call_manager(fn, d.args)
        except Exception:
            log.exception("claude.%s failed", d.action)
            out = None
        if d.action == "cancel":
            reply = STOPPED_LINE if out else NOTHING_RUNNING_LINE
        elif isinstance(out, str) and out.strip():
            reply = out.strip()
        elif out is None or out is False:
            reply = "I couldn't manage that, sir."
        else:
            reply = "Very good, sir."
        return CommandResult(handled=True, reply=reply, speak=True,
                             status=f"Claude: {d.action}",
                             done=d.action != "resume")

    def _setup_line(self, section: str, fallback: str) -> str:
        cfg = self._svc("assistant")
        if cfg is None:
            return fallback
        try:
            line = cfg.setup_line(section)
            return line if isinstance(line, str) and line else fallback
        except Exception:
            log.exception("setup_line(%s) failed", section)
            return fallback

    def _route_text(self, text: str) -> CommandResult:
        """Voice-command substitution, targeting, actions, brain, fallback.

        Ports _transcribe_worker 2655-2678 + _on_transcription 2841-2924.
        """
        screenshot = False

        # Apply voice commands (2655-2657)
        if CONFIG.voice_cmds:
            text = _apply_voice_commands(text)

        # Strip stop-recording phrases from end of text (2659-2664)
        for phrase in STOP_RECORDING_PHRASES:
            idx = text.lower().rfind(phrase)
            if idx >= 0 and idx > len(text) - len(phrase) - 5:
                text = text[:idx].rstrip(" ,.-")
                break

        # Screenshot trigger — strip phrase, flag for after typing (2666-2675)
        text_lower = text.lower()
        for phrase in SCREENSHOT_PHRASES:
            idx = text_lower.rfind(phrase)
            if idx >= 0:
                text = (text[:idx] + text[idx + len(phrase):]
                        ).strip().rstrip(" ,.-")
                screenshot = True
                log.info("Screenshot requested, remaining text: %r", text)
                break

        # Voice targeting commands (2841-2853)
        if text and CONFIG.voice_cmds:
            tl = text.strip().lower().rstrip(".")
            desktop = self._svc("desktop")
            if tl in TARGET_RESET_PHRASES:
                if desktop is not None:
                    desktop.reset_target()
                return CommandResult(handled=True, status="Target: auto")
            match = TARGET_PATTERN.match(tl)
            # "switch to the vss project" is a Claude project switch (router
            # action), not a window target; "focus session on the thesis"
            # is a study session (jarvis/focus.py), not "focus <window>".
            if match and not _PROJECT_SWITCH_RX.match(tl) and not _m_focus_start(tl):
                query = match.group(1).strip().rstrip(".")
                if desktop is not None:
                    desktop.target_window(query)
                return CommandResult(handled=True, status=f"Target: {query}")

        # Action commands from _apply_voice_commands (2855-2860)
        if text.startswith("__ACTION__"):
            action = text.replace("__ACTION__", "")
            desktop = self._svc("desktop")
            if desktop is not None:
                self._bg(lambda: desktop.handle_action(action))
            return CommandResult(handled=True, status=f"Action: {action}")

        if not text:
            return CommandResult(handled=False, status="No speech detected")

        # Habit learning (2891-2892)
        memory = self._svc("memory")
        if memory is not None:
            try:
                memory.log_habit(text[:50])
            except Exception:
                log.exception("habit logging failed")

        # Jarvis Mode — send to the brain (2894-2901). The clock and the
        # three courtesies are answered locally first, so Tier 2 never
        # guesses a time or muddles a good night.
        if CONFIG.jarvis_mode:
            kind = clock_kind(text)
            if kind:
                return _h_clock(self, text, kind)
            kind = courtesy_kind(text)
            if kind:
                return _h_courtesy(self, text, kind)
            # Voice I/O answered locally: quiet, say again, pronounce,
            # spelled names / vocabulary, read aloud, continue reading
            # (only mid-reading).
            if quiet_kind(text):
                return _h_quiet(self, text, True)
            if repeat_kind(text):
                return _h_repeat(self, text, True)
            pm = _PRONOUNCE_RX.match(text.strip())
            if pm:
                return _h_pronounce(self, text, pm)
            sm = _SPELL_NAME_RX.match(text.strip())
            if sm:
                return _h_spell_name(self, text, sm)
            am = _ADD_VOCAB_RX.match(text.strip())
            if am:
                return _h_add_vocab(self, text, am)
            if self._svc("reader") is not None:
                rk = read_kind(text)
                if rk:
                    return _h_read_aloud(self, text, rk)
                if continue_kind(text):
                    res = _h_continue(self, text, True)
                    if res is not None:
                        return res
            # Assistant Tier 1 (timers, reminders, alarms, notes, briefing).
            res = self._try_assistant(text)
            if res is not None:
                return res
            # Router: local model / Claude / one question / session action.
            res = self._dispatch_router(text)
            if res.handled:
                return res
            log.warning("jarvis_mode enabled but brain service missing")

        # Intent enhancement — add context if relevant (2903-2908)
        type_text = text
        context = self._svc("context")
        if context is not None:
            try:
                enhanced = context.interpret_intent(text)
                if enhanced:
                    type_text = enhanced
                    log.info("Intent enhanced: +%d chars context",
                             len(enhanced) - len(text))
            except Exception:
                log.exception("interpret_intent failed")

        # Auto-type, then screenshot if requested (2910-2924)
        desktop = self._svc("desktop")
        if CONFIG.auto_type and desktop is not None:
            if screenshot:
                self._bg(lambda: desktop.screenshot(text=type_text))
            else:
                self._bg(lambda: desktop.type_text(type_text))
            return CommandResult(handled=True,
                                 status=f"Typed {len(text)} chars")
        if screenshot and desktop is not None:
            self._bg(lambda: desktop.screenshot())
            return CommandResult(handled=True, status="Screenshot")

        # No route — surface the text so the UI can still show it.
        return CommandResult(handled=False, reply=text,
                             status="No route (auto-type off)")
