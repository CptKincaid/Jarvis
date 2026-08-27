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
               stop(), pending_chunks

Personal-assistant services (spec 2026-08-26, sections 2.2 and 5.2; every
one optional — a missing member falls back to the legacy path):

    assistant  AssistantConfig: get(dotted, default), setup_line(section)
    router     Router: route(text, active_project) -> RouteDecision,
               pending(), resolve_answer(text), clear_pending()
    brain      + chat(text, force_tool=None), local_line(instruction, text,
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
-> pending router question -> desktop chains -> registry -> voice intent
gate -> jarvis-mode Tier 1 -> Router -> (local: brain.chat | claude:
claude.submit | ask: one question | action: claude.<action>).
"""
from __future__ import annotations

import inspect
import json
import random
import re
from datetime import datetime
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from jarvis import pronounce
from jarvis.config import CONFIG, PATHS
from jarvis.events import JarvisReply, Status, bus
from jarvis.logs import get_logger
from jarvis.router import ROUTER_QUESTION, RouteDecision, estimate_size

log = get_logger("commander")

# Fixed persona lines (spec 3.4) — prewarmed in the speech cache by the app.
ALLOWED_LINE = "Allowed, sir."
DECLINED_LINE = "Declined, sir."
STOPPED_LINE = "Stopped, sir."
NOTHING_RUNNING_LINE = "Nothing's running, sir."
CLAUDE_ACK_FALLBACK = "Right away, sir."
CLAUDE_SETUP_LINE = "I'll need Claude set up first, sir."
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

    INTENT_LOG = Path.home() / ".aiws_trainer" / "intent_log.json"
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
        "go ahead", "please", "take a look", "look at",
        "check the screen", "screenshot", "take a screenshot",
        "take screenshot",
    ]

    # Patterns that suggest casual/side conversation
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
            if any(lower.startswith(p) for p in (
                    "run", "fix", "check", "stop", "test", "take",
                    "commit", "push", "show", "open", "screenshot",
                    "save", "deploy", "start", "build", "install")):
                return self.YES, 0.9
            if "screenshot" in lower:
                return self.YES, 0.9
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

        # Long text without positive signals
        if len(words) >= 8 and pos_score == 0:
            neg_score += 1

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


def clock_kind(text: str) -> Optional[str]:
    """'time' / 'date' / 'day' when the text asks for the clock, else None."""
    for kind, rx in _CLOCK_KINDS:
        if rx.search(text or ""):
            return kind
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


def _h_read_aloud(c, t, m):
    reader = c._svc("reader")
    kind, arg = m
    if kind == "clipboard":
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
    if reader is None or not reader.pending_chunks:
        return None                     # not a reading session: fall through
    res = reader.continue_reading()
    if res.ok:
        return CommandResult(handled=True, status=res.message)
    return CommandResult(handled=True, reply=res.message, speak=True,
                         status="Reading finished")


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


def _h_remember(c, t, m):                                  # 3109-3118
    note = m.group(1).strip()
    # PERSISTENT store (jarvis.memory) — fixes the monolith's data loss.
    # Key on the note text (monolith reused one "user_note" key, which
    # silently overwrote every previous note).
    c._svc("memory").remember(note[:60], note)
    c._speak("Noted. I'll remember that.")
    return CommandResult(handled=True, reply=f"Remembered: {note}")


def _h_recall(c, t, m):                                    # 3120-3133
    query = m.group(1).strip()
    results = c._svc("memory").recall(query)
    if results:
        text = "\n".join(f"- {r['value']}" for r in results[:3])
        c._speak(f"I recall: {results[0]['value']}")
        return CommandResult(handled=True, reply=f"I recall:\n{text}")
    return CommandResult(handled=True,
                         reply="I don't have anything stored about that.")


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


def _h_search(c, t, m):                                    # 3202-3215
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
_BRIEFING_RX = re.compile(
    r"^(?:good morning(?:[, ]+jarvis)?|(?:morning|daily|my|the) briefing|briefing|"
    r"what'?s my briefing|(?:give me|read me|run|do) (?:the|my) (?:morning |daily )?briefing|"
    r"brief me)[.!?]*$", re.I)
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
        tk.add_timer(seconds, label or f"{words} timer")
        line = f"{words}, sir; I'll let you know."
        if label:
            what = label if re.match(r"^(?:the|my|a|an|your)\b", label, re.I) \
                else f"the {label}"
            line = f"{words} for {what}, sir; I'll let you know."
        return CommandResult(handled=True, reply=line, speak=True,
                             status=f"Timer set: {words}")
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
    tk.add_alarm(due, label, repeat)
    desc = _describe(tk, due, now, when)
    tail = {"daily": " Every day.", "weekdays": " Weekdays."}.get(repeat, "")
    return CommandResult(handled=True, reply=f"Alarm {desc}, sir.{tail}",
                         speak=True, status=f"Alarm {desc}")


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


def _h_goodnight(c, t, m):                                 # 3233-3238
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
        notes.add("note", note)
        return CommandResult(handled=True, reply="Noted, sir.", speak=True,
                             status=f"Note: {note[:40]}")
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
    notes.add("todo", text)
    return CommandResult(handled=True, reply="Added to your list, sir.",
                         speak=True, status=f"To-do: {text[:40]}")


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
    tk.add_reminder(due, task)
    desc = _describe(tk, due, now, when)
    what = task if re.match(r"^(?:that|about)\b", task, re.I) else f"to {task}"
    return CommandResult(handled=True,
                         reply=f"Very good, sir; I'll remind you {what} {desc}.",
                         speak=True, status=f"Reminder {desc}: {task[:30]}")


# Ordered registry — mirrors _check_quick_command branch order (3036-3485).
# The single insertion is "autonomous" before "workflow" (V3 spec: "deploy"
# and "autonomous:" phrases route to brain.execute_autonomous).
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
    Command("read aloud", read_kind, _h_read_aloud, needs=("reader",)),
    Command("continue reading", continue_kind, _h_continue,
            needs=("reader",)),
    Command("workflow", lambda t: True, _h_workflow, needs=("workflows",)),
    Command("suggest",
            _m_contains("suggest", "what should i do", "any suggestions"),
            _h_suggest, needs=("memory",)),
    Command("remember",
            _m_re(r"remember (?:that )?(.+)"),
            _h_remember, needs=("memory",)),
    Command("recall",
            _m_re(r"(?:recall|what did i say about|remember about)\s+(.+)"),
            _h_recall, needs=("memory",)),
    Command("windows",
            _m_exact("what's open", "whats open", "list windows",
                     "show windows", "what windows are open"),
            _h_windows, needs=("desktop",)),
    Command("launch", _m_re(r"launch\s+(.+)"), _h_launch),
    Command("type", _m_re(r"type\s+(.+)"), _h_type),
    Command("clipboard",
            _m_contains("clipboard", "what did i copy", "read clipboard"),
            _h_clipboard),
    Command("web search",
            _m_re(r"(?:search|google|look up)\s+(?:for\s+)?(.+)"),
            _h_search),
    Command("timer", _TIMER_RX.match, _h_timer),
    Command("alarm", _ALARM_RX.match, _h_alarm),
    Command("list schedule", _LIST_SCHED_RX.match, _h_list_schedule,
            needs=("timekeeper",)),
    Command("cancel schedule", _CANCEL_SCHED_RX.match, _h_cancel_schedule,
            needs=("timekeeper",)),
    Command("briefing", _BRIEFING_RX.match, _h_briefing, needs=("brain",)),
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
    Command("todo done", _TODO_DONE_RX.match, _h_todo_done, needs=("notes",)),
    Command("todo add", _TODO_ADD_RX.match, _h_todo_add, needs=("notes",)),
    Command("todo list", _TODO_LIST_RX.match, _h_todo_list, needs=("notes",)),
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
]

# The assistant's Tier 1 without the "jarvis" prefix (jarvis mode): the
# same handlers, in the same order, run right before the router so that a
# typed "timer for 5 minutes" is instant and never a model round trip.
ASSISTANT_TIER1: list[Command] = [
    cmd for cmd in REGISTRY
    if cmd.name in ("timer", "alarm", "list schedule", "cancel schedule",
                    "briefing", "greeting", "todo done", "todo add",
                    "todo list",
                    "take note", "show notes", "answer question", "remind me")
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
_YES_PHRASES = ("go ahead", "please do", "do it")
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


class Commander:
    """Routes a user utterance (voice or typed) through the V3 pipeline:

    dictation → desktop (jarvis-prefixed) → registry (jarvis-prefixed) →
    intent classification (voice only) → voice-command substitution /
    targeting / actions → jarvis-mode brain → fallback type-to-window.

    Ported from _transcribe_worker (2559-2680) + _on_transcription
    (2820-2940). Speaker verification and the confidence gate stay in the
    transcription pipeline; handle() receives accepted text only.
    """

    def __init__(self, services):
        self.services = services
        self.intent = IntentClassifier()
        self.dictation = False
        self._raw_text = ""
        # Set when the Claude manager refuses an out-of-project task and
        # offers the terminal; the next "yes" opens it (spec 7 / OUTSIDE_LINE).
        self._pending_terminal_slug = ""
        # UI hook for uncertain intent ("Was this for me?"); wired by the
        # main window. Falls back to a warn Status event.
        self.on_uncertain: Optional[Callable[[str], None]] = None

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
    def handle(self, text: str, source: str = "voice") -> CommandResult:
        text = (text or "").strip()
        if not text:
            return CommandResult(handled=False, status="No speech detected")
        log.info("handle %r source=%s", text, source)
        self._raw_text = text          # original casing for handlers that need it

        # 1. Dictation mode — type directly, don't route (2611-2630)
        if self.dictation:
            return self._handle_dictation(text)

        # 2. A ringing alarm owns the next words (spec 5.2 a).
        res = self._try_ringing(text)
        if res is not None:
            return res
        # 3. A pending permission question owns yes / no (spec 5.2 b).
        res = self._try_approval(text, source)
        if res is not None:
            return res
        # 3b. Claude offered the terminal after refusing an outside-dir
        #     task: "yes" / "open it" opens it.
        res = self._try_terminal_offer(text)
        if res is not None:
            return res
        # 4. A pending router question: resolve it and dispatch the
        #    remembered utterance (spec 5.2 c).
        res = self._try_router_answer(text)
        if res is not None:
            return res

        cmd_text = strip_jarvis_prefix(text)
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
        if source == "voice" and cmd_text is None:
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
        self.intent.log_feedback(text, yes)
        if yes:
            return self._route_text(text)
        return CommandResult(handled=True, status="Discarded")

    # -- pipeline stages -----------------------------------------------
    def _handle_dictation(self, text: str) -> CommandResult:
        if "end dictation" in text.lower():
            self.dictation = False
            log.info("Dictation mode ended")
            return CommandResult(handled=True, reply="Dictation mode: OFF")
        if CONFIG.auto_type:
            self._type_raw(text + " ")
        return CommandResult(handled=True, reply=text, status="Dictating")

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
        brain.chat(strip_address(text))
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
            # action), not a window target.
            if match and not _PROJECT_SWITCH_RX.match(tl):
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
            # read aloud, continue reading (only mid-reading).
            if quiet_kind(text):
                return _h_quiet(self, text, True)
            if repeat_kind(text):
                return _h_repeat(self, text, True)
            pm = _PRONOUNCE_RX.match(text.strip())
            if pm:
                return _h_pronounce(self, text, pm)
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
