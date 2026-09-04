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
from datetime import date, datetime, timedelta
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from jarvis import address
from jarvis import arc as arc_mod
from jarvis import aside as aside_mod
from jarvis import board as board_mod
from jarvis import cast as cast_mod
from jarvis import identity as identity_mod
from jarvis import dialogue as dialogue_mod
from jarvis import faults as faults_mod
from jarvis import lecture as lecture_mod
from jarvis import mathspeak
from jarvis import objections as objections_mod
from jarvis import outbox
from jarvis import leavetime as leave_mod
from jarvis import pronounce, standup
from jarvis import reader as reader_mod
from jarvis import soundbar as soundbar_mod
from jarvis.config import CONFIG, PATHS
from jarvis.tools.location import clock_words
from jarvis.events import (ClearTranscript, JarvisReply, SensingChanged,
                           Status, bus)
from jarvis.logs import get_logger
from jarvis.tools.briefing import OFFER_TTL_S
from jarvis.memory import parse_person_statement, parse_since
from jarvis import selfstate
from jarvis.tools import notes as notes_mod
from jarvis.tools import journal as journal_mod
from jarvis.tools import mail as mail_mod
from jarvis.tools import filepick
from jarvis.tools import oracle as oracle_mod
from jarvis.tools import remote as remote_mod
from jarvis.tools import quiz as quiz_mod
from jarvis.tools.calendar import add_event
from jarvis.tools.docs import EmbedError, INDEXING_LINE, course_chunks
from jarvis.tools.notes import number_word
from jarvis.tools import spotify as spotify_mod
from jarvis import syllabus as syllabus_mod
from jarvis.router import (ROUTER_QUESTION, WEB_CUE_RX, RouteDecision,
                           estimate_size, local_cues, normalise)

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
        # --- ambient (2026-08-30): the room tone switch ---
        "room tone", "ambience", "ambient sound", "ambient noise",
        # --- round 2 (2026-08-30): teach-me mode and the study ledger ---
        "teach me", "tutor me", "walk me through", "how much did i study",
        "streak", "study this week",
        # --- named lists (2026-08-30) ---
        # The Tier-1 probe covers the exact phrasings; these carry the
        # looser ones ("anything else on the shopping list?") past the gate.
        "shopping list", "grocery list", "packing list", "reading list",
        "on the list", "off the list", "on my list", "my lists",
        # --- memory round (2026-08-30): episodic recall, the memory garden
        # and the weekly report. The Tier-1 probe bypasses the gate for an
        # exact match; these cover the paraphrases that reach the classifier
        # ("so how long since I looked at the thesis?").
        "when did i last", "the last time i", "how long since",
        "you filed", "memory report", "filed this week", "garden",
        "my week", "week in review", "how was my week",
        # --- syllabus ingestion (2026-08-30) ---
        # "go through my syllabus" reaches no Tier-1 matcher, so without
        # these words it is the same silent NO the media and study words
        # were before it.
        "syllabus", "syllabi", "due dates", "exam schedule",
        # --- the Aside's kill phrase (2026-08-30, jarvis/aside.py) ---
        # The Tier-1 matcher bypasses this gate, so these are the belt to
        # its braces: "no more asides" is three short words and every
        # one-to-three-word phrase used to classify NO and vanish.
        "aside", "asides", "no more",
        # --- the Board (2026-08-30) ---
        # The Tier-1 probe bypasses the gate for the exact phrasings; these
        # carry the loose ones ("can you bring the board up?", "anything on
        # the sessions panel?") past the classifier, which would otherwise
        # read a three-word surface command as background chat -- the same
        # silent drop the media, study and quiz words each hit in turn.
        "the board", "board up", "board down", "focus on the", "panel",
        "mission control",
        # --- room control (2026-08-30): light, level and the scenes ---
        # The Tier-1 probe bypasses the gate for the exact phrasings; this
        # vocabulary covers the looser ones that reach the classifier
        # anyway ("it's a bit bright in here, dim the display").
        "dim", "lights", "brighter", "darker", "brightness", "night light",
        "warmer", "cooler", "too bright", "too dark", "daylight",
        "wind down", "power down", "workshop", "lights out",
        # --- learned walks (2026-08-30): jarvis/leavetime.py ---
        # "it takes ten minutes to get to Wisenbaker" and "make that ten
        # next time" are teaching, not chat; the Tier-1 probe covers the
        # exact phrasings and these cover the looser ones.
        "walk to", "how long to", "minutes to get", "make that", "get to",
        "leave for", "walking",
        # --- persona (2026-08-30): register control and the self answer ---
        # "formal mode" and "banter up" are two-word utterances with no verb
        # the gate knows; without these they classify NO and vanish, the
        # same silent drop as the media and study words before them.
        "formal", "banter", "less formal", "more formal", "normal mode",
        "loosen up", "how do you feel", "how are you",
        # --- dialogue board (2026-08-30): working sessions, faults, runs ---
        # The Tier-1 probe above bypasses the gate for an EXACT match, but
        # these reach the classifier the moment they are said loosely
        # ("sort out my week for me"), and a dropped session opener is
        # worse than most: it is the first turn of a conversation, so the
        # silence reads as him ignoring you rather than mishearing.
        # "plan the week" is spelt out per verb rather than as a bare
        # "plan", which would call "we should plan a trip" a command.
        "plan the week", "plan my week", "sort out my week", "my week",
        "map out the week", "block out the week", "the week ahead",
        "anything wrong", "what went wrong", "the matter", "the fault",
        "quietly", "narrat", "keep it down", "fewer updates",
        "less updates", "how's the run", "the training", "the trainer",
        # --- the Oracle box (2026-08-31): jarvis/tools/oracle.py ---
        # The Tier-1 probe bypasses the gate for an exact match; these carry
        # the loose phrasings ("is that bot of mine still alive?"). Every
        # word here names the box or one of its NINE services, so none of
        # them can drag an unrelated sentence past the classifier.
        "oracle", "demon bot", "still up", "still running", "the server",
        "the bots", "haymaker", "court of awe", "exoshock", "vrider",
        "timecard", "knightfall", "elevation api", "monday sync",
        "bot dashboard",
        # --- the sink sentinel (2026-08-31, jarvis/soundbar.py) ---
        # Asked while staring at a speaker that has gone quiet, which is
        # when the wake word is least likely to have been heard: "where's
        # your voice coming out", "which speaker are you on". The Tier-1
        # probe covers the exact phrasings; these carry the loose ones.
        "coming out of", "which speaker", "what speaker", "the soundbar",
        "the speakers", "audio output", "sound output", "the monitor's",
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
        # room control 2026-08-30: "dim it", "lights up", "warmer"
        "dim", "brighten", "warm", "cool", "lights", "brighter", "darker",
        "warmer", "cooler",
        # dialogue board 2026-08-30: the OPENERS "plan my week" and
        # "quietly please". A session's own answers ("Thursday", "skip
        # it") never reach the classifier -- _try_session is rung 3a at
        # _handle_inner:4025, well above the gate at :4126 -- so the bare
        # weekdays deliberately stay out of this tuple.
        "plan", "quietly", "narrate",
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


# Two commands in one breath ("set a timer for ten minutes and add milk to
# my to-dos"). The conjunctions are the ones _try_desktop has chained on
# since the monolith; one regex for both so the splitters cannot drift.
_CHAIN_SPLIT_RX = re.compile(r"\s+and then\s+|\s+then\s+|\s+and\s+|,\s*", re.I)
MAX_CLAUSES = 2


def split_clauses(text: str) -> list:
    """The halves of a compound utterance, or [].

    EXACTLY two, and only for a caller that has already failed to match the
    whole utterance. A three-way split is far more often one command with a
    list in its body ("add milk, eggs and bread to my to-dos") than three
    commands, and refusing costs nothing: the model still answers.
    """
    parts = [p.strip(" ,.!?") for p in _CHAIN_SPLIT_RX.split(text or "")]
    parts = [p for p in parts if p]
    if len(parts) != MAX_CLAUSES:
        return []
    return [strip_jarvis_prefix(p) or p for p in parts]


def local_tool_clause(text: str) -> str:
    """The local tool the router would name for this clause ALONE, "" for
    none.

    STRONG cues only, which is the whole point: a topic noun on its own
    ("...and eggs", "a half minutes", "and take an umbrella") is a fragment
    of one thought, not a second request, and a weak-cue test would split
    sentences that were never compound. Used by _compound_hijack to tell a
    genuine second intent from the tail of the first one.
    """
    try:
        strong, _weak, kind, _at = local_cues(normalise(text or ""))
    except Exception:
        log.exception("local cue probe failed for %r", text)
        return ""
    return kind if strong else ""


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
    # The STRUCTURED thing the handler actually created -- a timekeeper Item
    # with its epoch, a calendar Event. jarvis/aside.py anchors on this and
    # on nothing else: "set for seven" only becomes "…due at 11:59 that
    # night" if "seven" is a real datetime, and re-parsing it out of the
    # reply text is how an aside lands on the wrong day. Display code must
    # ignore it; it is never spoken or shown.
    action: Any = None


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


# ...and "what time is my class" names something in his DIARY. The wall
# clock answered it live on 2026-09-02 10:10 -- "It's 10:25 in the
# morning, sir." -- because _CLOCK_KINDS' "time" branch only needs the
# words "what time". The answer is the timetable (Command("next class"))
# or the calendar; it is never the current time.
#
# ANCHORED to the clock clause, not the sentence. An earlier `.*?` between
# "what time" and the possessive let any later mention of a room disable
# the wall clock: measured, "what time is it in the lab" and "what time is
# it now that the lab is over" both went from 'time' to None and lost the
# 0-latency answer to a question that really was about the current time.
# The determiner has to follow "what <kind>" within one copula.
#
# Built from the same time|date|day alternation _CLOCK_KINDS carries,
# because "what date is my exam" answered with today's wall date -- one
# word away from "what time is my exam", which this guard already caught.
_CLOCK_MINE_RX = re.compile(
    r"\bwhat\s+(?:time|date|day)\s*(?:'s|s|is|are|was|were)?\s+"
    r"(?:my|our|the|this|that)\s+(?:[a-z][\w'-]*\s+){0,3}?"
    r"(?:class(?:es)?|lectures?|labs?|seminars?|tutorials?|lessons?|"
    r"exams?|midterms?|finals?|quiz(?:zes)?|meetings?|appointments?|"
    r"flights?|trains?|bus|shift)\b", re.I)


def clock_kind(text: str) -> Optional[str]:
    """'time' / 'date' / 'day' when the text asks for the clock, else None.
    A question that names a place is left to the router and get_time; one
    that names something of his is left to the timetable / the calendar."""
    text = text or ""
    for kind, rx in _CLOCK_KINDS:
        if rx.search(text):
            if _CLOCK_PLACE_RX.search(text) or _CLOCK_MINE_RX.search(text):
                return None
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


# ---- Tier 1 next class ----------------------------------------------
# "What's my next class?" answered from the calendar cache with no model
# turn, the way _h_next_exam answers "when's my next exam" from the same
# sources.  LIVE 2026-09-02 10:10:33 and again at 10:10:38 it was
# `route claude (code-cue)` instead -- a billed Opus session that then
# said it could not see his schedule.  jarvis/router.class_diary stops the
# misroute; this is the answer, and the answer was always there: run
# against the live cache that morning, recurring_courses + next_class gave
# ("MAGNETIC RESONANCE ENGR", 12:40, ETB 1003).
#
# NOT a get_calendar path.  range="next" is the next EVENT of any kind:
# measured on the same cache at 09:00 the following day it answers with an
# MBA admissions Zoom while next_class answers ELECTRICAL DESIGN LAB II.
# courses.next_class (jarvis/courses.py) had no caller in the repo at all;
# this is its first one.
#
# Registered AHEAD of Command("clock", ...) -- REGISTRY index 4, and the
# first thing _route_text tries in jarvis mode -- because clock_kind
# matches "what time is my class" and answered it "It's 10:25 in the
# morning, sir.".
NO_CLASS_LINE = "Nothing on your timetable for the next two weeks, sir."
CLASS_LOOKAHEAD_DAYS = 14

_CLASS_KIND = (r"class(?:es)?|lectures?|labs?|seminars?|tutorials?|"
               r"recitations?|lessons?")
# The words between "my" and the noun are the course he named, if any:
# "my biosensors class", "my electrical design lab"; the noun ITSELF is
# captured too (k1/k2/k3), because throwing it away answered "when is my
# next lab" with the 12:40 lecture -- measured on his own cache, where he
# owns a course called ELECTRICAL DESIGN LAB II at 16:10.
#
# NO day qualifier. It used to accept a trailing today/tonight/tomorrow and
# then ignore it: frozen at Wed 2026-09-02 08:00 against his real cache,
# "when is my class tomorrow" answered 'BIOSENSORS in an hour, at 9:10 am'
# -- a class that is TODAY -- and frozen on Saturday "when is my class
# today" named Monday's. A day-scoped question is the router's new
# local:calendar route and get_calendar, which honours a range.
#
# NO "are" either: "what are my classes today" asks for a LIST and was
# answered with one row.
_NEXT_CLASS_RX = re.compile(
    r"^(?:what\s+time|when|where|what|which)\s*(?:'s|s|is|was)?\s+"
    r"(?:my|our)\s+(?:(?:next|upcoming)\s+)?(?P<c1>(?:[a-z][\w'-]*\s+){0,3}?)"
    r"(?P<k1>" + _CLASS_KIND + r")(?:\s+(?:session|period|block))?"
    r"\W*$"
    r"|^(?:what|which)\s+(?P<c2>(?:[a-z][\w'-]*\s+){0,2}?)"
    r"(?P<k2>" + _CLASS_KIND + r")\s+(?:do\s+i\s+have|have\s+i\s+got|is|'s|s)\s+"
    r"(?:up\s+)?next\W*$"
    r"|^when\s+(?:does|do|will)\s+(?:my|our)\s+(?:(?:next|upcoming)\s+)?"
    r"(?P<c3>(?:[a-z][\w'-]*\s+){0,3}?)(?P<k3>" + _CLASS_KIND + r")\s+"
    r"(?:start|begin|kick\s+off)\W*$", re.I)

# "lab" has to reach ELECTRICAL DESIGN LAB II, whose folded title is
# 'electrical design lab ii'; the rest of the nouns appear in a title the
# way they are said.
_KIND_TITLE = {"lab": r"labs?|laborator(?:y|ies)"}
# A one-off sitting is invisible to courses.recurring_courses (MIN_DATES =
# 2), so a make-up lab or the first meeting of a term is silently dropped
# and a LATER class is named "your next class". Probed: a single SENIOR
# DESIGN SEMINAR at now+1h beside a recurring lecture at now+2h answered
# with the lecture and never mentioned the seminar. When something that
# reads like a session sits in the gap, the honest move is the model and
# get_calendar, which lists everything.
_CLASS_TITLE_RX = re.compile(
    r"\b(?:class|lecture|lab|seminar|recitation|tutorial|lesson|studio|"
    r"discussion)\b", re.I)
# ...except a Canvas row, which is coursework and never a sitting. Every
# one of them in his cache wears the section tag: 'Prelab for Lab 3 (canvas
# quiz) [BMEN-427:501,502,503,504,BME...]', 'HW#1 [MSEN-222:599,M99]',
# 'Update Presentation #1 [ECEN-404:901,902,903]'. Without this the Prelab
# row -- timed 12:40 on 2026-09-14, before his 16:10 lab -- sent "when is
# my next lab" to the model for no reason.
_CANVAS_TAG_RX = re.compile(r"\[[A-Z]{2,6}-\d{3}")
_EXPLICIT_NEXT_RX = re.compile(r"\b(?:next|upcoming)\b", re.I)


def _class_kind(word: str) -> str:
    """The narrowing noun the question keyed on -- 'lab' from "my next lab"
    -- or "" for the generic class / classes, which means any course."""
    w = " ".join((word or "").lower().split())
    if w.startswith("class"):
        return ""
    return w[:-1] if w.endswith("s") else w


def _kinded_courses(kind: str, names, fold) -> list:
    """``names`` narrowed to the courses whose title carries ``kind``."""
    rx = re.compile(r"\b(?:" + _KIND_TITLE.get(kind, kind + r"s?") + r")\b",
                    re.I)
    return [n for n in names if rx.search(fold(n))]


def _h_next_class(c, t, m):
    from jarvis import courses as courses_mod
    from jarvis.dossier import room_words
    from jarvis.tools.calendar import describe_due
    from jarvis.tools.location import clock_words
    # The same source boundary the exam lookup crosses: a feed that is
    # unconfigured or down is [] and never an exception on the spoken path.
    from jarvis.tools.canvas import _calendar_events
    events = _calendar_events(c._svc("calendar"))
    if not events:
        return None          # nothing READ is not nothing on: let the model
    names = _course_names(c, events)
    if not names:
        return None          # no repeating slot in the cache: not a timetable
    query = " ".join((m.group("c1") or m.group("c2") or
                      m.group("c3") or "").split())
    kind = _class_kind(m.group("k1") or m.group("k2") or m.group("k3") or "")
    course = ""
    if query:
        # "when is my thermodynamics class" for a course he does not take
        # falls through rather than answering with a different one.
        course = courses_mod.course_for(query, list(names))
        if not course:
            return None
        cand = [course]
    elif kind:
        # He said "lab", not "class": naming a lecture here is the confident
        # wrong answer this command exists to remove. No course of that kind
        # means the model looks, exactly as an unknown course name does.
        cand = _kinded_courses(kind, names, courses_mod.fold)
        if not cand:
            return None
    else:
        cand = list(names)
    now = datetime.now().astimezone()
    if not _EXPLICIT_NEXT_RX.search(t or ""):
        # Asked mid-lecture, "where is my class" used to answer with the
        # 16:10 lab: frozen at Wed 12:50, ten minutes into his 12:40-13:30
        # slot in ETB 1003, it said 'ELECTRICAL DESIGN LAB II ... in
        # Emerging Technologies 1020'. He is asking about the room he is
        # standing in.
        live, live_course = _class_in_progress(events, cand, now, courses_mod)
        if live is not None:
            where = room_words(getattr(live, "location", ""))
            room = f", in {where}" if where else ""
            return CommandResult(
                handled=True, speak=True,
                reply=f"{live_course} is on now until "
                      f"{clock_words(live.end)}{room}, sir.",
                status=f"{live_course[:24]} now")
    ev, found = courses_mod.next_class(
        events, cand, now, within=timedelta(days=CLASS_LOOKAHEAD_DAYS))
    if ev is None:
        return CommandResult(handled=True, reply=NO_CLASS_LINE, speak=True,
                             status="No class")
    if not course and _unlisted_session(events, names, now, ev, courses_mod):
        return None          # something class-shaped is sooner: let it list
    when = describe_due(ev.start, now)
    if when.startswith("in ") or when == "now":
        # format_events words its "next" the same way: the countdown alone
        # ("in 2 hours") is not a time he can write down.
        when = f"{when}, at {clock_words(ev.start)}"
    where = room_words(getattr(ev, "location", ""))
    room = f", in {where}" if where else ""
    noun = kind or "class"
    head = f"Your next {found} is" if course else f"Your next {noun} is {found}"
    return CommandResult(handled=True, speak=True,
                         reply=f"{head} {when}{room}, sir.",
                         status=f"{found[:24]} {when[:24]}")


def _class_in_progress(events, cand, now, courses_mod):
    """(event, course) for a class of ``cand`` running right now."""
    for ev in events:
        end = getattr(ev, "end", None)
        if end is None or courses_mod.slot(ev) is None:
            continue
        try:
            if not (ev.start <= now < end):
                continue
        except TypeError:            # a naive start in a hand-built feed
            continue
        name = courses_mod.course_for(ev.title, cand)
        if name:
            return ev, name
    return None, ""


def _unlisted_session(events, names, now, chosen, courses_mod) -> bool:
    """Is a class-shaped event he does NOT have a recurring slot for due
    before ``chosen``?  See _CLASS_TITLE_RX for why this is worth a turn."""
    for ev in events:
        if courses_mod.slot(ev) is None:
            continue
        try:
            if not (now < ev.start < chosen.start):
                continue
        except TypeError:
            continue
        if courses_mod.course_for(ev.title, names):
            continue
        title = str(getattr(ev, "title", "") or "")
        if _CLASS_TITLE_RX.search(title) and not _CANVAS_TAG_RX.search(title):
            return True
    return False


def _course_names(c, events) -> tuple:
    """His course titles: the dossier's memoised list when it is running
    (it also folds in any Canvas names already cached), else derived from
    the events -- jarvis/courses.recurring_courses is the only source that
    works on this box, where canvas.token is empty."""
    from jarvis import courses as courses_mod
    dossier = c._svc("dossier")
    if dossier is not None:
        try:
            names = tuple(dossier.courses(events))
            if names:
                return names
        except Exception:            # noqa: BLE001 - source boundary
            log.debug("next class: dossier course list unavailable",
                      exc_info=True)
    try:
        # recurring_courses sorts on the first start of each slot, so ONE
        # naive datetime in a hand-built or future feed raises TypeError
        # straight out of Commander.handle (reproduced: two naive-start
        # rows beside the 30 real cache events). A source that cannot be
        # read is the model's problem, not a traceback on the spoken path.
        return tuple(courses_mod.recurring_courses(events))
    except Exception:                # noqa: BLE001 - source boundary
        log.debug("next class: course list unavailable", exc_info=True)
        return ()


def _h_clock(c, t, m):
    # Spoken regardless of talk-back: a clock question asked aloud wants
    # the answer aloud, exactly as the brain's replies are always spoken.
    return CommandResult(handled=True, reply=clock_reply(datetime.now(), m),
                         speak=True, status="Clock")


# ---- Tier 1 arithmetic and unit conversion --------------------------
# "What's 18 percent of 74" used to be a full gemma turn: router's
# _QUESTION_RX sends calculate/compute/convert to the brain, which is
# seconds of wait for something stdlib does exactly (jarvis/mathspeak.py).
# The matcher IS the evaluator: it returns the finished sentence or None,
# so an unrecognised phrasing falls through to the model rather than being
# claimed and half-answered.
def math_kind(text: str):
    """The finished ``mathspeak.Answer`` when this is a sum or conversion
    he can do, else None (the ladder carries on to the router)."""
    return mathspeak.solve(text)


def _h_math(c, t, m):
    # Spoken regardless of talk-back, like the clock: a sum asked aloud
    # wants its answer aloud.
    return CommandResult(handled=True, reply=m.text, speak=True,
                         status="Maths" if m.ok else "Maths (declined)")


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
    # "how do you feel" joined the family with the self-state answer: it is
    # the same question and it used to fall through to a full model turn.
    ("wellbeing", re.compile(
        r"^" + _JV + r"(?:how are you(?: doing| today| feeling| holding up)?|"
        r"how do you feel(?: today| about it)?|how are you feeling|"
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
    # "All systems nominal, sir." is gone from here and from the
    # diagnostics sheet in the same change: brain.VOICE_RULES bans the
    # phrase "all systems", and it was the one line in the product that
    # sounded like a toy. Wellbeing is answered from real state now
    # (jarvis/selfstate.py); these remain the fallback when there is none.
    "wellbeing": ["Very well, sir.", "Never better, sir.",
                  "Quite well, sir."],
    "availability": ["Never too busy for you, sir.", "Quite free, sir.",
                     "Nothing pressing, sir."],
}

# Register variants (brain.REGISTERS). A kind a register does not override
# falls back to COURTESY_REPLIES. Courtesy is his highest-frequency
# utterance, so this is where the register is actually felt -- and every
# line here is fixed, so app._canned_phrases prewarms all of them and the
# register costs no latency.
COURTESY_BY_REGISTER = {
    "formal": {
        "presence": ["Yes, sir.", "Here, sir.", "At your service, sir."],
        "thanks": ["Of course, sir.", "My pleasure, sir.", "Not at all, sir."],
        "goodnight": ["Good night, sir.", "Good night, sir; until tomorrow.",
                      "Sleep well, sir."],
        "availability": ["Quite free, sir.", "At your disposal, sir.",
                         "Nothing pressing, sir."],
    },
    "banter": {
        "presence": ["Always, sir.", "Where else would I be, sir.",
                     "Still here, sir."],
        "thanks": ["Any time, sir.", "It's rather what I'm for, sir.",
                   "Don't mention it, sir."],
        "goodnight": ["Good night, sir; do try to actually sleep.",
                      "Good night, sir; I'll keep the lights on.",
                      "Sleep well, sir."],
        "availability": ["Never too busy for you, sir.", "Idle as ever, sir.",
                         "Free as anything, sir."],
    },
}

# Spoken the instant the register changes -- fixed strings, prewarmed, and
# said WITHOUT a model turn, because the prefix reprocess the change just
# triggered is running behind them (brain.set_register -> warm_static).
REGISTER_LINES = {
    "formal": "Formal it is, sir.",
    "normal": "Back to my usual, sir.",
    "banter": "Very well, sir; I'll be rather less restrained.",
}
REGISTER_ALREADY_LINES = {
    "formal": "I'm already being formal, sir.",
    "normal": "That is my usual, sir.",
    "banter": "I'm already at my least restrained, sir.",
}


def _register_name(c=None) -> str:
    """The spoken register in force; "normal" when the brain is absent."""
    if c is not None:
        brain = c._svc("brain")
        get = getattr(brain, "register", None) if brain is not None else None
        if callable(get):
            try:
                return str(get() or "normal")
            except Exception:
                log.debug("register read failed", exc_info=True)
    try:
        from jarvis import brain as brain_mod
        return brain_mod.register()
    except Exception:                              # noqa: BLE001
        return "normal"


def _register_lines(kind: str, register: Optional[str] = None) -> list:
    """The variants for one courtesy kind in one register."""
    over = COURTESY_BY_REGISTER.get(register or "", {})
    return over.get(kind) or COURTESY_REPLIES[kind]


def _greeting_line(now: Optional[datetime] = None,
                   register: Optional[str] = None) -> str:
    """The time-appropriate line from the greeting variants.

    The bands live in jarvis/arc.py now, not here: brain.ground_greeting
    corrects the MODEL's greeting against the same three, and the canned
    courtesy and the guard disagreeing about what hour it is would be a
    quieter version of the 2026-09-02 14:29 defect ("Good evening" at
    2:29 pm), not a fix for it."""
    return _register_lines("greeting", register)[arc_mod.greeting_index(now)]


def courtesy_kind(text: str) -> Optional[str]:
    """'presence' / 'thanks' / 'goodnight' for a whole-utterance courtesy,
    else None (so "thanks, now open the terminal" is not swallowed)."""
    for kind, rx in _COURTESY_KINDS:
        if rx.match((text or "").strip()):
            return kind
    return None


def courtesy_reply(kind: str, rng=None, register: Optional[str] = None) -> str:
    # "Good afternoon" answered with "Good morning" is worse than no
    # variation at all, so the greeting is chosen by the clock.
    if kind == "greeting":
        return _greeting_line(register=register)
    return (rng or random).choice(_register_lines(kind, register))


def _start_winddown(c) -> bool:
    """The physical half of "good night" (jarvis/winddown.py): music down,
    screen warm and dim, do-not-disturb armed. Off by default, and it fires
    BEFORE _goodnight_preview because that one bails when briefings are
    switched off -- the room should still go to bed.

    Returns True when it actually TOOK the night.  There is exactly one
    owner per good night: winddown.py and the "wind down" scene reach for
    the same xrandr output, the same Spotify and the same quiet window."""
    wd = c._svc("winddown")
    if wd is None:
        return False
    try:
        if wd.start():
            log.info("wind-down started")
            return True
    except Exception:
        log.exception("wind-down failed to start")
    return False


def _restore_winddown(c) -> None:
    """The morning half: idempotent, so every greeting may call it."""
    wd = c._svc("winddown")
    if wd is None:
        return
    try:
        if wd.restore():
            log.info("wind-down restored")
    except Exception:
        log.exception("wind-down restore failed")


def _h_courtesy(c, t, m):
    if m == "goodnight":
        # "Good night" is a courtesy first (this runs ahead of the registry's
        # own good-night entry and again in _route_text): the wind-down
        # preview hangs off it here, and falls back to the plain line.
        started = _start_winddown(c)
        # The room scene is a side effect of the SAME call site rather than
        # a second good-night handler, and only when he has asked for it
        # (room.wind_down_on_goodnight, off by default).  One owner per
        # night: when the wind-down took it, the scene reaches for the same
        # knobs and would only answer with scenes.WIND_DOWN_HELD_LINE.
        scene = "" if started else _maybe_wind_down(c)
        res = _goodnight_preview(c, t)
        if res is not None:
            # The scene line is the ONLY announcement that the desktop
            # moved -- _start_winddown is deliberately silent -- so
            # returning the preview unchanged dimmed the screen, paused the
            # music and rearmed quiet hours with nothing said about any of
            # it.  CommandResult is a plain dataclass; mutate it in place.
            if scene:
                res.reply = f"{scene} {res.reply or ''}".strip()
            return res
        line = courtesy_reply(m, register=_register_name(c))
        return CommandResult(handled=True,
                             reply=f"{scene} {line}".strip() if scene else line,
                             speak=True, status="Courtesy")
    return CommandResult(handled=True,
                         reply=courtesy_reply(m, register=_register_name(c)),
                         speak=True, status="Courtesy")


def greeting_kind(text: str) -> Optional[str]:
    """'greeting' / 'wellbeing' / 'availability' for a whole-utterance
    greeting, else None."""
    for kind, rx in _GREETING_KINDS:
        if rx.match((text or "").strip()):
            return kind
    return None


def _self_state(c, full=True) -> Optional[dict]:
    """The app's one self-state sheet, or None. Pure Python on the app side
    (/proc/meminfo, a jsonl scan, a guarded nvidia-smi Popen), so this stays
    Tier 1 -- it must never be routed to the model."""
    fn = c._svc("self_state")
    if fn is None:
        return None
    try:
        return fn(full=full)
    except TypeError:
        try:
            return fn()
        except Exception:
            log.exception("self state failed")
    except Exception:
        log.exception("self state failed")
    return None


def _h_greeting(c, t, m):
    """"How are you?" is answered from what he IS, not from a list of three.

    full=False: the sink probe and the tmux pane count say nothing about
    his wellbeing and would put two subprocesses on the fastest exchange in
    the system. Availability still falls back to the templated line, which
    is prewarmed and instant; only wellbeing may pay a cache miss.
    """
    if m == "greeting":
        # "Good morning" / "hello": undo last night's wind-down. Not for
        # "how are you" or "are you busy", which are said all day.
        _restore_winddown(c)
        # "Good morning" is the reverse of the wind-down: whatever the
        # scene moved goes back before he has sat down. Silent when no
        # scene is running (the usual case).
        _scene_wake(c)
    line = None
    if m in ("wellbeing", "availability"):
        state = _self_state(c, full=False)
        if state:
            if m == "wellbeing":
                line = selfstate.wellbeing_line(state, register=_register_name(c))
            else:
                line = selfstate.availability_line(state)
    if not line:
        line = courtesy_reply(m, register=_register_name(c))
    return CommandResult(handled=True, reply=line, speak=True,
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


# "When did I last talk to my advisor?" -- episodic recall over the activity
# journal (jarvis/tools/journal.py).  ABOVE "recall" in the registry: the
# recall matcher does not claim these words, but the two are neighbours and
# the order documents which one owns "when did I last say X".
_LAST_SEEN_RX = re.compile(
    r"^(?P<gap>how long (?:has it been |is it |it's been |have i gone )?since)"
    r"\s+i(?:'ve| have)?\s+(?P<tail1>.+)$"
    r"|^when did i last\s+(?P<tail2>.+)$"
    r"|^when(?:'s| was| is) the last time (?:that )?i\s+(?P<tail3>.+)$", re.I)


def _h_last_seen(c, t, m):
    """The last time the JOURNAL saw this — not the last time it happened.

    The journal records what flowed through Jarvis (turns, tool calls,
    Claude results, sampled window titles), so a meeting he never mentioned
    aloud is invisible; the wording says "you said" / "you had X open" so
    the answer never overclaims. A miss falls back to long-term memory,
    which may hold a fact he TOLD me about the same thing."""
    memory = c._svc("memory")
    tail = m.group("tail1") or m.group("tail2") or m.group("tail3") or ""
    target = journal_mod.mention_target(tail)
    if not target:
        return None                    # "when did I last?" — let the model try
    engine = c._svc("context_engine") or c._svc("conversation")
    journal_dir = None
    try:
        if hasattr(engine, "journal_dir"):
            journal_dir = engine.journal_dir()
    except Exception:
        log.exception("journal dir lookup failed")
    if journal_dir is None:
        journal_dir = PATHS.MEMORY_DIR / "journal"
    now = journal_mod._now()       # the module's clock seam; tests pin it
    try:
        row = journal_mod.find_last_mention(
            journal_dir, journal_mod.mention_terms(target, memory), now=now)
    except Exception:
        log.exception("journal mention scan failed")
        return CommandResult(handled=True, speak=True, status="Last mention failed",
                             reply="I couldn't read the journal, sir.")
    if row is not None:
        line = journal_mod.last_mention_line(row, target, now=now,
                                             duration=bool(m.group("gap")))
        return CommandResult(handled=True, reply=line, speak=True,
                             status="Last mention")
    hits = []
    try:
        if hasattr(memory, "recall"):
            hits = memory.recall(target)
    except Exception:
        log.exception("recall after a journal miss failed")
    if not isinstance(hits, list):
        hits = []                      # a stubbed memory must not reach the voice
    value = str(hits[0].get("value", "")).strip() if hits else ""
    line = journal_mod.no_mention_line(target)
    if value:
        line = (f"Nothing in the journal about {target}, sir, but I have "
                f"this stored: {value}.")
    return CommandResult(handled=True, reply=line, speak=True,
                         status="Last mention: nothing")


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
    r"(?:\s+(?P<conn>for|to|called|named|labell?ed)\s+(?P<label>.+?))?[.!]*$", re.I)
# A timer label that is a VERB PHRASE ("to put chicken away") is spoken as
# "8 minutes to put chicken away", never "for the put chicken away" (live
# 2026-09-01 20:40 and 21:03). The connector word decides; this list catches
# the same shape when the transcript dropped the "to".
_VERB_LEAD_RX = re.compile(
    r"^(?:put|pack|take|check|flip|turn|call|get|go|move|stir|start|stop|pull|"
    r"pick|feed|let|bring|wake|change|drain|remove|add|send|text|email|finish|"
    r"switch|grab|make)\b", re.I)


def timer_label_phrase(conn: str, label: str) -> str:
    """How a timer's label reads after the duration: 'to put chicken away'
    for an action, 'for the tea' for a thing. '' with no label."""
    label = (label or "").strip()
    if not label:
        return ""
    conn = (conn or "").strip().lower()
    if conn == "to" or _VERB_LEAD_RX.match(label):
        return f"to {label}"
    if re.match(r"^(?:the|my|a|an|your)\b", label, re.I):
        return f"for {label}"
    return f"for the {label}"


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


# ---- Tier 1 extend / shorten (2026-09-01) --------------------------------
# 20:40:54 "Set a timer for 10 minutes to pack up the chicken." -> Tier-1.
# 20:41:49 "Extend that timer by 10 minutes." -> nothing here matched, so it
# went to the model, whose manage_schedule tool had no extend action; it
# picked `list` and read the timer back ("One timer, sir: pack up the
# chicken in 9 minutes."). The tool now has extend/shorten, but the route
# must not depend on the model's choice again: every phrasing below lands
# here deterministically, like set/list/cancel do.
#
# Python's re forbids a group name twice, so each alternative numbers its
# own groups (k1/n1/u1 ...) and _adj_group() coalesces them: v* the verb,
# d* a direction word (back/later/longer = later; up/earlier/forward/
# shorter/sooner = sooner), n*/u* the amount, k* the kind word, p* a
# pronoun standing in for it ("that" / "it" -> kind all, which last), l*
# label words after for/about/to/called/named.
def _adj_obj(i: int) -> str:
    # Three shapes of object: "[all] [the] <kind> [for <label>]", the label
    # BEFORE the kind word ("the chicken timer" -- at least as natural in
    # speech as "the timer for the chicken", and missing it sent the
    # utterance down the very path that failed on 09-01), and a bare
    # pronoun.
    return (r"(?:(?:all\s+(?:of\s+)?|every\s+)?(?:(?:the|my|that|this|our)\s+)?"
            r"(?P<k%d>timers?|reminders?|alarms?)"
            r"(?:\s+(?:for|about|to|called|named)\s+(?P<l%d>.+?))?"
            r"|(?:the|my|that|this|our)\s+(?P<l%dz>[\w'’-]+(?:\s+[\w'’-]+){0,3}?)\s+"
            r"(?P<k%dz>timers?|reminders?|alarms?)"
            r"|(?P<p%d>that|it|this)(?:\s+one)?)" % (i, i, i, i, i))


def _adj_amt(i: int, more: bool = False, opt_unit: bool = False) -> str:
    """The amount. ``opt_unit`` lets the unit be left off -- "extend the
    timer by five" is a common voice shape and minutes is the only sane
    reading; it is allowed only after a verb that can mean nothing else
    (extend / shorten / ...), never after "add" or "put"."""
    mid = r"(?:(?P<v%d>more|extra)\s+)?" % i if more else r"(?:(?:more|extra)\s+)?"
    unit = r"(?P<u%d>" % i + _UNIT_ALT + r")"
    if opt_unit:
        unit = r"(?:" + unit + r")?"
    return (r"(?:another\s+|an\s+extra\s+)?(?P<n%d>" % i + _NUM_ALT + r")\s*[- ]?"
            + mid + unit)


_ADJ_BY = r"(?:by|for|with)?\s*"
_ADJ_TAIL = r"(?:[, ]+(?:jarvis|please|sir|thanks|thank you))*[.!?]*$"
_ADJUST_SCHED_RX = re.compile(
    r"^(?:(?:please|jarvis)[, ]+)?(?:"
    # extend / shorten <obj> by <amount>
    r"(?P<v1>extend|lengthen|prolong|delay|postpone|shorten|cut|reduce)\s+"
    + _adj_obj(1) + r"\s+" + _ADJ_BY + _adj_amt(1, opt_unit=True)
    # add / put <amount> to / on <obj>
    + r"|(?P<v2>add|put)\s+" + _adj_amt(2) + r"(?:\s+more)?\s+(?:to|on|onto)\s+" + _adj_obj(2)
    # give <obj> another <amount> | give me <amount> more on <obj>
    + r"|(?P<v3>give)\s+(?:me\s+)?(?:" + _adj_obj(3) + r"\s+" + _adj_amt(3) + r"(?:\s+more)?"
    + r"|" + _adj_amt(4) + r"(?:\s+more)?\s+(?:on|for|to|onto)\s+" + _adj_obj(4) + r")"
    # <n> more <unit> on <obj>  (the bare "10 more minutes" is the ringing
    # branch's snooze and, with nothing ringing, the model's: no object, no
    # match here)
    + r"|" + _adj_amt(5, more=True) + r"\s+(?:on|for|to|onto)\s+" + _adj_obj(5)
    # push / move / bump <obj> back|later|up|earlier <amount>
    + r"|(?P<v6>push|move|bump)\s+" + _adj_obj(6)
    + r"\s+(?P<d6>back|later|up|earlier)\s+" + _ADJ_BY + _adj_amt(6)
    # push back <obj> <amount>
    + r"|(?P<v7>push back|move back)\s+" + _adj_obj(7) + r"\s+" + _ADJ_BY + _adj_amt(7)
    # bring <obj> forward <amount> | bring forward <obj> <amount>
    + r"|(?P<v8>bring)\s+(?:" + _adj_obj(8) + r"\s+forward|forward\s+" + _adj_obj(9) + r")\s+"
    + _ADJ_BY + _adj_amt(8)
    # make <obj> <amount> longer|shorter|later|earlier|sooner
    + r"|make\s+" + _adj_obj(10) + r"\s+" + _adj_amt(10)
    + r"\s+(?P<d10>longer|later|shorter|earlier|sooner)"
    # take / knock / shave / cut <amount> off <obj>
    + r"|(?P<v11>take|knock|shave|cut)\s+" + _adj_amt(11)
    + r"\s+(?:off|from)\s+(?:of\s+)?" + _adj_obj(11)
    + r")" + _ADJ_TAIL, re.I)
# Verbs that can mean nothing but "move a scheduled thing": with a bare
# pronoun for an object ("give IT another five minutes") and nothing on the
# books, everything else is far more likely to be about a download, a song
# or a meeting, and must reach the model rather than be answered with a
# schedule negative.
_ADJ_SCHEDULE_VERBS = frozenset(("extend", "shorten", "lengthen", "prolong", "postpone"))
# Words that move an item SOONER; everything else in the table moves it later.
_ADJ_SOONER = frozenset((
    "shorten", "cut", "reduce", "take", "knock", "shave", "bring",
    "forward", "up", "earlier", "sooner", "shorter"))
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
# "Cross the second one off THE LIST" -- no name, so _LIST_STRIKE_RX (which
# requires "<name> list") cannot see it, no other rung matched either, and
# the utterance fell all the way to the router, which classified it
# "claude" with low confidence and offered the handoff: "Shall I hand that
# to Claude, sir, or is it a quick one for me?" (his feature #32, still
# there after two fix passes; reproduced 2026-09-01). The plain form only
# LOOKED fine because "cross bread off the list" is five words and the
# router's SHORT_WORDS prior sends it to gemma4's notes tool -- one word
# longer and it would have failed the same way, so this rung answers both.
_LIST_STRIKE_ANON_RX = re.compile(
    r"^(?:take|cross|scratch|strike|knock|tick|check|rub)\s+(?:off\s+)?"
    r"(?P<item>.+?)\s+(?:off(?:\s+of)?|from|out of)\s+(?:my|the|our)\s+"
    r"list[.!]*$", re.I)
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
# The comma alternative used to win the race on "milk, eggs, and bread":
# ", " matched first and left "and bread" standing as an ITEM, so the list
# read back "eggs, and ... and bread" (reported 2026-08-31, feature #31 --
# "it kept the and bread i said"). The Oxford "and" belongs to the
# SEPARATOR, so it is matched with the comma, before the bare-"and" arm.
_ITEM_SPLIT_RX = re.compile(r"\s*,\s*(?:and\s+|&\s*)?|\s+and\s+|\s*&\s*", re.I)
# Belt and braces for a transcript the splitter above cannot un-pick
# ("milk and, eggs"): no item he dictates starts with a bare conjunction.
_LEADING_CONJ_RX = re.compile(r"^(?:and|&)\s+", re.I)
LIST_SPLIT_CHARS = 60
LIST_SPOKEN_LIMIT = 10          # == NotesStore.resolve's ordinal window


def _split_items(text: str) -> list:
    text = " ".join(str(text or "").split()).strip(" .,")
    if not text:
        return []
    if len(text) > LIST_SPLIT_CHARS:
        return [text]
    parts = [_LEADING_CONJ_RX.sub("", p.strip(" .,")).strip()
             for p in _ITEM_SPLIT_RX.split(text)]
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


def _touch_list(store, kind) -> None:
    """Remember which list is in play, so a later ordinal has something to
    resolve against ("cross the second one off the list"). Guarded: a
    duck-typed/stub store in a test has no touch()."""
    fn = getattr(store, "touch", None)
    if callable(fn):
        try:
            fn(kind)
        except Exception:                       # noqa: BLE001 - a stub store
            log.debug("list touch failed", exc_info=True)


def _list_in_play(store):
    """The kind an unqualified "the list" means, or None when nothing has
    been read or named recently enough to say."""
    fn = getattr(store, "in_play", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:                           # noqa: BLE001 - a stub store
        log.debug("list in_play failed", exc_info=True)
        return None


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
    _touch_list(store, kind)          # "...and cross the second one off the list"
    items = _split_items(mm.group("item"))
    if not items:
        return None
    # add_items, not a bare add() per item: a thing already on the list is
    # not added again (feature #31 -- "milk, milk"). It reports what it
    # skipped so the confirmation stays honest about the count.
    adder = getattr(store, "add_items", None)
    if callable(adder):
        ids, added, dups = adder(kind, items)
    else:                                   # duck-typed / stub store
        ids, added, dups = [store.add(kind, i) for i in items], list(items), []
    if not added:
        # Every one of them was already there: say so rather than claim an
        # add that did not happen.
        subject = notes_mod.join_spoken(dups)
        subject = subject[:1].upper() + subject[1:]
        verb = "is" if len(dups) == 1 else "are"
        return CommandResult(
            handled=True, speak=True,
            reply=f"{subject} {verb} already on your {name} list, sir.",
            status=f"{name}: already there")
    line = f"Added to your {name} list, sir." if len(added) == 1 else \
        f"{number_word(len(added)).capitalize()} added to your {name} list, sir."
    if dups:
        line = line[:-1] + (f"; {notes_mod.join_spoken(dups)} "
                            f"{'was' if len(dups) == 1 else 'were'} already there.")
    return CommandResult(handled=True, reply=line, speak=True,
                         status=f"{name}: {', '.join(added)[:40]}",
                         undo=_undo_notes(store, kind, ids,
                                          f"Off the {name} list again, sir."))


def _h_list_read(c, t, m):
    store, name, kind = _list_target(c, m)
    if kind is None:
        return None
    if not kind:
        return _no_such_list(name)
    # The list he just heard read out is the list "the second one" counts
    # down -- the ordinals index exactly this order (NotesStore.list).
    _touch_list(store, kind)
    return CommandResult(handled=True,
                         reply=store.list_text(kind, LIST_SPOKEN_LIMIT),
                         speak=True, status=f"{name} list")


def _h_list_strike(c, t, m):
    store, name, kind = _list_target(c, m)
    if kind is None:
        return None
    if not kind:
        return _no_such_list(name)
    _touch_list(store, kind)
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


def _h_list_strike_anon(c, t, m):
    """"Cross the second one off the list" -- the list is not named.

    Resolves against the list most recently read, added to or named
    (NotesStore.last_touch, stamped by the handlers above and by the
    `notes` tool). ``store.remove`` already understands "the second one",
    "the last one" and "number two" -- notes.parse_which/resolve index the
    same order list_text speaks -- so the ONLY thing missing was which
    list they were ordinals INTO.

    The safety rule this rung exists to keep: a positional reference with
    no list in play asks rather than guesses. Guessing would delete the
    second item of some list he was not talking about, silently, and the
    undo only helps if he notices.
    """
    store = c._svc("notes")
    if store is None:
        return None
    which = (m.group("item") or "").strip(" .")
    if not which:
        return None
    mode, _ = notes_mod.parse_which(which)
    if mode == "all":
        # "cross everything off the list" is a wipe: leave it to the clear
        # path, which reads it back before doing it.
        return None
    kind = _list_in_play(store)
    if kind is None:
        if mode not in ("index", "last"):
            # A named item ("cross bread off the list") still says what it
            # means, so let the model's notes tool find it as it does now.
            return None
        return CommandResult(
            handled=True, speak=True, status="Which list?",
            reply="Which list, sir? I've lost track of the one you mean.")
    name = notes_mod.list_name(kind) or "to-do"
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
    _touch_list(store, kind)
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
    "study": "study", "flashcard": "study", "flashcards": "study",
    "revision": "study",
}
_PREF_SECTION_WORDS = {"todos": "the to-dos", "alarms": "the alarms",
                       "reminders": "the reminders", "canvas": "Canvas",
                       "weather": "the weather", "calendar": "the calendar",
                       "study": "the study line"}
_PREF_SECTION_RX = re.compile(
    r"^(?:(?P<off>no|skip|drop|leave out|lose|without|i don'?t want|i do not want|"
    r"don'?t (?:read|include|give me|do|mention)|stop (?:reading|including|giving me))"
    r"|(?P<on>include|add|put|bring back|read|give me|i want|i'?d like|"
    r"start (?:reading|including)|mention))"
    r"\s+(?:the\s+|my\s+|any\s+)?"
    r"(?P<section>news|sports?|stocks?|weather|calendar|canvas|coursework|to-?dos?|tasks|"
    r"alarms?|reminders?|study|flash ?cards?|revision)"
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
# The first-wake briefing's offer (app._offer_first_wake_briefing) is
# answered here. The decline is the study offer's word for word: one
# protocol, one sound.
BRIEFING_DECLINED_LINE = "Very good, sir."
BRIEFING_BUSY_LINE = "I'm still on the last one, sir; ask me for it in a moment."
# This offer is answered by its OWN end-anchored grammar and NOT by
# parse_yes_no, which is a word BAG: it waives its own overheard-speech
# guard whenever the first word is a yes/no word, so the ten-word
# "Yeah, so you should be able to look that up." -- a real line from
# jarvis.log.1:19499 that was not an answer to anything and routed to a web
# lookup -- read as a yes and delivered the whole briefing, and "no, turn
# the lights off" was answered "Very good, sir." with the lights still on.
# An answer to a yes/no question is the word and a courtesy and nothing
# else, which is the shape _RING_STOP_RX below already uses.
#
# The vocabulary is wider than parse_yes_no's on BOTH sides and is kept
# local for the same reason the decline was: "later" must not start
# meaning no to a destructive read-back, and "okay" -- absent from
# _YES_WORDS, so "okay" used to answer this question with total silence --
# must not start meaning yes to one.
_BRIEFING_TAIL = r"(?:[,\s]+(?:jarvis|sir|please|thanks|thank you|then|now))*[?.!]*$"
# An affirmative is a CHAIN: "Yes, go ahead." / "Yeah, sure." / "Okay, do
# it." are how he actually answers, and a grammar that took one yes-word
# plus a courtesy refused 26 of 37 natural answers (measured, 09-03, F36)
# and routed "yes go ahead" to the model as a fresh command -- the day's
# only offer gone. Still end-anchored, so "yes, turn the lights off" falls
# through to the command it is.
_BRIEFING_YES_WORD = (
    r"(?:yes|yeah|yep|yup|aye|affirmative|certainly|absolutely|definitely|"
    r"of course|sure(?: thing)?|ok(?:ay)?|alright|all right|sounds good|"
    r"very well|please do|do it|do that|run it|go ahead|go for it|why not|"
    r"let'?s (?:hear it|do it)|i would|if you would|please|"
    r"that would be great)")
_BRIEFING_YES_RX = re.compile(
    r"^(?:jarvis[,\s]+)?" + _BRIEFING_YES_WORD
    + r"(?:[,\s]+(?:" + _BRIEFING_YES_WORD
    + r"|jarvis|sir|please|thanks|thank you|then|now))*[?.!]*$", re.I)
# "go on" / "carry on" / "continue" are deliberately NOT here and neither
# is a bare "skip": _READ_CTL_RX owns the first three and "skip" alone is
# in the live log as a real command routed to local:music
# (jarvis.log.1:20928). "skip it" / "skip that" are declines, but only when
# nothing is being read -- see the read-control guard in
# _try_briefing_offer, which this rung sits ABOVE.
_BRIEFING_NO_RX = re.compile(
    r"^(?:jarvis[,\s]+)?(?:no[,\s]+)?"
    r"(?:no|nope|nah|negative|not now|not today|not right now|not just now|"
    r"later|maybe later|another time|some other time|in a bit|"
    r"skip it|skip that|leave it|never mind|nevermind|forget it|no need|"
    r"i'?m good|i'?m fine)" + _BRIEFING_TAIL, re.I)
# The wake alarm's 180 s OFFER_TTL_S is the wrong size for THIS offer twice
# over. It arms its own microphone (app._offer_first_wake_briefing sets
# _followup_after_speech), so the answer lands inside the follow-up window
# the question itself opened -- quiz.window_s, 15 s -- and an answer two
# minutes later is not an answer. And app._question_open reads the same
# stamp; that predicate also gates _salvage_low_confidence, so a
# three-minute offer force-accepted sub-threshold garble for three minutes
# after a question he may never have heard. That argument is already
# written out against _pending_leave in app._question_open; this is the
# same one. 60 s: the offer's own line, the 15 s window it opens, and room
# for one wake-word retry on top.
BRIEFING_OFFER_TTL_S = 60.0
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
        line = f"{words}, sir; I'll let you know."
        # "8 minutes to put chicken away" / "5 minutes for the tea": the
        # connector he used decides the grammar (live 2026-09-01 21:03:
        # "8 minutes for the put chicken away, sir"). The stored label is
        # the phrase as spoken, so the fire line and "extend the timer for
        # the chicken" keep working on it.
        what = timer_label_phrase(m.group("conn"), label)
        if what:
            line = f"{words} {what}, sir; I'll let you know."
        def _run():
            item = tk.add_timer(seconds, label or f"{words} timer")
            # action=item so the aside engine sees the structured result; it
            # then declines a timer on purpose (aside.SKIP_KINDS) -- a
            # countdown is a kitchen device, not a point in the day.
            return CommandResult(handled=True, reply=line, speak=True,
                                 status=f"Timer set: {words}", action=item,
                                 undo=_undo_timekeeper(tk, item, "timer",
                                                       "Timer scrapped, sir."))
        # A shaky transcript reads the parsed timer back first (_confirm_or_run)
        # "An 8-minute timer to put chicken away, sir?" -- "an" before a
        # spoken 8 / 11 / 18 / 80..., "a" otherwise.
        article = "An" if str(n).startswith("8") or n in (11, 18) else "A"
        ask = f"{article} {n}-{_unit_word(unit, 1)} timer {what}, sir?" if what \
            else f"A timer for {words}, sir?"
        return _confirm_or_run(c, _run, ask)
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
    desc = _describe(tk, due, now, when)
    tail = {"daily": " Every day.", "weekdays": " Weekdays."}.get(repeat, "")

    # Split so the whole "set it" half is a callable: an objection stashes
    # THIS and runs it unchanged on a yes, so the reply, the speak flag and
    # the status of an overruled alarm are byte-identical to one that was
    # never objected to (tests/test_objections.py asserts exactly that).
    def _do_set_alarm() -> CommandResult:
        item = tk.add_alarm(due, label, repeat)
        return CommandResult(handled=True, reply=f"Alarm {desc}, sir.{tail}",
                             speak=True, status=f"Alarm {desc}", action=item,
                             undo=_undo_timekeeper(tk, item, "alarm",
                                                   "Alarm cancelled, sir."))

    def _run():
        # The objection is raised INSIDE the read-back's callable, never
        # beside it: _confirm_or_run's pending slot has already been
        # resolved by the time this runs, so a shaky-transcript read-back
        # and a dissent offer can never be live at the same time
        # (spec 2 correction 5).
        obj = c.objection_for_alarm(due, now)
        if obj is None:
            return _do_set_alarm()
        line = obj.line(f"a {clock_words(datetime.fromtimestamp(due))} alarm")
        c.stash_objection(_do_set_alarm, line, obj)
        return CommandResult(handled=True, reply=line, speak=True,
                             status="Advising against")

    # A misheard hour is the daily cost of a confident guess: on a shaky
    # transcript the parsed time is read back before anything is set.
    asked = {"daily": ", every day", "weekdays": ", weekdays"}.get(repeat, "")
    return _confirm_or_run(c, _run, f"An alarm {desc}{asked}, sir?")


def _m_no_asides(t):
    """The aside kill, and ONLY the aside kill.

    Deliberately its own matcher rather than a widening of _QUIET_RX: "stop
    that" already means barge-in and read-aloud steering, and a kill the
    router can confuse with either of those is worse than no kill at all
    (jarvis/aside.py, KILL_PHRASES)."""
    return aside_mod.kill_phrase(t)


def _h_no_asides(c, t, m):
    engine = c._svc("aside")
    if engine is None:
        # Nothing to silence, but never argue about it: the user asked for
        # quiet and getting a lecture back is the joke writing itself.
        return CommandResult(handled=True, reply=aside_mod.KILL_LINE, speak=True,
                             status="Asides off")
    return CommandResult(handled=True, reply=engine.silence(t), speak=True,
                         status="Asides off for today")


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


def _confirm_or_run(c, run: Callable[[], CommandResult],
                    question: str) -> CommandResult:
    """Commit the creation, or read the PARSED result back first when the
    transcript scraped in under confirm.shaky_logprob.

    "5:15" and "5:50" differ by one phoneme and the cost of the wrong one
    lands hours later, in the dark; the same for "remind me at two" heard
    as "at ten". A confident transcript stays zero-friction -- this branch
    only fires on the doubtful tail the confidence gate already measures --
    and the yes costs one second through the follow-up window that any
    spoken reply opens. Reusing stash_destructive means the answer is
    resolved by _try_destructive_confirm, so the offer expires after
    DESTRUCTIVE_TTL_S and a change of subject drops it, exactly like a bulk
    cancel: an unanswered read-back must never set an alarm later.
    """
    if _assistant_get(c, "confirm.read_back", True) and c.shaky_transcript():
        c.stash_destructive(run, question)
        return CommandResult(handled=True, reply=question, speak=True,
                             status="Confirm?")
    return run()


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


def _adj_group(m, prefix: str) -> Optional[str]:
    """The first non-None group whose name starts with ``prefix`` -- the
    numbered alternatives of _ADJUST_SCHED_RX share one meaning per letter."""
    for name, val in m.groupdict().items():
        if name.startswith(prefix) and val is not None:
            return val
    return None


def _tk_now(tk) -> float:
    """The timekeeper's clock when it has one (tests drive a fake), else
    the wall clock."""
    fn = getattr(tk, "_now", None)
    if callable(fn):
        try:
            return float(fn())
        except (TypeError, ValueError):
            pass
    return time.time()


def _describe_adjusted(tk, items, now: float) -> list[str]:
    """One '<label> in 20 minutes' per adjusted item: the timekeeper's own
    _describe_item when it has one (a MagicMock's answer is not a str and
    is skipped), else the label plus describe_due."""
    out = []
    fn = getattr(tk, "_describe_item", None)
    for it in items:
        text = None
        if callable(fn):
            try:
                got = fn(it, now)
                if isinstance(got, str) and got.strip():
                    text = got.strip()
            except Exception:
                log.exception("_describe_item failed")
        if text is None:
            label = getattr(it, "label", None)
            label = str(label).strip() if isinstance(label, str) and label.strip() else "it"
            due = getattr(it, "effective_due", None)
            if not isinstance(due, (int, float)):
                due = getattr(it, "due", None)
            if isinstance(due, (int, float)):
                text = f"{label} {_describe(tk, float(due), datetime.fromtimestamp(now), 'later')}"
            else:
                text = label
        out.append(text)
    return out


def _undo_adjust(tk, items, kind: str, delta: int):
    """Put each adjusted item back.

    Exactly where it was when Timekeeper.adjust handed back a ``previous``
    snapshot: re-applying the opposite delta is NOT an undo once a due was
    clamped to now (take five minutes off a two-minute timer and adding
    five back leaves it LATER than it started, under a reply that claims
    otherwise). Without a snapshot -- a stubbed timekeeper, or a ring that
    cannot be un-killed -- the opposite delta is still the best there is."""
    marks = []
    for i in items:
        item_id = getattr(i, "id", None)
        if not isinstance(item_id, str) or not item_id:
            continue
        item_kind = getattr(i, "kind", None)
        prev = getattr(i, "previous", None)
        if not (isinstance(prev, tuple) and len(prev) == 3
                and isinstance(prev[0], (int, float))):
            prev = None
        marks.append((item_id,
                      item_kind if isinstance(item_kind, str) and item_kind else kind,
                      prev))
    if tk is None or not marks:
        return None

    def _undo() -> str:
        back = 0
        restore = getattr(tk, "restore", None)
        for item_id, item_kind, prev in marks:
            try:
                if prev is not None and callable(restore) and restore(item_id, prev):
                    back += 1
                    continue
                if tk.adjust(item_id, item_kind, -delta):
                    back += 1
            except Exception:
                log.exception("undo adjust failed")
        return "Back to where it was, sir." if back else "That one had gone already, sir."
    return _undo


def _h_adjust_schedule(c, t, m):
    """"Extend that timer by 10 minutes" (live 2026-09-01 20:41). No
    read-back even on a shaky transcript: the change is reversible by
    "scratch that" (undo=), and the reply speaks the new due, so a misheard
    number is heard straight away. A RINGING item reached here with a
    positive delta is snoozed by Timekeeper.adjust (that is what "give me
    ten more minutes on the alarm" means while it rings); the ringing
    branch's own "10 more minutes" / "snooze" words are untouched."""
    tk = c._svc("timekeeper")
    if tk is None or not callable(getattr(tk, "adjust", None)):
        return None
    n = _num(_adj_group(m, "n"))
    if n is None:
        return None
    unit = (_adj_group(m, "u") or "minutes").lower()
    verb = (_adj_group(m, "v") or "").lower()
    direction = (_adj_group(m, "d") or "").lower()
    word = direction or verb
    sign = -1 if word in _ADJ_SOONER else 1
    delta = sign * _seconds(n, unit)
    if delta == 0:
        return None
    kind_word = (_adj_group(m, "k") or "").lower()
    kind = "reminder" if kind_word.startswith("remind") else \
        "timer" if kind_word.startswith("timer") else \
        "alarm" if kind_word.startswith("alarm") else "all"
    pronoun = _adj_group(m, "p") is not None
    if direction == "up" and (pronoun or kind == "timer"):
        # "Move the ALARM up ten minutes" is the calendar idiom -- earlier.
        # "Bump that timer up five minutes" is at least as often "give it
        # five more", so the confident opposite is worse than no Tier-1
        # match at all: it falls to the model.
        return None
    which = (_adj_group(m, "l") or "").strip()
    if not which:
        # "extend my timers by ten minutes": the plural is all of them, as
        # it is for cancel; the singular or a pronoun is the latest one.
        which = "all" if kind_word.endswith("s") else "last"
    try:
        from jarvis.tools.timekeeper import adjust_empty_line, adjust_line, applied_delta
    except Exception:                                 # pragma: no cover
        adjust_line = adjust_empty_line = None

        def applied_delta(items, requested):
            return requested
    items = tk.adjust(which, kind, float(delta))
    items = list(items) if isinstance(items, (list, tuple)) else []
    later = delta > 0
    if not items:
        if pronoun and not kind_word and verb not in _ADJ_SCHEDULE_VERBS:
            # "Give it another five minutes" with an empty schedule is
            # almost certainly about something else; only the unambiguous
            # schedule verbs answer for the schedule here.
            return None
        line = adjust_empty_line(tk, kind, delta) if adjust_empty_line else \
            ("Nothing to extend, sir." if later else "Nothing to shorten, sir.")
        return CommandResult(handled=True, reply=line, speak=True,
                             status="Nothing to adjust")
    now = _tk_now(tk)
    described = _describe_adjusted(tk, items, now)
    if adjust_line is not None:
        # applied_delta, not delta: a shorten past now is clamped to now,
        # and "999 minutes off, sir: the tea now" was wrong twice.
        line = adjust_line(applied_delta(items, delta), described)
    else:                                             # pragma: no cover
        amount = f"{n} {_unit_word(unit, n)}"
        line = f"{amount} {'added' if later else 'off'}, sir: {'; '.join(described)}."
    noun = kind if kind != "all" else \
        (getattr(items[0], "kind", None) if isinstance(getattr(items[0], "kind", None), str)
         else "schedule")
    status = f"{str(noun).capitalize()} {'extended' if later else 'shortened'}"
    log.info("tier-1 adjust: %s %s %r by %+d s -> %s", kind, which,
             [getattr(i, "id", None) for i in items], delta, line)
    return CommandResult(handled=True, reply=line, speak=True, status=status,
                         action=items[0] if len(items) == 1 else items,
                         undo=_undo_adjust(tk, items, kind, delta))


def _h_briefing(c, t, m):
    enabled = bool(_assistant_get(c, "briefing.enabled", False))
    explicit = not re.match(r"^good morning", t, re.I)
    if not explicit:
        # "Good morning" reaches the briefing BEFORE the greeting handler,
        # so the wind-down's morning half has to be undone here too --
        # ahead of the enabled check, which returns None on a box with
        # briefings switched off.
        _restore_winddown(c)
        # "Good morning" reaches the briefing before the greeting handler,
        # so the wind-down's reverse has to be hung off BOTH call sites.
        _scene_wake(c)
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
# A deliberate note from a source the mode does not listen to: `jarvis
# "note: the demo is on friday"` while the lecture runs on the microphone.
_NOTE_PREFIX_RX = re.compile(r"^note\s*[:\-]\s*(?P<body>\S.*)$", re.I)


def _dictation_end(text: str) -> bool:
    """"end dictation" closes the mode from ANY source. The mode itself is
    voice-only (_handle_inner), so a terminal that sees it in "status"
    needs a way to close it."""
    return "end dictation" in (text or "").lower()


def _lecture_end(text: str) -> bool:
    """Same rule for lecture notes: the end phrase is honoured from any
    source, so "jarvis 'end notes'" from a shell closes the capture."""
    return bool(_LECTURE_END_RX.match(strip_address(text).strip().lower()))


def _note_prefix(text: str) -> Optional[str]:
    """"note: the demo is on friday" -> "the demo is on friday", else None."""
    m = _NOTE_PREFIX_RX.match(strip_address(text).strip())
    return m.group("body").strip() if m else None


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
    """The film's "run diagnostics": uptime, models, today's turns, the box.

    ONE answer per turn. This used to speak the film register AND publish
    the plain sheet as a "card", which is not a thing the bus has: a
    JarvisReply IS the answer, so the UI drew both as Jarvis bubbles and
    cmdsock streamed both as {"kind": "reply"}. He heard the film line and
    read the plain one, reported 2026-08-31 as "(didnt say this but in
    transcript) said this", and `jarvis --quiet "run diagnostics"` printed
    two whole different status sentences on 2026-09-01.

    The film register is the answer, in every register but formal: it was
    written for the Iron-Man round, it is what the room hears, and it
    carries the numbers that matter (clock and utilisation, not power
    draw -- idle reads ~15 W both wedged and healthy). Nothing is lost by
    dropping the second line: the plain sheet is still what `jarvis
    status`, ask.py --status, the phone (webapp) and the formal register
    render, all off this same selfstate dict.

    No model turn either: routing a fact sheet through gemma4 buys nothing
    and buys back the hallucination risk, and "never invent a figure" is a
    prompt, not a guarantee.
    """
    fn = c._svc("diagnostics")
    if fn is None:
        return None
    state = _self_state(c)
    try:
        # Both renderings come off the SAME dict when the app can hand one
        # over, so the spoken line and the card can never disagree -- one
        # state source in two registers, never a second that drifts. The
        # `diagnostics` service stays the gate and the fallback.
        plain = selfstate.diagnostics_line(state) if state else fn()
    except Exception:
        log.exception("diagnostics failed")
        return CommandResult(handled=True, speak=True, status="Diagnostics",
                             reply="I'm afraid the diagnostics didn't complete, sir.")
    spoken = plain
    if state and _register_name(c) != "formal":
        # Formal gets the plain sheet: the film register is an aside, and
        # the register that bans asides bans this one too.
        spoken = selfstate.stark_line(state) or plain
    # Deliberately NO second publish here. See the docstring: the plain
    # sheet as a companion "card" was a second answer to the same
    # question, in a different voice, on the same bus.
    return CommandResult(handled=True, reply=spoken, speak=True, status="Diagnostics")


# ---- Tier 1 register: "formal mode" / "banter up" ---------------------
# The preference is persisted to assistant.json AND memory (the same
# _persist_preference both briefing knobs use) and baked into the STATIC
# Tier 2 prompt. That is the whole design: one cache miss at change time,
# paid in the background, never one per turn.
_REGISTER_RX = re.compile(
    r"^(?:"
    r"(?P<formal>(?:be |go |switch to |use |turn on )?(?:more )?formal(?: mode| register| please)?"
    r"|(?:be |go )?(?:more )?(?:serious|businesslike|professional)(?: mode| please)?"
    r"|(?:less|no|cut the|drop the|enough) (?:banter|jokes|joking|wit|quips))"
    r"|(?P<banter>banter up|more banter|(?:be |go )?(?:more )?(?:playful|cheeky|witty)"
    r"|(?:turn|dial) (?:up|on) the (?:banter|wit|jokes)|loosen up)"
    r"|(?P<normal>(?:back to |go back to )?(?:your |the )?(?:usual|normal)(?: mode| register| self| voice)?"
    r"|(?:be |go )?normal again|banter down|less formal|stop being (?:so )?formal"
    r"|(?:dial|turn) (?:down|off) the (?:banter|wit|jokes))"
    r")[.!\s]*$", re.I)


def register_kind(text: str) -> Optional[str]:
    """'formal' / 'banter' / 'normal' for a whole-utterance register
    request, else None."""
    m = _REGISTER_RX.match((text or "").strip())
    if not m:
        return None
    for name in ("formal", "banter", "normal"):
        if m.group(name):
            return name
    return None


def _h_register(c, t, m):
    """"Formal mode" / "banter up" / "back to normal"."""
    want = m if isinstance(m, str) else register_kind(t)
    if want is None:
        return None
    brain = c._svc("brain")
    setter = getattr(brain, "set_register", None) if brain is not None else None
    if not callable(setter):
        return None
    # assistant.json is the source of truth read at app start; memory's
    # preferences.json is the record of what he asked for. A config service
    # that will not take it means the preference cannot survive a restart,
    # and a register that forgets itself overnight is worse than none.
    if not _persist_preference(c, "persona.register", want):
        return None
    try:
        changed = bool(setter(want))
    except Exception:
        log.exception("set_register(%s) failed", want)
        return None
    line = REGISTER_LINES[want] if changed else REGISTER_ALREADY_LINES[want]
    return CommandResult(handled=True, reply=line, speak=True,
                         status=f"Register: {want}")


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

# The weekly self-review (jarvis/dayreview.py week_*): cross-day trends over
# the nightly digests. Registered BEFORE the "week" briefing command, which
# owns "how's my week looking" -- the forecast is about his calendar, this is
# about Jarvis. Neither regex reaches the other's words; the order says which
# would win if a future phrasing straddled them.
_WEEKREVIEW_RX = re.compile(
    r"^(?:(?:my |the |your )?week(?:ly)? (?:review|report|self[- ]review|digest)"
    r"|(?:my |the )?week in review"
    r"|how (?:was|did) (?:my|the|your) week(?:\s+go)?"
    r"|what went wrong last week"
    r"|review (?:my |the )?last week)\W*$", re.I)
# The weekly memory garden (jarvis/garden.py): what Jarvis filed about him
# out of his own journal, and the one sentence that takes it back.
_GARDEN_UNDO_RX = re.compile(
    r"^(?:forget (?:the )?(?:last )?(?:garden|memory) pass"
    r"|undo (?:the )?(?:last )?(?:garden|memory) pass"
    r"|forget what you (?:filed|learned)(?: (?:this|last) week)?)\W*$", re.I)
_GARDEN_REPORT_RX = re.compile(
    r"^(?:memory report"
    r"|what (?:did|have) you file[d]?(?: this week| last week| from the journal)?"
    r"|what did you learn(?: about me)?(?: this week| last week)?"
    r"|what have you learned about me)\W*$", re.I)


def _h_week_review(c, t, m):
    """"How was my week": the two spoken sentences, the table on a card."""
    fn = c._svc("week_review")
    if fn is None:
        return None
    card = ""
    try:
        out = fn()
        spoken, card = (out if isinstance(out, tuple) else (str(out), ""))
    except Exception:
        log.exception("week review failed")
        spoken = "I'm afraid the weekly review didn't complete, sir."
    if card:
        bus.publish(JarvisReply(text=card, speak=False))
    return CommandResult(handled=True, reply=spoken, speak=True,
                         status="Week review")


def _h_garden_report(c, t, m):
    fn = c._svc("garden_report")
    if fn is None:
        return None
    try:
        line = fn()
    except Exception:
        log.exception("memory report failed")
        line = "I'm afraid I couldn't read what I filed, sir."
    return CommandResult(handled=True, reply=line, speak=True,
                         status="Memory report")


def _h_garden_undo(c, t, m):
    """The correction path for the garden. No read-back: everything it
    removes is something JARVIS wrote, never something Hunter said, and
    the state file keeps the values so the line can name the damage."""
    fn = c._svc("garden_undo")
    if fn is None:
        return None
    try:
        line = fn()
    except Exception:
        log.exception("memory garden undo failed")
        line = "I'm afraid I couldn't undo that, sir."
    return CommandResult(handled=True, reply=line, speak=True,
                         status="Garden undone")


# The sink sentinel (jarvis/soundbar.py). Asked at the desk, staring at a
# speaker that is not playing, so it must answer without the wake word --
# and it is a QUESTION only: nothing here moves a sink.
_AUDIO_OUT_RX = re.compile(
    r"^(?:where(?:'s| is| are)? (?:your |my |the )?"
    r"(?:voice|audio|sound|speech|you) (?:coming out|going|playing|coming from)"
    r"(?: of| from| to)?"
    # "coming out of" belongs here too: it is the phrasing a person actually
    # uses at a speaker that has gone quiet, and it was in the gate
    # vocabulary while no matcher accepted it (found live 2026-08-31).
    r"|(?:which|what) (?:speaker|sink|output|device) (?:are you|is that|is it)"
    r" (?:on|using|coming out of|playing (?:on|through)|going (?:to|through))"
    r"|(?:what|which) (?:is |are )?(?:my |your )?(?:audio|sound) output)\W*$", re.I)


def _h_audio_out(c, t, m):
    """"Where's your voice coming out?" -- the sentinel's last reading.

    Returns None with no sentinel wired rather than an invented answer: on
    a box without pactl the honest reply is the model's shrug, not a
    confident sentence about a speaker nobody probed.
    """
    sentinel = c._svc("soundbar")
    if sentinel is None:
        return None
    try:
        line = sentinel.status_line()
    except Exception:
        log.exception("soundbar status failed")
        line = soundbar_mod.BLIND_LINE
    return CommandResult(handled=True, reply=line, speak=True,
                         status="Audio output")


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


# "What's wrong?" -- the fault lane's drill-down. The live fault first
# (jarvis/faults.py holds it long after the 4-6 s Status chip is gone),
# and when the board is clear it falls straight through to the existing
# log triage rather than answering "nothing" without looking.
_WHATS_WRONG_RX = re.compile(
    r"^(?:so )?(?:what(?:'s| is| has)? (?:wrong|the matter|the problem|broken)|"
    r"is (?:anything|something) (?:wrong|the matter|up)|"
    r"anything wrong|what went wrong|"
    r"(?:what|how)(?:'s| is) (?:the )?(?:fault|trouble))"
    r"(?: with (?:you|the box|the spark|the machine|it))?"
    r"(?:[,]?\s*(?:sir|jarvis))?[?.!\s]*$", re.I)


def _h_whats_wrong(c, t, m):
    """The live fault in his own words; when nothing is latched, the log
    triage answers instead (the drill-down the fault lane promises)."""
    board = c._svc("faults")
    line = ""
    if board is not None:
        try:
            line = board.describe()
        except Exception:
            log.exception("fault board read failed")
    if line:
        return CommandResult(handled=True, reply=line, speak=True, status="Fault")
    res = _h_log_triage(c, t, m)
    if res is not None:
        return res
    return CommandResult(handled=True, reply=faults_mod.NOTHING_WRONG_LINE,
                         speak=True, status="No faults")


# "Quietly, please" -- the run ledger's narration toggle (jarvis/runwatch.py).
# Deliberately NOT part of quiet_kind: "quiet" is barge-in (cut the speech
# now), this holds the epoch beats for the run that is going and lifts by
# itself when that run ends, so there is nothing left switched off.
_QUIETLY_RX = re.compile(
    r"^(?:narrate quietly|quietly(?:,)? please|quietly|keep it down|"
    r"(?:stop|no more) narrating|don'?t narrate(?: the run| that)?|"
    r"(?:less|fewer) updates)"
    r"(?:[,]?\s*(?:please|sir|jarvis))*[.!\s]*$", re.I)
QUIETLY_LINE = "Quietly it is, sir; I'll tell you when it's done."
QUIETLY_IDLE_LINE = "Nothing is running to narrate, sir."


def _h_quietly(c, t, m):
    """Hold the run narration for the run in progress."""
    wd = c._svc("health_watchdog")
    ledger = getattr(wd, "runs", None)
    if ledger is None:
        return None                      # no ledger: "quietly" is the model's
    if not getattr(ledger, "active", None):
        return CommandResult(handled=True, reply=QUIETLY_IDLE_LINE, speak=True,
                             status="Nothing running")
    ledger.muted = True
    log.info("run narration muted for the current run")
    return CommandResult(handled=True, reply=QUIETLY_LINE, speak=True,
                         status="Narration quiet")


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
    # "good night"/"goodnight" never arrive here: the courtesy entry
    # (REGISTRY index 6) matches them first and never returns None, so only
    # "go to sleep" and "shut down jarvis" reach this handler -- and they
    # mean the same thing, so they get the same wind-down, scene and
    # preview the courtesy path gets.
    started = _start_winddown(c)
    scene = "" if started else _maybe_wind_down(c)  # off unless he asked
    res = _goodnight_preview(c, t)
    if res is not None:
        # The scene line is the only word about the moved desktop; the
        # preview would otherwise swallow it (see _h_courtesy).
        if scene:
            res.reply = f"{scene} {res.reply or ''}".strip()
        return res
    line = "Good night sir. I'll be here when you need me."
    return CommandResult(
        handled=True,
        reply=f"{scene} {line}".strip() if scene else line,
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


# ---- Tier 1 Spotify: "play my liked songs" (jarvis/tools/spotify.py) -----
# LIVE 2026-09-01 19:59:26, by voice: "Play my like songs." went
# local:music -> brain -> gemma4, which picked spotify_liked AND filled
# shuffle=true on its own, and Jarvis announced "Your Liked Songs on
# shuffle, sir" for a request that never said the word.  Two faults in
# that one turn, and this route removes both: a 3-4 s model round trip
# for a request whose shape is fixed, and a model deciding a knob that
# only the utterance may decide.  The tool is forced (force_tool /
# force_args -- the recap / briefing / last-mail pattern) and ``shuffle``
# is wants_shuffle(<what he said>, default=spotify.liked_shuffle), the same
# rule the regex play() path applies, and comes from nowhere else.  The
# model never sees the turn.
#
# Anchored, and the tail may hold only things this route understands (an
# order word, a shuffle word, a Connect device, "please"): "play my liked
# songs by Drake" or "what's in my liked songs" fall through to the router
# and the model -- which cannot set shuffle either any more, because the
# tool schema no longer offers it and the registry drops it when guessed.
_LIKED_WHAT_RX = (
    # "like songs": Whisper drops the d (his 21:14:37 and 19:59:26 turns).
    r"(?:liked?\s+(?:songs|tracks|music)|likes|"
    r"saved\s+(?:songs|tracks|music)|favou?rites?|library|"
    r"(?:songs|tracks|music)\s+(?:that\s+)?i(?:'ve|\s+have)?\s+(?:liked|saved))")
_LIKED_RX = re.compile(
    r"^(?:(?:please|can you|could you|would you|go ahead and)[\s,]+)*"
    # The verb list is a door, not a filter: a phrasing that misses it does
    # NOT fall back to a model that can honour "on shuffle" -- the model's
    # shuffle is reserved -- so "put my liked songs on shuffle" landed on
    # the newest-first default and was announced as such (2026-09-02
    # review).  Bare "put"/"queue"/"stick"/"throw", the trailing particle,
    # and a leading "shuffle play" are all in now; the utterance-derived
    # fallback in the tool spec (jarvis/tools/spotify.py) covers the rest.
    r"(?:(?:shuffle|randomly)\s+)?"
    r"(?:(?:play|put|start|shuffle|queue|stick|throw|fire\s+up)"
    r"(?:\s+(?:on|up))?\s+)?"
    r"(?:(?:all\s+)?(?:of\s+)?(?:my|the|our)\s+)?(?:spotify\s+)?"
    + _LIKED_WHAT_RX +
    r"(?:\s+(?:playlist|collection))?(?:\s+on\s+spotify)?"
    r"(?P<tail>(?:[\s,]+.*)?)$", re.I)
# Tail words that mean nothing to the tool but everything to a sentence.
# Removed BEFORE the order/shuffle words: "in the order I added them" is
# one phrase here, where _INORDER_RX would take "in the order" and leave
# "i added them" behind as a stranger.
_LIKED_FILLER_RX = re.compile(
    r"\b(?:please|now|for me|thanks|thank you|and|but|then|them|it|from the top|"
    r"from the start|from the beginning|"
    r"in the order (?:that )?(?:i|they were) (?:added|saved|liked)(?: them)?|"
    # "in a random order": the shuffle word is read off the WHOLE utterance
    # by wants_shuffle, so the phrase only has to leave the tail empty
    r"in\s+(?:a|an|any)?\s*(?:random|shuffled|mixed[- ]up)\s+order|"
    # "stick my liked songs ON", "turn them back ON" -- a trailing particle,
    # never the "on" of "on my phone" (the device match runs after this)
    r"(?:back\s+)?on(?=\s*$)|"
    r"(?:on|in|with)\s+(?:shuffle|random)(?:\s+mode)?|shuffle mode)\b", re.I)
# "on my phone" / "on hpcomputer" / "on the computer": the device is
# matched case-insensitively by the tool (_match_device runs _norm), so the
# lowercased form the matchers see is enough.  "on repeat" is a mode the
# tool cannot honour here, not a speaker called Repeat.
_LIKED_DEVICE_RX = re.compile(
    r"^(?:on|onto|to|over to|through)\s+(?:my\s+|the\s+)?"
    r"(?P<dev>(?!(?:repeat|loop|spotify)\b)[\w' -]{2,40})$", re.I)


def _liked_tail(tail: str) -> Optional[tuple[str, Optional[str]]]:
    """(leftover, device) for the words after the noun, or None when the
    tail holds something this route does not understand."""
    rest = " " + (tail or "") + " "
    rest = _LIKED_FILLER_RX.sub(" ", rest)
    rest = spotify_mod._INORDER_RX.sub(" ", rest)
    rest = spotify_mod._SHUFFLE_RX.sub(" ", rest)
    rest = re.sub(r"[,\s]+", " ", rest).strip()
    if not rest:
        return "", None
    m = _LIKED_DEVICE_RX.match(rest)
    if not m:
        return None
    return "", m.group("dev").strip(" '-")


def liked_songs_kind(text: str):
    """The match for a Liked-Songs request, else None (falsy)."""
    m = _LIKED_RX.match(str(text or "").strip().rstrip(".!?"))
    if not m:
        return None
    return m if _liked_tail(m.group("tail")) is not None else None


def _h_liked_songs(c, t, m):
    brain = c._svc("brain")
    if brain is None or not hasattr(brain, "chat"):
        return None
    # The tool's own reading of spotify.liked_shuffle when it is wired
    # (services.spotify), the config directly when it is not: either way
    # ONE default, and "he did not say" resolves to it here, not in the
    # model.
    default = getattr(c._svc("spotify"), "liked_shuffle", None)
    if not isinstance(default, bool):
        default = bool(_assistant_get(c, "spotify.liked_shuffle", False))
    args = {"shuffle": bool(spotify_mod.wants_shuffle(t, default=default))}
    parsed = _liked_tail(m.group("tail"))
    device = parsed[1] if parsed else None
    if device:
        args["device"] = device
    log.info("liked songs: forcing spotify_liked %s for %r", args, t)
    brain.chat(t, force_tool="spotify_liked", force_args=args)
    return CommandResult(handled=True, status="Liked Songs…", done=False)


# ---- Tier 1 Spotify: "start my music" (spotify_control resume) -----------
# LIVE 2026-09-01 20:56:42, by voice: "Say hello to my family and then add
# milk to my shopping list and then start playing my Spotify."  The
# greeting clause is not Tier-1, so _try_multi's all-or-nothing rule sent
# the compound to gemma4 whole -- which answered "I've added milk to your
# shopping list, sir, and I'm starting your music now" and called NO tool
# (no 'brain INFO tool' line in the log).  The brain's unbacked-action
# guard (jarvis/brain.py) is the fix for the narration; this route is the
# fix for the request itself: "start playing my Spotify" has one meaning
# and one tool action, spotify_control resume, and the model was the only
# thing between the words and the call.  Forced (force_tool/force_args),
# like liked songs above, so the tool's own "Resumed, sir." is spoken and
# no model turn runs.  Extends the transport-word vocabulary the read-aloud
# steering (_READ_CTL_RX) and the media keys (desktop.MEDIA_PLAY_EXACT)
# already share: those keep bare "play"/"pause"/"resume", this takes the
# forms that NAME the music.
#
# Anchored and narrow: an artist, a genre, a playlist, "liked songs" or any
# tail the route does not understand falls through to the router as
# before ("play Drake", "play some jazz", "play my liked songs" -- the
# route above -- and "play my playlist" are all still the model's).
_MUSIC_NOUN = r"(?:music|tunes|spotify|jams|playback)"
_MUSIC_RESUME_RX = re.compile(
    r"^(?:(?:please|can you|could you|would you|go ahead and|jarvis)[\s,]+)*"
    r"(?:"
    # "start playing my Spotify", "resume my music", "play me some music",
    # "start up the music", "unpause the music" -- a determiner names the
    # music as HIS, which is what separates "start my Spotify" from "start
    # Spotify" (launching the app, the desktop's word)
    r"(?:start|resume|play|unpause|continue|restart|fire up|start up|turn on|put on)"
    r"(?:\s+playing)?(?:\s+me)?\s+(?:my|the|some|our)(?:\s+spotify)?\s+" + _MUSIC_NOUN +
    r"(?:\s+(?:back\s+)?(?:on|up|again))?"
    # "put my music on", "put some music on", "turn the music back on"
    r"|(?:put|turn|switch)(?:\s+me)?\s+(?:my|the|some|our)\s+" + _MUSIC_NOUN +
    r"\s+(?:back\s+)?on"
    # the bare noun after a transport verb: "play music", "play spotify",
    # "resume playback", "start playing spotify"
    r"|(?:play|resume|unpause|continue|start\s+playing|keep\s+playing)\s+" + _MUSIC_NOUN +
    r"|start\s+(?:music|tunes|playback)"
    r")"
    r"(?:[\s,]+(?:please|now|for me|thanks|thank you|again|would you|will you))*"
    r"(?:\s+(?:on|through|over)\s+(?:my\s+|the\s+)?(?P<dev>[\w' -]{2,40}))?$", re.I)
# Words a device cannot be called: "play my music on shuffle" and "put the
# music on repeat" are modes, and the route does not set modes.
_MUSIC_NOT_A_DEVICE_RX = re.compile(
    r"^(?:shuffle|repeat|loop|random|spotify|full|max|low|quiet|mute)\b", re.I)


def music_resume_kind(text: str):
    """The match for a bare "start/resume/play my music" request (no
    artist, no playlist, no Liked Songs), else None (falsy)."""
    t = str(text or "").strip().rstrip(".!?")
    if _LIKED_RX.match(t):
        return None                       # Liked Songs is the route above
    m = _MUSIC_RESUME_RX.match(t)
    if not m:
        return None
    dev = m.group("dev")
    if dev and _MUSIC_NOT_A_DEVICE_RX.match(dev.strip()):
        return None
    return m


def _h_music_resume(c, t, m):
    brain = c._svc("brain")
    if brain is None or not hasattr(brain, "chat"):
        return None
    args = {"action": "resume"}
    dev = (m.group("dev") or "").strip(" '-")
    if dev:
        # matched case-insensitively by the tool (_match_device runs _norm)
        args["device"] = dev
    log.info("music: forcing spotify_control %s for %r", args, t)
    brain.chat(t, force_tool="spotify_control", force_args=args)
    return CommandResult(handled=True, status="Music…", done=False)


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


# ---- the Oracle Cloud VM (jarvis/tools/oracle.py) ------------------------
# OUTBOUND ONLY: four doors out to Hunter's Oracle Linux box `demon-bot`,
# none back in. Every one of them reads the config first and answers ONE
# honest line when the lane is off or unconfigured -- so on a fresh install
# none of this opens a socket. Not a character of the transcript ever reaches
# a shell: a spoken name is resolved against the `oracle.services` table and
# the command is BUILT in oracle.py from a fixed template plus that row's
# unit name, or the words are refused out loud.
_ORACLE_BOX = (r"(?:oracle(?:\s+(?:box|server|vm|cloud))?|demon[\s-]*bot)")
# The nine service names, longest first -- see oracle.service_word_pattern
# for why this is built from the shipped table rather than his live config.
_ORACLE_SVC = oracle_mod.service_word_pattern()
_ORACLE_STATUS_RX = re.compile(
    r"^(?:"
    r"(?:how(?:'s|s| is| are|'re)|what(?:'s|s| is))\s+(?:the\s+|my\s+)?"
    + _ORACLE_BOX + r"(?:\s+(?:doing|looking|getting on))?|"
    r"(?:is|are)\s+(?:the\s+|my\s+)?" + _ORACLE_BOX +
    r"\s+(?:up|running|alive|ok|okay|online|still up|still running)|"
    r"what(?:'s|s| is)\s+(?:running|up)\s+on\s+(?:the\s+|my\s+)?"
    + _ORACLE_BOX + r"|"
    r"(?:check|check on|look at)\s+(?:the\s+|my\s+)?" + _ORACLE_BOX + r"|"
    r"oracle\s+status|status\s+of\s+(?:the\s+|my\s+)?" + _ORACLE_BOX + r"|"
    # the roll-call by its subject rather than its host: nine services, and
    # "are all the bots up" is how a person with nine of them asks
    r"(?:are|is)\s+(?:all\s+)?(?:the\s+|my\s+)?(?:bots|services)\s+"
    r"(?:up|running|alive|ok|okay|online)|"
    r"(?:how(?:'s|s| are|'re))\s+(?:all\s+)?(?:the\s+|my\s+)?"
    r"(?:bots|services)(?:\s+doing)?"
    r")\W*$", re.I)
# "is the server up", "how's the bot": phrasings that name no box at all. He
# asked for them and he has exactly one box, so they answer -- but ONLY once
# the lane is switched on. With oracle.enabled false every handler returns
# None on them and the words go to the model, because answering "the Oracle
# box is switched off in my settings" to a question about a local dev server
# would be a confident wrong answer, and those are the expensive kind.
# _ORACLE_NAMED_RX is that test, and the SERVICE names are deliberately not
# in it: ~/haymaker-digest is a job on THIS machine, so with the lane off
# "how's the haymaker" belongs to the model, not to a server he has not
# told Jarvis about.
_ORACLE_NAMED_RX = re.compile(_ORACLE_BOX, re.I)
_ORACLE_VAGUE_RX = re.compile(
    r"^(?:(?:how(?:'s|s| is)|what(?:'s|s| is))\s+(?:the\s+|my\s+)?"
    r"(?:server|bot)(?:\s+doing)?|"
    r"(?:is|are)\s+(?:the\s+|my\s+)?(?:server|bot)\s+"
    r"(?:up|running|alive|ok|okay|online))\W*$", re.I)
# ONE service by name: "how's the haymaker bot", "is knightfall up",
# "what about monday sync". The captured words still have to RESOLVE to a
# row of his services table -- the regex only narrows the field.
_ORACLE_SERVICE_RX = re.compile(
    r"^(?:"
    r"(?:how(?:'s|s| is| are|'re)|what(?:'s|s| is))\s+(?:the\s+|my\s+)?"
    r"(?P<svc>" + _ORACLE_SVC + r")(?:\s+(?:doing|looking|getting on))?|"
    r"(?:is|are)\s+(?:the\s+|my\s+)?(?P<svc2>" + _ORACLE_SVC + r")\s+"
    r"(?:up|down|running|alive|ok|okay|online|still up|still running)|"
    r"what\s+about\s+(?:the\s+|my\s+)?(?P<svc3>" + _ORACLE_SVC + r")|"
    r"(?:check|check on|look at)\s+(?:the\s+|my\s+)?"
    r"(?P<svc4>" + _ORACLE_SVC + r")|"
    r"(?P<svc5>" + _ORACLE_SVC + r")\s+status"
    r")\W*$", re.I)
# The box or a service is REQUIRED in every log branch: Jarvis has his own
# log ("anything wrong in your log", _h_log_triage), and a bare "check the
# logs" belongs to whichever of the two he was just talking about.
_ORACLE_VERB = r"(?:(?:show|read|check|tail|pull up|get|give)\s+(?:me\s+)?)?"
_ORACLE_LOG_SUBJ = r"(?:" + _ORACLE_SVC + r"|" + _ORACLE_BOX + r")"
_ORACLE_LOGS_RX = re.compile(
    r"^(?:"
    + _ORACLE_VERB + r"(?:the\s+|my\s+)?(?P<svc>" + _ORACLE_LOG_SUBJ +
    r")(?:'s)?\s+logs?|"
    + _ORACLE_VERB + r"(?:the\s+|my\s+)?logs?\s+(?:from|for|on)\s+"
    r"(?:the\s+|my\s+)?(?P<svc2>" + _ORACLE_LOG_SUBJ + r")|"
    r"what\s+(?:do|does)\s+(?:the\s+|my\s+)?(?P<svc3>" + _ORACLE_LOG_SUBJ +
    r")\s+logs?\s+say"
    r")\W*$", re.I)
# The named-action door. A verb and a service, either way round the box may
# be named. The verb set is WIDER than what Jarvis will do on purpose:
# "stop haymaker" has to be refused out loud, not dropped into the model,
# which would leave him thinking it had been stopped.
_ORACLE_VERBS = r"(?:restart|reboot|bounce|reload|stop|start|kill|disable)"
_ORACLE_ACTION_RX = re.compile(
    r"^(?:"
    r"(?:please\s+)?(?P<verb>" + _ORACLE_VERBS + r")\s+(?:the\s+|my\s+)?"
    r"(?P<svc>" + _ORACLE_SVC + r")"
    r"(?:\s+on\s+(?:the\s+|my\s+)?" + _ORACLE_BOX + r")?|"
    r"(?:on\s+)?(?:the\s+|my\s+)?" + _ORACLE_BOX + r"\s*[,:]?\s*"
    r"(?:please\s+)?(?P<verb2>" + _ORACLE_VERBS + r")\s+(?:the\s+|my\s+)?"
    r"(?P<svc2>" + _ORACLE_SVC + r")|"
    # "restart the bot" named exactly one thing when there was one bot.
    # There are nine, so this branch captures no service and the handler
    # asks which -- with the list on the card.
    r"(?:please\s+)?(?P<vague>" + _ORACLE_VERBS + r")\s+(?:the\s+|my\s+)?"
    r"(?:bot|service)s?(?:\s+on\s+(?:the\s+|my\s+)?" + _ORACLE_BOX + r")?"
    r")\W*$", re.I)
# The refusal door, LAST. Anything else said AT the box by name ends here
# with "I only do status, logs and a restart" rather than reaching the model,
# which would otherwise be asked to invent an answer about a server it
# cannot see. The box has to be named, so ordinary talk cannot arrive.
_ORACLE_FREEFORM_RX = re.compile(
    r"^(?:"
    r"(?:on\s+)?(?:the\s+|my\s+)?" + _ORACLE_BOX + r"\s*[,:]?\s*"
    r"(?:please\s+)?(?:run\s+)?(?P<cmd>[a-z][a-z0-9' -]{0,40}?)|"
    r"(?:please\s+)?run\s+(?P<cmd2>[a-z][a-z0-9' -]{0,40}?)\s+on\s+"
    r"(?:the\s+|my\s+)?" + _ORACLE_BOX +
    r")\W*$", re.I)
ORACLE_BUSY_STATUS = "Asking Oracle…"
ORACLE_RESTART_VERBS = ("restart", "reboot", "bounce", "reload")


def _m_oracle_status(t):
    return _ORACLE_STATUS_RX.match(t) or _ORACLE_VAGUE_RX.match(t)


def _oracle_conf(c):
    return oracle_mod.read_config(c._svc("assistant"))


def _oracle_unnamed_and_off(conf, t: str) -> bool:
    """True when the words never named the box and the lane is off -- see
    _ORACLE_NAMED_RX. The handler falls through to the model instead of
    claiming a server it has not been told about."""
    return not conf.enabled and not _ORACLE_NAMED_RX.search(t)


def _oracle_blocked(conf):
    """The CommandResult for an unconfigured lane, or None. One line naming
    exactly what is missing, and no socket opened to find out."""
    reason = oracle_mod.missing_reason(conf)
    if reason is None:
        return None
    return CommandResult(handled=True, reply=reason, speak=True,
                         status="Oracle: not set up")


def _oracle_group(m, *names):
    """The first named group that matched, stripped. The alternations each
    carry their own capture name because Python has no branch reset."""
    for name in names:
        try:
            val = m.group(name)
        except IndexError:
            continue
        if val:
            return str(val).strip()
    return ""


def _oracle_reading(c, conf, then):
    """Fetch the roll-call (or reuse the cached one) and hand it to `then`,
    which says the sentence. The whole-box and single-service questions share
    this: one round trip answers either, and the second question inside
    cache_s answers ON the turn instead of opening another."""
    hit = oracle_mod.cached(conf)
    if hit is not None:
        bus.publish(JarvisReply(text=oracle_mod.card(conf, hit), speak=False))
        return CommandResult(handled=True, speak=True, status="Oracle",
                             reply=then(hit))

    def _run():
        reading, why = oracle_mod.status(conf)
        if reading is None:
            line = oracle_mod.fail_line(conf, why)
            bus.publish(Status(text="Oracle: no answer", kind="warn"))
            c._speak(line)
            bus.publish(JarvisReply(text=line, speak=False))
            return
        bus.publish(JarvisReply(text=oracle_mod.card(conf, reading), speak=False))
        bus.publish(Status(text="Oracle", kind="ok"))
        c._speak(then(reading))

    c._bg(_run)
    return CommandResult(handled=True, status=ORACLE_BUSY_STATUS, done=False)


def _h_oracle_status(c, t, m):
    conf = _oracle_conf(c)
    if _oracle_unnamed_and_off(conf, t):
        return None                    # bare "is the server up": see above
    blocked = _oracle_blocked(conf)
    if blocked is not None:
        return blocked
    return _oracle_reading(c, conf, lambda r: oracle_mod.speak_line(conf, r))


def _h_oracle_service(c, t, m):
    """ONE service by name. The words matched a SHIPPED service name, but
    they still have to resolve against his live table -- and if they do not,
    this falls through to the model rather than answering for a box that
    never heard of it."""
    spoken = _oracle_group(m, "svc", "svc2", "svc3", "svc4", "svc5")
    conf = _oracle_conf(c)
    if _oracle_unnamed_and_off(conf, t):
        return None
    svc = oracle_mod.resolve_service(conf, spoken)
    if svc is None:
        return None
    blocked = _oracle_blocked(conf)
    if blocked is not None:
        return blocked
    return _oracle_reading(c, conf,
                           lambda r: oracle_mod.service_line(conf, r, svc))


def _oracle_run_action(c, conf, svc, action: str, command: str):
    """Run one allow-listed action in the background and say what came back.
    ``command`` came from oracle.action_command and nowhere else."""
    def _run():
        ok, text = oracle_mod.run_action(conf, command)
        if not ok:
            bus.publish(Status(text="Oracle: no answer", kind="warn"))
            c._speak(text)
            bus.publish(JarvisReply(text=text, speak=False))
            return
        if action == oracle_mod.ACTION_LOGS:
            line, sheet = oracle_mod.log_summary(svc, text)
        else:
            sheet = text[:oracle_mod.CARD_CHAR_CAP]
            line = oracle_mod.restart_line(svc, text)
        if sheet:
            bus.publish(JarvisReply(text=sheet, speak=False))
        bus.publish(Status(text=f"Oracle: {action}"[:30], kind="ok"))
        c._speak(line)
        # The roll-call cache is stale the moment an action ran: a restart
        # reported off a 20-second-old reading would be the one wrong answer
        # this lane must never give.
        oracle_mod.clear_cache()

    c._bg(_run)
    return CommandResult(handled=True, status=f"Oracle: {action}"[:30],
                         done=False)


def _h_oracle_logs(c, t, m):
    conf = _oracle_conf(c)
    if _oracle_unnamed_and_off(conf, t):
        return None
    blocked = _oracle_blocked(conf)
    if blocked is not None:
        return blocked
    spoken = _oracle_group(m, "svc", "svc2", "svc3")
    svc = oracle_mod.resolve_service(conf, spoken)
    if svc is None:
        # "show me the oracle logs" names the box but no service, and nine
        # journals are not one answer. Ask, with the list on the card.
        if _ORACLE_NAMED_RX.search(spoken or t):
            bus.publish(JarvisReply(text=oracle_mod.services_card(conf),
                                    speak=False))
            return CommandResult(handled=True, speak=True, status="Which one?",
                                 reply=oracle_mod.WHICH_SERVICE_LINE)
        return None
    command = oracle_mod.action_command(conf, oracle_mod.ACTION_LOGS, svc.unit)
    if command is None:                       # cannot happen; refuse anyway
        return CommandResult(handled=True, speak=True, status="Not on the list",
                             reply=oracle_mod.UNKNOWN_ACTION_LINE)
    return _oracle_run_action(c, conf, svc, oracle_mod.ACTION_LOGS, command)


def _h_oracle_action(c, t, m):
    conf = _oracle_conf(c)
    if _oracle_unnamed_and_off(conf, t):
        return None                    # bare "restart the bot": see above
    verb = _oracle_group(m, "verb", "verb2", "vague").lower()
    spoken = _oracle_group(m, "svc", "svc2")
    svc = oracle_mod.resolve_service(conf, spoken)
    blocked = _oracle_blocked(conf)
    if svc is None:
        if not spoken:                 # "restart the bot": which of nine?
            if blocked is not None:
                return blocked
            bus.publish(JarvisReply(text=oracle_mod.services_card(conf),
                                    speak=False))
            return CommandResult(handled=True, speak=True, status="Which one?",
                                 reply=oracle_mod.WHICH_RESTART_LINE)
        return None                    # not his box's service
    if blocked is not None:
        return blocked
    if verb not in ORACLE_RESTART_VERBS:
        # "stop haymaker" is refused OUT LOUD rather than dropped: silence
        # would leave him believing the bot had been stopped.
        log.info("oracle: %r is not an action I do", verb)
        return CommandResult(handled=True, speak=True, status="Not on the list",
                             reply=oracle_mod.UNKNOWN_ACTION_LINE)
    command = oracle_mod.action_command(conf, oracle_mod.ACTION_RESTART,
                                        svc.unit)
    if command is None:
        return CommandResult(handled=True, speak=True, status="Not on the list",
                             reply=oracle_mod.UNKNOWN_ACTION_LINE)
    # The same rung a bulk cancel uses: read it back NAMING the service, and
    # the next yes runs it (_try_destructive_confirm). Restarting one of his
    # nine bots is not a thing to do on a maybe-heard word.
    line = oracle_mod.READ_BACK_LINE.format(name=svc.name)
    c.stash_destructive(
        lambda: _oracle_run_action(c, conf, svc, oracle_mod.ACTION_RESTART,
                                   command), line)
    return CommandResult(handled=True, reply=line, speak=True,
                         status="Confirm?")


def _h_oracle_freeform(c, t, m):
    """The last door: words said AT the box that are not one of the three
    things Jarvis does there. Refused out loud, never guessed at and never
    handed to the model -- "run whatever on the oracle box" ends here."""
    conf = _oracle_conf(c)
    if _oracle_unnamed_and_off(conf, t):
        return None
    said = _oracle_group(m, "cmd", "cmd2")
    if not oracle_mod.key_tokens(said):
        # Nothing but the box's own name ("the oracle box"): that is not an
        # instruction, so it goes to the model rather than being refused.
        return None
    blocked = _oracle_blocked(conf)
    if blocked is not None:
        return blocked
    svc = oracle_mod.resolve_service(conf, said)
    log.info("oracle: %r is not on the allow-list", said)
    reply = oracle_mod.UNKNOWN_ACTION_LINE if svc is not None \
        else oracle_mod.unknown_service_line(conf, said)
    return CommandResult(handled=True, speak=True, status="Not on the list",
                         reply=reply)


# ---- HPCOMPUTER: files and a short question list (jarvis/tools/remote.py) --
# Five doors out to his other machine, none back in. Two of them move a file
# and both are READ BACK before anything opens; one answers a fixed list of
# read-only questions unattended; one refuses everything else out loud.
#
# The host words are deliberately narrow, and "desktop" ALONE is not among
# them. His own phrasing for the mail lane is "this file ... on my desktop",
# where "my desktop" is the FOLDER the file sits in -- so a bare "desktop"
# must never be read as the destination machine. "Put this on my desktop"
# stays a local request and falls through to the model; only "the desktop
# MACHINE" (or HPCOMPUTER, or "the other machine") names the host.
_HPC = (r"(?:hp\s*computer|the\s+hp\b|"
        r"(?:my|the)\s+(?:desktop|other)\s+(?:machine|computer|box|pc)|"
        r"(?:my|the)\s+other\s+machine)")
_HPC_RX = re.compile(_HPC, re.I)

# A file phrase: "this file", "the budget spreadsheet", "budget.xlsx".
_FILE_WORD = r"(?P<what>[\w][\w '.\-()+]{0,80}?)"

# "is HPCOMPUTER up", "is it awake", "how's HPCOMPUTER"
_REMOTE_STATUS_RX = re.compile(
    r"^(?:"
    r"(?:is|are)\s+" + _HPC + r"\s+(?:up|on|awake|alive|online|running|"
    r"ok|okay|there)|"
    r"(?:how(?:'s|s| is))\s+" + _HPC + r"(?:\s+(?:doing|looking))?|"
    r"(?:check|check on|ping)\s+" + _HPC + r"|"
    r"(?:can|could)\s+you\s+(?:see|reach)\s+" + _HPC +
    r")\W*$", re.I)

# The read-only question list. The KEY is chosen by these words; the command
# itself is a constant in remote.QUERIES, so a misheard word can only pick a
# different question from the table or none at all.
_REMOTE_QUERY_RX = re.compile(
    r"^(?:what(?:'s|s| is)|how(?:'s|s| is)|who(?:'s|s| is)|show me)\s+"
    r"(?:the\s+|my\s+)?(?P<q>disk|space|drive|storage|room|uptime|load|"
    r"logged\s+in|logged\s+on|on\s+it|inbox|in\s+the\s+inbox)\b"
    r".{0,20}?\bon\s+" + _HPC + r"\W*$", re.I)

# PUSH: "put the budget on HPCOMPUTER", "send this file to HPCOMPUTER".
_REMOTE_PUSH_RX = re.compile(
    r"^(?:put|copy|send|move|push|transfer)\s+"
    r"(?:the\s+|my\s+|this\s+|that\s+)?" + _FILE_WORD +
    r"(?:\s+file)?\s+(?:on(?:to)?|to|over\s+to|across\s+to)\s+"
    + _HPC + r"\W*$", re.I)

# PULL: "get the budget from HPCOMPUTER", "grab that off HPCOMPUTER".
# The optional trailing folder word is an ALLOW-LIST key, not a path.
_REMOTE_PULL_RX = re.compile(
    r"^(?:get|grab|fetch|bring|pull|copy|download)\s+(?:me\s+)?"
    r"(?:the\s+|my\s+|this\s+|that\s+)?" + _FILE_WORD +
    r"(?:\s+file)?\s+(?:from|off(?:\s+of)?|out\s+of)\s+" + _HPC +
    r"(?:(?:'s)?\s+(?P<where>outbox|desktop|downloads))?\W*$", re.I)

# The refusal door, LAST: anything else aimed at the host that reads like an
# instruction. "run the build on HPCOMPUTER", "delete the logs on the HP".
# Three shapes, because an order can put the host anywhere: "run the build ON
# HPCOMPUTER", "HPCOMPUTER, run the build", and the bare "shut down
# HPCOMPUTER" that names no preposition at all -- that last one is how a
# reboot gets said, so leaving it out would leave the loudest order unrefused.
_REMOTE_FREEFORM_RX = re.compile(
    r"^(?:(?P<cmd>.{2,120}?)\s+on\s+" + _HPC + r"|"
    r"(?:on\s+)?" + _HPC + r"[,:]?\s+(?P<cmd2>.{2,120}?)|"
    r"(?P<cmd3>.{2,120}?)\s+" + _HPC + r")\W*$", re.I)

# Words that make a phrase an ORDER rather than a mention. "how's HPCOMPUTER"
# is a question the status door already took; "is HPCOMPUTER a good machine"
# is conversation and belongs to the model.
#
# ANCHORED at the head of the clause, which an unanchored `search` was not,
# and that difference is the whole of the door's manners. An order is an
# IMPERATIVE -- the verb comes first. A clause that merely CONTAINS one of
# these words is a question or a report, and every one of these was being
# refused out loud: "did you install anything on the HP", "have you run the
# tests on the HP", "I need to update the HP", "the build failed on the HP",
# "remind me to run the backup on the HP". The last is worse than noise --
# it is a reminder he asked for and did not get. Same rule _SEND_NOT_RX
# applies to the mail lane, stated as an anchor rather than a veto list.
_REMOTE_ORDER_RX = re.compile(
    r"^(?:(?:please|just|go\s+ahead\s+and|can\s+you|could\s+you|"
    r"would\s+you|will\s+you)[,\s]+)*"
    r"(?:run|start|stop|restart|reboot|shut\s*down|kill|delete|remove|rm\b|"
    r"install|update|upgrade|build|make|compile|deploy|launch|open|execute|"
    r"format|wipe|clear|empty|move|rename|chmod|sudo)\b", re.I)

_QUERY_KEYS = {"disk": "disk", "space": "disk", "drive": "disk",
               "storage": "disk", "room": "disk", "uptime": "up",
               "load": "load", "logged in": "who", "logged on": "who",
               "on it": "who", "inbox": "inbox", "in the inbox": "inbox"}


try:                                        # the mail lane's phrase layer
    from jarvis import filephrase as _filephrase
except ImportError:                         # pragma: no cover - lane dropped
    _filephrase = None


def _remote_conf(c):
    return remote_mod.read_config(c._svc("assistant"))


def _remote_resolve_local(said: str, conf):
    """Spoken words -> ONE local file, the rivals, or a reason.

    COORDINATION, not duplication. `jarvis/tools/filepick.py` is the shared
    resolver both lanes rank and vet files with; the mail lane's
    `jarvis/filephrase.py` is a PHRASE layer on top of it that understands
    the shapes a bare name cannot -- "that file on my desktop" (a folder is
    the handle, there is no name), "the PDF I just downloaded" (a type and a
    recency). Those are exactly the phrases he uses for this lane too, so it
    is used here rather than re-derived, and the two lanes cannot disagree
    about what "that file" means.

    ONE argument differs and it is deliberate: ``allow_explicit_outside`` is
    FALSE here. The mail lane lets him attach a path he names outright from
    anywhere (guarded by its DENY_ROOTS); this lane keeps filepick's hard
    containment, because a push has a second machine's filesystem on the far
    end and "put /etc/... on HPCOMPUTER" should not be sayable at all.

    Falls back to the shared resolver alone if the phrase layer is absent,
    so this lane still stands on its own.
    """
    if _filephrase is not None:
        return _filephrase.resolve(said, roots=conf.local_roots,
                                   max_mb=conf.max_mb,
                                   allow_explicit_outside=False)
    return filepick.pick(said, roots=conf.local_roots, max_mb=conf.max_mb)


def _remote_blocked(c, conf):
    """One honest line naming exactly what is missing, and no socket opened
    to find it out. None when the lane is ready to try."""
    reason = remote_mod.missing_reason(conf)
    if not reason:
        return None
    return CommandResult(handled=True, speak=True,
                         reply=remote_mod.fail_line(conf, reason),
                         status=f"{conf.name}: not set up")


def _remote_fail(conf, reason: str, status: str = ""):
    return CommandResult(handled=True, speak=True,
                         reply=remote_mod.fail_line(conf, reason),
                         status=status or f"{conf.name}: {reason}")


def _h_remote_status(c, t, m):
    """Is it there? Answered from the LOCAL tailnet view first, so a machine
    that is off or has never joined costs no socket and no wait -- and gets
    a different sentence from one that is merely slow. Those three are not
    the same problem and he should not have to guess which he has."""
    conf = _remote_conf(c)
    blocked = _remote_blocked(c, conf)
    if blocked is not None:
        return blocked
    state = remote_mod.tailnet_state(conf)
    if state == "absent":
        return _remote_fail(conf, "off-tailnet", f"{conf.name}: absent")
    if state == "offline":
        return _remote_fail(conf, "asleep", f"{conf.name}: asleep")
    res = remote_mod.ask(conf, "up")
    if not res.ok:
        return _remote_fail(conf, res.reason)
    up = " ".join((res.out or "").split())[:120]
    line = f"{conf.name} is up, sir" + (f" -- {up}." if up else ".")
    return CommandResult(handled=True, reply=line, speak=True,
                         status=f"{conf.name}: up")


def _h_remote_query(c, t, m):
    """One row of the read-only table. Unattended by design: the spoken words
    choose a KEY, never a command."""
    conf = _remote_conf(c)
    blocked = _remote_blocked(c, conf)
    if blocked is not None:
        return blocked
    said = " ".join((m.group("q") or "").lower().split())
    key = _QUERY_KEYS.get(said)
    if not key:
        return None
    res = remote_mod.ask(conf, key)
    if not res.ok:
        return _remote_fail(conf, res.reason)
    body = " ".join((res.out or "").split())[:200]
    if not body:
        return CommandResult(handled=True, speak=True,
                             status=f"{conf.name}: nothing",
                             reply="Nothing to report there, sir.")
    say = remote_mod.QUERIES[key]["say"]
    return CommandResult(handled=True, speak=True, status=f"{conf.name}: {key}",
                         reply=f"On {conf.name}, {say}: {body}.")


def _remote_ask_which(c, conf, names, resume):
    """More than one file could be meant. ASK -- never pick the newer one.

    Parked in the shared "Which one?" slot (``stash_filepick``) so the
    answer can actually be heard: without it the question was spoken, the
    follow-up microphone got the short window, and "the second one" was
    classified as background chat and dropped in silence.
    """
    line = (f"I've more than one that could be, sir: "
            f"{filepick.describe(names)}. Which one?")
    c.stash_filepick(names, resume)
    return CommandResult(handled=True, reply=line, speak=True,
                         status="Which one?")


def _h_remote_push(c, t, m):
    """Send ONE local file to the host's inbox, after reading it back.

    Read back EVERY time, not only on a shaky transcript the way an alarm
    is: an alarm set wrong is an annoyance, and this puts a file of his on
    another machine, where it cannot be taken back."""
    conf = _remote_conf(c)
    blocked = _remote_blocked(c, conf)
    if blocked is not None:
        return blocked
    said = (m.group("what") or "").strip()

    def _armed(path):
        """Read the transfer back and wait. STRICT: the yes that spends this
        is parse_send_answer's end-anchored grammar, not parse_yes_no --
        a file on another machine is as irreversible as one in an email."""
        if not remote_mod.inbox_target(conf, path.name):
            return _remote_fail(conf, "odd-name", "Bad name")
        question = f"Send {path.name} to {conf.name}'s inbox, sir?"

        def _run():
            res = remote_mod.push(conf, path)
            if not res.ok:
                return _remote_fail(conf, res.reason)
            return CommandResult(handled=True, speak=True,
                                 status=f"Sent to {conf.name}",
                                 reply=f"{path.name} is on {conf.name}, sir.")

        c.stash_destructive(_run, question, strict=True)
        return CommandResult(handled=True, reply=question, speak=True,
                             status="Confirm?")

    pick = _remote_resolve_local(said, conf)
    if pick.ambiguous:
        def _resume(path):
            again = _remote_resolve_local(str(path), conf)
            if not again.ok:
                size_mb = getattr(again, "size", 0) / (1024 * 1024)
                return CommandResult(
                    handled=True, speak=True, status="No such file",
                    reply=filepick.reason_line(again.reason, conf.max_mb,
                                               size_mb))
            return _armed(again.path)
        return _remote_ask_which(c, conf, pick.candidates, _resume)
    if not pick.ok:
        size_mb = getattr(pick, "size", 0) / (1024 * 1024)
        return CommandResult(handled=True, speak=True, status="No such file",
                             reply=filepick.reason_line(pick.reason,
                                                        conf.max_mb, size_mb))
    return _armed(pick.path)


def _h_remote_pull(c, t, m):
    """Fetch ONE file from an allow-listed remote folder, after reading back
    the name the REMOTE reported -- not the one that was said."""
    conf = _remote_conf(c)
    blocked = _remote_blocked(c, conf)
    if blocked is not None:
        return blocked
    said = (m.group("what") or "").strip()
    key = (m.group("where") or "outbox").lower()
    names, why = remote_mod.list_remote(conf, key)
    if why:
        return _remote_fail(conf, why)
    if not names:
        return CommandResult(handled=True, speak=True, status="Empty",
                             reply=f"There's nothing in {conf.name}'s "
                                   f"{key}, sir.")

    def _armed(name: str):
        dest = remote_mod.pull_target(conf, name)
        question = (f"Bring {name} from {conf.name} to your "
                    f"{dest.parent.name}, sir?")

        def _run():
            res = remote_mod.pull(conf, key, name)
            if not res.ok:
                return _remote_fail(conf, res.reason)
            return CommandResult(handled=True, speak=True, status="Fetched",
                                 reply=f"{name} is on your "
                                       f"{dest.parent.name}, sir.")

        # STRICT, for the same reason the push is: this one lands a
        # stranger's file on HIS disk and can overwrite nothing, but it is
        # still a transfer he must have actually said yes to.
        c.stash_destructive(_run, question, strict=True)
        return CommandResult(handled=True, reply=question, speak=True,
                             status="Confirm?")

    pick = remote_mod.match_remote(said, names)
    if pick.ambiguous:
        return _remote_ask_which(
            c, conf, pick.candidates,
            lambda path: _armed(Path(path).name))
    if not pick.ok:
        return CommandResult(handled=True, speak=True, status="No such file",
                             reply=f"I can't see anything by that name in "
                                   f"{conf.name}'s {key}, sir.")
    return _armed(pick.path.name)


def _h_remote_freeform(c, t, m):
    """The last door, and the one that defines the lane: an instruction
    aimed at the other machine that is not a file move or a listed question
    is REFUSED, out loud, and never handed to the model.

    This is the whole safety argument in one function. A voice channel with
    a measurable false-accept rate cannot be given a shell on a second
    machine -- "delete the logs on the HP" and "delete the block on the HP"
    differ by one phoneme, and only one of them is recoverable. The useful
    part of a remote shell is already covered by the read-only table above;
    what is left is unbounded, so it does not exist.
    """
    conf = _remote_conf(c)
    said = _oracle_group(m, "cmd", "cmd2", "cmd3")
    if not said or not _REMOTE_ORDER_RX.match(said.strip()):
        # A mention, not an order ("the music's playing on HPCOMPUTER"):
        # not this lane's business, so it goes to the model.
        return None
    log.info("remote: refusing a free-form order aimed at %s", conf.name)
    return CommandResult(handled=True, speak=True, status="Refused",
                         reply=remote_mod.FREEFORM_REFUSAL.format(
                             name=conf.name))


# ---- Email a file (jarvis/outbox.py, 2026-09-02) --------------------------
# "Email this file to this person from this location on my desktop", which is
# how he asked for it. Two irreversible things happen at once -- a file
# leaves the machine, and it leaves it wearing one of his three identities --
# so this family arms NOTHING on the first utterance. It builds a draft,
# reads back the file, the size, the address and the account, and waits
# (_try_send_confirm). Anything it is not sure of becomes a question.
#
# Placed AFTER the HPCOMPUTER family on purpose. "Send the budget to
# HPCOMPUTER" is a transfer, not an email, and _REMOTE_PUSH_RX must have the
# first claim on it; _SEND_NOT_RX repeats the guard by name so the ordering
# is belt and braces rather than the only thing holding it.
_SEND_OPENER = (r"^(?:jarvis[,\s]+)?"
                r"(?:(?:please|can you|could you|would you|will you|"
                r"i(?:'d| would) like you to|i want you to|"
                r"go ahead and)[,\s]+)*")
_SEND_VERB = r"(?:e-?mail|send|share|forward on|shoot|fire)"
# Two shapes, and only two:
#   A  "email <the lab report> to <Heather> [from my <school> account]"
#      ("share <X> WITH <Y>" is the same shape; "with" is in the
#      preposition list because that is how a share is phrased, and the
#      worst it can do is make an odd sentence ask who the recipient is.)
#   B  "send <Heather> <the lab report>"
# B is the loose one -- "send me the weather" fits it perfectly -- so its
# handler refuses to act unless the recipient RESOLVES to a real address in
# the people book or the contacts map. A is anchored by an explicit "to".
_SEND_FILE_RX = re.compile(
    _SEND_OPENER + _SEND_VERB + r"\s+"
    r"(?:(?P<file_a>\S.*?)\s+(?:to|over to|across to|with)\s+(?P<who_a>\S.*?)"
    r"|(?P<who_b>[a-z][\w'.\-]*(?:\s+[a-z][\w'.\-]*)?)\s+"
    r"(?P<file_b>(?:the|my|that|this|a|an)\s+\S.*?))"
    r"(?:\s+(?:from|using|via|out of|off)\s+(?:my\s+|the\s+)?"
    r"(?P<acct>[\w'\-]+(?:\s+[\w'\-]+)?)\s+"
    r"(?:account|address|mailbox|e-?mail))?"
    r"[\s,.!?]*$", re.I)

# The NEGATIVE table. He says "send", "email" and "file" in ordinary
# sentences all day, and every line here is one of those sentences. A hit
# returns None, so the utterance keeps whatever meaning it already had --
# usually the model's.
_SEND_NOT_RX = re.compile(
    # 1. A QUESTION or a report about sending, not an order to send. The
    #    polite openers (can you / could you / would you / please) are
    #    consumed by _SEND_OPENER above and are deliberately absent here.
    r"^(?:did|do|does|has|have|had|is|are|was|were|should|shall|am|ain'?t|"
    r"when|what|why|how|who|whom|whose|where|which)\b"
    # 2. NARRATION. "I need to send the lab report to Heather" is him
    #    thinking out loud about a chore, not handing it to me.
    r"|^(?:i|we)\s+(?:need|want|have|had|ought|meant|forgot|should|must|"
    r"will|'ll|am|was|might|may|could|would|still|just)\b"
    r"|\b(?:already |just )?(?:sent|emailed|e-mailed|forwarded|mailed)\b"
    # 3. ANOTHER CHANNEL. Texts and calls were ruled OUT of this queue by
    #    name ("email send with read-back, NO texts/calls"), and Discord
    #    and Spotify are other lanes in this same file.
    r"|" + _SEND_VERB + r"\s+(?:\w+\s+){0,3}?"
    r"(?:a |an |the |my |him |her |them |me )?"
    r"(?:text|texts|sms|imessage|dm|voicemail|whatsapp)\b"
    r"|\bto\s+(?:discord|slack|whatsapp|my\s+phone|the\s+group\s+chat)\b"
    # 4. HPCOMPUTER. The remote lane owns every phrasing that names the
    #    other machine; _HPC is its own definition, shared so the two
    #    cannot drift apart.
    r"|\b(?:to|onto|on|over to|across to)\s+" + _HPC +
    # 5. MAIL VERBS THIS IS NOT. Replying, forwarding and unsubscribing all
    #    act on a message that already exists; this feature makes a new one.
    r"|^(?:reply|respond|forward|unsubscribe|archive)\b"
    r"|\breply\s+to\b|\brespond\s+to\b"
    # 6. "file" as a VERB. "File that away", "file a report".
    r"|\bfile\s+(?:it|that|this|them|these)?\s*(?:away|under|a |an )\b"
    # 7. Objects that are not files. Each of these fits shape A perfectly
    #    ("send my location to Heather") and none of them is an attachment.
    r"|" + _SEND_VERB + r"\s+(?:\w+\s+){0,2}?"
    r"(?:a |an |the |my |your )?"
    r"(?:reminder|invite|invitation|meeting|link|url|password|money|"
    # `apolog` was the one stem here, and \b after it never matched:
    # "apolog" + "y" is not a word boundary, so "send an apology to my
    # professor" reached the send lane and was answered "I can't find a
    # file by that name, sir." Both endings, spelled out.
    r"payment|song|track|playlist|location|weather|apolog(?:y|ies)|"
    r"regards|love|thanks|note to self)\b",
    re.I)

# The read-back's answer, with its OWN end-anchored grammar and NOT
# parse_yes_no. parse_yes_no is a word BAG that waives its overheard-speech
# guard whenever the first word is a yes word, which is how the ten-word
# "Yeah, so you should be able to look that up." (jarvis.log.1:19499, a real
# line that answered nothing) read as a yes and delivered a whole briefing.
# Delivering a briefing by accident costs a briefing. Sending a file by
# accident cannot be undone, so this grammar is the briefing offer's shape
# and a NARROWER vocabulary: an answer here is the word, a courtesy, and
# nothing else.
# The separator is [.!?,\s]+ and not [,\s]+ because Whisper punctuates:
# "Yes. Thank you." is one of his own logged answers (jarvis.log.1:17109)
# and a comma-only separator made the full stop end the sentence, so a
# clean yes fell through to the silent-drop branch while "yes, thank you"
# worked. The words that may follow are unchanged -- this widens the
# PUNCTUATION, not the vocabulary.
_SEND_TAIL = (r"(?:[.!?,\s]+(?:jarvis|sir|please|thanks|thank you|now|then|"
              r"it|that|send it|send that|go ahead|do it))*[?.!]*$")
# "ok" / "okay" / "sure" / "alright" are deliberately ABSENT. They are the
# words a man says while still reading the read-back, and this is the one
# question in the app where "probably yes" must not be enough. They are
# caught by _SEND_MAYBE_RX below and asked again rather than dropped in
# silence -- silence is how he learns the feature does not work.
_SEND_YES_RX = re.compile(
    r"^(?:jarvis[,\s]+)?"
    r"(?:yes|yeah|yep|yup|aye|affirmative|correct|confirmed?|certainly|"
    r"absolutely|definitely|of course|go ahead|do it|send it|send that|"
    r"send it now|please do|that'?s right|fire away|off you go)"
    + _SEND_TAIL, re.I)
_SEND_NO_RX = re.compile(
    r"^(?:jarvis[,\s]+)?(?:no[,\s]+)?"
    r"(?:no|nope|nah|negative|don'?t|do not|stop|cancel|abort|"
    r"not now|not yet|not that one|wrong one|wrong file|wrong person|"
    r"hold on|hold off|wait|never ?mind|forget it|scratch that|leave it|"
    r"no thanks|no thank you|that'?s wrong)" + _SEND_TAIL, re.I)
_SEND_MAYBE_RX = re.compile(
    r"^(?:jarvis[,\s]+)?"
    r"(?:ok(?:ay)?|alright|all right|sure|fine|right|very well|mhm|mm|"
    r"uh huh|i guess|i suppose|maybe|probably|whatever|sounds good|"
    r"i think so|if you like|why not)" + _SEND_TAIL, re.I)


# ---- "Which one, sir?" ---------------------------------------------------
# BOTH file lanes ask this ("I've 2 that could be the lab report, sir: lab
# report.pdf or lab report final.pdf. Which one?") and until now neither
# could hear the answer: prepare() returned a question with no draft, so
# nothing was parked, question_open() stayed False, the app sized the
# follow-up microphone at 4 s, and "the final one" was then classified as
# background chat and dropped in SILENCE. That is the same failure his
# issue #8 was about, and it lands on the exact case the ambiguous branch
# was written for -- lab_report.pdf against lab_report_final.pdf.
#
# So the offer is parked, like every other question Jarvis asks, and this
# is its grammar. Deliberately narrow: an ordinal, or a name that scores
# clearly against ONE of the candidates it just read out. Anything else
# drops the slot and keeps its own meaning -- a wrong pick here sends the
# wrong file, which is the mistake the question exists to prevent.
FILEPICK_TTL_S = 45.0

_PICK_ORDINALS = {"first": 0, "1st": 0, "one": 0, "1": 0,
                  "second": 1, "2nd": 1, "two": 1, "2": 1, "other": 1,
                  "third": 2, "3rd": 2, "three": 2, "3": 2,
                  "fourth": 3, "4th": 3, "four": 3, "4": 3,
                  "fifth": 4, "5th": 4, "five": 4, "5": 4}
_PICK_ORDINAL_RX = re.compile(
    r"^(?:jarvis[,\s]+)?(?:(?:the|number|no\.?|option)\s+)*"
    r"(?P<n>first|second|third|fourth|fifth|last|latest|newest|other|"
    r"1st|2nd|3rd|4th|5th|one|two|three|four|five|[1-5])"
    r"(?:\s+one)?(?:[,\s]+(?:please|sir|jarvis|thanks))*[\s,.!?]*$", re.I)
_PICK_CANCEL_RX = re.compile(
    r"^(?:jarvis[,\s]+)?(?:no|nope|nah|neither|none|not (?:that|those|"
    r"either|any)|cancel|stop|forget it|never ?mind|leave it|"
    r"don'?t bother|skip it)\b", re.I)


def pick_from_answer(text, candidates) -> tuple:
    """(the candidate he named, was it a near miss).

    Two readings and no third. An ORDINAL ("the second one", "the last
    one", "the other one") indexes the list exactly as it was read out. A
    NAME is scored against the candidates with the same scorer that offered
    them, and has to beat its nearest rival by more than
    ``filephrase.TIE_SCORE`` -- the very band that called these two
    ambiguous in the first place. Anything narrower would let him repeat
    the ambiguous phrase and get a guess, which is the mistake the question
    exists to prevent.

    (None, True) is the near miss: he was plainly naming one of them and
    the margin was not there. The caller asks once more rather than
    dropping it in silence. (None, False) is "not an answer at all".
    """
    said = " ".join(str(text or "").split())
    cands = [Path(c) for c in (candidates or ())]
    if not said or not cands:
        return None, False
    m = _PICK_ORDINAL_RX.match(said)
    if m:
        word = m.group("n").lower()
        if word in ("last", "latest", "newest"):
            return cands[-1], False
        # "the other one" is only meaningful when there are exactly two.
        if word == "other" and len(cands) != 2:
            return None, True
        idx = _PICK_ORDINALS.get(word)
        if idx is not None and idx < len(cands):
            return cands[idx], False
        return None, True
    trimmed = re.sub(r"^(?:jarvis[,\s]+)?(?:the\s+)?", "", said, flags=re.I)
    trimmed = re.sub(r"\s+(?:one|file|please|sir)\b[\s,.!?]*$", "", trimmed,
                     flags=re.I).strip(" ,.!?")
    if not trimmed:
        return None, False
    scored = sorted(((filepick.score(trimmed, c.name), c) for c in cands),
                    key=lambda t: (-t[0], len(t[1].name), t[1].name))
    if scored[0][0] < filepick.FUZZY_FLOOR:
        return None, False
    tie = getattr(_filephrase, "TIE_SCORE", 0.12) if _filephrase else 0.12
    if len(scored) > 1 and scored[1][0] > scored[0][0] - tie:
        return None, True
    return scored[0][1], False


def parse_send_answer(text) -> Optional[bool]:
    """True / False / None for a read-back answer. None is "not an answer".

    NO is tested first: "no, send it" is a contradictory sentence and the
    safe reading of it is the one where nothing leaves the machine.
    """
    t = " ".join(str(text or "").split())
    if not t:
        return None
    if _SEND_NO_RX.match(t):
        return False
    if _SEND_YES_RX.match(t):
        return True
    return None


def _send_file_pieces(c, t, m=None):
    """The pieces of a send-a-file request -- or None, meaning this lane has
    no claim on the sentence and it keeps whatever meaning it already had.

    ONE claim rule, used by the handler AND by the intent-gate probe
    (``Commander._match_assistant``), so the two cannot disagree about what
    counts as a send. The gate is why it matters: a Tier-1 name match turns
    the classifier OFF, and the raw ``_SEND_FILE_RX`` is far too generous to
    be that switch on its own.

    Three tests, cheapest first:

    1. the NEGATIVE table gets first refusal, as it always did -- but now
       BEFORE the gate is switched off rather than after, so "send my
       regards to Heather" and "send a text to Heather" are background
       chat again instead of sentences the model answers out loud;
    2. the RECIPIENT is resolved BEFORE the file, because that is the half
       that says whether the sentence was addressed to Jarvis at all.
       "Send the kids to bed", "send flowers to my mom", "share my screen
       with the class" all fit shape A perfectly and name nobody he can
       write to. Shape B always demanded this; shape A never did, and it
       claimed 28 of 33 everyday sentences measured against it;
    3. a recipient that does not resolve is fatal -- UNLESS the file phrase
       names something really on his disk. That exception is the whole
       reason "email the biosensors handout to Dana" is still answered
       "I've no address for Dana, sir. What is it?" instead of vanishing:
       he named a real file, so he was plainly talking to me.
    """
    if _SEND_NOT_RX.search(t):
        return None
    cfg = c._svc("assistant")
    if cfg is None:
        return None
    lm = m or _SEND_FILE_RX.match(t)
    if lm is None:
        return None
    # The lower-cased match gives the SHAPE; the file name has to come back
    # from the original casing, or "Lab_Report.pdf" is looked for as
    # "lab_report.pdf" -- which is a different file on a case-sensitive
    # filesystem and no file at all on the day he has both.
    raw = _raw_cmd_text(c)
    rm = (_SEND_FILE_RX.match(raw) if raw else None) or lm
    shape_a = bool(lm.group("file_a"))
    said_file = (rm.group("file_a") or rm.group("file_b") or "").strip()
    who = (rm.group("who_a") or rm.group("who_b") or "").strip()
    hint = (rm.group("acct") or "").strip()
    memory = c._svc("memory")
    addr, _ = outbox.resolve_recipient(cfg, memory, who)
    if not addr:
        if not shape_a:
            log.info("send-file: %r names no one I can write to", who)
            return None
        if not outbox.names_a_real_file(cfg, said_file):
            log.info("send-file: %r names neither a file nor a correspondent",
                     t[:60])
            return None
    return said_file, who, hint, addr


def _send_file_offer(c, prep, who: str, hint: str):
    """"Which one, sir?", parked so the answer can be heard.

    Without the slot this branch was a dead end: the question was spoken,
    nothing was armed, the follow-up microphone got the short window and
    every natural answer ("the final one", "lab report final") was then
    called background chat and dropped without a word.
    """
    cfg = c._svc("assistant")

    def _resume(path):
        again = outbox.prepare(cfg, c._svc("memory"), str(path), who,
                               account_hint=hint, chosen=path)
        if again.draft is None:
            return CommandResult(handled=True, reply=again.ask, speak=True,
                                 status=again.status)
        c.stash_send(again.draft)
        return CommandResult(handled=True, speak=True, status=again.status,
                             reply=outbox.read_back(again.draft))

    c.stash_filepick(prep.candidates, _resume)
    return CommandResult(handled=True, reply=prep.ask, speak=True,
                         status=prep.status)


def _h_send_file(c, t, m):
    """Arm a file send and read it back. Nothing is sent from here."""
    pieces = _send_file_pieces(c, t, m)
    if pieces is None:
        log.info("send-file: %r is not a send-a-file request", t)
        return None
    said_file, who, hint, _addr = pieces
    # ONE read-back at a time. Every ordinary path spends _pending_send in
    # _try_send_confirm before a second send can arm, so this is reached
    # only inside a single turn -- the compound "email A to Heather and
    # email B to Heather", where arming twice speaks two questions and
    # leaves only the second one answerable.
    live = getattr(c, "_pending_send", None)
    if live is not None and not live.stale():
        return CommandResult(handled=True, speak=True, status="One at a time",
                             reply="There's one waiting on your yes already, "
                                   "sir; that one first.")
    prep = outbox.prepare(c._svc("assistant"), c._svc("memory"), said_file,
                          who, account_hint=hint)
    if prep.draft is None:
        if prep.candidates:
            return _send_file_offer(c, prep, who, hint)
        return CommandResult(handled=True, reply=prep.ask, speak=True,
                             status=prep.status)
    c.stash_send(prep.draft)
    return CommandResult(handled=True, reply=outbox.read_back(prep.draft),
                         speak=True, status=prep.status)


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
    _touch_list(notes, "todo")        # "...and cross the second one off the list"
    return CommandResult(handled=True, reply="Added to your list, sir.",
                         speak=True, status=f"To-do: {text[:40]}",
                         undo=_undo_notes(notes, "todo", todo_id,
                                          "Off your list again, sir."))


def _h_todo_list(c, t, m):
    notes = c._svc("notes")
    if notes is None:
        return None
    _touch_list(notes, "todo")
    return CommandResult(handled=True, reply=notes.list_text("todo"),
                         speak=True, status="To-dos")


def _h_todo_done(c, t, m):
    notes = c._svc("notes")
    if notes is None or not hasattr(notes, "complete"):
        return None
    which = (m.group("w1") or m.group("w2") or m.group("w3") or "last").strip()
    _touch_list(notes, "todo")
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
    desc = _describe(tk, due, now, when)
    what = task if re.match(r"^(?:that|about)\b", task, re.I) else f"to {task}"

    def _run():
        item = tk.add_reminder(due, task)
        # action=item is the aside's anchor: "remind me to hand in the lab at
        # nine" resolves to a real datetime here, and jarvis/aside.py refuses
        # to volunteer anything without one rather than re-parse the words.
        return CommandResult(handled=True,
                             reply=f"Very good, sir; I'll remind you {what} {desc}.",
                             speak=True, status=f"Reminder {desc}: {task[:30]}",
                             action=item,
                             undo=_undo_timekeeper(tk, item, "reminder",
                                                   "Reminder cancelled, sir."))
    # Both halves can be misheard here -- the hour and the errand itself --
    # so a shaky transcript reads the whole parse back (_confirm_or_run).
    return _confirm_or_run(c, _run, f"A reminder {what} {desc}, sir?")


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

# Room tone (jarvis/roomtone.py). The bed ships OFF and its own proposal
# requires an explicit SPOKEN opt-in, so this is the switch -- and it has to
# be reversible in one utterance, because the failure mode of ambience is
# that it is quietly costing you wake words while you wonder why.
_ROOM_TONE_NAME = r"(?:room ?tone|ambience|ambient (?:sound|noise)|the bed)"
_ROOM_TONE_RX = re.compile(
    r"^" + _JV + r"(?:"
    r"(?P<status>(?:is|is the|what's|whats)\s+(?:the\s+)?" + _ROOM_TONE_NAME +
    r"\s*(?:on|off|running|playing)?)"
    r"|(?:turn|switch|put)\s+(?P<state1>on|off)\s+(?:the\s+|my\s+)?" + _ROOM_TONE_NAME +
    r"|(?P<verb2>turn|switch|start|stop|kill|end|play|enable|disable)?\s*"
    r"(?:the\s+|my\s+)?" + _ROOM_TONE_NAME + r"(?:\s+(?P<state2>on|off))?"
    r")[?.!\s]*$", re.I)
_ROOM_TONE_ON_VERBS = ("start", "play", "enable")
_ROOM_TONE_OFF_VERBS = ("stop", "kill", "end", "disable")
ROOM_TONE_ON_LINE = ("Room tone on, sir. Tell me if it costs me a wake word.")
ROOM_TONE_OFF_LINE = "Room tone off, sir."
ROOM_TONE_STATUS_ON = "The room tone is on, sir."
ROOM_TONE_STATUS_OFF = "The room tone is off, sir."

# The window's LOOK (jarvis/ui/theme.py, 2026-09-01): "holo" is the
# blue-holographic overhaul, "classic" is the 08-31 console kept token for
# token as the fallback he asked for. The switch is a config write that the
# window reads ONCE at create(), so the line says "after a restart" out
# loud: a look that half-applies mid-session is the only way this can look
# broken. Either a switching verb ("switch to classic") or a surface noun
# ("classic visuals") is required -- a bare "classic" or "hologram" is not
# an order. "ui look" is Tier 1 and so BYPASSES the intent gate; the review
# of 09-01 found "give me classic" / "i want the classic" / "use classic"
# matching the noun-less path, and an overheard sentence in an open
# listening window would have rewritten the config. So only the four verbs
# that cannot mean anything else with a look name after "to" may drop the
# noun; every softer opener needs it.
_UI_LOOK_WORD = r"(?P<look%s>classic|holo(?:graphic|gram)?)"   # one name per branch
_UI_LOOK_NOUN = r"(?:visuals?|look|theme|ui|skin|style|console|interface|display|graphics)"
_UI_LOOK_TO = r"(?:\s+(?:it|me|us))?(?:\s+back)?(?:\s+over)?\s+to"
_UI_LOOK_RX = re.compile(
    r"^" + _JV + r"(?:"
    # "switch to (the) classic (look)", "flip back to holo", "set it to classic mode"
    r"(?:switch|change|set|flip)" + _UI_LOOK_TO +
    r"\s+(?:the\s+)?" + _UI_LOOK_WORD % 1 + r"(?:\s+mode)?(?:\s+" + _UI_LOOK_NOUN + r")?"
    # "go back to the holographic visuals", "use the holo look", "give me the
    # classic visuals" -- the noun is mandatory here
    r"|(?:(?:go|put|take)" + _UI_LOOK_TO +
    r"|use|give me|show me|i want|i'd like|i would like|let's have|lets have)"
    r"\s+(?:the\s+)?" + _UI_LOOK_WORD % 4 + r"(?:\s+mode)?\s+" + _UI_LOOK_NOUN +
    # "switch the visuals to classic", "change the look to holographic"
    r"|(?:switch|change|set|flip|put|turn)\s+(?:the\s+|your\s+|my\s+)?" + _UI_LOOK_NOUN +
    r"(?:\s+back)?\s+(?:to|over to|into)\s+(?:the\s+)?" + _UI_LOOK_WORD % 2 + r"(?:\s+(?:mode|one))?"
    # "classic mode visuals", "holographic visuals", "classic look please"
    r"|(?:the\s+)?" + _UI_LOOK_WORD % 3 + r"(?:\s+mode)?\s+" + _UI_LOOK_NOUN +
    # Whisper writes the vocative with a comma ("..., please"), so the tail
    # takes [,\s] not just whitespace.
    r")(?:[,\s]+(?:please|now|sir))*[.!\s]*$", re.I)
UI_LOOK_OPTION = "console.look"          # == jarvis.ui.theme.OPTION_KEY (tested)
UI_LOOK_ENV = "JARVIS_LOOK"              # == jarvis.ui.theme.ENV_KEY (tested)
UI_LOOK_LINES = {
    "classic": "Classic visuals, sir \u2014 it applies after a restart.",
    "holo": "Holographic visuals, sir \u2014 after a restart.",
}
# The env var outranks the saved option in theme.resolve_look (a one-off
# run, the judge harness), so a launcher that exports it would make the
# restart promise above a lie. Say so instead of promising.
UI_LOOK_PINNED_LINE = ("Saved, sir \u2014 but the environment pins the look to "
                       "{pinned} until the {env} variable is removed.")
_UI_LOOK_SPOKEN = {"classic": "classic", "holo": "holographic"}
UI_LOOK_NO_CONFIG_LINE = ("I can't reach the settings to change the visuals, "
                          "sir \u2014 the assistant config isn't wired.")
UI_LOOK_SAVE_FAILED_LINE = "I couldn't save the visuals setting, sir."


def _h_ui_look(c, t, m):
    """"Switch to classic visuals" / "holographic visuals".

    Writes console.look for the NEXT start (main_window.create reads it
    through theme.resolve_look) and says so. Not gated by needs=: with no
    assistant service the honest answer is spoken, not a silent fall-through
    to the router, which would hand "switch to classic visuals" to a model
    that cannot do it.
    """
    word = next((v for k, v in m.groupdict().items()
                 if k.startswith("look") and v), "").lower()
    name = "classic" if word.startswith("classic") else "holo"
    cfg = c._svc("assistant")
    if cfg is None or not hasattr(cfg, "set"):
        return CommandResult(handled=True, speak=True, reply=UI_LOOK_NO_CONFIG_LINE,
                             status="Visuals: no config")
    try:
        cfg.set(UI_LOOK_OPTION, name)
    except Exception:
        log.exception("%s could not be saved", UI_LOOK_OPTION)
        return CommandResult(handled=True, speak=True, reply=UI_LOOK_SAVE_FAILED_LINE,
                             status="Visuals: save failed")
    # Our own environment is the one the restart inherits (the autostart
    # entry re-runs the same launcher), so a valid JARVIS_LOOK here that
    # disagrees with the write means resolve_look will ignore the write.
    pinned = (os.environ.get(UI_LOOK_ENV) or "").strip().lower()
    if pinned in UI_LOOK_LINES and pinned != name:
        log.info("%s=%s overrides the saved %s=%s at the next start",
                 UI_LOOK_ENV, pinned, UI_LOOK_OPTION, name)
        return CommandResult(handled=True, speak=True,
                             reply=UI_LOOK_PINNED_LINE.format(
                                 pinned=_UI_LOOK_SPOKEN[pinned], env=UI_LOOK_ENV),
                             status=f"Visuals: {name} saved, env pins {pinned}")
    return CommandResult(handled=True, speak=True, reply=UI_LOOK_LINES[name],
                         status=f"Visuals: {name} (restart)")


# The conversation pane (jarvis/ui/views.py TranscriptView), 2026-09-02.
# 23:26:01 he said "Clear the transcript" and the intent gate answered
# "Ignored (background chat, conf=0.80)": three words, no rung, dropped in
# silence. This is that rung.
#
# THE SCREEN ONLY, and the line says so. Wiping the pane is cosmetic and
# costs nothing; wiping the conversation the model sees (jarvis/memory.py,
# the context engine) changes what Jarvis knows mid-sentence. He asked for
# "the transcript", which is the thing in front of him, so that is what he
# gets -- and he is told the memory is intact rather than left to wonder
# whether Jarvis has just forgotten the last ten minutes.
#
# The collision this grammar exists to survive: "clear" is already his verb
# for his LISTS ("clear the shopping list", 09-01 12:12 and 20:58). Two
# defences, because this repo has shipped this bug before -- a widened undo
# grammar quietly ate "cancel that one". First, the noun set below is a
# closed list of words that can only mean the pane, and "list" is not in it
# nor is it an accepted trailing word, so "clear my shopping list" cannot
# match at all. Second, this rung sits BELOW the named-list family in the
# registry, so even a future widening hands "... list" to the lists.
#
# Nouns considered and REFUSED. "memory" outright: that is the context
# wipe, which this rung must never do. "history" and "log" on their own,
# because "clear the history" is the clipboard's and "clear the log" is a
# file; they are accepted only AFTER a pane noun, where "clear the chat
# history" can mean nothing else.
_TRANSCRIPT_NOUN = r"(?:transcript|screen|display|console|chat|conversation)"
# THE LEFT EDGE (added 2026-09-03, review).  _JV alone is the house
# convention -- 41 rungs open with it and exactly one, _ADJUST_SCHED_RX,
# also takes a leading "please" -- but on THIS rung that convention lands
# his ordinary phrasing in the very silent drop the rung was built to end:
# measured through the shipping ladder, "please clear the transcript",
# "can you clear the transcript", "go ahead and clear the transcript" and
# even the explicitly-addressed "jarvis please clear the transcript" all
# reached NO rung and were handed to a model that has no tool to clear
# anything.  Repeated (`*`) because "please can you" and "jarvis, go ahead
# and" are each one breath; every alternative eats a whole word plus its
# space, so the group cannot spin on an empty match.  It sits on BOTH
# sides of _JV -- he says "jarvis, please clear..." and "please, jarvis,
# clear..." interchangeably.
#
# Widening the LEFT edge cannot reach his lists.  The language is still
# end-anchored on the closed pane-noun set below, and "list" is in neither
# that set nor the trailing words, so "please clear the shopping list"
# still cannot match at all -- the negative table in
# tests/test_clear_transcript.py asserts every one of his logged list
# utterances against the shipping dispatch order, not against this regex
# alone.
_LEAD_COURTESY = (r"(?:(?:please|can you|could you|would you|will you|just|"
                  r"go ahead and|let's|lets),?\s+)*")
_TRANSCRIPT_CLEAR_RX = re.compile(
    r"^" + _LEAD_COURTESY + _JV + _LEAD_COURTESY +
    r"(?:clear|wipe|erase|empty|blank|clean|scrub|reset)\s+(?:out\s+)?"
    # "that" was missing while "this" and "your" were in, so "clear that
    # transcript" was dropped in silence -- the same class of miss as the
    # 23:26:01 log line this rung answers.  The set is now the union of
    # this one and _LIST_CLEAR_RX's (my|the|our), so the two no longer
    # differ for no stated reason; a determiner cannot cause a collision,
    # the noun after it can, and that set is closed.
    r"(?:the|my|this|that|your|our)?\s*(?:(?:whole|entire|full)\s+)?"
    + _TRANSCRIPT_NOUN +
    r"(?:\s+(?:pane|panel|window|view|log|history|area))?"
    r"(?:\s+(?:clean|out|off))?"
    r"(?:\s+(?:right\s+)?now)?"
    r"(?:[, ]+(?:please|jarvis|sir|for me|would you|will you|thanks))*"
    r"[?.!]*$", re.I)
# Both facts in one breath: the memory is untouched, and the cards do not
# come back. No read-back and no undo= go with it -- see _h_transcript_clear.
TRANSCRIPT_CLEAR_LINE = ("Screen's clear, sir. Nothing forgotten — "
                         "and nothing to bring back.")


def _standing_questions(c) -> int:
    """How many approval questions are still waiting on him.

    TranscriptView.clear_all deliberately KEEPS an unanswered approval
    card -- it carries the only hand-answerable ALLOW / DENY for a Claude
    run that is blocked on it -- so in that one case the pane is NOT clear
    when the wipe lands, and "Screen's clear, sir" would be the wrong
    thing to say.  The wipe itself is fire-and-forget (the commander runs
    on worker threads and must never touch a Tk surface), so the fact is
    read from the approvals SERVICE, which is the same fact the pane is
    keying on: ApprovalService.pending() holds exactly the requests whose
    cards are unanswered, and answer()/_resolve() is the only thing that
    empties it.

    Reading it here rather than reporting it back from the window is also
    the only version that cannot lose a race: a Status published from
    inside _ev_transcript_clear lands BEFORE the CommandResult's own
    status (the bus is a queue, and _emit_result publishes the reply and
    the status after the wipe was queued), so the window's correction
    would be overwritten by the flat "Transcript cleared".

    Never raises and never blocks the wipe: with no approvals service, or
    a service that throws, this answers 0 and the plain line is spoken.
    """
    ap = c._svc("approvals") if hasattr(c, "_svc") else None
    if ap is None:
        return 0
    try:
        return len(ap.pending() or ())
    except Exception:                           # noqa: BLE001 - cosmetic
        log.exception("approvals.pending failed; reporting a plain wipe")
        return 0


def transcript_clear_line(held: int) -> str:
    """The spoken confirmation, told plainly: what he actually got.

    The brief's rule for this rung was to say what happened rather than
    leave it to be guessed at, which is why the plain line names both the
    memory and the missing undo.  The held case is the same rule: he hears
    that one card stayed and why, instead of hearing "Screen's clear" and
    seeing a card, with an 1800 ms toast as the only correction.
    """
    if held <= 0:
        return TRANSCRIPT_CLEAR_LINE
    what = "one question" if held == 1 else f"{held} questions"
    those = "that card" if held == 1 else "those cards"
    return (f"Screen's clear bar {what} still waiting on you, sir — "
            f"I've left {those} up. Nothing forgotten, and nothing to "
            f"bring back.")


def transcript_clear_status(held: int) -> str:
    """The status strip's line -- it outlives the toast, which is the
    point: the toast under the held card is gone in 1.8 s."""
    if held <= 0:
        return "Transcript cleared"
    s = "" if held == 1 else "s"
    return f"Transcript cleared — {held} question{s} left standing"


def _h_transcript_clear(c, t, m):
    """Empty the console's conversation pane.

    Fire-and-forget on the bus: the commander runs on worker threads and
    must never touch a Tk surface, so the window subscribes and does the
    work (main_window._ev_transcript_clear). The event is queued BEFORE
    _emit_result publishes this reply, and the bus is FIFO, so the wipe
    lands first and the confirmation is the one card left on the glass.

    No needs= and no service check. The transcript is the main window, not
    an optional second surface like the Board, and with no window running
    there is no one listening to mislead.

    Deliberately NOT destructive-confirmed and deliberately NOT undoable.
    _h_list_clear stashes a read-back because it destroys DATA he would have
    to dictate again; nothing is lost here, so a "are you sure, sir?" on a
    screen wipe would only stand between him and an empty pane. undo=None
    leaves "scratch that" its old meaning rather than offering a restore
    this rung cannot honour -- which is exactly why the line says so.

    The one case where the pane does NOT come out empty -- a question still
    waiting on him, which clear_all keeps on purpose -- is counted BEFORE
    the wipe is queued and said out loud, so the reply and the status strip
    both match the glass. See _standing_questions.
    """
    held = _standing_questions(c)
    bus.publish(ClearTranscript())
    return CommandResult(handled=True, reply=transcript_clear_line(held),
                         speak=True, status=transcript_clear_status(held))


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


def _h_room_tone(c, t, m):
    """"room tone on" / "stop the room tone" / "is the room tone on".

    Writes ambience.room_tone through _persist_preference, so the switch he
    speaks survives a restart and is recorded as a preference. The player
    reads the key every tick, so there is nothing to restart.
    """
    cfg = c._svc("assistant")
    if cfg is None:
        return None
    if m.group("status"):
        on = bool(cfg.get("ambience.room_tone", False))
        return CommandResult(handled=True, speak=True,
                             reply=ROOM_TONE_STATUS_ON if on else ROOM_TONE_STATUS_OFF,
                             status=f"Room tone {'on' if on else 'off'}")
    state = (m.group("state1") or m.group("state2") or "").lower()
    verb = (m.group("verb2") or "").lower()
    if not state:
        if verb in _ROOM_TONE_ON_VERBS:
            state = "on"
        elif verb in _ROOM_TONE_OFF_VERBS:
            state = "off"
    if not state:
        # A bare "room tone" is a question, not an order: acting on it would
        # start audio he never asked for.
        on = bool(cfg.get("ambience.room_tone", False))
        return CommandResult(handled=True, speak=True,
                             reply=ROOM_TONE_STATUS_ON if on else ROOM_TONE_STATUS_OFF,
                             status=f"Room tone {'on' if on else 'off'}")
    want = state == "on"
    if not _persist_preference(c, "ambience.room_tone", want):
        return None
    tone = c._svc("roomtone")
    if tone is not None and not want:
        try:
            tone.settle()        # down now, rather than at the next tick
        except Exception:
            log.exception("room tone could not be stopped")
    return CommandResult(handled=True, speak=True,
                         reply=ROOM_TONE_ON_LINE if want else ROOM_TONE_OFF_LINE,
                         status=f"Room tone {'on' if want else 'off'}")


# ------------------------------------------------------------------
# Offline mode -- the privacy switch, spoken (jarvis/sensing.py)
# ------------------------------------------------------------------
# His ruling of 2026-09-02: "It also needs to have a Jarvis offline mode
# and it will shutdown/disable sensors and cameras", worked "by a voice
# command that says offline mode or deactivate presence or something of
# that nature", and the curfew window changeable "with voice command or UI
# settings buttons". So this is a FAMILY, not one blessed phrase.
#
# WHAT IS TIER 1 HERE, AND WHY THE REST IS NOT.
#   Tier 1: the switch (on / off), the status question, a BOUNDED offline
#   whose end is a duration or a clock time, and the daily window given as
#   two clock times or one moved edge. All four are closed vocabularies
#   over an unambiguous time token, they are what he actually says, and --
#   the deciding reason -- they have to work when the GPU is lent out and
#   the local model is unloaded. A privacy switch that needs a 26B model
#   resident is not a privacy switch.
#   NOT Tier 1: any end that refers to an event rather than a clock ("keep
#   it off until I'm back from class", "no cameras while my brother's
#   here"). A regex cannot ask a clarifying question, and a mis-parsed
#   time silently rewrites a privacy schedule; those fall to the router,
#   which can ask.
#   And when a Tier-1 SHAPE matches but the TIME does not parse, the
#   sensors go off NOW, open-endedly, and the line says the "until" was
#   not caught. Failing toward more privacy and saying so out loud beats
#   both guessing and refusing.
#
# The microphone is deliberately absent from every one of these handlers
# (tests/test_offline_mode.py asserts it by reading their source): the
# switch is spoken off again, so gating the mic would make it one-way.
_SENSE_NOUN = (r"(?:the\s+|my\s+|your\s+)?"
               r"(?:cameras?|webcams?|web\s?cams?|sensors?|sensing|radar|"
               r"presence(?:\s+(?:sensor|sensing|detection))?|"
               r"lens(?:es)?|eyes)")
# "Turn off the camera and the radar" is ONE order, not a sentence this
# family may drop on the floor because it names both sensors.
_SENSE_NOUNS = _SENSE_NOUN + r"(?:\s+and\s+" + _SENSE_NOUN + r")?"
_SENSE_MODE = r"(?:offline|privacy)"
# He puts the politeness in FRONT at least as often as behind ("please stop
# watching", "can you stop watching"), and the whole reason this family is
# Tier 1 is that a privacy order must never reach a model that cannot
# switch a sensor. A leading modal is the cheapest way to lose one.
_SENSE_ASK = r"(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+)?"
# Whisper writes the vocative with a comma, so the tail takes [,\s].
_SENSE_TAIL = r"(?:[,\s]+(?:please|now|sir))*[?.!\s]*$"
# "No sensors tonight" is the same order as "no sensors": off until he says
# otherwise. Deliberately NOT read as a window -- inventing an end time is
# the one direction that puts a lens back on by itself.
_SENSE_OFF_TAIL = r"(?:[,\s]+(?:please|now|sir|tonight|today))*[?.!\s]*$"

_SENSING_STATUS_RX = re.compile(
    r"^" + _JV + r"(?:"
    r"(?:are|is)\s+(?:you|we|it)\s+(?:in\s+)?" + _SENSE_MODE + r"(?:\s+mode)?"
    r"|is\s+" + _SENSE_MODE + r"\s+mode(?:\s+(?:on|off|active|running))?"
    r"|(?:are|is)\s+you\s+(?:watching|looking|recording|filming|seeing)"
    r"(?:\s+(?:at\s+)?(?:me|us|the\s+room|the\s+office))?"
    r"|(?:are|is)\s+" + _SENSE_NOUN + r"\s+(?:on|off|running|live|active|up)"
    r"|(?:what(?:'s| is)?\s+)?(?:the\s+)?"
    r"(?:sensing|sensor|camera|privacy)\s+status"
    r"|(?:when|what)\s+(?:is|are|time is)\s+(?:the\s+|my\s+)?"
    r"(?:camera\s+)?curfew"
    r")" + _SENSE_TAIL, re.I)

_SENSING_CURFEW_RX = re.compile(
    r"^" + _JV + r"(?:"
    r"(?P<off>(?:turn\s+off|switch\s+off|cancel|remove|disable|drop|delete|"
    r"no\s+more)\s+(?:the\s+|my\s+)?(?:cameras?\s+)?curfew)"
    # a whole window: "camera curfew from nine to seven"
    r"|(?:(?:set|change|move|make|put)\s+(?:the\s+|my\s+)?(?:cameras?\s+)?"
    r"curfew\s+(?:to\s+|at\s+)?(?:from\s+)?"
    r"|(?:the\s+)?(?:cameras?\s+)?curfew\s+(?:is\s+)?(?:from\s+)?"
    r"|no\s+cameras?\s+from\s+|cameras?\s+off\s+from\s+)"
    r"(?P<a>.+?)\s+(?:to|until|till|through)\s+(?P<b>.+?)"
    # one edge: "start the camera curfew at ten tonight"
    r"|(?:start|begin|move)\s+(?:the\s+|my\s+)?(?:cameras?\s+)?curfew\s+"
    r"(?:at|to)\s+(?P<start>.+?)"
    # the other edge: "extend the camera curfew until noon"
    r"|(?:extend|push|end|lift|stretch)\s+(?:the\s+|my\s+)?(?:cameras?\s+)?"
    r"curfew\s+(?:to|until|till|at)\s+(?P<end>.+?)"
    r")" + _SENSE_TAIL, re.I)

_SENSING_HOLD_RX = re.compile(
    r"^" + _JV + _SENSE_ASK + r"(?:"
    r"(?:keep|leave)\s+" + _SENSE_NOUN + r"\s+(?:off|down)\s+"
    r"(?P<mode1>for|until|till|through)\s+(?P<when1>.+?)"
    r"|no\s+(?:more\s+)?(?:cameras?|sensors?|radar|watching)\s+"
    r"(?P<mode2>for|until|till|through)\s+(?P<when2>.+?)"
    r"|(?:turn|switch|shut)\s+(?:off\s+)?" + _SENSE_NOUN + r"(?:\s+off)?\s+"
    r"(?P<mode3>for|until|till|through)\s+(?P<when3>.+?)"
    r"|(?:go\s+offline|offline(?:\s+mode)?|privacy\s+mode)\s+"
    r"(?P<mode4>for|until|till|through)\s+(?P<when4>.+?)"
    # "stop watching for ten minutes" -- the bare verb already goes off
    # open-endedly, so without this the BOUNDED form was the one that fell
    # through to the router.
    r"|stop\s+(?:watching|looking|sensing|spying|staring)"
    r"(?:\s+(?:at\s+)?(?:me|us|the\s+room|the\s+office))?\s+"
    r"(?P<mode5>for|until|till|through)\s+(?P<when5>.+?)"
    # ...and the verbless "camera off for an hour".
    r"|(?:cameras?|sensors?|radar|presence|sensing)\s+(?:off|down)\s+"
    r"(?P<mode6>for|until|till|through)\s+(?P<when6>.+?)"
    r")" + _SENSE_TAIL, re.I)

_SENSING_OFF_RX = re.compile(
    r"^" + _JV + _SENSE_ASK + r"(?:"
    r"go(?:ing)?\s+offline"
    r"|go\s+dark"
    r"|" + _SENSE_MODE + r"\s+mode(?:\s+on)?"
    r"|(?:turn|switch|flip)\s+on\s+" + _SENSE_MODE + r"\s+mode"
    r"|(?:enable|activate|engage|start)\s+" + _SENSE_MODE + r"\s+mode"
    r"|(?:go|switch|drop|flip)\s+(?:in)?to\s+" + _SENSE_MODE + r"\s+mode"
    r"|(?:deactivate|disable|kill|stop|shut\s+down|shut\s+off|turn\s+off|"
    r"switch\s+off|power\s+down|cut)\s+" + _SENSE_NOUNS +
    r"|(?:turn|switch|shut|power)\s+" + _SENSE_NOUNS + r"\s+(?:off|down)"
    r"|stop\s+(?:watching|looking|sensing|spying|staring)"
    r"(?:\s+(?:at\s+)?(?:me|us|the\s+room|the\s+office))?"
    r"|(?:don'?t|do\s+not)\s+watch\s+(?:me|us|the\s+room)"
    r"|(?:close|shut)\s+your\s+eyes"
    r"|look\s+away"
    r"|no\s+(?:more\s+)?(?:cameras?|sensors?|watching|radar)"
    r"|(?:cameras?|sensors?|radar|presence|sensing)\s+(?:off|down)"
    r")" + _SENSE_OFF_TAIL, re.I)

_SENSING_ON_RX = re.compile(
    r"^" + _JV + _SENSE_ASK + r"(?:"
    r"(?:(?:come|go|get)\s+)?back\s+online"
    r"|(?:come|go|get)\s+online"
    r"|online\s+mode"
    r"|(?:turn|switch)\s+off\s+" + _SENSE_MODE + r"\s+mode"
    r"|(?:exit|leave|end|cancel|disable|stop|quit)\s+" + _SENSE_MODE + r"\s+mode"
    r"|" + _SENSE_MODE + r"\s+mode\s+off"
    r"|(?:re-?activate|re-?enable|enable|resume|restore|start|wake)"
    r"(?:\s+up)?\s+" + _SENSE_NOUNS +
    r"|(?:turn|switch|power)\s+on\s+" + _SENSE_NOUNS +
    r"|(?:turn|switch|power)\s+" + _SENSE_NOUNS + r"\s+(?:back\s+)?on"
    r"|start\s+watching(?:\s+(?:again|me|us|the\s+room|the\s+office))?"
    r"|(?:open|use)\s+your\s+eyes"
    r"|(?:cameras?|sensors?|radar|presence|sensing)\s+(?:back\s+)?on"
    r"|you\s+can\s+watch(?:\s+(?:me|us|the\s+room|the\s+office))?\s+again"
    r")" + _SENSE_TAIL, re.I)

SENSING_NO_POLICY_LINE = ("I can't reach the sensing switch, sir — offline "
                          "mode isn't wired in this build.")
SENSING_MIC_LINE = "The microphone stays on."
SENSING_NOTHING_LINE = "Nothing was sensing to stop."
SENSING_UNSAVED_LINE = "I couldn't save that, so it won't hold if I restart."
SENSING_CURFEW_SET_LINE = "Camera curfew is now {start} to {end}, sir."
SENSING_CURFEW_OFF_LINE = "The camera curfew is off, sir."
SENSING_CURFEW_OFF_FAIL_LINE = "I couldn't switch the curfew off, sir."
SENSING_CURFEW_CLAUSE = "The camera stays off until {end} for the curfew."


def _sensing_join(names) -> str:
    """('camera', 'radar') -> 'the camera and the radar'."""
    parts = ["the %s" % n for n in names]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _sensing_end_words(hm) -> str:
    from jarvis.quiet import fmt_clock
    return fmt_clock(hm[0], hm[1])


def _sensing_off_line(out, when_text: str = "") -> str:
    """What OFFLINE actually did -- never what it intended.

    The clauses that can appear are all things he would otherwise only
    discover by finding a camera light on: a device that did not stop, a
    device that was only stopped as far as this process reaches (the
    radar's live configuration -- no power switch wired, so it keeps
    radiating and Jarvis merely stops asking), a state file that did not
    save (so the next start comes up online), and the honest "there was
    nothing to stop" on a box where nothing is wired yet.
    """
    # The bound comes from the OUTCOME, not from what the caller asked
    # for: disable() drops an 'until' that is not in the future, and a
    # head reading "offline until 2:47" over an open-ended switch would be
    # the one lie the whole family exists to avoid.
    bounded = bool(when_text) and out.state.until is not None
    parts = ["Offline until %s, sir." % when_text if bounded else "Offline, sir."]
    if out.stopped:
        names = _sensing_join(out.stopped)
        parts.append("%s%s %s down." % (names[0].upper(), names[1:],
                                        "are" if len(out.stopped) > 1 else "is"))
    elif not out.failed and not out.partial:
        parts.append(SENSING_NOTHING_LINE)
    if out.partial:
        parts.append("I've stopped reading %s, but %s power isn't switched, "
                     "so %s still sensing the room." % (
                         _sensing_join(out.partial),
                         "their" if len(out.partial) > 1 else "its",
                         "they're" if len(out.partial) > 1 else "it's"))
    if out.failed:
        parts.append("I couldn't stop %s, so %s still be running." % (
            _sensing_join(out.failed),
            "they may" if len(out.failed) > 1 else "it may"))
    if not out.persisted:
        parts.append(SENSING_UNSAVED_LINE)
    parts.append(SENSING_MIC_LINE)
    return " ".join(parts)


def _sensing_on_line(out) -> str:
    st = out.state
    parts = ["Back online, sir."]
    if out.failed:
        parts.append("I couldn't bring %s back." % _sensing_join(out.failed))
    if st.reason == "curfew" and st.curfew:
        parts.append(SENSING_CURFEW_CLAUSE.format(
            end=_sensing_end_words(st.curfew[1])))
    elif st.camera and st.radar:
        parts.append("Camera and radar are live.")
    if not out.persisted:
        parts.append(SENSING_UNSAVED_LINE)
    return " ".join(parts)


def _sensing_status_line(state) -> str:
    """Reads the state back in his own terms; never changes it."""
    from jarvis import sensing as sensing_mod
    if state.reason == sensing_mod.REASON_FAILSAFE:
        return ("I'm offline, sir: I couldn't read my last sensing state, so "
                "I've stayed off. Say “come back online” when you want "
                "me watching.")
    if state.reason == sensing_mod.REASON_TIMED and state.until:
        end = datetime.fromtimestamp(state.until)
        return ("I'm offline, sir — camera and radar are off until %s."
                % _sensing_end_words((end.hour, end.minute)))
    if state.offline:
        return ("I'm offline, sir — camera and radar are off, and they "
                "stay off until you tell me otherwise.")
    if state.reason == sensing_mod.REASON_CURFEW and state.curfew:
        return ("The camera's off until %s for the curfew, sir; the radar's on."
                % _sensing_end_words(state.curfew[1]))
    if state.curfew:
        return ("Camera and radar are on, sir. The camera goes off at %s."
                % _sensing_end_words(state.curfew[0]))
    return "Camera and radar are on, sir. There's no curfew set."


def _sensing_seconds(mode: str, when: str, now: datetime):
    """'for two hours' / 'until noon' -> seconds from now; None = no idea.

    Deliberately narrower than the do-not-disturb parser: no timekeeper
    fallback, so the ONLY things that resolve here are a duration and a
    clock time. Everything else is handed back as "I didn't catch when",
    which the caller turns into an open-ended offline rather than a guess.
    """
    when = (when or "").strip()
    if not when:
        return None
    if mode in ("until", "till", "through"):
        from jarvis.quiet import parse_clock
        ends = []
        for default in ("am", "pm"):
            hm = parse_clock(when, default=default)
            if hm is None:
                continue
            end = now.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
            if end <= now:
                end += timedelta(days=1)
            ends.append(end)
        return (min(ends) - now).total_seconds() if ends else None
    try:
        from jarvis.tools.timekeeper import parse_duration
        return parse_duration(when)
    except Exception:  # noqa: BLE001 - an unparseable duration is "no idea"
        log.debug("sensing: %r is not a duration", when, exc_info=True)
        return None


def _sensing_group(m, prefix: str) -> str:
    for key, value in m.groupdict().items():
        if key.startswith(prefix) and value:
            return value
    return ""


def _sensing_hold_mode(m) -> str:
    return _sensing_group(m, "mode").lower()


def _sensing_hold_when(m) -> str:
    """The 'when' with the filler his speech carries stripped: "for the
    next two hours" is a duration once "the next" is gone."""
    when = _sensing_group(m, "when").strip()
    when = re.sub(r"^(?:the\s+)?next\s+", "", when, flags=re.I)
    return re.sub(r"^the\s+", "", when, flags=re.I).strip()


def _sensing_publish(state) -> None:
    bus.publish(SensingChanged(camera=state.camera, radar=state.radar,
                               offline=state.offline, reason=state.reason,
                               until=state.until))


def _sensing_result(reply: str, status: str) -> "CommandResult":
    return CommandResult(handled=True, speak=True, reply=reply, status=status)


def _h_sensing_off(c, t, m, until=None, when_text=""):
    """"Offline mode" / "deactivate presence" / "stop watching".

    Not gated by needs=: with no policy the honest answer is SPOKEN. A
    privacy order that falls through to a language model is a privacy
    order that did nothing, and he would have no way to tell.
    """
    pol = c._svc("sensing")
    if pol is None:
        return _sensing_result(SENSING_NO_POLICY_LINE, "Offline mode: not wired")
    out = pol.disable(until=until, source="voice")
    _sensing_publish(out.state)
    bus.publish(Status(text="Sensing offline", kind="warn"))
    return _sensing_result(_sensing_off_line(out, when_text), "Sensing offline")


def _h_sensing_on(c, t, m):
    """"Come back online" / "reactivate presence" / "start watching again"."""
    pol = c._svc("sensing")
    if pol is None:
        return _sensing_result(SENSING_NO_POLICY_LINE, "Offline mode: not wired")
    out = pol.enable(source="voice")
    _sensing_publish(out.state)
    bus.publish(Status(text="Sensing on", kind="info"))
    return _sensing_result(_sensing_on_line(out), "Sensing on")


def _h_sensing_status(c, t, m):
    """"Are you watching?" -- a question, and it changes nothing."""
    pol = c._svc("sensing")
    if pol is None:
        return _sensing_result(SENSING_NO_POLICY_LINE, "Offline mode: not wired")
    state = pol.state()
    return _sensing_result(_sensing_status_line(state),
                           "Sensing off" if state.offline else "Sensing on")


def _h_sensing_hold(c, t, m):
    """"No cameras for the next two hours" / "keep the camera off until noon".

    An end he gave that this cannot parse does NOT become a guess and does
    not become a refusal: everything goes off open-endedly and the line
    says the "until" was missed, which is the only failure direction that
    cannot leave a lens open by accident.
    """
    pol = c._svc("sensing")
    if pol is None:
        return _sensing_result(SENSING_NO_POLICY_LINE, "Offline mode: not wired")
    now = datetime.now()
    secs = _sensing_seconds(_sensing_hold_mode(m), _sensing_hold_when(m), now)
    if secs is None or secs <= 0:
        out = pol.disable(source="voice")
        _sensing_publish(out.state)
        line = _sensing_off_line(out)
        return _sensing_result(
            line.replace("Offline, sir.",
                         "Offline, sir — I didn't catch until when, so it "
                         "stays off until you say otherwise.", 1),
            "Sensing offline")
    end = now + timedelta(seconds=secs)
    return _h_sensing_off(c, t, m, until=end.timestamp(),
                          when_text=_sensing_end_words((end.hour, end.minute)))


def _h_sensing_curfew(c, t, m):
    """The nightly camera window, by voice: "camera curfew from nine to
    seven", "start the camera curfew at ten tonight", "extend it to noon".

    Overnight by convention, exactly like quiet hours: a bare start hour
    is pm and a bare end hour is am, so "nine to seven" is 21:00-07:00 and
    not the working day.
    """
    from jarvis.quiet import parse_clock
    from jarvis.tools.timekeeper import CANT_PARSE_LINE
    from jarvis import sensing as sensing_mod
    pol = c._svc("sensing")
    if pol is None:
        return _sensing_result(SENSING_NO_POLICY_LINE, "Offline mode: not wired")
    if m.group("off"):
        # Checked, exactly like the set-window branch twenty lines below: a
        # config that refused the write must not be spoken back as done.
        if not pol.set_curfew(None, None):
            return _sensing_result(SENSING_CURFEW_OFF_FAIL_LINE,
                                   "Camera curfew: save failed")
        _sensing_publish(pol.state())
        return _sensing_result(SENSING_CURFEW_OFF_LINE, "Camera curfew off")
    current = pol.curfew() or (sensing_mod.DEFAULT_CURFEW_START,
                               sensing_mod.DEFAULT_CURFEW_END)
    if m.group("a"):
        start = parse_clock(m.group("a"), default="pm")
        end = parse_clock(m.group("b"), default="am")
    elif m.group("start"):
        start, end = parse_clock(m.group("start"), default="pm"), current[1]
    else:
        start, end = current[0], parse_clock(m.group("end"), default="am")
    if start is None or end is None or start == end:
        return _sensing_result(CANT_PARSE_LINE, "Camera curfew: when?")
    if not pol.set_curfew(start, end):
        return _sensing_result("I couldn't save the curfew window, sir.",
                               "Camera curfew: save failed")
    _sensing_publish(pol.state())
    words = (_sensing_end_words(start), _sensing_end_words(end))
    return _sensing_result(
        SENSING_CURFEW_SET_LINE.format(start=words[0], end=words[1]),
        "Camera curfew %s–%s" % words)


# ---- Tier 1 face enrolment: the way in, and what the gallery knows -------
# THE ENTRY POINT HE ASKED FOR ("add the enrollment option into Jarvis"),
# and it is deliberately an entry point rather than a capture surface. The
# guided run needs the camera device the running Jarvis owns, a key press
# between stations, and produces thirty lines of numbers he PASTES -- three
# things a spoken assistant is the wrong shell for. So "enrol my face" hands
# over the exact command -- with his label in it, and with --pose and
# --append too when he named what he will be doing ("enrol my face looking
# at my phone") -- and puts it on the clipboard, while "who do you recognise"
# is answered in full, here, because none of that needs a lens. See
# jarvis/enrolentry.py.
#
# DELETING IS HANDED OVER TOO. "Forget Heather's face" arrives as a
# speech-recognition result, and a misheard word may not destroy biometric
# data; the typed confirmation stays in the terminal.
#
# WHY WHO IS TWO TOKENS AND NOT A FREE SPAN. The first cut of this grammar
# let 31 arbitrary characters -- SPACES INCLUDED -- sit between the verb and
# the word "face", which turns every ordinary "<verb> <something> face"
# utterance into a face command: "add a reminder to wash my face" enrolled
# somebody called "a-reminder-to-wash-my", spoke a forty-word consent
# paragraph and overwrote his clipboard, and because these three are in
# ASSISTANT_TIER1 it did that WITHOUT the background-chat intent gate ever
# getting a say. So WHO is at most two name-shaped tokens, and no token may
# be a function word: with that, "add cream for my face", "remove the hair
# from my face", "delete that photo of my face" and "forget what I said
# about her face" all fail to match at all and fall through to the model,
# which is what they were always meant to do. Swept against 582 real
# utterances (his live log plus every tests/ handle()) -- nothing he has
# ever said changes hands.
_FACE_STOP = (r"for|to|from|of|about|that|this|on|in|at|by|with|and|but|or|it|"
              r"is|was|what|which|when|while|if|so|there|here|all|any|some")
_FACE_TOK = r"(?!(?:%s)\b)[a-z][a-z0-9'\u2019_-]{0,19}" % _FACE_STOP
_FACE_WHO = r"(?P<who>%s(?:\s+%s)?)" % (_FACE_TOK, _FACE_TOK)
# NAMING THE POSE IS THE FEATURE, not decoration: "add more ways for me to be
# recognised (looking at my phone, looking away)" is what was asked for, and
# without this the --pose half of the script was unreachable from inside
# Jarvis and --append with it -- so every spoken enrolment was a pool-
# REPLACING run. The clause is bounded and introduced by a fixed word, and
# enrolentry.command_line shlex-quotes it before it reaches the clipboard.
_FACE_POSE = (r"(?:\s+(?P<pose>(?:looking|facing|turned|wearing|holding|with|"
              r"while|when|without)\s+[a-z0-9][a-z0-9 '\u2019_-]{0,60}?))?"
              )
_FACE_TAIL = r"(?:[,\s]+(?:please|now|sir|again))*[?.!\s]*$"
_FACE_ENROL_RX = re.compile(
    r"^(?:please\s+|can\s+you\s+|could\s+you\s+)?"
    r"(?:(?:enroll?|register|add|remember|learn|memori[sz]e)\s+" + _FACE_WHO +
    r"(?:'s|\u2019s)?\s+face"
    r"(?:\s+(?:in|into|to)\s+(?:the\s+)?(?:gallery|camera))?"
    r"|(?:enroll?|register)\s+(?P<mine>me|myself))" + _FACE_POSE + _FACE_TAIL,
    re.I)
_FACE_FORGET_RX = re.compile(
    r"^(?:please\s+|can\s+you\s+|could\s+you\s+)?"
    r"(?:forget|delete|remove|unenrol|unenroll)\s+" + _FACE_WHO +
    r"(?:'s|\u2019s)?\s+face"
    r"(?:\s+from\s+(?:the\s+)?gallery)?" + _FACE_TAIL, re.I)
_FACE_GALLERY_RX = re.compile(
    r"^(?:"
    r"(?:who(?:se)?|what|how\s+many)\s+faces?\s+do\s+you\s+"
    r"(?:know|recogni[sz]e|have)"
    r"|who\s+do\s+you\s+recogni[sz]e"
    r"|who(?:'s|\u2019s| is)\s+in\s+(?:the\s+)?face\s+gallery"
    r"|(?:what(?:'s| is)\s+)?(?:in\s+)?(?:the\s+)?face\s+gallery"
    r"|(?:whose\s+)?faces?\s+(?:are\s+)?enrol(?:l)?ed"
    r"|am\s+i\s+enrol(?:l)?ed"
    r"|face\s+enrol(?:l)?ment\s+status"
    r"|which\s+(?:of\s+my\s+)?(?:pose|take)s?\s+is\s+(?:the\s+)?weak(?:est)?"
    r")" + _FACE_TAIL, re.I)


def _face_owner(c) -> str:
    """His label, from HIS config -- never from the gallery.

    A DELEGATE now, and that is the point. This used to derive the label
    itself, which made it the second of three independent copies of the same
    string (jarvis/cast.py's literal and scripts/face_enrol.owner_label were
    the others). The ruling it anchors is unchanged: the set of ENROLLED
    names may grow without the set of PRIVILEGED names growing by one,
    because "owner" is a config value and a recognised face cannot write the
    config. It is anchored in jarvis/identity.owner_label now, once."""
    return identity_mod.owner_label(c._svc("assistant"))


def _face_gallery(c):
    """The gallery for the CONFIGURED face backend -- never the default one.

    ``camera.face_backend`` decides which model wrote the vectors on disk, and
    a gallery opened without it is a gallery for the wrong model. This built
    ``default_gallery()`` with no backend, so the voice path resolved the
    module default (or ``JARVIS_FACE_BACKEND``) while the camera path resolved
    his config: "who do you recognise" would answer "nobody" over an
    enrolment sitting right there, and "forget Heather's face" would read an
    empty pool. The backend comes from HIS config, the same way
    ``_face_owner`` reads his name -- and a missing config service means "",
    which is the documented "let facemodels decide", not a guess at a model.

    RETURNS None WHEN THE CONFIG CANNOT ANSWER, and the two ways it cannot are
    both real. ``_face_owner`` right above has a try/except and this had none,
    so a config service that RAISES took the whole voice path down with a
    traceback; and ``camera.face_backend`` with a typo in it raises out of
    ``facemodels.backend_for`` BY DESIGN, because a typo that silently kept
    the old models is the class of bug that subsystem is written against.

    None rather than a fallback, and that is the point: falling back to the
    default gallery here would open a store for a model that may not be the
    one that wrote his vectors, and "who do you recognise" would answer
    "nobody" over a live enrolment. The three handlers say the config is
    unreadable instead. A gallery is not a thing to guess at."""
    from jarvis.camera import face_backend_from_config
    from jarvis.facegallery import default_gallery
    try:
        backend = face_backend_from_config(c._svc("assistant"))
    except Exception:  # noqa: BLE001 - a config that cannot say is not fatal
        log.warning("face: could not read camera.face_backend", exc_info=True)
        return None
    try:
        return default_gallery(backend=backend)
    except Exception:  # noqa: BLE001 - an unknown backend name, and it RAISES
        log.warning("face: no gallery for face backend %r", backend,
                    exc_info=True)
        return None


# What the face commands say when the gallery cannot be opened at all. It
# names the setting, because that is the one thing he can act on.
_FACE_CONFIG_REPLY = (
    "I can't tell which face model to open the gallery with, sir -- check "
    "camera.face_backend in the config. I won't guess at a model: the wrong "
    "one reads an enrolment as nobody.")
_FACE_CONFIG_STATUS = "Face gallery: camera.face_backend unreadable"


def _face_config_result():
    return CommandResult(handled=True, speak=True, reply=_FACE_CONFIG_REPLY,
                         status=_FACE_CONFIG_STATUS)


def _h_face_enrol(c, t, m):
    """"Enrol my face" / "add Heather's face to the gallery" / "enrol my face
    looking at my phone"."""
    from jarvis import enrolentry as ee
    owner = _face_owner(c)
    # "enrol me" names him with no WHO group at all; anything else names the
    # person, or names nobody and gets asked.
    spoken = m.group("who") or ("my" if m.groupdict().get("mine") else "")
    who = ee.spoken_label(spoken, owner)
    if not who:
        return CommandResult(
            handled=True, speak=True,
            reply="Whose face, sir? Say \"enrol my face\", or give me the "
                  "name to store it under.",
            status="Enrolment: whose?")
    pose = ee.spoken_pose(m.groupdict().get("pose") or "")
    gallery = _face_gallery(c)
    if gallery is None:
        return _face_config_result()
    out = ee.enrol_answer(gallery, who, owner=owner,
                          poses=(pose,) if pose else ())
    return CommandResult(handled=True, speak=True, reply=out["reply"],
                         status=out["status"])


def _h_face_forget(c, t, m):
    """"Forget Heather's face" -- which deletes nothing. See the block
    comment above: a misheard word may not destroy biometric data."""
    from jarvis import enrolentry as ee
    owner = _face_owner(c)
    who = ee.spoken_label(m.group("who") or "", owner)
    if not who:
        return CommandResult(handled=True, speak=True,
                             reply="Whose face, sir?",
                             status="Face gallery: whose?")
    gallery = _face_gallery(c)
    if gallery is None:
        return _face_config_result()
    out = ee.forget_answer(gallery, who, owner=owner)
    return CommandResult(handled=True, speak=True, reply=out["reply"],
                         status=out["status"])


def _h_face_gallery(c, t, m):
    """"Who do you recognise?" / "which pose is weakest?" -- the half of
    this feature that genuinely belongs in the window, because it reads a
    file and opens nothing."""
    from jarvis import enrolentry as ee
    gallery = _face_gallery(c)
    if gallery is None:
        return _face_config_result()
    out = ee.gallery_answer(gallery, owner=_face_owner(c))
    return CommandResult(handled=True, speak=True, reply=out["reply"],
                         status=out["status"])


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


# ---- Tier 1 room control: light, level and the scenes -------------------
# "Dim it a little", "lights down", "warmer", "lights up", "power down the
# workshop".  There are no bulbs in this room: what moves is the 43-inch
# panel's gamma and GNOME's night light (jarvis/room.py), and the spoken
# lines say so -- the screen dims, the room does not.  Every one of these
# is reversible and is put back at "lights up", at boot and at quit.
#
# "lights up" is deliberately the RESTORE verb rather than one step
# brighter: it is the sentence he will say when he wants the room back, and
# "brighter" is there for the stepped version.
_ROOM_LIGHT_KINDS = (
    ("dim", re.compile(
        r"^" + _JV + r"(?:dim(?: it| that| the (?:screen|display|lights?|room))?"
        r"(?:\s+(?:a little|a bit|a touch|down|slightly))?|"
        r"lights?\s+down|turn (?:the )?(?:lights?|screen|display) down|"
        r"darker|make it darker|(?:it|the screen|the display)'?s? too bright)"
        r"(?:[,]?\s*jarvis)?[.!?\s]*$", re.I)),
    ("brighter", re.compile(
        r"^" + _JV + r"(?:brighten(?: (?:it|the (?:screen|display)))?"
        r"(?:\s+(?:a little|a bit|a touch|up))?|brighter|make it brighter|"
        r"(?:a (?:little|bit) )?brighter|"
        r"(?:it|the screen|the display)'?s? too dark)"
        r"(?:[,]?\s*jarvis)?[.!?\s]*$", re.I)),
    ("up", re.compile(
        r"^" + _JV + r"(?:lights?\s+(?:up|on)|full brightness|"
        r"turn (?:the )?(?:lights?|screen|display) (?:up|back up)|"
        r"(?:bring|put) (?:the )?(?:lights?|screen|display) back(?: up)?)"
        r"(?:[,]?\s*jarvis)?[.!?\s]*$", re.I)),
    ("warm", re.compile(
        r"^" + _JV + r"(?:warm(?:er)?(?: the (?:screen|display))?|"
        r"warm (?:it|the screen|the display)(?: up)?|"
        r"(?:turn on|switch on) (?:the )?night ?light|night ?light on)"
        r"(?:[,]?\s*jarvis)?[.!?\s]*$", re.I)),
    ("cool", re.compile(
        r"^" + _JV + r"(?:cool(?:er)?(?: the (?:screen|display))?|"
        r"cool (?:it|the screen|the display) down|"
        r"(?:turn off|switch off) (?:the )?night ?light|night ?light off|"
        r"back to daylight)(?:[,]?\s*jarvis)?[.!?\s]*$", re.I)),
)
# "lights out" is the SCENE (music, quiet hours and the light together);
# "lights down" above is one step of the light alone.
_SCENE_KINDS = (
    ("down", re.compile(
        r"^" + _JV + r"(?:(?:power|shut|close|lock) (?:down|up) (?:the )?"
        r"(?:workshop|shop|lab|room|studio)|"
        r"wind (?:it |things |the (?:workshop|room) )?down|"
        r"(?:let'?s )?call it a night|lights? out)"
        r"(?:[,]?\s*jarvis)?[.!?\s]*$", re.I)),
    ("up", re.compile(
        r"^" + _JV + r"(?:(?:wake|power|open|start) (?:up )?(?:the )?"
        r"(?:workshop|shop|lab|room|studio)(?: (?:back )?up)?|"
        r"bring (?:the )?(?:workshop|shop|lab|room|studio) back(?: up)?)"
        r"(?:[,]?\s*jarvis)?[.!?\s]*$", re.I)),
)


def room_light_kind(text: str) -> Optional[str]:
    """'dim' / 'brighter' / 'up' / 'warm' / 'cool' for a whole-utterance
    light command, else None."""
    for kind, rx in _ROOM_LIGHT_KINDS:
        if rx.match((text or "").strip()):
            return kind
    return None


def scene_kind(text: str) -> Optional[str]:
    """'down' / 'up' for a whole-utterance scene command, else None."""
    for kind, rx in _SCENE_KINDS:
        if rx.match((text or "").strip()):
            return kind
    return None


def _room_enabled(c) -> bool:
    return bool(_assistant_get(c, "room.enabled", True))


def _h_room_light(c, t, m):
    """The light verbs.  The xrandr/gsettings probes are a handful of short
    subprocess calls (~200 ms), run inline like the timekeeper's database
    work: the answer has to carry the line that says what actually
    happened, and a spoken reply costs far longer than the probe."""
    light = c._svc("room_light")
    if light is None or not _room_enabled(c):
        return None
    if m == "up":
        # "Lights up" is the whole room back, not one notch: restore the
        # scene when one is running, and the light's baseline otherwise.
        scenes = c._svc("scenes")
        if scenes is not None and (scenes.active() or light.changed):
            res = scenes.restore()
            return CommandResult(handled=True, reply=res.line, speak=True,
                                 status="Lights up")
        if light.changed:
            _, line = light.restore()
            return CommandResult(handled=True, reply=line, speak=True,
                                 status="Lights up")
        _, line = light.brighten()
        return CommandResult(handled=True, reply=line, speak=True,
                             status="Lights up")
    verb = {"dim": light.dim, "brighter": light.brighten,
            "warm": light.warmer, "cool": light.cooler}[m]
    _, line = verb()
    return CommandResult(handled=True, reply=line, speak=True,
                         status=f"Room light: {m}")


def _h_scene(c, t, m):
    scenes = c._svc("scenes")
    if scenes is None or not _room_enabled(c):
        return None
    from jarvis.scenes import WIND_DOWN
    res = scenes.restore() if m == "up" else scenes.apply(WIND_DOWN)
    return CommandResult(handled=True, reply=res.line, speak=True,
                         status="Workshop up" if m == "up" else "Workshop down")


def _maybe_wind_down(c) -> str:
    """The good-night hook.  OFF by default (room.wind_down_on_goodnight):
    "good night" already has a handler, and a scene is a change to his
    desktop that he did not ask for by saying good night.  Returns the
    scene's line when it ran, "" otherwise."""
    if not _assistant_get(c, "room.wind_down_on_goodnight", False):
        return ""
    scenes = c._svc("scenes")
    if scenes is None or not _room_enabled(c):
        return ""
    try:
        from jarvis.scenes import WIND_DOWN
        return scenes.apply(WIND_DOWN).line or ""
    except Exception:
        log.exception("wind-down scene failed")
        return ""


def _scene_wake(c) -> bool:
    """"Good morning": whatever the wind-down moved goes back.  True when
    something was actually restored."""
    scenes = c._svc("scenes")
    if scenes is None:
        return False
    light = c._svc("room_light")
    if not scenes.active() and not (light is not None and light.changed):
        return False
    try:
        scenes.restore()
    except Exception:
        log.exception("scene restore at wake failed")
        return False
    return True


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
            # course_chunks, not topic_chunks: "quiz me on biosensors" must
            # not be answered out of another course's PDF (the global
            # embedding leg happily crosses courses). Falls back to
            # topic_chunks when the topic names no course of his.
            chunks = course_chunks(index, c._svc("assistant"), topic, k=k)
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


# ------------------------------------------------------------ teach me
# "Teach me chapter three" is explain and quiz as one verb: the chunks are
# retrieved ONCE, explained, and stashed; a yes to the offer builds the
# quiz from the SAME chunks, so the questions are provably about what was
# just said (_h_quiz retrieves from scratch, which can land on other text).
_TEACH_RX = re.compile(
    r"^" + _JV + r"(?:(?:teach|tutor|coach)\s+me\s+(?:about\s+|on\s+|over\s+)?|"
    r"walk\s+me\s+through\s+)"
    r"(?:the\s+|my\s+|some\s+)?(?P<topic>.+?)(?:\s+please)?[.!?\s]*$", re.I)
# "teach me how to ..." / "teach me why ..." is a question for the router,
# and "teach me a lesson" is not a request at all.
_TEACH_NOT_A_TOPIC_RX = re.compile(
    r"^(?:how|why|what|when|where|who|whether|if)\b|"
    r"^(?:a\s+)?lessons?$|^(?:something|anything|everything|yourself|nothing)$", re.I)
TEACH_ACK_LINE = "Let me pull together what I have on {topic}, sir."
TEACH_OFFER_LINE = "Say quiz me and I'll test you on it, sir."
TEACH_FAIL_LINE = "I couldn't make sense of what I have on {topic}, sir."
TEACH_DECLINED_LINE = "Very good, sir."
# A bare "quiz me" / "go on" takes the teach offer. quiz_kind needs an
# "on <topic>", so these reach no registry command of their own.
_TAKE_QUIZ_RX = re.compile(
    r"^" + _JV + r"(?:(?:quiz|test|drill|grill) me|go on|go ahead|"
    r"(?:let's |lets )?(?:do|try) it)[.!?\s]*$", re.I)


def teach_kind(text: str) -> Optional[str]:
    m = _TEACH_RX.match((text or "").strip())
    if not m:
        return None
    topic = " ".join(m.group("topic").split())
    if len(topic) < 3 or _TEACH_NOT_A_TOPIC_RX.match(topic):
        return None
    return topic


def _teach_no_material(c, index, topic: str) -> None:
    """The same three honest answers the quiz gives when nothing matched:
    no such topic, still indexing, or no documents at all."""
    if index.document_count() > 0:
        _deliver(c, quiz_mod.NO_TOPIC_LINE.format(topic=topic))
    elif index.scan():
        index.start_background()
        _deliver(c, INDEXING_LINE)
    else:
        _deliver(c, quiz_mod.NO_DOCS_LINE)


# ---------------------------------------------------------- study ledger
# focus_session.json is overwritten by the next session, so "three blocks
# done" used to be spoken once and lost. focus.study_days() merges the
# durable ledger with the timekeeper's own never-pruned block rows; these
# two phrases are what read it back.
_STUDY_TOTAL_RX = re.compile(
    r"^" + _JV + r"(?:how (?:much|long) (?:did|have) i (?:studied|study|been studying)|"
    r"how much (?:study|studying|focus|revision)(?: time)? (?:did|have) i (?:do|done|log(?:ged)?)|"
    r"how much did i (?:get )?(?:study|studied|done)|"
    r"(?:what's|what is|show me) my (?:study|focus|revision) (?:time|total|hours))"
    r"(?:\s+(?P<when>today|yesterday|this week|last week|this month))?"
    r"[.!?\s]*$", re.I)
_STREAK_RX = re.compile(
    r"^" + _JV + r"(?:what'?s|what is|how(?:'s| is)|hows)?\s*(?:my|the)?\s*"
    r"(?:study |focus |revision )?streak(?:\s+(?:at|now|going|looking))?[.!?\s]*$", re.I)


def _study_paths(c):
    """(session-state path, timekeeper db path) -- both from the live
    services when the app wired them, so a test session's tmp files are
    used and the real ones are never touched."""
    focus = c._svc("focus")
    state = getattr(focus, "_state_path", None) if focus is not None else None
    tk = c._svc("timekeeper")
    db = getattr(tk, "db_path", None) if tk is not None else None
    return state, db


def _study_table(c) -> dict:
    from jarvis import focus as focus_mod
    state, db = _study_paths(c)
    days = focus_mod.study_days(state_path=state, db_path=db)
    # A session running right now is not in the ledger yet (end() writes
    # it), and "how much did I study today" must still count this morning.
    focus = c._svc("focus")
    try:
        if focus is not None and focus.active and focus.blocks_done > 0:
            per = int(focus.state.get("block_min") or 0)
            started = float(focus.state.get("started") or 0.0)
            if started > 0:
                day = focus_mod._day(started)
                cell = days.setdefault(day, {"blocks": 0, "minutes": 0})
                cell["blocks"] += focus.blocks_done
                cell["minutes"] += focus.blocks_done * max(0, per)
    except Exception:                          # noqa: BLE001 - live session boundary
        log.debug("study ledger: live session unreadable", exc_info=True)
    return days


def _h_study_total(c, t, m):
    from datetime import date, timedelta

    from jarvis import focus as focus_mod
    when = (m.group("when") or "this week").strip().lower() if hasattr(m, "group") else "this week"
    today = date.today()
    since = {"today": today, "yesterday": today - timedelta(days=1),
             "this week": focus_mod.week_start(today),
             "last week": focus_mod.week_start(today) - timedelta(days=7),
             "this month": today.replace(day=1)}.get(when, focus_mod.week_start(today))
    days = _study_table(c)
    if when in ("yesterday", "last week"):
        # a closed window: drop everything after it, or "yesterday" would
        # quietly include today
        stop = (today if when == "yesterday" else focus_mod.week_start(today)).isoformat()
        days = {d: cell for d, cell in days.items() if d < stop}
    return CommandResult(handled=True, reply=focus_mod.summary_line(days, since, when),
                         speak=True, status="Study ledger")


def _h_study_streak(c, t, m):
    from jarvis import focus as focus_mod
    return CommandResult(handled=True, reply=focus_mod.streak_line(_study_table(c)),
                         speak=True, status="Study streak")


def _h_teach(c, t, m):
    topic = m
    index = c._svc("docs")
    if index is None:
        return CommandResult(handled=True, reply=quiz_mod.NO_DOCS_LINE, speak=True,
                             status="No documents")
    brain = c._svc("brain")
    if brain is None or not hasattr(brain, "explain_text"):
        return CommandResult(handled=True, reply=EXPLAIN_NO_MODEL_LINE, speak=True,
                             status="No model")
    k = _int_setting(c, "quiz.chunks", quiz_mod.DEFAULT_CHUNKS)
    c._pending_teach = None                      # a new lesson replaces the old

    def _work():
        try:
            chunks = course_chunks(index, c._svc("assistant"), topic, k=k)
        except EmbedError as exc:
            log.warning("teach: embed failed: %s", exc)
            _deliver(c, quiz_mod.INDEX_DOWN_LINE)
            return
        except Exception:                        # noqa: BLE001 - store boundary
            log.exception("teach: index failed")
            _deliver(c, quiz_mod.INDEX_DOWN_LINE)
            return
        if not chunks:
            _teach_no_material(c, index, topic)
            return
        # study_text, not fact_sheet: the quiz half is built from this
        # exact string, so both halves read the same material.
        body = quiz_mod.study_text(chunks)
        try:
            lead, summary = brain.explain_text(body, name=topic)
        except Exception:
            log.exception("teach: explain_text failed")
            lead, summary = "", ""
        if not lead:
            _deliver(c, TEACH_FAIL_LINE.format(topic=topic))
            return
        if summary and summary != lead:
            bus.publish(JarvisReply(text=f"{topic}\n{summary}", speak=False))
        # Stashed BEFORE the offer is spoken: the reply closes the turn and
        # opens the follow-up window, and the yes can arrive at once.
        c._pending_teach = (topic, body, time.monotonic())
        _deliver(c, f"{lead} {TEACH_OFFER_LINE}")

    c._bg(_work)
    return CommandResult(handled=True, reply=TEACH_ACK_LINE.format(topic=topic),
                         speak=True, ack=True, done=False, status=f"Teaching {topic}")


def _start_review(c, n: int, topic: str = "") -> CommandResult:
    """Open a flashcard session over the cards due now, optionally on one
    deck. Shared by "review my flashcards" and the briefing's exam-week
    offer, so both end up in the same _pending_quiz rung."""
    try:
        store = _quiz_store(c)
    except Exception:
        log.exception("flashcard store unavailable")
        return CommandResult(handled=True, reply=quiz_mod.NO_CARDS_LINE, speak=True,
                             status="No store")
    cards = store.due(limit=n, topic=topic)
    if not cards:
        line = quiz_mod.NO_CARDS_LINE if store.count() == 0 else quiz_mod.NOTHING_DUE_LINE
        return CommandResult(handled=True, reply=line, speak=True, status="No cards due")
    session = quiz_mod.QuizSession(cards, topic=topic or "review")
    c._pending_quiz = session
    return CommandResult(handled=True, reply=f"{_cards_line(len(cards))} {session.ask()}",
                         speak=True, status=f"Flashcards 1/{len(cards)}")


# "Scan the syllabus": the exam a professor never put in Canvas. The chunks
# come from the documents index (topic_chunks on the syllabus topics), ONE
# gemma call proposes {title, course, due} rows, and every row is READ BACK
# before anything is filed -- the destructive-confirm rung, because a model
# reading dates out of a PDF is exactly where a wrong year files a reminder
# for the wrong week. Accepted rows land in jarvis/syllabus.py's store,
# which deadlines.tick and canvas.find_next_exam both merge.
_SYLLABUS_RX = re.compile(
    r"^" + _JV + r"(?:(?:scan|read|check|go through|look through|go over|read through|"
    r"import|ingest|pull the dates out of|get the dates out of)\s+"
    r"(?:the\s+|my\s+|through\s+my\s+)?(?:syllabus|syllabi|syllabuses)"
    r"(?:\s+for\s+(?:the\s+)?(?:dates?|deadlines?|exams?|due dates?))?|"
    r"add (?:the |my )?syllabus (?:dates?|deadlines?|exams?)|"
    r"(?:what's|what is|what are)\s+(?:on|in)\s+my\s+(?:syllabus|syllabi))"
    r"(?:\s+please)?[.!?\s]*$", re.I)


def syllabus_kind(text: str) -> bool:
    return bool(_SYLLABUS_RX.match((text or "").strip()))


def _do_file_syllabus(c, rows) -> CommandResult:
    try:
        n = syllabus_mod.add(rows)
    except Exception:                            # noqa: BLE001 - store boundary
        log.exception("syllabus: filing failed")
        return CommandResult(handled=True, reply="I couldn't file those, sir.",
                             speak=True, status="Syllabus: failed")
    if n <= 0:
        # Every row was already on the books: a second scan of the same
        # syllabus must not read back "filed three" and change nothing.
        return CommandResult(handled=True, reply="Already on the books, sir.",
                             speak=True, status="Syllabus: nothing new")
    line = syllabus_mod.FILED_ONE_LINE if n == 1 else syllabus_mod.FILED_LINE.format(n=n)
    return CommandResult(handled=True, reply=line, speak=True,
                         status=f"Syllabus: {n} filed")


def _h_scan_syllabus(c, t, m):
    index = c._svc("docs")
    if index is None:
        return CommandResult(handled=True, reply=syllabus_mod.NO_SYLLABUS_LINE,
                             speak=True, status="No documents")
    brain = c._svc("brain")
    if brain is None or not hasattr(brain, "read_syllabus"):
        return CommandResult(handled=True, reply=syllabus_mod.NO_DATES_LINE,
                             speak=True, status="No model")

    def _work():
        try:
            chunks = syllabus_mod.gather(index)
        except EmbedError as exc:
            log.warning("syllabus: embed failed: %s", exc)
            _deliver(c, quiz_mod.INDEX_DOWN_LINE)
            return
        except Exception:                        # noqa: BLE001 - store boundary
            log.exception("syllabus: index failed")
            _deliver(c, quiz_mod.INDEX_DOWN_LINE)
            return
        if not chunks:
            # Files present but nothing stored yet is "indexing", not "no
            # syllabus": blaming the folder would send him to check a
            # folder that is fine (the ask_docs precedent).
            if index.document_count() == 0 and index.scan():
                index.start_background()
                _deliver(c, INDEXING_LINE)
            else:
                _deliver(c, syllabus_mod.NO_SYLLABUS_LINE)
            return
        now = datetime.now().astimezone()
        try:
            raw = brain.read_syllabus(syllabus_mod.source_text(chunks),
                                      today=now.strftime("%A %d %B %Y"))
        except Exception:                        # noqa: BLE001 - model boundary
            log.exception("read_syllabus failed")
            raw = []
        rows = syllabus_mod.parse_rows(raw, now)
        if not rows:
            _deliver(c, syllabus_mod.NO_DATES_LINE)
            return
        line = syllabus_mod.read_back_line(rows, now)
        # Stash BEFORE speaking: the follow-up window opens the moment the
        # line is delivered, so a fast "yes" must find the offer waiting.
        c.stash_destructive(lambda: _do_file_syllabus(c, rows), line)
        _deliver(c, line)

    c._bg(_work)
    return CommandResult(handled=True, reply=syllabus_mod.SCANNING_LINE, speak=True,
                         ack=True, done=False, status="Reading the syllabus")


def _h_review(c, t, m):
    return _start_review(c, _int_setting(c, "quiz.questions", quiz_mod.DEFAULT_QUESTIONS))


def _h_quiz_stop(c, t, m):
    session = getattr(c, "_pending_quiz", None)
    if session is None:
        return None                              # no quiz: "stop the test" is the model's
    c._pending_quiz = None
    return CommandResult(handled=True, reply=session.score_line(), speak=True,
                         status="Quiz stopped")


# ------------------------------------------------------------------
# Working sessions (jarvis/dialogue.py). The first tenant: "let us plan
# the week" walks this week's Canvas deadlines and to-dos, proposes one
# slot each, and takes yes / move that to Thursday / skip it / that's
# enough. Writes go to timekeeper.add_reminder ONLY -- calendar.add_event
# exists, but a plan built in ninety seconds should stay cheap to undo.
# ------------------------------------------------------------------
_PLAN_WEEK_RX = re.compile(
    r"^(?:let(?:'s| us)|lets|shall we|can we|help me|i want to|"
    r"time to|we should)?\s*"
    r"(?:plan|sort out|map out|lay out|block out|work out|plan out)\s+"
    r"(?:my |the |this |our )?week(?: ahead| out| together)?"
    r"(?:[,]?\s*(?:please|sir|jarvis))?[?.!\s]*$", re.I)
PLAN_PREPARING_LINE = "Let me see what the week is carrying, sir."
PLAN_BUSY_LINE = "I can't plan while I'm taking notes, sir."
SESSION_LOST_LINE = "I've lost the thread of that, sir."
PLAN_DUE_DAYS = 7
PLAN_TODO_LIMIT = 6
# Reminders are filed a little before the slot so the nudge lands while
# there is still time to start; the slot itself is the working hour.
PLAN_REMINDER_LEAD_MIN = 5


def _plan_items(c) -> list:
    """This week's Canvas deadlines and open to-dos, as PlanItems. Each
    source fails on its own -- no token, no Canvas, still a plan from the
    to-do list."""
    deadlines: list = []
    try:
        from jarvis.tools.canvas import read_due
        # read_due, not fetch_due: without a token the coursework still
        # arrives, from the Canvas calendar feed (jarvis/tools/canvas_ical.py).
        deadlines = list(read_due(c._svc("assistant"), PLAN_DUE_DAYS,
                                  c._svc("calendar")).items)
    except Exception:                       # noqa: BLE001 - outage, bad token
        log.info("plan the week: Canvas unavailable", exc_info=True)
    todos: list = []
    notes = c._svc("notes")
    if notes is not None:
        try:
            todos = list(notes.list("todo", limit=PLAN_TODO_LIMIT))
        except Exception:                   # noqa: BLE001
            log.exception("plan the week: to-do read failed")
    return dialogue_mod.collect_items(deadlines, todos)


def _file_plan(c, slots: list) -> bool:
    """Write the agreed slots as timekeeper reminders. Returns False when
    nothing could be filed, which changes the read-back line rather than
    pretending the week is booked."""
    tk = c._svc("timekeeper")
    if tk is None or not hasattr(tk, "add_reminder"):
        return False
    filed = 0
    for slot in slots:
        try:
            due = slot.when_epoch() - PLAN_REMINDER_LEAD_MIN * 60
            tk.add_reminder(due, slot.item.title)
            filed += 1
        except Exception:                   # noqa: BLE001 - one bad slot only
            log.exception("plan the week: could not file %r", slot.item.title)
    log.info("plan the week: filed %d of %d slots", filed, len(slots))
    return filed > 0


def _h_plan_week(c, t, m):
    """Open the planning session. The gather is a Canvas round trip, so it
    runs on a worker and the first question arrives through services.reply
    (_deliver), which closes the turn and arms the follow-up mic -- the
    same door the first quiz question uses."""
    if c.dictation or getattr(c, "lecture_course", None):
        return CommandResult(handled=True, reply=PLAN_BUSY_LINE, speak=True,
                             status="Plan refused")

    def _work():
        try:
            items = _plan_items(c)
        except Exception:                   # noqa: BLE001
            log.exception("plan the week: gather failed")
            items = []
        if not items:
            _deliver(c, dialogue_mod.PLAN_NOTHING_LINE)
            return
        session = dialogue_mod.WeekPlanner(
            items=items, today=date.today(),
            filer=lambda slots: _file_plan(c, slots))
        if not c.open_session(session):
            _deliver(c, PLAN_BUSY_LINE)
            return
        _deliver(c, f"{dialogue_mod.open_line(len(items))} {session.ask()}")

    c._bg(_work)
    return CommandResult(handled=True, reply=PLAN_PREPARING_LINE, speak=True,
                         ack=True, done=False, status="Planning the week")


def _int_setting(c, key: str, default: int) -> int:
    try:
        return max(1, int(_assistant_get(c, key, default) or default))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------
# The Board (jarvis/board.py + jarvis/ui/board.py): the docked
# mission-control panel on the empty right flank of the 4K display.
# ------------------------------------------------------------------
# The commander runs on worker threads and must never hold a reference to a
# Tk surface, so these three reach the window through services.board, whose
# methods publish a BoardCommand on the bus. "focus on the sessions" is NOT
# a highlight animation: it resolves to the one-line SPOKEN read of that
# panel's state, which is the only version of the verb worth having.
_BOARD_SHOW_RX = re.compile(
    r"^(?:bring|put|pull)\s+up\s+(?:the\s+)?board\b"
    r"|^(?:show|open|raise|give)\s+(?:me\s+)?(?:the\s+)?board\b"
    r"|^board\s+up\b")
_BOARD_HIDE_RX = re.compile(
    r"^(?:close|hide|dismiss|kill|drop)\s+(?:the\s+)?board\b"
    r"|^(?:take|put)\s+(?:the\s+)?board\s+(?:down|away)\b"
    r"|^board\s+(?:down|off)\b")
# "focus on the sessions" / "what's on the vitals panel". The panel NAME is
# resolved by jarvis.board.resolve_panel, not here: the alias table lives
# beside the panels it names so a new panel cannot be spoken about before
# it exists.
_BOARD_FOCUS_RX = re.compile(
    r"^focus\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+panel)?$"
    r"|^what'?s?\s+on\s+(?:the\s+)?(.+?)\s+panel$"
    r"|^read\s+(?:me\s+)?(?:the\s+)?(.+?)\s+panel$")


def _board_panel_name(text: str) -> str:
    """The Board panel key a "focus on ..." names, else "".

    Read by TWO callers: the handler below, and the voice-targeting chain in
    _handle_inner, which owns the word "focus" (TARGET_PATTERN) and would
    otherwise turn "focus on the sessions" into a window target before any
    registry entry ran. That chain already carves out the Claude project
    switch and the pomodoro for the same reason; this is the third."""
    m = _BOARD_FOCUS_RX.match(str(text or "").strip().lower().rstrip(".!?"))
    if not m:
        return ""
    for group in m.groups():
        if group:
            return board_mod.resolve_panel(group)
    return ""


def _board_svc(c):
    return c._svc("board")


# ---- the cast: grab-and-throw by voice (jarvis/gesturecast.py) ---------
# Hunter, 2026-09-03: "reach out and grab at the screen ... throwing the
# cast onto the HPCOMPUTER". The gesture itself is the camera's; these are
# the same verbs by voice, with the camera off, and the cancel for a live
# carry. EVERY pattern is whole-utterance and ends in a KNOWN target (the
# alias table lives in jarvis/cast.py beside the sinks), because the words
# collide with half the registry otherwise: "put milk on the shopping list"
# (list add), "throw" in the list-add opener, "drop the board" (board hide),
# "drop" in the cancel-schedule family, "let go" in ordinary speech. An
# unanchored "put it on" was exactly the class of Tier-1 hijack that ate
# longer utterances before, so the object is pinned to this/it/that and the
# target to the table.
_CAST_SINK = (
    # "the desktop" is NOT here: the file lane rules a bare "desktop" is a
    # FOLDER on this box ("put this on my desktop" stays a local request),
    # and the merge of the two lanes had this table quietly overruling that.
    r"(?P<sink>hp\s*computer|the\s+hp|the\s+pc|my\s+pc|"
    r"the\s+windows\s+(?:machine|box)|"
    r"the\s+other\s+(?:computer|machine)|(?:the\s+)?board|(?:the\s+)?spark|"
    r"(?:my|this)\s+screen|the\s+console|(?:the\s+)?handoff(?:\s+page)?|"
    r"the\s+page)")
_CAST_THROW_RX = re.compile(
    r"^(?:throw|cast|fling|toss|chuck|send)\s+(?:this|it|that)(?:\s+one)?\s+"
    r"(?:on|onto|at|to|over\s+to|up\s+on|across\s+to)\s+" + _CAST_SINK
    + r"[.!]*$", re.I)
_CAST_PUT_RX = re.compile(
    r"^put\s+(?:this|it|that)(?:\s+one)?\s+(?:on|onto|up\s+on)\s+"
    + _CAST_SINK + r"[.!]*$", re.I)
_CAST_DROP_RX = re.compile(
    r"^(?:drop\s+(?:it|that|this)|put\s+(?:it|that|this)\s+(?:down|back)|"
    r"let\s+(?:it|that)\s+go|let\s+go(?:\s+of\s+(?:it|that))?)[.!]*$", re.I)
_CAST_HOLDING_RX = re.compile(
    r"^(?:what\s+(?:am\s+i|are\s+you)\s+holding|"
    r"what(?:'s|\s+is)\s+in\s+(?:your|my)\s+hand)[?.!]*$", re.I)
_CAST_SIDE_RX = re.compile(
    r"^(?:which|what)\s+side\s+(?:is\s+)?" + _CAST_SINK
    + r"(?:\s+on)?[?.!]*$", re.I)
# The whole family, for the one rule that a sentence outranks a gesture:
# any OTHER utterance puts a live carry down (Commander._cast_spoken_over).
_CAST_FAMILY_RX = re.compile("|".join(
    "(?:%s)" % rx.pattern.replace("(?P<sink>", "(?:")
    for rx in (_CAST_THROW_RX, _CAST_PUT_RX, _CAST_DROP_RX,
               _CAST_HOLDING_RX, _CAST_SIDE_RX)), re.I)


def _cast_svc(c):
    return c._svc("gesture")


def _h_cast_throw(c, t, m):
    courier = _cast_svc(c)
    if courier is None:
        return None
    try:
        line, status = courier.throw_by_voice(m.group("sink"))
    except Exception:                            # noqa: BLE001 - service boundary
        log.exception("cast by voice failed")
        return CommandResult(handled=True, status="Cast",
                             reply="I couldn't manage that throw, sir.",
                             speak=True)
    return CommandResult(handled=True, reply=line or None, speak=bool(line),
                         status="Cast: %s" % status)


def _h_cast_drop(c, t, m):
    courier = _cast_svc(c)
    if courier is None:
        return None
    try:
        line = courier.drop_by_voice()
    except Exception:                            # noqa: BLE001 - service boundary
        log.exception("cast drop failed")
        return None
    return CommandResult(handled=True, reply=line, speak=True, status="Cast")


def _h_cast_holding(c, t, m):
    courier = _cast_svc(c)
    if courier is None:
        return None
    return CommandResult(handled=True, reply=courier.holding_line(),
                         speak=True, status="Cast")


def _h_cast_teach(c, t, m):
    """"HPCOMPUTER is on my right": m is (side, sink) from
    cast.parse_side_teaching, which only parses a KNOWN target."""
    courier = _cast_svc(c)
    if courier is None:
        return None
    side, sink = m
    try:
        line = courier.teach(side, sink)
    except Exception:                            # noqa: BLE001 - config boundary
        log.exception("cast teach failed")
        line = cast_mod.TAUGHT_FAILED_LINE
    return CommandResult(handled=True, reply=line, speak=True,
                         status="Cast: %s is %s" % (side, sink))


def _h_cast_side(c, t, m):
    courier = _cast_svc(c)
    if courier is None:
        return None
    return CommandResult(handled=True, reply=courier.side_line(m.group("sink")),
                         speak=True, status="Cast")


def _h_board_show(c, t, m):
    board = _board_svc(c)
    if board is None:
        return None
    try:
        already = bool(board.show())
    except Exception:                            # noqa: BLE001 - service boundary
        log.exception("board show failed")
        return CommandResult(handled=True, status="Board",
                             reply="I couldn't raise the board, sir.",
                             speak=True)
    line = "Already up, sir." if already else "The board, sir."
    return CommandResult(handled=True, reply=line, speak=True, status="Board")


def _h_board_hide(c, t, m):
    board = _board_svc(c)
    if board is None:
        return None
    try:
        board.hide()
    except Exception:                            # noqa: BLE001 - service boundary
        log.exception("board hide failed")
    return CommandResult(handled=True, reply="Board down, sir.", speak=True,
                         status="Board")


def _h_board_focus(c, t, m):
    """"Focus on the sessions" -> that panel lights AND he speaks its state.

    Returns None on a name that resolves to no panel, deliberately: "focus
    on the thesis" is a study session, not a board verb, and the matcher is
    loose enough to catch it. Falling through leaves it to the registry
    entries that own those words."""
    board = _board_svc(c)
    if board is None:
        return None
    name = ""
    for group in (m.groups() if hasattr(m, "groups") else ()):
        if group:
            name = str(group)
            break
    if not name:
        return None
    try:
        line = str(board.read(name) or "")
    except Exception:                            # noqa: BLE001 - service boundary
        log.exception("board read failed")
        return None
    if not line:
        return None
    return CommandResult(handled=True, reply=line, speak=True, status="Board")
# Time to leave (jarvis/leavetime.py, 2026-08-30)
# ------------------------------------------------------------------
# The walk to each building is LEARNED, never guessed: Jarvis asks once in
# passing after a heads-up and stores the answer in long-term memory. These
# three commands are the manual doors onto the same table -- teaching a walk
# outright, amending the last one ("make that ten next time", which
# correction_kind() does NOT match), and asking what he has. Every one of
# them returns None rather than inventing a building: an unresolvable place
# belongs to the model, not to a table of walks.
def _note_leave_touch(c) -> None:
    """Stamp the moment THIS conversation last named a building.

    In-memory and monotonic on purpose: LeaveTimes.last_key is restored from
    disk, and a background tick (leavetime._file_reminder / _maybe_ask) can
    re-point it with no user turn at all. Neither of those may make "make
    that ten" mean a building from days ago.
    """
    c._leave_touch = time.monotonic()


def _leave_key_fresh(c, lt) -> bool:
    """Is ``lt.last_key`` still the building he is talking about?"""
    stamp = getattr(c, "_leave_touch", 0.0) or 0.0
    # Seam: if jarvis/leavetime.py ever stamps its own touches (last_touch,
    # monotonic), honour whichever is newer -- it sees the ones the
    # commander never handles.
    other = getattr(lt, "last_touch", None)
    if isinstance(other, (int, float)) and not isinstance(other, bool):
        stamp = max(float(stamp), float(other))
    if not stamp:
        return False
    return time.monotonic() - float(stamp) <= LEAVE_AMEND_WINDOW_S


def _h_leave_set(c, t, m):
    lt = c._svc("leavetime")
    place, minutes = m
    key = lt.resolve(place)
    if key is None:
        return None                              # not a building he has: the model's
    value = lt.learn(key, minutes)
    _note_leave_touch(c)
    return CommandResult(handled=True, speak=True,
                         reply=leave_mod.LEARNED_LINE.format(
                             place=leave_mod.speech_name(key), minutes=value),
                         status=f"Walk: {leave_mod.speech_name(key)} {value} min")


def _h_leave_amend(c, t, m):
    lt = c._svc("leavetime")
    key = lt.last_key
    if not key:
        return None                              # nothing to amend: the model's
    if not _leave_key_fresh(c, lt):
        # No building has been on the table this conversation, so "make it
        # twenty" is about something else entirely (a timer, a volume, a
        # recipe). Overwriting a persisted walk off a disk-loaded key is
        # the silent damage; the model can have the words.
        log.info("leavetime: refusing a stale amend of %s", key)
        return None
    value = lt.learn(key, m)
    _note_leave_touch(c)
    return CommandResult(handled=True, speak=True,
                         reply=leave_mod.LEARNED_LINE.format(
                             place=leave_mod.speech_name(key), minutes=value),
                         status=f"Walk: {leave_mod.speech_name(key)} {value} min")


def _h_leave_query(c, t, m):
    lt = c._svc("leavetime")
    key = lt.resolve(m)
    if key is None:
        return None
    place = leave_mod.speech_name(key)
    minutes = lt.table.get(key)
    lt.note_key(key)
    _note_leave_touch(c)                         # he named it: "make that ten" may follow
    if minutes is None:
        # Asking back and then dropping the answer on the floor is the
        # dead-end this repo has been bitten by before ("Was that for me?"
        # was a toast nothing listened to). Arm the SAME pending slot the
        # proactive ask uses, so "about twelve minutes" lands in the table.
        # A refusal means another question owns the floor and the answer
        # would be eaten by its rung -- say what he asked and stop there
        # rather than asking a question nothing is listening for.
        if c.ask_leave_time(key, place) is False:
            return CommandResult(handled=True, speak=True, status="Walk unknown",
                                 reply=leave_mod.UNKNOWN_LINE)
        return CommandResult(handled=True, speak=True, status="Walk unknown",
                             reply=f"{leave_mod.UNKNOWN_LINE} "
                                   f"{leave_mod.ASK_LINE.format(place=place)}")
    return CommandResult(handled=True, speak=True,
                         reply=f"{place} is a {minutes} minute walk, sir.",
                         status=f"Walk: {place} {minutes} min")


def _h_leave_forget(c, t, m):
    """"Forget the walk to Wisenbaker" -- the way back out of the table.

    A walk is taught from a half-heard number in passing, so a wrong one is
    routine; without this the only exit was to overwrite it with another
    guess. Returns None for anything that is not a building he has taught,
    so the words go to the model rather than becoming a bogus confirmation.
    """
    lt = c._svc("leavetime")
    key = lt.resolve(m)
    if key is None or lt.table.get(key) is None:
        return None                              # not a walk he has: the model's
    lt.forget(key)
    place = leave_mod.speech_name(key)
    return CommandResult(handled=True, speak=True,
                         reply=leave_mod.FORGOT_LINE.format(place=place),
                         status=f"Walk forgotten: {place}")


REGISTRY: list[Command] = [
    # Offline mode FIRST, ahead of everything (jarvis/sensing.py). Not for
    # ambiguity -- every one of these matchers is a whole-utterance regex
    # over a closed vocabulary -- but because "stop watching" and "turn off
    # the sensors" are the two orders in this app that must never be
    # shadowed by a future entry that starts claiming "stop" or "turn off".
    # Status before the setters, so "are the sensors off" is a question.
    Command("sensing status", _SENSING_STATUS_RX.match, _h_sensing_status),
    # ...and the schedule before the switch, so "no cameras for two hours"
    # is a bounded offline and not an open-ended one.
    Command("sensing curfew", _SENSING_CURFEW_RX.match, _h_sensing_curfew),
    Command("sensing hold", _SENSING_HOLD_RX.match, _h_sensing_hold),
    Command("sensing on", _SENSING_ON_RX.match, _h_sensing_on),
    Command("sensing off", _SENSING_OFF_RX.match, _h_sensing_off),
    # Face enrolment, in the same family and for the same reason: it is
    # about the lens and about biometric data, and it must never be shadowed
    # by a later entry that starts claiming "add" or "forget". All three
    # matchers are whole-utterance regexes ending in "face" or "gallery", so
    # they shadow nothing themselves. The QUESTION goes first, so "who do
    # you recognise" is never read as an instruction.
    Command("face gallery", _FACE_GALLERY_RX.match, _h_face_gallery),
    Command("face forget", _FACE_FORGET_RX.match, _h_face_forget),
    Command("face enrol", _FACE_ENROL_RX.match, _h_face_enrol),
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
    # BEFORE the clock: clock_kind matches "what time is my class" and
    # answered it with the wall clock (live 2026-09-02, "It's 10:25 in the
    # morning, sir."). The handler returns None when the calendar was
    # never read, so an unconfigured box still falls through to the model.
    Command("next class", _NEXT_CLASS_RX.match, _h_next_class),
    Command("clock", clock_kind, _h_clock),              # Tier 1 clock
    Command("math", math_kind, _h_math),                 # Tier 1 arithmetic
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
    Command("scan syllabus", syllabus_kind, _h_scan_syllabus),
    Command("review flashcards", review_kind, _h_review),
    Command("stop quiz", quiz_stop_kind, _h_quiz_stop),
    # after "quiz": "quiz me on X" is its own verb, not a lesson
    Command("teach me", teach_kind, _h_teach),
    Command("plan week", _PLAN_WEEK_RX.match, _h_plan_week),
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
    # Before "recall": episodic ("when did I last …") vs semantic ("what did
    # I say about …"). Neither matcher claims the other's words, but the
    # pair is read together and the order records which owns "when".
    Command("last seen", _LAST_SEEN_RX.match, _h_last_seen, needs=("memory",)),
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
    # the ledger reads files, not the live session: no needs=("focus",)
    Command("study total", _STUDY_TOTAL_RX.match, _h_study_total),
    Command("study streak", _STREAK_RX.match, _h_study_streak),
    # AFTER the focus family, deliberately: "focus session on the thesis"
    # is a pomodoro and _BOARD_FOCUS_RX is loose enough to claim it. The
    # board handler also returns None on a name that resolves to no panel,
    # so an unrelated "focus on ..." still reaches the model.
    Command("board show", _BOARD_SHOW_RX.match, _h_board_show,
            needs=("board",)),
    Command("board hide", _BOARD_HIDE_RX.match, _h_board_hide,
            needs=("board",)),
    Command("board focus", _BOARD_FOCUS_RX.match, _h_board_focus,
            needs=("board",)),
    # Grab and throw by voice (jarvis/gesturecast.py). BEFORE the list
    # family: "put it on the board" and "add milk to the shopping list"
    # share an opener, and each pattern here ends in a known target so the
    # list's "<name> list" tail can never match it -- the collision tests
    # in tests/test_commander.py pin both directions.
    Command("cast throw", _CAST_THROW_RX.match, _h_cast_throw,
            needs=("gesture",)),
    Command("cast put", _CAST_PUT_RX.match, _h_cast_throw,
            needs=("gesture",)),
    Command("cast drop", _CAST_DROP_RX.match, _h_cast_drop,
            needs=("gesture",)),
    Command("cast holding", _CAST_HOLDING_RX.match, _h_cast_holding,
            needs=("gesture",)),
    Command("cast side", _CAST_SIDE_RX.match, _h_cast_side,
            needs=("gesture",)),
    Command("cast teach", cast_mod.parse_side_teaching, _h_cast_teach,
            needs=("gesture",)),
    Command("timer", _TIMER_RX.match, _h_timer),
    Command("alarm", _ALARM_RX.match, _h_alarm),
    Command("no asides", _m_no_asides, _h_no_asides),
    Command("list schedule", _LIST_SCHED_RX.match, _h_list_schedule,
            needs=("timekeeper",)),
    Command("cancel schedule", _CANCEL_SCHED_RX.match, _h_cancel_schedule,
            needs=("timekeeper",)),
    # After set/list/cancel, so "set a timer for 10 minutes" is untouched;
    # "extend that timer by 10 minutes" never reaches the model (2026-09-01).
    Command("adjust schedule", _ADJUST_SCHED_RX.match, _h_adjust_schedule,
            needs=("timekeeper",)),
    Command("briefing", _BRIEFING_RX.match, _h_briefing, needs=("brain",)),
    Command("preview", _PREVIEW_RX.match, _h_preview, needs=("brain",)),
    # Jarvis's own week before Hunter's: "weekly review" is the self-review,
    # "how's my week looking" the calendar forecast below it.
    Command("week review", _WEEKREVIEW_RX.match, _h_week_review),
    Command("week", _WEEK_RX.match, _h_week, needs=("brain",)),
    Command("briefing section", _PREF_SECTION_RX.match, _h_pref_section,
            needs=("assistant",)),
    Command("verbosity", _PREF_VERBOSITY_RX.match, _h_verbosity,
            needs=("assistant",)),
    Command("last mail", _LAST_MAIL_RX.search, _h_last_mail,
            needs=("brain",)),
    # Liked Songs, forced onto the tool with shuffle decided from the words
    # (never by the model). needs only the brain: the tool is registered
    # with it, and services.spotify is a convenience, not a requirement.
    Command("liked songs", liked_songs_kind, _h_liked_songs, needs=("brain",)),
    # "start playing my Spotify": one meaning, one tool action, forced the
    # same way. After "liked songs" so "play my liked songs" is never read
    # as a bare resume (the matcher also refuses it, belt and braces).
    Command("music resume", music_resume_kind, _h_music_resume, needs=("brain",)),
    Command("diagnostics", _DIAG_RX.match, _h_diagnostics),
    Command("register", register_kind, _h_register, needs=("brain",)),
    Command("next exam", _NEXT_EXAM_RX.match, _h_next_exam),
    # The learned walks. Ahead of "answer question" / "quick command", which
    # would swallow "how long to Wisenbaker" as a general question.
    Command("leave time", leave_mod.leave_set_kind, _h_leave_set,
            needs=("leavetime",)),
    Command("leave time amend", leave_mod.leave_amend_kind, _h_leave_amend,
            needs=("leavetime",)),
    Command("leave time query", leave_mod.leave_query_kind, _h_leave_query,
            needs=("leavetime",)),
    Command("leave time forget", leave_mod.leave_forget_kind, _h_leave_forget,
            needs=("leavetime",)),
    Command("day review", _DAYREVIEW_RX.match, _h_dayreview),
    # Undo before report: "forget what you filed" must never be read as a
    # question about what was filed.
    Command("garden undo", _GARDEN_UNDO_RX.match, _h_garden_undo),
    Command("garden report", _GARDEN_REPORT_RX.match, _h_garden_report),

    # Before "whats wrong": "where's your voice coming out" is a question
    # about the speaker, not about the fault lane.
    Command("audio out", _AUDIO_OUT_RX.match, _h_audio_out, needs=("soundbar",)),

    Command("quietly", _QUIETLY_RX.match, _h_quietly, needs=("health_watchdog",)),
    Command("whats wrong", _WHATS_WRONG_RX.match, _h_whats_wrong),
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
    # The Oracle box, outbound only. Status FIRST: "what's running on the
    # oracle box" would otherwise be read by the action door as a command
    # named "what's running" and refused instead of answered. `needs` stays
    # empty on purpose -- these read the assistant config, not a service, so
    # they must still be reachable on a box where nothing else is wired.
    Command("oracle status", _m_oracle_status, _h_oracle_status),
    Command("oracle logs", _ORACLE_LOGS_RX.match, _h_oracle_logs),
    Command("oracle action", _ORACLE_ACTION_RX.match, _h_oracle_action),
    # The bare-name question ("is knightfall up") comes AFTER the three
    # above: "oracle status" would otherwise be read as a service called
    # "oracle" and refused instead of answered. Its handler returns None on
    # a name that is not one of his services, so the words carry on to the
    # rest of the table and then to the model.
    Command("oracle service", _ORACLE_SERVICE_RX.match, _h_oracle_service),
    # ...and the refusal door LAST of the five, so every phrasing that IS
    # answerable has already been taken by one of them.
    Command("oracle freeform", _ORACLE_FREEFORM_RX.match, _h_oracle_freeform),
    # HPCOMPUTER (jarvis/tools/remote.py). Same shape as the Oracle five and
    # for the same reason: the two doors that MOVE something come before the
    # ones that only ask, and the refusal door is LAST of the family, so
    # "put the budget on HPCOMPUTER" is a transfer and never an order that
    # gets refused. `remote push`/`remote pull` are ahead of `remote query`
    # because "send me the disk report from HPCOMPUTER" is a file, not the
    # `disk` row of the question table.
    Command("remote push", _REMOTE_PUSH_RX.match, _h_remote_push),
    Command("remote pull", _REMOTE_PULL_RX.match, _h_remote_pull),
    Command("remote status", _REMOTE_STATUS_RX.match, _h_remote_status),
    Command("remote query", _REMOTE_QUERY_RX.match, _h_remote_query),
    # ...and the refusal door is NOT here. It is the loosest matcher in the
    # family (any clause of 2-120 characters beside the host's name), and
    # sitting at this index it outranked "list add" and "remind me": "clear
    # the shopping list on my desktop computer" was refused instead of
    # clearing the list. It is registered below both of them instead, so
    # every command that has a real answer is tried first and only then is
    # an order to the other machine refused. Its entry is below "remind me".
    # Email a file (jarvis/outbox.py). AFTER the HPCOMPUTER family: "send
    # the budget to HPCOMPUTER" is a transfer and `remote push` must have
    # it, and _SEND_NOT_RX names _HPC again so the two guards are
    # independent. The handler returns None for everything the negative
    # table catches, so an ordinary sentence with "send" in it falls
    # through to the router exactly as it did before.
    Command("send file", _SEND_FILE_RX.match, _h_send_file),
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
    # After the named form (which is the more specific match) and before
    # the to-do commands, which never see "... off the list" at all.
    Command("list strike anon", _LIST_STRIKE_ANON_RX.match,
            _h_list_strike_anon, needs=("notes",)),
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
    # HPCOMPUTER's refusal door, moved down out of its own family (see the
    # note there). LAST thing that may claim a sentence naming the host:
    # everything above it either answers the request or leaves it alone.
    Command("remote freeform", _REMOTE_FREEFORM_RX.match, _h_remote_freeform),
    # ambient (jarvis/quiet.py): before "free" so "I'm free until seven"
    # does not read as a DND request, and status before the hours setter.
    # "room tone" comes first of all: "stop the room tone" must not be read
    # by the media handlers as "stop", and it needs only the config.
    Command("room tone", _ROOM_TONE_RX.match, _h_room_tone, needs=("assistant",)),
    # the window look (jarvis/ui/theme.py): a config write, spoken with its
    # "after a restart" caveat; no needs= so a missing config is SAID.
    Command("ui look", _UI_LOOK_RX.match, _h_ui_look),
    # the conversation pane: BELOW the named-list family on purpose, so
    # that "clear the shopping list" is claimed by "list clear" before this
    # rung is ever asked. No needs=: the transcript is the main window.
    Command("clear transcript", _TRANSCRIPT_CLEAR_RX.match, _h_transcript_clear),
    Command("quiet status", _QUIET_STATUS_RX.match, _h_quiet_status, needs=("quiet",)),
    Command("quiet hours off", _QUIET_HOURS_OFF_RX.match, _h_quiet_hours_off,
            needs=("quiet",)),
    Command("quiet hours", _QUIET_HOURS_RX.match, _h_quiet_hours, needs=("quiet",)),
    Command("do not disturb", _DND_RX.match, _h_dnd, needs=("quiet",)),
    Command("free", _FREE_RX.match, _h_free, needs=("quiet",)),
    # room control (jarvis/room.py, jarvis/scenes.py): the scene BEFORE the
    # light, so "lights out" is the whole wind-down and not one dim step.
    Command("scene", scene_kind, _h_scene, needs=("scenes",)),
    Command("room light", room_light_kind, _h_room_light, needs=("room_light",)),
]

# The assistant's Tier 1 without the "jarvis" prefix (jarvis mode): the
# same handlers, in the same order, run right before the router so that a
# typed "timer for 5 minutes" is instant and never a model round trip.
ASSISTANT_TIER1: list[Command] = [
    cmd for cmd in REGISTRY
    if cmd.name in ("explain document", "quiz", "scan syllabus", "teach me",
                    "review flashcards", "stop quiz",
                    "plan week",
                    "focus start", "focus left", "focus end", "lecture notes",
                    "study total", "study streak",
                    # the Board: "bring up the board" is said at the desk
                    # without a wake-word prefix, like every other surface verb
                    "board show", "board hide", "board focus",
                    # the cast: "throw this on HPCOMPUTER" / "drop it" /
                    # "HPCOMPUTER is on my right" are said at the desk,
                    # mid-gesture, with no wake word left to strip
                    "cast throw", "cast put", "cast drop", "cast holding",
                    "cast side", "cast teach",
                    "timer", "alarm", "no asides",
                    "list schedule", "cancel schedule", "adjust schedule",
                    "briefing", "preview", "week", "briefing section", "verbosity",
                    "last mail", "diagnostics", "register", "next exam",
                    # "email the lab report to Heather" arrives at the desk
                    # with the wake word already eaten, like every other
                    # instruction he gives standing up. Without this name
                    # the registry pass never runs on it and the intent
                    # gate calls it background chat.
                    "send file",
                    # "what's my next class" arrives with the wake word
                    # already eaten, like every other question at the desk
                    "next class",
                    # "play my liked songs" arrives by voice with the wake
                    # word already eaten; without this name it would reach
                    # the router and the model round trip it exists to skip
                    "liked songs",
                    # "start playing my Spotify" (20:56:42) arrived the same
                    # way -- and as a clause of a compound, which _try_multi
                    # can only run when every clause is a Tier-1 name
                    "music resume",
                    # "good night" is the other half of "good morning": the
                    # hotword eats the wake word, so the courtesy arrives
                    # bare, the prefixed registry pass is skipped and the
                    # intent gate called two words background chat -- which
                    # left the whole wind-down (music fade, screen dim, DND)
                    # unreachable by voice, its only call site being
                    # _h_courtesy on "goodnight".
                    "greeting", "courtesy", "day review",
                    # the weekly self-review and the memory garden's two
                    # answers: all three are asked without the wake word
                    "week review", "garden report", "garden undo",
                    # the learned walks: "it takes ten minutes to get to
                    # Wisenbaker" arrives by voice with no prefix left to strip
                    "leave time", "leave time amend", "leave time query",
                    # ...and the way back out: "forget the walk to
                    # Wisenbaker" arrives bare like the other three.
                    "leave time forget",
                    "todo done", "todo add",
                    "todo list",
                    # named lists: spoken in the aisle and read back over
                    # SSH from the phone, neither with a wake-word prefix
                    "list add", "list read", "list strike", "list clear",
                    # ...and the unnamed form, for the same reason: "cross
                    # the second one off the list" is said straight after
                    # the list was read out, with the wake word already
                    # eaten. Without this name it is a REGISTRY entry only,
                    # which unprefixed text never reaches -- which is why
                    # his #32 still ended at the Claude offer.
                    "list strike anon",
                    "lists",
                    "take note", "show notes", "answer question", "remind me",
                    # long-term memory, the people book and the day recap
                    # answer without the wake-word prefix too: unprefixed
                    # "remember that ..." used to reach the router and the
                    # notes tool instead of the memory it was pitched for
                    "person", "remember", "recall", "last seen", "who is", "recap",
                    "quiet status", "quiet hours off", "quiet hours", "do not disturb",
                    "free", "room tone",
                    # offline mode: the hotword eats the wake word, so
                    # "stop watching" arrives bare. Without Tier 1 it would
                    # reach the router and be answered by a model that
                    # cannot switch a sensor off -- the one failure this
                    # feature cannot survive.
                    "sensing status", "sensing curfew", "sensing hold",
                    "sensing on", "sensing off",
                    # "enrol my face" and "who do you recognise" are said at
                    # the desk with the wake word already eaten, like every
                    # other surface verb -- and without Tier 1 they would
                    # reach a model that cannot open a gallery and would
                    # answer the question by inventing an answer.
                    "face gallery", "face forget", "face enrol",
                    # "switch to classic visuals" is said AT the window he
                    # is looking at, wake word already eaten like the rest
                    "ui look",
                    # ...and so is "clear the transcript", which arrived bare
                    # at 23:26:01 on 09-02 and was answered "Ignored
                    # (background chat, conf=0.80)": three words with no
                    # Tier-1 name are exactly what the intent gate drops.
                    "clear transcript",
                    # "where's your voice coming out" is asked AT the dead
                    # speaker, which is exactly when the wake word is least
                    # likely to have been heard.
                    "audio out",
                    # room control: "dim it a little" / "lights up" / "power
                    # down the workshop" arrive by voice with the wake word
                    # already consumed, so they need the unprefixed pass too
                    "scene", "room light",
                    "standup", "gpu reclaim", "gpu lend",
                    # the Oracle box: "how's the haymaker bot" is asked at
                    # the desk with the wake word already eaten by the
                    # hotword, like every other surface question
                    "oracle status", "oracle logs", "oracle action",
                    "oracle service", "oracle freeform",
                    # HPCOMPUTER, for exactly the reason the Oracle five are
                    # here and by the same oversight that kept them out
                    # once: the hotword eats the wake word, so "put the lab
                    # report on HPCOMPUTER" arrives bare, the prefixed
                    # registry pass never runs on it, and every one of the
                    # five doors was unreachable by voice -- three dropped
                    # by the intent gate, two handed to the model. The
                    # refusal door is here too, because a refusal he cannot
                    # hear is not the safety story this lane claims.
                    "remote push", "remote pull", "remote status",
                    "remote query", "remote freeform",
                    "log triage", "whats wrong", "quietly", "slow turn",
                    # the hotword consumes the wake word, so spoken text never
                    # reaches the prefixed registry: without this the router
                    # would hand Claude the bare words "fix what i copied".
                    "clip to claude",
                    # arithmetic and unit conversion answer unprefixed too:
                    # "what's 18 percent of 74" is a question, not a command,
                    # and nobody says "jarvis" before a sum.
                    "math",
                    "read control")
]


def _tier1_send_file(t):
    """The Tier-1 probe for the one irreversible family in the app.

    A Tier-1 name match switches the intent classifier OFF for the whole
    utterance (``_handle_inner``: "tier-1 match %r bypasses the intent
    gate"), so whatever matcher stands here IS the gate for this family --
    and the raw ``_SEND_FILE_RX`` is far too generous to be it. Unaddressed
    speech of the shape "send X to Y" armed a live draft and read it back
    out loud, and a backchannel "yeah" then sent the file; sentences the
    negative table exists to veto ("send my regards to Heather", "send a
    text to Heather", "send the money to Ali") bypassed the gate too and
    were answered by the model instead of dropped as background chat.

    So the veto runs HERE, before the gate is switched off, rather than
    only inside the handler afterwards. The registry keeps the wide
    matcher: a sentence he prefixed with "jarvis" was addressed to me by
    construction and needs no gate at all.

    The rest of the claim rule -- a recipient he can actually write to, or
    a file that is really on his disk -- needs services a matcher does not
    get, so ``Commander._match_assistant`` applies it beside this.
    """
    if _SEND_NOT_RX.search(t):
        return None
    return _SEND_FILE_RX.match(t)


ASSISTANT_TIER1 = [
    cmd if cmd.name != "send file"
    else Command(cmd.name, _tier1_send_file, cmd.handler, cmd.needs)
    for cmd in ASSISTANT_TIER1
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
# Day-shift follow-ups: "and the next day" (2026-08-31)
# ------------------------------------------------------------------
# Live, 21:30:37: "What do I have going on tomorrow?" was answered from
# get_calendar. 21:30:50: "and the next day" -- four words, no subject, no
# day of its own -- reached the model as itself (route local (short)), the
# model never called get_calendar again, and it read TOMORROW'S list back
# verbatim. The very next utterance, "and what about the day after that?",
# went through the classifier instead (route local (classify)), DID call
# get_calendar and answered Wednesday correctly: the model can resolve the
# day when it bothers to look, so the defect is that a bare fragment is
# handed over with nothing to look at.
#
# The repair is anaphora, not a new tool: take the day word out of the
# question he just asked, move it on by one, and re-dispatch HIS OWN
# sentence with the new day in it. That keeps the subject ("what do I have
# going on", "what's the weather") instead of guessing that every
# follow-up is about the calendar, and it hands the router a full question
# with an explicit day, which is what makes the tool call happen.
FOLLOWUP_DAY_WINDOW_S = 180.0
_WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday",
                  "saturday", "sunday")
# The fragment must BE the whole utterance -- "the next day I'm free" is a
# sentence, not a follow-up.
_DAY_SHIFT_RX = re.compile(
    r"^(?:and|so|ok(?:ay)?|well)?[,\s]*"
    r"(?:what|how)\s+about\s+|"
    r"^(?:and|so|ok(?:ay)?|well)?[,\s]*", re.I)
_DAY_SHIFT_TAIL_RX = re.compile(
    r"^(?:the\s+)?(?:next|following)\s+day$|"
    r"^(?:the\s+)?day\s+after(?:\s+that)?$", re.I)
_DAY_ANCHOR_RX = re.compile(
    r"\b(today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)\b", re.I)


def day_shift_followup(prev_text: str, text: str,
                       today=None) -> Optional[str]:
    """His previous question re-asked one day later, or None.

    ``None`` means "not this shape" and the caller carries on exactly as
    before: either the utterance is not a bare day-shift fragment, or the
    question before it named no day to move.
    """
    t = (text or "").strip().strip("?.!,")
    if not t:
        return None
    tail = _DAY_SHIFT_RX.sub("", t, count=1).strip()
    if not _DAY_SHIFT_TAIL_RX.match(tail):
        return None
    prev = (prev_text or "").strip()
    if not prev:
        return None
    hits = list(_DAY_ANCHOR_RX.finditer(prev))
    if not hits:
        return None
    m = hits[-1]                      # the day he ended on
    word = m.group(1).lower()
    today = today or date.today()
    if word in ("today", "tonight"):
        base = today
    elif word == "tomorrow":
        base = today + timedelta(days=1)
    else:
        # The NEXT such day counting today -- the same rule
        # tools.calendar.format_events uses, so the two cannot drift.
        want = _WEEKDAY_NAMES.index(word)
        base = today + timedelta(days=(want - today.weekday()) % 7)
    target = base + timedelta(days=1)
    delta = (target - today).days
    if delta == 1:
        label = "tomorrow"
    elif 2 <= delta <= 6:
        # A weekday name inside a week is unambiguous; at 7 days out it
        # would name TODAY to every downstream parser, so refuse instead
        # of answering about the wrong day.
        label = _WEEKDAY_NAMES[target.weekday()].capitalize()
    else:
        return None
    return prev[:m.start()] + label + prev[m.end():]


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
# The leave-time question ("how long do you need to get to Wisenbaker,
# sir?") stays answerable for a few minutes -- he is usually packing a bag
# when it lands -- but only a duration-shaped reply answers it.
LEAVE_ANSWER_WINDOW_S = 180.0
# "make that ten next time" amends the building last talked about, and that
# subject goes stale like every other pending state here. LeaveTimes.last_key
# is LOADED FROM DISK at construction, so without a window a bare "make it
# twenty" on a fresh boot -- or hours after a timer, an alarm, or the
# background reminder tick quietly re-pointed the key -- silently rewrote a
# stored walk for a building nobody had mentioned that day.
LEAVE_AMEND_WINDOW_S = 180.0
LEAVE_DROPPED_LINE = "As you wish, sir; I'll not ask again."
_LEAVE_DECLINE_RX = re.compile(
    r"^(?:no|nope|nah|never\s?mind|forget it|skip it|don'?t worry|"
    r"i (?:don'?t|do not) know|dunno|no idea|not sure|who knows)\b", re.I)

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
# "Belay that last order" NAMES the thing it takes back; it is not a stray
# two-word transcript, so it is honoured for longer than a bare "scratch
# that". Live 2026-09-01: the timer was set at 21:03:18 and the belay came
# at 21:10:23 -- seven minutes, and he meant it.
UNDO_EXPLICIT_WINDOW_S = 15 * 60.0
_UNDO_OBJECT = r"(?:order|command|instruction|request|thing|one|action|step)"
_UNDO_LAST = r"(?:that|the|my)\s+last(?:\s+" + _UNDO_OBJECT + r")?"
# Two classes of verb. "Scratch", "undo", "belay" and "take back" mean
# nothing but undo, so they may name their object outright ("belay that
# order"). "Cancel" and "forget" have a life of their own -- "cancel that
# one" right after a read-out of the timers is a cancellation, not a
# rewind, and it reached the model at d38b493 -- so they take back only the
# forms that say LAST, and with nothing to take back they still fall
# through to the model (see undo_explicit / _try_undo).
_UNDO_RX = re.compile(
    r"^(?:(?:no|nope)[,.!]?\s+)?(?:"
    r"(?:scratch|undo|belay|take back)\s+"
    r"(?:(?P<x1>" + _UNDO_LAST + r"|that\s+" + _UNDO_OBJECT + r")"
    r"|the last(?: one| thing)?|that|it|this)"
    r"|(?:cancel|forget)\s+"
    r"(?:" + _UNDO_LAST + r"|the last(?: one| thing)?|that|it|this)"
    r")"
    r"|^(?:scratch|undo|belay) that"
    r"|^undo(?: the)?(?: last)?(?: one| thing| action)?"
    r"|^take that back"
    r"|^(?:on second thought[s]?|actually)[,.]?\s+(?:scratch|undo|cancel) that",
    re.I)
_UNDO_TAIL_RX = re.compile(r"^[\s,.!]*(?:please|jarvis|sir|instead)?[\s,.!]*$", re.I)
# Spoken fillers Whisper writes down: "BELAY THAT LAST Uhhh... ORDER" (live
# 2026-09-01 21:10:23) was a perfect undo with a hesitation in it, and the
# hesitation sent it to the intent classifier, which asked "Was that for
# me?" and then let the model answer "I'll stand down" -- doing nothing.
_FILLER_RX = re.compile(
    r",?\s*(?<![a-z])(?:uh+m*|um+|er+m?|ah+|hmm+|mm+|like)(?![a-z])[,.!?…]*", re.I)
_ELLIPSIS_RX = re.compile(r"\.{2,}|…")


def strip_fillers(text: str) -> str:
    """The utterance without its ums, uhs and ellipses, whitespace
    collapsed. Case-insensitive and punctuation-tolerant; used only where
    a filler can carry no meaning (the undo phrase)."""
    t = _ELLIPSIS_RX.sub(" ", str(text or ""))
    t = _FILLER_RX.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip(" ,")


def _undo_match(text: str):
    """The _UNDO_RX match for a whole-utterance undo, else None."""
    t = strip_fillers(text)
    m = _UNDO_RX.match(t)
    if m and _UNDO_TAIL_RX.match(t[m.end():]):
        return m
    return None


def undo_kind(text: str) -> bool:
    """True when the utterance asks for the last action to be taken back."""
    return _undo_match(text) is not None


def undo_explicit(text: str) -> bool:
    """True for an undo that names its object with a verb that can mean
    nothing else -- "belay that last order", "scratch that last command",
    "undo my last request" -- as opposed to the bare "scratch that" or
    anything led by "cancel"/"forget". The explicit form is honoured for
    UNDO_EXPLICIT_WINDOW_S and, with nothing to take back, is answered
    rather than handed on; "cancel that last one" with nothing to take
    back keeps falling through to the model, which can still cancel the
    schedule item he means."""
    m = _undo_match(text)
    return bool(m) and m.group("x1") is not None


@dataclass
class LastTurn:
    text: str
    status: str
    ts: float


def _merge_acks(replies: list) -> Optional[str]:
    """One acknowledgement for a compound whose clauses answered alike.

    "Noted, sir: your mom is Heather." + "Noted, sir: your dad is Ali."
    -> "Noted, sir: your mom is Heather and your dad is Ali." Only when
    every clause used the SAME lead-in (the text before the first colon)
    and every one has a tail; anything else is two different answers and
    is joined as two sentences, as before.
    """
    if len(replies) < 2:
        return None
    heads, tails = [], []
    for reply in replies:
        head, sep, tail = str(reply).partition(":")
        tail = tail.strip().rstrip(".")
        if not sep or not tail or "." in head:
            return None            # no lead-in, or the "head" is a sentence
        heads.append(head.strip())
        tails.append(tail)
    if len(set(heads)) != 1 or len(set(tails)) != len(tails):
        return None                # different answers, or the same fact twice
    return f"{heads[0]}: {notes_mod.join_spoken(tails)}."


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
    # (source, strict, the tuple it describes) for the slot above. Kept
    # BESIDE the tuple rather than inside it because the tuple's 3-item
    # shape is written and read all over the app and its tests; the `ref`
    # leg means stale metadata can never be applied to a slot somebody set
    # by hand -- an unrecognised tuple falls back to the old, lenient rules.
    _pending_destructive_meta: Optional[tuple] = None
    # (candidates, resume, source, made_at) for "Which one, sir?"
    _pending_filepick: Optional[tuple] = None
    # The source of the turn being handled, so a question armed inside a
    # handler knows which channel it was asked on.
    _turn_source: str = "voice"
    # Set by the rungs that answer one of THIS commander's own questions,
    # read by _drop_stranded_questions.
    _answered_pending: bool = False
    # One re-ask per STRICT read-back, the same one-shot the send draft
    # carries on itself. Reset when the offer is armed and when it is spent.
    _strict_reasked: bool = False
    _pending_objection: Optional[tuple] = None
    _pending_send: Any = None                 # outbox.Draft awaiting a yes
    _objection_timer = None
    _objections = None
    # Monotonic; 0.0 means "no building has been named this session", which
    # is exactly what a disk-loaded LeaveTimes.last_key must count as.
    _leave_touch: float = 0.0

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
        # The lesson whose "say quiz me" offer is open: (topic, study text,
        # monotonic stamp). The study text is kept so the quiz is built from
        # exactly what was explained, with no second retrieval.
        self._pending_teach = None
        # Working sessions (jarvis/dialogue.py): the Session whose open
        # question the next utterance answers. The quiz's pattern
        # generalised -- see _try_session, one rung above the quiz.
        self._pending_session = None
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
        self._pending_destructive_meta: Optional[tuple] = None
        self._pending_filepick: Optional[tuple] = None
        self._turn_source = "voice"
        self._answered_pending = False
        self._strict_reasked = False
        self._pending_send = None
        # He advised against something and asked "shall I set it anyway?":
        # (run, spoken line, Objection, stamp). Its own slot because its
        # default on ambiguity is the OPPOSITE of the read-back's -- see
        # _try_objection_confirm.
        self._pending_objection: Optional[tuple] = None
        self._objection_timer = None
        self._objections = None
        # "How long do you need to get to Wisenbaker, sir?" is on the table:
        # (building key, spoken place, monotonic stamp). See ask_leave_time.
        self._pending_leave: Optional[tuple] = None
        # When a building was last named in conversation, so "make that ten
        # next time" cannot rewrite a walk restored from disk (_leave_key_fresh).
        self._leave_touch: float = 0.0

    # -- service access ------------------------------------------------
    def _svc(self, name: str):
        return getattr(self.services, name, None)

    def _bg(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def _speak(self, text: str):
        """Speak via the TTS service when talk-back is enabled.

        While ``_try_multi`` is running the clauses of ONE compound
        utterance the line is diverted into ``_suppress_speak`` instead of
        spoken on the spot: two clauses that each speak for themselves are
        two utterances for one breath ("Noted, sir: your mom is Heather."
        then "Noted: your dad is Ali." -- reported 2026-08-31, feature #57,
        "two noted said though"). The JOIN speaks the pair once.
        """
        held = getattr(self, "_suppress_speak", None)
        if held is not None:
            if text:
                held.append(text)
            return
        if not CONFIG.talkback or not text:
            return
        tts = self._svc("tts")
        if tts is None:
            return
        try:
            tts.speak(text)
        except Exception:
            log.exception("tts speak failed")

    # None = speak now; a list = a compound is running, park the lines
    # (see _speak / _try_multi).
    _suppress_speak = None

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
        self._cast_spoken_over(text)
        with self._turn_lock:
            self._confidence = confidence
            self._turn_source = source
            armed = (getattr(self, "_pending_send", None),
                     getattr(self, "_pending_filepick", None),
                     getattr(self, "_pending_destructive", None),
                     getattr(self, "_pending_destructive_meta", None))
            self._answered_pending = False
            result = self._handle_inner(text, source)
            self._drop_stranded_questions(armed, result, source)
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

    def _drop_stranded_questions(self, armed: tuple, result,
                                 source: str = "voice") -> None:
        """A question of MINE that survived a turn somebody ELSE answered is
        dropped, not left armed for the next yes-shaped sentence.

        Every pending rung clears its own slot before it does anything, and
        each is written as if it were the only one on the floor. But a
        ringing kitchen timer, a Claude permission prompt, an alarm offer
        and the first-wake briefing offer all arrive OUT OF BAND, and each
        of them sits ABOVE the read-back rungs and returns before they run.
        Measured: with a draft armed, "yes" answered a parked alarm offer,
        set the alarm -- and left the draft live for the rest of its 90 s,
        so the NEXT yes-shaped utterance, aimed at anything at all, sent
        the file. Same for the study offer, the briefing offer and "stop"
        to a ringing timer.

        The rungs that DID answer set ``_answered_pending``; anything else
        that consumed the turn spends the questions it talked over. Identity
        comparison, not truthiness: a question armed DURING this turn is a
        new object and is left alone.

        A turn from ANOTHER ROOM spends nothing. A Discord message or a
        tmux turn could not have answered a question put out loud at the
        desk, so it is not its cancellation either -- the question stays
        parked for the channel it was asked on, which is the same rule the
        confirm rungs themselves apply.
        """
        if self._answered_pending or result is None:
            return
        send, pick, destructive, meta = armed
        if send is not None and getattr(self, "_pending_send", None) is send \
                and self._same_room(getattr(send, "asked_from", "voice"), source):
            log.info("send read-back dropped: the turn was answered elsewhere")
            self._pending_send = None
        if pick is not None and getattr(self, "_pending_filepick", None) is pick \
                and self._same_room(pick[2] if len(pick) == 5 else "voice", source):
            log.info("which-one dropped: the turn was answered elsewhere")
            self._pending_filepick = None
        asked = meta[0] if (isinstance(meta, tuple) and len(meta) == 3
                            and meta[2] is destructive) else ""
        if destructive is not None and \
                getattr(self, "_pending_destructive", None) is destructive \
                and (not asked or self._same_room(asked, source)):
            log.info("read-back dropped: the turn was answered elsewhere")
            self._pending_destructive = None
            self._pending_destructive_meta = None

    def _cast_spoken_over(self, text: str) -> None:
        """A sentence outranks a gesture: any utterance that is not itself
        one of the cast verbs puts a LIVE carry down, quietly
        (jarvis/gesturecast.GestureCast.spoken_over). The cast verbs are
        exempt because "throw this on the board" while carrying IS the
        throw. getattr chains, not _svc: a slim test commander made with
        __new__ has no services at all."""
        courier = getattr(getattr(self, "services", None), "gesture", None)
        if courier is None:
            return
        t = text.strip().lower().rstrip(".!?")
        if _CAST_FAMILY_RX.match(t) or cast_mod.parse_side_teaching(t):
            return
        try:
            courier.spoken_over()
        except Exception:                        # noqa: BLE001 - service boundary
            log.debug("cast spoken_over failed", exc_info=True)

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

    def question_open(self) -> bool:
        """Jarvis put a question and is waiting on the answer.

        One predicate for every rung of ``handle`` that owns the floor, so
        callers stop each growing their own half-list. Two bugs came out of
        that: ``app._question_open`` sized the follow-up mic at 4 s through
        the study offer and the objection (both spoken yes/no questions
        Jarvis asked), and ``_ask_leave_time`` armed a leave question on top
        of a live quiz, where the duration answer is graded as a flashcard.

        Every entry carries its own expiry, so this is only ever True while
        a question is genuinely live -- an offer the rung would drop is not
        a question.
        """
        quiz = getattr(self, "_pending_quiz", None)
        if quiz is not None and not getattr(quiz, "finished", True):
            try:
                if not quiz.stale():
                    return True
            except Exception:  # noqa: BLE001 - a slim/duck-typed session
                log.debug("question_open: quiz staleness failed", exc_info=True)
        session = getattr(self, "_pending_session", None)
        if session is not None:
            try:
                if not (session.finished or session.stale()):
                    return True
            except Exception:  # noqa: BLE001 - a bad session
                log.debug("question_open: session state failed", exc_info=True)
        # A file send read back and not yet answered ("...Send it, sir?").
        # Its own expiry, outbox.DRAFT_TTL_S, which is longer than the
        # destructive read-back's because there are four facts to check.
        draft = getattr(self, "_pending_send", None)
        if draft is not None:
            try:
                if not draft.stale():
                    return True
            except Exception:  # noqa: BLE001 - a slim/duck-typed draft
                log.debug("question_open: send draft staleness failed",
                          exc_info=True)
        # "Which one, sir?" -- the ambiguous-file question both file lanes
        # ask. It holds the floor exactly as the read-back does: it is a
        # question Jarvis put out loud and is waiting on, and while it was
        # missing from this list the app gave its answer the 4 s window and
        # the strict confidence gate instead of the 15 s question window.
        pick = getattr(self, "_pending_filepick", None)
        if isinstance(pick, tuple) and len(pick) == 5:
            try:
                if time.monotonic() - float(pick[3]) <= FILEPICK_TTL_S:
                    return True
            except (TypeError, ValueError):
                pass
        # A read-back ("Cancel all three alarms, sir?") and the objection
        # ("Shall I set it anyway?") both hold the floor for DESTRUCTIVE_TTL_S.
        for name, size in (("_pending_destructive", 3), ("_pending_objection", 4)):
            pend = getattr(self, name, None)
            if isinstance(pend, tuple) and len(pend) == size:
                try:
                    if time.monotonic() - float(pend[-1]) <= DESTRUCTIVE_TTL_S:
                        return True
                except (TypeError, ValueError):
                    pass
        # "Shall I hand that to Claude, sir, or is it a quick one for me?"
        # is a question Jarvis asked too. The router owns it (its own
        # ASK_TTL_S, and .pending() self-expires), and it was missing here:
        # on 2026-08-31 "Cross the second one off the list" ended in that
        # question, and the yes/no that answers it got the 4 s window and
        # the strict confidence gate rather than the 15 s question window
        # and the salvage (app._salvage_low_confidence).
        # getattr, not _svc: a slim test commander has no .services at all.
        router = getattr(getattr(self, "services", None), "router", None)
        probe = getattr(router, "pending", None)
        if callable(probe):
            try:
                if probe() is not None:
                    return True
            except Exception:  # noqa: BLE001 - a slim/duck-typed router
                log.debug("question_open: router pending failed", exc_info=True)
        # The wake-alarm offer and the exam-week study offer are parked on
        # the SERVICES namespace by briefing.make_tools, and the first-wake
        # briefing offer by app._offer_first_wake_briefing -- not on the
        # commander, and all three are stamped in wall-clock seconds. The
        # briefing one carries a shorter life than the other two because it
        # opens the microphone for its own answer (BRIEFING_OFFER_TTL_S).
        services = getattr(self, "services", None)   # a slim test commander has none
        for name, ttl in (("alarm_offer", OFFER_TTL_S),
                          ("study_offer", OFFER_TTL_S),
                          ("briefing_offer", BRIEFING_OFFER_TTL_S)):
            offer = getattr(services, name, None)
            if isinstance(offer, dict) and offer:
                try:
                    made = float(offer.get("made_at") or 0.0)
                except (TypeError, ValueError):
                    made = 0.0
                if not made or time.time() - made <= ttl:
                    return True
        return False

    def ask_leave_time(self, key: str, place: str) -> bool:
        """The leave-time watch asked how long the walk is; the next
        duration-shaped utterance answers it (``_try_leave_answer``).

        False means "not armed, do not ask": another question already owns
        the floor. ``_try_leave_answer`` is the LAST pending rung on
        purpose, so a duration said while a quiz or a working session is
        open is eaten by that rung instead -- a flashcard marked wrong and
        the walk never learned. Worse, ``leavetime._maybe_ask`` burns its
        once-ever ask the moment the asker returns True, so the collision
        would spend the question for good rather than delay it. The watch
        retries on the next tick.
        """
        if self.question_open():
            log.info("leavetime: not asking about %s -- a question is already "
                     "on the table", key)
            return False
        self._pending_leave = (key, place, time.monotonic())
        # "make that ten next time" is only about this building while the
        # exchange is fresh (see _leave_key_fresh).
        self._leave_touch = time.monotonic()
        return True

    def stash_destructive(self, run: Callable[[], CommandResult], line: str,
                          strict: bool = False):
        """A handler read an action back instead of doing it; the next yes
        runs it (``_try_destructive_confirm``).

        ``strict`` swaps the answer grammar for ``parse_send_answer`` -- the
        end-anchored one the mail lane was built with -- and adds its one
        re-ask on a vague word. Pass it for anything as irreversible as an
        email: the two HPCOMPUTER transfers do, and they are the reason the
        flag exists. ``parse_yes_no`` waives its overheard-speech guard
        whenever the first word is a yes word and counts "sure" as one, so
        "Yeah, so you should be able to look that up." -- a real line from
        his log that answered nothing -- spent a read-back once already.
        Cancelling three alarms that way is a bad afternoon; putting a file
        on another machine that way cannot be taken back.
        """
        self._pending_destructive = (run, line, time.monotonic())
        self._pending_destructive_meta = (self._turn_source, bool(strict),
                                          self._pending_destructive)
        self._strict_reasked = False
        # The mirror of stash_send's clear. Two questions can never share
        # one yes, and until now that invariant held in one direction only
        # -- by the accident that _try_send_confirm sits above this rung
        # and drops its draft on any non-answer. A read-back armed outside
        # handle() (a proactive offer) had no such accident protecting it.
        self._pending_send = None

    def stash_filepick(self, candidates, resume: Callable):
        """"Which one, sir?" -- park the rivals so the answer can be heard.

        ``resume(path)`` carries on with the candidate he names and returns
        the CommandResult for that turn; for both lanes it ends in the
        ordinary read-back, so choosing a file still confirms nothing.

        This is the slot whose absence made the ambiguous branch a silent
        dead end: the question was asked, nothing was parked,
        ``question_open()`` said no, the app gave the follow-up microphone
        the short window and the answer was then dropped as background chat.
        """
        keep = [Path(c) for c in (candidates or ())]
        if not keep:
            self._pending_filepick = None
            return
        self._pending_filepick = (keep, resume, self._turn_source,
                                  time.monotonic(), False)
        # One question on the floor at a time, the same rule stash_send and
        # stash_destructive keep between themselves.
        self._pending_send = None
        self._pending_destructive = None
        self._pending_destructive_meta = None

    def stash_send(self, draft):
        """A file send was read back; only an explicit yes spends it.

        Its own slot, NOT ``_pending_destructive``: that one is answered by
        ``parse_yes_no``, a word bag wide enough that a passing "yeah" in a
        ten-word sentence counts. Cancelling three alarms by accident is a
        bad afternoon; sending a file by accident is permanent, so this slot
        has a narrower grammar of its own (``parse_send_answer``) and one
        re-ask for a vague answer.
        """
        try:
            draft.asked_from = self._turn_source
        except Exception:  # noqa: BLE001 - a slim/duck-typed draft in a test
            log.debug("stash_send: draft would not record its source",
                      exc_info=True)
        self._pending_send = draft
        # A new question replaces the old one, the way a new quiz replaces
        # the last: without this a remote-push read-back armed a moment
        # earlier would still be sitting in _pending_destructive, and the
        # yes that sends the email would leave IT armed for the next one.
        self._pending_destructive = None
        self._pending_destructive_meta = None
        self._pending_filepick = None

    def _same_room(self, asked: str, answering: str) -> bool:
        """Could a turn from ``answering`` be the answer to a question put
        on ``asked``?

        Voice and typed are the same desk -- he can say it or type it into
        the window the question is on. Everything else is a different room:
        a Discord message, a phone-client turn, a tmux `jarvis "..."` and a
        socket turn never heard the question, and a "yes" from one of them
        is a yes to something else. ``_try_briefing_offer`` already draws
        this line for an offer that is entirely reversible.
        """
        desk = ("voice", "typed")
        if asked in desk:
            return answering in desk
        return answering == asked

    # -- reasoned dissent (jarvis/objections.py) -----------------------
    def _objection_ledger(self):
        """One ledger per process, built lazily: a Commander made with
        __new__ (tests/test_custom_phrases.py) has no __init__ state."""
        ledger = getattr(self, "_objections", None)
        if ledger is None:
            ledger = objections_mod.ObjectionLedger(
                state_path=PATHS.MEMORY_DIR / "objections_state.json")
            self._objections = ledger
        return ledger

    def objection_for_alarm(self, due: float, now: datetime):
        """The reason to advise against this alarm, or None.

        Every gate that keeps dissent from becoming insufferable is here
        rather than in the predicates: the switch (``confirm.dissent``), the
        once-a-day-per-thing ledger, and the rule that an offer already on
        the floor (a destructive read-back, the evening preview's "shall I
        wake you at seven?") is never talked over by a second question."""
        if not _assistant_get(self, "confirm.dissent", True):
            return None
        if self._pending_destructive is not None or self._svc("alarm_offer"):
            return None
        try:
            due_dt = datetime.fromtimestamp(float(due)).astimezone()
        except (TypeError, ValueError, OSError):
            return None
        cal = self._svc("calendar")
        events = []
        if cal is not None:
            try:
                conf = getattr(cal, "configured", True)
                if callable(conf):
                    conf = conf()
                # events() is the CACHE and says so in its own docstring; a
                # fetch here would put the network inside "wake me at two".
                events = list(cal.events()) if conf else []
            except Exception:  # noqa: BLE001 - source boundary
                log.debug("objections: calendar unavailable", exc_info=True)
        deadlines = self._svc("deadlines")
        items = []
        if deadlines is not None:
            try:
                items = list(deadlines.snapshot())   # never canvas.fetch_due
            except Exception:  # noqa: BLE001 - source boundary
                log.debug("objections: deadline snapshot unavailable", exc_info=True)
        tk, pending = self._svc("timekeeper"), []
        if tk is not None:
            try:
                pending = list(tk.list("alarm"))
            except Exception:  # noqa: BLE001 - source boundary
                log.debug("objections: timekeeper list failed", exc_info=True)
        try:
            floor = float(_assistant_get(self, "confirm.sleep_floor_h",
                                         objections_mod.DEFAULT_SLEEP_FLOOR_H))
        except (TypeError, ValueError):
            floor = objections_mod.DEFAULT_SLEEP_FLOOR_H
        obj = objections_mod.for_alarm(due_dt, now.astimezone(), events=events,
                                       items=items, pending=pending,
                                       quiet=self._svc("quiet"),
                                       sleep_floor_h=floor)
        if obj is None:
            return None
        ledger = self._objection_ledger()
        if ledger.already_said(obj):
            # Said once today. A second "wake me at two" is a man who has
            # heard the reason and wants the alarm.
            log.info("objection %s suppressed: already said today (%s)",
                     obj.source, obj.row)
            return None
        return obj

    def stash_objection(self, run: Callable[[], CommandResult], line: str, obj):
        """He advised against it; the pending slot decides what happens next.

        A slot of its own, NOT ``_pending_destructive``: that one drops the
        action on anything but a clear yes, which is right when doing
        nothing is the safe end (a delete) and wrong here -- the user asked
        for this alarm out loud, so an unclear answer must not silently
        leave him without one."""
        self._objection_cancel_timer()
        self._pending_objection = (run, line, obj, time.monotonic())
        self._objection_ledger().record(obj)
        log.info("objection %s: %s (row: %s)", obj.source, obj.reason, obj.row)
        journal = self._svc("journal_objection")
        if callable(journal):
            try:
                journal(obj.source, obj.reason, obj.row)
            except Exception:  # noqa: BLE001 - the journal is best-effort
                log.debug("objection journal failed", exc_info=True)
        # Silence must not lose the alarm either. Without this the pending
        # slot only resolves when the user next says ANYTHING, so a man who
        # heard the objection, agreed with it and went to bed would wake up
        # to no alarm at all -- the failure the objection was warning about.
        timer = threading.Timer(DESTRUCTIVE_TTL_S, self.objection_timeout)
        timer.daemon = True
        self._objection_timer = timer
        timer.start()

    def _objection_cancel_timer(self):
        timer, self._objection_timer = getattr(self, "_objection_timer", None), None
        if timer is not None:
            timer.cancel()

    def _speak_now(self, text: str) -> bool:
        """One line through the app's own TTS door (services.speak ->
        JarvisApp._say), explicitly NOT proactive: this is the direct
        consequence of something the user just asked for, so quiet hours
        must not hold it back for a digest hours later."""
        speak = self._svc("speak")
        if not callable(speak) or not text:
            return False
        try:
            speak(text, proactive=False, kind="message")
            return True
        except Exception:  # noqa: BLE001 - speech must not break the turn
            log.exception("objection ack failed to speak")
            return False

    def _run_objection(self, pend, why: str) -> Optional[CommandResult]:
        run, _line, obj, _stamp = pend
        log.info("objection %s resolved: overruled=True (%s)", obj.source, why)
        try:
            return run()
        except Exception:
            log.exception("overruled objection failed to run")
            return None

    def _try_objection_confirm(self, text: str) -> Optional[CommandResult]:
        """Resolve "Shall I set it anyway?" -- the default is RUN.

        The inverted default is the whole point. ``_try_destructive_confirm``
        drops its pending action on anything that is not a clear yes and on
        expiry, because refusing to delete something is the safe end of a
        delete. Dissent is the other way round: the alarm was ASKED for and
        the objection was only an opinion, so a mumbled reply must not
        silently leave him without one.

        Three outcomes, and only the first two consume the utterance:
          * an explicit no -> drop it;
          * a yes or an override ("set it anyway", "I know") -> run it and
            answer exactly as an unobjected alarm would have;
          * anything else -- a changed subject, a shrug, an expired offer --
            -> run it, SAY so, and let the words he actually said have their
            own turn. Swallowing "play some jazz" as an answer about an
            alarm would be the second mistake after the objection itself.
        """
        pend, self._pending_objection = getattr(self, "_pending_objection", None), None
        if pend is None:
            return None
        self._objection_cancel_timer()
        obj, stamp = pend[2], pend[3]
        answer = parse_yes_no(text)
        if answer is None and objections_mod.is_override(text):
            answer = True
        if time.monotonic() - stamp > DESTRUCTIVE_TTL_S:
            answer = None            # a minute later, those words are not an answer
        if answer is False:
            log.info("objection %s resolved: overruled=False (declined)", obj.source)
            return CommandResult(handled=True, reply=objections_mod.DROPPED_LINE,
                                 speak=True, status="Dropped")
        result = self._run_objection(pend, f"heard {text[:40]!r}")
        if result is None:
            return CommandResult(handled=True, reply="I couldn't manage that, sir.",
                                 speak=True, status="error")
        if answer is True:
            return result
        self._speak_now(f"{objections_mod.RUN_ANYWAY_LINE} {result.reply or ''}".strip())
        bus.publish(Status(text=result.status or "Set anyway", kind="info"))
        return None                  # his sentence keeps its own meaning

    def objection_timeout(self) -> Optional[CommandResult]:
        """Nothing was said at all. Same default: set it, and say so."""
        with self._turn_lock:
            pend, self._pending_objection = getattr(self, "_pending_objection", None), None
            self._objection_timer = None
            if pend is None:
                return None
            result = self._run_objection(pend, "no reply")
        if result is None:
            return None
        self._speak_now(f"{objections_mod.RUN_ANYWAY_LINE} {result.reply or ''}".strip())
        bus.publish(Status(text=result.status or "Set anyway", kind="info"))
        return result

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
        # 1a-1b. The sticky modes belong to the MICROPHONE. Dictation and
        #    lecture notes swallow every later utterance, and they used to do
        #    it whatever the source: with notes open, `jarvis "what's the
        #    weather"` from a tmux shell was filed as a lecture line and a
        #    Discord message became a quiz answer -- and in dictation mode a
        #    CLI turn was typed into whatever window happened to be focused.
        #    A turn that arrives down a socket is a different room, so it
        #    routes normally. Two escapes keep a terminal in control of a
        #    mode it cannot see: the explicit end phrase works from ANY
        #    source (with "status" naming the open modes), and "note: ..."
        #    files a deliberate line while a lecture is open.
        if self.dictation:
            if source == "voice" or _dictation_end(text):
                return self._handle_dictation(text)
        # 1b. Lecture notes open: file it, unless it is "end notes".
        if getattr(self, "lecture_course", None):
            note = None if source == "voice" else _note_prefix(text)
            # getattr(self, "_lecture", None) is None: the flag outlived its
            # file (a failed write). That recovery clears the flag and
            # re-dispatches, so it must run for EVERY source -- otherwise a
            # box that only ever sees CLI turns keeps a ghost mode forever.
            if source == "voice" or note is not None or _lecture_end(text) \
                    or getattr(self, "_lecture", None) is None:
                return self._handle_lecture(text, source, note=note)
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
        # 3a. A working session is holding a question open ("Tuesday at
        #     four?"): this utterance is the answer. Above the quiz because
        #     a session may itself be a quiz-shaped thing, and below the
        #     sticky modes for the same reason the quiz is -- dictation and
        #     lecture notes swallow every utterance, so open_session()
        #     refuses to start one while either is active.
        res = self._try_session(text)
        if res is not None:
            return res
        # 3a'. A quiz question is on the table: this is the answer (or
        #      "skip" / "stop the quiz").
        res = self._try_quiz_answer(text, source)
        if res is not None:
            return res
        # 3a''. "Teach me X" ended with "say quiz me and I'll test you on
        #       it": a plain yes -- or a bare "quiz me", which carries no
        #       topic of its own -- builds that quiz from the chunks the
        #       lesson already read.
        res = self._try_teach_offer(text)
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
        # 3d'. The morning briefing asked "Shall we run ten now, sir?" of
        #      the deck for this week's exam; a plain yes deals those cards.
        res = self._try_study_offer(text)
        if res is not None:
            return res
        # 3d''. The first wake of the day asked "Shall I run your briefing,
        #       sir?"; a plain yes runs THAT, and is not a new command.
        #       Below the study offer because that is the narrower
        #       question; the two can never be live together anyway, since
        #       this rung clears its own offer BEFORE the delivery that
        #       parks a study one. It takes the SOURCE because a spoken
        #       question cannot be answered from Discord or a tmux shell.
        res = self._try_briefing_offer(text, source)
        if res is not None:
            return res
        # 3d'''. A FILE SEND was read back ("lab report.pdf, 2.4
        #       megabytes, to Heather ... Send it, sir?"). Above the
        #       destructive rung because it is the stricter question of
        #       the two -- its yes is an end-anchored grammar of its own,
        #       not parse_yes_no -- and because the two can never be live
        #       together anyway: a send arms its own slot and nothing in
        #       the app arms both.
        res = self._try_send_confirm(text, source)
        if res is not None:
            return res
        # 3d-iv. "Which one, sir?" -- the ambiguous-file question both file
        #        lanes ask. Below the yes/no rungs because its own grammar
        #        is an ORDINAL or a NAME and it must not be handed a bare
        #        "yes" that belongs to one of them; above the destructive
        #        rung because "the second one" is not a yes and that rung
        #        would drop this question's slot without answering it.
        res = self._try_filepick_answer(text, source)
        if res is not None:
            return res
        # 3e. A destructive action was read back ("Cancel all three alarms,
        #     sir?"): a plain yes runs it, anything else drops it.
        res = self._try_destructive_confirm(text, source)
        if res is not None:
            return res
        # 3f. He advised against something ("Shall I set it anyway?"). AFTER
        #     the read-back on purpose: that one drops on ambiguity, this one
        #     runs on ambiguity, and the two can never be live together
        #     (objection_for_alarm refuses while a read-back is pending). It
        #     returns None when the reply was neither a yes nor a no -- the
        #     alarm is set by then and the words keep their own meaning.
        res = self._try_objection_confirm(text)
        if res is not None:
            return res
        # 4. A pending router question: resolve it and dispatch the
        #    remembered utterance (spec 5.2 c).
        res = self._try_router_answer(text)
        if res is not None:
            return res
        # 4a. "How long do you need to get to Wisenbaker, sir?" is on the
        #     table. LAST of the pending stages on purpose: it must not eat
        #     a bare "no" that belongs to the alarm offer or the read-back
        #     above, and it returns None for anything that is not plainly a
        #     duration, so an unrelated command in the window still runs.
        res = self._try_leave_answer(text)
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
            # 3'. The same table again, per clause, for a compound ask.
            #     After the whole-utterance attempt, never before it.
            res = self._try_multi(cmd_text, self._try_registry)
            if res is not None:
                return res

        # 3''. Barge-in, said WITHOUT the address. "quiet" / "hush" / "be
        #      quiet" / "stop talking" (#9, 2026-08-31: "does not
        #      understand"). Every one of these is a one- or two-word
        #      phrase, so with no "jarvis" prefix cmd_text is None, the
        #      registry above never runs, and the intent gate below calls
        #      them background chat and drops them in SILENCE -- measured:
        #      classify("quiet") = ("no", 0.80), and only the three-word
        #      "stop talking" scraped through. Nobody prefixes the word
        #      they are using to interrupt, so this is the one Tier-1
        #      command that can never be addressed.
        #      _QUIET_RX is anchored to the whole utterance, so this fires
        #      only when the words ARE the interruption -- "why are you
        #      being quiet?" (also live, 20:33) still goes to the model.
        #      Placed after the pending yes/no stages and the custom
        #      phrases (both of which own "cancel that" / may shadow a
        #      built-in) and before the gate, exactly like read_control.
        if quiet_kind(text):
            return _h_quiet(self, text, True)

        # 3''a. "Say again", the other half of the same hole, and the same
        #       silence -- his #8, "doesnt understand say again. or any of
        #       these". From the log, twice, thirty seconds apart:
        #         20:36:36 handle 'Say again' source=voice
        #         20:36:36 Ignored (background chat, conf=0.80): 'Say again'
        #         20:37:01 the same two lines, verbatim
        #       Correctly transcribed both times, then thrown away. Nobody
        #       says "jarvis" before asking for a repeat any more than
        #       before an interruption, so cmd_text is None, the prefixed
        #       registry pass (which owns Command("repeat", ...)) is
        #       skipped, and the Tier-1 voice-I/O block that answers it
        #       lives in _route_text, one rung BELOW the gate.
        #       ASSISTANT_TIER1 could not save it either: it is a
        #       name-filtered view of REGISTRY and "repeat" is not in the
        #       list. ("say that again" and "what was that" happened to
        #       survive by matching read_control_kind -- an accident, not a
        #       design, and it does not cover "say again" or "repeat
        #       that".) _REPEAT_RX is anchored like _QUIET_RX, so this
        #       fires only when the words ARE the request.
        if repeat_kind(text):
            return _h_repeat(self, text, True)

        # 3''b. "and the next day": re-ask HIS previous question one day
        #       on. Before the gate, which called this four-word fragment
        #       UNCERTAIN and made him click a card (live, 21:30:50), and
        #       before the router, whose "short" rule handed it to the
        #       model as itself -- which answered with tomorrow's list
        #       again, word for word.
        res = self._try_day_shift(text, source)
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
            # A compound of two Tier-1 commands is addressed to Jarvis for
            # the same reason each half is: without this the classifier
            # calls the pair background chat and drops it in silence.
            if not name and self._multi_match(text):
                name = "multi-intent"
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

    def _try_day_shift(self, text: str,
                       source: str) -> Optional[CommandResult]:
        """"and the next day" -> the last question with the next day in it.

        None unless the utterance is a bare day-shift fragment AND the
        question before it named a day; every other utterance routes
        exactly as it did before.
        """
        prev = self._last_turn
        if prev is None:
            return None
        # A fragment this small is only a follow-up while the exchange is
        # still live -- said cold it is the room talking, and the gate
        # below is the right place for it.
        if time.monotonic() - prev.ts > FOLLOWUP_DAY_WINDOW_S:
            return None
        meant = day_shift_followup(prev.text, strip_address(text))
        if not meant:
            return None
        log.info("day-shift follow-up: %r + %r -> %r", prev.text, text, meant)
        # gate=False: a follow-up to a question Jarvis just answered is
        # addressed to Jarvis, the same reason a correction bypasses it.
        res = self._handle_inner(meant, source, gate=False)
        # The app records the exchange under the resolved sentence, which
        # is also what _last_turn keeps -- so a SECOND "and the next day"
        # has a day to move instead of a fragment with none.
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
        # This rung sits AHEAD of the intent gate (step 3a' in
        # _handle_inner), so an undo is a Tier-1 match by construction: it
        # never reaches the classifier or the model. Live 2026-09-01
        # 21:10:23, "BELAY THAT LAST Uhhh... ORDER": the filler kept
        # undo_kind from matching, the classifier called it Uncertain, and
        # after his "yes" the MODEL said "I'll stand down" and did nothing.
        explicit = undo_explicit(text)
        log.info("undo phrase %r bypasses the intent gate%s", text,
                 " (explicit)" if explicit else "")
        window = UNDO_EXPLICIT_WINDOW_S if explicit else UNDO_WINDOW_S
        pend, self._last_undo = self._last_undo, None
        if pend is None:
            # A calendar add made through the TOOL (the confident path, which
            # never asks) carries no CommandResult, so its undo is parked on
            # the source instead. Same staleness rule; nothing else is looked
            # for here.
            parked = self._calendar_add_undo()
            if parked is not None:
                return parked
            log.info("undo asked for with nothing to undo: %r", text)
            if explicit:
                # "Belay that last order" with nothing on the books is still
                # addressed to Jarvis; it must not fall through to the model.
                return CommandResult(handled=True, speak=True, status="Nothing to undo",
                                     reply="Nothing to take back, sir.")
            return None
        if time.monotonic() - pend[1] > window:
            log.info("undo expired (%.0fs)", time.monotonic() - pend[1])
            if explicit:
                return CommandResult(handled=True, speak=True, status="Nothing to undo",
                                     reply="Nothing to take back, sir.")
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

    def _calendar_add_undo(self) -> Optional[CommandResult]:
        """"Scratch that" after an event was added by the tool.

        The entry is CLEARED whether or not the removal worked: a second
        "scratch that" must not try to delete the same event twice, and the
        honest "I can't take it back" is not made truer by repeating it.
        """
        source = self._svc("calendar")
        entry = getattr(source, "last_add", None) if source is not None else None
        if not isinstance(entry, dict) or not callable(entry.get("undo")):
            return None
        try:
            source.last_add = None
        except Exception:
            log.debug("could not clear the parked calendar undo", exc_info=True)
        try:
            at = float(entry.get("at") or 0.0)
        except (TypeError, ValueError):
            at = 0.0
        if at and time.monotonic() - at > UNDO_WINDOW_S:
            log.info("calendar undo expired (%.0fs)", time.monotonic() - at)
            return None
        try:
            line = entry["undo"]()
        except Exception:
            log.exception("calendar undo failed")
            line = "I couldn't take that back, sir."
        return CommandResult(handled=True, reply=str(line), speak=True,
                             status="Undone")

    def _try_filepick_answer(self, text: str,
                             source: str = "voice") -> Optional[CommandResult]:
        """Answer "Which one, sir?" -- the ambiguous-file question.

        Narrow on purpose (``pick_from_answer``): an ordinal, or a name that
        beats its rivals by more than the band that called them ambiguous.
        An explicit "neither" says so out loud; a NEAR miss -- most often
        the ambiguous phrase repeated -- gets one more question rather than
        a guess; anything else drops the slot and keeps its own meaning.
        The chosen file is then READ BACK like any other, so a wrong pick
        here still cannot send or move anything on its own.
        """
        pend = getattr(self, "_pending_filepick", None)
        if not isinstance(pend, tuple) or len(pend) != 5:
            return None
        cands, resume, asked, made, reasked = pend
        if time.monotonic() - float(made or 0.0) > FILEPICK_TTL_S:
            self._pending_filepick = None
            log.info("which-one: expired; %r is a new subject", text[:40])
            return None
        # A question put at the desk is not answered from Discord or a tmux
        # shell. Left PARKED rather than dropped: that turn is not its
        # answer, but it is not its cancellation either.
        if not self._same_room(asked, source):
            log.debug("which-one: a %s turn is not its answer", source)
            return None
        choice, near = pick_from_answer(text, cands)
        if choice is None:
            said = " ".join(str(text or "").split())
            if _PICK_CANCEL_RX.match(said):
                self._pending_filepick = None
                self._answered_pending = True
                return CommandResult(handled=True, reply="Very good, sir.",
                                     speak=True, status="Dropped")
            # A NEAR MISS -- he named one of them and the margin was not
            # there, which is most often the ambiguous phrase repeated. One
            # more question rather than a guess or a silence; the second
            # near miss spends it, exactly as the send read-back does.
            if near and not reasked:
                self._pending_filepick = (cands, resume, asked, made, True)
                self._answered_pending = True
                log.info("which-one: %r still fits both; asking again",
                         text[:60])
                return CommandResult(
                    handled=True, speak=True, status="Which one?",
                    reply=f"Still either, sir: {filepick.describe(cands)}. "
                          f"Which of them?")
            self._pending_filepick = None
            log.info("which-one: %r names none of them", text[:60])
            return None
        self._pending_filepick = None
        self._answered_pending = True
        log.info("which-one: %r means %s", text[:40], choice.name)
        try:
            return resume(choice)
        except Exception:
            log.exception("which-one: could not carry on with %s", choice.name)
            return CommandResult(handled=True, speak=True, status="error",
                                 reply="I couldn't manage that, sir.")

    def _try_send_confirm(self, text: str,
                          source: str = "voice") -> Optional[CommandResult]:
        """Resolve a file-send read-back ("...Send it, sir?").

        Four answers, and the difference between them is the whole feature:

        * a clear YES sends it, in the background, because an attachment is
          the one tool result in this app whose transfer can take seconds;
        * a clear NO drops it and says so;
        * a VAGUE answer ("okay", "sure", "mhm") is asked once more rather
          than obeyed or ignored. Obeying it is how a file reaches the wrong
          person; ignoring it in silence is how he concludes the feature
          does not work. The second vague answer spends the draft;
        * ANYTHING ELSE -- a new command, a changed subject -- drops the
          draft and keeps its own meaning, exactly as the destructive
          read-back does. Changing the subject is not consent.

        The slot is cleared FIRST, before any of that, so no path through
        this method can leave a live draft behind for a later stray yes.
        """
        draft, self._pending_send = getattr(self, "_pending_send", None), None
        if draft is None:
            return None
        if draft.stale():
            log.info("send read-back expired: %s", draft.path.name)
            return None
        # The question was SPOKEN, at the desk. A Discord message, a phone
        # turn, a `jarvis "..."` from tmux and a socket turn never heard it,
        # so their "yes" is a yes to something else -- and this one sends an
        # attachment. The draft is put BACK: that turn is not its answer,
        # but it is not its cancellation either, and it is still his file
        # waiting on his word at the desk where he was asked.
        if not self._same_room(getattr(draft, "asked_from", "voice"), source):
            self._pending_send = draft
            log.info("send read-back: a %s turn is not its answer", source)
            return None
        self._answered_pending = True
        answer = parse_send_answer(text)
        if answer is None:
            said = " ".join(str(text or "").split())
            # Two ways to be vague, and BOTH get the one re-ask rather than
            # the silent drop. _SEND_MAYBE_RX catches the bare fillers
            # ("okay", "sure"); the second leg catches a yes this grammar
            # does not take but parse_yes_no does -- "um, yes", "okay yes",
            # "I think so yes", "yeah go ahead and send it", and his own
            # logged "system, yes." Six words is parse_yes_no's own
            # overheard-speech line, and it is what keeps the ten-word
            # "Yeah, so you should be able to look that up." on the silent
            # branch where it belongs. Safety is unchanged either way:
            # nothing is sent, he is asked once more.
            vague = bool(_SEND_MAYBE_RX.match(said)) or (
                len(said.split()) <= 6 and parse_yes_no(said) is True)
            if vague and not draft.reasked:
                draft.reasked = True
                self._pending_send = draft
                log.info("send read-back: %r is not a yes; asking again", text)
                return CommandResult(handled=True, reply=outbox.UNSURE_LINE,
                                     speak=True, status="Confirm?")
            self._answered_pending = False
            log.info("send draft dropped, the subject changed: %r", text)
            return None
        if not answer:
            log.info("send declined: %s", draft.path.name)
            return CommandResult(handled=True, reply=outbox.DROPPED_LINE,
                                 speak=True, status="Not sent")

        smtp = self._svc("smtp")
        cap = outbox.max_mb(self._svc("assistant"))
        name = draft.path.name

        def _run():
            try:
                line = outbox.send(draft, smtp=smtp, cap_mb=cap)
                kind, status = "ok", "Sent"
            except outbox.DraftChanged as exc:
                # Its message IS the sentence: the file moved between the
                # read-back and the yes, and he needs to hear which.
                line, kind, status = str(exc), "error", "Not sent"
            except mail_mod.MailSendFailed as exc:
                # A transport failure's text is a class name, never a line.
                log.warning("send failed for %s (%s)", name, exc)
                line, kind, status = mail_mod.SEND_FAILED_LINE, "error", "Send failed"
            except Exception:                          # noqa: BLE001 - source
                log.exception("send blew up for %s", name)
                line, kind, status = mail_mod.SEND_FAILED_LINE, "error", "Send failed"
            bus.publish(Status(text=status, kind=kind))
            # Through the app's own door (services.reply -> _async_reply)
            # when there is one: it shows, speaks, arms the follow-up
            # window AND closes the turn. bus + _speak_now did the first
            # two only, so after "Sent to Heather, sir." the wake word
            # stayed dead for the 60 s watchdog (F20, 09-03). The fallback
            # keeps _speak_now (the not-proactive door) rather than
            # _deliver's talk-back-gated _speak: this line is the direct
            # consequence of a yes he just gave.
            reply = self._svc("reply")
            if callable(reply):
                try:
                    reply(line, speak=True)
                    return
                except Exception:
                    log.exception("services.reply failed")
            bus.publish(JarvisReply(text=line))
            self._speak_now(line)

        self._bg(_run)
        # ack=True: "Sending it now" is an acknowledgement, and the result
        # follows on its own event. done=False keeps the turn open so the
        # UI does not call it finished while the file is still on the wire.
        return CommandResult(handled=True, reply="Sending it now, sir.",
                             speak=True, ack=True, done=False,
                             status=f"Sending {name}")

    def _try_destructive_confirm(self, text: str,
                                 source: str = "voice") -> Optional[CommandResult]:
        """Resolve a read-back ("Cancel all three alarms, sir?").

        As with a calendar add, anything that is not a clear yes or no
        DROPS the offer: changing the subject is not consent, and a stale
        offer would attach the next stray "yes" to an old cancel. An offer
        older than DESTRUCTIVE_TTL_S is dropped even on a yes.

        Two things the plain read-back does not have:

        * a STRICT slot (``stash_destructive(..., strict=True)``, which the
          two HPCOMPUTER transfers use) is answered by ``parse_send_answer``
          instead, with the send lane's one re-ask on a vague word. Its yes
          moves a file onto another machine and there is no undo closure
          waiting on the far side;
        * the answer has to come from the channel the question was PUT on.
          A "yes" typed into Discord or arriving down the phone socket never
          heard the question, and one of these read-backs is irreversible.
        """
        pend, self._pending_destructive = self._pending_destructive, None
        meta, self._pending_destructive_meta = \
            getattr(self, "_pending_destructive_meta", None), None
        # Only trust metadata that names THIS tuple: _pending_destructive is
        # also written by hand (tests, jarvis/app.py reads it), and stale
        # metadata applied to somebody else's slot would be worse than none.
        asked, strict = "", False
        if isinstance(meta, tuple) and len(meta) == 3 and meta[2] is pend:
            asked, strict = str(meta[0] or ""), bool(meta[1])
        if pend is not None and asked and not self._same_room(asked, source):
            self._pending_destructive = pend
            self._pending_destructive_meta = meta
            log.info("read-back: a %s turn is not its answer", source)
            return None
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
        if strict:
            said = " ".join(str(text or "").split())
            answer = parse_send_answer(said)
            if answer is None:
                # The send lane's one re-ask, and for the same reason:
                # obeying "okay" is how a file lands on the wrong machine,
                # and dropping it in silence is how he learns the feature
                # does not work. The second vague answer spends the offer.
                vague = bool(_SEND_MAYBE_RX.match(said)) or (
                    len(said.split()) <= 6 and parse_yes_no(said) is True)
                if vague and not self._strict_reasked:
                    self._strict_reasked = True
                    self._answered_pending = True
                    self._pending_destructive = pend
                    self._pending_destructive_meta = meta
                    log.info("read-back: %r is not a yes; asking again", text)
                    return CommandResult(handled=True, speak=True,
                                         reply=outbox.UNSURE_LINE,
                                         status="Confirm?")
                log.info("read-back dropped, the subject changed: %r", text)
                return None
            self._strict_reasked = False
        else:
            answer = parse_yes_no(text)
            if answer is None:
                return None
        self._answered_pending = True
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

    def _handle_lecture(self, text: str, source: str = "voice",
                        note: Optional[str] = None) -> CommandResult:
        """File one line. ``note`` is the body of an explicit "note: ..."
        from a source the mode does not otherwise listen to -- it is filed
        verbatim, so "note: end notes" writes a line instead of closing."""
        body = note if note is not None else strip_address(text).strip()
        if note is None and _LECTURE_END_RX.match(body.lower()):
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
        # "switch to classic visuals" is the window LOOK, not a window
        # called "classic visuals": the desktop parser owns every "switch
        # to X" and would target a window that does not exist (2026-09-01).
        if _UI_LOOK_RX.match(cmd_text):
            return False

        # Split on "and then", "then", "and", commas for chained commands
        parts = _CHAIN_SPLIT_RX.split(cmd_text)
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
            # The prefixed table has the same hijack in it: "Jarvis, what's
            # on my calendar and what's on my latest email?" matches "last
            # mail" here too. Same guard, same fall-through (_try_multi
            # runs right after this pass, then the router).
            if self._compound_hijack(cmd_text, cmd):
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

    def _try_study_offer(self, text: str) -> Optional[CommandResult]:
        """Resolve "Shall we run ten now, sir?" from the exam-week study
        section (briefing.make_tools parks it on services.study_offer).

        Same rule as _try_alarm_offer: only a clear yes takes it, a no
        declines, anything else DROPS it, and so does an offer older than
        OFFER_TTL_S -- the briefing is spoken at breakfast and a "yes" to
        something else at lunchtime must not start a quiz. The cards come
        from the deck for the exam's course, so it is revision for THAT
        exam and not a general review.
        """
        offer = getattr(self.services, "study_offer", None)
        if not isinstance(offer, dict) or not offer:
            return None
        try:
            self.services.study_offer = None
        except Exception:
            log.debug("could not clear the study offer", exc_info=True)
        try:
            made = float(offer.get("made_at") or 0.0)
        except (TypeError, ValueError):
            made = 0.0
        if made and time.time() - made > OFFER_TTL_S:
            log.info("study offer expired; %r is a new subject", text[:40])
            return None
        answer = parse_yes_no(text)
        if answer is None:
            return None
        if not answer:
            return CommandResult(handled=True, reply="Very good, sir.", speak=True,
                                 status="No study")
        try:
            n = max(1, int(offer.get("n") or quiz_mod.DEFAULT_QUESTIONS))
        except (TypeError, ValueError):
            n = quiz_mod.DEFAULT_QUESTIONS
        return _start_review(self, n, topic=str(offer.get("course") or ""))

    def _try_briefing_offer(self, text: str,
                            source: str = "voice") -> Optional[CommandResult]:
        """Resolve "Shall I run your briefing, sir?" -- the offer that
        replaced 40 seconds of unbidden monologue on 2026-09-02
        (app._offer_first_wake_briefing parks it on services.briefing_offer).

        Same rule as _try_study_offer: a clear yes runs it, a no declines,
        ANYTHING else drops the offer and routes as a new subject, and so
        does an offer older than BRIEFING_OFFER_TTL_S.

        What is NOT the same is what counts as a clear yes or no. The other
        offers ride parse_yes_no; this one has _BRIEFING_YES_RX /
        _BRIEFING_NO_RX, which are end-anchored, because this offer is put
        once a day behind an ARBITRARY first request rather than inside a
        briefing he just heard, and it holds an open microphone. A word bag
        at that exposure swallowed commands ("skip this song", "later today
        remind me to call mom", "no, turn the lights off") and delivered
        the briefing to overheard speech that merely opened with "yeah".

        The delivery is a callback carried in the offer: the commander has
        no handle on the app, and the app is where brain.chat and the
        day's state file live.
        """
        offer = getattr(self.services, "briefing_offer", None)
        if not isinstance(offer, dict) or not offer:
            return None
        # A Discord message or a `jarvis "..."` shell turn is a different
        # room -- app._handle_inner runs this rung for every source -- and
        # it cannot be the answer to a question that was SPOKEN here. It
        # used to consume the offer anyway, and because the day is closed
        # when the question is PUT that spent his only briefing offer of
        # the day. The offer is LEFT parked: it is not this turn's subject
        # either. Same line the debrief draws (app._debrief_reply).
        if source not in ("voice", "typed"):
            log.debug("briefing offer: %s turn is not its answer", source)
            return None
        # "skip it" / "skip that" are declines here and read controls at
        # rung 4b, which is BELOW this one. Rung 4b's own comment says the
        # read control must outrank the chains that "would otherwise eat
        # 'skip' / 'back' / 'pause' / 'go on'" -- so while something is
        # actually being read, those words belong to the reader and this
        # rung stands aside with the offer still parked. Narrow on purpose:
        # a "yes" is not a read control and still answers the question.
        if read_control_kind(text) is not None and \
                getattr(self._svc("reader"), "active", False) is True:
            log.debug("briefing offer: a reading owns %r", text[:40])
            return None
        try:
            self.services.briefing_offer = None
        except Exception:
            log.debug("could not clear the briefing offer", exc_info=True)
        try:
            made = float(offer.get("made_at") or 0.0)
        except (TypeError, ValueError):
            made = 0.0
        if made and time.time() - made > BRIEFING_OFFER_TTL_S:
            log.info("briefing offer expired; %r is a new subject", text[:40])
            return None
        stripped = str(text or "").strip()
        if _BRIEFING_YES_RX.match(stripped):
            answer = True
        elif _BRIEFING_NO_RX.match(stripped):
            answer = False
        else:
            # Not answer-SHAPED, so not an answer: the offer is gone and
            # the words keep their own meaning.
            log.info("briefing offer: %r is a new subject", text[:40])
            return None
        if not answer:
            # The day is already closed (the offer closed it when it was
            # put), so this is the end of it until tomorrow -- he can still
            # say "my briefing" at any hour and get one.
            return CommandResult(handled=True, reply=BRIEFING_DECLINED_LINE,
                                 speak=True, status="Briefing declined")
        deliver = offer.get("deliver")
        if not callable(deliver):
            log.warning("briefing offer had no deliver callback")
            return None
        try:
            ran = bool(deliver())
        except Exception:
            log.exception("briefing offer: delivery failed")
            ran = False
        if not ran:
            # A yes that vanishes is worse than a refusal: the model was
            # busy (or the app threw), so nothing was said and nothing was
            # marked. Tell him, and let him ask again.
            return CommandResult(handled=True, reply=BRIEFING_BUSY_LINE,
                                 speak=True, status="Briefing held")
        # done=False, like _h_briefing: the spoken briefing is the model's
        # reply and lands on a later beat.
        return CommandResult(handled=True, status="Briefing…", done=False)

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
            line, undo = add_event(source.icloud_calendars(), pending["title"],
                                   pending["start"], pending["end"],
                                   calendar_name=pending.get("calendar"))
        except Exception as exc:             # noqa: BLE001 - refusal or server
            log.exception("calendar write failed")
            line = f"I couldn't add that, sir — {type(exc).__name__}."
            return CommandResult(handled=True, reply=line, speak=True,
                                 status="Add failed")
        # "Scratch that" now reaches the event: the closure deletes it, or
        # says plainly that this server gives no way to. Falling through in
        # silence -- what happened before -- reads as success while the
        # event sits in his calendar.
        return CommandResult(handled=True, reply=line, speak=True, status="Added",
                             undo=undo)

    def _try_quiz_answer(self, text: str,
                         source: str = "voice") -> Optional[CommandResult]:
        """While a quiz question is open, the SPOKEN utterance is the answer.

        Only the spoken one: a `jarvis "..."` from a shell or a Discord
        message arrives while the quiz sits on the microphone, and grading
        it as the answer both loses the command and marks a card wrong.
        The stop words are honoured from any source so a terminal can end a
        quiz it can see in "status".

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
        if source != "voice":
            return None            # not the answer: route it as a command
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

    def open_session(self, session) -> bool:
        """Park a working session (jarvis/dialogue.py) so the next
        utterance answers its question. Refused while dictation or lecture
        notes are open: those rungs run FIRST and swallow every utterance,
        so a session opened under them could never be answered.

        Returns False when it was refused; the caller then says something
        else rather than asking a question into a closed door."""
        if self.dictation or getattr(self, "lecture_course", None):
            log.info("session %s refused: a sticky mode owns the mic",
                     getattr(session, "name", "?"))
            return False
        self._pending_session = session
        return True

    def _try_session(self, text: str) -> Optional[CommandResult]:
        """While a working session holds a question open, the utterance is
        its answer.

        The rung, in order: a finished or stale session is dropped and the
        words route normally; "that's enough" closes it WITH its read-back
        (dialogue.enough_kind, checked before quiet_kind -- which matches
        the same words but means barge-in, say nothing); "quiet" / "cancel
        that" drop it silently, cutting speech, exactly as they end a quiz;
        anything else goes to settle(). settle() returning None means the
        session did not recognise the words -- it is dropped and the text
        routes as a new subject, so nothing is ever trapped in a dialogue.

        Results are always done=True, speak=True: app._after_dispatch arms
        the follow-up mic only for a done, spoken reply, and without it the
        next answer would need the wake word.
        """
        session = getattr(self, "_pending_session", None)   # slim test commander
        if session is None:
            return None
        name = getattr(session, "name", "session")
        t = (strip_jarvis_prefix(text) or text).strip()
        try:
            if session.finished or session.stale():
                self._pending_session = None
                return None
        except Exception:                       # noqa: BLE001 - a bad session
            log.exception("session %s state failed", name)
            self._pending_session = None
            return None
        if dialogue_mod.enough_kind(t):
            self._pending_session = None
            return CommandResult(handled=True, reply=self._session_stop(session),
                                 speak=True, status=f"{name} ended")
        if quiet_kind(t) or cancel_kind(t):
            self._pending_session = None
            _cut_speech(self)
            return CommandResult(handled=True, reply="Very good, sir.", speak=False,
                                 status=f"{name} stopped")
        try:
            line = session.settle(t)
        except Exception:                       # noqa: BLE001 - tenant boundary
            log.exception("session %s settle failed", name)
            self._pending_session = None
            return CommandResult(handled=True, reply=SESSION_LOST_LINE, speak=True,
                                 status=f"{name} failed")
        if line is None:
            self._pending_session = None       # not an answer: route the words
            return None
        if session.finished:
            self._pending_session = None
            tail = self._session_stop(session)
            return CommandResult(handled=True, reply=f"{line} {tail}".strip(),
                                 speak=True, status=f"{name} finished")
        try:
            question = session.ask()
        except Exception:                       # noqa: BLE001
            log.exception("session %s ask failed", name)
            self._pending_session = None
            return CommandResult(handled=True, reply=line, speak=True,
                                 status=f"{name} failed")
        status = f"{name} {session.status_text()}".strip()
        return CommandResult(handled=True, reply=f"{line} {question}".strip(),
                             speak=True, status=status)

    @staticmethod
    def _session_stop(session) -> str:
        """The closing line, never an exception: stop() is where a tenant
        writes its results (the planner files its reminders), and a failed
        write must still leave him with something spoken."""
        try:
            return session.stop() or "Very good, sir."
        except Exception:                       # noqa: BLE001
            log.exception("session %s stop failed", getattr(session, "name", "?"))
            return SESSION_LOST_LINE

    def _try_leave_answer(self, text: str) -> Optional[CommandResult]:
        """Answer the once-ever "how long do you need to get to X, sir?".

        Deliberately hard to trigger: only a plainly duration-shaped reply
        counts (leavetime.answer_minutes), so "set a timer for five
        minutes" spoken inside the window is still a timer. Anything else
        leaves the question standing until it expires."""
        pend = getattr(self, "_pending_leave", None)
        if not pend:
            return None
        key, place, ts = pend
        if time.monotonic() - ts > LEAVE_ANSWER_WINDOW_S:
            self._pending_leave = None
            return None
        lt = self._svc("leavetime")
        if lt is None:
            self._pending_leave = None
            return None
        minutes = leave_mod.answer_minutes(text)
        if minutes is None:
            if _LEAVE_DECLINE_RX.match(strip_address(text) or text or ""):
                self._pending_leave = None
                return CommandResult(handled=True, speak=True,
                                     reply=LEAVE_DROPPED_LINE,
                                     status="Walk unknown")
            return None
        self._pending_leave = None
        try:
            value = lt.learn(key, minutes)
        except Exception:  # noqa: BLE001 - a store failure must not eat the turn
            log.exception("leavetime: learn(%r) failed", key)
            return None
        _note_leave_touch(self)     # "make that ten" may follow straight on
        log.info("leavetime: %s answered as %d min", key, value)
        return CommandResult(handled=True, speak=True,
                             reply=leave_mod.LEARNED_LINE.format(
                                 place=place, minutes=value),
                             status=f"Walk: {place} {value} min")

    def _try_teach_offer(self, text: str) -> Optional[CommandResult]:
        """Resolve "say quiz me and I'll test you on it, sir" from _h_teach.

        The whole point of the lesson is that the quiz comes from the SAME
        material, so the stashed study text is handed to make_quiz and no
        second retrieval happens. Same rule as every other offer rung: only
        a clear yes takes it, a no declines it, anything else DROPS it (a
        stray yes an hour later must not start a quiz), and so does an
        offer older than OFFER_TTL_S.
        """
        pending = getattr(self, "_pending_teach", None)   # a slim test commander has none
        if not pending:
            return None
        topic, body, made = pending
        self._pending_teach = None
        if time.monotonic() - float(made) > OFFER_TTL_S:
            log.info("teach offer expired; %r is a new subject", text[:40])
            return None
        t = (strip_jarvis_prefix(text) or text).strip()
        answer = parse_yes_no(t)
        if answer is None:
            # "quiz me" on its own carries no topic (quiz_kind wants an
            # "on ..."), so the registry would never route it; here it is
            # the plainest way to say yes.
            if not _TAKE_QUIZ_RX.match(t.rstrip(".!?")):
                return None
            answer = True
        if not answer:
            return CommandResult(handled=True, reply=TEACH_DECLINED_LINE, speak=True,
                                 status="No quiz")
        brain = self._svc("brain")
        if brain is None or not hasattr(brain, "make_quiz"):
            return CommandResult(handled=True,
                                 reply=quiz_mod.NO_QUESTIONS_LINE.format(topic=topic),
                                 speak=True, status="No model")
        try:
            store = _quiz_store(self)
        except Exception:
            log.exception("flashcard store unavailable")
            return CommandResult(handled=True,
                                 reply=quiz_mod.NO_QUESTIONS_LINE.format(topic=topic),
                                 speak=True, status="No store")
        n = _int_setting(self, "quiz.questions", quiz_mod.DEFAULT_QUESTIONS)
        self._pending_quiz = None

        def _work():
            try:
                pairs = brain.make_quiz(body, n=n, topic=topic)
            except Exception:
                log.exception("teach: make_quiz failed")
                pairs = []
            if not pairs:
                _deliver(self, quiz_mod.NO_QUESTIONS_LINE.format(topic=topic))
                return
            # source is the topic, not a file name: the cards came from the
            # lesson's whole reading, and store.due(topic=) LIKE-matches
            # either column, so "review my biosensors cards" still finds them.
            cards = store.add_cards(pairs, source=topic, topic=topic)
            session = quiz_mod.QuizSession(cards, topic=topic)
            self._pending_quiz = session
            _deliver(self, session.ask())

        self._bg(_work)
        return CommandResult(handled=True,
                             reply=quiz_mod.PREPARING_LINE.format(topic=topic),
                             speak=True, ack=True, done=False,
                             status=f"Quiz: {topic}")

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

    def _try_multi(self, text: str, run_one) -> Optional[CommandResult]:
        """Two Tier-1 commands in one utterance.

        Called ONLY after the whole utterance failed to match, and that
        ordering is the whole safety argument: "remind me to buy milk and
        eggs" matches _REMIND_RX whole, so its body is never split. What
        reaches here already had no meaning as one command.

        Every clause must match a Tier-1 command on its own or nothing runs
        and the compound goes to the model untouched -- half an answer is
        worse than none, and a clause that is really part of one thought
        ("...and eggs") matches nothing, which is what keeps this honest.
        """
        parts = self._multi_match(text)
        if not parts:
            return None
        if self.shaky_transcript():
            # The split is itself a guess about the words. On a transcript
            # that scraped in under confirm.shaky_logprob, running TWO
            # actions off it is the wrong kind of confident. The model gets
            # the compound whole, as it did before this feature.
            log.info("multi-intent: declining a shaky compound %r", text)
            return None
        # Each clause is its own turn for the handlers that re-read the raw
        # utterance (_h_remind_me, _h_lecture_start): left whole, _REMIND_RX
        # would re-match the compound and take the other clause as the body.
        raw = getattr(self, "_raw_text", "")
        results = []
        # _pending_destructive is ONE slot and a compound has two clauses:
        # "clear the shopping list and cancel all the alarms" read both
        # questions aloud but the second stash overwrote the first, so the
        # single "yes" ran only the second clause and the first was dropped
        # in silence. Take each clause's question as it is stashed, then
        # re-stash the pair as one.
        before = self._pending_destructive
        self._pending_destructive = None
        pending = []
        # Handlers that speak for themselves (_h_add_person, _h_who_is,
        # _h_remember) are diverted here for the length of the compound so
        # the pair is spoken ONCE, by the JOIN below. #57.
        held_speech: list = []
        self._suppress_speak = held_speech
        try:
            for part in parts:
                self._raw_text = part
                try:
                    res = run_one(part)
                except Exception:
                    log.exception("multi-intent: clause %r failed", part)
                    res = None
                stash = self._pending_destructive
                if stash is not None:
                    meta = getattr(self, "_pending_destructive_meta", None)
                    strict = bool(isinstance(meta, tuple) and len(meta) == 3
                                  and meta[2] is stash and meta[1])
                    pending.append((stash, strict))
                    self._pending_destructive = None
                    self._pending_destructive_meta = None
                if res is None or not res.handled:
                    log.warning("multi-intent: clause %r matched but did "
                                "nothing", part)
                    continue
                results.append(res)
        finally:
            self._raw_text = raw
            self._suppress_speak = None
        if not results:
            self._pending_destructive = before   # nothing ran: leave the floor as it was
            return None
        log.info("multi-intent: %d of %d clauses ran for %r",
                 len(results), len(parts), text)
        if pending:
            self._stash_multi_confirm(pending)
        replies = [str(r.reply).strip() for r in results if r.reply]
        # A handler that spoke eagerly returned reply without speak=True
        # (it had already said it). Now that its line was diverted, the
        # joined result has to carry it, or the turn goes silent.
        spoke_eagerly = bool(held_speech)
        if not replies and held_speech:
            replies = [t.strip() for t in held_speech if t and t.strip()]
        statuses = [r.status for r in results if r.status]
        # Both clauses' undo closures, chained: handle() only replaces
        # _last_undo when the NEW result carries one, so a compound that
        # dropped them left the PREVIOUS turn's closure armed and "scratch
        # that" took back the wrong thing -- while the two things the
        # compound had just created stayed.
        undos = [r.undo for r in results if getattr(r, "undo", None) is not None]
        # aside.py anchors on the structured thing just created; the last
        # clause is the one "...and make it 9 pm" is about.
        acts = [r.action for r in results if getattr(r, "action", None) is not None]
        # THE JOIN: a compound turn ("set a ten minute timer and add milk to
        # the list") speaks two finished authored lines as one burst, and
        # each of them signs off. Thinned while they are still separate
        # fragments -- jarvis/address.py.
        # ONE acknowledgement for one breath: "Noted, sir: your mom is
        # Heather" + "Noted, sir: your dad is Ali" is a single sentence
        # with two facts, not two announcements. #57 -- and combining is
        # deliberately preferred over dropping the second fact.
        merged = _merge_acks(replies)
        return CommandResult(
            handled=True,
            reply=merged or address.join_fragments(replies) or None,
            speak=any(r.speak for r in results) or spoke_eagerly,
            status=" + ".join(statuses) or None,
            # one follow-up window for the pair: it opens once the last
            # clause is done
            done=all(r.done for r in results),
            ack=any(r.ack for r in results),
            undo=self._chain_undos(undos) if undos else None,
            action=acts[-1] if acts else None)

    @staticmethod
    def _chain_undos(undos: list):
        """One closure that takes back a whole compound turn."""
        def _undo_all() -> str:
            # Reverse order: unwind the way a person would, last thing first.
            lines = []
            for fn in reversed(undos):
                try:
                    line = fn()
                except Exception:
                    log.exception("multi-intent: clause undo failed")
                    continue
                if line:
                    lines.append(str(line))
            return address.join_fragments(lines)
        return _undo_all

    def _stash_multi_confirm(self, pending: list) -> None:
        """Re-arm the clauses' read-backs as a single pending question.

        One question, one yes, every clause run: _try_destructive_confirm
        pops one slot, so anything left unchained is silently lost.
        """
        # Entries are (the 3-tuple that was stashed, was it strict).
        stashes = [p[0] for p in pending]
        strict = any(p[1] for p in pending)
        runs = [p[0] for p in stashes]
        lines = [str(p[1]).strip() for p in stashes if p[1]]

        def _run_all() -> CommandResult:
            replies, statuses = [], []
            for fn in runs:
                try:
                    res = fn()
                except Exception:
                    log.exception("multi-intent: confirmed clause failed")
                    continue
                if res is None:
                    continue
                if getattr(res, "reply", None):
                    replies.append(str(res.reply).strip())
                if getattr(res, "status", None):
                    statuses.append(str(res.status))
            return CommandResult(handled=True,
                                 reply=address.join_fragments(replies) or None,
                                 speak=True,
                                 status=" + ".join(statuses) or "Done")
        # One STRICT clause makes the joined question strict: the yes that
        # answers it would run that clause too, and it is the irreversible
        # one in the pair.
        self.stash_destructive(_run_all, address.join_fragments(lines),
                               strict=strict)

    def _multi_match(self, text: str) -> list:
        """The clauses of a compound whose EVERY half is a Tier-1 command
        on its own, else []. A probe: no handler runs."""
        parts = split_clauses(text)
        if parts and all(self._match_assistant(p) for p in parts):
            return parts
        return []

    def _compound_hijack(self, text: str, cmd: Command) -> bool:
        """True when ``cmd`` matched a compound utterance but means only
        one half of it, and the other half is a request in its own right.

        LIVE 2026-08-31 14:35, by voice: "What's on my calendar and what's
        on my latest email?" -- one breath, two questions. "last mail" is a
        `search`, so it matched the TAIL ("latest email"); _try_assistant
        claimed the WHOLE utterance and forced get_mail on it, and the
        calendar half was dropped without a word (the log reads "tier-1
        match 'last mail' bypasses the intent gate").

        The whole-utterance pass running BEFORE split_clauses is right and
        stays -- "remind me to buy milk and eggs" must never be split -- so
        this is a precondition on that pass, not a new ordering: a Tier-1
        shortcut claims the turn only when the request it recognises IS the
        utterance. The same rule forced_call already applies to the router
        short-cut ("a second clause means a second intent: the full loop
        handles both"), stated narrowly enough that the bare commands keep
        their fast path -- a naive "contains 'and'" test would cost "set a
        timer for one and a half minutes" its shortcut.

        Two shapes, and nothing else:

        1. Both halves are Tier-1 commands. _try_multi runs BOTH, which is
           exactly what it was built for; it just never got the chance,
           because the whole-utterance pass answered first. Deferred only
           when _try_multi would actually take it (a shaky transcript
           declines the split, and then the whole match is still the best
           reading there is).
        2. The match sits in the TRAILING clause and the utterance OPENS
           with a request the router names a tool for. An anchored matcher
           cannot do this; an unanchored `search` can, and skipping over a
           whole question to answer the second one is the 14:35 bug.
           This direction only: a command that LEADS the utterance keeps
           the turn, because its handler is usually the fuller answer
           ("good morning and what's on my calendar" is the briefing,
           calendar included) and handing it to the model would lose it.
        """
        # "Jarvis, X and Y" reaches _try_assistant with the address still
        # on it (only the registry pass gets a stripped copy), and a comma
        # after the name is a clause separator: left in, the sentence
        # splits three ways and split_clauses refuses it -- which read as
        # "not a compound" and let the hijack straight through.
        bare = strip_jarvis_prefix(text)
        if bare is None:
            bare = strip_address(text)
        parts = split_clauses(bare)
        if len(parts) != MAX_CLAUSES:
            return False
        hits = []
        for part in parts:
            try:
                hits.append(bool(cmd.matcher(part.strip().lower().rstrip(".!?"))))
            except Exception:
                log.exception("matcher %s failed on clause %r", cmd.name, part)
                return False
        if sum(hits) == 2:
            # BOTH halves are this command. "email the notes to Heather and
            # email the handout to Heather" is two requests, and bailing
            # here let the whole-utterance match stand: _SEND_FILE_RX's
            # `to`-split is non-greedy but $-anchored, so who_a swallowed
            # the second clause and Jarvis asked for the address of a
            # person called "Heather and email the biosensors handout to
            # Heather". _try_multi is what this sentence wanted all along.
            if self._multi_match(bare) and not self.shaky_transcript():
                log.info("tier-1 %r declines the compound %r: both halves "
                         "are the same command", cmd.name, text)
                return True
            return False
        if sum(hits) != 1:
            # Neither half on its own ("add milk and bread to my shopping
            # list" matches only whole), so the match is not one clause of
            # a two-request sentence.
            return False
        first = hits.index(True) == 0
        other = parts[1] if first else parts[0]
        opener = "" if first else local_tool_clause(other)
        if self._multi_match(bare) and not self.shaky_transcript():
            why = "both halves are Tier-1 commands"
        elif opener:
            why = f"it opens with a {opener} request"
        else:
            return False
        log.info("tier-1 %r declines the compound %r: %s", cmd.name, text, why)
        return True

    def _match_assistant(self, text: str) -> Optional[str]:
        """The name of the ASSISTANT_TIER1 command whose matcher accepts
        the bare utterance, else None. A probe only -- no handler runs, no
        service is consulted -- used to spare exact command matches the
        intent classifier's guess.

        Deliberately NOT guarded by _compound_hijack: its other caller is
        the intent gate, where the question is only "was this addressed to
        Jarvis", and a compound of two requests plainly was. Making the
        probe stricter there would hand the 14:35 utterance to the
        classifier, which calls long sentences background chat and drops
        them in silence -- trading a half answer for none."""
        t = text.strip().lower().rstrip(".!?")
        for cmd in ASSISTANT_TIER1:
            try:
                if not cmd.matcher(t):
                    continue
            except Exception:
                log.exception("matcher %s failed", cmd.name)
                continue
            # The one family where the gate must not be switched off on the
            # SHAPE of a sentence alone. _tier1_send_file has already run
            # the negative table; this is the other half of the same claim
            # rule -- a recipient he can write to, or a file really on his
            # disk -- and it needs services a matcher does not get. Without
            # it "send the deposit to the landlord" is addressed to Jarvis.
            if cmd.name == "send file":
                try:
                    claimed = _send_file_pieces(self, t) is not None
                except Exception:  # noqa: BLE001 - a slim test commander
                    log.debug("send-file: the claim rule needs services I "
                              "haven't got", exc_info=True)
                    claimed = True
                if not claimed:
                    continue
            return cmd.name
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
            # A shortcut only claims the turn when it IS the request: the
            # compound falls through to _try_multi and then to the full
            # tool loop, which answers both halves (live 14:35).
            if self._compound_hijack(t, cmd):
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
            # is a study session (jarvis/focus.py), not "focus <window>";
            # "focus on the sessions" names a Board panel, which is the
            # third thing this chain's bare "focus" would otherwise eat;
            # and "switch to classic visuals" is the window's own look
            # (Tier 1 "ui look", further down this same path).
            if match and not _PROJECT_SWITCH_RX.match(tl) \
                    and not _m_focus_start(tl) and not _board_panel_name(tl) \
                    and not _UI_LOOK_RX.match(tl):
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
            # Ahead of the clock, for the same reason the registry entry
            # is: clock_kind claims "what time is my class" and this is
            # the rung it would be claimed on when he says it bare.
            cm = _NEXT_CLASS_RX.match(text.strip())
            if cm:
                res = _h_next_class(self, text, cm)
                if res is not None:
                    return res
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
            # ... then the same rung per clause for a compound ask, which
            # the model would otherwise answer half of.
            res = self._try_multi(text, self._try_assistant)
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
