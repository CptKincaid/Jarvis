"""Persistent Memory — facts, habits, preferences, sessions, notes.

THE single memory store for Jarvis V3. Survives across sessions. All data
lives in ``PATHS.MEMORY_DIR`` (~/.aiws_trainer/jarvis_memory/).

Absorbs the legacy jarvis_agent.py stores: on first run, anything in
``PATHS.LEGACY_AGENT_DIR`` (~/.aiws_trainer/jarvis_data/) is migrated in —
habit entries are merged (so habit counts merge), voice notes are copied —
and the old dir is renamed ``jarvis_data.migrated``.

``log_habit`` is the ONE habit sink (the V1 double-log — agent.log_command +
memory.log_habit — is gone; all callers route here exactly once per
utterance).

Usage:
    mem = JarvisMemory()
    mem.remember("project", "training uses batch size 16")
    results = mem.recall("training")
    mem.log_habit("check gpu")
    suggestion = mem.suggest_by_habit()
    mem.save_session("worked on the detector")   # app shutdown
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("memory")


class JarvisMemory:
    """Persistent memory across Jarvis sessions (single store)."""

    MAX_HABITS = 500
    MAX_SESSIONS = 20
    MAX_INTENTS = 500

    def __init__(self, memory_dir: Path | str | None = None,
                 legacy_dir: Path | str | None = None):
        self._dir = Path(memory_dir) if memory_dir else PATHS.MEMORY_DIR
        self._legacy_dir = (Path(legacy_dir) if legacy_dir
                            else PATHS.LEGACY_AGENT_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._facts = self._load("facts.json", {})
        self._habits = self._load("habits.json", [])
        self._preferences = self._load("preferences.json", {})
        self._sessions = self._load("sessions.json", [])
        self._intent_log = self._load("intent_log.json", [])
        self._migrate_legacy()

    # ------------------------------------------------------------------
    # File I/O (atomic writes)
    # ------------------------------------------------------------------
    def _load(self, filename, default):
        path = self._dir / filename
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                log.exception("failed to load %s; using default", filename)
        return default

    def _save(self, filename, data):
        path = self._dir / filename
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, path)
        except Exception:
            log.exception("save failed (%s)", filename)

    # ------------------------------------------------------------------
    # One-time migration of the legacy jarvis_agent store
    # ------------------------------------------------------------------
    def _migrate_legacy(self):
        """Merge ~/.aiws_trainer/jarvis_data into this store, once."""
        legacy = self._legacy_dir
        if not legacy.is_dir():
            return
        log.info("migrating legacy agent data from %s", legacy)
        try:
            # Merge habit entries (this merges the per-command counts too).
            legacy_habits_file = legacy / "habits.json"
            if legacy_habits_file.exists():
                try:
                    legacy_habits = json.loads(legacy_habits_file.read_text())
                except Exception:
                    log.exception("legacy habits.json unreadable; skipping")
                    legacy_habits = []
                if isinstance(legacy_habits, list) and legacy_habits:
                    merged = legacy_habits + self._habits
                    try:
                        merged.sort(key=lambda h: str(h.get("time", "")))
                    except Exception:
                        log.exception("habit merge sort failed; keeping order")
                    self._habits = merged[-self.MAX_HABITS:]
                    self._save("habits.json", self._habits)
                    log.info("merged %d legacy habit entries",
                             len(legacy_habits))

            # Copy voice notes into the single notes dir.
            notes_src = legacy / "voice_notes"
            if notes_src.is_dir():
                notes_dst = self._dir / "notes"
                notes_dst.mkdir(exist_ok=True)
                copied = 0
                for f in sorted(notes_src.glob("note_*.txt")):
                    dst = notes_dst / f.name
                    if not dst.exists():
                        shutil.copy2(f, dst)
                        copied += 1
                log.info("copied %d legacy voice notes", copied)

            # Rename the old dir so migration never runs twice.
            migrated = legacy.with_name(legacy.name + ".migrated")
            if migrated.exists():
                log.warning("%s already exists; leaving legacy dir in place",
                            migrated)
            else:
                legacy.rename(migrated)
                log.info("legacy dir renamed to %s", migrated.name)
        except Exception:
            log.exception("legacy migration failed")

    # ------------------------------------------------------------------
    # Facts — key/value store with timestamps
    # ------------------------------------------------------------------
    def remember(self, key, value):
        """Store a fact persistently. Overwrites if key exists."""
        self._facts[key] = {
            "value": value,
            "time": datetime.now().isoformat(),
        }
        self._save("facts.json", self._facts)
        log.info("remembered: %s = %s", key, str(value)[:50])

    def recall(self, query):
        """Search facts by key or value substring."""
        q = query.lower()
        matches = []
        for key, entry in self._facts.items():
            if q in key.lower() or q in str(entry["value"]).lower():
                matches.append({"key": key, **entry})
        return matches[:5]

    def forget(self, key):
        """Remove a fact."""
        if key in self._facts:
            del self._facts[key]
            self._save("facts.json", self._facts)

    def get_all_facts(self):
        """Return all stored facts."""
        return self._facts

    # ------------------------------------------------------------------
    # Habits — command logging with time patterns (the ONE habit log)
    # ------------------------------------------------------------------
    def log_habit(self, command, context=None):
        """Log a command execution for pattern learning."""
        self._habits.append({
            "time": datetime.now().isoformat(),
            "hour": datetime.now().hour,
            "day": datetime.now().strftime("%A"),
            "command": command[:100],
            "context": context,
        })
        self._habits = self._habits[-self.MAX_HABITS:]
        self._save("habits.json", self._habits)

    def suggest_by_habit(self):
        """Suggest a command based on current time patterns."""
        if len(self._habits) < 10:
            return None
        hour = datetime.now().hour
        counts = {}
        for h in self._habits:
            if abs(h.get("hour", -1) - hour) <= 1:
                cmd = h["command"]
                counts[cmd] = counts.get(cmd, 0) + 1
        if counts:
            best = max(counts, key=counts.get)
            if counts[best] >= 3:
                return best
        return None

    def get_habit_summary(self):
        """Summary of habit patterns for context injection."""
        if not self._habits:
            return "No habits recorded yet."
        total = len(self._habits)
        counts = {}
        for h in self._habits:
            cmd = h["command"]
            counts[cmd] = counts.get(cmd, 0) + 1
        top = sorted(counts.items(), key=lambda x: -x[1])[:5]
        lines = [f"Total commands logged: {total}"]
        for cmd, count in top:
            lines.append(f"  {cmd}: {count}x")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Preferences — user settings/preferences
    # ------------------------------------------------------------------
    def set_preference(self, key, value):
        self._preferences[key] = value
        self._save("preferences.json", self._preferences)

    def get_preference(self, key, default=None):
        return self._preferences.get(key, default)

    def get_all_preferences(self):
        return self._preferences

    # ------------------------------------------------------------------
    # Session summaries — compressed conversation history
    # ------------------------------------------------------------------
    def save_session(self, summary):
        """Save a session summary for cross-session memory.

        Called from app shutdown so the next session remembers this one.
        """
        if not summary:
            return
        self._sessions.append({
            "time": datetime.now().isoformat(),
            "summary": str(summary)[:500],
        })
        self._sessions = self._sessions[-self.MAX_SESSIONS:]
        self._save("sessions.json", self._sessions)
        log.info("session saved: %s", str(summary)[:50])

    def get_recent_sessions(self, n=3):
        """Get last N session summaries."""
        return self._sessions[-n:]

    def format_sessions_for_prompt(self):
        """Format session summaries for context injection."""
        sessions = self.get_recent_sessions()
        if not sessions:
            return ""
        parts = ["Previous sessions:"]
        for s in sessions:
            parts.append(f"  [{s['time'][:16]}] {s['summary'][:100]}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Intent learning — "Was this for me?" feedback
    # ------------------------------------------------------------------
    def log_intent(self, text, is_for_assistant):
        """Record user feedback on intent classification."""
        self._intent_log.append({
            "text": text[:200],
            "label": "yes" if is_for_assistant else "no",
        })
        self._intent_log = self._intent_log[-self.MAX_INTENTS:]
        self._save("intent_log.json", self._intent_log)

    def get_intent_log(self):
        return self._intent_log

    # ------------------------------------------------------------------
    # Voice notes — timestamped text memos
    # ------------------------------------------------------------------
    def save_note(self, text):
        """Save a timestamped voice note."""
        notes_dir = self._dir / "notes"
        notes_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = notes_dir / f"note_{ts}.txt"
        path.write_text(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]\n{text}\n")
        log.info("note saved: %s", path.name)
        return str(path)

    def get_notes(self, n=5):
        """List recent voice notes."""
        notes_dir = self._dir / "notes"
        if not notes_dir.exists():
            return []
        files = sorted(notes_dir.glob("note_*.txt"), reverse=True)[:n]
        return [{"file": f.name, "content": f.read_text().strip()[:100]}
                for f in files]

    # ------------------------------------------------------------------
    # Full memory dump for context
    # ------------------------------------------------------------------
    def format_for_context(self):
        """Format all memory for LLM context injection."""
        parts = []

        if self._facts:
            parts.append(f"Known facts ({len(self._facts)}):")
            for key, entry in list(self._facts.items())[-5:]:
                parts.append(f"  {key}: {entry['value']}")

        suggestion = self.suggest_by_habit()
        if suggestion:
            parts.append(
                f"Habit suggestion: user often runs '{suggestion}' at this time")

        sessions_text = self.format_sessions_for_prompt()
        if sessions_text:
            parts.append(sessions_text)

        return "\n".join(parts) if parts else ""
