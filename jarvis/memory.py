"""Persistent Memory — facts, habits, preferences, sessions, notes.

Survives across sessions. All data stored in ~/.aiws_trainer/jarvis_memory/.

Usage:
    mem = JarvisMemory()
    mem.remember("project", "training uses batch size 16")
    results = mem.recall("training")
    mem.log_habit("check gpu")
    suggestion = mem.suggest_by_habit()
"""

import json
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path.home() / ".aiws_trainer" / "jarvis_memory"


from jarvis.jarvis_logging import get_logger
_log = get_logger("MEM")


class JarvisMemory:
    """Persistent memory across Jarvis sessions."""

    def __init__(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self._facts = self._load("facts.json", {})
        self._habits = self._load("habits.json", [])
        self._preferences = self._load("preferences.json", {})
        self._sessions = self._load("sessions.json", [])
        self._intent_log = self._load("intent_log.json", [])

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def _load(self, filename, default):
        path = MEMORY_DIR / filename
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return default

    def _save(self, filename, data):
        try:
            target = MEMORY_DIR / filename
            # Atomic write — crash mid-write cannot corrupt persistent state.
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            import os
            os.replace(tmp, target)
        except Exception as e:
            _log(f"Save error ({filename}): {e}")

    # ------------------------------------------------------------------
    # Facts — key/value store with timestamps
    # ------------------------------------------------------------------
    def remember(self, key, value):
        """Store a fact. Overwrites if key exists."""
        self._facts[key] = {
            "value": value,
            "time": datetime.now().isoformat(),
        }
        self._save("facts.json", self._facts)
        _log(f"Remembered: {key} = {str(value)[:50]}")

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
    # Habits — command logging with time patterns
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
        self._habits = self._habits[-500:]
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
        # Most common commands
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
        """Save a session summary for cross-session memory."""
        self._sessions.append({
            "time": datetime.now().isoformat(),
            "summary": summary[:500],
        })
        self._sessions = self._sessions[-20:]
        self._save("sessions.json", self._sessions)
        _log(f"Session saved: {summary[:50]}")

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
        self._intent_log = self._intent_log[-500:]
        self._save("intent_log.json", self._intent_log)

    def get_intent_log(self):
        return self._intent_log

    # ------------------------------------------------------------------
    # Voice notes — timestamped text memos
    # ------------------------------------------------------------------
    def save_note(self, text):
        """Save a timestamped voice note."""
        notes_dir = MEMORY_DIR / "notes"
        notes_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = notes_dir / f"note_{ts}.txt"
        path.write_text(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]\n{text}\n")
        _log(f"Note saved: {path.name}")
        return str(path)

    def get_notes(self, n=5):
        """List recent voice notes."""
        notes_dir = MEMORY_DIR / "notes"
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

        # Facts
        if self._facts:
            parts.append(f"Known facts ({len(self._facts)}):")
            for key, entry in list(self._facts.items())[-5:]:
                parts.append(f"  {key}: {entry['value']}")

        # Habits
        suggestion = self.suggest_by_habit()
        if suggestion:
            parts.append(f"Habit suggestion: user often runs '{suggestion}' at this time")

        # Sessions
        sessions_text = self.format_sessions_for_prompt()
        if sessions_text:
            parts.append(sessions_text)

        return "\n".join(parts) if parts else ""
