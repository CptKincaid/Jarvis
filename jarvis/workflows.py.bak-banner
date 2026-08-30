"""Workflows + persisted reminders for Jarvis V3.

Ported from (READ-ONLY legacy sources):
  jarvis_agent.py  DEFAULT_WORKFLOWS 257-276, define/get_workflow 244-255
  voice_input_gui.py  _run_workflow 3767-3798  -> Workflows.run
  voice_input_gui.py  _set_reminder 3514-3546  -> Reminders.set / _fire

Differences from V1 (by spec): no widget calls — progress and completion go
out as Status events and the runner RETURNS its text result for the caller's
CommandResult. Reminders persist to PATHS.REMINDERS, are restored on boot,
are cancellable, and fire a ReminderFired event plus the injected tts
callback. Hardcoded /home/hunterp/vss_env becomes PATHS.VSS_ENV.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid

from jarvis.config import CONFIG, PATHS
from jarvis.events import ReminderFired, Status, bus
from jarvis.logs import get_logger

log = get_logger("workflows")

# Custom (user-defined) workflows live with the rest of jarvis memory now;
# the legacy jarvis_agent location is read as a fallback until memory.py's
# migration renames it.
WORKFLOWS_FILE = PATHS.MEMORY_DIR / "workflows.json"
LEGACY_WORKFLOWS_FILE = PATHS.LEGACY_AGENT_DIR / "workflows.json"

# Default workflows — port of jarvis_agent.DEFAULT_WORKFLOWS 258-276 with
# PATHS.VSS_ENV substituted for the hardcoded /home/hunterp/vss_env.
DEFAULT_WORKFLOWS = {
    "deploy": [
        ("shell", f"cd {PATHS.VSS_ENV} && python scripts/agents/test_agent.py"),
        ("shell", f"cd {PATHS.VSS_ENV} && python scripts/agents/security_agent.py"),
        ("shell", f"cd {PATHS.VSS_ENV} && git add -A && git status -s"),
        ("speak", "Tests and security scan complete. Ready to commit."),
    ],
    "morning": [
        ("shell", "nvidia-smi --query-gpu=name,memory.used,utilization.gpu"
                  " --format=csv"),
        ("shell", "df -h / /storage 2>/dev/null"),
        ("shell", "uptime"),
        ("speak", "Good morning sir. Systems are online. GPU and storage"
                  " look healthy."),
    ],
    "training check": [
        ("shell", "nvidia-smi --query-gpu=utilization.gpu,memory.used,"
                  "temperature.gpu --format=csv"),
        ("shell", f"ls -lt {PATHS.VSS_ENV}/aiws_system/training_data/*/images/"
                  " 2>/dev/null | head -5"),
        ("speak", "Training status retrieved."),
    ],
}


class Workflows:
    """Named multi-step workflows: shell / speak / key / wait steps.

    `say` is the injected tts callback (e.g. tts.speak); speak steps and the
    talkback gate go through it. run() blocks — call from a worker thread.
    """

    def __init__(self, say=None):
        self._say = say or (lambda text: None)
        self._custom = self._load_custom()

    # -------------------------------------------------- definitions
    def _load_custom(self):
        for path in (WORKFLOWS_FILE, LEGACY_WORKFLOWS_FILE):
            try:
                if path.exists():
                    data = json.loads(path.read_text())
                    # JSON round-trip turns step tuples into lists; normalize
                    return {name: [tuple(s) for s in steps]
                            for name, steps in data.items()}
            except Exception:
                log.exception("failed loading workflows from %s", path)
        return {}

    def define(self, name, steps):
        """Define (and persist) a reusable workflow. Port of 244-251."""
        self._custom[name] = [tuple(s) for s in steps]
        try:
            WORKFLOWS_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = WORKFLOWS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {n: [list(s) for s in steps_]
                 for n, steps_ in self._custom.items()}, indent=2))
            os.replace(tmp, WORKFLOWS_FILE)
        except Exception:
            log.exception("failed saving workflows")
        log.info("Workflow defined: %s (%d steps)", name, len(steps))

    def get(self, name):
        """Steps for a workflow name (custom overrides defaults), or None."""
        return self._custom.get(name) or DEFAULT_WORKFLOWS.get(name)

    def names(self):
        return sorted(set(DEFAULT_WORKFLOWS) | set(self._custom))

    # ------------------------------------------------------- runner
    def run(self, name):
        """Execute a workflow by name. Returns its text result ("[name]\\n…")
        or None if no such workflow. Port of _run_workflow 3767-3798 with
        Status events instead of widget calls."""
        steps = self.get(name)
        if not steps:
            return None
        log.info("Workflow: %s (%d steps)", name, len(steps))
        bus.publish(Status(f"Workflow '{name}' running...", "busy"))

        results = []
        for step_type, step_data in steps:
            if step_type == "shell":
                try:
                    r = subprocess.run(
                        step_data, shell=True, capture_output=True,
                        text=True, timeout=60,
                    )
                    results.append(r.stdout.strip()[:200])
                    log.info("Workflow step: %s → OK", step_data[:40])
                except Exception as e:
                    log.exception("workflow shell step failed")
                    results.append(f"Error: {e}")
            elif step_type == "speak":
                if CONFIG.talkback:
                    self._say(step_data)
            elif step_type == "key":
                try:
                    subprocess.run(
                        ["xdotool", "key", "--clearmodifiers", step_data],
                        capture_output=True, timeout=2,
                    )
                except Exception:
                    log.exception("workflow key step failed")
            elif step_type == "wait":
                time.sleep(int(step_data))
            time.sleep(0.3)

        output = "\n".join(r for r in results if r)
        bus.publish(Status(f"Workflow '{name}' complete", "ok"))
        return f"[{name}]\n{output[:500]}"


class Reminders:
    """Timed reminders: persisted to PATHS.REMINDERS, restored on boot,
    cancellable. Firing publishes ReminderFired + Status, sends a desktop
    notification, and speaks through the injected tts callback (talkback-
    gated). Evolves _set_reminder 3514-3546 (V1 lost reminders on restart)."""

    def __init__(self, say=None, restore=True):
        self._say = say or (lambda text: None)
        self._timers = {}          # id -> threading.Timer
        self._items = {}           # id -> {"task": str, "due": float}
        self._lock = threading.Lock()
        if restore:
            self.restore()

    # -------------------------------------------------- persistence
    def _load(self):
        try:
            if PATHS.REMINDERS.exists():
                data = json.loads(PATHS.REMINDERS.read_text())
                return {r["id"]: {"task": r["task"], "due": float(r["due"])}
                        for r in data.get("reminders", [])}
        except Exception:
            log.exception("failed loading reminders")
        return {}

    def _save(self):
        try:
            PATHS.REMINDERS.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {"reminders": [
                    {"id": rid, "task": it["task"], "due": it["due"]}
                    for rid, it in self._items.items()]}
            tmp = PATHS.REMINDERS.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, PATHS.REMINDERS)
        except Exception:
            log.exception("failed saving reminders")

    # ------------------------------------------------------ public
    def restore(self):
        """Re-arm persisted reminders on boot. Overdue ones fire shortly
        after startup instead of being lost. Returns count restored."""
        items = self._load()
        count = 0
        now = time.time()
        for rid, item in items.items():
            delay = max(0.5, item["due"] - now)
            with self._lock:
                self._items[rid] = item
                timer = threading.Timer(delay, self._fire, args=(rid,))
                timer.daemon = True
                self._timers[rid] = timer
            timer.start()
            count += 1
            if item["due"] < now:
                log.info("Reminder overdue on restore, firing: %r",
                         item["task"])
        if count:
            log.info("Restored %d reminder(s)", count)
        return count

    def set(self, seconds, task):
        """Set a timed reminder. Port of _set_reminder 3514-3524 (phrasing
        verbatim); persistence and cancellability are new. Returns
        confirmation text for the caller's reply."""
        log.info("Reminder set: %r in %ss", task, seconds)
        mins = int(seconds) // 60
        bus.publish(Status(f"Reminder set: {mins}m — {task[:30]}", "ok"))

        if CONFIG.talkback:
            self._say(f"Reminder set for {mins} minutes."
                      f" I will remind you to {task}.")

        rid = uuid.uuid4().hex[:12]
        due = time.time() + seconds
        timer = threading.Timer(seconds, self._fire, args=(rid,))
        timer.daemon = True
        with self._lock:
            self._items[rid] = {"task": task, "due": due}
            self._timers[rid] = timer
        self._save()
        timer.start()
        return f"Reminder set for {mins} minutes: {task}"

    def cancel(self, rid):
        """Cancel one reminder by id. Returns True if it existed."""
        with self._lock:
            timer = self._timers.pop(rid, None)
            existed = self._items.pop(rid, None) is not None
        if timer:
            timer.cancel()
        if existed:
            self._save()
            log.info("Reminder cancelled: %s", rid)
        return existed

    def cancel_all(self):
        """Cancel every pending reminder. Returns count cancelled."""
        with self._lock:
            timers = list(self._timers.values())
            count = len(self._items)
            self._timers.clear()
            self._items.clear()
        for t in timers:
            t.cancel()
        if count:
            self._save()
            log.info("Cancelled %d reminder(s)", count)
        return count

    def list(self):
        """Pending reminders as [(id, task, due_epoch)], soonest first."""
        with self._lock:
            items = [(rid, it["task"], it["due"])
                     for rid, it in self._items.items()]
        return sorted(items, key=lambda r: r[2])

    # ------------------------------------------------------ firing
    def _fire(self, rid):
        """Port of the _remind closure 3525-3544: notify-send + speak, plus
        ReminderFired on the bus instead of widget calls."""
        with self._lock:
            item = self._items.pop(rid, None)
            self._timers.pop(rid, None)
        if item is None:
            return                          # cancelled between timer and fire
        task = item["task"]
        log.info("Reminder fired: %r", task)
        self._save()
        # Desktop notification
        try:
            subprocess.Popen(
                ["notify-send", "-t", "10000", "-i", "appointment-soon",
                 "Jarvis Reminder", task],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            log.warning("notify-send unavailable for reminder %r", task)
        # Speak it
        if CONFIG.talkback:
            try:
                self._say(f"Sir, this is your reminder. {task}")
            except Exception:
                log.exception("reminder tts failed")
        bus.publish(ReminderFired(task))
        bus.publish(Status(f"Reminder: {task[:50]}", "warn"))
