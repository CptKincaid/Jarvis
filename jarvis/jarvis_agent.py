"""Jarvis Agent — intelligence layer for proactive assistance.

Handles: screen awareness, habit learning, contextual memory,
multi-step workflows, proactive alerts, notification reading,
and intent interpretation.
"""

import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("/tmp/vss_voice")
DATA_DIR = Path.home() / ".aiws_trainer" / "jarvis_data"
HABITS_FILE = DATA_DIR / "habits.json"
MEMORY_FILE = DATA_DIR / "context_memory.json"
WORKFLOWS_FILE = DATA_DIR / "workflows.json"


def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"{ts} [Agent] {msg}"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "gui_debug.log", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


class JarvisAgent:
    """Proactive intelligence layer for Jarvis."""

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._habits = []
        self._memory = {
            "last_windows": deque(maxlen=20),
            "last_commands": deque(maxlen=50),
            "session_notes": [],
            "current_app": None,
        }
        self._alerts_active = False
        self._workflows = {}
        self._load_data()

    def _load_data(self):
        try:
            if HABITS_FILE.exists():
                self._habits = json.loads(HABITS_FILE.read_text())
        except Exception:
            self._habits = []
        try:
            if WORKFLOWS_FILE.exists():
                self._workflows = json.loads(WORKFLOWS_FILE.read_text())
        except Exception:
            self._workflows = {}

    def _save_habits(self):
        try:
            HABITS_FILE.write_text(json.dumps(self._habits[-500:], indent=2))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 1. Screen Awareness
    # ------------------------------------------------------------------
    def analyze_screen(self):
        """Capture screen and describe what's visible."""
        try:
            from PIL import ImageGrab
            capture_dir = Path("/tmp/vss_screen")
            capture_dir.mkdir(parents=True, exist_ok=True)
            img = ImageGrab.grab()
            img.save(str(capture_dir / "latest.png"))

            # Get active window info
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=2,
            )
            active_window = result.stdout.strip()

            # Get window geometry
            result2 = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowgeometry"],
                capture_output=True, text=True, timeout=2,
            )
            geometry = result2.stdout.strip()

            self._memory["current_app"] = active_window
            _log(f"Screen: active={active_window}")

            return {
                "active_window": active_window,
                "geometry": geometry,
                "screenshot": str(capture_dir / "latest.png"),
            }
        except Exception as e:
            _log(f"Screen analysis error: {e}")
            return None

    def find_on_screen(self, target):
        """Try to find a UI element on screen by text/description.

        Uses OCR to find text on screen and return coordinates.
        """
        try:
            from PIL import ImageGrab
            import pytesseract

            img = ImageGrab.grab()
            # Run OCR to find text positions
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            target_lower = target.lower()
            matches = []
            for i, text in enumerate(data["text"]):
                if text.strip().lower() and target_lower in text.strip().lower():
                    x = data["left"][i] + data["width"][i] // 2
                    y = data["top"][i] + data["height"][i] // 2
                    conf = data["conf"][i]
                    if conf > 30:
                        matches.append({"text": text, "x": x, "y": y, "conf": conf})

            _log(f"Find on screen '{target}': {len(matches)} matches")
            return matches
        except ImportError:
            _log("pytesseract not installed — OCR unavailable")
            return []
        except Exception as e:
            _log(f"Find on screen error: {e}")
            return []

    def click_on_text(self, target):
        """Find text on screen and click it."""
        matches = self.find_on_screen(target)
        if matches:
            best = max(matches, key=lambda m: m["conf"])
            _log(f"Clicking '{target}' at ({best['x']}, {best['y']})")
            subprocess.run(
                ["xdotool", "mousemove", str(best["x"]), str(best["y"])],
                capture_output=True, timeout=2,
            )
            time.sleep(0.1)
            subprocess.run(
                ["xdotool", "click", "1"],
                capture_output=True, timeout=2,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # 2. Habit Learning
    # ------------------------------------------------------------------
    def log_command(self, command, context=None):
        """Log a command execution for habit learning."""
        entry = {
            "time": datetime.now().isoformat(),
            "hour": datetime.now().hour,
            "day": datetime.now().strftime("%A"),
            "command": command,
            "context": context or self._memory.get("current_app"),
        }
        self._habits.append(entry)
        self._memory["last_commands"].append(entry)
        self._save_habits()

    def suggest_command(self):
        """Suggest a command based on current time and habits."""
        if len(self._habits) < 10:
            return None

        current_hour = datetime.now().hour
        current_day = datetime.now().strftime("%A")

        # Find commands commonly run at this hour
        hour_commands = {}
        for h in self._habits:
            if abs(h.get("hour", -1) - current_hour) <= 1:
                cmd = h["command"]
                hour_commands[cmd] = hour_commands.get(cmd, 0) + 1

        if hour_commands:
            # Most common command at this time
            best = max(hour_commands, key=hour_commands.get)
            count = hour_commands[best]
            if count >= 3:  # Only suggest if done 3+ times at this hour
                # Check if already done recently
                recent = [c["command"] for c in list(self._memory["last_commands"])[-5:]]
                if best not in recent:
                    return best
        return None

    # ------------------------------------------------------------------
    # 3. Contextual Memory
    # ------------------------------------------------------------------
    def remember(self, key, value):
        """Store a contextual memory."""
        self._memory["session_notes"].append({
            "time": datetime.now().isoformat(),
            "key": key,
            "value": value,
        })
        # Keep last 100 notes
        self._memory["session_notes"] = self._memory["session_notes"][-100:]
        _log(f"Memory: {key} = {str(value)[:50]}")

    def recall(self, query):
        """Search contextual memory for relevant info."""
        query_lower = query.lower()
        matches = []
        for note in reversed(self._memory["session_notes"]):
            if (query_lower in note.get("key", "").lower() or
                    query_lower in str(note.get("value", "")).lower()):
                matches.append(note)
            if len(matches) >= 5:
                break
        return matches

    def track_window(self, window_name):
        """Track window focus changes."""
        self._memory["last_windows"].append({
            "time": datetime.now().isoformat(),
            "window": window_name,
        })
        self._memory["current_app"] = window_name

    def get_last_window(self):
        """Get the previously focused window (for 'go back')."""
        windows = list(self._memory["last_windows"])
        if len(windows) >= 2:
            return windows[-2]["window"]
        return None

    # ------------------------------------------------------------------
    # 4. Multi-step Workflows
    # ------------------------------------------------------------------
    def define_workflow(self, name, steps):
        """Define a reusable workflow."""
        self._workflows[name] = steps
        try:
            WORKFLOWS_FILE.write_text(json.dumps(self._workflows, indent=2))
        except Exception:
            pass
        _log(f"Workflow defined: {name} ({len(steps)} steps)")

    def get_workflow(self, name):
        """Get a workflow by name."""
        return self._workflows.get(name)

    # Default workflows
    DEFAULT_WORKFLOWS = {
        "deploy": [
            ("shell", "cd /home/hunterp/vss_env && python scripts/agents/test_agent.py"),
            ("shell", "cd /home/hunterp/vss_env && python scripts/agents/security_agent.py"),
            ("shell", "cd /home/hunterp/vss_env && git add -A && git status -s"),
            ("speak", "Tests and security scan complete. Ready to commit."),
        ],
        "morning": [
            ("shell", "nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv"),
            ("shell", "df -h / /storage 2>/dev/null"),
            ("shell", "uptime"),
            ("speak", "Good morning sir. Systems are online. GPU and storage look healthy."),
        ],
        "training check": [
            ("shell", "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv"),
            ("shell", "ls -lt /home/hunterp/vss_env/aiws_system/training_data/*/images/ 2>/dev/null | head -5"),
            ("speak", "Training status retrieved."),
        ],
    }

    # ------------------------------------------------------------------
    # 5. Proactive Alerts
    # ------------------------------------------------------------------
    def start_monitoring(self, speak_func=None):
        """Start background monitoring for proactive alerts."""
        if self._alerts_active:
            return
        self._alerts_active = True
        self._speak_func = speak_func
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        _log("Proactive monitoring started")

    def stop_monitoring(self):
        self._alerts_active = False

    def _monitor_loop(self):
        """Background loop checking system health."""
        while self._alerts_active:
            try:
                self._check_gpu_temp()
                self._check_disk_space()
            except Exception as e:
                _log(f"Monitor error: {e}")
            time.sleep(300)  # Check every 5 minutes

    def _check_gpu_temp(self):
        """Alert if GPU temperature is too high."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().splitlines():
                temp = int(line.strip())
                if temp > 85:
                    msg = f"Warning: GPU temperature is {temp} degrees. Consider reducing workload."
                    _log(f"ALERT: GPU temp {temp}C")
                    self._alert(msg)
        except Exception:
            pass

    def _check_disk_space(self):
        """Alert if disk space is low."""
        try:
            import shutil
            usage = shutil.disk_usage("/")
            free_gb = usage.free / (1024 ** 3)
            if free_gb < 10:
                msg = f"Warning: Only {free_gb:.1f} gigabytes of disk space remaining."
                _log(f"ALERT: Low disk {free_gb:.1f}GB")
                self._alert(msg)
        except Exception:
            pass

    def _alert(self, message):
        """Send a proactive alert."""
        # Desktop notification
        try:
            subprocess.Popen(
                ["notify-send", "-u", "critical", "-i", "dialog-warning",
                 "Jarvis Alert", message],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
        # Speak if available
        if self._speak_func:
            try:
                from jarvis.jarvis_speak_queue import say
                say(message)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 6. Notification Reading
    # ------------------------------------------------------------------
    def read_notification(self, title, body):
        """Read a desktop notification aloud."""
        text = f"Notification from {title}. {body}"
        _log(f"Reading notification: {title}")
        try:
            from jarvis.jarvis_speak_queue import say
            say(text)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 7. Intent Interpretation
    # ------------------------------------------------------------------
    def interpret_intent(self, raw_text):
        """Interpret user's intent and craft a better prompt for Claude.

        Returns (enhanced_text, context) or None if no enhancement needed.
        """
        lower = raw_text.lower()

        # "fix the bug" → gather context first
        if "fix" in lower and ("bug" in lower or "error" in lower):
            context = self._gather_error_context()
            if context:
                enhanced = (f"{raw_text}\n\nHere's the relevant context:\n"
                            f"```\n{context}\n```")
                return enhanced

        # "what's the status" → gather system info
        if "status" in lower or "how's it going" in lower:
            status = self._gather_status()
            if status:
                enhanced = f"{raw_text}\n\nCurrent system status:\n{status}"
                return enhanced

        return None

    def _gather_error_context(self):
        """Gather recent error logs for bug fixing context."""
        try:
            # Check recent GUI debug log for errors
            result = subprocess.run(
                ["grep", "-i", "error\\|exception\\|traceback",
                 str(LOG_DIR / "gui_debug.log")],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout:
                return result.stdout.strip()[-500:]
        except Exception:
            pass
        return None

    def get_greeting(self):
        """Generate a contextual greeting based on time of day."""
        hour = datetime.now().hour
        if hour < 6:
            return "Burning the midnight oil, sir?"
        elif hour < 12:
            return "Good morning sir. All systems are online."
        elif hour < 17:
            return "Good afternoon sir."
        elif hour < 21:
            return "Good evening sir."
        else:
            return "Working late tonight sir. All systems operational."

    def get_status_summary(self):
        """Quick one-line status for greeting."""
        parts = []
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.stdout.strip():
                vals = r.stdout.strip().split(", ")
                if len(vals) >= 2:
                    parts.append(f"GPU at {vals[0]}% utilization, {vals[1]} degrees")
        except Exception:
            pass

        cmd_count = len(self._habits)
        if cmd_count > 0:
            parts.append(f"{cmd_count} commands logged")

        return ". ".join(parts) if parts else "All systems nominal"

    def _gather_status(self):
        """Gather current system status."""
        parts = []
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if r.stdout.strip():
                parts.append(f"GPU: {r.stdout.strip()}")
        except Exception:
            pass
        try:
            r = subprocess.run(["uptime"], capture_output=True, text=True, timeout=5)
            if r.stdout.strip():
                parts.append(f"Uptime: {r.stdout.strip()}")
        except Exception:
            pass
        return "\n".join(parts) if parts else None

    # ------------------------------------------------------------------
    # 8. Process Monitor
    # ------------------------------------------------------------------
    def list_heavy_processes(self, top_n=5):
        """List the heaviest processes by CPU/memory."""
        try:
            r = subprocess.run(
                ["ps", "aux", "--sort=-%cpu"],
                capture_output=True, text=True, timeout=5,
            )
            lines = r.stdout.strip().splitlines()[1:top_n + 1]
            processes = []
            for line in lines:
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    processes.append({
                        "user": parts[0], "cpu": parts[2],
                        "mem": parts[3], "cmd": parts[10][:40],
                    })
            return processes
        except Exception:
            return []

    # ------------------------------------------------------------------
    # 9. Git Intelligence
    # ------------------------------------------------------------------
    def git_summary(self, repo_path="/home/hunterp/vss_env"):
        """Get a quick git status summary."""
        try:
            def _run(cmd):
                return subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=5, cwd=repo_path,
                ).stdout.strip()

            branch = _run(["git", "branch", "--show-current"])
            status = _run(["git", "status", "--porcelain"])
            changed = len(status.splitlines()) if status else 0
            last_commit = _run(["git", "log", "--oneline", "-1"])
            ahead = _run(["git", "rev-list", "--count", f"origin/{branch}..HEAD"])

            return {
                "branch": branch,
                "changed_files": changed,
                "last_commit": last_commit,
                "commits_ahead": int(ahead) if ahead.isdigit() else 0,
            }
        except Exception as e:
            _log(f"Git summary error: {e}")
            return None

    # ------------------------------------------------------------------
    # 10. Network / Connectivity
    # ------------------------------------------------------------------
    def check_connectivity(self):
        """Check internet and local network status."""
        results = {}
        # Internet
        try:
            r = subprocess.run(
                ["ping", "-c", "1", "-W", "2", "8.8.8.8"],
                capture_output=True, text=True, timeout=5,
            )
            results["internet"] = r.returncode == 0
            if r.returncode == 0:
                # Extract latency
                m = re.search(r"time=(\d+\.?\d*)", r.stdout)
                if m:
                    results["latency_ms"] = float(m.group(1))
        except Exception:
            results["internet"] = False

        # Ollama
        try:
            r = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 "http://localhost:11434/api/tags"],
                capture_output=True, text=True, timeout=3,
            )
            results["ollama"] = r.stdout.strip() == "200"
        except Exception:
            results["ollama"] = False

        return results

    # ------------------------------------------------------------------
    # 11. File Operations
    # ------------------------------------------------------------------
    def find_file(self, name, search_dir="/home/hunterp/vss_env"):
        """Find files matching a name pattern."""
        try:
            r = subprocess.run(
                ["find", search_dir, "-iname", f"*{name}*",
                 "-not", "-path", "*/.*", "-not", "-path", "*/node_modules/*",
                 "-type", "f"],
                capture_output=True, text=True, timeout=10,
            )
            files = r.stdout.strip().splitlines()[:10]
            return files
        except Exception:
            return []

    def recent_files(self, directory="/home/hunterp/vss_env", count=5):
        """List recently modified files."""
        try:
            r = subprocess.run(
                ["find", directory, "-maxdepth", "3", "-type", "f",
                 "-not", "-path", "*/.*", "-not", "-path", "*/__pycache__/*",
                 "-printf", "%T@ %p\n"],
                capture_output=True, text=True, timeout=10,
            )
            lines = sorted(r.stdout.strip().splitlines(), reverse=True)[:count]
            return [l.split(" ", 1)[1] for l in lines if " " in l]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # 12. Smart Responses
    # ------------------------------------------------------------------
    def answer_question(self, question):
        """Try to answer simple questions without Claude."""
        q = question.lower()

        if "time" in q and ("what" in q or "current" in q):
            return datetime.now().strftime("It is %I:%M %p on %A, %B %d.")

        if "date" in q and ("what" in q or "today" in q):
            return datetime.now().strftime("Today is %A, %B %d, %Y.")

        if "day" in q and ("what" in q):
            return datetime.now().strftime("Today is %A.")

        if "uptime" in q or "how long" in q and "running" in q:
            try:
                r = subprocess.run(["uptime", "-p"], capture_output=True,
                                   text=True, timeout=5)
                return f"The system has been {r.stdout.strip()}."
            except Exception:
                pass

        if "ip" in q and ("what" in q or "my" in q):
            try:
                r = subprocess.run(
                    ["hostname", "-I"], capture_output=True,
                    text=True, timeout=5,
                )
                return f"Your IP address is {r.stdout.strip().split()[0]}."
            except Exception:
                pass

        if "battery" in q or "power" in q:
            try:
                r = subprocess.run(
                    ["upower", "-i", "/org/freedesktop/UPower/devices/battery_BAT0"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in r.stdout.splitlines():
                    if "percentage" in line:
                        return f"Battery is at {line.split(':')[1].strip()}."
            except Exception:
                return "No battery detected. You appear to be on a desktop."

        if "weather" in q:
            try:
                r = subprocess.run(
                    ["curl", "-s", "wttr.in/?format=3"],
                    capture_output=True, text=True, timeout=5,
                )
                return r.stdout.strip()
            except Exception:
                return "Could not fetch weather data."

        return None

    # ------------------------------------------------------------------
    # 13. Clipboard History
    # ------------------------------------------------------------------
    def _init_clipboard(self):
        if not hasattr(self, '_clipboard_history'):
            self._clipboard_history = deque(maxlen=20)
            self._last_clip = ""

    def poll_clipboard(self):
        self._init_clipboard()
        try:
            r = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True, timeout=2,
            )
            current = r.stdout.strip()
            if current and current != self._last_clip:
                self._clipboard_history.append({
                    "time": datetime.now().isoformat(),
                    "text": current[:500],
                })
                self._last_clip = current
        except Exception:
            pass

    def get_clipboard_history(self, n=5):
        self._init_clipboard()
        return list(self._clipboard_history)[-n:]

    def paste_from_history(self, index):
        self._init_clipboard()
        items = list(self._clipboard_history)
        if 0 <= index < len(items):
            text = items[-(index + 1)]["text"]
            try:
                proc = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE,
                )
                proc.communicate(input=text.encode(), timeout=2)
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                    timeout=2, capture_output=True,
                )
                return text[:50]
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # 14. Voice Notes
    # ------------------------------------------------------------------
    def save_voice_note(self, text):
        notes_dir = DATA_DIR / "voice_notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        note_file = notes_dir / f"note_{ts}.txt"
        note_file.write_text(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]\n{text}\n")
        _log(f"Voice note saved: {note_file.name}")
        return str(note_file)

    def list_voice_notes(self, n=5):
        notes_dir = DATA_DIR / "voice_notes"
        if not notes_dir.exists():
            return []
        files = sorted(notes_dir.glob("note_*.txt"), reverse=True)[:n]
        return [{"file": f.name, "content": f.read_text().strip()[:100]}
                for f in files]

    # ------------------------------------------------------------------
    # 15. Shell Piping
    # ------------------------------------------------------------------
    def run_shell(self, command):
        _log(f"Shell: {command[:60]}")
        try:
            r = subprocess.run(
                command, shell=True, capture_output=True,
                text=True, timeout=30,
                cwd="/home/hunterp/jarvis",
            )
            output = r.stdout.strip()
            if r.returncode != 0 and r.stderr:
                output += f"\nError: {r.stderr.strip()[:200]}"
            return output[:1000]
        except subprocess.TimeoutExpired:
            return "Command timed out."
        except Exception as e:
            return f"Error: {e}"

    # ------------------------------------------------------------------
    # 16. Multi-Monitor
    # ------------------------------------------------------------------
    def move_window_to_monitor(self, direction="next"):
        try:
            r = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, timeout=2,
            )
            wid = r.stdout.strip()
            r2 = subprocess.run(
                ["xdotool", "getwindowgeometry", wid],
                capture_output=True, text=True, timeout=2,
            )
            m = re.search(r"Position: (\d+),(\d+)", r2.stdout)
            if not m:
                return False
            x, y = int(m.group(1)), int(m.group(2))

            r3 = subprocess.run(
                ["xrandr", "--query"],
                capture_output=True, text=True, timeout=2,
            )
            monitors = []
            for line in r3.stdout.splitlines():
                match = re.search(r"(\d+)x(\d+)\+(\d+)\+(\d+)", line)
                if match and " connected" in line:
                    monitors.append({
                        "w": int(match.group(1)), "h": int(match.group(2)),
                        "x": int(match.group(3)), "y": int(match.group(4)),
                    })
            if len(monitors) < 2:
                return False

            current_idx = 0
            for i, mon in enumerate(monitors):
                if mon["x"] <= x < mon["x"] + mon["w"]:
                    current_idx = i
                    break

            target_idx = ((current_idx + 1) if direction == "next"
                          else (current_idx - 1)) % len(monitors)
            target = monitors[target_idx]
            new_x = target["x"] + (x - monitors[current_idx]["x"])
            new_y = target["y"] + (y - monitors[current_idx]["y"])

            subprocess.run(
                ["xdotool", "windowmove", wid, str(new_x), str(new_y)],
                capture_output=True, timeout=2,
            )
            _log(f"Moved window to monitor {target_idx}")
            return True
        except Exception as e:
            _log(f"Move monitor error: {e}")
            return False

    # ------------------------------------------------------------------
    # 17. Conditional Triggers
    # ------------------------------------------------------------------
    def set_trigger(self, condition, action_msg):
        if not hasattr(self, '_triggers'):
            self._triggers = []
        self._triggers.append({"condition": condition, "action": action_msg})
        _log(f"Trigger set: when {condition} → {action_msg}")

    def check_triggers(self):
        if not hasattr(self, '_triggers'):
            return []
        fired = []
        remaining = []
        for trigger in self._triggers:
            cond = trigger["condition"].lower()
            if "gpu" in cond and "below" in cond:
                m = re.search(r"(\d+)", cond)
                if m:
                    try:
                        r = subprocess.run(
                            ["nvidia-smi", "--query-gpu=utilization.gpu",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5,
                        )
                        if int(r.stdout.strip().splitlines()[0]) < int(m.group(1)):
                            fired.append(trigger)
                            continue
                    except Exception:
                        pass
            remaining.append(trigger)
        self._triggers = remaining
        return fired
