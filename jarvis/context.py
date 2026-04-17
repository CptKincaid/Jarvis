"""Context Engine — rich context for all intelligence tiers.

Gathers system state, git info, active window, recent files,
conversation history, errors, and screen info. Feeds this
context to Ollama (Tier 2) and Claude (Tier 3) for smarter responses.

Usage:
    ctx = ContextEngine()
    snapshot = ctx.get_context("standard")
    prompt_text = ctx.format_for_prompt(snapshot)
"""

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("/tmp/vss_voice")


def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "gui_debug.log", "a") as f:
            f.write(f"{ts} [Context] {msg}\n")
    except Exception:
        pass


class ContextEngine:
    """Builds rich context snapshots for AI tiers."""

    CACHE_TTL = 30  # seconds

    def __init__(self, project_dir=str(Path.home() / "jarvis"),
                 vss_dir=str(Path.home() / "vss_env")):
        self._cache = {}
        self._cache_times = {}
        self._conversation = []
        self._project_dir = project_dir
        self._vss_dir = vss_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_context(self, detail="standard"):
        """Get context snapshot.

        detail levels:
            'minimal'  — time + active window (for Tier 1)
            'standard' — + git, recent files, conversation (for Tier 2)
            'full'     — + system state, errors, screen, memory (for Tier 3)
        """
        ctx = {
            "time": datetime.now().strftime("%I:%M %p, %A %B %d %Y"),
            "active_window": self._get_active_window(),
        }

        if detail in ("standard", "full"):
            ctx["git"] = self._get_git_state()
            ctx["recent_files"] = self._get_recent_files()
            ctx["conversation"] = self._conversation[-10:]

        if detail == "full":
            ctx["system"] = self._get_system_state()
            ctx["errors"] = self._get_recent_errors()
            ctx["screen"] = self._get_screen_path()
            ctx["processes"] = self._get_top_processes()

        return ctx

    def add_exchange(self, user_text, jarvis_response):
        """Record a conversation exchange for context."""
        self._conversation.append({
            "time": datetime.now().isoformat(),
            "user": user_text[:200],
            "jarvis": jarvis_response[:200] if jarvis_response else "",
        })
        self._conversation = self._conversation[-20:]

    def format_for_prompt(self, ctx):
        """Format context dict into text suitable for LLM prompt injection."""
        parts = [f"Current time: {ctx['time']}"]

        if ctx.get("active_window"):
            parts.append(f"Active window: {ctx['active_window']}")

        if ctx.get("git"):
            g = ctx["git"]
            parts.append(
                f"Git: branch={g.get('branch', '?')} | "
                f"{g.get('changed', 0)} changed files | "
                f"last commit: {g.get('last_commit', '?')}"
            )
            if g.get("ahead", 0) > 0:
                parts.append(f"  {g['ahead']} commits ahead of remote")

        if ctx.get("recent_files"):
            names = [Path(f.split(" ", 1)[-1] if " " in f else f).name
                     for f in ctx["recent_files"][:5]]
            parts.append(f"Recently modified: {', '.join(names)}")

        if ctx.get("conversation"):
            parts.append("Recent conversation:")
            for ex in ctx["conversation"][-4:]:
                parts.append(f"  User: {ex['user'][:80]}")
                if ex.get("jarvis"):
                    parts.append(f"  Jarvis: {ex['jarvis'][:80]}")

        if ctx.get("system"):
            parts.append(f"System: {ctx['system']}")

        if ctx.get("errors"):
            parts.append(f"Recent errors ({len(ctx['errors'])}):")
            for err in ctx["errors"][-3:]:
                parts.append(f"  {err[:100]}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Cached data sources
    # ------------------------------------------------------------------
    def _cached(self, key, getter):
        now = time.monotonic()
        if (key in self._cache and
                now - self._cache_times.get(key, 0) < self.CACHE_TTL):
            return self._cache[key]
        try:
            value = getter()
        except Exception as e:
            _log(f"Cache fetch error ({key}): {e}")
            value = self._cache.get(key)  # Return stale if available
        self._cache[key] = value
        self._cache_times[key] = now
        return value

    def _get_active_window(self):
        return self._cached("window", self._fetch_active_window)

    def _fetch_active_window(self):
        try:
            r = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=2,
            )
            return r.stdout.strip()
        except Exception:
            return "unknown"

    def _get_git_state(self):
        return self._cached("git", self._fetch_git)

    def _fetch_git(self):
        # Check both Jarvis and VSS repos
        for repo_dir in [self._project_dir, self._vss_dir]:
            try:
                def _run(cmd):
                    return subprocess.run(
                        cmd, capture_output=True, text=True,
                        timeout=5, cwd=repo_dir,
                    ).stdout.strip()

                branch = _run(["git", "branch", "--show-current"])
                if not branch:
                    continue

                status = _run(["git", "status", "--porcelain"])
                changed = len(status.splitlines()) if status else 0
                last = _run(["git", "log", "--oneline", "-1"])

                # Check ahead/behind
                ahead = 0
                try:
                    ahead_str = _run(
                        ["git", "rev-list", "--count", f"origin/{branch}..HEAD"])
                    ahead = int(ahead_str) if ahead_str.isdigit() else 0
                except Exception:
                    pass

                return {
                    "repo": Path(repo_dir).name,
                    "branch": branch,
                    "changed": changed,
                    "last_commit": last,
                    "ahead": ahead,
                }
            except Exception:
                continue
        return {}

    def _get_recent_files(self):
        return self._cached("files", self._fetch_recent_files)

    def _fetch_recent_files(self):
        try:
            r = subprocess.run(
                ["find", self._project_dir, "-maxdepth", "3", "-type", "f",
                 "-not", "-path", "*/.*", "-not", "-path", "*/__pycache__/*",
                 "-not", "-name", "*.pyc",
                 "-printf", "%T@ %p\n"],
                capture_output=True, text=True, timeout=5,
            )
            lines = sorted(r.stdout.strip().splitlines(), reverse=True)
            return lines[:5]
        except Exception:
            return []

    def _get_system_state(self):
        return self._cached("system", self._fetch_system)

    def _fetch_system(self):
        parts = []
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if r.stdout.strip():
                parts.append(f"GPU: {r.stdout.strip()}")
        except Exception:
            pass

        try:
            r = subprocess.run(
                ["uptime", "-p"], capture_output=True, text=True, timeout=5,
            )
            if r.stdout.strip():
                parts.append(f"Uptime: {r.stdout.strip()}")
        except Exception:
            pass

        try:
            import shutil
            usage = shutil.disk_usage("/")
            free_gb = usage.free / (1024 ** 3)
            parts.append(f"Disk: {free_gb:.1f}GB free")
        except Exception:
            pass

        return " | ".join(parts) if parts else "unavailable"

    def _get_recent_errors(self):
        try:
            log = LOG_DIR / "gui_debug.log"
            if not log.exists():
                return []
            # Read last 100 lines efficiently
            text = log.read_text()
            lines = text.splitlines()[-100:]
            errors = [
                l for l in lines
                if any(kw in l.lower() for kw in
                       ("error", "exception", "traceback", "failed"))
            ]
            return errors[-5:]
        except Exception:
            return []

    def _get_screen_path(self):
        p = Path("/tmp/vss_screen/latest.png")
        return str(p) if p.exists() else None

    def _get_top_processes(self):
        return self._cached("procs", self._fetch_processes)

    def _fetch_processes(self):
        try:
            r = subprocess.run(
                ["ps", "aux", "--sort=-%cpu"],
                capture_output=True, text=True, timeout=5,
            )
            lines = r.stdout.strip().splitlines()[1:4]
            procs = []
            for line in lines:
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    procs.append(f"{parts[10][:30]} (CPU:{parts[2]}%)")
            return procs
        except Exception:
            return []
