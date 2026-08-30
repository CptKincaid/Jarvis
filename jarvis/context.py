"""Context Engine — rich context for all intelligence tiers.

Gathers system state, git info, active window, recent files, conversation
history, session summaries, errors, and screen info. Feeds this context to
Ollama (Tier 2) and Claude (Tier 3) for smarter responses.

V3 notes:
- project_dir defaults to PATHS.REPO_ROOT (fixes the /home/hunterp/jarvis
  case bug).
- GB10/unified-memory truth: GPU stats come from nvidia-smi *utilization/
  temperature only* (memory.used/total read N/A there); RAM comes from
  /proc/meminfo.
- Absorbs jarvis_agent.py's screen capture (agent lines 73-103) and window
  tracking; the capture is the writer of /tmp/vss_screen/latest.png that
  full context reports.
- Session summaries from an injected JarvisMemory appear in standard/full
  context.
- Every key gathered by get_context() is rendered by format_for_prompt()
  (the V1 gather/render mismatch is gone).
- Activity journal (2026-08-30): every exchange, tool call, Claude task
  result and window-focus change is appended as one JSON line to
  ``PATHS.MEMORY_DIR/journal/<YYYY-MM-DD>.jsonl`` (untruncated, one file per
  day so rollover is free). ``journal_rows()`` reads a time window back for
  "recap my day" (jarvis/tools/journal.py). A write failure logs and drops
  the row; it never reaches the turn.

Usage:
    ctx = ContextEngine(memory=mem)
    snapshot = ctx.get_context("standard")
    prompt_text = ctx.format_for_prompt(snapshot)
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path

from jarvis.config import PATHS
from jarvis.logs import LOG_FILE, get_logger

log = get_logger("context")

_LEGACY_LOG = PATHS.LOG_DIR / "gui_debug.log"
# One journal row keeps the whole utterance and reply (add_exchange's
# 200-char cap is for the prompt window, not the record), but a pasted
# document or a Claude transcript must not turn the day file into a dump.
JOURNAL_TEXT_CAP = 4000
JOURNAL_KINDS = ("exchange", "tool", "claude", "window")


class ContextEngine:
    """Builds rich context snapshots for AI tiers."""

    CACHE_TTL = 30  # seconds
    # Exchanges older than this are not shown to the model: "what about
    # tomorrow?" refers to the last minute, not to this morning.
    CONVERSATION_TTL_S = 10 * 60
    # Class-level defaults so an engine built without __init__ (the tests'
    # ContextEngine.__new__ pattern) can still record an exchange.
    _journal_dir = None
    _journal_lock = threading.Lock()
    _journal_failed_logged = False
    _last_journaled_window = None

    def __init__(self, project_dir=None, vss_dir=None, memory=None,
                 journal_dir=None):
        self._cache = {}
        self._cache_times = {}
        self._conversation = []
        self._project_dir = str(project_dir or PATHS.REPO_ROOT)
        self._vss_dir = str(vss_dir or PATHS.VSS_ENV)
        self._memory = memory          # injected JarvisMemory (optional)
        self._windows = deque(maxlen=20)
        self._current_app = None
        # journal_dir=None resolves PATHS.MEMORY_DIR at write time, so the
        # test firewall's env redirect is honoured whenever it was set.
        self._journal_dir = Path(journal_dir) if journal_dir else None
        self._journal_lock = threading.Lock()
        self._journal_failed_logged = False
        self._last_journaled_window = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_context(self, detail="standard"):
        """Get context snapshot.

        detail levels:
            'minimal'  — time + active window (for Tier 1)
            'standard' — + git, recent files, conversation, sessions (Tier 2)
            'full'     — + system state, errors, screen, processes (Tier 3)
        """
        ctx = {
            "time": datetime.now().strftime("%I:%M %p, %A %B %d %Y"),
            "active_window": self._get_active_window(),
        }

        if detail in ("standard", "full"):
            ctx["git"] = self._get_git_state()
            ctx["recent_files"] = self._get_recent_files()
            ctx["conversation"] = self._recent_conversation()
            ctx["sessions"] = self._get_sessions()

        if detail == "full":
            ctx["system"] = self._get_system_state()
            ctx["errors"] = self._get_recent_errors()
            ctx["screen"] = self._get_screen_path()
            ctx["processes"] = self._get_top_processes()

        return ctx

    def add_exchange(self, user_text, jarvis_response):
        """Record a conversation exchange for context (and the journal)."""
        self._conversation.append({
            "time": datetime.now().isoformat(),
            "user": (user_text or "")[:200],
            "jarvis": jarvis_response[:200] if jarvis_response else "",
        })
        self._conversation = self._conversation[-20:]
        if user_text or jarvis_response:
            self._journal_write("exchange", user=user_text or "",
                                jarvis=jarvis_response or "")

    # ------------------------------------------------------------------
    # Activity journal — one JSON line per event, one file per day
    # ------------------------------------------------------------------
    def journal_dir(self) -> Path:
        return self._journal_dir or (PATHS.MEMORY_DIR / "journal")

    def _journal_write(self, kind, **fields):
        """Append {time, kind, ...} to today's file. Mirrors
        turnclock._default_emit: any failure is logged once and dropped."""
        row = {"time": datetime.now().isoformat(timespec="seconds"), "kind": kind}
        for key, value in fields.items():
            if isinstance(value, str) and len(value) > JOURNAL_TEXT_CAP:
                value = value[:JOURNAL_TEXT_CAP] + "…"
            row[key] = value
        try:
            line = json.dumps(row, ensure_ascii=False, default=str)
            with self._journal_lock:
                d = self.journal_dir()
                d.mkdir(parents=True, exist_ok=True)
                with open(d / f"{date.today():%Y-%m-%d}.jsonl", "a",
                          encoding="utf-8") as fh:
                    fh.write(line + "\n")
            self._journal_failed_logged = False
        except Exception:
            if not self._journal_failed_logged:
                log.exception("journal write failed (further failures at debug)")
                self._journal_failed_logged = True
            else:
                log.debug("journal write failed", exc_info=True)

    def journal_tool(self, name, args=None, ok=True, text=""):
        """A tool the model (or a Tier 1 handler) ran."""
        try:
            args = dict(args or {})
        except (TypeError, ValueError):
            args = {"args": str(args)}
        self._journal_write("tool", name=str(name), args=args, ok=bool(ok),
                            text=(text or "")[:600])

    def journal_claude(self, project, state, text=""):
        """A Claude task reached a terminal state (done / failed)."""
        self._journal_write("claude", project=str(project or ""),
                            state=str(state or ""), text=text or "")

    def journal_window(self, title):
        """A focus sample from the activity sampler: journaled only when it
        changed, so an hour in one window is one row, not sixty."""
        title = (title or "").strip()
        if not title or title == "unknown":
            return False          # xdotool failed / no focused window (locked)
        if title == self._last_journaled_window:
            return False
        self._last_journaled_window = title
        self.track_window(title)
        self._journal_write("window", title=title)
        return True

    def journal_rows(self, since=None, until=None, kinds=None):
        """Rows between ``since`` and ``until`` (datetimes; default: today
        so far), oldest first. ``kinds`` restricts to JOURNAL_KINDS
        members. Unreadable lines are skipped."""
        now = datetime.now()
        until = until or now
        since = since or until.replace(hour=0, minute=0, second=0, microsecond=0)
        if since > until:
            since, until = until, since
        want = set(kinds) if kinds else None
        rows = []
        day = since.date()
        last = until.date()
        while day <= last:
            path = self.journal_dir() / f"{day:%Y-%m-%d}.jsonl"
            if path.exists():
                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    log.debug("journal read failed: %s", path, exc_info=True)
                    lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        when = datetime.fromisoformat(str(row.get("time", "")))
                    except (ValueError, TypeError):
                        continue
                    if not (since <= when <= until):
                        continue
                    if want and row.get("kind") not in want:
                        continue
                    row["_when"] = when
                    rows.append(row)
            day += timedelta(days=1)
        rows.sort(key=lambda r: r["_when"])
        return rows

    def _recent_conversation(self, limit=10):
        """The last exchanges that are still fresh (CONVERSATION_TTL_S)."""
        cutoff = datetime.now().timestamp() - self.CONVERSATION_TTL_S
        fresh = []
        for e in self._conversation[-limit:]:
            try:
                if datetime.fromisoformat(e["time"]).timestamp() >= cutoff:
                    fresh.append(e)
            except (KeyError, ValueError):
                continue
        return fresh


    def recent_conversation_text(self, limit=4) -> str:
        """The last exchanges as plain lines and nothing else -- for a
        prompt that leaves the machine (the web lookup). format_for_prompt
        also carries the git state, the active window and the session log."""
        lines = []
        for e in self._recent_conversation()[-limit:]:
            if e.get("user"):
                lines.append(f"User: {e['user'][:200]}")
            if e.get("jarvis"):
                lines.append(f"Jarvis: {e['jarvis'][:200]}")
        return "\n".join(lines)
    def format_for_prompt(self, ctx, spoken=False):
        """Format context dict into text for LLM prompt injection.

        Renders every key get_context() gathers (time, active_window, git,
        recent_files, conversation, sessions, system, errors, screen,
        processes). The git line is spelled out in words ("40 files changed
        and not yet committed") because the small Tier 2 model misread
        "40 changed files" as "up to date".

        spoken=True renders for the Tier 2 (read-aloud) prompt: the
        recently-modified file names collapse to a count so there is
        nothing for the model to recite ("brain.py, tts.py, ...").
        """
        parts = [f"Current time: {ctx['time']}"]
        # Gathered since V3 but never rendered -- and recorded with the user's
        # side blank -- so the model had never seen a previous exchange and
        # "what about tomorrow?" could not follow a calendar question. This
        # is the ONE rendering: a second, older block further down rendered
        # the same exchanges again until 2026-08-30.
        convo = [e for e in (ctx.get("conversation") or []) if e.get("user") or e.get("jarvis")]
        if convo:
            lines = ["Recent conversation (newest last; answer follow-ups in its light):"]
            for e in convo[-4:]:
                if e.get("user"):
                    lines.append(f"  User: {e['user']}")
                if e.get("jarvis"):
                    lines.append(f"  Jarvis: {e['jarvis']}")
            parts.append("\n".join(lines))

        if ctx.get("active_window"):
            parts.append(f"Active window: {ctx['active_window']}")

        if ctx.get("git"):
            g = ctx["git"]
            changed = g.get("changed", 0) or 0
            if changed:
                state = (f"{changed} file{'s' if changed != 1 else ''} "
                         f"changed and not yet committed")
            else:
                state = "nothing uncommitted"
            last = str(g.get("last_commit", "?"))
            if spoken:      # "6bfe446 V3 overhaul" -> "V3 overhaul"
                last = re.sub(r"^[0-9a-f]{7,40}\s+", "", last)
            parts.append(
                f"Git: on branch {g.get('branch', '?')}, {state}, "
                f"last commit: {last}"
            )
            if g.get("ahead", 0) > 0:
                parts.append(f"  {g['ahead']} commits ahead of remote")

        if ctx.get("recent_files"):
            files = ctx["recent_files"]
            if spoken:
                n = len(files)
                parts.append(f"Recently modified: {n} project "
                             f"file{'s' if n != 1 else ''}")
            else:
                names = [Path(f.split(" ", 1)[-1] if " " in f else f).name
                         for f in files[:5]]
                parts.append(f"Recently modified: {', '.join(names)}")

        if ctx.get("sessions"):
            parts.append("Previous sessions:")
            for s in ctx["sessions"][-3:]:
                parts.append(
                    f"  [{str(s.get('time', ''))[:16]}] "
                    f"{str(s.get('summary', ''))[:100]}")

        if ctx.get("system"):
            parts.append(f"System: {ctx['system']}")

        if ctx.get("errors"):
            parts.append(f"Recent errors ({len(ctx['errors'])}):")
            for err in ctx["errors"][-3:]:
                parts.append(f"  {err[:100]}")

        if ctx.get("screen"):
            parts.append(f"Screen capture available at: {ctx['screen']}")

        if ctx.get("processes"):
            parts.append("Top processes: " + "; ".join(ctx["processes"][:3]))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Screen awareness (absorbed from jarvis_agent.analyze_screen, 73-103)
    # ------------------------------------------------------------------
    def capture_screen(self):
        """Capture the screen to PATHS.SCREEN_DIR/latest.png and describe it.

        Returns {'active_window', 'geometry', 'screenshot'} or None.
        """
        try:
            from PIL import ImageGrab
            capture_dir = PATHS.SCREEN_DIR
            capture_dir.mkdir(parents=True, exist_ok=True)
            img = ImageGrab.grab()
            img.save(str(capture_dir / "latest.png"))

            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=2,
            )
            active_window = result.stdout.strip()

            result2 = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowgeometry"],
                capture_output=True, text=True, timeout=2,
            )
            geometry = result2.stdout.strip()

            self._current_app = active_window
            log.info("screen captured: active=%s", active_window)

            return {
                "active_window": active_window,
                "geometry": geometry,
                "screenshot": str(capture_dir / "latest.png"),
            }
        except Exception:
            log.exception("screen capture failed")
            return None

    # ------------------------------------------------------------------
    # Window tracking (absorbed from jarvis_agent.track_window)
    # ------------------------------------------------------------------
    def track_window(self, window_name):
        """Track window focus changes."""
        self._windows.append({
            "time": datetime.now().isoformat(),
            "window": window_name,
        })
        self._current_app = window_name

    def get_last_window(self):
        """Get the previously focused window (for 'go back')."""
        windows = list(self._windows)
        if len(windows) >= 2:
            return windows[-2]["window"]
        return None

    @property
    def current_app(self):
        return self._current_app

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
        except Exception:
            log.exception("cache fetch error (%s)", key)
            value = self._cache.get(key)  # return stale if available
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
            return r.stdout.strip() or "unknown"
        except Exception:
            log.debug("active window probe failed", exc_info=True)
            return "unknown"

    def _get_git_state(self):
        return self._cached("git", self._fetch_git)

    def _fetch_git(self):
        # Check both the Jarvis repo and the VSS tree
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
                # hash, subject and a relative age ("(6 hours ago)") so the
                # model has the real timing instead of inventing one
                last = _run(["git", "log", "-1", "--format=%h %s (%cr)"])

                ahead = 0
                try:
                    ahead_str = _run(
                        ["git", "rev-list", "--count",
                         f"origin/{branch}..HEAD"])
                    ahead = int(ahead_str) if ahead_str.isdigit() else 0
                except Exception:
                    log.debug("git ahead-count probe failed", exc_info=True)

                return {
                    "repo": Path(repo_dir).name,
                    "branch": branch,
                    "changed": changed,
                    "last_commit": last,
                    "ahead": ahead,
                }
            except Exception:
                log.debug("git probe failed for %s", repo_dir, exc_info=True)
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
            log.debug("recent files probe failed", exc_info=True)
            return []

    def _get_sessions(self):
        """Recent session summaries from the injected memory store."""
        if self._memory is None:
            return []
        try:
            return list(self._memory.get_recent_sessions(3))
        except Exception:
            log.exception("session summary fetch failed")
            return []

    # ------------------------------------------------------------------
    # System state — GB10-honest probes
    # ------------------------------------------------------------------
    def _get_system_state(self):
        return self._cached("system", self._fetch_system)

    def _fetch_system(self):
        parts = []
        # GPU: utilization + temperature ONLY. On GB10 unified memory,
        # nvidia-smi memory.used/memory.total read "N/A" — do not query them.
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
            vals = [v.strip() for v in line.split(",")]
            if len(vals) >= 2 and vals[0]:
                parts.append(f"GPU: {vals[0]}% util, {vals[1]}C")
        except Exception:
            log.debug("gpu probe failed", exc_info=True)

        # System memory from /proc/meminfo (the working source on GB10).
        try:
            mem = self._read_meminfo()
            total_kb = mem.get("MemTotal", 0)
            avail_kb = mem.get("MemAvailable", 0)
            if total_kb:
                used_gb = (total_kb - avail_kb) / (1024 ** 2)
                total_gb = total_kb / (1024 ** 2)
                parts.append(f"RAM: {used_gb:.1f}/{total_gb:.1f}GB used")
        except Exception:
            log.debug("meminfo probe failed", exc_info=True)

        try:
            r = subprocess.run(
                ["uptime", "-p"], capture_output=True, text=True, timeout=5,
            )
            if r.stdout.strip():
                parts.append(f"Uptime: {r.stdout.strip()}")
        except Exception:
            log.debug("uptime probe failed", exc_info=True)

        try:
            import shutil
            usage = shutil.disk_usage("/")
            free_gb = usage.free / (1024 ** 3)
            parts.append(f"Disk: {free_gb:.1f}GB free")
        except Exception:
            log.debug("disk probe failed", exc_info=True)

        return " | ".join(parts) if parts else "unavailable"

    @staticmethod
    def _read_meminfo():
        """Parse /proc/meminfo into {key: kB}."""
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            fields = rest.split()
            if fields:
                try:
                    info[key.strip()] = int(fields[0])
                except ValueError:
                    continue
        return info

    # ------------------------------------------------------------------
    # Errors / screen / processes
    # ------------------------------------------------------------------
    def _get_recent_errors(self):
        try:
            log_path = LOG_FILE if LOG_FILE.exists() else _LEGACY_LOG
            if not log_path.exists():
                return []
            lines = log_path.read_text(errors="replace").splitlines()[-100:]
            errors = [
                ln for ln in lines
                if any(kw in ln.lower() for kw in
                       ("error", "exception", "traceback", "failed"))
            ]
            return errors[-5:]
        except Exception:
            log.exception("error-context read failed")
            return []

    def _get_screen_path(self):
        p = PATHS.SCREEN_DIR / "latest.png"
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
            log.debug("process probe failed", exc_info=True)
            return []
