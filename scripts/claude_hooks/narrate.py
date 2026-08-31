#!/usr/bin/env python3
"""Claude Code hook: narrate the user's OWN terminal sessions through Jarvis.

Runs as a Claude Code *hook* (see scripts/claude_hooks/install.py and
docs/assistant-setup.md, "Claude Code hooks").  Claude Code pipes one JSON
object describing the event on stdin; this script decides whether the event
deserves a spoken line and, if so, appends it to the speak-queue file that
the running Jarvis tails (jarvis/speak_queue.py Watcher).  Nothing is ever
written to stdout: for several hook events stdout is fed back to Claude as
context, and a narration line is not context.

STDLIB ONLY, by design.  A hook runs on every tool call of every session,
so it has to start in a few tens of milliseconds; jarvis.claude_session
drags the bus / config / logging stack in.  The test-outcome regexes and
verdict function below are therefore COPIES of the ones in
jarvis/claude_session.py, and tests/test_claude_hooks.py fails the moment
they drift.

Events (hook_event_name):
  PostToolUse (Bash)  a pytest / ruff / … command -> "3 tests failed in
                      jarvis, sir." — the same verdicts the Jarvis-driven
                      sessions already get.
  Notification        "Claude is waiting on you in <project>, sir." (idle) or
                      "Claude needs your say-so in <project>, sir."
                      (permission prompt).
  UserPromptSubmit    silent; stamps the start of the turn for the Stop gate.
  Stop                "Claude has finished in <project>, sir." — but Stop
                      fires at the end of EVERY assistant turn, so it only
                      speaks when the turn ran a test command or lasted at
                      least MIN_TURN_S; otherwise it would be a tic.

Suppression, cheapest check first:
  * JARVIS_DRIVEN is set in the environment — the session was started by
    Jarvis himself (ClaudeSessionManager.build_command puts the variable on
    the launch line), and he already narrates those panes from the
    transcript; hearing every line twice is worse than hearing none.
  * /tmp/vss_voice/jarvis.pid absent or its pid dead — Jarvis is not
    running, so there is nobody to speak.
  * alerts.claude_hooks is false in ~/.config/jarvis/assistant.json (read
    with plain json; no jarvis import).
  * the same line was spoken for this session less than REPEAT_S ago (idle
    notifications repeat on their own).

Paths follow the app's own defaults and env overrides (jarvis/config.py):
JARVIS_LOG_DIR (speak queue + pid file), JARVIS_ASSISTANT_CONFIG,
JARVIS_CACHE_DIR; JARVIS_HOOK_STATE_DIR pins the per-session state directory.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# --- copied from jarvis/claude_session.py; the drift test pins them -------
TEST_CMD_RX = re.compile(
    r"(?:^|[\s;&|(])(?:python3?\s+-m\s+)?(?:pytest|npm\s+(?:run\s+)?test|"
    r"pnpm\s+test|yarn\s+test|cargo\s+test|go\s+test|make\s+test|ruff|mypy|"
    r"jest|vitest|tox)\b")
_PASSED_RX = re.compile(r"\b(\d+)\s+passed\b")
_FAILED_RX = re.compile(r"\b(\d+)\s+failed\b")
_ERRORS_RX = re.compile(r"\b(\d+)\s+errors?\b")


def _test_outcome(text: str) -> Optional[str]:
    """Spoken verdict for a test / lint run's output, or None."""
    text = str(text or "")
    failed = _FAILED_RX.search(text)
    if failed and int(failed.group(1)) > 0:
        n = int(failed.group(1))
        return f"{n} test{'s' if n != 1 else ''} failed, sir."
    errors = _ERRORS_RX.search(text)
    if errors and int(errors.group(1)) > 0 and not _PASSED_RX.search(text):
        return "The tests errored, sir."
    if _PASSED_RX.search(text) or re.search(r"\bAll checks passed\b|\bSuccess: no issues\b", text):
        return "Tests passed, sir."
    if re.search(r"\bFAILED\b|\bTraceback\b|\berror\b", text):
        return "The tests errored, sir."
    return None
# --------------------------------------------------------------------------


DRIVEN_ENV = "JARVIS_DRIVEN"
LIVE_LOG_DIR = "/tmp/vss_voice"             # jarvis.config.PATHS.LOG_DIR default
STATE_SUBDIR = "claude_hooks"

MIN_TURN_S = 45.0        # a Stop this soon after the prompt is a chat reply, not a job
REPEAT_S = 30.0          # the same verdict twice in half a minute is noise
WAITING_REPEAT_S = 120.0  # idle_prompt re-fires on its own while he waits
STATE_MAX_AGE_S = 7 * 24 * 3600.0   # per-session state files older than this are dropped

WAITING_LINE = "Claude is waiting on you in {project}, sir."
PERMISSION_LINE = "Claude needs your say-so in {project}, sir."
FINISHED_LINE = "Claude has finished in {project}, sir."
UNKNOWN_PROJECT = "the terminal"


# ------------------------------------------------------------------ paths
def log_dir() -> Path:
    return Path(os.environ.get("JARVIS_LOG_DIR") or LIVE_LOG_DIR)


def speak_queue_path() -> Path:
    return log_dir() / "speak_queue.txt"


def pid_path() -> Path:
    return log_dir() / "jarvis.pid"


def config_path() -> Path:
    return Path(os.environ.get("JARVIS_ASSISTANT_CONFIG") or
                (Path.home() / ".config" / "jarvis" / "assistant.json"))


def state_dir() -> Path:
    explicit = os.environ.get("JARVIS_HOOK_STATE_DIR")
    if explicit:
        return Path(explicit)
    cache = Path(os.environ.get("JARVIS_CACHE_DIR") or (Path.home() / ".cache" / "jarvis"))
    return cache / STATE_SUBDIR


# ------------------------------------------------------------------ gates
def jarvis_alive() -> bool:
    """True when the pid file names a live process.  A stale file after a
    crash must not queue lines that a later Jarvis would then read out as
    backlog (the Watcher skips pre-start content, but only once)."""
    try:
        pid = int(pid_path().read_text().strip())
    except (OSError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, just not ours to signal
    except OSError:
        return False
    return True


def hooks_enabled() -> bool:
    """alerts.claude_hooks from the assistant config; a missing or broken
    file means the default (on)."""
    try:
        data = json.loads(config_path().read_text())
    except (OSError, ValueError):
        return True
    alerts = data.get("alerts") if isinstance(data, dict) else None
    if not isinstance(alerts, dict) or "claude_hooks" not in alerts:
        return True
    return bool(alerts.get("claude_hooks"))


# ------------------------------------------------------------------ state
def _state_file(session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "") or "no-session"
    return state_dir() / f"{safe}.json"


def load_state(session_id: str) -> dict:
    try:
        data = json.loads(_state_file(session_id).read_text())
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def save_state(session_id: str, state: dict, now: float) -> None:
    d = state_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        _state_file(session_id).write_text(json.dumps(state))
    except OSError:
        return
    # One tiny file per Claude session would otherwise pile up forever.
    try:
        for f in d.glob("*.json"):
            if now - f.stat().st_mtime > STATE_MAX_AGE_S:
                f.unlink(missing_ok=True)
    except OSError:
        pass


# ------------------------------------------------------------------ event
def project_name(event: dict) -> str:
    cwd = str(event.get("cwd") or "").strip()
    name = Path(cwd).name if cwd else ""
    # "haymaker-digest" reads as a hyphen in TTS; "haymaker digest" is what
    # he would say for the managed sessions.
    name = re.sub(r"[-_]+", " ", name).strip()
    return name or UNKNOWN_PROJECT


def _response_text(resp) -> str:
    """The Bash tool's output, whatever shape the CLI hands it in: a dict
    with stdout/stderr, a content list of text blocks, or a bare string."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        return "\n".join(_response_text(x) for x in resp)
    if isinstance(resp, dict):
        parts = []
        for key in ("stdout", "stderr", "output", "text", "content"):
            if key in resp:
                parts.append(_response_text(resp[key]))
        return "\n".join(p for p in parts if p)
    return str(resp)


def _with_project(verdict: str, project: str) -> str:
    """'3 tests failed, sir.' -> '3 tests failed in jarvis, sir.' — the
    project matters when several of your own sessions are running."""
    if verdict.endswith(", sir."):
        return f"{verdict[:-len(', sir.')]} in {project}, sir."
    return verdict


def line_for(event: dict, state: dict, now: float) -> Optional[str]:
    """The spoken line for this hook event, or None.  Mutates `state`
    (turn timing, test-run flag) but does NOT apply the repeat limiter;
    main() does that so this stays easy to test."""
    kind = str(event.get("hook_event_name") or "")
    project = project_name(event)

    if kind == "UserPromptSubmit":
        state["turn_started"] = now
        state["tested"] = False
        return None

    if kind == "PostToolUse":
        if str(event.get("tool_name") or "") != "Bash":
            return None
        inp = event.get("tool_input") or {}
        command = str(inp.get("command") or "") if isinstance(inp, dict) else ""
        if not TEST_CMD_RX.search(command):
            return None
        state["tested"] = True
        verdict = _test_outcome(_response_text(event.get("tool_response")))
        return _with_project(verdict, project) if verdict else None

    if kind == "Notification":
        ntype = str(event.get("notification_type") or "")
        message = str(event.get("message") or "").lower()
        if ntype == "permission_prompt" or "permission" in message:
            return PERMISSION_LINE.format(project=project)
        if ntype in ("idle_prompt", "elicitation_dialog", "") or "waiting" in message:
            return WAITING_LINE.format(project=project)
        return None                       # auth_success and the like

    if kind == "Stop":
        if event.get("stop_hook_active"):
            return None                   # a Stop hook already resumed Claude once
        started = state.pop("turn_started", None)
        tested = bool(state.pop("tested", False))
        long_enough = isinstance(started, (int, float)) and (now - float(started)) >= MIN_TURN_S
        if tested or long_enough:
            return FINISHED_LINE.format(project=project)
        return None

    return None


def _repeat_window(line: str) -> float:
    return WAITING_REPEAT_S if line.startswith("Claude is waiting") else REPEAT_S


def append_line(line: str) -> None:
    path = speak_queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")


# ------------------------------------------------------------------ main
def main(argv=None, stdin=None, now: Optional[float] = None) -> int:
    """Always returns 0: a narration hook must never block or fail the CLI."""
    try:
        return _main(stdin if stdin is not None else sys.stdin,
                     time.time() if now is None else now)
    except Exception:
        return 0


def _main(stdin, now: float) -> int:
    if os.environ.get(DRIVEN_ENV):
        return 0
    if not jarvis_alive():
        return 0
    if not hooks_enabled():
        return 0
    try:
        event = json.load(stdin)
    except (ValueError, OSError):
        return 0
    if not isinstance(event, dict):
        return 0
    # A background agent's session (a workflow builder, a spawned subagent)
    # is not a terminal Hunter is watching: on 2026-08-30 eight builders'
    # pytest runs narrated "Tests passed in wf..., sir" to the soundbar at
    # 11 pm. Their transcripts live under .../subagents/; a session whose
    # hook payload points there is never narrated.
    if "/subagents/" in str(event.get("transcript_path") or ""):
        return 0

    session_id = str(event.get("session_id") or "")
    state = load_state(session_id)
    line = line_for(event, state, now)
    if line:
        last = state.get("last") if isinstance(state.get("last"), dict) else {}
        stamp = last.get(line)
        if not (isinstance(stamp, (int, float)) and now - float(stamp) < _repeat_window(line)):
            append_line(line)
            last[line] = now
        # keep the limiter table small
        state["last"] = {k: v for k, v in last.items()
                         if isinstance(v, (int, float)) and now - float(v) < 3600.0}
    save_state(session_id, state, now)
    return 0


if __name__ == "__main__":
    sys.exit(main())
