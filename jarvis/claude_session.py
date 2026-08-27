"""Claude session manager — Jarvis drives the real Claude Code TUI in tmux.

Spec: docs/specs/2026-08-26-jarvis-personal-assistant.md, section 7, as
amended by the 2026-08-27 "live Claude terminal" decision.

One tmux session per project (`jarvis-<slug>`), one task at a time per
project (others queue), a second project in parallel only when asked.

The pane runs Claude **interactively** — the genuine interface, thinking
and tool rendering included — because `claude -p --output-format
stream-json` is a machine event stream and can only ever look like a flat
log (that was the user's complaint).  A task is therefore a prompt typed
into that session with `tmux send-keys`, and Jarvis's own progress comes
from the transcript Claude Code writes as it works,
`~/.claude/projects/<encoded-cwd>/<session-id>.jsonl` — the same files
`discover_sessions()` reads.  Milestones become `ClaudeProgress` events
(the app speaks the `milestone=True` ones), the lifecycle becomes
`ClaudeTaskState`, and the final assistant message of the turn becomes
`Task.result_text` for the spoken two-sentence summary.

The pop-out attaches READ-ONLY (`tmux attach -r`): keystrokes in that
window cannot drive or derail the session — Jarvis stays the input path,
and `terminal_open()` tells the router and the UI when someone is watching.

Permissions: inside the project everything is pre-approved by the
`.claude/settings.local.json` this module writes (7.3); everything else
reaches the MCP tool `mcp__jarvis__approve` (jarvis/mcp_permissions.py)
which asks the app's ApprovalBroker (jarvis/approvals.py).

Threads: every public method returns quickly; runners are daemon threads
that publish on the bus.  Speech is the app's job — this module only
returns persona strings or publishes events.

Subcommand: `python -m jarvis.claude_session render --out <file>` — the
print-mode renderer, kept for offline replay of an old `-p` stream; the
live pane no longer uses it.
"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from jarvis.events import (ActiveProject, ApprovalRequested, ApprovalResolved,
                           ClaudeProgress, ClaudeTaskState, bus)
from jarvis.logs import get_logger

log = get_logger("claude_session")

REPO_ROOT = Path(__file__).resolve().parents[1]
PERMISSION_TOOL = "mcp__jarvis__approve"
MODEL_ALIASES = ("opus", "sonnet", "fable", "haiku")
DEFAULT_MODEL = "opus"
DEFAULT_BIG_MODEL = "fable"
DEFAULT_PROJECTS_ROOT = str(Path.home() / "projects")
RESUME_PROMPT = ("Pick up where we left off; summarise the state in two "
                 "sentences first, then continue.")
SYSTEM_SUFFIX = ("You are being driven by Jarvis, Hunter's voice assistant. "
                 "Your final message is read aloud: end with one or two plain-"
                 "prose sentences saying what you did and anything he must "
                 "decide.")

# ------------------------------------------------------------ lines (3.4)
BUSY_LINE = "Claude's still on the last one for {project}, sir; I've queued it."
CANCELLED_LINE = "Stopped, sir."
NO_PROJECT_LINE = ("I don't have a project to work in, sir; name one, or say "
                   "'start a new project'.")
NO_SESSION_LINE = "I can't find a session to pick up, sir."
OUTSIDE_LINE = ("That folder isn't one you've cleared for me, sir, so I'd "
                "rather not touch it. Say the word and I'll open the terminal "
                "there instead.")
SETUP_LINE_FALLBACK = ("I'll need the Claude command line set up, sir; the "
                       "notes are in docs/assistant-setup.md.")
UNSAFE_DIR_LINE = ("That's your home or a configuration folder, sir, not a "
                   "project, so I'd rather not turn Claude loose in it. Say "
                   "the word and I'll open the terminal there instead.")
NO_RESULT_LINE = "Claude stopped without a result, sir."

# ------------------------------------------------- the interactive pane
# VERIFIED live on tmux 3.4 / Claude Code 2.1.247, 2026-08-27:
#  * `tmux new-session -d -x -y` is IGNORED while no client is attached —
#    the window stays 80x24 (`window-size latest` falls back to
#    `default-size`) and the TUI wraps into soup.  `window-size manual`
#    plus `resize-window` is what actually takes.
#  * a first interactive run in a folder shows the workspace-trust dialog
#    ("Quick safety check … 1. Yes, I trust this folder"), which a
#    read-only pane could never answer; `_await_ready` accepts it only for
#    a project the user has already cleared for Claude.
#  * the ready TUI draws an `❯` input line and a "shift+tab to cycle"
#    footer; while a turn runs the footer carries "esc to interrupt".
TMUX_COLS, TMUX_ROWS = 200, 50
TRUST_RX = re.compile(r"trust this folder|Quick safety check", re.I)
SETTINGS_RX = re.compile(r"Settings Warning", re.I)
# a choice menu: "❯ 1. Continue" with "Enter to confirm" underneath.  It
# must NOT be read as the input box, which is a bare `❯` between two rules.
MENU_RX = re.compile(r"^\s*❯\s*\d+\.", re.M)
READY_RX = re.compile(r"^\s*❯\s*$|shift\+tab to cycle", re.M)
BUSY_PANE_RX = re.compile(r"esc to interrupt", re.I)
# control characters never reach send-keys: a prompt is text, not keys
KEY_STRIP_RX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# ------------------------------- the CLI permission contract, VERIFIED 7.3
# Verified live on 2026-08-27 against Claude Code **2.1.247** by driving real
# `claude -p` runs against this module's own MCP server and ApprovalBroker.
# These are findings, not assumptions; re-check them when the CLI major moves.
#
#   1. `--permission-prompt-tool` EXISTS in 2.1.247 and takes the MCP tool
#      name as `mcp__<server>__<tool>` (`mcp__jarvis__approve`).  It is hidden
#      from `--help`, but it parses: an unknown flag dies with "unknown
#      option", this one gets through to the prompt check.
#   2. `--mcp-config <file>` ACCEPTS the `{"mcpServers": {...}}` shape written
#      by `write_mcp_config()`.  The init event reports
#      `mcp_servers: [{"name": "jarvis", "status": "connected"}]`.
#   3. The tool-result payload assumed by mcp_permissions.permission_result()
#      is CORRECT: ONE text block holding json.dumps of
#      {"behavior": "allow", "updatedInput": <input>} — the tool then runs for
#      real — or {"behavior": "deny", "message": "..."}, which comes back as
#      an is_error tool_result carrying our message verbatim, tagged
#      `non_execution_kind: "permission-rule"`.  Both end the task rc 0.
#   4. `--permission-mode acceptEdits` DOES still route out-of-project edits
#      to the prompt tool, and only those: a Write inside the project cwd is
#      auto-accepted and the tool never fires (probe: asked=0), while a Write
#      outside it fires the tool every time.  That is exactly the split we
#      want — no spoken question for ordinary in-project work.
#   5. The MCP subprocess DOES inherit the `env` block (JARVIS_APPROVAL_SOCK,
#      JARVIS_PROJECT, PYTHONPATH): the server reached the broker over the
#      socket and imported `jarvis` by that PYTHONPATH.
#   6. Extra finding: the broad `Write(/**)`-style rules in ALLOW_RULES do NOT
#      suppress the prompt tool for out-of-project paths — the question still
#      fires with them present, so the allow list and the ask-aloud path
#      coexist and ALLOW_RULES needs no project scoping.
#
# NOTE (user decision 2026-08-26): work is NOT restricted to
# `claude.allowed_dirs` — `claude.auto_approve_anywhere` defaults to true, so
# `submit()` accepts any project and writes the same broad allow rules into
# that project's `.claude/settings.local.json`.  With that flag off, an
# outside dir is no longer refused outright: it runs under the prompt tool and
# every out-of-project action becomes a spoken question.  OUTSIDE_LINE now
# means only "and the prompt tool is unavailable" — a real fallback, not a
# silent one.

# settings.local.json (7.3).  Bare `Edit` / `Write` rules would grant writes
# for ANY path (the CLI warns exactly that) and the outside-dir approval
# flow would never fire, so the file tools are workspace-scoped with `/**`.
# `mcp__*` was invalid and is now poison: CLI 2.1.247 refuses a bare
# wildcard tool name in an allow rule ("an allow pattern must name the scope
# it widens"), and INTERACTIVELY it stops on a modal Settings Warning — a
# dialog the read-only pane could never answer.  Verified live 2026-08-27.
# The valid form names the server, and STALE_ALLOW_RULES prunes the bad one
# out of the files an older Jarvis already wrote.
ALLOW_RULES = ["Bash(*)", "Read(/**)", "Edit(/**)", "Write(/**)",
               "MultiEdit(/**)", "NotebookEdit(/**)", "Glob", "Grep",
               "WebFetch", "WebSearch", "Task", "TodoWrite", "mcp__jarvis__*"]
STALE_ALLOW_RULES = ("mcp__*",)
DENY_RULES = ["Bash(sudo *)", "Bash(rm -rf /*)", "Bash(rm -rf ~*)"]

NEVER = float("-inf")           # "no milestone yet" for the 20 s limiter

EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
TEST_CMD_RX = re.compile(
    r"(?:^|[\s;&|(])(?:python3?\s+-m\s+)?(?:pytest|npm\s+(?:run\s+)?test|"
    r"pnpm\s+test|yarn\s+test|cargo\s+test|go\s+test|make\s+test|ruff|mypy|"
    r"jest|vitest|tox)\b")
PLAN_HEADING_RX = re.compile(r"^\s*(?:#{1,3}\s*)?(?:implementation\s+|the\s+)?plan\b\s*[:\-—]?",
                             re.I)
_PASSED_RX = re.compile(r"\b(\d+)\s+passed\b")
_FAILED_RX = re.compile(r"\b(\d+)\s+failed\b")
_ERRORS_RX = re.compile(r"\b(\d+)\s+errors?\b")
_SHELLS = frozenset({"bash", "zsh", "sh", "fish", "dash"})
# NOTE (2026-08-26): "jarvis" is deliberately NOT a stopword.  It is the name
# of the user's own repo (the first entry in the shipped claude.allowed_dirs),
# so dropping it made ~/Jarvis unnameable — "pick up the jarvis session" scored
# 0 on the name match and the wrong session resumed.  The *address* form
# ("Jarvis, pick up ...") is stripped by _ADDRESS_RX below instead.
_STOPWORDS = frozenset("""
a an and are at be by continue could did do for from get go i in is it its
last left me my of off on one our pick please project projects resume session
that the then there this to up us was we were what where which with work
working yesterday today week you your claude let lets keep going back
""".split())
# The vocative, not the project name: "hey jarvis, ...", "jarvis: ...",
# "..., jarvis".  A bare mid-sentence "jarvis" ("the jarvis one") is a NAME.
_ADDRESS_RX = re.compile(
    r"^\s*(?:hey|ok|okay)\s+jarvis\b[\s,:;.!?-]*"      # "hey jarvis, ..."
    r"|^\s*jarvis\b\s*[,:;.!?-]+\s*"                   # "jarvis, ..."
    r"|[,;]\s*jarvis\b\s*[.!?]*\s*$",                  # "..., jarvis"
    re.I)
_DATE_WORDS = ("yesterday", "today", "last week", "this week", "last night",
               "this morning")


# ------------------------------------------------------------------- seam
def _run(argv, timeout: float = 15.0, input: Optional[str] = None,
         cwd: Optional[str] = None, env: Optional[dict] = None):
    """Subprocess seam (tests replace it with a recorder).  Never raises;
    a missing binary is rc 127, a timeout rc 124."""
    try:
        return subprocess.run(list(argv), capture_output=True, text=True,
                              timeout=timeout, input=input, cwd=cwd, env=env)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(list(argv), 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(list(argv), 124,
                                           exc.stdout or "", "timeout")
    except OSError as exc:
        return subprocess.CompletedProcess(list(argv), 126, "", str(exc))


# ------------------------------------------------------------- dataclasses
@dataclass
class Project:
    slug: str
    path: str
    session_id: str = ""
    model: str = ""
    last_used: float = 0.0

    @property
    def display(self) -> str:
        return display_name(self.slug)


@dataclass
class Task:
    task_id: str
    project: str
    prompt: str
    model: str
    state: str = "queued"          # queued | running | waiting | done | failed | cancelled
    started: float = 0.0
    files_touched: set = field(default_factory=set)
    result_text: str = ""
    rc: Optional[int] = None
    # -- bookkeeping beyond the spec's fields (defaults keep it compatible)
    cwd: str = ""
    session_id: str = ""
    parallel: bool = False
    finished: float = 0.0
    turns: int = 0
    error: str = ""
    events: int = 0
    attempt: int = 1
    paths: dict = field(default_factory=dict)
    tool_uses: dict = field(default_factory=dict)      # tool_use id -> (name, input)
    test_ids: set = field(default_factory=set)
    error_streak: int = 0
    pending_files: list = field(default_factory=list)
    # -inf so the FIRST milestone of a task is never eaten by the 20 s
    # limiter (a task that starts at now=1 would otherwise be silenced).
    last_edit_milestone: float = NEVER
    last_milestone: float = NEVER

    @property
    def final(self) -> bool:
        return self.state in ("done", "failed", "cancelled")


@dataclass
class Pane:
    """The interactive Claude living in one project's tmux window."""
    slug: str
    session_id: str = ""       # --session-id we launched with (or resumed)
    transcript: str = ""       # ~/.claude/projects/<enc>/<id>.jsonl
    offset: int = 0            # bytes of it that predate the current task
    started: float = 0.0


@dataclass
class SessionInfo:
    session_id: str
    cwd: str
    mtime: float
    first_user: str = ""
    last_assistant: str = ""
    turns: int = 0
    path: str = ""
    title: str = ""
    score: float = 0.0

    @property
    def slug(self) -> str:
        return slugify(os.path.basename(self.cwd.rstrip("/")))


# ------------------------------------------------------------ pure helpers
def slugify(name: str) -> str:
    """lowercase [a-z0-9-] from a spoken name (7.6)."""
    text = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower())
    return text.strip("-")


def display_name(slug: str) -> str:
    return str(slug or "").replace("-", " ").replace("_", " ").strip()


def _cfg_get(cfg, dotted: str, default=None):
    if cfg is None:
        return default
    get = getattr(cfg, "get", None)
    if callable(get):
        try:
            value = get(dotted, default)
        except Exception:                            # noqa: BLE001 - config shim
            value = default
        return default if value is None else value
    if isinstance(cfg, dict):
        node = cfg
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node
    return default


def _allowed_dirs(cfg) -> list[str]:
    dirs = getattr(cfg, "allowed_dirs", None)
    if callable(dirs):
        dirs = dirs()
    if dirs is None:
        dirs = _cfg_get(cfg, "claude.allowed_dirs", []) or []
    return [os.path.abspath(os.path.expanduser(str(d))) for d in dirs
            if isinstance(d, str) and d.strip()]


def _same_dir(a: str, b: str) -> bool:
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except OSError:
        return False


# --------------------------------------------------- unsafe dirs (C2, 7.3)
# `.claude/settings.local.json` is a PROJECT file in a project — but in $HOME
# it is Claude Code's USER-LEVEL allowlist, so writing ALLOW_RULES there would
# hand Bash(*)/Edit(/**)/Write(/**)/mcp__* to every future interactive Claude
# session on this machine, permanently.  These dirs are therefore never
# treated as projects.  Ordinary project directories ANYWHERE on disk stay
# allowed (claude.auto_approve_anywhere, user decision 2026-08-26) — this is
# only about home, config roots and system dirs.
_SYSTEM_ROOTS = ("/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/lib32",
                 "/opt", "/boot", "/proc", "/sys", "/dev", "/var", "/srv",
                 "/run", "/snap", "/mnt", "/media", "/root")
_SCRATCH_DIRS = ("/tmp", "/var/tmp", "/dev/shm")
# Any one of these makes a directory a plausible project (7.3); a directory
# with none of them is still allowed unless it trips a rule above — the marker
# list is what `looks_like_project()` reports, not a second gate.
PROJECT_MARKERS = (".git", ".hg", "pyproject.toml", "package.json", "CLAUDE.md",
                   "setup.py", "setup.cfg", "Cargo.toml", "go.mod", "Makefile",
                   "requirements.txt")


def unsafe_dir_reason(path, home=None) -> str:
    """Why Claude must not be given this directory, or "" when it is fine.

    Refuses $HOME itself, anything $HOME sits inside (`/`, `/home`), any
    dotted configuration directory (`~/.claude`, `~/.config/...`), the system
    roots, the scratch roots, and an existing directory the user does not own.
    Everything else — a real project dir anywhere on disk — is fine, including
    one that does not exist yet (that is the caller's problem, not a
    permissions one).
    """
    home_dir = os.path.realpath(os.path.expanduser(str(home or Path.home())))
    raw = str(path or "").strip()
    if not raw:
        return "an empty path"
    try:
        real = os.path.realpath(os.path.expanduser(raw))
    except OSError:
        return "a path I can't resolve"
    # Shape first, so a path that does not exist yet is judged the same way.
    if real == home_dir:
        return "your home folder itself"
    if _under(home_dir, real):
        return "a folder your home folder lives in"
    for part in Path(real).parts[1:]:
        if part.startswith("."):
            return "a configuration folder"
    for root in _SYSTEM_ROOTS:
        if real == root or real.startswith(root + "/"):
            return "a system folder"
    if real in _SCRATCH_DIRS:
        return "a scratch folder"
    if not os.path.isdir(real):
        return ""        # nothing there to protect; the caller's own problem
    getuid = getattr(os, "getuid", None)
    if getuid is not None:
        try:
            if os.stat(real).st_uid != getuid():
                return "a folder you don't own"
        except OSError:
            return "a folder I can't read"
    return ""


def looks_like_project(path) -> bool:
    """True when the directory carries one of PROJECT_MARKERS (advisory)."""
    try:
        return any((Path(path) / m).exists() for m in PROJECT_MARKERS)
    except OSError:
        return False


def _under(path: str, root: str) -> bool:
    try:
        real, top = os.path.realpath(path), os.path.realpath(root)
    except OSError:
        return False
    return real == top or real.startswith(top.rstrip("/") + "/")


def _short_path(path: str) -> str:
    home = str(Path.home())
    if path == home:
        return "~"
    if path.startswith(home + "/"):
        return "~/" + path[len(home) + 1:]
    return path


def _rel(path: str, cwd: str) -> str:
    path = str(path or "")
    if cwd and (path == cwd or path.startswith(cwd.rstrip("/") + "/")):
        return path[len(cwd.rstrip("/")) + 1:] or "."
    return _short_path(path)


def _excerpt(text, n: int = 90) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def _first_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"^#+\s*", "", text)
    m = re.search(r"[.!?](\s|$)", text)
    return text[:m.end()].strip() if m else _excerpt(text, 120)


def _join_names(names: list[str]) -> str:
    names = list(dict.fromkeys(names))
    if len(names) > 3:
        names = names[:3] + [f"{len(names) - 3} more"]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _ago(seconds: float) -> str:
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven", "twelve", "thirteen",
             "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
             "nineteen"]
    s = max(0, int(seconds))
    if s < 60:
        return "just now"
    m = s // 60
    if m < 60:
        if m == 1:
            return "a minute ago"
        return f"{words[m] if m < 20 else m} minutes ago"
    h = m // 60
    if h == 1:
        return "an hour ago"
    return f"{words[h] if h < 20 else h} hours ago"


_OBJECT_WORDS = ("code", "function", "class", "module", "file", "files",
                 "test", "tests", "bug", "feature", "branch", "repo", "pr",
                 "script", "error", "traceback", "package", "dependency",
                 "config", "pipeline", "dataset", "training", "docker",
                 "service", "endpoint", "database", "schema", "migration")
_LARGE_CUES = re.compile(
    r"\b(refactor|feature|across|all the|every file|migrate|migration|"
    r"multi-file|end to end|end-to-end|rewrite|redesign|overhaul|"
    r"this is a big one|think hard)\b|\bdebug\b.*\band\b", re.I)


def estimate_size(prompt: str) -> str:
    """'small' | 'large' (7.1): escalate on the router's cue tables."""
    text = str(prompt or "")
    low = text.lower()
    if _LARGE_CUES.search(low):
        return "large"
    tokens = re.findall(r"[a-z]+", low)
    objects = {t for t in tokens if t in _OBJECT_WORDS}
    if len(objects) >= 2:
        return "large"
    if len(tokens) > 60:
        return "large"
    return "small"


# ------------------------------------------------ stream events (7.4)
def _tool_line(name: str, inp: dict, cwd: str = "") -> str:
    inp = inp if isinstance(inp, dict) else {}
    path = inp.get("file_path") or inp.get("notebook_path") or inp.get("path") or ""
    if name in EDIT_TOOLS:
        verb = "Write" if name == "Write" else "Edit"
        return f"{verb} {_rel(str(path), cwd)}" if path else verb
    if name == "Read":
        return f"Read {_rel(str(path), cwd)}" if path else "Read"
    if name == "Bash":
        return f"Bash {_excerpt(inp.get('command', ''), 80)}"
    if name in ("Glob", "Grep"):
        return f"Search \"{_excerpt(inp.get('pattern', ''), 60)}\""
    if name == "WebFetch":
        return f"Fetch {_excerpt(inp.get('url', ''), 80)}"
    if name == "WebSearch":
        return f"Search web \"{_excerpt(inp.get('query', ''), 60)}\""
    if name == "Task":
        return f"Agent {_excerpt(inp.get('description') or inp.get('prompt', ''), 70)}"
    if name == "TodoWrite":
        todos = inp.get("todos")
        return f"Todo {len(todos)} items" if isinstance(todos, list) else "Todo"
    if name == "ExitPlanMode":
        return "Plan ready"
    if name == "AskUserQuestion":
        qs = inp.get("questions")
        q = qs[0].get("question", "") if isinstance(qs, list) and qs and isinstance(qs[0], dict) else ""
        return f"Question {_excerpt(q, 70)}"
    for value in inp.values():
        if isinstance(value, str) and value.strip():
            return f"{name} {_excerpt(value, 60)}"
    return name


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return "" if content is None else str(content)


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


def parse_stream_event(event, task: Task, now: Optional[float] = None) -> list:
    """One event -> compact `ClaudeProgress` lines (pure: no I/O; only the
    task's own bookkeeping fields change).  Tolerant of unknown types and
    missing keys; never raises."""
    now = time.time() if now is None else now
    out: list[tuple[str, bool, bool]] = []      # (line, milestone, exempt)
    try:
        _parse(event, task, now, out)
    except Exception:                            # noqa: BLE001 - never break the tail
        log.exception("parse_stream_event failed on %r", str(event)[:200])
    task.events += 1
    lines = []
    for line, milestone, exempt in out:
        if milestone and not exempt:
            if now - task.last_milestone < 20.0:
                milestone = False
            else:
                task.last_milestone = now
        lines.append(ClaudeProgress(project=task.project, task_id=task.task_id,
                                    line=line, milestone=milestone))
    return lines


def _parse(event, task: Task, now: float, out: list):
    if not isinstance(event, dict):
        return
    etype = event.get("type")
    if etype == "system":
        if event.get("subtype") == "init":
            task.session_id = str(event.get("session_id") or task.session_id)
            model = str(event.get("model") or task.model or "")
            out.append((f"Started · {model}" if model else "Started", False, False))
        return
    if etype == "assistant":
        msg = event.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for block in content or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = str(block.get("text") or "")
                if not text.strip():
                    continue
                if PLAN_HEADING_RX.match(text):
                    body = PLAN_HEADING_RX.sub("", text, count=1)
                    out.append((f"Plan chosen: {_first_sentence(body)}", True, False))
                else:
                    out.append((_excerpt(text, 90), False, False))
            elif btype == "tool_use":
                _parse_tool_use(block, task, now, out)
        return
    if etype == "user":
        msg = event.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else None
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                _parse_tool_result(block, task, out)
        return
    if etype == "result":
        is_error = bool(event.get("is_error")) or \
            str(event.get("subtype", "")).startswith("error")
        text = _result_text(event.get("result"))
        task.turns = int(event.get("num_turns") or task.turns or 0)
        task.session_id = str(event.get("session_id") or task.session_id)
        if is_error:
            task.rc = 1
            task.error = text or str(event.get("subtype") or "error")
            errors = event.get("errors")
            if not text and isinstance(errors, list) and errors:
                task.error = _result_text(errors[0]) or task.error
            out.append((f"Claude stopped, sir: {_excerpt(task.error, 80)}", True, True))
        else:
            task.rc = 0
            task.result_text = text
            dur = event.get("duration_ms")
            secs = f" · {int(dur) // 1000} s" if isinstance(dur, (int, float)) else ""
            out.append((f"Done · {task.turns} turns{secs}", False, False))
        return
    # stream_event, rate_limit_event, tool_progress, …: silent


def _parse_tool_use(block: dict, task: Task, now: float, out: list):
    name = str(block.get("name") or "tool")
    inp = block.get("input") if isinstance(block.get("input"), dict) else {}
    tid = str(block.get("id") or "")
    if tid:
        task.tool_uses[tid] = (name, inp)
    line = _tool_line(name, inp, task.cwd)
    if name in EDIT_TOOLS:
        path = str(inp.get("file_path") or inp.get("notebook_path") or "")
        new = bool(path) and path not in task.files_touched
        if path:
            task.files_touched.add(path)
        if new:
            task.pending_files.append(os.path.basename(path))
        if task.pending_files and now - task.last_edit_milestone >= 20.0:
            names = _join_names(task.pending_files)
            task.pending_files = []
            task.last_edit_milestone = now
            out.append((line, False, False))
            out.append((f"Editing {names}, sir.", True, False))
            return
        out.append((line, False, False))
        return
    if name == "Bash":
        out.append((line, False, False))
        if TEST_CMD_RX.search(str(inp.get("command") or "")):
            if tid:
                task.test_ids.add(tid)
            out.append(("Running the tests, sir.", True, False))
        return
    if name == "ExitPlanMode":
        plan = str(inp.get("plan") or "")
        body = PLAN_HEADING_RX.sub("", plan, count=1)   # drop a "# Plan" heading
        out.append((f"Plan chosen: {_first_sentence(body or plan)}"
                    if plan.strip() else "Plan chosen, sir.", True, False))
        return
    if name == "AskUserQuestion":
        out.append((line, False, False))
        out.append((f"Claude has a question, sir: {line[9:] if line.startswith('Question ') else 'see the terminal'}",
                    True, True))
        return
    out.append((line, False, False))


def _parse_tool_result(block: dict, task: Task, out: list):
    tid = str(block.get("tool_use_id") or "")
    name, _inp = task.tool_uses.get(tid, ("tool", {}))
    text = _result_text(block.get("content"))
    if block.get("is_error"):
        # A test run that FAILS exits non-zero, and the CLI marks every
        # non-zero Bash result is_error — so the verdict ("2 tests failed,
        # sir.") lives on this path, not the clean one.  Returning here made
        # a failing suite silent (the app only speaks milestones).
        if tid in task.test_ids:
            verdict = _test_outcome(text)
            if verdict:
                task.test_ids.discard(tid)
                out.append((f"Error: {_excerpt(text or name, 90)}", False, False))
                out.append((verdict, True, True))
                return                  # a failed suite is work, not a tool fault
        task.error_streak += 1
        out.append((f"Error: {_excerpt(text or name, 90)}", False, False))
        if task.error_streak == 3:
            out.append(("Hitting errors, sir; carrying on.", True, True))
        return
    task.error_streak = 0
    if tid in task.test_ids:
        task.test_ids.discard(tid)
        verdict = _test_outcome(text)
        if verdict:
            out.append((verdict, True, True))
        else:
            out.append((f"Tests: {_excerpt(text.strip().splitlines()[-1] if text.strip() else 'done', 80)}",
                        False, False))


def describe_event(event, cwd: str = "") -> list[str]:
    """Readable pane lines for the pop-out terminal (stateless)."""
    if not isinstance(event, dict):
        return []
    etype = event.get("type")
    if etype == "system" and event.get("subtype") == "init":
        return [f"── session {event.get('session_id', '')} · {event.get('model', '')} ──"]
    if etype == "assistant":
        msg = event.get("message") or {}
        lines = []
        for block in (msg.get("content") if isinstance(msg, dict) else None) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and str(block.get("text", "")).strip():
                lines.append(str(block.get("text")).rstrip())
            elif block.get("type") == "tool_use":
                lines.append("▸ " + _tool_line(str(block.get("name") or "tool"),
                                               block.get("input") or {}, cwd))
        return lines
    if etype == "user":
        msg = event.get("message") or {}
        lines = []
        for block in (msg.get("content") if isinstance(msg, dict) else None) or []:
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error"):
                lines.append("  ✗ " + _excerpt(_result_text(block.get("content")), 160))
        return lines
    if etype == "result":
        bad = bool(event.get("is_error")) or str(event.get("subtype", "")).startswith("error")
        dur = event.get("duration_ms")
        secs = f"{int(dur) // 1000} s" if isinstance(dur, (int, float)) else "?"
        return [f"── {'FAILED' if bad else 'done'} · {event.get('num_turns', '?')} turns · {secs} ──"]
    return []


# --------------------------------------------- the live transcript (7.4b)
def encode_project_dir(cwd) -> str:
    """Claude Code's ~/.claude/projects/<dir> encoding: every character
    that is not a letter or a digit becomes '-'.  Verified against the
    directories on this machine ('/home/hunterp/vss_env' ->
    '-home-hunterp-vss-env', '/home/hunterp/.claude/projects/-home-hunterp/
    memory' -> '-home-hunterp--claude-projects--home-hunterp-memory')."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(cwd or ""))


def transcript_dir(cwd, projects_dir=None) -> Path:
    root = Path(projects_dir or (Path.home() / ".claude" / "projects"))
    return root / encode_project_dir(cwd)


def find_transcript(cwd, projects_dir=None, session_id: str = "",
                    since: float = 0.0) -> Optional[Path]:
    """The JSONL Claude Code is writing for a session in `cwd`: the file
    named by `session_id` when it exists, else the newest one in the
    project's directory that has been touched at or after `since`."""
    d = transcript_dir(cwd, projects_dir)
    if session_id:
        named = d / f"{session_id}.jsonl"
        if named.is_file():
            return named
    best, best_m = None, -1.0
    try:
        entries = list(d.glob("*.jsonl"))
    except OSError:
        return None
    for path in entries:
        try:
            m = path.stat().st_mtime
        except OSError:
            continue
        if m + 0.001 < since or m <= best_m:
            continue
        best, best_m = path, m
    return best


def sanitize_keys(text) -> str:
    """A prompt as one line of literal text for `tmux send-keys -l`: no
    newlines (Enter submits), no tabs (the TUI completes on them), no
    control characters at all."""
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = KEY_STRIP_RX.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def pane_state(text) -> str:
    """What `tmux capture-pane -p` says the pane is showing.

    'trust'    the workspace-trust dialog
    'settings' the Settings Warning dialog (an invalid rule was skipped)
    'menu'     some other blocking choice menu — we do not guess at those
    'ready'    the TUI is up and its input box takes text
    'starting' anything else: still booting, or already gone

    The three dialogs matter because the pane is READ-ONLY: a dialog is a
    question nobody watching can answer, so Jarvis answers the two it
    recognises and refuses to touch the rest."""
    text = str(text or "")
    if MENU_RX.search(text) or "Enter to confirm" in text:
        if TRUST_RX.search(text):
            return "trust"
        if SETTINGS_RX.search(text):
            return "settings"
        return "menu"
    if READY_RX.search(text):
        return "ready"
    return "starting"


def pane_working(text) -> bool:
    """True while the TUI shows a turn in flight ('esc to interrupt')."""
    return bool(BUSY_PANE_RX.search(str(text or "")))


def entry_turn_text(entry) -> Optional[str]:
    """The assistant text of a transcript entry that ENDS a turn (Claude
    has stopped and is waiting for input), else None.  A sidechain entry
    is a subagent's turn, never the session's."""
    if not isinstance(entry, dict) or entry.get("type") != "assistant":
        return None
    if entry.get("isSidechain"):
        return None
    msg = entry.get("message")
    if not isinstance(msg, dict):
        return None
    if str(msg.get("stop_reason") or "") not in ("end_turn", "stop_sequence"):
        return None
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    parts = [str(b.get("text") or "") for b in content or []
             if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p.strip()).strip()


def parse_transcript_entry(entry, task: Task, now: Optional[float] = None) -> list:
    """One transcript line -> `ClaudeProgress` lines.  The transcript's
    `assistant` / `user` records carry exactly the `message` shape that
    print mode streams, so the milestone rules (edits coalesced, test
    verdicts INCLUDING failures, plans, error streaks) are the same code;
    sidechains (subagents) and Claude Code's own bookkeeping records
    ('attachment', 'last-prompt', …) are dropped."""
    if not isinstance(entry, dict) or entry.get("isSidechain"):
        return []
    if entry.get("type") not in ("assistant", "user"):
        return []
    return parse_stream_event(entry, task, now)


# ------------------------------------------------ session discovery (7.5)
def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block.get("text") or "")
    return ""


def _has_tool_result(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def _is_meta_text(text: str) -> bool:
    t = text.lstrip()
    return (not t or t.startswith("<") or t.startswith("You are Jarvis")
            or t.startswith("Caveat:"))


def read_session(path, max_bytes: int = 40_000_000) -> Optional[SessionInfo]:
    """Parse one ~/.claude/projects/<cwd>/<id>.jsonl (None when unusable)."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return None
    info = SessionInfo(session_id=path.stem, cwd="", mtime=st.st_mtime,
                       path=str(path))
    turns = 0
    read = 0
    try:
        with open(path, "rb") as fh:
            for raw in fh:
                read += len(raw)
                if read > max_bytes:
                    break
                # cheap prefilter — tolerant of both compact and pretty
                # JSON separators (`"type":"user"` and `"type": "user"`)
                if not any(k in raw for k in (b'"user"', b'"assistant"',
                                              b'"ai-title"', b'"summary"')):
                    continue
                try:
                    d = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(d, dict):
                    continue
                t = d.get("type")
                if t == "ai-title":
                    info.title = str(d.get("aiTitle") or "")[:120]
                    continue
                if t == "summary":
                    info.title = info.title or str(d.get("summary") or "")[:120]
                    continue
                if d.get("isSidechain"):
                    continue
                if not info.cwd and isinstance(d.get("cwd"), str):
                    info.cwd = d["cwd"]
                if isinstance(d.get("sessionId"), str) and d["sessionId"]:
                    info.session_id = d["sessionId"]
                msg = d.get("message") if isinstance(d.get("message"), dict) else {}
                content = msg.get("content")
                if t == "user":
                    if _has_tool_result(content):
                        continue
                    text = _text_of(content)
                    if text.lstrip().startswith("You are Jarvis"):
                        return None      # one of Jarvis's own Tier-3 sessions
                    if _is_meta_text(text):
                        continue
                    turns += 1
                    if not info.first_user:
                        info.first_user = _excerpt(text, 120)
                elif t == "assistant":
                    text = _text_of(content)
                    if text.strip():
                        info.last_assistant = _excerpt(text, 120)
                    turns += 1
    except OSError:
        return None
    info.turns = turns
    if not info.cwd or not os.path.isdir(info.cwd):
        return None
    if turns <= 2 or not info.first_user:
        return None
    return info


def discover_sessions(projects_dir=None, limit: int = 40) -> list[SessionInfo]:
    """Most recent usable sessions on this machine, newest first (7.5)."""
    root = Path(projects_dir or (Path.home() / ".claude" / "projects"))
    try:
        files = [p for p in root.glob("*/*.jsonl") if p.is_file()]
    except OSError:
        return []
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    out: list[SessionInfo] = []
    for path in files[: max(limit * 4, 40)]:
        info = read_session(path)
        if info is not None:
            out.append(info)
        if len(out) >= limit:
            break
    return out


def _tokens(text: str) -> set[str]:
    text = _ADDRESS_RX.sub(" ", str(text or ""))
    return {t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in _STOPWORDS and len(t) >= 2}


def _name_tokens(cwd: str) -> set[str]:
    parts = [p for p in cwd.split("/") if p][-2:]
    toks = set()
    for p in parts:
        low = p.lower()
        toks.add(re.sub(r"[^a-z0-9]", "", low))
        toks.update(re.findall(r"[a-z0-9]+", low))
    return {t for t in toks if t}


def _day_word(mtime: float, now: float) -> str:
    then, today = datetime.fromtimestamp(mtime).date(), datetime.fromtimestamp(now).date()
    delta = (today - then).days
    if delta <= 0:
        return "today"
    if delta == 1:
        return "yesterday"
    if delta < 7:
        return then.strftime("%A")
    return f"the {then.day}{_ordinal(then.day)} of {then.strftime('%B')}"


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def session_summary(info: SessionInfo, now: Optional[float] = None) -> str:
    now = time.time() if now is None else now
    what = info.title or info.first_user
    what = _excerpt(what, 50) if what else ""
    tail = f", on {what}" if what else ""
    return f"{display_name(info.slug)}{tail}, from {_day_word(info.mtime, now)}"


def pick_session(utterance: str, sessions: list, now: Optional[float] = None):
    """(best, runner_up, question) per 7.5; question is None unless the top
    two are within 0.15 of each other."""
    now = time.time() if now is None else now
    if not sessions:
        return None, None, None
    low = str(utterance or "").lower()
    utoks = _tokens(low)
    today = datetime.fromtimestamp(now).date()
    for s in sessions:
        score = 0.0
        ntoks = _name_tokens(s.cwd)
        if utoks & ntoks or any(len(u) >= 4 and any(u in n for n in ntoks) for u in utoks):
            score += 0.6
        day = datetime.fromtimestamp(s.mtime).date()
        age_days = (now - s.mtime) / 86400.0
        if "yesterday" in low or "last night" in low:
            if day == today - timedelta(days=1):
                score += 0.4
        elif "today" in low or "this morning" in low:
            if day == today:
                score += 0.4
        elif "last week" in low or "this week" in low:
            if age_days <= 14:
                score += 0.4
        score += 0.3 * math.exp(-max(age_days, 0.0) / 2.0)
        blob = (s.first_user + " " + s.title + " " + s.last_assistant).lower()
        if any(len(u) >= 4 and u in blob for u in utoks):
            score += 0.2
        s.score = round(score, 4)
    ranked = sorted(sessions, key=lambda s: (s.score, s.mtime), reverse=True)
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    question = None
    if runner is not None and abs(best.score - runner.score) < 0.15:
        question = (f"Two candidates, sir: {session_summary(best, now)}; or "
                    f"{session_summary(runner, now)} — which?")
    return best, runner, question


# ------------------------------------------------------------- the manager
class ClaudeSessionManager:
    def __init__(self, cfg, brain, approvals, state_path, task_dir,
                 run: Callable = _run, claude_bin: Optional[str] = None,
                 python: str = sys.executable, projects_dir=None,
                 now: Callable[[], float] = time.time, poll_s: float = 0.25,
                 home=None, ready_s: float = 90.0, settle_s: float = 1.5,
                 idle_s: float = 300.0, clients_ttl_s: float = 1.0,
                 ready_grace_s: float = 5.0, key_wait_s: float = 3.0):
        self.cfg = cfg
        self.brain = brain
        self.approvals = approvals
        self.state_path = Path(state_path)
        self.task_dir = Path(task_dir)
        self._run = run
        if claude_bin is None:
            claude_bin = os.environ.get("JARVIS_CLAUDE_BIN") or ""
            if not claude_bin:
                try:
                    from jarvis.config import MACHINE
                    claude_bin = MACHINE.claude_bin
                except Exception:                    # noqa: BLE001 - detection is optional
                    claude_bin = shutil.which("claude") or ""
        self.claude_bin = claude_bin or ""
        self.python = python
        self.projects_dir = Path(projects_dir or (Path.home() / ".claude" / "projects"))
        # the root a discovered session's cwd must sit under before we add it
        # to allowed_dirs (7.5); a constructor arg so tests can use tmp
        self.home = str(Path(home).expanduser()) if home else str(Path.home())
        self._now = now
        self.poll_s = poll_s
        # interactive-pane timings: how long the TUI may take to come up,
        # how long after a turn-ending message we wait for a late record,
        # and how long a silent transcript with an idle pane may last
        self.ready_s = float(ready_s)
        self.ready_grace_s = float(ready_grace_s)
        self.key_wait_s = float(key_wait_s)
        self.settle_s = float(settle_s)
        self.idle_s = float(idle_s)
        self.clients_ttl_s = float(clients_ttl_s)
        self._panes: dict[str, Pane] = {}
        self._clients: dict[str, tuple] = {}     # slug -> (asked_at, attached)
        self._lock = threading.RLock()
        self._state: dict = {"active": "", "model": "", "fast_mode": None,
                             "projects": {}}
        self._projects: dict[str, Project] = {}
        self._tasks: dict[str, Task] = {}
        self._running: dict[str, Task] = {}
        self._queue: list[Task] = []
        self._threads: dict[str, threading.Thread] = {}
        self._resume_candidates: Optional[tuple] = None
        self._fast_mode_warned = False
        self.max_task_s = float(_cfg_get(cfg, "claude.max_task_s", 7200) or 7200)
        self._load_state()
        self._refresh_projects()
        bus.subscribe(ApprovalRequested, self._on_approval_requested)
        bus.subscribe(ApprovalResolved, self._on_approval_resolved)

    def close(self):
        bus.unsubscribe(ApprovalRequested, self._on_approval_requested)
        bus.unsubscribe(ApprovalResolved, self._on_approval_resolved)

    # ----------------------------------------------------------- state
    def _load_state(self):
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._state.update({k: data.get(k, v) for k, v in self._state.items()})
                if not isinstance(self._state.get("projects"), dict):
                    self._state["projects"] = {}
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            log.warning("claude_projects.json unreadable (%s); starting fresh", exc)

    def _save_state(self):
        with self._lock:
            projects = {}
            for slug, p in self._projects.items():
                projects[slug] = {"path": p.path, "session_id": p.session_id,
                                  "model": p.model, "last_used": p.last_used}
            self._state["projects"] = projects
            data = dict(self._state)
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError:
            log.exception("claude_projects.json save failed")

    def _refresh_projects(self):
        """Allowed dirs (config) ∪ remembered projects (state)."""
        with self._lock:
            known = self._projects
            for path in _allowed_dirs(self.cfg):
                slug = slugify(os.path.basename(path.rstrip("/"))) or "project"
                hit = next((p for p in known.values() if _same_dir(p.path, path)), None)
                if hit is None:
                    base, n = slug, 2
                    while slug in known:
                        slug = f"{base}-{n}"
                        n += 1
                    known[slug] = Project(slug=slug, path=path)
            for slug, raw in (self._state.get("projects") or {}).items():
                if not isinstance(raw, dict) or slug in known:
                    if isinstance(raw, dict) and slug in known:
                        p = known[slug]
                        p.session_id = p.session_id or str(raw.get("session_id") or "")
                        p.model = p.model or str(raw.get("model") or "")
                        p.last_used = max(p.last_used, float(raw.get("last_used") or 0))
                    continue
                path = str(raw.get("path") or "")
                if path and os.path.isdir(path):
                    known[slug] = Project(slug=slug, path=path,
                                          session_id=str(raw.get("session_id") or ""),
                                          model=str(raw.get("model") or ""),
                                          last_used=float(raw.get("last_used") or 0))

    # -------------------------------------------------------- projects
    def projects(self) -> list[Project]:
        self._refresh_projects()
        with self._lock:
            return list(self._projects.values())

    @property
    def active_project(self) -> Optional[str]:
        with self._lock:
            slug = self._state.get("active") or ""
            return slug if slug in self._projects else None

    def project_for(self, name) -> Optional[Project]:
        if not name:
            return None
        self._refresh_projects()
        raw = str(name).strip()
        low = re.sub(r"^(the|my)\s+", "", raw.lower()).strip()
        low = re.sub(r"\s+(project|repo|repository|folder)$", "", low).strip()
        slug = slugify(low)
        with self._lock:
            projects = list(self._projects.values())
        for p in projects:
            if p.slug == slug or _same_dir(p.path, os.path.expanduser(raw)):
                return p
        want = _tokens(low) or {slug}
        best, best_n = None, 0
        for p in projects:
            have = set(re.findall(r"[a-z0-9]+", p.slug)) | {p.slug.replace("-", "")}
            n = len([w for w in want if w in have or any(len(w) >= 4 and w in h for h in have)])
            if n > best_n:
                best, best_n = p, n
        return best

    def _resolve_project(self, project) -> Optional[Project]:
        if isinstance(project, Project):
            return project
        if project:
            return self.project_for(project)
        active = self.active_project
        if active:
            return self._projects.get(active)
        projects = self.projects()
        return projects[0] if projects else None

    def unsafe_dir(self, path) -> str:
        """"" when Claude may work in `path`, else the spoken reason (C2)."""
        return unsafe_dir_reason(path, self.home)

    def project_allowed(self, project) -> bool:
        """True when Jarvis may work in this project dir.

        With `claude.auto_approve_anywhere` (the default, user decision
        2026-08-26) every project is allowed and gets the same auto-approval
        as the configured roots; with it false, only dirs under
        `claude.allowed_dirs` are (7.3).
        """
        proj = project if isinstance(project, Project) else self.project_for(project)
        if proj is None or not proj.path:
            return False
        if self.unsafe_dir(proj.path):
            return False                 # home / config / system dir (C2)
        if self.auto_approve_anywhere:
            return True
        return any(_under(proj.path, root) for root in _allowed_dirs(self.cfg))

    def _set_active(self, project: Project):
        with self._lock:
            self._state["active"] = project.slug
            project.last_used = self._now()
        self._save_state()
        bus.publish(ActiveProject(slug=project.slug, path=project.path))

    def work_on(self, name) -> str:
        proj = self.project_for(name)
        if proj is None:
            return f"I don't know a project called {str(name).strip()}, sir; shall I set one up?"
        # C2 (round 2): never make active a project the manager will then
        # refuse every task in (a config-rooted one persisted in state.json,
        # say) -- refuse it here, once, instead of on every later utterance.
        if self.unsafe_dir(proj.path):
            log.warning("refusing to activate %s: %s is %s", proj.slug,
                        proj.path, self.unsafe_dir(proj.path))
            return UNSAFE_DIR_LINE
        self._set_active(proj)
        return f"{proj.display.capitalize()} it is, sir."

    def new_project(self, name) -> str:
        spoken = re.sub(r"\s+", " ", str(name or "")).strip()
        slug = slugify(spoken)
        if not slug:
            return "I'll need a name for it, sir."
        root = str(_cfg_get(self.cfg, "claude.projects_root", "") or
                   getattr(self.cfg, "projects_root", "") or DEFAULT_PROJECTS_ROOT)
        root = os.path.abspath(os.path.expanduser(root))
        path = os.path.join(root, slug)
        if not _under(path, root) or os.path.realpath(path) == os.path.realpath(root):
            return f"I only set projects up under {_short_path(root)}, sir."
        # C2 (round 2): check before committing.  A projects_root that points
        # (or links) into a config dir would otherwise get a directory made,
        # a project confirmed in persona and made active -- and then every
        # task in it refused.
        why = self.unsafe_dir(path)
        if why:
            log.warning("refusing to create project in %s: %s", path, why)
            return (f"That's {why}, sir, not somewhere I'd set a project up. "
                    "Point me at a proper projects folder and I'll make one "
                    "there.")
        existed = os.path.isdir(path)
        try:
            os.makedirs(path, exist_ok=True)
            if not os.path.isdir(os.path.join(path, ".git")):
                r = self._run(["git", "-C", path, "init", "-q"], timeout=15)
                if getattr(r, "returncode", 1) != 0:
                    log.warning("git init failed in %s: %s", path, getattr(r, "stderr", ""))
            claude_md = Path(path) / "CLAUDE.md"
            if not claude_md.exists():
                claude_md.write_text(
                    f"# {spoken}\n\n"
                    "- Python 3 / ~/vss_env unless told otherwise.\n"
                    "- Tests under tests/, run with `pytest -q`.\n"
                    "- Jarvis drives this repo; keep replies short — the last "
                    "one or two sentences are read aloud.\n", encoding="utf-8")
            readme = Path(path) / "README.md"
            if not readme.exists():
                readme.write_text(f"# {spoken}\n\nCreated by Jarvis on "
                                  f"{datetime.now():%Y-%m-%d}.\n", encoding="utf-8")
        except OSError as exc:
            log.exception("new_project %s failed", path)
            return f"I couldn't create the {spoken} project, sir: {exc.strerror or exc}."
        add = getattr(self.cfg, "add_allowed_dir", None)
        if callable(add):
            try:
                add(path)
            except Exception:                        # noqa: BLE001 - config hook
                log.exception("add_allowed_dir failed for %s", path)
        with self._lock:
            proj = next((p for p in self._projects.values() if _same_dir(p.path, path)), None)
            if proj is None:
                proj = Project(slug=slug, path=path)
                self._projects[slug] = proj
        self.ensure_project_settings(proj)
        self._set_active(proj)
        if existed:
            return f"The {spoken} project already exists, sir; it's active now."
        return f"The {spoken} project is set up, sir; ready when you are."

    # --------------------------------------------------- models / modes
    def _model_alias(self, text) -> Optional[str]:
        low = str(text or "").lower()
        for alias in MODEL_ALIASES:
            if re.search(rf"\b{alias}\b", low):
                return alias
        return None

    def set_model(self, alias) -> str:
        found = self._model_alias(alias)
        if found is None:
            return "I only know opus, sonnet, fable and haiku, sir."
        with self._lock:
            self._state["model"] = found
        self._save_state()
        return f"{found.capitalize()} it is, sir."

    def set_fast_mode(self, on: bool) -> str:
        with self._lock:
            self._state["fast_mode"] = bool(on)
        self._save_state()
        return "Fast mode on, sir." if on else "Fast mode off, sir."

    @property
    def model(self) -> str:
        with self._lock:
            return str(self._state.get("model") or "")

    @property
    def auto_approve_anywhere(self) -> bool:
        """True when any project may be worked in, not only allowed_dirs.

        User decision 2026-08-26: auto-approval is not limited to the
        configured roots.  Set `claude.auto_approve_anywhere` false to
        restore the fail-closed behaviour.
        """
        return bool(_cfg_get(self.cfg, "claude.auto_approve_anywhere", True))

    @property
    def permission_prompt_tool(self) -> bool:
        """True: the CLI contract is verified (see the block at the top of
        this module).  Set `claude.permission_prompt_tool` false to run
        without the prompt tool, which makes out-of-project work unaskable
        and so brings OUTSIDE_LINE back."""
        return bool(_cfg_get(self.cfg, "claude.permission_prompt_tool", True))

    @property
    def fast_mode(self) -> bool:
        with self._lock:
            fm = self._state.get("fast_mode")
        if fm is None:
            return bool(_cfg_get(self.cfg, "claude.fast_mode", False))
        return bool(fm)

    def model_for(self, prompt: str, override: Optional[str] = None) -> str:
        """Explicit override > sticky voice override > default with the
        size-estimate escalation (7.1)."""
        if override:
            return self._model_alias(override) or str(override)
        sticky = self.model
        if sticky:
            return sticky
        default = str(_cfg_get(self.cfg, "claude.model", DEFAULT_MODEL) or DEFAULT_MODEL)
        big = str(_cfg_get(self.cfg, "claude.big_model", DEFAULT_BIG_MODEL) or DEFAULT_BIG_MODEL)
        return big if estimate_size(prompt) == "large" else default

    # -------------------------------------------------------- settings
    def ensure_project_settings(self, project) -> Path:
        proj = project if isinstance(project, Project) else self.project_for(project)
        path = Path(proj.path) / ".claude" / "settings.local.json"
        # C2: never widen the user's own ~/.claude/settings.local.json (or any
        # config/system dir's).  Returned unwritten so callers still have the
        # path to log; submit()/_resume_session() refuse the dir outright.
        why = self.unsafe_dir(proj.path)
        if why:
            log.warning("refusing to write %s: %s is %s", path, proj.path, why)
            return path
        # C2 (round 2): judge the file we are about to WRITE, not merely the
        # project that nominally contains it.  A project `.claude` symlinked at
        # the user's own config dir (a common dotfiles habit) would otherwise
        # let an ordinary-looking project widen the user-level allowlist.
        real_dir = Path(os.path.realpath(path.parent))
        if not _under(str(real_dir), proj.path):
            log.warning("refusing to write %s: %s really lives at %s, outside "
                        "the project", path, path.parent, real_dir)
            return path
        outside = unsafe_dir_reason(str(real_dir.parent), self.home)
        if outside:
            log.warning("refusing to write %s: it really sits in %s", path, outside)
            return path
        if path.is_symlink():
            log.warning("refusing to write %s: it is a symlink to %s",
                        path, os.path.realpath(path))
            return path
        path = real_dir / path.name
        data: dict = {}
        try:
            if path.exists():
                # O_NOFOLLOW: never read (and so never copy out) a settings
                # file that is itself a link to the user's own.
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "r", encoding="utf-8") as fh:
                    data = json.loads(fh.read())
                if not isinstance(data, dict):
                    raise ValueError("top level is not an object")
        except ValueError as exc:
            bad = path.with_name(path.name + ".bad")
            log.warning("%s is corrupt (%s); kept as %s", path, exc, bad)
            try:
                os.replace(path, bad)
            except OSError:
                pass
            data = {}
        except OSError as exc:
            log.warning("cannot read %s: %s", path, exc)
        before = json.dumps(data, sort_keys=True)
        perms = data.get("permissions")
        if not isinstance(perms, dict):
            perms = {}
            data["permissions"] = perms
        allow = [r for r in perms.get("allow", []) if isinstance(r, str)
                 and r not in STALE_ALLOW_RULES] \
            if isinstance(perms.get("allow"), list) else []
        for rule in ALLOW_RULES:
            if rule not in allow:
                allow.append(rule)
        perms["allow"] = allow
        deny = [r for r in perms.get("deny", []) if isinstance(r, str)] \
            if isinstance(perms.get("deny"), list) else []
        for rule in DENY_RULES:
            if rule not in deny:
                deny.append(rule)
        perms["deny"] = deny
        if json.dumps(data, sort_keys=True) != before:
            try:
                real_dir.mkdir(parents=True, exist_ok=True)
                tmp = real_dir / "settings.local.tmp"
                tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                os.replace(tmp, path)
                log.info("wrote %s", path)
            except OSError:
                log.exception("could not write %s", path)
        return path

    def _sock_path(self) -> str:
        sock = getattr(self.approvals, "sock_path", None)
        if sock:
            return str(sock)
        return str(self.task_dir.parent / "approvals.sock")

    def write_mcp_config(self, project: Project) -> Path:
        """Per-project MCP config naming the permission server (7.3)."""
        env = {"JARVIS_APPROVAL_SOCK": self._sock_path(),
               "JARVIS_PROJECT": project.slug,
               "PYTHONPATH": str(REPO_ROOT)}
        cfg = {"mcpServers": {"jarvis": {"command": self.python,
                                         "args": ["-m", "jarvis.mcp_permissions"],
                                         "env": env}}}
        path = self.task_dir / project.slug / "mcp_jarvis.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        return path

    def _system_suffix(self) -> Path:
        path = self.task_dir / "system_suffix.txt"
        try:
            if not path.exists() or path.read_text(encoding="utf-8") != SYSTEM_SUFFIX:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(SYSTEM_SUFFIX, encoding="utf-8")
        except OSError:
            log.exception("system suffix write failed")
        return path

    # ---------------------------------------------------------- submit
    def _setup_line(self) -> str:
        fn = getattr(self.cfg, "setup_line", None)
        if callable(fn):
            try:
                return fn("claude")
            except Exception:                        # noqa: BLE001 - config shim
                pass
        return SETUP_LINE_FALLBACK

    def submit(self, prompt, project=None, parallel: bool = False,
               model: Optional[str] = None):
        prompt = str(prompt or "").strip()
        if not self.claude_bin:
            return self._setup_line()
        if not prompt:
            return "I'll need to know what to hand Claude, sir."
        proj = self._resolve_project(project)
        if proj is None:
            if project:
                return f"I don't know a project called {project}, sir; shall I set one up?"
            return NO_PROJECT_LINE
        why = self.unsafe_dir(proj.path)
        if why:
            log.warning("refusing %s: %s is %s", proj.slug, proj.path, why)
            return UNSAFE_DIR_LINE
        if not self.project_allowed(proj) and not self.permission_prompt_tool:
            # Real fallback, not a silent one: outside the cleared dirs the
            # prompt tool is the only way to ask, so without it we do not
            # start and offer the terminal instead (7.3).
            log.warning("refusing %s: %s is outside claude.allowed_dirs and "
                        "the permission prompt tool is off", proj.slug, proj.path)
            return OUTSIDE_LINE
        task = Task(task_id=f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
                    project=proj.slug, prompt=prompt,
                    model=self.model_for(prompt, model), cwd=proj.path,
                    parallel=bool(parallel))
        with self._lock:
            self._tasks[task.task_id] = task
            running = list(self._running.values())
            same = [t for t in running if t.project == proj.slug]
            other = [t for t in running if t.project != proj.slug]
            if same or (other and not parallel):
                self._queue.append(task)
                busy = (same or other)[0].project
                bus.publish(ClaudeTaskState(project=proj.slug, task_id=task.task_id,
                                            state="queued",
                                            text=f"Queued for {proj.display}: {_excerpt(prompt, 60)}"))
                return BUSY_LINE.format(project=display_name(busy))
            self._running[task.task_id] = task
        if self.active_project != proj.slug:
            self._set_active(proj)
        self._start(task, proj)
        return task

    def _start(self, task: Task, proj: Project):
        task.state = "running"
        task.started = self._now()
        proj.last_used = task.started
        bus.publish(ClaudeTaskState(project=proj.slug, task_id=task.task_id,
                                    state="running",
                                    text=f"Claude started on {proj.display}: {_excerpt(task.prompt, 60)}"))
        th = threading.Thread(target=self._runner, args=(task, proj),
                              name=f"claude-{proj.slug}", daemon=True)
        self._threads[task.task_id] = th
        th.start()

    def _paths(self, task: Task, attempt: int) -> dict:
        """Task bookkeeping on disk.  The prompt is kept for the audit
        trail; the pane no longer writes a stream, an rc or a stderr file
        because Claude runs interactively and its record IS the
        transcript, whose path is stored here once we know it."""
        d = self.task_dir / task.project
        return {"dir": d, "prompt": d / f"{task.task_id}.prompt",
                "transcript": ""}

    def build_command(self, proj: Project, model: str, session_id: str = "",
                      resume_id: str = "", mcp_path: Optional[Path] = None,
                      suffix_path: Optional[Path] = None,
                      prompt: str = "") -> str:
        """The shell line that starts the INTERACTIVE Claude in the pane.

        No `-p`, no `--output-format`, no pipe: that is the whole point —
        the pane must hold the real interface.  A brand-new pane gets a
        `--session-id` we generate, so the transcript's path is known
        before the first byte lands; a project we have worked in before
        gets `--resume <id>` (never `--continue`, which would hijack
        Hunter's own terminal session).

        The task's prompt rides along as the positional argument.  VERIFIED
        the hard way on 2026-08-27: keys typed at a TUI that is still
        claiming the tty are DROPPED — the pane sat at an empty `❯` while
        Jarvis waited for a turn that had never been asked for.  argv
        cannot be dropped."""
        q = shlex.quote
        parts = [q(self.claude_bin), "--model", q(model)]
        if resume_id:
            parts += ["--resume", q(resume_id)]
        elif session_id:
            parts += ["--session-id", q(session_id)]
        effort = str(_cfg_get(self.cfg, "claude.effort", "") or "")
        if effort:
            parts += ["--effort", q(effort)]
        if self.fast_mode:
            parts += ["--settings", q('{"fastMode":true}')]
        # The --mcp-config / --permission-prompt-tool pair is verified against
        # CLI 2.1.247 (see the block at the top of this module).  It stays
        # behind claude.permission_prompt_tool (default true) so the flags can
        # still be turned off in one place if a CLI upgrade breaks them.  It
        # matters MORE now: the pane is read-only, so a question the TUI asks
        # in-pane is a question nobody can answer.
        if bool(_cfg_get(self.cfg, "claude.dangerously_skip_permissions", False)):
            parts += ["--dangerously-skip-permissions"]
            if self.permission_prompt_tool and mcp_path is not None:
                parts += ["--mcp-config", q(str(mcp_path))]
        else:
            mode = str(_cfg_get(self.cfg, "claude.permission_mode", "acceptEdits") or "acceptEdits")
            parts += ["--permission-mode", q(mode)]
            if self.permission_prompt_tool and mcp_path is not None:
                parts += ["--mcp-config", q(str(mcp_path)),
                          "--permission-prompt-tool", PERMISSION_TOOL]
        if suffix_path is not None:
            parts += ["--append-system-prompt-file", q(str(suffix_path))]
        if prompt:
            parts += [q(prompt)]
        return f"clear; cd {q(proj.path)} && {' '.join(parts)}"

    # ------------------------------------------------------------ tmux
    def _tmux(self, *args, timeout: float = 10.0):
        return self._run(["tmux", *args], timeout=timeout)

    def _ensure_tmux(self, proj: Project) -> bool:
        name = f"jarvis-{proj.slug}"
        r = self._tmux("has-session", "-t", name)
        if getattr(r, "returncode", 1) != 0:
            r = self._tmux("new-session", "-d", "-s", name, "-c", proj.path,
                           "-x", str(TMUX_COLS), "-y", str(TMUX_ROWS))
            if getattr(r, "returncode", 1) != 0:
                log.error("tmux new-session %s failed: %s", name,
                          getattr(r, "stderr", ""))
                return False
        # A DETACHED session ignores -x/-y (window-size 'latest' falls back
        # to default-size, 80x24) and the Claude TUI wraps into soup.  Only
        # claim the geometry while nobody is watching: an attached client
        # must keep driving the size or the viewer sees a cropped pane.
        if not self._clients_attached(proj.slug, fresh=True):
            self._tmux("set-option", "-t", name, "window-size", "manual")
            self._tmux("resize-window", "-t", name, "-x", str(TMUX_COLS),
                       "-y", str(TMUX_ROWS))
        return True

    def _capture_pane(self, slug: str) -> str:
        r = self._tmux("capture-pane", "-p", "-t", f"jarvis-{slug}")
        if getattr(r, "returncode", 1) != 0:
            return ""
        return getattr(r, "stdout", "") or ""

    def _send_line(self, slug: str, text: str) -> bool:
        """One line of literal text plus Enter.  `-l` matters: without it
        tmux reads words like 'C-c' or 'Enter' inside a prompt as KEYS."""
        name = f"jarvis-{slug}"
        r = self._tmux("send-keys", "-t", name, "-l", text)
        if getattr(r, "returncode", 1) != 0:
            log.error("tmux send-keys -l %s failed: %s", name,
                      getattr(r, "stderr", ""))
            return False
        r = self._tmux("send-keys", "-t", name, "Enter")
        return getattr(r, "returncode", 1) == 0

    def _clients_attached(self, slug: str, fresh: bool = False) -> bool:
        """Is a tmux client watching this project's session?  Cached for
        `clients_ttl_s` so the router and the UI can ask freely."""
        now = self._now()
        if not fresh:
            hit = self._clients.get(slug)
            if hit and now - hit[0] < self.clients_ttl_s:
                return bool(hit[1])
        r = self._tmux("list-clients", "-t", f"jarvis-{slug}", "-F",
                       "#{client_name}", timeout=5)
        ok = (getattr(r, "returncode", 1) == 0
              and bool((getattr(r, "stdout", "") or "").strip()))
        self._clients[slug] = (now, ok)
        return ok

    def terminal_open(self, project=None) -> bool:
        """True while a terminal is attached to that project's session —
        i.e. the pop-out is on screen and Hunter can see what Claude is
        doing.  Cheap and non-blocking: safe to call from the router."""
        proj = self._resolve_project(project)
        if proj is None:
            return False
        return self._clients_attached(proj.slug)

    def open_projects(self) -> list[str]:
        """Slugs whose terminal is currently open (newest project first)."""
        projects = sorted(self.projects(), key=lambda p: -p.last_used)
        return [p.slug for p in projects if self._clients_attached(p.slug)]

    # ----------------------------------------------------------- runner
    def _runner(self, task: Task, proj: Project):
        try:
            attempt = 1
            resume_id = proj.session_id
            while True:
                outcome = self._run_attempt(task, proj, resume_id, attempt)
                if outcome == "retry" and attempt == 1:
                    log.warning("resume of %s failed; retrying without --resume",
                                resume_id)
                    proj.session_id = ""
                    self._save_state()
                    resume_id, attempt = "", 2
                    task.attempt = 2
                    continue
                break
        except Exception:                            # noqa: BLE001 - runner boundary
            log.exception("runner crashed for %s", task.task_id)
            self._finish(task, proj, "failed", "Claude's runner crashed, sir.")
        finally:
            self._threads.pop(task.task_id, None)

    def _run_attempt(self, task: Task, proj: Project, resume_id: str,
                     attempt: int) -> str:
        paths = self._paths(task, attempt)
        task.paths = {k: str(v) for k, v in paths.items()}
        try:
            paths["dir"].mkdir(parents=True, exist_ok=True)
            paths["prompt"].write_text(task.prompt + "\n", encoding="utf-8")
        except OSError:
            log.exception("task files for %s", task.task_id)
            self._finish(task, proj, "failed", "I couldn't write the task files, sir.")
            return "done"
        self.ensure_project_settings(proj)
        mcp_path = self.write_mcp_config(proj)
        suffix = self._system_suffix()
        if not self._ensure_tmux(proj):
            self._finish(task, proj, "failed",
                         "I couldn't open a tmux session for Claude, sir.")
            return "done"
        log.info("task %s [%s] model=%s resume=%s", task.task_id, proj.slug,
                 task.model, resume_id or "-")
        warm = self._panes.get(proj.slug) is not None and self._pane_busy(proj)
        # Where the transcript stands BEFORE the prompt goes in, so a
        # resumed session's history is never replayed as tonight's work.
        known = self._panes[proj.slug].session_id if warm else resume_id
        path = find_transcript(proj.path, self.projects_dir, known) if known else None
        try:
            offset = path.stat().st_size if path else 0
        except OSError:
            path, offset = None, 0
        sent_at = self._now()
        pane, delivered = self._ensure_claude(
            task, proj, resume_id, mcp_path, suffix,
            "" if warm else sanitize_keys(task.prompt))
        if pane is None:
            if task.state == "cancelled":
                return "done"
            if resume_id and attempt == 1:
                return "retry"
            self._finish(task, proj, "failed",
                         "I couldn't get Claude started in the terminal, sir.")
            return "done"
        task.session_id = pane.session_id or task.session_id
        pane.transcript = str(path) if path else ""
        pane.offset = offset
        if not delivered and not self._send_prompt(proj, task.prompt):
            self._finish(task, proj, "failed", "tmux wouldn't take the prompt, sir.")
            return "done"
        model = task.model or ""
        bus.publish(ClaudeProgress(project=task.project, task_id=task.task_id,
                                   line=f"Started · {model}" if model else "Started",
                                   milestone=False))
        return self._tail_transcript(task, proj, pane, sent_at)

    def _ensure_claude(self, task: Task, proj: Project, resume_id: str,
                       mcp_path: Path, suffix: Path,
                       launch_prompt: str = "") -> tuple:
        """Make sure an interactive Claude is live in the project's window
        and ready for input.  Returns (pane, prompt_delivered) — a pane we
        start ourselves carries the prompt in its argv, so there is nothing
        left to type."""
        pane = self._panes.get(proj.slug)
        alive = self._pane_busy(proj)
        if pane is not None and alive:
            return (pane, False) if self._await_ready(task, proj) else (None, False)
        self._panes.pop(proj.slug, None)
        delivered = False
        if alive:
            # Something already holds the pane — Hunter's own claude, most
            # likely.  Use it rather than fighting it; its transcript is
            # found by mtime because we did not choose its session id.
            log.info("jarvis-%s already has a live process; reusing it", proj.slug)
            pane = Pane(slug=proj.slug, started=self._now())
        else:
            sid = "" if resume_id else str(uuid.uuid4())
            cmd = self.build_command(proj, task.model, sid, resume_id,
                                     mcp_path, suffix, launch_prompt)
            pane = Pane(slug=proj.slug, session_id=resume_id or sid,
                        started=self._now())
            if not self._send_line(proj.slug, cmd):
                return None, False
            delivered = bool(launch_prompt)
        if not self._await_ready(task, proj):
            return None, False
        self._panes[proj.slug] = pane
        return pane, delivered

    def _await_ready(self, task: Task, proj: Project) -> bool:
        """Poll the pane until the TUI takes input.  Answers the
        workspace-trust dialog — which a read-only pane could never answer
        — but only for a folder the user has already cleared for Claude."""
        start = self._now()
        deadline = start + self.ready_s
        stable = 0
        while self._now() < deadline:
            if task.state == "cancelled":
                return False
            text = self._capture_pane(proj.slug)
            state = pane_state(text)
            if state == "settings":
                # "1. Continue" is preselected: the CLI has already skipped
                # the offending rule and the rest of the file is in effect,
                # exactly as print mode behaves today.
                log.warning("Claude skipped an invalid rule in %s/.claude/"
                            "settings.local.json; continuing", proj.path)
                self._tmux("send-keys", "-t", f"jarvis-{proj.slug}", "Enter")
                stable = 0
                time.sleep(min(1.0, self.poll_s * 4))
                continue
            if state == "menu":
                log.error("jarvis-%s is stopped on a dialog I don't know how "
                          "to answer; leaving it alone", proj.slug)
                return False
            if state == "trust":
                if self.unsafe_dir(proj.path) or not self.project_allowed(proj):
                    log.warning("Claude wants workspace trust for %s and I "
                                "won't grant it", proj.path)
                    # Esc cancels the dialog, so the refused session exits
                    # instead of sitting there holding the pane.
                    self._tmux("send-keys", "-t", f"jarvis-{proj.slug}", "Escape")
                    return False
                log.info("accepting Claude's workspace-trust dialog for %s",
                         proj.path)
                self._tmux("send-keys", "-t", f"jarvis-{proj.slug}", "Enter")
                stable = 0
                time.sleep(min(1.0, self.poll_s * 4))
                continue
            if state == "ready":
                stable += 1
                if stable >= 2:
                    return True
            else:
                stable = 0
                if self._now() - start > self.ready_grace_s \
                        and not self._pane_busy(proj):
                    log.error("claude is not running in jarvis-%s", proj.slug)
                    return False
            time.sleep(min(0.5, self.poll_s * 2))
        log.error("claude never became ready in jarvis-%s", proj.slug)
        return False

    def _send_prompt(self, proj: Project, prompt: str) -> bool:
        """Type a prompt into a session that is already up, and CONFIRM it
        landed in the input box before pressing Enter — a TUI that is busy
        redrawing silently drops keys, and a dropped prompt is a task that
        waits forever for a turn nobody asked for."""
        text = sanitize_keys(prompt)
        if not text:
            return False
        name = f"jarvis-{proj.slug}"
        probe = text[:24]
        for attempt in range(1, 4):
            r = self._tmux("send-keys", "-t", name, "-l", text)
            if getattr(r, "returncode", 1) != 0:
                log.error("tmux send-keys -l %s failed: %s", name,
                          getattr(r, "stderr", ""))
                return False
            deadline = self._now() + self.key_wait_s
            while self._now() < deadline:
                if probe in " ".join(self._capture_pane(proj.slug).split()):
                    r = self._tmux("send-keys", "-t", name, "Enter")
                    return getattr(r, "returncode", 1) == 0
                time.sleep(min(0.2, self.poll_s * 2))
            log.warning("the prompt did not reach jarvis-%s (try %d); clearing "
                        "the input box and retyping", proj.slug, attempt)
            self._tmux("send-keys", "-t", name, "C-u")
        return False

    # ------------------------------------------------------------ tail
    def _tail_transcript(self, task: Task, proj: Project, pane: Pane,
                         sent_at: float) -> str:
        """Follow the session transcript until the turn ends.

        Completion is the assistant message that stops the turn
        (`stop_reason` end_turn) with nothing after it — never a fixed
        sleep.  Its text becomes `Task.result_text`, which the app trims
        to the spoken two sentences."""
        fh = None
        path = Path(pane.transcript) if pane.transcript else None
        offset = pane.offset
        last_entry = sent_at
        last_liveness = sent_at
        final_at = 0.0
        try:
            while True:
                if task.state == "cancelled":
                    return "done"
                now = self._now()
                if path is None or not path.exists():
                    found = find_transcript(proj.path, self.projects_dir,
                                            pane.session_id, sent_at - 5.0)
                    if found is not None:
                        path, offset = found, 0
                        pane.transcript = str(found)
                        task.paths["transcript"] = str(found)
                if fh is None and path is not None and path.exists():
                    try:
                        fh = open(path, "rb")
                        fh.seek(offset)
                        pane.transcript = str(path)
                        task.paths["transcript"] = str(path)
                    except OSError:
                        log.exception("cannot read %s", path)
                        fh = None
                if fh is not None:
                    for entry in self._drain(fh):
                        last_entry = now = self._now()
                        for prog in parse_transcript_entry(entry, task, now):
                            bus.publish(prog)
                        sid = entry.get("sessionId") if isinstance(entry, dict) else ""
                        if isinstance(sid, str) and sid and sid != proj.session_id:
                            proj.session_id = pane.session_id = sid
                            self._save_state()
                        text = entry_turn_text(entry)
                        if text is None:
                            if isinstance(entry, dict) and not entry.get("isSidechain") \
                                    and entry.get("type") in ("assistant", "user"):
                                final_at = 0.0      # the turn carried on
                        else:
                            task.result_text = text
                            final_at = now
                if final_at and self._now() - final_at >= self.settle_s:
                    self._finish(task, proj, "done", task.result_text)
                    return "done"
                now = self._now()
                if now - task.started > self.max_task_s:
                    log.warning("task %s exceeded %.0f s; cancelling",
                                task.task_id, self.max_task_s)
                    self.cancel(proj.slug)
                    return "done"
                if now - last_entry > self.idle_s:
                    pane_text = self._capture_pane(proj.slug)
                    if not pane_working(pane_text):
                        log.warning("task %s: %.0f s of silence and an idle pane",
                                    task.task_id, now - last_entry)
                        self._finish(task, proj, "failed", NO_RESULT_LINE,
                                     speak=True)
                        return "done"
                    last_entry = now
                # liveness costs a subprocess, so ask every 5 s, not 4x/s
                if now - last_liveness >= 5.0:
                    last_liveness = now
                    if not self._pane_busy(proj):
                        self._panes.pop(proj.slug, None)
                        self._finish(task, proj, "failed",
                                     "Claude's process vanished, sir.", speak=True)
                        return "done"
                time.sleep(self.poll_s)
        finally:
            if fh is not None:
                fh.close()

    @staticmethod
    def _drain(fh) -> list:
        """Every COMPLETE JSON line that has landed since the last call (a
        half-written line is left for the next pass)."""
        out = []
        while True:
            pos = fh.tell()
            raw = fh.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                fh.seek(pos)
                break
            try:
                out.append(json.loads(raw))
            except ValueError:
                continue
        return out

    def _pane_busy(self, proj: Project) -> bool:
        r = self._run(["tmux", "list-panes", "-t", f"jarvis-{proj.slug}", "-F",
                       "#{pane_current_command}"], timeout=10)
        if getattr(r, "returncode", 1) != 0:
            return False
        cmds = [c.strip() for c in (getattr(r, "stdout", "") or "").splitlines() if c.strip()]
        return any(c not in _SHELLS for c in cmds)

    def _finish(self, task: Task, proj: Project, state: str, text: str,
                speak: bool = False):
        with self._lock:
            if task.final:
                return
            task.state = state
            task.finished = self._now()
            self._running.pop(task.task_id, None)
        log.info("task %s %s rc=%s", task.task_id, state, task.rc)
        if speak:
            bus.publish(ClaudeProgress(project=proj.slug, task_id=task.task_id,
                                       line=text, milestone=True))
        bus.publish(ClaudeTaskState(project=proj.slug, task_id=task.task_id,
                                    state=state, text=text or state))
        self._pump_queue()

    def _pump_queue(self):
        to_start = []
        with self._lock:
            running = list(self._running.values())
            for task in list(self._queue):
                busy_projects = {t.project for t in running} | {t.project for t in to_start}
                if task.project in busy_projects:
                    continue
                if (running or to_start) and not task.parallel:
                    continue
                self._queue.remove(task)
                self._running[task.task_id] = task
                to_start.append(task)
        for task in to_start:
            proj = self._projects.get(task.project)
            if proj is None:
                self._finish(task, Project(slug=task.project, path=task.cwd),
                             "failed", "That project has gone, sir.")
                continue
            self._start(task, proj)

    # ---------------------------------------------------------- cancel
    def cancel(self, project=None) -> bool:
        with self._lock:
            running = list(self._running.values())
            if project:
                proj = self.project_for(project)
                running = [t for t in running if proj and t.project == proj.slug]
            elif self.active_project and len(running) > 1:
                running = [t for t in running if t.project == self.active_project] or running
            task = running[0] if running else None
            if task is None:
                # a queued one may still be dropped
                queued = [t for t in self._queue
                          if not project or t.project == getattr(self.project_for(project), "slug", "")]
                if queued:
                    t = queued[-1]
                    self._queue.remove(t)
                    t.state = "cancelled"
                    bus.publish(ClaudeTaskState(project=t.project, task_id=t.task_id,
                                                state="cancelled", text=CANCELLED_LINE))
                    return True
                return False
            task.state = "cancelled"
        proj = self._projects.get(task.project) or Project(slug=task.project, path=task.cwd)
        # C-c interrupts the turn in the TUI (spec 7); it does NOT exit the
        # interactive session, so the pane keeps its scrollback and the next
        # task reuses the same Claude.  Nothing is killed: the pane's child
        # IS the session Hunter is watching.
        self._tmux("send-keys", "-t", f"jarvis-{task.project}", "C-c")
        with self._lock:
            task.finished = self._now()
            self._running.pop(task.task_id, None)
        log.info("task %s cancelled", task.task_id)
        bus.publish(ClaudeTaskState(project=proj.slug, task_id=task.task_id,
                                    state="cancelled", text=CANCELLED_LINE))
        self._pump_queue()
        return True

    # ---------------------------------------------------------- resume
    def resume(self, utterance: str = "") -> str:
        with self._lock:
            pending = self._resume_candidates
        if pending:
            choice = self._resolve_candidate(utterance, pending)
            if choice is not None:
                with self._lock:
                    self._resume_candidates = None
                return self._resume_session(choice)
        # C2: a session whose cwd is $HOME (or a config/system dir) is not a
        # project and must never be picked up — drop it before ranking.
        sessions = [s for s in discover_sessions(self.projects_dir)
                    if not self.unsafe_dir(s.cwd)]
        if not sessions:
            return NO_SESSION_LINE
        best, runner, question = pick_session(utterance, sessions, self._now())
        if question:
            with self._lock:
                self._resume_candidates = (best, runner)
            return question
        return self._resume_session(best)

    def pending_question(self) -> bool:
        with self._lock:
            return self._resume_candidates is not None

    def _resolve_candidate(self, text: str, pair) -> Optional[SessionInfo]:
        toks = _tokens(text)
        low = str(text or "").lower()
        hits = []
        for info in pair:
            ntoks = _name_tokens(info.cwd)
            if toks & ntoks or any(len(u) >= 4 and any(u in n for n in ntoks) for u in toks):
                hits.append(info)
        if len(hits) == 1:
            return hits[0]
        if re.search(r"\b(first|former|the first one)\b", low):
            return pair[0]
        if re.search(r"\b(second|latter|the other|other one)\b", low):
            return pair[1]
        return None

    def _resume_session(self, info: SessionInfo) -> str:
        why = self.unsafe_dir(info.cwd)
        if why:
            log.warning("refusing to resume %s: its cwd %s is %s",
                        info.session_id, info.cwd, why)
            return UNSAFE_DIR_LINE
        proj = next((p for p in self.projects() if _same_dir(p.path, info.cwd)), None)
        if proj is None:
            if not _under(info.cwd, self.home):
                return "That session lives outside your home folder, sir; I'd rather not."
            add = getattr(self.cfg, "add_allowed_dir", None)
            if callable(add):
                try:
                    add(info.cwd)
                except Exception:                    # noqa: BLE001 - config hook
                    log.exception("add_allowed_dir failed for %s", info.cwd)
            slug = info.slug or "project"
            with self._lock:
                base, n = slug, 2
                while slug in self._projects:
                    slug = f"{base}-{n}"
                    n += 1
                proj = Project(slug=slug, path=info.cwd)
                self._projects[slug] = proj
        proj.session_id = info.session_id
        self._set_active(proj)
        fallback = f"Right away, sir — picking up {proj.display} where we left off."
        ack = fallback
        local_line = getattr(self.brain, "local_line", None)
        if callable(local_line):
            try:
                what = info.title or info.first_user
                ack = local_line(
                    "Acknowledge in one short sentence that you're picking this "
                    "project up where we left off, naming it",
                    f"project: {proj.display}; last worked on: {_excerpt(what, 80)}",
                    max_sentences=1, timeout=2.0, fallback=fallback) or fallback
            except TypeError:
                try:
                    ack = local_line("Acknowledge picking this project up where we left off",
                                     proj.display) or fallback
                except Exception:                    # noqa: BLE001 - brain shim
                    ack = fallback
            except Exception:                        # noqa: BLE001 - brain shim
                log.exception("local_line failed")
                ack = fallback
        result = self.submit(RESUME_PROMPT, project=proj.slug)
        if isinstance(result, str):
            return result
        return ack

    # -------------------------------------------------------- terminal
    def open_terminal(self, slug=None) -> bool:
        proj = self._resolve_project(slug)
        if proj is None:
            log.warning("open_terminal: no project")
            return False
        if not self._ensure_tmux(proj):
            return False
        name = f"jarvis-{proj.slug}"
        title = f"Jarvis · {proj.slug}"
        r = self._run(["xdotool", "search", "--name", f"^{re.escape(title)}$"], timeout=10)
        ids = [w for w in (getattr(r, "stdout", "") or "").split() if w.strip()]
        if getattr(r, "returncode", 1) == 0 and ids:
            self._run(["xdotool", "windowactivate", ids[0]], timeout=10)
            return True
        # Hand the window size back to the client that is about to watch,
        # so the TUI redraws at the real terminal geometry instead of the
        # detached 200x50 we hold while nobody is looking.
        self._tmux("set-option", "-t", name, "window-size", "latest")
        # `-r` is the point: the pane shows the genuine Claude Code
        # interface, and keystrokes into it cannot drive the session.
        # Jarvis stays the input path.
        r = self._run(["gnome-terminal", "--title", title,
                       f"--geometry={TMUX_COLS}x{TMUX_ROWS}",
                       "--", "tmux", "attach", "-r", "-t", name], timeout=15)
        ok = getattr(r, "returncode", 1) == 0
        if not ok:
            log.error("gnome-terminal failed: %s", getattr(r, "stderr", ""))
        else:
            self._clients.pop(proj.slug, None)      # re-ask on the next poll
        return ok

    # ---------------------------------------------------------- status
    def tasks(self) -> list[Task]:
        with self._lock:
            return list(self._tasks.values())

    def running_tasks(self) -> list[Task]:
        with self._lock:
            return list(self._running.values())

    def task(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def status_text(self) -> str:
        with self._lock:
            running = list(self._running.values())
            queued = len(self._queue)
        now = self._now()
        if running:
            t = running[0]
            proj = self._projects.get(t.project)
            name = proj.display if proj else display_name(t.project)
            verb = "waiting on you about" if t.state == "waiting" else "working on"
            line = f"Claude's {verb} {name}, sir; started {_ago(now - t.started)}."
            if len(running) > 1:
                line = line[:-1] + f", and {display_name(running[1].project)} alongside."
        else:
            active = self.active_project
            line = "Claude's idle, sir."
            if active:
                line = f"Claude's idle, sir; the active project is {display_name(active)}."
        if queued:
            line += " One more is queued." if queued == 1 else f" {queued} more are queued."
        return line

    # --------------------------------------------------- approvals hook
    def _task_for_approval(self, project: str) -> Optional[Task]:
        with self._lock:
            running = [t for t in self._running.values() if not t.final]
            if project:
                hit = [t for t in running if t.project == project]
                if hit:
                    return hit[0]
            return running[0] if len(running) == 1 else None

    def _on_approval_requested(self, ev: ApprovalRequested):
        task = self._task_for_approval(ev.project)
        if task is None or task.state != "running":
            return
        task.state = "waiting"
        bus.publish(ClaudeTaskState(project=task.project, task_id=task.task_id,
                                    state="waiting", text=ev.question))

    def _on_approval_resolved(self, ev: ApprovalResolved):
        with self._lock:
            waiting = [t for t in self._running.values() if t.state == "waiting"]
        for task in waiting:
            task.state = "running"
            bus.publish(ClaudeTaskState(project=task.project, task_id=task.task_id,
                                        state="running",
                                        text="Allowed; carrying on." if ev.allowed else "Declined; carrying on."))


# --------------------------------------------------------- render subcmd
def render_main(argv) -> int:
    """stdin (stream-json) -> --out (raw, flushed per line) + readable pane."""
    import argparse
    ap = argparse.ArgumentParser(prog="jarvis.claude_session render")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cwd", default=os.getcwd())
    args = ap.parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = sys.stdout
    with open(out_path, "ab") as out:
        for raw in sys.stdin.buffer:
            out.write(raw)
            out.flush()
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            for line in describe_event(ev, args.cwd):
                try:
                    stdout.write(line + "\n")
                    stdout.flush()
                except OSError:
                    pass
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "render":
        return render_main(argv[1:])
    sys.stderr.write("usage: python -m jarvis.claude_session render --out <file>\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
