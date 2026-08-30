"""Git standup: "what did I do today?" across every cleared project.

Walks the repos the assistant config clears for Claude (``claude.allowed_dirs``
plus the children of ``claude.projects_root``, which may not exist) and the
VSS tree, reads each one's commit subjects for the day and its uncommitted
diff stat through ``ContextEngine.git_repos``, adds the Claude Code sessions
whose transcript was touched that day (title or opening line, from
``~/.claude/projects``), and renders two things: a text card of commit
subjects and a two-sentence spoken line in Jarvis's voice.

Everything here is local -- git subprocesses and jsonl parsing -- and the
spoken line is composed from the data, not by the model, so the standup still
works while the local model is lent out (jarvis/brain.py release()).

Session filtering: ``~/.claude/projects`` holds transcripts for scratchpads,
probes and subagent runs (cwd under /tmp or ~/.claude); those are not
projects and are dropped by ``is_project_session``. Transcripts are only
parsed when their mtime falls on the day asked for, and each is read up to
``SESSION_MAX_BYTES`` -- the title/summary and opening turn sit at the top
of the file, and a long Jarvis session can run to tens of megabytes.
"""
from __future__ import annotations

import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional

from jarvis.claude_session import (SessionInfo, _allowed_dirs, _excerpt,
                                   read_session)
from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("standup")

SESSION_MAX_BYTES = 1_000_000
MAX_CARD_COMMITS = 12          # per repo, on the card
MAX_SPOKEN_SESSIONS = 2        # session titles read aloud
SCRATCH_ROOTS = ("/tmp", "/var/tmp", "/dev/shm")
FAILED_LINE = "I couldn't put the standup together, sir."

# The command, without the "jarvis" prefix (ASSISTANT_TIER1 lowercases and
# strips trailing punctuation first). The day word is optional and defaults
# to today; "yesterday's standup" and "standup for yesterday" both carry it.
STANDUP_RX = re.compile(
    r"^(?:(?:give me |run |do )?(?:(?:my |the |a )?(?:daily |morning |quick |git )?)?"
    r"stand[- ]?up(?: (?:for|on) (?P<day1>today|yesterday))?"
    r"|(?P<day0>today|yesterday)'?s? stand[- ]?up"
    r"|what (?:did|have) i (?:do|done|get done|got done|change|changed|commit|committed|"
    r"work on|worked on|ship|shipped|build|built)"
    r"(?: (?:so far )?(?P<day2>today|yesterday|this morning|tonight|so far))?"
    r"|what(?:'s| is| was) (?:my|the) (?:git )?(?:progress|day) (?:like )?"
    r"(?P<day3>today|yesterday)?)\W*$", re.I)


def day_word(match) -> str:
    """'today' | 'yesterday' from a STANDUP_RX match (default today)."""
    for name in ("day0", "day1", "day2", "day3"):
        try:
            word = (match.group(name) or "").lower()
        except (IndexError, AttributeError):
            word = ""
        if word == "yesterday":
            return "yesterday"
        if word:
            return "today"
    return "today"


def day_for(word: str, now: Optional[float] = None) -> date:
    today = datetime.fromtimestamp(time.time() if now is None else now).date()
    return today - timedelta(days=1) if word == "yesterday" else today


def window(day: date) -> tuple[str, str]:
    """ISO [since, until) bounds for git log: the day, local time."""
    return (f"{day.isoformat()} 00:00:00",
            f"{(day + timedelta(days=1)).isoformat()} 00:00:00")


# ------------------------------------------------------------- repo list
def repo_dirs(cfg, extra: Iterable = (), isdir: Callable = os.path.isdir,
              listdir: Callable = os.listdir) -> list[str]:
    """claude.allowed_dirs, then the children of claude.projects_root (which
    may not exist), then ``extra`` (the VSS tree); deduplicated, only
    directories that carry a .git. Order is the primary-first order the
    prompt's git line relies on."""
    out: list[str] = []

    def add(path):
        path = os.path.abspath(os.path.expanduser(str(path)))
        if path in out:
            return
        if isdir(path) and isdir(os.path.join(path, ".git")):
            out.append(path)

    for d in _allowed_dirs(cfg):
        add(d)
    root = ""
    try:
        root = (getattr(cfg, "projects_root", None) or
                (cfg.get("claude.projects_root", "") if hasattr(cfg, "get") else ""))
    except Exception:
        log.debug("projects_root unreadable", exc_info=True)
    if root:
        root = os.path.abspath(os.path.expanduser(str(root)))
        try:
            children = sorted(listdir(root)) if isdir(root) else []
        except OSError:
            children = []
        for name in children:
            if not name.startswith("."):
                add(os.path.join(root, name))
    for d in extra:
        add(d)
    return out


# ------------------------------------------------------------- sessions
def is_project_session(cwd: str, home: Optional[str] = None) -> bool:
    """A transcript whose cwd is a real project: not a scratch root, not a
    dotted folder (~/.claude/projects/... subagent and memory dirs), not
    $HOME itself."""
    cwd = str(cwd or "").rstrip("/")
    if not cwd:
        return False
    home_dir = str(home or Path.home()).rstrip("/")
    if cwd == home_dir:
        return False
    if any(cwd == r or cwd.startswith(r + "/") for r in SCRATCH_ROOTS):
        return False
    return not any(part.startswith(".") for part in Path(cwd).parts[1:])


def _is_jarvis_own(info: SessionInfo) -> bool:
    """Jarvis's own one-shot `claude -p` sessions (the web lookup opens with
    "You are answering a spoken question ..."; read_session already drops
    the "You are Jarvis" Tier-3 ones) are not Hunter's work. A person does
    not open a session with "You are ...", so the prefix is the tell."""
    return (info.first_user or "").lstrip().startswith("You are ")


def sessions_on(day: date, projects_dir=None, home: Optional[str] = None,
                reader: Callable = read_session) -> list[SessionInfo]:
    """Claude Code sessions whose transcript was touched on ``day``, newest
    first, project sessions only. Never raises."""
    root = Path(projects_dir or (Path.home() / ".claude" / "projects"))
    try:
        files = [p for p in root.glob("*/*.jsonl") if p.is_file()]
    except OSError:
        return []
    hits = []
    for path in files:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if datetime.fromtimestamp(mtime).date() != day:
            continue
        hits.append((mtime, path))
    hits.sort(reverse=True)
    out: list[SessionInfo] = []
    seen: set[str] = set()
    for mtime, path in hits:
        try:
            info = reader(path, max_bytes=SESSION_MAX_BYTES)
        except Exception:
            log.debug("session %s unreadable", path, exc_info=True)
            continue
        if info is None or not is_project_session(info.cwd, home):
            continue
        if _is_jarvis_own(info):
            continue
        if info.session_id in seen:
            continue
        seen.add(info.session_id)
        out.append(info)
    return out


def session_label(info: SessionInfo, n: int = 60) -> str:
    return _excerpt(info.title or info.first_user or "", n)


# ------------------------------------------------------------- rendering
def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _join(names: list[str]) -> str:
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def card_text(repos: list[dict], sessions: list[SessionInfo], word: str,
              day: date) -> str:
    lines = [f"Standup — {word} ({day.strftime('%a %d %b')})"]
    if not repos:
        lines.append("No git repositories configured (claude.allowed_dirs).")
    for r in repos:
        bits = [r.get("branch") or "HEAD"]
        if r.get("ahead"):
            bits.append(f"{r['ahead']} ahead")
        if r.get("changed"):
            bits.append(f"{_plural(int(r['changed']), 'file')} uncommitted")
        else:
            bits.append("clean")
        commits = list(r.get("commits") or [])
        head = f"{r.get('repo', '?')} ({', '.join(bits)})"
        lines.append(head if commits else f"{head} — no commits {word}")
        for subject in commits[:MAX_CARD_COMMITS]:
            lines.append(f"  - {subject}")
        if len(commits) > MAX_CARD_COMMITS:
            lines.append(f"  … and {len(commits) - MAX_CARD_COMMITS} more")
        if r.get("diff_stat"):
            lines.append(f"  uncommitted: {r['diff_stat'].strip()}")
    lines.append(f"Claude sessions {word}:" if sessions
                 else f"No Claude sessions {word}.")
    for s in sessions:
        project = os.path.basename(s.cwd.rstrip("/")) or s.cwd
        lines.append(f"  - {project}: {session_label(s)}")
    return "\n".join(lines)


def spoken_line(repos: list[dict], sessions: list[SessionInfo],
                word: str) -> str:
    """Two sentences, in character, from the data: commits and what is
    still uncommitted, then the Claude sessions."""
    cap = word.capitalize()
    with_commits = [r for r in repos if r.get("commits")]
    total = sum(len(r["commits"]) for r in with_commits)
    dirty = [r for r in repos if r.get("changed")]
    if not repos:
        first = "I have no repositories to look at, sir"
    elif total:
        first = (f"{cap} you made {_plural(total, 'commit')} in "
                 f"{_join([r['repo'] for r in with_commits])}, sir")
    else:
        first = f"No commits {word}, sir"
    if dirty:
        parts = [f"{_plural(int(r['changed']), 'file')} in {r['repo']}"
                 for r in dirty[:2]]
        first += f", with {_join(parts)} still uncommitted"
    first += "."
    if sessions:
        names = [session_label(s, 40) for s in sessions[:MAX_SPOKEN_SESSIONS]]
        names = [n for n in names if n]
        second = f"Claude had {_plural(len(sessions), 'session')} {word}"
        if names:
            second += f", on {_join(names)}"
        second += "."
    else:
        second = f"No Claude sessions {word}."
    return f"{first} {second}"


def build(git_repos: Callable, word: str = "today", now: Optional[float] = None,
          projects_dir=None, home: Optional[str] = None) -> tuple[str, str]:
    """(spoken line, card) for ``word`` in {today, yesterday}.
    ``git_repos(since, until)`` is ContextEngine.git_repos."""
    word = "yesterday" if word == "yesterday" else "today"
    day = day_for(word, now)
    since, until = window(day)
    repos = list(git_repos(since=since, until=until) or [])
    sessions = sessions_on(day, projects_dir=projects_dir, home=home)
    return spoken_line(repos, sessions, word), card_text(repos, sessions, word, day)


def default_repo_dirs(cfg) -> list[str]:
    """The app's list: the cleared projects plus the VSS tree."""
    return repo_dirs(cfg, extra=(PATHS.VSS_ENV,))
