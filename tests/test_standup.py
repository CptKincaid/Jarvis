"""Git standup (jarvis/standup.py + ContextEngine.git_repos): the per-repo
probe over tmp repositories, the day window, the Claude-session filter
over a tmp ~/.claude/projects, the spoken line and card, the Tier-1 command
in the commander, and one build of the real app.

No network, no model: the spoken line is composed from the data. Git runs
only against repositories created under tmp_path.
"""
import json
import os
import subprocess
import time
import types
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import jarvis.commander as commander
from jarvis import standup
from jarvis.claude_session import SessionInfo
from jarvis.commander import Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.context import ContextEngine, probe_repo
from jarvis.events import JarvisReply, bus

# the app fixtures (real modules, hardware stubbed) for the wiring test
from tests.test_app_wiring import build, paths, seams  # noqa: F401


# ------------------------------------------------------------- fixtures
def _git(repo, *args, env=None):
    e = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
             GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    e.update(env or {})
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                   env=e)


def _commit(repo, name, subject, when=None):
    (Path(repo) / name).write_text(subject + "\n")
    _git(repo, "add", name)
    env = {}
    if when is not None:
        stamp = when.strftime("%Y-%m-%dT%H:%M:%S")
        env = {"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
    _git(repo, "commit", "-q", "-m", subject, env=env)


@pytest.fixture
def repos(tmp_path):
    """jarvis: two commits today + one yesterday + one dirty file;
    digest: one commit last week, clean; plain: a directory with no git."""
    now = datetime.now().replace(hour=12)
    jarvis = tmp_path / "Jarvis"
    jarvis.mkdir()
    _git(jarvis, "init", "-q", "-b", "jarvis-v3")
    _commit(jarvis, "a.py", "old: yesterday's commit", now - timedelta(days=1))
    _commit(jarvis, "b.py", "standup: walk every repo", now)
    _commit(jarvis, "c.py", "gpu: lend the model to a trainer", now)
    (jarvis / "dirty.py").write_text("x = 1\n")
    _git(jarvis, "add", "dirty.py")
    digest = tmp_path / "haymaker-digest"
    digest.mkdir()
    _git(digest, "init", "-q", "-b", "main")
    _commit(digest, "d.py", "digest: last week", now - timedelta(days=7))
    plain = tmp_path / "plain"
    plain.mkdir()
    return types.SimpleNamespace(jarvis=jarvis, digest=digest, plain=plain,
                                 missing=tmp_path / "projects" / "nope")


# --------------------------------------------------------------- probe
def test_probe_repo_reports_commits_in_the_window_only(repos):
    since, until = standup.window(standup.day_for("today"))
    info = probe_repo(repos.jarvis, since=since, until=until)
    assert info["repo"] == "Jarvis" and info["branch"] == "jarvis-v3"
    assert info["commits"] == ["gpu: lend the model to a trainer",
                               "standup: walk every repo"]
    assert info["changed"] == 1 and "1 file changed" in info["diff_stat"]
    assert info["ahead"] == 0                    # no origin: never a crash
    y_since, y_until = standup.window(standup.day_for("yesterday"))
    assert probe_repo(repos.jarvis, since=y_since, until=y_until)["commits"] == \
        ["old: yesterday's commit"]
    # the prompt path asks without a window and pays for no log walk
    assert probe_repo(repos.jarvis)["commits"] == []


def test_probe_repo_is_none_for_missing_or_plain_dirs(repos):
    assert probe_repo(repos.plain) is None
    assert probe_repo(repos.missing) is None


def test_git_repos_walks_every_configured_repo_primary_first(repos):
    eng = ContextEngine(project_dir=repos.jarvis, vss_dir=repos.plain,
                        repo_dirs=[repos.jarvis, repos.missing, repos.digest,
                                   repos.plain, repos.jarvis])
    since, until = standup.window(standup.day_for("today"))
    out = eng.git_repos(since=since, until=until)
    assert [r["repo"] for r in out] == ["Jarvis", "haymaker-digest"]  # deduped
    assert out[1]["commits"] == [] and out[1]["changed"] == 0
    # the prompt keeps ONE dict shaped as before, from the primary repo
    ctx = eng.get_context("standard")
    assert ctx["git"]["repo"] == "Jarvis" and isinstance(ctx["git"], dict)
    assert "Git: on branch jarvis-v3, 1 file changed and not yet committed" in \
        eng.format_for_prompt(ctx)
    # and "git status" gets the V1 shape from the same repo, not ~/vss_env
    assert eng.git_summary() == {"branch": "jarvis-v3", "changed_files": 1,
                                 "last_commit": ctx["git"]["last_commit"],
                                 "commits_ahead": 0}


def test_git_summary_is_none_without_a_repo(repos):
    eng = ContextEngine(project_dir=repos.plain, vss_dir=repos.missing)
    assert eng.git_summary() is None


# ----------------------------------------------------------- repo list
def test_repo_dirs_from_config_tolerates_a_missing_projects_root(tmp_path):
    from jarvis.assistant_config import AssistantConfig
    cfg = AssistantConfig({"claude": {
        "allowed_dirs": [str(tmp_path / "Jarvis"), str(tmp_path / "gone"),
                         str(tmp_path / "Jarvis")],
        "projects_root": str(tmp_path / "projects")}})
    existing = {str(tmp_path / "Jarvis"), str(tmp_path / "Jarvis" / ".git"),
                str(tmp_path / "vss_env"), str(tmp_path / "vss_env" / ".git"),
                str(tmp_path / "notgit")}
    out = standup.repo_dirs(cfg, extra=[tmp_path / "vss_env", tmp_path / "notgit"],
                            isdir=lambda p: p in existing, listdir=lambda p: [])
    assert out == [str(tmp_path / "Jarvis"), str(tmp_path / "vss_env")]


def test_repo_dirs_walks_the_children_of_projects_root(tmp_path):
    root = tmp_path / "projects"
    existing = {str(root), str(root / "b"), str(root / "b" / ".git"),
                str(root / "a"), str(root / "a" / ".git"), str(root / ".hidden")}
    cfg = types.SimpleNamespace(allowed_dirs=[], projects_root=str(root))
    out = standup.repo_dirs(cfg, isdir=lambda p: p in existing,
                            listdir=lambda p: [".hidden", "b", "a"])
    assert out == [str(root / "a"), str(root / "b")]


# ------------------------------------------------------------ sessions
def _session_file(root, cwd, title, mtime, session_id="s1",
                  first="fix the standup regex"):
    d = root / cwd.replace("/", "-")
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{session_id}.jsonl"
    rows = [{"type": "summary", "summary": title},
            {"type": "user", "cwd": cwd, "sessionId": session_id,
             "message": {"role": "user", "content": first}},
            {"type": "assistant", "cwd": cwd, "sessionId": session_id,
             "message": {"role": "assistant", "content": "Looking."}},
            {"type": "user", "cwd": cwd, "sessionId": session_id,
             "message": {"role": "user", "content": "and the tests"}},
            {"type": "assistant", "cwd": cwd, "sessionId": session_id,
             "message": {"role": "assistant", "content": "Done."}}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    os.utime(path, (mtime, mtime))
    return path


def test_sessions_on_keeps_project_sessions_touched_that_day(tmp_path, monkeypatch):
    projects = tmp_path / "claude-projects"
    home = tmp_path / "home"
    proj = home / "Jarvis"
    proj.mkdir(parents=True)
    # tmp_path itself sits under /tmp, a scratch root: the rule under test
    # is pointed at a scratch dir of its own so the project dir survives
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(standup, "SCRATCH_ROOTS", (str(scratch),))
    now = time.time()
    _session_file(projects, str(proj), "Standup feature", now, "today1")
    _session_file(projects, str(proj), "Old work", now - 86400 * 2, "old")
    # Jarvis's own one-shot web lookups (claude -p) are not Hunter's work
    _session_file(projects, str(proj), "", now, "web",
                  first="You are answering a spoken question for a voice assistant.")
    # scratchpad/probe/subagent transcripts are not projects
    _session_file(projects, str(scratch), "Probe", now, "probe")
    (home / ".claude" / "projects" / "x").mkdir(parents=True)
    _session_file(projects, str(home / ".claude" / "projects" / "x"), "Sub", now, "sub")
    _session_file(projects, str(home), "Home", now, "home")
    today = standup.day_for("today")
    got = standup.sessions_on(today, projects_dir=projects, home=str(home))
    assert [s.session_id for s in got] == ["today1"]
    assert got[0].title == "Standup feature"
    yesterday = standup.day_for("yesterday")
    assert standup.sessions_on(yesterday, projects_dir=projects, home=str(home)) == []
    assert standup.sessions_on(today, projects_dir=tmp_path / "nowhere") == []


@pytest.mark.parametrize("cwd,ok", [
    ("/home/h/Jarvis", True),
    ("/tmp/claude-1000/scratchpad", False),
    ("/home/h/.claude/projects/x", False),
    ("/home/h", False),
    ("", False),
])
def test_is_project_session(cwd, ok):
    assert standup.is_project_session(cwd, home="/home/h") is ok


# ----------------------------------------------------------- rendering
def _sess(title, cwd="/home/h/Jarvis"):
    return SessionInfo(session_id="s", cwd=cwd, mtime=0.0, title=title,
                       first_user="", turns=3)


def test_spoken_line_is_two_sentences_from_the_data():
    repos_ = [{"repo": "Jarvis", "branch": "jarvis-v3", "changed": 3, "ahead": 2,
               "commits": ["a", "b", "c"], "diff_stat": "3 files changed"},
              {"repo": "haymaker-digest", "branch": "main", "changed": 0,
               "ahead": 0, "commits": ["d"], "diff_stat": ""},
              {"repo": "vss_env", "branch": "main", "changed": 0, "ahead": 0,
               "commits": [], "diff_stat": ""}]
    line = standup.spoken_line(repos_, [_sess("Standup feature"),
                                        _sess("TTS shootout")], "today")
    assert line == ("Today you made 4 commits in Jarvis and haymaker-digest, sir, "
                    "with 3 files in Jarvis still uncommitted. Claude had 2 "
                    "sessions today, on Standup feature and TTS shootout.")
    assert line.count(". ") == 1                  # two sentences, no more
    quiet = standup.spoken_line([repos_[2]], [], "yesterday")
    assert quiet == "No commits yesterday, sir. No Claude sessions yesterday."
    assert standup.spoken_line([], [], "today").startswith("I have no repositories")
    one = standup.spoken_line([dict(repos_[1], changed=1)], [_sess("")], "today")
    assert "1 commit in haymaker-digest, sir, with 1 file in haymaker-digest" in one
    assert one.endswith("Claude had 1 session today.")


def test_card_lists_subjects_per_repo_and_the_sessions():
    day = standup.day_for("today")
    repos_ = [{"repo": "Jarvis", "branch": "jarvis-v3", "changed": 1, "ahead": 2,
               "commits": [f"c{i}" for i in range(14)],
               "diff_stat": " 1 file changed, 2 insertions(+)"},
              {"repo": "vss_env", "branch": "main", "changed": 0, "ahead": 0,
               "commits": [], "diff_stat": ""}]
    card = standup.card_text(repos_, [_sess("Standup feature")], "today", day)
    lines = card.splitlines()
    assert lines[0].startswith("Standup — today (")
    assert lines[1] == "Jarvis (jarvis-v3, 2 ahead, 1 file uncommitted)"
    assert lines[2] == "  - c0" and "  … and 2 more" in lines
    assert "  uncommitted: 1 file changed, 2 insertions(+)" in lines
    assert "vss_env (main, clean) — no commits today" in lines
    assert lines[-2] == "Claude sessions today:"
    assert lines[-1] == "  - Jarvis: Standup feature"
    empty = standup.card_text([], [], "yesterday", day)
    assert "No git repositories configured" in empty
    assert empty.endswith("No Claude sessions yesterday.")


def test_build_puts_the_repo_walk_and_sessions_together(repos, tmp_path, monkeypatch):
    calls = []

    def git_repos(since=None, until=None):
        calls.append((since, until))
        return [probe_repo(repos.jarvis, since=since, until=until)]

    line, card = standup.build(git_repos, "today",
                               projects_dir=tmp_path / "no-projects")
    day = standup.day_for("today")
    assert calls == [standup.window(day)]
    assert line.startswith("Today you made 2 commits in Jarvis, sir, with 1 file in "
                           "Jarvis still uncommitted.")
    assert "  - standup: walk every repo" in card


@pytest.mark.parametrize("text,word", [
    ("standup", "today"), ("stand up", "today"), ("daily standup", "today"),
    ("what did i do today", "today"), ("what did i do yesterday", "yesterday"),
    ("what did i change today", "today"), ("what have i done today", "today"),
    ("what did i work on yesterday", "yesterday"), ("what did i get done", "today"),
    ("standup for yesterday", "yesterday"), ("yesterday's standup", "yesterday"),
    ("what did i commit today", "today"), ("give me the standup", "today"),
])
def test_standup_phrases(text, word):
    m = standup.STANDUP_RX.match(text)
    assert m and standup.day_word(m) == word


@pytest.mark.parametrize("text", [
    "what did i say about the move", "stand up straight", "what do i do today",
    "standup meeting at 10", "what did i eat today", "what did i tell you about x",
])
def test_standup_regex_leaves_other_questions_alone(text):
    assert standup.STANDUP_RX.match(text) is None


# ----------------------------------------------------------- commander
@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = types.SimpleNamespace(desktop=MagicMock(), workflows=MagicMock(),
                                brain=MagicMock(), memory=MagicMock(),
                                context=MagicMock(), tts=MagicMock())
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.context.git_repos = lambda since=None, until=None: [
        {"repo": "Jarvis", "branch": "jarvis-v3", "changed": 0, "ahead": 1,
         "commits": ["standup: walk every repo"], "diff_stat": ""}]
    monkeypatch.setattr(standup, "sessions_on",
                        lambda day, projects_dir=None, home=None, reader=None: [])
    return Commander(svc)


def test_standup_command_speaks_the_line_and_shows_the_card(cmdr):
    got = []
    bus.subscribe(JarvisReply, got.append)
    try:
        res = cmdr.handle("jarvis, what did I do today", source="voice")
    finally:
        bus.unsubscribe(JarvisReply, got.append)
    assert res.handled and res.speak and res.status == "Standup"
    assert res.reply == ("Today you made 1 commit in Jarvis, sir. "
                         "No Claude sessions today.")
    bus.drain()
    cards = [e for e in got if e.text.startswith("Standup —")]
    assert len(cards) == 1 and not cards[0].speak
    assert "  - standup: walk every repo" in cards[0].text
    assert "Jarvis (jarvis-v3, 1 ahead, clean)" in cards[0].text


def test_standup_is_tier_one_without_the_prefix(cmdr):
    assert "standup" in [c.name for c in commander.ASSISTANT_TIER1]
    res = cmdr.handle("standup", source="typed")
    assert res.handled and res.status == "Standup"
    cmdr.services.brain.chat.assert_not_called()


def test_standup_failure_is_an_excuse_not_a_traceback(cmdr, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("git exploded")
    monkeypatch.setattr(standup, "build", boom)
    res = cmdr.handle("jarvis, standup", source="typed")
    assert res.reply == standup.FAILED_LINE and res.speak


def test_standup_needs_the_v3_probe(cmdr):
    del cmdr.services.context.git_repos
    cmdr.services.context.git_repos = None
    res = cmdr.handle("jarvis, what did I do yesterday", source="typed")
    assert res.status != "Standup"           # fell through to the next stage


# --------------------------------------------------------- app wiring
def test_the_app_wires_git_status_and_the_standup_to_the_v3_probe(build, repos):  # noqa: F811
    a = build()
    a.context.repo_dirs = [str(repos.jarvis), str(repos.digest)]
    assert a.services.context.git_summary()["branch"] == "jarvis-v3"
    res = a.commander.handle("jarvis, what did I do today", source="typed")
    assert res.status == "Standup"
    assert res.reply.startswith("Today you made 2 commits in Jarvis, sir")
    for fn in ("release", "reclaim", "is_lent"):
        assert callable(getattr(a.services.brain, fn))
