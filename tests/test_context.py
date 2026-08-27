"""Tests for jarvis.context — format_for_prompt renders every gathered key
(tmp git repo fixture), session injection, window tracking."""
import subprocess

import pytest

from jarvis.context import ContextEngine

# Keys get_context() gathers at each detail level — the render test below
# guarantees none of them is gathered-but-never-rendered (the V1 bug).
MINIMAL_KEYS = {"time", "active_window"}
STANDARD_KEYS = MINIMAL_KEYS | {"git", "recent_files", "conversation",
                                "sessions"}
FULL_KEYS = STANDARD_KEYS | {"system", "errors", "screen", "processes"}


class FakeMemory:
    def get_recent_sessions(self, n=3):
        return [{"time": "2026-08-24T21:14:00",
                 "summary": "ported the memory module"}]


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
            cwd=repo, check=True, capture_output=True)

    git("init", "-b", "main")
    (repo / "hello.py").write_text("print('hi')\n")
    git("add", "-A")
    git("commit", "-m", "initial commit")
    return repo


@pytest.fixture
def engine(git_repo, tmp_path):
    return ContextEngine(project_dir=git_repo, vss_dir=tmp_path / "novss",
                         memory=FakeMemory())


# ------------------------------------------------------------- gathering
def test_gathered_key_sets(engine):
    assert set(engine.get_context("minimal")) == MINIMAL_KEYS
    assert set(engine.get_context("standard")) == STANDARD_KEYS
    assert set(engine.get_context("full")) == FULL_KEYS


def test_git_context_from_tmp_repo(engine):
    ctx = engine.get_context("standard")
    g = ctx["git"]
    assert g["branch"] == "main"
    assert g["changed"] == 0
    assert "initial commit" in g["last_commit"]
    rendered = engine.format_for_prompt(ctx)
    assert "Git: on branch main, nothing uncommitted" in rendered
    assert "initial commit" in rendered


def test_recent_files_from_tmp_repo(engine):
    ctx = engine.get_context("standard")
    assert any("hello.py" in f for f in ctx["recent_files"])
    assert "hello.py" in engine.format_for_prompt(ctx)


def test_spoken_rendering_counts_files_instead_of_naming_them(engine):
    """Tier 2 (read aloud) gets a count, so there is nothing to recite."""
    ctx = {"time": "10:00 AM",
           "git": {"branch": "main", "changed": 40,
                   "last_commit": "6bfe446 V3 overhaul", "ahead": 0},
           "recent_files": ["1.0 /r/brain.py", "2.0 /r/tts.py"]}
    spoken = engine.format_for_prompt(ctx, spoken=True)
    assert "Recently modified: 2 project files" in spoken
    assert "brain.py" not in spoken and "tts.py" not in spoken
    assert ("Git: on branch main, 40 files changed and not yet committed, "
            "last commit: V3 overhaul") in spoken     # hash not read aloud
    written = engine.format_for_prompt(ctx)          # Tier 3 keeps names
    assert "brain.py, tts.py" in written
    assert "last commit: 6bfe446 V3 overhaul" in written  # Tier 3 keeps hash
    one = dict(ctx, git=dict(ctx["git"], changed=1),
               recent_files=["1.0 /r/brain.py"])
    assert "1 file changed and not yet committed" in \
        engine.format_for_prompt(one)
    assert "Recently modified: 1 project file\n" in \
        engine.format_for_prompt(one, spoken=True) + "\n"


def test_sessions_injected_from_memory(engine):
    ctx = engine.get_context("standard")
    assert ctx["sessions"][0]["summary"] == "ported the memory module"
    assert "ported the memory module" in engine.format_for_prompt(ctx)


def test_no_memory_means_empty_sessions(git_repo, tmp_path):
    eng = ContextEngine(project_dir=git_repo, vss_dir=tmp_path)
    assert eng.get_context("standard")["sessions"] == []


# ------------------------------------------------------------- rendering
def test_format_renders_every_gathered_key(engine):
    """Every key get_context('full') gathers must be rendered when truthy."""
    sentinels = {
        "time": "11:58 PM, Tuesday August 25 2026",
        "active_window": "SENTINEL-WINDOW",
        "git": {"repo": "r", "branch": "SENTINEL-BRANCH", "changed": 2,
                "last_commit": "abc SENTINEL-COMMIT", "ahead": 1},
        "recent_files": ["123.0 /tmp/SENTINEL-FILE.py"],
        "conversation": [{"user": "SENTINEL-USER-LINE",
                          "jarvis": "SENTINEL-JARVIS-LINE"}],
        "sessions": [{"time": "2026-08-24T21:14:00",
                      "summary": "SENTINEL-SESSION"}],
        "system": "GPU: 7% util, 55C | RAM: SENTINEL-RAM",
        "errors": ["ERROR SENTINEL-ERROR-LINE"],
        "screen": "/tmp/vss_screen/SENTINEL.png",
        "processes": ["SENTINEL-PROC (CPU:9%)"],
    }
    assert set(sentinels) == FULL_KEYS  # keep this test honest vs gather

    rendered = engine.format_for_prompt(sentinels)
    for key, marker in [
        ("time", "11:58 PM"),
        ("active_window", "SENTINEL-WINDOW"),
        ("git", "SENTINEL-BRANCH"),
        ("recent_files", "SENTINEL-FILE.py"),
        ("conversation", "SENTINEL-USER-LINE"),
        ("sessions", "SENTINEL-SESSION"),
        ("system", "SENTINEL-RAM"),
        ("errors", "SENTINEL-ERROR-LINE"),
        ("screen", "SENTINEL.png"),
        ("processes", "SENTINEL-PROC"),
    ]:
        assert marker in rendered, f"key '{key}' gathered but not rendered"


def test_format_skips_falsy_values(engine):
    ctx = {"time": "10:00 AM", "active_window": "", "git": {},
           "recent_files": [], "conversation": [], "sessions": [],
           "system": "", "errors": [], "screen": None, "processes": []}
    rendered = engine.format_for_prompt(ctx)
    assert rendered == "Current time: 10:00 AM"


# ---------------------------------------------------------- conversation
def test_add_exchange_and_cap(engine):
    for i in range(25):
        engine.add_exchange(f"question {i}", f"answer {i}")
    assert len(engine._conversation) == 20
    ctx = engine.get_context("standard")
    assert len(ctx["conversation"]) == 10
    assert "question 24" in engine.format_for_prompt(ctx)


# ------------------------------------------------------- window tracking
def test_window_tracking(engine):
    assert engine.get_last_window() is None
    engine.track_window("Firefox")
    engine.track_window("Terminal")
    assert engine.get_last_window() == "Firefox"
    assert engine.current_app == "Terminal"


# ------------------------------------------------------------ system bits
def test_read_meminfo_parses_proc():
    info = ContextEngine._read_meminfo()
    assert info["MemTotal"] > 0
    assert "MemAvailable" in info
