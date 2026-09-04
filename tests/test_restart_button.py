"""The Restart button (Hunter, 2026-09-04: "yes, button only").

Part 1 -- the code-status line the drawer shows above the button: the pure
formatter, the probe through a fake git, and the probe against a REAL git
repository built in tmp (so the exact rev-list / diff arguments are proven
against git itself, not against a mock that agrees with the code).

Nothing here goes near the live process: no pid file, no kill(), no spawn.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import jarvis.app as app_mod
from jarvis.app import format_code_status, probe_code_status


# ------------------------------------------------------- format (pure)
def test_format_up_to_date():
    assert format_code_status({"running": "7539478", "disk": "7539478",
                               "behind": 0, "dirty": False}) == \
        "Running 7539478, on disk 7539478 (up to date)"


def test_format_behind_counts_commits_singular_and_plural():
    assert format_code_status({"running": "ac934b0", "disk": "7539478",
                               "behind": 2, "dirty": False}) == \
        "Running ac934b0, on disk 7539478 (2 commits behind)"
    assert format_code_status({"running": "ac934b0", "disk": "7539478",
                               "behind": 1, "dirty": False}) == \
        "Running ac934b0, on disk 7539478 (1 commit behind)"


def test_format_appends_uncommitted_changes_when_dirty():
    assert format_code_status({"running": "7539478", "disk": "7539478",
                               "behind": 0, "dirty": True}) == \
        "Running 7539478, on disk 7539478 (up to date), uncommitted changes"
    assert format_code_status({"running": "ac934b0", "disk": "7539478",
                               "behind": 2, "dirty": True}).endswith(
        "(2 commits behind), uncommitted changes")


def test_format_says_running_unknown_without_git():
    assert format_code_status({"running": "", "disk": "", "behind": None,
                               "dirty": False}) == "Running unknown"
    assert format_code_status({}) == "Running unknown"


def test_format_names_a_running_commit_that_is_not_an_ancestor():
    """behind=None with two different hashes: the tree was rebased or
    switched under him, so a count would be a lie."""
    line = format_code_status({"running": "ac934b0", "disk": "7539478",
                               "behind": None, "dirty": False})
    assert line.startswith("Running ac934b0, on disk 7539478 (")
    assert "behind" not in line and "up to date" not in line


# ------------------------------------------------------ probe (fake git)
class _FakeGit:
    """Answers `git <args>` from a table; records what was asked."""

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    def __call__(self, repo, *args, timeout=5):
        self.asked.append((str(repo), args))
        return self.answers.get(args, "")


def test_probe_reads_disk_behind_and_dirty_through_git():
    git = _FakeGit({
        ("rev-parse", "--short", "HEAD"): "7539478",
        ("rev-list", "--count", "ac934b0..HEAD"): "2",
        ("rev-list", "--count", "HEAD..ac934b0"): "0",
        ("diff", "HEAD", "--shortstat"): " 1 file changed, 3 insertions(+)",
    })
    st = probe_code_status("ac934b0", "/repo", git=git)
    assert st == {"running": "ac934b0", "disk": "7539478", "behind": 2,
                  "dirty": True}
    assert all(repo == "/repo" for repo, _ in git.asked)


def test_probe_behind_is_none_when_running_is_not_an_ancestor():
    git = _FakeGit({
        ("rev-parse", "--short", "HEAD"): "7539478",
        ("rev-list", "--count", "ac934b0..HEAD"): "5",
        ("rev-list", "--count", "HEAD..ac934b0"): "1",   # it has its own
        ("diff", "HEAD", "--shortstat"): "",
    })
    st = probe_code_status("ac934b0", "/repo", git=git)
    assert st["behind"] is None and st["dirty"] is False


def test_probe_tolerates_no_git_at_all():
    git = _FakeGit({})
    st = probe_code_status("", "/nowhere", git=git)
    assert st == {"running": "", "disk": "", "behind": None, "dirty": False}
    # and no rev-list is attempted without a hash to compare
    assert not any(a and a[0] == "rev-list" for _, a in git.asked)


# ------------------------------------------------------ probe (real git)
def _git(repo: Path, *args) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                       text=True, timeout=20,
                       env={"PATH": "/usr/bin:/bin",
                            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                            "GIT_COMMITTER_NAME": "t",
                            "GIT_COMMITTER_EMAIL": "t@t",
                            "HOME": str(repo)})
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A real repository: three commits on main, the running commit being
    the first."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    heads = []
    for n in range(3):
        (repo / "f.txt").write_text(f"v{n}\n")
        _git(repo, "add", "f.txt")
        _git(repo, "commit", "-q", "-m", f"c{n}")
        heads.append(_git(repo, "rev-parse", "--short", "HEAD"))
    return repo, heads


def test_probe_against_a_real_repository(repo):
    from jarvis.context import _git as real_git
    path, heads = repo
    st = probe_code_status(heads[0], path, git=real_git)
    assert st == {"running": heads[0], "disk": heads[2], "behind": 2,
                  "dirty": False}
    assert format_code_status(st) == \
        f"Running {heads[0]}, on disk {heads[2]} (2 commits behind)"
    # HEAD itself: up to date
    st = probe_code_status(heads[2], path, git=real_git)
    assert st["behind"] == 0 and format_code_status(st).endswith("(up to date)")
    # an edit on top: dirty
    (path / "f.txt").write_text("edited\n")
    st = probe_code_status(heads[2], path, git=real_git)
    assert st["dirty"] is True
    assert format_code_status(st).endswith("(up to date), uncommitted changes")


def test_probe_against_a_real_repository_after_a_rebase(repo):
    """The running commit was rewritten (amend): it is no longer on HEAD's
    line at all, and the probe must say so with None rather than a count."""
    from jarvis.context import _git as real_git
    path, heads = repo
    _git(path, "commit", "-q", "--amend", "-m", "c2 rewritten")
    st = probe_code_status(heads[2], path, git=real_git)
    assert st["running"] == heads[2] and st["disk"] != heads[2]
    assert st["behind"] is None
    # a hash git has never heard of (a gc'd rewrite): still None, no crash
    st = probe_code_status("0000000", path, git=real_git)
    assert st["behind"] is None and st["disk"] != ""


def test_running_commit_is_read_from_the_repo_with_the_same_helper(repo):
    from jarvis.context import _git as real_git
    path, heads = repo
    assert app_mod.running_commit(repo=path, git=real_git) == heads[2]
    assert app_mod.running_commit(repo=path / "nope", git=real_git) == ""
