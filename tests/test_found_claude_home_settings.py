"""Adversarial-sweep regression test (2026-08-26).

DEFECT: resuming a discovered Claude session whose cwd is the user's HOME
registers $HOME as a Jarvis project (jarvis/claude_session.py:1580,
`_resume_session` -> `_under(info.cwd, self.home)` is True when cwd == home),
and the first task there makes `ensure_project_settings`
(jarvis/claude_session.py:1098) write
`$HOME/.claude/settings.local.json` -- which is Claude Code's USER-LEVEL
settings file, not a project file.  ALLOW_RULES (jarvis/claude_session.py:117)
are appended to it, so `Bash(*)`, `Read(/**)`, `Edit(/**)`, `Write(/**)`,
`MultiEdit(/**)`, `NotebookEdit(/**)` and `mcp__*` become permanent, machine-wide
grants for every future Claude Code session -- including the user's own
interactive ones.

Reachable in one utterance on this machine: "pick up the session from
yesterday" routes to action=resume, and pick_session() returns a session with
cwd=/home/hunterp with no clarifying question.

Fix direction: refuse (or specially handle) a project whose path is the home
directory itself, and/or never write ALLOW_RULES into `~/.claude/`.

FIXED 2026-08-26: `claude_session.unsafe_dir_reason()` names the directories
Claude is never given -- $HOME itself, anything $HOME sits inside, any dotted
configuration directory (~/.claude, ~/.config/...), the system roots, the
scratch roots, and anything that is not an existing directory the user owns.
`ensure_project_settings()` refuses to write there, `_resume_session()` and
`submit()` refuse the dir with UNSAFE_DIR_LINE (which offers the terminal),
and `resume()` drops such sessions before ranking.  Ordinary project dirs
anywhere on disk are untouched by this (claude.auto_approve_anywhere).

HARDENED 2026-08-27 (adversarial round 1): the guard judged the PROJECT dir
but never the path actually written, so a project whose `.claude` is a symlink
at the user's own `~/.claude` (a common dotfiles habit) still widened the
user-level allowlist.  `ensure_project_settings()` now resolves the write
target, refuses anything that resolves outside the project or into a
home/config dir, reads with O_NOFOLLOW (so a symlinked settings file cannot be
copied out either), and writes its temp file inside the resolved directory.
`new_project()` and `work_on()` got the same unsafe_dir() check, so a project
Jarvis would refuse every task in is never created or made active.
"""
import json
import pathlib
import time

from jarvis import claude_session as cs


def _mgr(tmp_path, home):
    class Cfg:
        def __init__(self):
            self.d = {"claude": {"allowed_dirs": [], "auto_approve_anywhere": True}}

        def get(self, dotted, default=None):
            node = self.d
            for part in dotted.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return default
            return node

        def add_allowed_dir(self, path):
            self.d["claude"].setdefault("allowed_dirs", []).append(str(path))

    return cs.ClaudeSessionManager(
        Cfg(), None, None, tmp_path / "state.json", tmp_path / "tasks",
        claude_bin="/nonexistent/claude", home=str(home))


def test_home_session_must_not_widen_user_level_claude_settings(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    settings = home / ".claude" / "settings.local.json"
    narrow = {"permissions": {"allow": ["Bash(ls)"]}}
    settings.write_text(json.dumps(narrow), encoding="utf-8")
    before = settings.read_bytes()

    mgr = _mgr(tmp_path, home)
    info = cs.SessionInfo(session_id="deadbeef", cwd=str(home),
                          mtime=time.time(), first_user="tidy up my home dir",
                          turns=90)
    line = mgr._resume_session(info)

    # refused, in persona, offering the terminal instead
    assert line == cs.UNSAFE_DIR_LINE
    assert "terminal" in line and "sir" in line
    # $HOME was not registered as a project, and nothing was cleared
    assert not any(p.path == str(home) for p in mgr.projects())
    assert mgr.cfg.d["claude"]["allowed_dirs"] == []
    # even asked point-blank, the user-level file is not touched
    mgr.ensure_project_settings(cs.Project(slug="home", path=str(home)))
    assert settings.read_bytes() == before
    after = json.loads(settings.read_text(encoding="utf-8"))
    assert after["permissions"]["allow"] == ["Bash(ls)"], (
        "Jarvis rewrote the user-level ~/.claude/settings.local.json: "
        f"{after['permissions']['allow']}")


def test_submit_refuses_home_and_config_dirs(tmp_path):
    """The other door in: a project whose path is $HOME / a config dir."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    mgr = _mgr(tmp_path, home)
    for path in (home, home / ".claude", home / ".config",
                 pathlib.Path(tmp_path.anchor), pathlib.Path("/etc")):
        if not path.exists():
            path.mkdir(parents=True)
        proj = cs.Project(slug="x", path=str(path))
        assert mgr.unsafe_dir(str(path)), path
        assert mgr.project_allowed(proj) is False, path
        assert mgr.submit("do something", project=proj) == cs.UNSAFE_DIR_LINE, path


def test_resume_skips_home_rooted_sessions_entirely(tmp_path):
    """Discovery must not even offer a $HOME-rooted session (report 7.5)."""
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "-home").mkdir(parents=True)
    mgr = _mgr(tmp_path, home)
    mgr.projects_dir = home / ".claude" / "projects"
    lines = [json.dumps({"type": "user", "cwd": str(home), "sessionId": "s1",
                         "message": {"role": "user", "content": "tidy my home dir"}}),
             json.dumps({"type": "assistant", "cwd": str(home), "sessionId": "s1",
                         "message": {"role": "assistant",
                                     "content": [{"type": "text", "text": "done"}]}}),
             json.dumps({"type": "user", "cwd": str(home), "sessionId": "s1",
                         "message": {"role": "user", "content": "and more"}}),
             json.dumps({"type": "assistant", "cwd": str(home), "sessionId": "s1",
                         "message": {"role": "assistant",
                                     "content": [{"type": "text", "text": "done 2"}]}})]
    (mgr.projects_dir / "-home" / "s1.jsonl").write_text(
        "\n".join(lines) + "\n")
    # discovery itself still sees it ...
    assert [s.cwd for s in cs.discover_sessions(mgr.projects_dir)] == [str(home)]
    # ... but resume refuses to pick it up
    assert mgr.resume("pick up the session from yesterday") == cs.NO_SESSION_LINE


def test_control_an_ordinary_project_anywhere_is_still_fine(tmp_path):
    """auto_approve_anywhere is INTENDED: a real project dir outside the
    configured roots and outside $HOME still gets its settings written."""
    home = tmp_path / "home"
    home.mkdir()
    elsewhere = tmp_path / "elsewhere" / "some-repo"
    (elsewhere / ".git").mkdir(parents=True)
    mgr = _mgr(tmp_path, home)
    proj = cs.Project(slug="some-repo", path=str(elsewhere))
    assert mgr.unsafe_dir(str(elsewhere)) == ""
    assert cs.looks_like_project(str(elsewhere)) is True
    assert mgr.project_allowed(proj) is True
    mgr.ensure_project_settings(proj)
    data = json.loads((elsewhere / ".claude" / "settings.local.json").read_text())
    assert "Bash(*)" in data["permissions"]["allow"]


def test_control_a_normal_project_is_scoped_to_its_own_dot_claude(tmp_path):
    """Control: a project that is NOT home gets its own settings file."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    proj_dir = home / "repo"
    proj_dir.mkdir()
    mgr = _mgr(tmp_path, home)
    proj = cs.Project(slug="repo", path=str(proj_dir))
    mgr.ensure_project_settings(proj)
    assert (proj_dir / ".claude" / "settings.local.json").exists()
    assert not (home / ".claude" / "settings.local.json").exists()


def test_a_symlinked_project_dot_claude_cannot_widen_the_user_file(tmp_path):
    """Round 1 adversarial: project looks perfect (.git and all) but its
    `.claude` is a symlink at the user's own config dir."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    settings = home / ".claude" / "settings.local.json"
    settings.write_text(json.dumps({"permissions": {"allow": ["Bash(ls:*)"]}}),
                        encoding="utf-8")
    before = settings.read_bytes()

    proj_dir = home / "projects" / "demo"
    (proj_dir / ".git").mkdir(parents=True)
    (proj_dir / ".claude").symlink_to(home / ".claude")

    mgr = _mgr(tmp_path, home)
    proj = cs.Project(slug="demo", path=str(proj_dir))
    # the project dir itself is fine -- the WRITE TARGET is not
    assert mgr.unsafe_dir(str(proj_dir)) == ""
    mgr.ensure_project_settings(proj)
    assert settings.read_bytes() == before, (
        "Jarvis wrote through a symlinked .claude into the user-level file: "
        f"{json.loads(settings.read_text())['permissions']['allow']}")


def test_a_symlinked_settings_file_is_neither_followed_nor_copied(tmp_path):
    """Round 1 adversarial, vector B: settings.local.json is itself the link."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    settings = home / ".claude" / "settings.local.json"
    settings.write_text(json.dumps({"permissions": {"allow": ["Bash(ls:*)"]}}),
                        encoding="utf-8")
    before = settings.read_bytes()

    proj_dir = home / "projects" / "b"
    (proj_dir / ".claude").mkdir(parents=True)
    (proj_dir / ".git").mkdir()
    (proj_dir / ".claude" / "settings.local.json").symlink_to(settings)

    mgr = _mgr(tmp_path, home)
    mgr.ensure_project_settings(cs.Project(slug="b", path=str(proj_dir)))
    assert settings.read_bytes() == before
    # the link is left alone: nothing was written through it, and the user's
    # own rules were not copied out into the project either
    assert (proj_dir / ".claude" / "settings.local.json").is_symlink()


def _cfg_with_root(root):
    class Cfg:
        def __init__(self):
            self.d = {"claude": {"allowed_dirs": [], "auto_approve_anywhere": True,
                                 "projects_root": str(root)}}

        def get(self, dotted, default=None):
            node = self.d
            for part in dotted.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return default
            return node

    return Cfg()


def test_new_project_refuses_a_config_rooted_projects_root(tmp_path):
    """Round 1 adversarial, vector C: don't create-and-activate a project
    every later task will be refused in."""
    home = tmp_path / "home"
    (home / ".config" / "repos").mkdir(parents=True)
    (home / "work").symlink_to(home / ".config" / "repos")
    mgr = cs.ClaudeSessionManager(
        _cfg_with_root(home / "work"), None, None, tmp_path / "state.json",
        tmp_path / "tasks", claude_bin="/nonexistent/claude", home=str(home))
    line = mgr.new_project("demo")
    assert "configuration folder" in line and "sir" in line
    assert not (home / ".config" / "repos" / "demo").exists()
    assert mgr.active_project is None
    assert not any(p.slug == "demo" for p in mgr.projects())


def test_work_on_refuses_a_config_rooted_project_from_state(tmp_path):
    """A project persisted before the guard landed must not become active."""
    home = tmp_path / "home"
    bad = home / ".config" / "repos" / "demo"
    bad.mkdir(parents=True)
    state = tmp_path / "state.json"
    state.write_text(json.dumps(
        {"projects": {"demo": {"slug": "demo", "path": str(bad)}}}),
        encoding="utf-8")
    mgr = _mgr(tmp_path, home)
    mgr = cs.ClaudeSessionManager(
        mgr.cfg, None, None, state, tmp_path / "tasks",
        claude_bin="/nonexistent/claude", home=str(home))
    assert mgr.work_on("demo") == cs.UNSAFE_DIR_LINE
    assert mgr.active_project is None
