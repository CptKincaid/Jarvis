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


# =====================================================================
# Part 2 -- the drawer: Services fields, the two-press arm, the wiring.
# Tk-free the way tests/test_ui_chrome.py builds views: the drawer is made
# with __new__ over fakes for the two widgets it touches and for after().
# =====================================================================
import threading  # noqa: E402
from dataclasses import fields  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from jarvis import relaunch  # noqa: E402
from jarvis.ui.main_window import Services  # noqa: E402
from jarvis.ui import views as views_mod  # noqa: E402
from jarvis.ui.views import RESTART_ARM_MS, RestartArm, SettingsDrawer  # noqa: E402


def test_services_declare_restart_and_code_status_unwired_by_default():
    names = {f.name for f in fields(Services)}
    assert {"restart", "code_status"} <= names
    svc = Services()
    assert svc.restart is None and svc.code_status is None


def test_the_drawer_formats_with_the_same_function_the_app_exports():
    assert views_mod.format_code_status is relaunch.format_code_status
    assert app_mod.format_code_status is relaunch.format_code_status


# ------------------------------------------------------------ RestartArm
class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_first_press_arms_second_press_within_the_window_fires():
    clock = _Clock()
    arm = RestartArm(window_s=5.0, clock=clock)
    assert arm.armed is False and arm.label == "Restart Jarvis"
    assert arm.press() == "armed"
    assert arm.armed is True and arm.label == "Press again to restart"
    clock.t += 4.9
    assert arm.press() == "fire"
    assert arm.armed is False and arm.label == "Restart Jarvis"


def test_a_press_after_the_window_re_arms_instead_of_firing():
    """Belt and braces with the Tk timer: even if after() has not run
    yet, a press 5.1 s later is a first press again, never a restart."""
    clock = _Clock()
    arm = RestartArm(window_s=5.0, clock=clock)
    arm.press()
    clock.t += 5.1
    assert arm.armed is False
    assert arm.press() == "armed"
    clock.t += 1.0
    assert arm.press() == "fire"


def test_expire_disarms():
    arm = RestartArm(window_s=5.0, clock=_Clock())
    arm.press()
    arm.expire()
    assert arm.armed is False
    assert arm.press() == "armed"


def test_the_arm_window_matches_the_tk_timer():
    assert RESTART_ARM_MS == 5000
    assert RestartArm().window_s == 5.0


# --------------------------------------------------------- drawer wiring
class _FakeButton:
    def __init__(self):
        self.texts = []

    def set_text(self, text):
        self.texts.append(text)


class _FakeLabel:
    def __init__(self):
        self.texts = []

    def configure(self, **kw):
        if "text" in kw:
            self.texts.append(kw["text"])


class _FakeToast:
    def __init__(self):
        self.shown = []

    def show(self, text, kind="info", ms=1800):
        self.shown.append((text, kind))


def _drawer(services, clock=None):
    d = SettingsDrawer.__new__(SettingsDrawer)       # no tk.Frame.__init__
    d.services = services
    d.toast = _FakeToast()
    d.after_calls = []
    d.cancelled = []
    d.after = lambda ms, fn: d.after_calls.append((ms, fn)) or f"job{len(d.after_calls)}"
    d.after_cancel = lambda job: d.cancelled.append(job)
    d._restart_arm = RestartArm(RESTART_ARM_MS / 1000.0,
                                clock=clock or _Clock())
    d._restart_job = None
    d._restart_btn = _FakeButton()
    d._code_status_lbl = _FakeLabel()
    return d


def test_one_press_arms_the_button_and_schedules_the_disarm_never_restarts():
    fired = []
    d = _drawer(SimpleNamespace(restart=lambda: fired.append(1)))
    d._restart_pressed()
    assert d._restart_btn.texts == ["Press again to restart"]
    assert d.after_calls == [(5000, d._restart_expired)]
    assert fired == []
    assert d.toast.shown == []


def test_the_timer_puts_the_label_back_and_the_next_press_only_arms():
    fired = []
    d = _drawer(SimpleNamespace(restart=lambda: fired.append(1)))
    d._restart_pressed()
    d._restart_expired()                        # what after() runs at 5 s
    assert d._restart_btn.texts[-1] == "Restart Jarvis"
    d._restart_pressed()
    assert d._restart_btn.texts[-1] == "Press again to restart"
    assert fired == []


def test_the_second_press_calls_the_restart_service_off_the_tk_thread():
    done = threading.Event()
    threads = []

    def restart():
        threads.append(threading.current_thread().name)
        done.set()
    d = _drawer(SimpleNamespace(restart=restart))
    d._restart_pressed()
    d._restart_pressed()
    assert done.wait(2.0), "restart service never ran"
    assert threads and threads[0] != threading.main_thread().name
    # the arm is spent, the timer cancelled, the label back, and he was told
    assert d.cancelled == ["job1"]
    assert d._restart_btn.texts[-1] == "Restart Jarvis"
    assert d.toast.shown and d.toast.shown[-1][0].startswith("Restarting")


def test_without_a_restart_service_the_toast_says_not_wired():
    d = _drawer(SimpleNamespace())                # no .restart attribute
    d._restart_pressed()
    d._restart_pressed()
    assert d.toast.shown == [("Restart not wired", "warn")]
    d = _drawer(None)
    d._restart_pressed()
    d._restart_pressed()
    assert d.toast.shown == [("Restart not wired", "warn")]


def test_a_failing_restart_service_is_logged_not_raised():
    done = threading.Event()

    def restart():
        done.set()
        raise RuntimeError("boom")
    d = _drawer(SimpleNamespace(restart=restart))
    d._restart_pressed()
    d._restart_pressed()
    assert done.wait(2.0)


def test_code_status_text_reads_the_service_and_degrades_to_unknown():
    d = _drawer(SimpleNamespace(code_status=lambda: {
        "running": "ac934b0", "disk": "7539478", "behind": 2, "dirty": False}))
    assert d._code_status_text() == \
        "Running ac934b0, on disk 7539478 (2 commits behind)"
    assert _drawer(SimpleNamespace())._code_status_text() == "Running unknown"
    assert _drawer(None)._code_status_text() == "Running unknown"

    def broken():
        raise RuntimeError("git exploded")
    assert _drawer(SimpleNamespace(code_status=broken))._code_status_text() == \
        "Running unknown"


def test_open_refreshes_the_code_status_line_so_it_is_current_when_he_looks():
    answers = [{"running": "a", "disk": "a", "behind": 0, "dirty": False},
               {"running": "a", "disk": "b", "behind": 3, "dirty": True}]
    d = _drawer(SimpleNamespace(code_status=lambda: answers.pop(0)))
    for name in ("_refresh_privacy", "lift", "focus_set"):
        setattr(d, name, lambda *a, **k: None)
    d.place = lambda *a, **k: None
    d._slide = lambda *a, **k: None
    d.host = None
    d._x = 0
    d.WIDTH = 320
    d._open = False
    d.open()
    assert d._code_status_lbl.texts == ["Running a, on disk a (up to date)"]
    d._open = False
    d.open()
    assert d._code_status_lbl.texts[-1] == \
        "Running a, on disk b (3 commits behind), uncommitted changes"


def test_the_system_section_builds_the_status_row_and_the_button():
    """The wiring, not the widgets: _build_sections must put the status
    line and the arming button in System, through the row helpers every
    other section uses. Read from the source so no Tk root is made."""
    import inspect
    src = inspect.getsource(SettingsDrawer._build_sections)
    sys_at = src.index('self._section("System")')
    assert "_code_status_lbl = self._info_row(" in src[sys_at:]
    assert "self._restart_row(box)" in src[sys_at:]
