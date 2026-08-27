"""Tests for jarvis.autostart — the GNOME autostart entry and never-sleep.

Firewall: HOME is a tmp dir for every test, so ~/.config/autostart of the
user is never written; gsettings is a recorded seam, never executed.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import jarvis.autostart as auto

REAL_DESKTOP = Path(os.path.expanduser("~")) / ".config" / "autostart" / "jarvis.desktop"


def _exists_stamp(p: Path):
    try:
        return p.stat().st_mtime_ns
    except FileNotFoundError:
        return None


@pytest.fixture(scope="module", autouse=True)
def _real_autostart_untouched():
    before = _exists_stamp(REAL_DESKTOP)
    yield
    assert _exists_stamp(REAL_DESKTOP) == before, "test touched the real autostart entry"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


class Recorder:
    """A fake `_run`: records argv, answers from a script."""

    def __init__(self, answers=None):
        self.calls = []
        self.answers = list(answers or [])

    def __call__(self, argv, timeout=10.0):
        self.calls.append(list(argv))
        if self.answers:
            item = self.answers.pop(0)
            if isinstance(item, BaseException):
                raise item
            rc, out = item
            return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


# ------------------------------------------------------------- desktop
def test_desktop_path_follows_home(home):
    assert auto.desktop_path() == home / ".config" / "autostart" / "jarvis.desktop"
    assert auto.desktop_path("~/x.desktop") == home / "x.desktop"


def test_render_matches_spec(home):
    text = auto.render_desktop("/home/hunterp/vss_env/bin/python -m jarvis.app",
                               "/home/hunterp/Jarvis")
    lines = text.splitlines()
    assert lines[0] == "[Desktop Entry]"
    assert "Type=Application" in lines
    assert "Name=Jarvis" in lines
    assert "Exec=/home/hunterp/vss_env/bin/python -m jarvis.app" in lines
    assert "Path=/home/hunterp/Jarvis" in lines
    assert "X-GNOME-Autostart-enabled=true" in lines
    assert "X-GNOME-Autostart-Delay=15" in lines
    assert "Terminal=false" in lines
    assert text.endswith("\n")


def test_default_exec_uses_venv_or_current_python(home, monkeypatch):
    assert auto.default_exec().endswith(" -m jarvis.app")
    venv = home / "vss_env" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text("")
    assert auto.default_python() == str(venv / "python")
    assert auto.REPO_ROOT.name == "Jarvis"


def test_install_is_idempotent(home):
    p1 = auto.install(exec_cmd="/x/python -m jarvis.app")
    assert p1 == home / ".config" / "autostart" / "jarvis.desktop"
    assert p1.exists()
    stamp = p1.stat().st_mtime_ns
    content = p1.read_text()
    os.utime(p1, ns=(stamp - 5_000_000_000, stamp - 5_000_000_000))
    p2 = auto.install(exec_cmd="/x/python -m jarvis.app")
    assert p2 == p1
    assert p1.read_text() == content
    assert p1.stat().st_mtime_ns == stamp - 5_000_000_000     # not rewritten
    assert [q.name for q in p1.parent.iterdir()] == ["jarvis.desktop"]
    # a changed Exec is rewritten
    auto.install(exec_cmd="/y/python -m jarvis.app")
    assert "Exec=/y/python -m jarvis.app" in p1.read_text()


def test_install_uninstall_status(home):
    assert auto.is_installed() is False
    assert auto.uninstall() is False
    auto.install(exec_cmd="/x/python -m jarvis.app")
    assert auto.is_installed() is True
    assert auto.uninstall() is True
    assert auto.is_installed() is False
    assert not auto.desktop_path().exists()


def test_is_installed_rejects_disabled_or_foreign(home):
    p = auto.desktop_path()
    p.parent.mkdir(parents=True)
    p.write_text("[Desktop Entry]\nType=Application\nExec=/x/python -m jarvis.app\n"
                 "X-GNOME-Autostart-enabled=false\n")
    assert auto.is_installed() is False
    p.write_text("[Desktop Entry]\nType=Application\nExec=/x/python -m jarvis.app\nHidden=true\n")
    assert auto.is_installed() is False
    p.write_text("[Desktop Entry]\nType=Application\nName=Jarvis\nExec=/usr/bin/other\n")
    assert auto.is_installed() is False
    p.write_text("[Desktop Entry]\nExec=/x/python -m jarvis.app\n")
    assert auto.is_installed() is True


def test_explicit_path_override(tmp_path, home):
    target = tmp_path / "elsewhere" / "j.desktop"
    assert auto.install(exec_cmd="/x -m jarvis.app", path=target) == target
    assert auto.is_installed(target) is True
    assert auto.is_installed() is False                # HOME location untouched
    assert auto.uninstall(target) is True


@pytest.mark.skipif(not shutil.which("desktop-file-validate"),
                    reason="desktop-file-validate not installed")
def test_rendered_entry_validates(home):
    p = auto.install(exec_cmd="/home/hunterp/vss_env/bin/python -m jarvis.app")
    out = subprocess.run(["desktop-file-validate", str(p)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


# ------------------------------------------------------------ gsettings
def test_disable_suspend_noop_when_already_nothing():
    rec = Recorder([(0, "'nothing'\n")])
    assert auto.disable_gnome_suspend(run=rec) is True
    assert rec.calls == [["gsettings", "get", auto.GSETTINGS_SCHEMA, auto.GSETTINGS_KEY]]


def test_disable_suspend_sets_when_suspend():
    rec = Recorder([(0, "'suspend'\n"), (0, "")])
    assert auto.disable_gnome_suspend(run=rec) is True
    assert rec.calls == [
        ["gsettings", "get", auto.GSETTINGS_SCHEMA, auto.GSETTINGS_KEY],
        ["gsettings", "set", auto.GSETTINGS_SCHEMA, auto.GSETTINGS_KEY, "nothing"],
    ]


def test_disable_suspend_reports_failures():
    rec = Recorder([(0, "'suspend'\n"), (1, "")])
    assert auto.disable_gnome_suspend(run=rec) is False
    rec = Recorder([(1, "")])                              # get fails
    assert auto.disable_gnome_suspend(run=rec) is False
    assert len(rec.calls) == 1
    rec = Recorder([FileNotFoundError("gsettings")])       # no gsettings at all
    assert auto.disable_gnome_suspend(run=rec) is False
    assert auto.suspend_setting(run=Recorder([(0, "'nothing'\n")])) == "'nothing'"
    assert auto.suspend_setting(run=Recorder([subprocess.TimeoutExpired("gsettings", 1)])) is None


def test_real_run_seam_is_not_used_when_recorder_given(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real subprocess called")
    monkeypatch.setattr(auto.subprocess, "run", boom)
    assert auto.disable_gnome_suspend(run=Recorder([(0, "'nothing'\n")])) is True


# ------------------------------------------------------------------ CLI
def test_cli_status_install_uninstall(home, monkeypatch, capsys):
    rec = Recorder([(0, "'nothing'\n"), (0, "'nothing'\n"), (0, "'nothing'\n")])
    monkeypatch.setattr(auto, "_run", rec)
    assert auto.main(["--status"]) == 0
    out = capsys.readouterr().out
    assert "installed: no" in out and "'nothing'" in out
    assert auto.main(["--install", "--exec", "/x/python -m jarvis.app"]) == 0
    out = capsys.readouterr().out
    assert "autostart entry:" in out and "disabled" in out
    assert auto.is_installed() is True
    assert auto.main(["--status"]) == 0
    assert "installed: yes" in capsys.readouterr().out
    assert auto.main(["--uninstall"]) == 0
    assert "removed" in capsys.readouterr().out
    assert auto.is_installed() is False
    assert all(c[0] == "gsettings" for c in rec.calls)
    assert not any(c[1] == "set" for c in rec.calls)      # already 'nothing'


def test_cli_keep_suspend_skips_gsettings(home, monkeypatch, capsys):
    rec = Recorder()
    monkeypatch.setattr(auto, "_run", rec)
    assert auto.main(["--install", "--exec", "/x -m jarvis.app", "--keep-suspend"]) == 0
    assert rec.calls == []
    assert auto.is_installed() is True


def test_cli_requires_one_action(home):
    with pytest.raises(SystemExit):
        auto.main([])
