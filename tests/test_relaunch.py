"""jarvis/relaunch.py -- the Restart button's helper, driven through seams.

Nothing here touches a real process: every pid is a number handed to a fake
process table, every signal lands in a list, every sleep advances a fake
clock, and every launch is a recorded popen. The live Jarvis (pid in
/tmp/vss_voice/jarvis.pid) is never looked at, not even with kill(pid, 0).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


from jarvis import relaunch

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ plan
def test_plan_is_the_bare_app_command_in_the_repo_with_this_environment(
        monkeypatch):
    monkeypatch.setenv("JARVIS_RELAUNCH_PROBE", "carried")
    cmd, cwd, env = relaunch.plan()
    assert cmd == [sys.executable, "-m", "jarvis.app"]
    assert Path(cwd) == REPO_ROOT
    # a COPY of the environment: DISPLAY / XAUTHORITY / DBUS ride along,
    # and a later edit to os.environ does not reach the plan
    assert env["JARVIS_RELAUNCH_PROBE"] == "carried"
    assert env is not os.environ
    monkeypatch.setenv("JARVIS_RELAUNCH_PROBE", "changed")
    assert env["JARVIS_RELAUNCH_PROBE"] == "carried"


# ------------------------------------------------------- spawn_relauncher
class _RecordingPopen:
    """A stand-in for subprocess.Popen: records argv + kwargs, hands back a
    process with a chosen pid."""

    def __init__(self, pid=4242, probe=b""):
        self.pid = pid
        self.probe = probe
        self.calls: list[tuple[list, dict]] = []

    def __call__(self, argv, **kw):
        self.calls.append((list(argv), dict(kw)))
        if self.probe:
            # what the real helper does with the handle it inherits: write
            # to it WHILE the parent still holds it open (the parent closes
            # its copy right after Popen, as a real Popen duplicates it)
            os.write(kw["stdout"], self.probe)

        class _P:
            pid = self.pid
        return _P()


def test_spawn_relauncher_starts_the_detached_helper_and_returns_its_pid(
        tmp_path):
    popen = _RecordingPopen(pid=777, probe=b"probe\n")
    log_path = tmp_path / "logs" / "relaunch.log"
    cmd = ["/venv/bin/python", "-m", "jarvis.app"]
    env = {"DISPLAY": ":1", "PATH": "/bin"}
    pid = relaunch.spawn_relauncher(31337, cmd, "/repo", env, log_path,
                                    grace_s=20, popen=popen)
    assert pid == 777
    assert len(popen.calls) == 1
    argv, kw = popen.calls[0]
    assert argv == [sys.executable, "-m", "jarvis.relaunch",
                    "--wait-pid", "31337", "--grace", "20",
                    "--log", str(log_path), "--cwd", "/repo",
                    "--", "/venv/bin/python", "-m", "jarvis.app"]
    # detached from the app: its own session, no terminal, fds closed
    assert kw["start_new_session"] is True
    assert kw["close_fds"] is True
    assert kw["stdin"] == subprocess.DEVNULL
    assert kw["cwd"] == "/repo"
    assert kw["env"] == env
    # stdout/stderr go to the log file (a traceback in the helper must
    # land somewhere he can read); the directory is created for it
    assert log_path.parent.is_dir()
    assert kw["stderr"] == kw["stdout"]
    assert log_path.read_text() == "probe\n"


def test_spawn_relauncher_grace_is_carried_as_the_helper_saw_it(tmp_path):
    popen = _RecordingPopen()
    relaunch.spawn_relauncher(1, ["py", "-m", "jarvis.app"], "/repo", {},
                              tmp_path / "relaunch.log", grace_s=7.5,
                              popen=popen)
    argv, _ = popen.calls[0]
    assert argv[argv.index("--grace") + 1] == "7.5"


# ------------------------------------------------- the helper's fake world
class FakeWorld:
    """A process table + clock the helper runs against. ``dies_at`` is the
    clock reading after which the old process is gone on its own; a
    signal it honours kills it at the next poll. ``reused_at`` hands the
    same pid to an UNRELATED command from that moment on (the kernel
    recycles pids; the helper must not wait on a stranger)."""

    def __init__(self, pid, dies_at=None, reused_at=None,
                 honours=("TERM", "KILL"), cmdline="python -m jarvis.app"):
        self.pid = pid
        self.dies_at = dies_at
        self.reused_at = reused_at
        self.honours = set(honours)
        self.cmdline = cmdline
        self.t = 0.0
        self.signals: list[tuple[float, int, str]] = []
        self.sleeps: list[float] = []
        self.dead_at = None

    # seams ------------------------------------------------------------
    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s

    def _exists(self, pid):
        if pid != self.pid:
            return False
        if self.dead_at is not None and self.t >= self.dead_at:
            return False
        if self.dies_at is not None and self.t >= self.dies_at:
            return False
        return True

    def kill(self, pid, sig):
        import signal as _s
        name = "NULL" if sig == 0 else _s.Signals(sig).name.replace("SIG", "")
        if not self._exists(pid) and not self._reused(pid):
            raise ProcessLookupError(pid)
        self.signals.append((self.t, pid, name))
        if name in self.honours and self._exists(pid):
            self.dead_at = self.t + 0.25       # dies before the next poll
        if sig == 0:
            return None

    def _reused(self, pid):
        return pid == self.pid and self.reused_at is not None and \
            self.t >= self.reused_at

    def read_cmdline(self, pid):
        if self._reused(pid):
            return "sleep 100000"
        if self._exists(pid):
            return self.cmdline
        raise ProcessLookupError(pid)

    def alive(self, pid):
        return relaunch.process_alive(pid, kill=self.kill,
                                      read_cmdline=self.read_cmdline)


# ---------------------------------------------------------- process_alive
def test_process_alive_is_false_once_the_kernel_says_esrch():
    def kill(pid, sig):
        raise ProcessLookupError(pid)
    assert relaunch.process_alive(5, kill=kill,
                                  read_cmdline=lambda p: "x") is False


def test_process_alive_treats_a_reused_pid_as_gone():
    """The pid-reuse guard: kill(pid, 0) succeeds for whatever now owns the
    number, so the cmdline must still say jarvis.app."""
    alive = relaunch.process_alive(
        5, kill=lambda p, s: None,
        read_cmdline=lambda p: "/usr/bin/sleep\x00100000\x00")
    assert alive is False
    assert relaunch.process_alive(
        5, kill=lambda p, s: None,
        read_cmdline=lambda p: "/venv/bin/python\x00-m\x00jarvis.app\x00") is True


def test_process_alive_stays_true_when_proc_is_unreadable_but_kill_works():
    """/proc/<pid>/cmdline can be empty for a zombie or racing exit; a
    read error is not evidence the process is a stranger."""
    def read_cmdline(p):
        raise OSError("no /proc here")
    assert relaunch.process_alive(5, kill=lambda p, s: None,
                                  read_cmdline=read_cmdline) is True


def test_process_alive_true_for_a_zombie_with_an_empty_cmdline():
    assert relaunch.process_alive(5, kill=lambda p, s: None,
                                  read_cmdline=lambda p: "") is True


# ---------------------------------------------------------- wait_for_exit
def _wait(world, grace_s=20.0, **kw):
    lines = []
    how = relaunch.wait_for_exit(world.pid, grace_s, alive=world.alive,
                                 kill=world.kill, sleep=world.sleep,
                                 clock=world.clock, log=lines.append, **kw)
    return how, lines


def test_a_process_that_quits_on_its_own_is_never_signalled():
    world = FakeWorld(pid=100, dies_at=1.4)
    how, lines = _wait(world)
    assert how == "exited"
    assert [s for s in world.signals if s[2] != "NULL"] == []
    # polled every 0.25 s, not spun
    assert world.sleeps and all(s == 0.25 for s in world.sleeps)
    assert 1.25 <= world.t <= 1.75
    assert any("gone" in ln for ln in lines)


def test_a_process_that_ignores_the_grace_gets_term_then_kill():
    world = FakeWorld(pid=100, honours=("KILL",))     # TERM is ignored
    how, lines = _wait(world, grace_s=20.0)
    named = [(t, s) for t, p, s in world.signals if s != "NULL"]
    assert named[0][1] == "TERM" and 20.0 <= named[0][0] < 20.5
    assert named[1][1] == "KILL" and 25.0 <= named[1][0] < 25.5
    assert how == "killed"
    assert any("SIGTERM" in ln for ln in lines) and \
        any("SIGKILL" in ln for ln in lines)


def test_a_process_that_honours_term_is_never_killed():
    world = FakeWorld(pid=100, honours=("TERM", "KILL"))
    how, _ = _wait(world, grace_s=2.0)
    named = [s for _, _, s in world.signals if s != "NULL"]
    assert named == ["TERM"]
    assert how == "terminated"


def test_a_reused_pid_counts_as_gone_and_the_stranger_is_not_signalled():
    """The old Jarvis died at 0.9 s and the kernel handed pid 100 to a
    `sleep`; the helper must launch, and must not SIGTERM the sleep."""
    world = FakeWorld(pid=100, dies_at=0.9, reused_at=0.9)
    how, _ = _wait(world, grace_s=1.0)
    assert how == "exited"
    assert [s for _, _, s in world.signals if s != "NULL"] == []
    assert world.t < 1.5


def test_a_process_that_survives_sigkill_is_given_up_on_not_waited_forever():
    world = FakeWorld(pid=100, honours=())            # D-state: nothing lands
    how, lines = _wait(world, grace_s=1.0, kill_wait_s=3.0)
    assert how == "stuck"
    assert world.t < 1.0 + 5.0 + 3.0 + 1.0
    assert any("still alive" in ln for ln in lines)


# ------------------------------------------------------------------ main
STAMP = r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} relaunch\[\d+\]: "


def _main(world, popen, log_path, grace="20", cmd=("py", "-m", "jarvis.app")):
    argv = ["--wait-pid", str(world.pid), "--grace", grace,
            "--log", str(log_path), "--cwd", "/repo", "--", *cmd]
    return relaunch.main(argv, alive=world.alive, kill=world.kill,
                         sleep=world.sleep, popen=popen, clock=world.clock)


def test_main_waits_for_the_old_pid_then_launches_once_with_a_timeline(
        tmp_path):
    import re
    world = FakeWorld(pid=100, dies_at=0.6)
    launched_at = []
    inner = _RecordingPopen(pid=4242)

    def popen(argv, **kw):
        launched_at.append(world.clock())
        return inner(argv, **kw)
    log_path = tmp_path / "relaunch.log"
    rc = _main(world, popen, log_path)
    assert rc == 0
    assert len(inner.calls) == 1
    argv, kw = inner.calls[0]
    assert argv == ["py", "-m", "jarvis.app"]
    assert kw["cwd"] == "/repo" and kw["start_new_session"] is True
    # launched only once the old pid was gone
    assert launched_at == [world.t] and world.t >= 0.6
    lines = log_path.read_text().splitlines()
    assert lines and all(re.match(STAMP, ln) for ln in lines), lines
    assert any("launched pid 4242" in ln for ln in lines)
    assert lines[-1].endswith("exit 0")


def test_main_logs_the_failed_launch_exits_1_and_never_retries(tmp_path):
    world = FakeWorld(pid=100, dies_at=0.1)
    attempts = []

    def popen(argv, **kw):
        attempts.append(list(argv))
        raise OSError("no such interpreter")
    log_path = tmp_path / "relaunch.log"
    rc = _main(world, popen, log_path)
    assert rc == 1
    assert len(attempts) == 1
    text = log_path.read_text()
    assert "launch failed" in text and "no such interpreter" in text
    assert text.rstrip().endswith("exit 1")


def test_main_does_not_launch_over_a_process_that_survived_sigkill(tmp_path):
    world = FakeWorld(pid=100, honours=())
    inner = _RecordingPopen()
    rc = _main(world, inner, tmp_path / "relaunch.log", grace="1")
    assert rc == 1
    assert inner.calls == []


def test_main_appends_to_an_existing_log(tmp_path):
    log_path = tmp_path / "relaunch.log"
    log_path.write_text("earlier\n")
    world = FakeWorld(pid=100, dies_at=0.1)
    _main(world, _RecordingPopen(), log_path)
    assert log_path.read_text().startswith("earlier\n")


# ------------------------------------------------- import-time hygiene
def _tree(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        st = p.stat()
        out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


def test_importing_and_helping_never_touches_home_or_a_config_module(
        tmp_path):
    """jarvis/config.py runs Config.load() at import and jarvis.brain loads
    the assistant config at import: a helper whose only job is a clean
    hand-over must not pull either in. Measured, not asserted by reading:
    ``python -X importtime`` in a subprocess with HOME at a tmp dir that
    HOLDS config files, before/after a byte-and-mtime snapshot of HOME."""
    home = tmp_path / "home"
    (home / ".config" / "jarvis").mkdir(parents=True)
    (home / ".config" / "jarvis" / "assistant.json").write_text(
        '{"user": {"name": "probe"}}')
    (home / ".aiws_trainer").mkdir()
    (home / ".aiws_trainer" / "voice_settings.json").write_text("{}")
    before = _tree(home)
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin")}

    r = subprocess.run([sys.executable, "-X", "importtime", "-c",
                        "import jarvis.relaunch"],
                       cwd=REPO_ROOT, env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    imported = [ln.split("|")[-1].strip() for ln in r.stderr.splitlines()
                if ln.startswith("import time:")]
    jarvis_mods = sorted(m for m in imported if m.startswith("jarvis"))
    assert jarvis_mods == ["jarvis", "jarvis.relaunch"], jarvis_mods
    for heavy in ("torch", "numpy", "tkinter", "sounddevice", "whisper"):
        assert not any(m == heavy or m.startswith(heavy + ".")
                       for m in imported), heavy

    r = subprocess.run([sys.executable, "-m", "jarvis.relaunch", "--help"],
                       cwd=REPO_ROOT, env=env, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "--wait-pid" in r.stdout and "--grace" in r.stdout
    assert _tree(home) == before
