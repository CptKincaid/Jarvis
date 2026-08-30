"""The F5 sidecar as a resident systemd --user unit, and the guards around it.

WHY. F5 is the local voice and the hosted voice's fallback, and the sidecar
takes ~180 s to become resident. Live on 2026-08-30: tts_engine=fish, no
f5_server running, a stale socket from the day before -- so the "fallback"
would have paid a full cold start at the first outage. The unit keeps the
model resident across app restarts and reboots; tts.py adopts it and must
never spawn a second copy beside it (f5_server.py unlinks and rebinds the
socket path on start, so two copies fight over it).

The sidecar itself is never run here: every test stubs the ping, the unit
query and Popen.
"""
import os
import re
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest

from jarvis import tts as tts_mod
from jarvis.config import PATHS
from jarvis.events import Status, bus
from jarvis.tts import TTS

REPO = Path(__file__).resolve().parent.parent
UNIT = REPO / "scripts" / "systemd" / "jarvis-f5.service"
SETUP = REPO / "scripts" / "setup_f5_service.sh"


def _unit_fields():
    fields = {}
    for line in UNIT.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "[")):
            continue
        k, _, v = line.partition("=")
        fields.setdefault(k.strip(), []).append(v.strip())
    return fields


def _home(path: str) -> str:
    return path.replace("%h", str(Path.home()))


# ------------------------------------------------------------ the unit file

def test_unit_is_a_restarting_user_unit():
    f = _unit_fields()
    assert f["Restart"] == ["always"]
    assert f["WantedBy"] == ["default.target"]
    assert f["Type"] == ["simple"]
    # a crash loop must stop, not hammer a GPU that is the system memory
    assert "StartLimitBurst" in f


def test_unit_uses_the_same_paths_as_tts_py():
    """One sidecar, one set of paths. If the unit and PATHS drift apart,
    Jarvis pings one socket while the unit serves another, and every launch
    spawns a second sidecar."""
    exec_start = _home(_unit_fields()["ExecStart"][0])
    argv = exec_start.split()
    assert argv[0] == str(PATHS.F5_PYTHON)
    assert argv[1].endswith("scripts/f5_server.py")
    assert argv[1] == str(Path.home() / "Jarvis" / "scripts" / "f5_server.py")
    opts = dict(zip(argv[2::2], argv[3::2]))
    assert opts["--ref"] == str(PATHS.VOICE_REF_F5)
    assert opts["--ref-text"] == str(PATHS.VOICE_REF_F5_TEXT)
    # PATHS.F5_SOCK is redirected by conftest; the unit targets the LIVE dir
    assert opts["--socket"] == "/tmp/vss_voice/" + PATHS.F5_SOCK.name


def test_unit_recreates_the_socket_dir_because_tmp_is_wiped_at_boot():
    pre = _unit_fields()["ExecStartPre"]
    assert any("mkdir -p /tmp/vss_voice" in p for p in pre), pre


def test_unit_flags_are_ones_f5_server_accepts():
    src = (REPO / "scripts" / "f5_server.py").read_text()
    argv = _unit_fields()["ExecStart"][0].split()
    for flag in argv[2::2]:
        assert f'"{flag}"' in src, f"f5_server.py does not take {flag}"


# --------------------------------------------------------- is-active query

class _Run:
    def __init__(self, stdout="inactive\n", rc=3, raise_=None):
        self.stdout, self.rc, self.raise_ = stdout, rc, raise_
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if self.raise_:
            raise self.raise_
        return subprocess.CompletedProcess(argv, self.rc, self.stdout, "")


@pytest.mark.parametrize("stdout,expected", [
    ("active\n", True),
    ("activating\n", True),     # Type=simple: active long before ready
    ("inactive\n", False),
    ("failed\n", False),
    ("deactivating\n", False),
    ("", False),
])
def test_is_active_parses_systemctl(monkeypatch, stdout, expected):
    run = _Run(stdout)
    monkeypatch.setattr(tts_mod.subprocess, "run", run)
    assert tts_mod._f5_unit_active() is expected
    assert run.calls[0][:3] == ["systemctl", "--user", "is-active"]
    assert run.calls[0][3] == tts_mod.F5_UNIT == "jarvis-f5.service"


def test_is_active_is_false_when_systemctl_cannot_be_asked(monkeypatch):
    """No systemctl / no user bus / a hang: better a duplicate than never
    starting a sidecar at all."""
    for exc in (FileNotFoundError("systemctl"),
                subprocess.TimeoutExpired("systemctl", 5)):
        monkeypatch.setattr(tts_mod.subprocess, "run", _Run(raise_=exc))
        assert tts_mod._f5_unit_active() is False


# ----------------------------------------------------- ensure: adopt / wait / spawn

class _Proc:
    """Stand-in for a Popen: alive until told otherwise."""
    pid = 4242

    def __init__(self):
        self.terminated = False
        self.rc = None

    def poll(self):
        return self.rc

    def terminate(self):
        self.terminated = True
        self.rc = -15

    def wait(self, timeout=None):
        return self.rc

    def kill(self):
        self.rc = -9


@pytest.fixture
def no_spawn(monkeypatch):
    """Popen must not be reached; the test fails loudly if it is."""
    def boom(*a, **kw):
        raise AssertionError(f"Popen called: {a[0]}")
    monkeypatch.setattr(tts_mod.subprocess, "Popen", boom)
    monkeypatch.setattr(tts_mod, "_f5_proc", None)


def _alive_sequence(monkeypatch, answers):
    """_f5_alive answers from the list, then repeats the last one."""
    seq = list(answers)

    def alive():
        if len(seq) > 1:
            return seq.pop(0)
        return seq[0]
    monkeypatch.setattr(tts_mod, "_f5_alive", alive)


def test_a_running_sidecar_is_adopted_without_asking_systemd(monkeypatch, no_spawn):
    _alive_sequence(monkeypatch, [True])
    asked = []
    monkeypatch.setattr(tts_mod, "_f5_unit_active",
                        lambda: asked.append(1) or True)
    assert tts_mod._ensure_f5_server() is True
    assert not asked, "a live socket is the answer; systemctl is not consulted"


def test_refuses_to_spawn_while_the_unit_is_active(monkeypatch, no_spawn):
    """The unit is loading the model: wait for its socket, never Popen."""
    _alive_sequence(monkeypatch, [False, False, True])
    monkeypatch.setattr(tts_mod, "_f5_unit_active", lambda: True)
    monkeypatch.setattr(tts_mod.time, "sleep", lambda s: None)
    assert tts_mod._ensure_f5_server(startup_timeout=5) is True
    assert tts_mod._f5_proc is None


def test_waiting_on_the_unit_gives_up_at_the_deadline(monkeypatch, no_spawn):
    _alive_sequence(monkeypatch, [False])
    monkeypatch.setattr(tts_mod, "_f5_unit_active", lambda: True)
    monkeypatch.setattr(tts_mod.time, "sleep", lambda s: None)
    t0 = time.monotonic()
    assert tts_mod._ensure_f5_server(startup_timeout=0.3) is False
    assert time.monotonic() - t0 < 5


def test_waiting_stops_early_when_the_unit_dies(monkeypatch, no_spawn):
    """A unit that crashed must not cost the whole 180 s budget -- and we
    must not then spawn under a unit systemd is about to restart."""
    _alive_sequence(monkeypatch, [False])
    answers = iter([True, False, False, False])
    monkeypatch.setattr(tts_mod, "_f5_unit_active", lambda: next(answers))
    # collapse the 5 s re-check cadence: every tick is a re-check
    clock = {"t": 1000.0}

    def mono():
        clock["t"] += 6
        return clock["t"]
    monkeypatch.setattr(tts_mod.time, "monotonic", mono)
    monkeypatch.setattr(tts_mod.time, "sleep", lambda s: None)
    assert tts_mod._ensure_f5_server(startup_timeout=10_000) is False


def test_spawns_its_own_when_nothing_owns_the_socket(monkeypatch, tmp_path):
    _alive_sequence(monkeypatch, [False, True])
    monkeypatch.setattr(tts_mod, "_f5_unit_active", lambda: False)
    monkeypatch.setattr(tts_mod, "_f5_proc", None)
    for name in ("F5_PYTHON", "F5_SERVER", "F5_REF", "F5_REF_TEXT"):
        f = tmp_path / name
        f.write_text("x")
        monkeypatch.setattr(tts_mod, name, f)
    monkeypatch.setattr(tts_mod, "F5_SOCK", tmp_path / "s" / "f5.sock")
    spawned = []
    proc = _Proc()
    monkeypatch.setattr(tts_mod.subprocess, "Popen",
                        lambda argv, **kw: spawned.append(argv) or proc)
    monkeypatch.setattr(tts_mod.time, "sleep", lambda s: None)
    assert tts_mod._ensure_f5_server(startup_timeout=5) is True
    assert len(spawned) == 1
    argv = spawned[0]
    assert argv[0] == str(tmp_path / "F5_PYTHON")
    assert "--socket" in argv and "--ref" in argv and "--ref-text" in argv
    assert tts_mod._f5_proc is proc


# ------------------------------------------- retiring our own duplicate

def test_our_sidecar_is_stopped_only_when_the_unit_has_taken_over(monkeypatch):
    proc = _Proc()
    monkeypatch.setattr(tts_mod, "_f5_proc", proc)
    monkeypatch.setattr(tts_mod, "_f5_unit_active", lambda: False)
    assert tts_mod.retire_own_f5_sidecar() is False
    assert not proc.terminated, "without the unit ours is the warm next launch"
    assert tts_mod._f5_proc is proc

    monkeypatch.setattr(tts_mod, "_f5_unit_active", lambda: True)
    assert tts_mod.retire_own_f5_sidecar() is True
    assert proc.terminated
    assert tts_mod._f5_proc is None


def test_retire_never_touches_a_process_we_did_not_spawn(monkeypatch):
    monkeypatch.setattr(tts_mod, "_f5_proc", None)
    asked = []
    monkeypatch.setattr(tts_mod, "_f5_unit_active", lambda: asked.append(1) or True)
    assert tts_mod.retire_own_f5_sidecar() is False
    assert not asked


def test_release_is_reachable_from_the_tts_and_swallows_errors(monkeypatch, tmp_path):
    def boom():
        raise RuntimeError("no bus")
    monkeypatch.setattr(tts_mod, "retire_own_f5_sidecar", boom)
    t = TTS(engine="edge", cache_dir=tmp_path / "c")
    assert t.release_f5_sidecar() is False


def test_app_quit_releases_the_sidecar():
    """The hook is wired at shutdown, next to the other cleanup, and is
    tolerant of a TTS stub that lacks it."""
    src = (REPO / "jarvis" / "app.py").read_text()
    quit_body = src[src.index("    def quit(self):"):]
    quit_body = quit_body[:quit_body.index("\n    def ", 10)]
    assert "release_f5_sidecar" in quit_body
    assert "self.tts.stop()" in quit_body


# ------------------------------------------ fish engine warms the fallback

@pytest.fixture
def fish_creds(monkeypatch):
    monkeypatch.setattr(tts_mod, "_fish_creds", lambda: ("key", "model"))


def test_fish_load_warms_f5_on_a_daemon_thread(monkeypatch, tmp_path, fish_creds):
    started = threading.Event()
    calls = []

    def ensure(startup_timeout=180):
        calls.append(threading.current_thread().name)
        started.set()
        return True
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", ensure)
    t = TTS(engine="fish", cache_dir=tmp_path / "c")
    t0 = time.monotonic()
    assert t.load() is True
    assert time.monotonic() - t0 < 1.0, "load() must not wait for the warm-up"
    assert t.engine == "fish"
    assert started.wait(3)
    assert calls and calls[0] != threading.main_thread().name
    assert t._f5_warm_thread.daemon


def test_fish_load_warms_only_once_per_instance(monkeypatch, tmp_path, fish_creds):
    gate = threading.Event()
    calls = []

    def ensure(startup_timeout=180):
        calls.append(1)
        gate.wait(5)
        return True
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", ensure)
    t = TTS(engine="fish", cache_dir=tmp_path / "c")
    for _ in range(4):               # load() runs before EVERY utterance
        assert t.load() is True
    gate.set()
    t._f5_warm_thread.join(5)
    assert calls == [1]


def test_a_cold_fallback_is_announced_not_hidden(monkeypatch, tmp_path, fish_creds):
    """The hosted voice still works, so nothing else would tell the user
    their outage plan is not there."""
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda startup_timeout=180: False)
    seen = []
    bus.subscribe(Status, seen.append)
    try:
        t = TTS(engine="fish", cache_dir=tmp_path / "c")
        assert t.load() is True
        t._f5_warm_thread.join(5)
    finally:
        bus.unsubscribe(Status, seen.append)
    warns = [e for e in seen if e.kind == "warn" and "F5" in e.text]
    assert warns, [e.text for e in seen]
    assert t.engine == "fish", "a cold fallback does not change the live engine"


def test_missing_fish_creds_do_not_start_a_warm_thread(monkeypatch, tmp_path):
    """The engine falls back to f5 directly; that path loads it in-line."""
    monkeypatch.setattr(tts_mod, "_fish_creds", lambda: (None, None))
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda startup_timeout=180: True)
    t = TTS(engine="fish", cache_dir=tmp_path / "c")
    assert t.load() is True
    assert t.engine == "f5"
    assert t._f5_warm_thread is None


# ------------------------------------- f5 unavailable: loud, local first

def _collect_status():
    seen = []
    bus.subscribe(Status, seen.append)
    return seen, lambda: bus.unsubscribe(Status, seen.append)


def test_f5_down_prefers_xtts_and_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda startup_timeout=180: False)
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    monkeypatch.setattr(tts_mod, "VOICE_REF", ref)
    loaded = []
    monkeypatch.setattr(TTS, "_load_xtts_locked",
                        lambda self: loaded.append(self.engine) or True)
    seen, done = _collect_status()
    try:
        t = TTS(engine="f5", cache_dir=tmp_path / "c")
        assert t.load() is True
    finally:
        done()
    assert t.engine == "xtts"
    assert loaded == ["xtts"]
    warns = [e for e in seen if e.kind == "warn"]
    assert warns and "XTTS" in warns[0].text and "local" in warns[0].text.lower()


def test_f5_down_without_an_xtts_clip_falls_to_edge_but_names_the_cloud(
        monkeypatch, tmp_path):
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda startup_timeout=180: False)
    monkeypatch.setattr(tts_mod, "VOICE_REF", tmp_path / "missing.wav")
    seen, done = _collect_status()
    try:
        t = TTS(engine="f5", cache_dir=tmp_path / "c")
        # edge can speak, so the utterance that triggered the load is kept
        assert t.load() is True
    finally:
        done()
    assert t.engine == "edge"
    warns = [e for e in seen if e.kind == "warn"]
    assert warns and "cloud" in warns[0].text.lower()


def test_f5_down_never_reaches_edge_silently(monkeypatch, tmp_path):
    """Whatever the fallback, a Status must be published -- the old code's
    single log.warning was invisible from the UI."""
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda startup_timeout=180: False)
    monkeypatch.setattr(tts_mod, "VOICE_REF", tmp_path / "missing.wav")
    seen, done = _collect_status()
    try:
        TTS(engine="f5", cache_dir=tmp_path / "c").load()
    finally:
        done()
    assert any(e.kind == "warn" for e in seen)


def test_when_the_sidecar_answers_the_engine_stays_f5(monkeypatch, tmp_path):
    monkeypatch.setattr(tts_mod, "_ensure_f5_server", lambda startup_timeout=180: True)
    t = TTS(engine="f5", cache_dir=tmp_path / "c")
    assert t.load() is True
    assert t.engine == "f5"


# ------------------------------------------------------- the install script

def test_install_script_is_executable_and_has_no_sudo():
    assert SETUP.stat().st_mode & stat.S_IXUSR
    src = SETUP.read_text()
    # no line RUNS sudo (the comments are allowed to say the word)
    assert not re.search(r"^\s*sudo\b", src, re.M)
    assert "daemon-reload" in src and "enable" in src


def _run_setup(tmp_path, monkeypatch):
    """Run the installer against a throwaway HOME and a stub systemctl that
    records its arguments. Nothing under the real ~/.config is touched."""
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "systemctl.log"
    stub = bindir / "systemctl"
    stub.write_text("#!/usr/bin/env bash\necho \"$@\" >> \"$SYSTEMCTL_LOG\"\n")
    stub.chmod(0o755)
    env = dict(os.environ, HOME=str(home), SYSTEMCTL_LOG=str(log),
               PATH=f"{bindir}:{os.environ.get('PATH', '')}")
    env.pop("XDG_CONFIG_HOME", None)
    res = subprocess.run(["bash", str(SETUP)], env=env, capture_output=True,
                         text=True, timeout=30)
    return res, home, (log.read_text() if log.exists() else "")


def test_install_script_copies_reloads_enables_and_never_starts(tmp_path, monkeypatch):
    res, home, calls = _run_setup(tmp_path, monkeypatch)
    assert res.returncode == 0, res.stderr
    unit = home / ".config" / "systemd" / "user" / "jarvis-f5.service"
    assert unit.exists()
    lines = calls.splitlines()
    assert "--user daemon-reload" in lines
    assert "--user enable jarvis-f5.service" in lines
    # The first start loads a ~5 GB model onto a GPU that IS the system
    # memory; that moment belongs to a person, not an installer.
    assert not any("start" in ln or "--now" in ln for ln in lines), lines
    assert "systemctl --user start jarvis-f5.service" in res.stdout


def test_install_script_points_the_unit_at_this_checkout(tmp_path, monkeypatch):
    """The unit says %h/Jarvis; a checkout elsewhere (a worktree, a clone)
    gets its real path, or the unit would run somebody else's f5_server."""
    res, home, _ = _run_setup(tmp_path, monkeypatch)
    assert res.returncode == 0, res.stderr
    installed = (home / ".config" / "systemd" / "user" / "jarvis-f5.service").read_text()
    exec_line = next(ln for ln in installed.splitlines() if ln.startswith("ExecStart="))
    # HOME is the throwaway above, so $HOME/Jarvis is never this checkout
    # and the substitution must always have happened
    assert f"{REPO}/scripts/f5_server.py" in exec_line, exec_line
    assert "%h/Jarvis/" not in exec_line
    # the ref-clip paths stay %h-relative either way: they follow the user
    assert "%h/.aiws_trainer/jarvis_voice_ref_f5.wav" in exec_line
