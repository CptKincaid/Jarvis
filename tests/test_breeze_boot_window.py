"""The Breeze sidecar's START window: boot ordering, and the one hole in it.

The render cost is fixed (tests/test_breeze_sidecar.py, RenderWorker). What
was left was the START: the model takes 13.5 GB of free pages and CUDA-graph
capture then demands ~18.2 GB more, a 31.2 GB peak measured against a floor
of MIN_FREE_GB=33 -- and once Jarvis is up, MemFree on this box is 19-30 GB,
so the gate correctly refuses and the voice never comes on.

The fix is ORDERING, and it rests on one fact about this box: nothing loads
gemma4:26b except jarvis.brain's residency loop. ollama.service holds no
model until something asks, so "Breeze before ollama" was never the
constraint -- "Breeze before JARVIS" is, and a login starts with more free
than the 49.9 GB a real start was measured surviving from (in at 49.87,
trough 18.66, a 31.2 GB peak). Two files carry it:

  * jarvis-breeze.service enabled at default.target, After=jarvis-f5.service
  * scripts/jarvis-autostart in place of `python -m jarvis.app` in the
    autostart entry, waiting for the sidecar's ping before Jarvis starts

and one file covers the case ordering cannot reach -- a GNOME log out
without a reboot, which leaves ollama holding a keep_alive -1 gemma4 that no
longer has an owner:

  * scripts/breeze_make_room.py, an ExecStartPre that unloads it, and ONLY
    when Jarvis is not running. A live Jarvis re-pins within 30 s, and a
    19.0 GB reload landing inside a 31.2 GB capture is 49.9 - 31.2 - 19.0 =
    -0.3 GB. That is the 2026-08-28 power-off, so the eviction is refused
    rather than risked, and the sidecar's own gate says so out loud.

No GPU, no ollama, no display, no Jarvis: every probe in breeze_make_room is
a module-level seam, and the wrapper is driven against a fake systemctl and
a real AF_UNIX server in tmp_path.
"""
import importlib.util
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
UNIT = REPO / "scripts" / "systemd" / "jarvis-breeze.service"
WRAPPER = REPO / "scripts" / "jarvis-autostart"
SETUP = REPO / "scripts" / "setup_breeze_service.sh"


def _load_make_room():
    spec = importlib.util.spec_from_file_location(
        "breeze_make_room", REPO / "scripts" / "breeze_make_room.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mr = _load_make_room()


@pytest.fixture
def room(monkeypatch):
    """breeze_make_room with every probe replaced and a call ledger."""
    state = {"free": 100.0, "loaded": [], "unloaded": [], "pid": None,
             "sleeps": []}

    monkeypatch.setattr(mr, "mem_gb", lambda field="MemFree": state["free"])
    monkeypatch.setattr(mr, "live_jarvis_pid", lambda pidfile: state["pid"])
    monkeypatch.setattr(mr, "ollama_loaded", lambda url: list(state["loaded"]))

    def _unload(url, model):
        state["unloaded"].append(model)
        state["free"] = state.get("free_after", state["free"])
        return True
    monkeypatch.setattr(mr, "ollama_unload", _unload)
    monkeypatch.setattr(mr.time, "sleep", lambda s: state["sleeps"].append(s))
    return state


def run_room(state, min_free=33.0, wait_s=30.0):
    return mr.make_room(min_free, wait_s, "/nonexistent/jarvis.pid",
                        "http://127.0.0.1:11434")


# ===========================================================================
# breeze_make_room: it does nothing unless doing nothing is the wrong answer
# ===========================================================================
def test_enough_free_is_a_no_op_and_never_touches_ollama(room, capsys):
    """The fresh-boot path. A login has more free than the 49.9 GB a real start
    was measured surviving from, so the whole helper is one printed line and
    no side effects -- it must not so much as ask ollama
    what it is holding."""
    room["free"] = 108.0
    room["loaded"] = ["gemma4:26b"]
    assert run_room(room) == 0
    assert room["unloaded"] == []
    assert "nothing to do" in capsys.readouterr().out


def test_a_live_jarvis_stops_the_eviction(room, capsys):
    """THE ARITHMETIC THAT DECIDES THIS FILE. brain.RESIDENCY_INTERVAL_S is
    30 s, so a live Jarvis re-pins gemma4 inside the capture window, and
    49.9 - 31.2 - 19.0 = -0.3 GB. Refuse the eviction, not the risk."""
    room["free"] = 29.0
    room["loaded"] = ["gemma4:26b"]
    room["pid"] = 1868758
    assert run_room(room) == 0
    assert room["unloaded"] == []
    out = capsys.readouterr().out
    assert "Jarvis is running (pid 1868758)" in out
    assert "F5" in out


def test_orphaned_model_is_unloaded_when_jarvis_is_gone(room, capsys):
    """The logout-without-reboot case this helper exists for: ollama is a
    system service and survives the session, so the keep_alive -1 pin outlives
    the process that set it."""
    room["free"] = 29.0
    room["free_after"] = 48.0
    room["loaded"] = ["gemma4:26b"]
    room["pid"] = None
    assert run_room(room) == 0
    assert room["unloaded"] == ["gemma4:26b"]
    out = capsys.readouterr().out
    assert "no Jarvis" in out
    assert "room made" in out


def test_every_loaded_model_goes_not_just_the_first(room):
    room["free"] = 20.0
    room["free_after"] = 50.0
    room["loaded"] = ["gemma4:26b", "nomic-embed-text:latest"]
    assert run_room(room) == 0
    assert room["unloaded"] == ["gemma4:26b", "nomic-embed-text:latest"]


def test_ollama_holding_nothing_is_reported_not_retried(room, capsys):
    room["free"] = 12.0
    room["loaded"] = []
    assert run_room(room) == 0
    assert room["unloaded"] == []
    assert "ollama holds nothing" in capsys.readouterr().out


def test_a_wait_that_never_pays_off_gives_up_and_leaves_the_gate_to_speak(
        room, capsys, monkeypatch):
    """The unload was accepted but the pages did not come back. We do not hold
    up the login for it -- the sidecar's refusal carries the real numbers."""
    clock = {"t": 0.0}
    monkeypatch.setattr(mr.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(mr.time, "sleep",
                        lambda s: clock.__setitem__("t", clock["t"] + s))
    room["free"] = 29.0
    room["free_after"] = 29.0
    room["loaded"] = ["gemma4:26b"]
    assert run_room(room, wait_s=5.0) == 0
    assert "still short" in capsys.readouterr().out


def test_it_always_exits_zero_even_when_a_probe_explodes(monkeypatch, capsys):
    """A failing ExecStartPre stops the unit BEFORE breeze_server.py can print
    its refusal, and that refusal -- the two numbers and the lever -- is the
    most useful thing a failed start produces."""
    def boom(*a, **k):
        raise RuntimeError("meminfo is on fire")
    monkeypatch.setattr(mr, "mem_gb", boom)
    assert mr.main(["--min-free-gb", "33"]) == 0
    assert "ignored" in capsys.readouterr().out


def test_the_floor_matches_the_servers_own(monkeypatch):
    """A drift between these two is a helper that frees the wrong amount and
    a gate that refuses anyway."""
    spec = importlib.util.spec_from_file_location(
        "breeze_server", REPO / "scripts" / "breeze_server.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    assert mr.DEFAULT_MIN_FREE_GB == float(bs.MIN_FREE_GB)
    assert any(f"--min-free-gb {int(bs.MIN_FREE_GB)}" in ln
               for ln in directives(UNIT))


# ---------------------------------------------------------------- liveness
def test_missing_pidfile_reads_as_no_jarvis(tmp_path):
    assert mr.live_jarvis_pid(str(tmp_path / "nope.pid")) is None


def test_a_stale_pid_nobody_is_using_reads_as_no_jarvis(tmp_path, monkeypatch):
    p = tmp_path / "jarvis.pid"
    p.write_text("424242")
    monkeypatch.setattr(mr, "pid_alive", lambda pid: False)
    assert mr.live_jarvis_pid(str(p)) is None


def test_a_recycled_pid_belonging_to_someone_else_reads_as_no_jarvis(
        tmp_path, monkeypatch):
    """The pid file is the anchor -- we never search the process table -- but
    a wiped /tmp and a reused pid can still point it at a stranger."""
    p = tmp_path / "jarvis.pid"
    p.write_text("999")
    monkeypatch.setattr(mr, "pid_alive", lambda pid: True)
    monkeypatch.setattr(mr, "read_cmdline", lambda pid: "/usr/bin/gnome-shell")
    assert mr.live_jarvis_pid(str(p)) is None


def test_a_real_jarvis_is_recognised(tmp_path, monkeypatch):
    p = tmp_path / "jarvis.pid"
    p.write_text("1868758")
    monkeypatch.setattr(mr, "pid_alive", lambda pid: True)
    monkeypatch.setattr(
        mr, "read_cmdline",
        lambda pid: "/home/hunterp/vss_env/bin/python -m jarvis.app")
    assert mr.live_jarvis_pid(str(p)) == 1868758


def test_an_unreadable_cmdline_counts_as_jarvis(tmp_path, monkeypatch):
    """Unreadable means we cannot rule him out, and the safe direction is the
    one where nothing is evicted."""
    p = tmp_path / "jarvis.pid"
    p.write_text("1868758")
    monkeypatch.setattr(mr, "pid_alive", lambda pid: True)

    def denied(pid):
        raise PermissionError
    monkeypatch.setattr(mr, "read_cmdline", denied)
    assert mr.live_jarvis_pid(str(p)) == 1868758


def test_unreadable_meminfo_reads_as_no_room_not_as_plenty(monkeypatch):
    def denied(path=mr.MEMINFO_PATH):
        raise OSError("nope")
    monkeypatch.setattr(mr, "read_meminfo", denied)
    assert mr.mem_gb("MemFree") == 0.0


def test_mem_gb_parses_kb_into_gib(monkeypatch):
    """/proc reports kB and every number in this story is GiB. The sample is a
    real /proc/meminfo from this box on 2026-09-02 with Jarvis, F5 and a
    resident gemma4:26b up -- MemFree 29.97 GiB, which is exactly why the
    sidecar cannot start from here."""
    monkeypatch.setattr(
        mr, "read_meminfo",
        lambda path=mr.MEMINFO_PATH:
            "MemTotal:  127606644 kB\nMemFree:   31428912 kB\n"
            "MemAvailable:  81515264 kB\n")
    assert round(mr.mem_gb("MemFree"), 2) == 29.97
    assert round(mr.mem_gb("MemAvailable"), 1) == 77.7
    assert mr.mem_gb("NoSuchField") == 0.0


# ===========================================================================
# the unit: ordering, and the ExecStartPre that covers the logout case
# ===========================================================================
def directives(path: Path) -> list[str]:
    """The lines systemd (or gnome-session) actually reads. Both files are
    mostly commentary, and every assertion below is about behaviour."""
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def test_unit_is_ordered_after_f5_not_after_ollama():
    """ollama.service is a SYSTEM unit and holds no model until asked; the
    thing that asks is Jarvis. Ordering against ollama would buy nothing and
    would couple a user unit to a system one."""
    lines = directives(UNIT)
    assert "After=jarvis-f5.service" in lines
    assert not [ln for ln in lines if "ollama" in ln]


def test_unit_does_not_pull_f5_in():
    """After= orders; Requires=/Wants= would make the Breeze voice depend on
    the fallback voice being installed."""
    lines = directives(UNIT)
    assert not [ln for ln in lines if ln.startswith(("Requires=", "Wants=",
                                                     "BindsTo="))]


def test_unit_runs_the_room_maker_before_the_server():
    lines = directives(UNIT)
    pre = [ln for ln in lines if ln.startswith("ExecStartPre=")]
    assert any("breeze_make_room.py" in ln for ln in pre)
    assert lines.index(next(ln for ln in pre if "breeze_make_room.py" in ln)) \
        < lines.index(next(ln for ln in lines if ln.startswith("ExecStart=")))


def test_room_maker_runs_on_the_system_python_not_the_breeze_venv():
    """ExecStartPre fires before the venv is relevant, and the helper is
    stdlib-only precisely so it can."""
    line = next(ln for ln in directives(UNIT) if "breeze_make_room.py" in ln)
    assert line.startswith("ExecStartPre=/usr/bin/python3 ")
    assert "venv" not in line


def test_room_maker_imports_nothing_outside_the_stdlib():
    src = (REPO / "scripts" / "breeze_make_room.py").read_text()
    for line in src.splitlines():
        if line.startswith(("import ", "from ")) and " import " not in line[:5]:
            mod = line.split()[1].split(".")[0]
            assert mod in {"__future__", "argparse", "json", "os", "sys",
                           "time", "urllib"}, mod


def test_unit_still_refuses_terminally_and_still_holds_the_gpu_lock():
    """The ordering work must not have loosened either guard."""
    lines = directives(UNIT)
    assert "RestartPreventExitStatus=2" in lines
    assert any("--gpu-lock" in ln for ln in lines)


# ===========================================================================
# the autostart entry: ONE renderer, and it knows about the wrapper
# ===========================================================================
def test_the_repo_ships_no_second_desktop_template():
    """app.py re-runs autostart.install() at every start once
    "autostart.enabled" is on. A template here would either lose to it or,
    once the two drifted, make it rewrite the entry on every login."""
    assert not (REPO / "scripts" / "systemd" / "jarvis.desktop").exists()


def test_jarvis_autostart_renders_the_wrapper_for_this_checkout():
    """The wrapper is in the tree and executable, so this is the Exec that
    every future install() produces -- the installer's and app.py's alike."""
    from jarvis import autostart
    assert autostart.default_exec() == str(REPO / "scripts" / "jarvis-autostart")
    assert autostart.is_installed  # the gate below reads the same file


# ===========================================================================
# the wrapper: it delays, it never blocks
# ===========================================================================
def fake_systemctl(bin_dir: Path, enabled: str, active: str) -> None:
    script = bin_dir / "systemctl"
    script.write_text(
        "#!/bin/sh\n"
        "for a in \"$@\"; do\n"
        f"  [ \"$a\" = is-enabled ] && {{ echo {enabled}; exit 0; }}\n"
        f"  [ \"$a\" = is-active ] && {{ echo {active}; exit 0; }}\n"
        "done\nexit 1\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def run_wrapper(tmp_path, *, enabled="enabled", active="active",
                sock=None, wait_s="3", timeout=40):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_systemctl(bin_dir, enabled, active)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["JARVIS_BREEZE_SOCK"] = str(sock or (tmp_path / "absent.sock"))
    env["JARVIS_BREEZE_WAIT_S"] = wait_s
    env["JARVIS_PYTHON"] = "/bin/echo"
    env["JARVIS_HOME"] = str(tmp_path)
    env["JARVIS_AUTOSTART_LOG"] = str(tmp_path / "autostart.log")
    t0 = time.monotonic()
    proc = subprocess.run([str(WRAPPER)], env=env, capture_output=True,
                          text=True, timeout=timeout)
    return proc, time.monotonic() - t0


def test_wrapper_execs_jarvis_immediately_when_the_unit_is_not_enabled(tmp_path):
    """Today's box. Nothing about Jarvis's start may change until the sidecar
    is deliberately turned on."""
    proc, elapsed = run_wrapper(tmp_path, enabled="disabled")
    assert proc.returncode == 0
    assert "-m jarvis.app" in proc.stdout
    assert elapsed < 3.0


def test_wrapper_stops_waiting_the_moment_the_unit_is_not_active(tmp_path):
    """A MemFree refusal exits 2 in about a second; Jarvis must not then sit
    out the whole 90 s budget."""
    proc, elapsed = run_wrapper(tmp_path, enabled="enabled", active="failed",
                                wait_s="30")
    assert "-m jarvis.app" in proc.stdout
    assert elapsed < 10.0


def test_wrapper_starts_jarvis_anyway_when_the_sidecar_never_answers(tmp_path):
    """The wait is a delay, never a dependency: Jarvis comes up on F5."""
    proc, elapsed = run_wrapper(tmp_path, enabled="enabled", active="active",
                                wait_s="3")
    assert "-m jarvis.app" in proc.stdout
    assert 2.0 < elapsed < 20.0


def test_wrapper_returns_at_once_when_the_sidecar_is_already_ready(tmp_path):
    """A hand restart of Jarvis, which he does often, must cost nothing."""
    sock = tmp_path / "breeze.sock"
    stop = threading.Event()

    def serve():
        srv = socket.socket(socket.AF_UNIX)
        srv.bind(str(sock))
        srv.listen(4)
        srv.settimeout(0.5)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            with conn:
                conn.recv(65536)
                conn.sendall(json.dumps(
                    {"ok": True, "ready": True, "graphs": True}).encode() + b"\n")
        srv.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
        proc, elapsed = run_wrapper(tmp_path, enabled="enabled",
                                    active="active", sock=sock, wait_s="30")
    finally:
        stop.set()
        t.join(timeout=5)
    assert "-m jarvis.app" in proc.stdout
    assert elapsed < 10.0


def test_wrapper_keeps_waiting_when_the_graphs_are_not_captured(tmp_path):
    """ready without graphs is the working-but-stuttering sidecar (RTF 1.131);
    it is not a reason to let a 19 GB model land on top of it either."""
    sock = tmp_path / "breeze.sock"
    stop = threading.Event()

    def serve():
        srv = socket.socket(socket.AF_UNIX)
        srv.bind(str(sock))
        srv.listen(4)
        srv.settimeout(0.5)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            with conn:
                conn.recv(65536)
                conn.sendall(json.dumps(
                    {"ok": True, "ready": True, "graphs": False}).encode() + b"\n")
        srv.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
        proc, elapsed = run_wrapper(tmp_path, enabled="enabled",
                                    active="active", sock=sock, wait_s="3")
    finally:
        stop.set()
        t.join(timeout=5)
    assert "-m jarvis.app" in proc.stdout
    assert elapsed > 2.0


# ===========================================================================
# the installer
# ===========================================================================
def test_setup_still_does_not_enable_or_start_by_default():
    """The 13.5 GB moment stays a choice."""
    text = SETUP.read_text()
    body = text.split('if [ "$MODE" = enable ]')[0]
    assert "systemctl --user enable" not in body
    assert "systemctl --user start" not in body


def test_setup_has_one_switch_to_turn_boot_ordering_on_and_one_to_undo_it():
    text = SETUP.read_text()
    assert "--enable-at-boot" in text and "--revert-boot" in text
    assert 'systemctl --user enable "$UNIT"' in text
    assert 'systemctl --user disable "$UNIT"' in text


def test_enabling_backs_up_the_existing_autostart_entry_exactly_once():
    """Re-running the installer must not overwrite the pre-breeze backup with
    our own file -- that is the only copy of what to revert to."""
    text = SETUP.read_text()
    assert "DESKTOP_BAK" in text
    assert '[ ! -f "$DESKTOP_BAK" ]' in text


def test_setup_refuses_to_install_a_unit_whose_execstartpre_is_missing():
    text = SETUP.read_text()
    assert "breeze_make_room.py" in text
    assert "jarvis-autostart" in text
    # a warning is not enough: the unit NAMES the helper, so a missing one is
    # a failed start
    line = next(ln for ln in text.splitlines()
                if "breeze_make_room.py" in ln and "[ -f" in ln)
    assert "exit 1" in line


def test_wrapper_and_room_maker_are_executable():
    assert os.access(WRAPPER, os.X_OK)


def run_setup(tmp_path, *args):
    """The real installer, against a sandbox HOME and a fake systemctl.

    Grepping the script proves what it says; this proves what it does -- and
    the thing that matters most about it (the ONE backup of the autostart
    entry that predates all of this) is only visible in a round trip.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "systemctl.log"
    stub = bin_dir / "systemctl"
    stub.write_text('#!/bin/sh\necho "systemctl $*" >> "$SYSTEMCTL_LOG"\nexit 0\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    # HOME as well: --enable-at-boot drives jarvis.autostart, whose
    # desktop_path() is HOME-relative. The real ~/.config/autostart is never
    # written by this suite.
    env["HOME"] = str(tmp_path)
    env["JARVIS_PYTHON"] = sys.executable
    env["SYSTEMCTL_LOG"] = str(log)
    proc = subprocess.run([str(SETUP), *args], env=env, capture_output=True,
                          text=True, timeout=60)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls


ORIGINAL_DESKTOP = (
    "[Desktop Entry]\nType=Application\nName=Jarvis\n"
    "Exec=/home/hunterp/vss_env/bin/python -m jarvis.app\n"
    "Path=/home/hunterp/Jarvis\nX-GNOME-Autostart-Delay=15\n")


@pytest.fixture
def sandbox(tmp_path):
    auto = tmp_path / "config" / "autostart"
    auto.mkdir(parents=True)
    (auto / "jarvis.desktop").write_text(ORIGINAL_DESKTOP)
    return tmp_path


def test_plain_install_writes_the_unit_and_enables_nothing(sandbox):
    proc, calls = run_setup(sandbox)
    assert proc.returncode == 0, proc.stderr
    unit = sandbox / "config" / "systemd" / "user" / "jarvis-breeze.service"
    assert unit.exists()
    assert calls == ["systemctl --user daemon-reload"]
    # and the autostart entry is untouched
    assert (sandbox / "config" / "autostart" / "jarvis.desktop").read_text() \
        == ORIGINAL_DESKTOP


def test_install_substitutes_the_real_repo_path_into_the_execstartpre(sandbox):
    """systemd has no "where the repo is" specifier and the helper is named in
    an ExecStartPre, so an unsubstituted %h there is a failed start."""
    run_setup(sandbox)
    unit = (sandbox / "config" / "systemd" / "user"
            / "jarvis-breeze.service").read_text()
    line = next(ln for ln in unit.splitlines()
                if "breeze_make_room.py" in ln and ln.startswith("ExecStartPre"))
    assert str(REPO) in line
    assert "%h/Jarvis" not in line
    assert Path(line.split()[1]).exists()


def test_enable_at_boot_backs_up_then_redirects_the_autostart_entry(sandbox):
    proc, calls = run_setup(sandbox, "--enable-at-boot")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    auto = sandbox / "config" / "autostart"
    assert (auto / "jarvis.desktop.pre-breeze").read_text() == ORIGINAL_DESKTOP
    installed = (auto / "jarvis.desktop").read_text()
    assert f"Exec={REPO}/scripts/jarvis-autostart" in installed
    assert "systemctl --user enable jarvis-breeze.service" in calls
    assert not [c for c in calls if " start " in c]


def test_the_entry_the_installer_writes_is_the_one_app_py_would_rewrite(sandbox):
    """The two must agree byte for byte, or install()'s idempotence check
    fails and Jarvis rewrites the entry at every single login."""
    run_setup(sandbox, "--enable-at-boot")
    written = (sandbox / "config" / "autostart" / "jarvis.desktop").read_text()
    from jarvis import autostart
    assert written == autostart.render_desktop()


def test_enabling_twice_keeps_the_backup_that_predates_all_of_this(sandbox):
    """Re-running the installer must not clobber the only copy of what to
    revert to with a copy of our own file."""
    run_setup(sandbox, "--enable-at-boot")
    run_setup(sandbox, "--enable-at-boot")
    assert (sandbox / "config" / "autostart"
            / "jarvis.desktop.pre-breeze").read_text() == ORIGINAL_DESKTOP


def test_revert_disables_the_unit_and_leaves_the_pass_through_alone(sandbox):
    """Reverting the ORDERING is one thing: the unit stops coming up at login.
    The entry keeps naming the wrapper because the wrapper does nothing once
    the unit is disabled -- and because putting the bare command back would
    not stick: jarvis.autostart renders the wrapper whenever it is there."""
    run_setup(sandbox, "--enable-at-boot")
    proc, calls = run_setup(sandbox, "--revert-boot")
    assert proc.returncode == 0, proc.stderr
    auto = sandbox / "config" / "autostart"
    assert "jarvis-autostart" in (auto / "jarvis.desktop").read_text()
    assert (auto / "jarvis.desktop.pre-breeze").read_text() == ORIGINAL_DESKTOP
    assert "systemctl --user disable jarvis-breeze.service" in calls
    # reverting the ordering must not stop the sidecar out from under him,
    # and must not rewrite the entry either
    assert not [c for c in calls if " stop " in c]


def test_revert_before_enable_is_harmless(sandbox):
    proc, calls = run_setup(sandbox, "--revert-boot")
    assert proc.returncode == 0
    assert (sandbox / "config" / "autostart"
            / "jarvis.desktop").read_text() == ORIGINAL_DESKTOP
    assert "systemctl --user disable jarvis-breeze.service" in calls


def test_an_unknown_flag_is_refused_rather_than_treated_as_install(sandbox):
    proc, calls = run_setup(sandbox, "--nonsense")
    assert proc.returncode != 0
    assert calls == []
