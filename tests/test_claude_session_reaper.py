"""Idle Claude sessions are reaped (jarvis/claude_session.py).

MEASURED 2026-09-12: tmux session jarvis-test held a `claude --resume …`
for 9 days 11 hours across three Jarvis restarts. Nothing ended a session:
close() only unsubscribed two bus events and cancel() sends C-c by design.
A session Jarvis started may be ended when it is IDLE — no task running,
nobody attached, the pane not mid-turn, and no activity for longer than
`claude.session_max_idle_s`. His own cc-* terminals are never touched.
Every subprocess goes through the `run` seam; nothing here touches tmux.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import jarvis.claude_session as cs
from jarvis.assistant_config import AssistantConfig

NOW = 1_800_000_000.0
H = 3600.0


class Tmux:
    """A scripted tmux: sessions with an activity time, an attached flag
    and a pane text; records every kill."""

    def __init__(self, sessions):
        # name -> (activity, attached, pane_text, pane_cmd)
        self.sessions = dict(sessions)
        self.killed = []
        self.fail = False

    def __call__(self, argv, timeout=15.0, input=None, cwd=None, env=None):
        ok = lambda out="": SimpleNamespace(returncode=0, stdout=out, stderr="")  # noqa: E731
        bad = SimpleNamespace(returncode=1, stdout="", stderr="no server running")
        if self.fail or argv[:1] != ["tmux"]:
            return bad
        sub = argv[1]
        if sub == "list-sessions":
            return ok("".join(f"{n}\t{int(a)}\t{1 if att else 0}\n"
                              for n, (a, att, _t, _c) in self.sessions.items()))
        name = argv[argv.index("-t") + 1] if "-t" in argv else ""
        if name not in self.sessions:
            return bad
        a, att, text, cmd = self.sessions[name]
        if sub == "capture-pane":
            return ok(text)
        if sub == "list-clients":
            return ok("client0\n" if att else "")
        if sub == "list-panes":
            return ok(cmd + "\n")
        if sub == "kill-session":
            self.killed.append(name)
            del self.sessions[name]
            return ok()
        return ok()


READY = "─" * 20 + "\n❯ \n" + "─" * 20 + "\n"
WORKING = READY + "  ✻ Thinking… (3s · esc to interrupt)\n"


def manager(tmp_path, tmux, max_idle_s=4 * H, **kw):
    cfg = AssistantConfig({"claude": {"session_max_idle_s": max_idle_s}},
                          path=tmp_path / "assistant.json")
    approvals = SimpleNamespace(sock_path=str(tmp_path / "a.sock"), pending=lambda: [])
    return cs.ClaudeSessionManager(cfg, SimpleNamespace(), approvals,
                                   tmp_path / "state.json", tmp_path / "tasks",
                                   run=tmux, claude_bin="/opt/bin/claude",
                                   projects_dir=tmp_path / "sessions",
                                   home=tmp_path, now=lambda: NOW, **kw)


def test_an_idle_unattached_session_past_the_cap_is_reaped(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="jarvis.claude_session")
    tmux = Tmux({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "claude")})
    m = manager(tmp_path, tmux)
    m._panes["test"] = cs.Pane(slug="test", started=NOW - 9 * 24 * H)
    assert m.reap_idle() == ["jarvis-test"]
    assert tmux.killed == ["jarvis-test"]
    assert "test" not in m._panes
    lines = [r.getMessage() for r in caplog.records if "reaped" in r.getMessage()]
    assert len(lines) == 1 and "jarvis-test" in lines[0] and "9 d" in lines[0]


def test_a_session_younger_than_the_cap_is_kept(tmp_path):
    tmux = Tmux({"jarvis-test": (NOW - 3 * H, False, READY, "claude")})
    assert manager(tmp_path, tmux).reap_idle() == []
    assert tmux.killed == []


def test_an_attached_session_is_kept_however_old(tmp_path):
    tmux = Tmux({"jarvis-test": (NOW - 9 * 24 * H, True, READY, "claude")})
    assert manager(tmp_path, tmux).reap_idle() == []
    assert tmux.killed == []


def test_a_pane_mid_turn_is_kept(tmp_path):
    tmux = Tmux({"jarvis-test": (NOW - 9 * 24 * H, False, WORKING, "claude")})
    assert manager(tmp_path, tmux).reap_idle() == []
    assert tmux.killed == []


def test_a_session_with_a_running_task_is_kept(tmp_path):
    tmux = Tmux({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "claude")})
    m = manager(tmp_path, tmux)
    t = cs.Task(task_id="t1", project="test", cwd=str(tmp_path), prompt="x", model="")
    m._running["t1"] = t
    assert m.reap_idle() == []
    assert tmux.killed == []


def test_his_own_terminals_and_strangers_are_never_touched(tmp_path):
    tmux = Tmux({"cc-hunterp-jarvis": (NOW - 30 * 24 * H, False, READY, "claude"),
                 "main": (NOW - 30 * 24 * H, False, "", "bash"),
                 "jarvis-old": (NOW - 30 * 24 * H, False, READY, "claude")})
    assert manager(tmp_path, tmux).reap_idle() == ["jarvis-old"]
    assert tmux.killed == ["jarvis-old"]


def test_a_cap_of_zero_switches_the_reaper_off(tmp_path):
    tmux = Tmux({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "claude")})
    assert manager(tmp_path, tmux, max_idle_s=0).reap_idle() == []
    assert tmux.killed == []


def test_tmux_being_down_is_nothing_to_reap(tmp_path):
    tmux = Tmux({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "claude")})
    tmux.fail = True
    assert manager(tmp_path, tmux).reap_idle() == []


def test_close_reaps_once_and_stops_the_thread(tmp_path):
    tmux = Tmux({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "claude")})
    m = manager(tmp_path, tmux)
    m.start_reaper(every_s=3600.0)
    assert m.reaper_running
    m.close()
    assert not m.reaper_running
    assert tmux.killed == ["jarvis-test"]


def test_the_default_cap_is_four_hours_and_configurable():
    assert cs.DEFAULT_SESSION_MAX_IDLE_S == 4 * H


def test_a_pane_running_something_other_than_claude_or_a_shell_is_kept(tmp_path):
    """The name pattern is Jarvis's, but a pane holding vim or a build is
    somebody's work in progress -- only a shell or the Claude TUI itself is
    a session the reaper may end."""
    tmux = Tmux({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "vim")})
    assert manager(tmp_path, tmux).reap_idle() == []
    assert tmux.killed == []
    for cmd in ("claude", "node", "bash"):
        tmux = Tmux({"jarvis-test": (NOW - 9 * 24 * H, False, READY, cmd)})
        assert manager(tmp_path, tmux).reap_idle() == ["jarvis-test"]


def test_the_reaper_sweeps_once_at_start_not_only_after_the_first_interval(tmp_path):
    """A 9-day orphan must not wait ten more minutes after a boot."""
    import time
    tmux = Tmux({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "claude")})
    m = manager(tmp_path, tmux)
    m.start_reaper(every_s=3600.0)
    deadline = time.monotonic() + 3.0
    while not tmux.killed and time.monotonic() < deadline:
        time.sleep(0.02)
    m.close()
    assert tmux.killed == ["jarvis-test"]


# ------------------------------------------ the attack round, 2026-09-12
class TmuxW(Tmux):
    """Tmux with a window count per session and hooks for the race tests."""

    def __init__(self, sessions, windows=None, on_capture=None, slow=None):
        super().__init__(sessions)
        self.windows = dict(windows or {})
        self.on_capture = on_capture
        self.slow = slow           # an Event a capture-pane blocks on

    def __call__(self, argv, timeout=15.0, input=None, cwd=None, env=None):
        if argv[:2] == ["tmux", "list-sessions"] and not self.fail:
            return SimpleNamespace(returncode=0, stderr="", stdout="".join(
                f"{n}\t{int(a)}\t{1 if att else 0}\t{self.windows.get(n, 1)}\n"
                for n, (a, att, _t, _c) in self.sessions.items()))
        if argv[:2] == ["tmux", "capture-pane"]:
            if self.slow is not None:
                self.slow.wait(5.0)
            if self.on_capture is not None:
                self.on_capture()
        return super().__call__(argv, timeout=timeout, input=input, cwd=cwd, env=env)


def test_idle_is_measured_from_the_last_task_not_from_the_sessions_age(tmp_path):
    """tmux's session_activity moves only on attach or a typed key -- never
    on send-keys or pane output -- so for a Jarvis-driven session it is
    the CREATION time. A 9-day-old session that finished a task a minute
    ago is busy by any honest measure and must be kept."""
    tmux = TmuxW({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "claude")})
    m = manager(tmp_path, tmux)
    proj = cs.Project(slug="test", path=str(tmp_path))
    m._projects["test"] = proj
    t = cs.Task(task_id="t1", project="test", cwd=str(tmp_path), prompt="x", model="")
    m._running["t1"] = t
    m._now = lambda: NOW - 60.0
    m._finish(t, proj, "done", "ok")            # the real path stamps the clock
    m._now = lambda: NOW
    assert m.reap_idle() == []
    assert tmux.killed == []
    m._now = lambda: NOW + 4 * H + 1            # four hours later it is idle
    assert m.reap_idle() == ["jarvis-test"]


def test_a_projects_persisted_last_use_survives_a_restart_as_the_floor(tmp_path):
    """After a restart the in-process clock is empty; the project's own
    last_used (persisted in claude_projects.json) still says 'used an hour
    ago', so a 9-day-old session is kept."""
    tmux = TmuxW({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "claude")})
    m = manager(tmp_path, tmux)
    m._projects["test"] = cs.Project(slug="test", path=str(tmp_path), last_used=NOW - H)
    assert m.reap_idle() == []
    assert tmux.killed == []


def test_the_busy_check_is_retaken_under_the_lock_before_the_kill(tmp_path):
    """A submit that lands after the first busy snapshot and before the
    kill registers its task under the lock; the kill must see it."""
    tmux = TmuxW({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "claude")})
    m = manager(tmp_path, tmux)

    def sneak_in():
        with m._lock:
            m._running["late"] = cs.Task(task_id="late", project="test",
                                         cwd=str(tmp_path), prompt="x", model="")
    tmux.on_capture = sneak_in
    assert m.reap_idle() == []
    assert tmux.killed == []


def test_a_session_with_more_than_one_window_is_kept(tmp_path):
    """Jarvis opens exactly one window; a second one means he has been in
    the session himself (vim, a build, a Claude of his own) and the guards
    only ever looked at the current window."""
    tmux = TmuxW({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "claude")},
                 windows={"jarvis-test": 2})
    assert manager(tmp_path, tmux).reap_idle() == []
    assert tmux.killed == []


def test_a_queued_task_protects_its_session(tmp_path):
    tmux = TmuxW({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "claude")})
    m = manager(tmp_path, tmux)
    m._queue.append(cs.Task(task_id="q1", project="test", cwd=str(tmp_path),
                            prompt="x", model=""))
    assert m.reap_idle() == []


def test_a_node_pane_that_is_not_a_resting_claude_tui_is_kept(tmp_path):
    """`node` is the Claude TUI -- and also `npm run dev`. Only a pane
    showing the TUI's idle prompt is a resting Claude."""
    tmux = TmuxW({"jarvis-test": (NOW - 9 * 24 * H, False, "compiled 3 modules\n", "node")})
    assert manager(tmp_path, tmux).reap_idle() == []
    tmux = TmuxW({"jarvis-test": (NOW - 9 * 24 * H, False, "compiled 3 modules\n", "claude")})
    assert manager(tmp_path, tmux).reap_idle() == []
    tmux = TmuxW({"jarvis-test": (NOW - 9 * 24 * H, False, "", "bash")})
    assert manager(tmp_path, tmux).reap_idle() == ["jarvis-test"]


def test_a_boolean_cap_is_not_a_one_second_cap(tmp_path):
    """`true` must not become float(True) == 1.0."""
    m = manager(tmp_path, TmuxW({}), max_idle_s=True)
    assert m.session_max_idle_s == cs.DEFAULT_SESSION_MAX_IDLE_S
    m = manager(tmp_path, TmuxW({}), max_idle_s="4h")
    assert m.session_max_idle_s == cs.DEFAULT_SESSION_MAX_IDLE_S
    def cfg(v):
        return AssistantConfig({"claude": {"reap_every_s": v}})
    assert cs.reap_every_s(cfg("10m")) == cs.DEFAULT_REAP_EVERY_S
    assert cs.reap_every_s(cfg(True)) == cs.DEFAULT_REAP_EVERY_S
    assert cs.reap_every_s(cfg(900)) == 900.0


def test_close_does_not_ask_tmux_when_the_reaper_never_ran(tmp_path):
    """A manager built by a test or a script never started the reaper;
    its close() must not reach for tmux (the desktop firewall counts)."""
    tmux = TmuxW({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "claude")})
    m = manager(tmp_path, tmux)
    m.close()
    assert tmux.killed == []


def test_a_timed_out_stop_keeps_the_thread_and_sweeps_never_overlap(tmp_path):
    """A sweep parked inside tmux outlives a bounded join: the thread
    reference must stay (so reaper_running is honest and a later start
    does not build a second loop) and close()'s own sweep must wait for it
    rather than kill the same session twice."""
    import threading
    gate = threading.Event()
    tmux = TmuxW({"jarvis-test": (NOW - 9 * 24 * H, False, READY, "claude")}, slow=gate)
    m = manager(tmp_path, tmux)
    m.start_reaper(every_s=3600.0)
    deadline = __import__("time").monotonic() + 3.0
    while not m._sweeping and __import__("time").monotonic() < deadline:
        __import__("time").sleep(0.01)
    assert m._sweeping
    m.stop_reaper(join_s=0.05)
    assert m.reaper_running                    # still alive, still reported
    m.start_reaper(every_s=3600.0)             # must not start a second loop
    assert sum(1 for t in threading.enumerate() if t.name == "claude-reaper") == 1
    gate.set()
    m.close()
    assert tmux.killed == ["jarvis-test"]      # once, not twice
    assert not m.reaper_running


def test_the_closing_sweep_is_bounded(tmp_path):
    """A wedged tmux may not hold a quit: the close() sweep has a budget."""
    sessions = {f"jarvis-s{i}": (NOW - 9 * 24 * H, False, READY, "claude") for i in range(6)}
    tmux = TmuxW(sessions)
    m = manager(tmp_path, tmux)
    ticks = iter(range(0, 1000))
    m._monotonic = lambda: next(ticks) * 1.0    # every call costs "a second"
    m.reap_idle(budget_s=2.0)
    assert 0 < len(tmux.killed) < 6
