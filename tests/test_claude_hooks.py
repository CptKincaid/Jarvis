"""scripts/claude_hooks: narration of the user's OWN Claude Code sessions.

narrate.py is driven through main() with hook payloads on a StringIO stdin
and every path pointed at tmp (JARVIS_LOG_DIR for the queue + pid file,
JARVIS_ASSISTANT_CONFIG, JARVIS_HOOK_STATE_DIR); the clock is injected.
install.py is exercised against a tmp settings file — the real
~/.claude/settings.json is never opened.  The drift tests pin the copied
regexes / verdict function to jarvis.claude_session's.
"""
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest

import jarvis.claude_session as cs

HOOKS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "claude_hooks"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"claude_hooks_{name}", HOOKS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # __name__ != "__main__", so no run
    return mod


narrate = _load("narrate")
install = _load("install")

SID = "sess-1234"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A running Jarvis (our own pid), a tmp queue, no config, fresh state."""
    log_dir = tmp_path / "vss"
    log_dir.mkdir()
    (log_dir / "jarvis.pid").write_text(str(os.getpid()))
    monkeypatch.setenv("JARVIS_LOG_DIR", str(log_dir))
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    monkeypatch.setenv("JARVIS_HOOK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("JARVIS_DRIVEN", raising=False)
    return log_dir


def _queue(log_dir):
    p = log_dir / "speak_queue.txt"
    return p.read_text().splitlines() if p.exists() else []


def run(payload, now=1000.0, capsys=None):
    rc = narrate.main(stdin=io.StringIO(json.dumps(payload)), now=now)
    assert rc == 0
    if capsys is not None:
        assert capsys.readouterr().out == "", "a hook must never write stdout"
    return rc


def ev(kind, cwd="/home/hunterp/Jarvis", **extra):
    base = {"hook_event_name": kind, "session_id": SID, "cwd": cwd,
            "transcript_path": "/x/y.jsonl", "permission_mode": "default"}
    base.update(extra)
    return base


def bash(command, stdout="", stderr="", cwd="/home/hunterp/Jarvis"):
    return ev("PostToolUse", cwd=cwd, tool_name="Bash",
              tool_input={"command": command},
              tool_response={"stdout": stdout, "stderr": stderr,
                             "interrupted": False, "isImage": False})


# --------------------------------------------------------------- drift
def test_regexes_are_the_ones_claude_session_uses():
    for name in ("TEST_CMD_RX", "_PASSED_RX", "_FAILED_RX", "_ERRORS_RX"):
        ours, theirs = getattr(narrate, name), getattr(cs, name)
        assert ours.pattern == theirs.pattern, name
        assert ours.flags == theirs.flags, name
    assert narrate.DRIVEN_ENV == cs.DRIVEN_ENV


@pytest.mark.parametrize("text", [
    "3 failed, 10 passed in 1.2s", "1 failed", "12 passed in 0.1s",
    "2 errors", "1 error, 5 passed", "All checks passed!",
    "Success: no issues found in 3 source files", "Traceback (most recent call last)",
    "FAILED tests/x.py::t", "error: something", "nothing to see here", "",
    "Exit code 1\n2 failed, 3 passed in 0.42s",
])
def test_verdicts_match_claude_session(text):
    assert narrate._test_outcome(text) == cs._test_outcome(text)


# ------------------------------------------------------------ PostToolUse
def test_failing_pytest_speaks_the_verdict_with_the_project(env, capsys):
    run(bash("~/vss_env/bin/python -m pytest tests/test_x.py -q",
             stdout="3 failed, 10 passed in 1.2s"), capsys=capsys)
    assert _queue(env) == ["3 tests failed in Jarvis, sir."]


def test_ruff_clean_run_speaks_passed(env):
    run(bash("ruff check jarvis/", stdout="All checks passed!"))
    assert _queue(env) == ["Tests passed in Jarvis, sir."]


def test_non_test_commands_and_other_tools_are_silent(env):
    run(bash("ls -la", stdout="total 3 failed attempts"))
    run(ev("PostToolUse", tool_name="Edit", tool_input={"file_path": "x"},
           tool_response="3 failed"))
    assert _queue(env) == []


def test_bare_string_and_content_list_responses_are_read(env):
    e = bash("pytest -q")
    e["tool_response"] = "1 failed, 2 passed"
    run(e)
    e2 = bash("pytest -q")
    e2["tool_response"] = {"content": [{"type": "text", "text": "4 passed in 0.2s"}]}
    run(e2, now=2000.0)
    assert _queue(env) == ["1 test failed in Jarvis, sir.", "Tests passed in Jarvis, sir."]


def test_project_name_is_spoken_without_hyphens(env):
    run(bash("pytest", stdout="2 passed", cwd="/home/hunterp/haymaker-digest"))
    run(bash("pytest", stdout="2 passed", cwd=""), now=2000.0)
    assert _queue(env) == ["Tests passed in haymaker digest, sir.",
                           "Tests passed in the terminal, sir."]


# ----------------------------------------------------------- Notification
def test_notifications_say_where_claude_is_waiting(env, capsys):
    run(ev("Notification", notification_type="idle_prompt",
           message="Claude is waiting for your input"), capsys=capsys)
    run(ev("Notification", notification_type="permission_prompt",
           message="Claude needs your permission to use Bash",
           cwd="/home/hunterp/haymaker-digest"))
    assert _queue(env) == ["Claude is waiting on you in Jarvis, sir.",
                           "Claude needs your say-so in haymaker digest, sir."]


def test_auth_notifications_are_not_spoken(env):
    run(ev("Notification", notification_type="auth_success", message="Signed in"))
    assert _queue(env) == []


def test_repeated_idle_notifications_are_rate_limited(env):
    idle = ev("Notification", notification_type="idle_prompt", message="waiting")
    run(idle, now=1000.0)
    run(idle, now=1000.0 + narrate.WAITING_REPEAT_S - 1)
    assert len(_queue(env)) == 1
    run(idle, now=1000.0 + narrate.WAITING_REPEAT_S + 1)
    assert len(_queue(env)) == 2


def test_the_limiter_is_per_session(env):
    idle = ev("Notification", notification_type="idle_prompt", message="waiting")
    run(idle)
    other = dict(idle, session_id="sess-other")
    run(other)
    assert len(_queue(env)) == 2


# ------------------------------------------------------------------- Stop
def test_a_quick_chat_turn_does_not_say_finished(env, capsys):
    run(ev("UserPromptSubmit", prompt="hi"), now=1000.0)
    run(ev("Stop", stop_hook_active=False), now=1000.0 + 5, capsys=capsys)
    assert _queue(env) == []


def test_a_long_turn_says_finished(env):
    run(ev("UserPromptSubmit", prompt="refactor everything"), now=1000.0)
    run(ev("Stop", stop_hook_active=False), now=1000.0 + narrate.MIN_TURN_S)
    assert _queue(env) == ["Claude has finished in Jarvis, sir."]


def test_a_turn_that_ran_tests_says_finished_even_if_short(env):
    run(ev("UserPromptSubmit", prompt="run the tests"), now=1000.0)
    run(bash("pytest -q", stdout="5 passed"), now=1003.0)
    run(ev("Stop", stop_hook_active=False), now=1006.0)
    assert _queue(env) == ["Tests passed in Jarvis, sir.", "Claude has finished in Jarvis, sir."]


def test_stop_without_a_prompt_stamp_needs_a_test_run(env):
    """Without the UserPromptSubmit hook installed there is no turn clock;
    only a test run opens the gate, so Stop never becomes a tic."""
    run(ev("Stop", stop_hook_active=False), now=5000.0)
    assert _queue(env) == []


def test_the_gate_resets_after_each_stop(env):
    run(ev("UserPromptSubmit", prompt="x"), now=1000.0)
    run(ev("Stop", stop_hook_active=False), now=1100.0)
    run(ev("Stop", stop_hook_active=False), now=1200.0)
    assert _queue(env) == ["Claude has finished in Jarvis, sir."]


def test_stop_hook_active_is_silent(env):
    run(ev("UserPromptSubmit", prompt="x"), now=1000.0)
    run(ev("Stop", stop_hook_active=True), now=1100.0)
    assert _queue(env) == []


# ---------------------------------------------------------------- gates
def test_jarvis_driven_sessions_are_silent(env, monkeypatch):
    monkeypatch.setenv("JARVIS_DRIVEN", "1")
    run(bash("pytest", stdout="3 failed"))
    run(ev("Notification", notification_type="idle_prompt", message="waiting"))
    assert _queue(env) == []


def test_no_pid_file_means_nobody_to_speak(env):
    (env / "jarvis.pid").unlink()
    run(bash("pytest", stdout="3 failed"))
    assert _queue(env) == []
    assert not (env / "speak_queue.txt").exists()


def test_a_stale_pid_file_is_the_same_as_none(env):
    (env / "jarvis.pid").write_text("4194303")       # PID_MAX_LIMIT: never alive
    run(bash("pytest", stdout="3 failed"))
    (env / "jarvis.pid").write_text("garbage")
    run(bash("pytest", stdout="3 failed"))
    assert _queue(env) == []


def test_alerts_claude_hooks_false_turns_it_off(env, tmp_path):
    cfg = tmp_path / "assistant.json"
    cfg.write_text(json.dumps({"alerts": {"desktop": True, "claude_hooks": False}}))
    run(bash("pytest", stdout="3 failed"))
    assert _queue(env) == []
    cfg.write_text(json.dumps({"alerts": {"desktop": True}}))    # key absent = on
    run(bash("pytest", stdout="3 failed"))
    cfg.write_text("{not json")                                   # broken = default on
    run(bash("pytest", stdout="3 failed"), now=2000.0)
    assert _queue(env) == ["3 tests failed in Jarvis, sir."] * 2


def test_the_default_config_has_the_key():
    from jarvis.assistant_config import DEFAULTS
    assert DEFAULTS["alerts"]["claude_hooks"] is True


def test_garbage_stdin_exits_zero_quietly(env, capsys):
    assert narrate.main(stdin=io.StringIO("{not json"), now=1.0) == 0
    assert narrate.main(stdin=io.StringIO("[1, 2]"), now=1.0) == 0
    assert capsys.readouterr().out == ""
    assert _queue(env) == []


def test_default_paths_are_the_apps(monkeypatch):
    monkeypatch.delenv("JARVIS_LOG_DIR", raising=False)
    monkeypatch.delenv("JARVIS_HOOK_STATE_DIR", raising=False)
    monkeypatch.delenv("JARVIS_CACHE_DIR", raising=False)
    monkeypatch.delenv("JARVIS_ASSISTANT_CONFIG", raising=False)
    assert narrate.speak_queue_path() == Path("/tmp/vss_voice/speak_queue.txt")
    assert narrate.pid_path() == Path("/tmp/vss_voice/jarvis.pid")
    assert narrate.config_path() == Path.home() / ".config" / "jarvis" / "assistant.json"
    assert narrate.state_dir() == Path.home() / ".cache" / "jarvis" / "claude_hooks"


def test_state_files_older_than_a_week_are_pruned(env, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    old = state / "ancient.json"
    old.write_text("{}")
    os.utime(old, (1.0, 1.0))
    run(bash("pytest", stdout="1 passed"), now=10 ** 9)
    assert not old.exists()
    assert (state / f"{SID}.json").exists()


def test_stdlib_only():
    import re
    src = (HOOKS_DIR / "narrate.py").read_text()
    assert not re.search(r"^\s*(?:import|from)\s+jarvis\b", src, re.M)


# ------------------------------------------------------------- build_command
def test_jarvis_driven_panes_carry_the_marker(tmp_path):
    from types import SimpleNamespace
    mgr = cs.ClaudeSessionManager(cfg=None, brain=None, approvals=None,
                                  state_path=tmp_path / "s.json", task_dir=tmp_path,
                                  claude_bin="/opt/bin/claude")
    proj = SimpleNamespace(path=str(tmp_path), slug="p", display="p")
    cmd = mgr.build_command(proj, "opus", session_id="abc", prompt="do it")
    assert cmd.startswith(f"clear; cd {tmp_path} && JARVIS_DRIVEN=1 /opt/bin/claude ")


# ---------------------------------------------------------------- install
def _cmd():
    return install.hook_command("/home/hunterp/vss_env/bin/python")


def test_snippet_covers_the_four_events_with_bash_matcher():
    snip = install.hooks_snippet(_cmd())
    assert set(snip) == {"PostToolUse", "Notification", "Stop", "UserPromptSubmit"}
    assert snip["PostToolUse"][0]["matcher"] == "Bash"
    assert "matcher" not in snip["Stop"][0]
    hook = snip["Stop"][0]["hooks"][0]
    assert hook["type"] == "command" and hook["command"].endswith("narrate.py")
    assert hook["timeout"] == install.TIMEOUT_S


def test_merge_is_idempotent_and_keeps_other_hooks():
    theirs = {"type": "command", "command": "/somewhere/else.sh"}
    settings = {"model": "opus",
                "hooks": {"PostToolUse": [{"matcher": "Edit", "hooks": [theirs]}],
                          "SessionStart": [{"hooks": [theirs]}]}}
    marker = str(install.NARRATE)
    once = install.merge(settings, install.hooks_snippet(_cmd()), marker)
    twice = install.merge(once, install.hooks_snippet(_cmd()), marker)
    assert once == twice
    assert once["model"] == "opus"
    assert once["hooks"]["SessionStart"] == [{"hooks": [theirs]}]
    assert once["hooks"]["PostToolUse"][0] == {"matcher": "Edit", "hooks": [theirs]}
    assert len(once["hooks"]["PostToolUse"]) == 2
    assert settings["hooks"]["PostToolUse"] == [{"matcher": "Edit", "hooks": [theirs]}]


def test_remove_takes_only_ours():
    theirs = {"type": "command", "command": "/somewhere/else.sh"}
    marker = str(install.NARRATE)
    merged = install.merge({"hooks": {"Stop": [{"hooks": [theirs]}]}},
                           install.hooks_snippet(_cmd()), marker)
    back = install.remove(merged, marker)
    assert back == {"hooks": {"Stop": [{"hooks": [theirs]}]}}
    assert install.remove(install.merge({}, install.hooks_snippet(_cmd()), marker), marker) == {}


def test_main_edits_only_the_given_file(tmp_path, capsys):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": "opus"}))
    assert install.main(["--settings", str(settings), "--python", sys.executable,
                         "--dry-run"]) == 0
    assert json.loads(settings.read_text()) == {"model": "opus"}
    assert "narrate.py" in capsys.readouterr().out

    assert install.main(["--settings", str(settings), "--python", sys.executable]) == 0
    data = json.loads(settings.read_text())
    assert data["model"] == "opus" and set(data["hooks"]) == set(install.EVENTS)
    assert (tmp_path / "settings.json.bak-claude-hooks").exists()
    assert install.main(["--settings", str(settings), "--python", sys.executable]) == 0
    assert "already up to date" in capsys.readouterr().out

    assert install.main(["--settings", str(settings), "--uninstall"]) == 0
    assert json.loads(settings.read_text()) == {"model": "opus"}


def test_main_creates_a_missing_settings_file(tmp_path):
    settings = tmp_path / "nested" / "settings.json"
    assert install.main(["--settings", str(settings), "--python", "python3"]) == 0
    data = json.loads(settings.read_text())
    assert data["hooks"]["PostToolUse"][0]["hooks"][0]["command"].startswith("python3 ")
