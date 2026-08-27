"""Tests for jarvis.claude_session — the Claude session manager (spec 7).

Pure and offline: every subprocess goes through the `run` seam (a recorder
that scripts tmux / xdotool / gnome-terminal answers, plays the part of an
INTERACTIVE Claude in the pane — a launch line makes the pane a ready TUI,
the prompt that follows writes a session transcript into a fake
~/.claude/projects tree exactly as Claude Code would), Ollama / Claude are
never called, and all paths live under tmp (tests/conftest.py firewalls
/tmp/vss_voice).
"""
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import jarvis.claude_session as cs
from jarvis.assistant_config import AssistantConfig
from jarvis.events import (ActiveProject, ApprovalRequested, ApprovalResolved,
                           ClaudeProgress, ClaudeTaskState, bus)

FIXTURE = Path(__file__).parent / "fixtures" / "claude_stream_sample.jsonl"
PY = sys.executable


# ------------------------------------------------------------------ helpers
class Recorder:
    """The `_run` seam: records argv, scripts tmux, and plays an
    interactive Claude.

    `tmux send-keys -l <line>` + `Enter` is either the LAUNCH line (it
    starts `clear; cd `) — after which the pane reports `claude` and shows
    a ready TUI, and any prompt in its argv is already delivered — or a
    prompt typed at a warm session, which shows up in the pane's input box
    (`drop_keys` makes the TUI swallow it, as a real one does while it is
    still claiming the tty).  A delivered prompt writes `stream` into
    `<projects_dir>/<encoded cwd>/<session id>.jsonl`.  A bare `Enter`
    with no pending line is the workspace-trust answer.
    """

    READY_PANE = ("─" * 20 + "\n❯ \n" + "─" * 20 +
                  "\n  ⏵⏵ accept edits on (shift+tab to cycle)\n")
    TRUST_PANE = (" Quick safety check: Is this a project you created or one "
                  "you trust?\n ❯ 1. Yes, I trust this folder\n   2. No, exit\n")
    WORKING_PANE = READY_PANE + "  ✻ Thinking… (3s · esc to interrupt)\n"

    def __init__(self, stream=None, projects_dir=None, complete=True,
                 window_ids=None, tmux_exists=False, trust=False,
                 fail_launches=0, clients=(), working=False, drop_keys=0):
        self.calls = []
        self.stream = stream            # transcript entries, or [entries, …]
        self.projects_dir = Path(projects_dir) if projects_dir else None
        self.complete = complete        # False: the prompt writes nothing
        self.window_ids = window_ids or []
        self.tmux_sessions = set()
        self.tmux_exists = tmux_exists
        self.trust = trust              # show the workspace-trust dialog once
        self.fail_launches = fail_launches   # first N launches exit at once
        self.clients = set(clients)     # slugs with a terminal attached
        self.working = working          # the pane shows 'esc to interrupt'
        self.drop_keys = drop_keys      # first N typed prompts are swallowed
        self.launches = []              # every launch line, in order
        self.prompts = []               # (slug, text) every prompt, in order
        self.cancelled = []             # slugs sent C-c
        self.panes = {}                 # slug -> {cwd, sid, alive}
        self.trusted = set()
        self.transcripts = {}           # slug -> Path
        self.plays = 0
        self.typed = {}                 # slug -> text showing in the input box
        self._pending = {}

    # -- the seam ---------------------------------------------------------
    def __call__(self, argv, timeout=15.0, input=None, cwd=None, env=None):
        argv = list(argv)
        self.calls.append(argv)
        rc, out = 0, ""
        if argv[:2] == ["tmux", "has-session"]:
            rc = 0 if (self.tmux_exists or argv[3] in self.tmux_sessions) else 1
        elif argv[:2] == ["tmux", "new-session"]:
            self.tmux_sessions.add(argv[4])
        elif argv[:2] == ["tmux", "list-clients"]:
            out = "/dev/pts/9\n" if self._slug(argv[3]) in self.clients else ""
        elif argv[:2] == ["tmux", "capture-pane"]:
            out = self._pane_text(self._slug(argv[-1]))
        elif argv[:2] == ["tmux", "send-keys"]:
            self._send(argv)
        elif argv[:2] == ["tmux", "list-panes"]:
            out = ("1234\n" if "#{pane_pid}" in argv
                   else self._pane_cmd(self._slug(argv[3])) + "\n")
        elif argv[:2] == ["xdotool", "search"]:
            out = "\n".join(self.window_ids) + ("\n" if self.window_ids else "")
            rc = 0 if self.window_ids else 1
        elif argv[0] == "pgrep":
            rc = 1
        return subprocess.CompletedProcess(argv, rc, out, "")

    # -- the pane ---------------------------------------------------------
    @staticmethod
    def _slug(target):
        return str(target).split("jarvis-", 1)[-1]

    def _pane_cmd(self, slug):
        pane = self.panes.get(slug)
        return "claude" if pane and pane["alive"] else "bash"

    def _pane_text(self, slug):
        pane = self.panes.get(slug)
        if not pane or not pane["alive"]:
            return ""
        if self.trust and slug not in self.trusted:
            return self.TRUST_PANE
        base = self.WORKING_PANE if self.working else self.READY_PANE
        return base.replace("❯ ", "❯ " + self.typed.get(slug, ""))

    def _send(self, argv):
        slug = self._slug(argv[3])
        rest = argv[4:]
        if rest[:1] == ["-l"]:
            text = rest[1]
            self._pending[slug] = text
            if not text.startswith("clear; cd "):
                if self.drop_keys > 0:      # the TUI swallowed the keys
                    self.drop_keys -= 1
                    self._pending.pop(slug, None)
                else:
                    self.typed[slug] = text
            return
        key = rest[0] if rest else ""
        if key == "C-c":
            self.cancelled.append(slug)
            return
        if key == "C-u":                    # the input box is cleared
            self.typed.pop(slug, None)
            self._pending.pop(slug, None)
            return
        if key != "Enter":
            return
        line = self._pending.pop(slug, None)
        if line is None:
            self.trusted.add(slug)          # the trust dialog's answer
        elif line.startswith("clear; cd "):
            self._launch(slug, line)
        else:
            self.typed.pop(slug, None)
            self.prompts.append((slug, line))
            self._play(slug)

    def _launch(self, slug, line):
        self.launches.append(line)
        alive = len(self.launches) > self.fail_launches
        argv = shlex.split(line.split("&&", 1)[1])
        cwd = shlex.split(re.search(r"cd (\S+) &&", line).group(1))[0]
        sid = re.search(r"--(?:session-id|resume) (\S+)", line)
        self.panes[slug] = {"cwd": cwd, "alive": alive,
                            "sid": shlex.split(sid.group(1))[0] if sid else ""}
        # a trailing positional is the prompt: argv is never dropped
        prompt = argv[-1] if len(argv) > 1 and not argv[-2].startswith("-") \
            and not argv[-1].startswith("-") else ""
        if prompt and alive:
            self.prompts.append((slug, prompt))
            self._play(slug)

    def _play(self, slug):
        """Write the transcript Claude Code would write for this turn."""
        if not self.complete or self.projects_dir is None:
            return
        pane = self.panes.get(slug) or {"cwd": "", "sid": ""}
        entries = self.stream or []
        if entries and isinstance(entries[0], list):
            entries = entries[min(self.plays, len(self.stream) - 1)]
        self.plays += 1
        d = self.projects_dir / cs.encode_project_dir(pane["cwd"])
        d.mkdir(parents=True, exist_ok=True)
        path = d / ("%s.jsonl" % (pane["sid"] or "unknown"))
        with open(path, "a", encoding="utf-8") as fh:
            for entry in entries:
                row = dict(entry)
                row.setdefault("cwd", pane["cwd"])
                row.setdefault("sessionId", pane["sid"])
                fh.write(json.dumps(row) + "\n")
        self.transcripts[slug] = path

def fixture_events():
    return [json.loads(ln) for ln in FIXTURE.read_text().splitlines() if ln.strip()]


def transcript_events(final="I changed the router's default and added a test."):
    """The same work as `claude_stream_sample.jsonl`, in the shape Claude
    Code writes to ~/.claude/projects/<enc>/<id>.jsonl: only assistant /
    user records, each with `isSidechain` and a `stop_reason`, and a final
    `end_turn` message that ENDS the turn (there is no `result` record in
    a transcript — that is a print-mode invention)."""
    out = []
    for ev in fixture_events():
        if ev.get("type") not in ("assistant", "user"):
            continue
        row = dict(ev, isSidechain=False)
        msg = dict(row["message"])
        msg.setdefault("stop_reason", "tool_use")
        row["message"] = msg
        out.append(row)
    out.append({"type": "assistant", "isSidechain": False,
                "message": {"role": "assistant", "stop_reason": "end_turn",
                            "content": [{"type": "text", "text": final}]}})
    return out

def wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class Sink:
    """Collects bus events of the given types."""

    def __init__(self, *types):
        self.events = []
        self._types = types
        for t in types:
            bus.subscribe(t, self.events.append)

    def close(self):
        for t in self._types:
            bus.unsubscribe(t, self.events.append)

    def of(self, etype):
        return [e for e in self.events if isinstance(e, etype)]


# ----------------------------------------------------------------- fixtures
@pytest.fixture
def work(tmp_path):
    """Two allowed project dirs, a config on tmp, state/task dirs on tmp,
    and a fake ~/.claude/projects the recorder writes transcripts into."""
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta-tools"
    alpha.mkdir()
    beta.mkdir()
    root = tmp_path / "projects"
    cfg = AssistantConfig(
        {"claude": {"allowed_dirs": [str(alpha), str(beta)],
                    "projects_root": str(root),
                    "model": "opus", "big_model": "fable", "effort": ""}},
        path=tmp_path / "assistant.json")
    sessions = tmp_path / "claude-projects"
    sessions.mkdir()
    return SimpleNamespace(tmp=tmp_path, alpha=alpha, beta=beta, root=root,
                           cfg=cfg, state=tmp_path / "claude_projects.json",
                           task_dir=tmp_path / "log" / "claude",
                           sessions=sessions,
                           sock=tmp_path / "log" / "approvals.sock")


@pytest.fixture
def sink():
    s = Sink(ClaudeTaskState, ClaudeProgress, ActiveProject)
    yield s
    s.close()


def make_manager(work, run, brain=None, claude_bin="/opt/bin/claude", **kw):
    approvals = SimpleNamespace(sock_path=work.sock, pending=lambda: [])
    kw.setdefault("home", work.tmp)           # tmp stands in for ~ (7.5)
    kw.setdefault("ready_grace_s", 0.05)      # how long a dead pane may look busy
    kw.setdefault("settle_s", 0.02)           # quiet after the turn-ending message
    kw.setdefault("idle_s", 30.0)             # silence + an idle pane = no result
    kw.setdefault("key_wait_s", 0.2)          # confirm typed keys landed
    if isinstance(run, Recorder) and run.projects_dir is None:
        run.projects_dir = work.sessions
    m = cs.ClaudeSessionManager(work.cfg, brain or SimpleNamespace(), approvals,
                                work.state, work.task_dir, run=run,
                                claude_bin=claude_bin, python=PY, poll_s=0.01,
                                projects_dir=work.sessions, **kw)
    return m

def join_runners(m, timeout=5.0):
    for th in list(m._threads.values()):
        th.join(timeout)
    assert not m._threads, "a runner is still alive"


def sent_commands(rec):
    """Every prompt typed into a pane (the launch lines live in .launches)."""
    return [text for _slug, text in rec.prompts]

# ------------------------------------------------------------ pure helpers
def test_slugify_and_display():
    assert cs.slugify("Weather Bot!") == "weather-bot"
    assert cs.slugify("  My__Great  App 2 ") == "my-great-app-2"
    assert cs.slugify("../../etc") == "etc"
    assert cs.slugify("") == ""
    assert cs.display_name("haymaker-digest") == "haymaker digest"


@pytest.mark.parametrize("prompt,size", [
    ("fix the typo in README.md", "small"),
    ("add a docstring to route()", "small"),
    ("refactor the router across all the modules", "large"),
    ("implement the calendar feature with tests", "large"),
    ("debug why the tests fail and the pipeline hangs", "large"),
    ("migrate the config to toml", "large"),
    ("rename the function", "small"),
])
def test_estimate_size(prompt, size):
    assert cs.estimate_size(prompt) == size


def test_ago_wording():
    assert cs._ago(5) == "just now"
    assert cs._ago(70) == "a minute ago"
    assert cs._ago(4 * 60 + 3) == "four minutes ago"
    assert cs._ago(25 * 60) == "25 minutes ago"
    assert cs._ago(3600) == "an hour ago"
    assert cs._ago(3 * 3600) == "three hours ago"


# --------------------------------------------------------- parse_stream_event
def test_parse_fixture_stream_milestones():
    task = cs.Task("t1", "w7-smoke", "p", "opus", cwd="/home/hunterp/projects/w7-smoke")
    lines = []
    now = 1000.0
    for ev in fixture_events():
        now += 1.0                       # every event a second apart
        lines += cs.parse_stream_event(ev, task, now=now)
    assert task.session_id == "11111111-2222-3333-4444-555555555555"
    texts = [(p.milestone, p.line) for p in lines]
    assert all(p.project == "w7-smoke" and p.task_id == "t1" for p in lines)
    # compact lines, project-relative paths
    assert (False, "Read jarvis/router.py") in texts
    assert (False, "Edit jarvis/router.py") in texts
    assert (False, "Write tests/test_router.py") in texts
    # first edit per file -> spoken, coalesced: the Write 2 s later is held
    assert (True, "Editing router.py, sir.") in texts
    assert not any("test_router" in ln and m for m, ln in texts)
    assert task.pending_files == ["test_router.py"]
    assert task.files_touched == {"/home/hunterp/projects/w7-smoke/jarvis/router.py",
                                  "/home/hunterp/projects/w7-smoke/tests/test_router.py"}
    # a test run and its verdict (the verdict is exempt from the 20 s limiter)
    assert any("Bash cd /home/hunterp/projects/w7-smoke && python -m pytest" in ln for _, ln in texts)
    assert (True, "Tests passed, sir.") in texts
    # one error is a compact line, not spoken
    assert (False, "Error: cat: /nonexistent: No such file or directory") in texts
    # prose excerpt, unknown event type ignored, result recorded
    assert (False, "I'll start by reading the router.") in texts
    assert task.result_text.startswith("I changed the router's default")
    assert task.rc == 0 and task.turns == 6
    assert (False, "Done · 6 turns · 34 s") in texts
    assert task.events == len(fixture_events())


def test_parse_rate_limit_and_exemptions():
    task = cs.Task("t", "p", "x", "opus", cwd="/w")
    def ev_edit(f, i):
        return {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": i, "name": "Edit",
             "input": {"file_path": f"/w/{f}"}}]}}

    a = cs.parse_stream_event(ev_edit("a.py", "1"), task, now=100.0)
    b = cs.parse_stream_event(ev_edit("b.py", "2"), task, now=105.0)
    c = cs.parse_stream_event(ev_edit("c.py", "3"), task, now=125.0)
    assert [p.line for p in a if p.milestone] == ["Editing a.py, sir."]
    assert not any(p.milestone for p in b)
    assert [p.line for p in c if p.milestone] == ["Editing b.py and c.py, sir."]
    # an edit to a file already touched is not a new milestone name
    d = cs.parse_stream_event(ev_edit("a.py", "4"), task, now=200.0)
    assert not any(p.milestone for p in d)
    # three consecutive errors -> one spoken line, exempt from the limiter
    task2 = cs.Task("t2", "p", "x", "opus")
    tool = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "b", "name": "Bash", "input": {"command": "pytest -q"}}]}}
    cs.parse_stream_event(tool, task2, now=1.0)
    err = {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "b", "content": "boom", "is_error": True}]}}
    out = []
    for i in range(3):
        out += cs.parse_stream_event(err, task2, now=2.0 + i)
    spoken = [p.line for p in out if p.milestone]
    assert spoken == ["Hitting errors, sir; carrying on."]


def test_parse_test_verdicts_plan_and_failure():
    task = cs.Task("t", "p", "x", "opus")
    run = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "npm test"}}]}}
    def res(text):
        return {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "b1", "content": text}]}}

    assert [p.line for p in cs.parse_stream_event(run, task, now=1) if p.milestone] == \
        ["Running the tests, sir."]
    assert [p.line for p in cs.parse_stream_event(res("3 passed, 2 failed in 1s"), task, now=2)
            if p.milestone] == ["2 tests failed, sir."]
    cs.parse_stream_event(run, task, now=30)
    assert [p.line for p in cs.parse_stream_event(res("Traceback (most recent call last)"), task, now=31)
            if p.milestone] == ["The tests errored, sir."]
    plan = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "p1", "name": "ExitPlanMode",
         "input": {"plan": "# Plan\nSplit the router into two modules. Then add tests."}}]}}
    lines = cs.parse_stream_event(plan, task, now=100)
    assert [p.line for p in lines if p.milestone] == ["Plan chosen: Split the router into two modules."]
    prose = {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "## Plan: rewrite the tail loop. Then test it."}]}}
    lines = cs.parse_stream_event(prose, task, now=200)
    assert [p.line for p in lines if p.milestone] == ["Plan chosen: rewrite the tail loop."]
    fail = {"type": "result", "subtype": "error_during_execution", "is_error": True,
            "result": "Context window exceeded", "num_turns": 2}
    lines = cs.parse_stream_event(fail, task, now=300)
    assert lines[0].milestone and lines[0].line == "Claude stopped, sir: Context window exceeded"
    assert task.rc == 1 and task.error == "Context window exceeded"


def test_parse_is_tolerant():
    task = cs.Task("t", "p", "x", "opus")
    for ev in [None, 3, "str", {}, {"type": "assistant"}, {"type": "assistant", "message": None},
               {"type": "assistant", "message": {"content": [None, {"type": "tool_use"}]}},
               {"type": "user", "message": {"content": "plain"}},
               {"type": "result"}, {"type": "system", "subtype": "compact_boundary"},
               {"type": "stream_event", "event": {}}]:
        cs.parse_stream_event(ev, task, now=1)      # must not raise
    assert task.events == 11


def test_describe_event_pane_lines():
    evs = fixture_events()
    assert cs.describe_event(evs[0]) == ["── session 11111111-2222-3333-4444-555555555555 · claude-opus-4-1-20250805 ──"]
    assert cs.describe_event(evs[1]) == ["I'll start by reading the router."]
    assert cs.describe_event(evs[2], "/home/hunterp/projects/w7-smoke") == ["▸ Read jarvis/router.py"]
    assert cs.describe_event(evs[11]) == ["  ✗ cat: /nonexistent: No such file or directory"]
    assert cs.describe_event(evs[-1]) == ["── done · 6 turns · 34 s ──"]
    assert cs.describe_event({"type": "rate_limit_event"}) == []


def test_render_subcommand_copies_raw_and_prints(tmp_path):
    out = tmp_path / "copy.jsonl"
    raw = FIXTURE.read_bytes() + b"not json\n"
    env = dict(os.environ, JARVIS_LOG_DIR=str(tmp_path / "log"),
               PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    r = subprocess.run([PY, "-m", "jarvis.claude_session", "render", "--out", str(out),
                        "--cwd", "/home/hunterp/projects/w7-smoke"],
                       input=raw, capture_output=True, env=env, timeout=30)
    assert r.returncode == 0, r.stderr.decode()
    assert out.read_bytes() == raw                 # byte-exact copy, bad line included
    pane = r.stdout.decode()
    assert "▸ Edit jarvis/router.py" in pane
    assert "── done · 6 turns · 34 s ──" in pane
    assert "not json" not in pane


# ------------------------------------------------------------------ projects
def test_projects_from_allowed_dirs_and_lookup(work):
    m = make_manager(work, Recorder())
    try:
        slugs = [p.slug for p in m.projects()]
        assert slugs == ["alpha", "beta-tools"]
        assert m.active_project is None
        assert m.project_for("alpha").path == str(work.alpha)
        assert m.project_for("the beta tools project").slug == "beta-tools"
        assert m.project_for("Beta").slug == "beta-tools"
        assert m.project_for(str(work.alpha)).slug == "alpha"
        assert m.project_for("gamma") is None
        assert m.work_on("gamma") == "I don't know a project called gamma, sir; shall I set one up?"
    finally:
        m.close()


def test_work_on_sets_active_and_publishes(work, sink):
    m = make_manager(work, Recorder())
    try:
        assert m.work_on("beta tools") == "Beta tools it is, sir."
        assert m.active_project == "beta-tools"
        ev = sink.of(ActiveProject)[-1]
        assert (ev.slug, ev.path) == ("beta-tools", str(work.beta))
        # persisted
        data = json.loads(work.state.read_text())
        assert data["active"] == "beta-tools"
        m2 = make_manager(work, Recorder())
        assert m2.active_project == "beta-tools"
        m2.close()
    finally:
        m.close()


def test_missing_claude_binary_returns_setup_line(work):
    m = make_manager(work, Recorder(), claude_bin="")
    try:
        line = m.submit("fix it", project="alpha")
        assert line == work.cfg.setup_line("claude")
        assert "Claude command line" in line
    finally:
        m.close()


# ---------------------------------------------------------------- the runner
def test_submit_runs_pipeline_exactly(work, sink):
    """The pane runs Claude INTERACTIVELY (no -p, no stream-json, no pipe),
    the prompt is typed into it, and every milestone comes from the session
    transcript Claude Code writes as it works."""
    rec = Recorder(stream=transcript_events())
    m = make_manager(work, rec)
    try:
        task = m.submit("add a docstring to route()", project="alpha")
        assert isinstance(task, cs.Task)
        join_runners(m)
        assert task.state == "done"
        assert task.result_text == "I changed the router's default and added a test."
        # tmux argv exact: create the session, then CLAIM ITS GEOMETRY —
        # a detached session ignores -x/-y and 80x24 wraps the TUI
        assert rec.calls[0] == ["tmux", "has-session", "-t", "jarvis-alpha"]
        assert rec.calls[1] == ["tmux", "new-session", "-d", "-s", "jarvis-alpha",
                                "-c", str(work.alpha), "-x", "200", "-y", "50"]
        assert ["tmux", "set-option", "-t", "jarvis-alpha", "window-size",
                "manual"] in rec.calls
        assert ["tmux", "resize-window", "-t", "jarvis-alpha", "-x", "200",
                "-y", "50"] in rec.calls
        # the launch line
        assert len(rec.launches) == 1
        cmd = rec.launches[0]
        mcp = work.task_dir / "alpha" / "mcp_jarvis.json"
        suffix = work.task_dir / "system_suffix.txt"
        sid = rec.panes["alpha"]["sid"]
        expected = (
            f"clear; cd {shlex.quote(str(work.alpha))} && /opt/bin/claude "
            f"--model opus --session-id {sid} "
            f"--permission-mode acceptEdits "
            f"--mcp-config {shlex.quote(str(mcp))} "
            f"--permission-prompt-tool mcp__jarvis__approve "
            f"--append-system-prompt-file {shlex.quote(str(suffix))} "
            f"'add a docstring to route()'")
        assert cmd == expected
        assert "-p " not in cmd and "--output-format" not in cmd
        assert "|" not in cmd and "--resume" not in cmd
        # a session we start ourselves carries the prompt in its argv (keys
        # typed at a TUI that is still claiming the tty get dropped), and
        # the whole line goes in with -l so tmux reads no word as a KEY
        assert rec.prompts == [("alpha", "add a docstring to route()")]
        idx = [i for i, c in enumerate(rec.calls)
               if c[:2] == ["tmux", "send-keys"] and c[4:5] == ["-l"]][0]
        assert rec.calls[idx] == ["tmux", "send-keys", "-t", "jarvis-alpha",
                                  "-l", expected]
        assert rec.calls[idx + 1] == ["tmux", "send-keys", "-t", "jarvis-alpha",
                                      "Enter"]
        # files
        p = task.paths
        assert Path(p["prompt"]).read_text() == "add a docstring to route()\n"
        assert Path(p["transcript"]) == rec.transcripts["alpha"]
        assert suffix.read_text() == cs.SYSTEM_SUFFIX
        mcp_cfg = json.loads(mcp.read_text())["mcpServers"]["jarvis"]
        assert mcp_cfg["command"] == PY
        assert mcp_cfg["env"]["JARVIS_PROJECT"] == "alpha"
        settings = json.loads((work.alpha / ".claude" / "settings.local.json").read_text())
        assert settings["permissions"]["allow"] == cs.ALLOW_RULES
        assert settings["permissions"]["deny"] == cs.DENY_RULES
        # the transcript's session id is stored per project
        assert m.project_for("alpha").session_id == sid
        data = json.loads(work.state.read_text())
        assert data["projects"]["alpha"]["session_id"] == sid
        assert data["active"] == "alpha"
        # events: running -> done, progress lines, milestones
        states = [(e.state, e.project) for e in sink.of(ClaudeTaskState)]
        assert states[0] == ("running", "alpha") and states[-1] == ("done", "alpha")
        assert sink.of(ClaudeTaskState)[-1].text == task.result_text
        progress = sink.of(ClaudeProgress)
        assert any(p.line == "Started · opus" and not p.milestone for p in progress)
        assert any(p.milestone and p.line == "Editing router.py, sir." for p in progress)
        assert any(p.milestone and p.line == "Tests passed, sir." for p in progress)
        assert m.status_text() == "Claude's idle, sir; the active project is alpha."
    finally:
        m.close()


def test_second_task_reuses_the_live_session(work):
    """One interactive Claude per project: the second task is another
    prompt typed into the SAME session, not a second `claude` run."""
    rec = Recorder(stream=transcript_events())
    work.cfg.set("claude.effort", "high")
    m = make_manager(work, rec)
    try:
        m.submit("first", project="alpha")
        join_runners(m)
        assert m.set_model("use sonnet") == "Sonnet it is, sir."
        assert m.set_fast_mode(True) == "Fast mode on, sir."
        m.submit("second", project="alpha")
        join_runners(m)
        assert len(rec.launches) == 1                    # one Claude, two turns
        assert sent_commands(rec) == ["first", "second"]
        assert "--effort high" in rec.launches[0]
        assert "--model opus" in rec.launches[0]
        assert m.set_fast_mode(False) == "Fast mode off, sir."
        assert m.set_model("gpt-9") == "I only know opus, sonnet, fable and haiku, sir."
        # sticky override survives a new manager on the same state file
        m2 = make_manager(work, Recorder())
        assert m2.model == "sonnet"
        m2.close()
    finally:
        m.close()


def test_a_dead_pane_is_resumed_with_the_stored_session_id(work):
    """When the pane no longer holds a Claude, the next task starts one and
    picks the conversation up with --resume (never --continue, which would
    hijack Hunter's own terminal session)."""
    rec = Recorder(stream=transcript_events())
    m = make_manager(work, rec)
    try:
        m.submit("first", project="alpha")
        join_runners(m)
        sid = rec.panes["alpha"]["sid"]
        rec.panes["alpha"]["alive"] = False              # /exit, or a crash
        m.submit("second", project="alpha")
        join_runners(m)
        assert len(rec.launches) == 2
        assert f"--resume {sid}" in rec.launches[1]
        assert "--session-id" not in rec.launches[1]
        assert "--continue" not in rec.launches[1]
        assert "--model sonnet" in rec.launches[1] or "--model opus" in rec.launches[1]
    finally:
        m.close()


def test_model_escalation_and_override(work):
    m = make_manager(work, Recorder())
    try:
        assert m.model_for("fix the typo") == "opus"
        assert m.model_for("refactor the router across all the modules") == "fable"
        assert m.model_for("refactor everything", override="haiku") == "haiku"
        assert m.model_for("refactor everything", override="use opus please") == "opus"
        m.set_model("fable")
        assert m.model_for("fix the typo") == "fable"
    finally:
        m.close()


def test_dangerous_flag_only_when_configured(work):
    rec = Recorder(stream=transcript_events())
    work.cfg.set("claude.dangerously_skip_permissions", True)
    m = make_manager(work, rec)
    try:
        m.submit("x", project="alpha")
        join_runners(m)
        cmd = rec.launches[0]
        assert "--dangerously-skip-permissions" in cmd
        # nothing can be asked when permissions are skipped, so the prompt
        # tool is not attached; the mcp config still is
        assert "--permission-prompt-tool" not in cmd
        assert "--mcp-config" in cmd
        assert "--permission-mode" not in cmd
    finally:
        m.close()


def test_permission_prompt_tool_on_by_default_and_switchable(work):
    """7.3, verified against CLI 2.1.247 on 2026-08-27: the pair is emitted by
    default (and matters more now the pane is read-only — an in-pane question
    is one nobody can answer), and `claude.permission_prompt_tool` false still
    turns it off."""
    rec = Recorder(stream=transcript_events())
    m = make_manager(work, rec)
    try:
        assert m.permission_prompt_tool is True
        m.submit("x", project="alpha")
        join_runners(m)
        cmd = rec.launches[0]
        mcp = work.task_dir / "alpha" / "mcp_jarvis.json"
        assert f"--mcp-config {shlex.quote(str(mcp))}" in cmd
        assert "--permission-prompt-tool mcp__jarvis__approve" in cmd
        assert cs.PERMISSION_TOOL == "mcp__jarvis__approve"
        work.cfg.set("claude.permission_prompt_tool", False)
        assert m.permission_prompt_tool is False
        rec.panes["alpha"]["alive"] = False       # force a fresh launch
        m.submit("y", project="alpha")
        join_runners(m)
        assert "--permission-prompt-tool" not in rec.launches[1]
    finally:
        m.close()


def test_outside_allowed_dirs_runs_under_the_prompt_tool(work, sink):
    """With `claude.auto_approve_anywhere` off a stray dir is no longer
    refused: the verified prompt tool gives Jarvis something to ask through,
    so the task starts and each out-of-project action becomes a question."""
    rec = Recorder(stream=transcript_events())
    stray = work.tmp / "stray"
    stray.mkdir()
    work.cfg.set("claude.auto_approve_anywhere", False)
    m = make_manager(work, rec)
    try:
        m._projects["stray"] = cs.Project(slug="stray", path=str(stray))
        assert m.project_allowed("alpha") is True
        assert m.project_allowed("stray") is False
        task = m.submit("touch it", project="stray")
        assert isinstance(task, cs.Task)                 # not OUTSIDE_LINE
        join_runners(m)
        assert "--permission-prompt-tool mcp__jarvis__approve" in rec.launches[0]
        states = [e.state for e in sink.of(ClaudeTaskState)]
        assert states[0] == "running" and states[-1] == "done"
    finally:
        m.close()


def test_outside_allowed_dirs_fails_closed_without_the_prompt_tool(work, sink):
    """The real fallback: no prompt tool means nothing to ask through, so
    work outside the cleared dirs is refused in persona and the terminal is
    offered instead."""
    rec = Recorder(stream=transcript_events())
    stray = work.tmp / "stray"
    stray.mkdir()
    work.cfg.set("claude.auto_approve_anywhere", False)
    work.cfg.set("claude.permission_prompt_tool", False)
    m = make_manager(work, rec)
    try:
        m._projects["stray"] = cs.Project(slug="stray", path=str(stray))
        line = m.submit("touch it", project="stray")
        assert line == cs.OUTSIDE_LINE
        assert "cleared for me, sir" in line and "terminal" in line
        assert rec.launches == [] and rec.prompts == []
        assert sink.of(ClaudeTaskState) == []
        # the pop-out terminal is still available for it
        assert m.open_terminal("stray") is True
    finally:
        m.close()


def test_outside_allowed_dirs_runs_by_default(work, sink):
    """User decision 2026-08-26: auto-approval is NOT limited to the configured
    roots, so a stray dir is accepted and actually started."""
    rec = Recorder(stream=transcript_events())
    stray = work.tmp / "stray"
    stray.mkdir()
    m = make_manager(work, rec)
    try:
        assert m.auto_approve_anywhere is True
        m._projects["stray"] = cs.Project(slug="stray", path=str(stray))
        assert m.project_allowed("stray") is True
        task = m.submit("touch it", project="stray")
        assert isinstance(task, cs.Task)                 # not OUTSIDE_LINE
        join_runners(m)
        assert task.state == "done"
        assert len(rec.launches) == 1
        assert f"cd {shlex.quote(str(stray))} &&" in rec.launches[0]
        states = [e.state for e in sink.of(ClaudeTaskState)]
        assert states[0] == "running" and states[-1] == "done"
        # the stray project gets the same broad allow rules as a cleared one
        settings = json.loads((stray / ".claude" / "settings.local.json").read_text())
        assert settings["permissions"]["allow"] == cs.ALLOW_RULES
    finally:
        m.close()


def test_project_allowed_outside_dirs_follows_the_flag(work, tmp_path):
    """A project outside claude.allowed_dirs: allowed by default, refused with
    `claude.auto_approve_anywhere` off."""
    stray = tmp_path / "stray-project"
    stray.mkdir()
    m = make_manager(work, Recorder())
    try:
        m._projects["stray-project"] = cs.Project(slug="stray-project",
                                                  path=str(stray))
        assert str(stray) not in work.cfg.allowed_dirs
        assert m.project_allowed("stray-project") is True
        assert m.project_allowed(cs.Project(slug="ad-hoc", path=str(stray))) is True
        assert m.project_allowed("gamma") is False       # not a project at all
        work.cfg.set("claude.auto_approve_anywhere", False)
        assert m.project_allowed("stray-project") is False
        assert m.project_allowed(cs.Project(slug="ad-hoc", path=str(stray))) is False
        assert m.project_allowed("alpha") is True        # a cleared dir still is
    finally:
        m.close()


def test_auto_approve_anywhere_property(work):
    m = make_manager(work, Recorder())
    try:
        assert m.auto_approve_anywhere is True           # DEFAULTS supply it
        work.cfg.set("claude.auto_approve_anywhere", False)
        assert m.auto_approve_anywhere is False
        work.cfg.set("claude.auto_approve_anywhere", True)
        assert m.auto_approve_anywhere is True
        # a config that has never heard of the key still defaults to true
        work.cfg._data["claude"].pop("auto_approve_anywhere")
        assert work.cfg.get("claude.auto_approve_anywhere") is None
        assert m.auto_approve_anywhere is True
    finally:
        m.close()


def test_resume_failure_retries_without_resume(work, sink):
    """A stale --resume makes the interactive Claude exit at once; the pane
    falls back to the shell, and the task retries with a fresh session."""
    rec = Recorder(stream=transcript_events(), fail_launches=1)
    m = make_manager(work, rec)
    proj = m.project_for("alpha")
    proj.session_id = "stale-session-id"
    try:
        task = m.submit("continue", project="alpha")
        join_runners(m)
        assert len(rec.launches) == 2
        assert "--resume stale-session-id" in rec.launches[0]
        assert "--resume" not in rec.launches[1]
        assert "--session-id" in rec.launches[1]
        assert task.state == "done" and task.attempt == 2
        assert m.project_for("alpha").session_id == rec.panes["alpha"]["sid"]
    finally:
        m.close()


def test_a_launch_that_never_comes_up_is_failed(work, sink):
    rec = Recorder(stream=transcript_events(), fail_launches=5)
    m = make_manager(work, rec)
    try:
        task = m.submit("x", project="alpha")
        join_runners(m)
        assert task.state == "failed"
        assert sink.of(ClaudeTaskState)[-1].text == \
            "I couldn't get Claude started in the terminal, sir."
        assert rec.prompts == []          # nothing was typed at a dead pane
    finally:
        m.close()


def test_claude_leaving_the_pane_mid_task_is_failed(work, sink):
    rec = Recorder(complete=False)
    m = make_manager(work, rec)
    try:
        task = m.submit("x", project="alpha")
        assert wait_for(lambda: "alpha" in m._panes)     # the tail has started
        rec.panes["alpha"]["alive"] = False              # /exit, or a crash
        # wait for the EVENT, not the field: _finish sets task.state first
        assert wait_for(lambda: any(e.state == "failed"
                                    for e in sink.of(ClaudeTaskState)), timeout=15)
        last = sink.of(ClaudeTaskState)[-1]
        assert last.state == "failed" and last.text == "Claude's process vanished, sir."
        assert task.state == "failed"
        assert [p.line for p in sink.of(ClaudeProgress) if p.milestone] == \
            ["Claude's process vanished, sir."]
        assert "alpha" not in m._panes                   # the pane is forgotten
    finally:
        join_runners_soft(m)
        m.close()


def test_silence_with_an_idle_pane_is_failed_with_a_spoken_line(work, sink):
    """The prompt went in, the transcript stayed empty and the TUI is back
    at its prompt: Claude produced nothing, and Jarvis says so."""
    rec = Recorder(complete=False)
    m = make_manager(work, rec, idle_s=0.05)
    try:
        task = m.submit("x", project="alpha")
        join_runners(m)
        assert task.state == "failed"
        spoken = [p.line for p in sink.of(ClaudeProgress) if p.milestone]
        assert spoken == [cs.NO_RESULT_LINE] == ["Claude stopped without a result, sir."]
    finally:
        m.close()


def test_a_working_pane_is_not_mistaken_for_silence(work, sink):
    """'esc to interrupt' in the pane means a turn is genuinely in flight,
    so a long quiet Bash run is never cut short."""
    rec = Recorder(complete=False, working=True)
    m = make_manager(work, rec, idle_s=0.05)
    try:
        task = m.submit("x", project="alpha")
        assert wait_for(lambda: rec.prompts != [])
        time.sleep(0.3)
        assert task.state == "running"
    finally:
        task.state = "cancelled"
        join_runners_soft(m)
        m.close()


# ------------------------------------------------------------ queue / cancel
def test_queue_rules_and_cancel(work, sink):
    rec = Recorder(complete=False)
    m = make_manager(work, rec)
    try:
        t1 = m.submit("one", project="alpha")
        assert isinstance(t1, cs.Task) and t1.state == "running"
        # same project -> queued with the busy line
        line = m.submit("two", project="alpha")
        assert line == "Claude's still on the last one for alpha, sir; I've queued it."
        # another project without parallel -> queued too
        line = m.submit("three", project="beta")
        assert line == "Claude's still on the last one for alpha, sir; I've queued it."
        assert [e.state for e in sink.of(ClaudeTaskState)] == ["running", "queued", "queued"]
        assert wait_for(lambda: rec.tmux_sessions == {"jarvis-alpha"})
        assert m.status_text().startswith("Claude's working on alpha, sir; started just now.")
        assert m.status_text().endswith("2 more are queued.")
        # parallel explicitly -> runs alongside
        t4 = m.submit("four", project="beta", parallel=True)
        assert isinstance(t4, cs.Task) and t4.state == "running"
        assert wait_for(lambda: "jarvis-beta-tools" in rec.tmux_sessions)
        # cancel the alpha task: C-c into its pane, and it is done
        assert m.cancel("alpha") is True
        assert t1.state == "cancelled"
        assert ["tmux", "send-keys", "-t", "jarvis-alpha", "C-c"] in rec.calls
        cancelled = [e for e in sink.of(ClaudeTaskState) if e.state == "cancelled"]
        assert cancelled and cancelled[0].task_id == t1.task_id
        assert cancelled[0].text == cs.CANCELLED_LINE
        # "two" still waits: a non-parallel task never starts alongside the
        # beta one that is still running
        assert all(t.state == "queued" for t in m._queue)
        assert [t.prompt for t in m.running_tasks()] == ["four"]
        assert m.cancel("beta") is True and t4.state == "cancelled"
        assert wait_for(lambda: any(t.prompt == "two" and t.state == "running"
                                    for t in m.running_tasks()))
        join_runners_soft(m)
        assert m.cancel("gamma") is False
    finally:
        for t in m.running_tasks():
            t.state = "cancelled"
        join_runners_soft(m)
        m.close()


def join_runners_soft(m, timeout=2.0):
    for th in list(m._threads.values()):
        th.join(timeout)


def test_cancel_interrupts_without_killing_the_session(work, sink):
    """C-c interrupts the turn in the TUI; the interactive Claude — the very
    session Hunter is watching — is never killed."""
    rec = Recorder(complete=False)
    m = make_manager(work, rec)
    try:
        t = m.submit("hang", project="alpha")
        assert isinstance(t, cs.Task)
        assert wait_for(lambda: rec.prompts != [])
        assert m.cancel() is True
        assert rec.cancelled == ["alpha"]
        assert t.state == "cancelled"
        assert not any(c[0] == "pgrep" for c in rec.calls)
        assert rec.panes["alpha"]["alive"] is True       # still there to watch
    finally:
        join_runners_soft(m)
        m.close()


def test_cancel_drops_a_queued_task(work, sink):
    rec = Recorder(complete=False)
    m = make_manager(work, rec)
    try:
        m.submit("one", project="alpha")
        m.submit("two", project="alpha")
        assert len(m._queue) == 1
        assert m.cancel("alpha") is True           # the running one
        join_runners_soft(m)
        # "two" started after the cancel; cancel again stops it
        assert wait_for(lambda: any(t.prompt == "two" for t in m.running_tasks()))
        assert m.cancel("alpha") is True
        join_runners_soft(m)
        assert m.cancel("alpha") is False
    finally:
        for t in m.running_tasks():
            t.state = "cancelled"
        join_runners_soft(m)
        m.close()


# --------------------------------------------------------- approvals hook
def test_approval_events_toggle_waiting(work, sink):
    rec = Recorder(complete=False)
    m = make_manager(work, rec)
    try:
        t = m.submit("x", project="alpha")
        bus.publish(ApprovalRequested(request_id="r1", question="Shall I?", tool_name="Write",
                                      detail="/etc/x", project="alpha"))
        assert t.state == "waiting"
        assert m.status_text().startswith("Claude's waiting on you about alpha, sir")
        bus.publish(ApprovalResolved(request_id="r1", allowed=True, source="typed"))
        assert t.state == "running"
        states = [e.state for e in sink.of(ClaudeTaskState)]
        assert states == ["running", "waiting", "running"]
    finally:
        t.state = "cancelled"
        join_runners_soft(m)
        m.close()


# ------------------------------------------------------ settings.local.json
def test_ensure_project_settings_merges(work):
    m = make_manager(work, Recorder())
    try:
        proj = m.project_for("alpha")
        path = work.alpha / ".claude" / "settings.local.json"
        path.parent.mkdir()
        path.write_text(json.dumps({
            "permissions": {"allow": ["Bash(npm run build:*)", "Read(/**)"],
                            "deny": ["WebFetch(domain:evil.example)"],
                            "additionalDirectories": ["../shared"]},
            "env": {"FOO": "1"}, "model": "sonnet"}))
        out = m.ensure_project_settings(proj)
        assert out == path
        data = json.loads(path.read_text())
        allow = data["permissions"]["allow"]
        assert allow[:2] == ["Bash(npm run build:*)", "Read(/**)"]     # user's first, kept
        assert allow.count("Read(/**)") == 1
        for rule in cs.ALLOW_RULES:
            assert rule in allow
        assert data["permissions"]["deny"] == ["WebFetch(domain:evil.example)"] + cs.DENY_RULES
        assert data["permissions"]["additionalDirectories"] == ["../shared"]
        assert data["env"] == {"FOO": "1"} and data["model"] == "sonnet"
        # idempotent: unchanged content is not rewritten
        before = path.stat().st_mtime_ns
        m.ensure_project_settings("alpha")
        assert path.stat().st_mtime_ns == before
        # a corrupt file is kept as .bad and replaced
        path.write_text("{not json")
        m.ensure_project_settings(proj)
        assert (path.with_name("settings.local.json.bad")).read_text() == "{not json"
        assert json.loads(path.read_text())["permissions"]["allow"] == cs.ALLOW_RULES
    finally:
        m.close()


def test_the_invalid_mcp_rule_is_pruned_from_a_file_we_already_wrote(work):
    """An older Jarvis wrote `mcp__*` into project settings; leaving it there
    stops every interactive session on a modal warning, so the merge drops
    it."""
    m = make_manager(work, Recorder())
    try:
        path = work.alpha / ".claude" / "settings.local.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"permissions": {
            "allow": ["mcp__*", "Bash(*)", "Keep(me)"], "deny": []}}))
        m.ensure_project_settings("alpha")
        allow = json.loads(path.read_text())["permissions"]["allow"]
        assert "mcp__*" not in allow
        assert "mcp__jarvis__*" in allow
        assert "Keep(me)" in allow            # the user's own rules survive
    finally:
        m.close()


def test_allow_rules_are_workspace_scoped():
    """Bare Edit/Write rules would allow ANY path (the CLI warns so) and the
    outside-dir approval would never fire."""
    for tool in ("Read", "Edit", "Write", "MultiEdit", "NotebookEdit"):
        assert tool not in cs.ALLOW_RULES
        assert f"{tool}(/**)" in cs.ALLOW_RULES
    assert "Bash(*)" in cs.ALLOW_RULES
    # `mcp__*` is INVALID in an allow rule (CLI 2.1.247) and stops the
    # interactive TUI on a Settings Warning nobody in a read-only pane can
    # answer; the valid form names the server.
    assert "mcp__*" not in cs.ALLOW_RULES
    assert "mcp__jarvis__*" in cs.ALLOW_RULES
    assert cs.STALE_ALLOW_RULES == ("mcp__*",)
    assert cs.DENY_RULES == ["Bash(sudo *)", "Bash(rm -rf /*)", "Bash(rm -rf ~*)"]


# ---------------------------------------------------------------- new_project
def test_new_project_scaffold(work, sink):
    rec = Recorder()
    m = make_manager(work, rec)
    try:
        line = m.new_project("Weather Bot")
        assert line == "The Weather Bot project is set up, sir; ready when you are."
        path = work.root / "weather-bot"
        assert path.is_dir()
        assert ["git", "-C", str(path), "init", "-q"] in rec.calls
        claude_md = (path / "CLAUDE.md").read_text()
        assert claude_md.startswith("# Weather Bot\n")
        assert "~/vss_env" in claude_md and "pytest -q" in claude_md and "Jarvis drives" in claude_md
        assert (path / "README.md").read_text().startswith("# Weather Bot")
        assert (path / ".claude" / "settings.local.json").exists()
        assert str(path) in work.cfg.allowed_dirs
        assert json.loads((work.tmp / "assistant.json").read_text())["claude"]["allowed_dirs"][-1] == str(path)
        assert m.active_project == "weather-bot"
        assert sink.of(ActiveProject)[-1].slug == "weather-bot"
        assert m.project_for("weather bot").path == str(path)
        # again: no clobbering, just activation
        (path / "CLAUDE.md").write_text("custom")
        assert m.new_project("weather bot") == "The weather bot project already exists, sir; it's active now."
        assert (path / "CLAUDE.md").read_text() == "custom"
    finally:
        m.close()


def test_new_project_refusals(work):
    m = make_manager(work, Recorder())
    try:
        assert m.new_project("") == "I'll need a name for it, sir."
        assert m.new_project("!!!") == "I'll need a name for it, sir."
        # slug rules keep everything under the root: no traversal survives
        assert m.new_project("../../etc") == "The ../../etc project is set up, sir; ready when you are."
        assert (work.root / "etc").is_dir() and not (work.tmp / "etc").exists()
        assert not (work.tmp.parent / "etc" / "CLAUDE.md").exists()
    finally:
        m.close()


# ---------------------------------------------------------------- discovery
def _write_session(dirpath, sid, cwd, turns, first_text="fix the parser", mtime=None,
                   title=""):
    dirpath.mkdir(parents=True, exist_ok=True)
    lines = []
    if title:
        lines.append({"type": "ai-title", "aiTitle": title, "sessionId": sid})
    lines.append({"type": "queue-operation", "operation": "enqueue", "sessionId": sid})
    for i in range(turns):
        if i % 2 == 0:
            text = first_text if i == 0 else f"and step {i}"
            lines.append({"type": "user", "cwd": cwd, "sessionId": sid,
                          "message": {"role": "user", "content": text}})
        else:
            lines.append({"type": "assistant", "cwd": cwd, "sessionId": sid,
                          "message": {"role": "assistant",
                                      "content": [{"type": "text", "text": f"done {i}"}]}})
    # noise that must be skipped: meta user lines and tool results
    lines.insert(1, {"type": "user", "cwd": cwd, "sessionId": sid,
                     "message": {"role": "user", "content": "<command-name>/clear</command-name>"}})
    lines.append({"type": "user", "cwd": cwd, "sessionId": sid,
                  "message": {"role": "user", "content": [
                      {"type": "tool_result", "tool_use_id": "x", "content": "ok"}]}})
    p = dirpath / f"{sid}.jsonl"
    p.write_text("\n".join(json.dumps(ln) for ln in lines) + "\n")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


@pytest.fixture
def sessions_dir(tmp_path):
    now = time.time()
    projects = tmp_path / "projects"
    vss = tmp_path / "vss-project"
    hay = tmp_path / "haymaker-digest"
    vss.mkdir()
    hay.mkdir()
    yesterday = now - 86400 - 3600
    _write_session(projects / "-x-vss-project", "sid-vss", str(vss), 6,
                   "improve the SAM3 labeler", mtime=yesterday, title="VSS labeler fixes")
    _write_session(projects / "-x-haymaker-digest", "sid-hay", str(hay), 4,
                   "tune the feed parser", mtime=now - 5 * 86400)
    _write_session(projects / "-x-haymaker-digest", "sid-jarvis-prompt", str(hay), 6,
                   "You are Jarvis, an AI voice assistant. Respond with structured commands",
                   mtime=now - 60)
    _write_session(projects / "-x-haymaker-digest", "sid-short", str(hay), 2, mtime=now - 30)
    _write_session(projects / "-x-gone", "sid-gone", str(tmp_path / "gone"), 6, mtime=now - 10)
    return SimpleNamespace(root=projects, vss=vss, hay=hay, now=now)


def test_discover_sessions_filters(sessions_dir):
    found = cs.discover_sessions(sessions_dir.root)
    ids = [s.session_id for s in found]
    assert ids == ["sid-vss", "sid-hay"]           # newest first; the rest skipped
    vss = found[0]
    assert vss.cwd == str(sessions_dir.vss)
    assert vss.first_user == "improve the SAM3 labeler"
    assert vss.last_assistant == "done 5"
    assert vss.turns == 6
    assert vss.title == "VSS labeler fixes"
    assert vss.slug == "vss-project"
    assert cs.discover_sessions(sessions_dir.root, limit=1)[0].session_id == "sid-vss"
    assert cs.discover_sessions(sessions_dir.root / "missing") == []


def test_pick_session_scoring_and_tie(sessions_dir):
    found = cs.discover_sessions(sessions_dir.root)
    now = sessions_dir.now
    best, runner, q = cs.pick_session("continue the vss project", found, now)
    assert best.session_id == "sid-vss" and q is None
    best, runner, q = cs.pick_session("what we were working on yesterday", found, now)
    assert best.session_id == "sid-vss" and q is None
    best, runner, q = cs.pick_session("pick up the haymaker digest", found, now)
    assert best.session_id == "sid-hay" and q is None
    # no cue at all: recency prior only; both are days old -> a tie -> one question
    for s in found:
        os.utime(s.path, (now - 10 * 86400, now - 10 * 86400))
    found = cs.discover_sessions(sessions_dir.root)
    best, runner, q = cs.pick_session("pick up where we left off", found, now)
    assert q is not None
    assert q.startswith("Two candidates, sir: ")
    assert q.endswith(" — which?")
    assert "haymaker digest" in q and "vss project" in q
    assert "VSS labeler fixes" in q                 # the title is the one-line summary
    assert cs.pick_session("x", [], now) == (None, None, None)


def test_resume_end_to_end(work, sessions_dir, sink):
    rec = Recorder(stream=transcript_events(), projects_dir=sessions_dir.root)
    acks = []

    def local_line(instruction, text, max_sentences=1, timeout=2.0, fallback=""):
        acks.append((instruction, text, fallback))
        return "Right away, sir — back to the VSS labeler."

    m = make_manager(work, rec, brain=SimpleNamespace(local_line=local_line))
    m.projects_dir = sessions_dir.root
    try:
        line = m.resume("pick up the vss project from yesterday")
        assert line == "Right away, sir — back to the VSS labeler."
        assert acks and "vss project" in acks[0][1] and acks[0][2].startswith("Right away, sir")
        join_runners(m)
        # the cwd was outside the allowed dirs but under ~ -> added, active, resumed
        proj = m.project_for("vss project")
        assert proj is not None and proj.path == str(sessions_dir.vss)
        assert str(sessions_dir.vss) in work.cfg.allowed_dirs
        assert m.active_project == proj.slug
        assert "--resume sid-vss" in rec.launches[0]
        assert sent_commands(rec) == [cs.RESUME_PROMPT]
        assert Path(m.tasks()[0].paths["prompt"]).read_text().strip() == cs.RESUME_PROMPT
        # the resumed session keeps its id: the transcript is the same file
        assert proj.session_id == "sid-vss"
    finally:
        m.close()


def test_resume_tie_then_answer(work, sessions_dir, sink):
    rec = Recorder(stream=transcript_events(), projects_dir=sessions_dir.root)
    m = make_manager(work, rec, brain=SimpleNamespace())
    m.projects_dir = sessions_dir.root
    now = sessions_dir.now
    for s in cs.discover_sessions(sessions_dir.root):
        os.utime(s.path, (now - 10 * 86400, now - 10 * 86400))
    try:
        q = m.resume("pick up where we left off")
        assert q.startswith("Two candidates, sir")
        assert m.pending_question()
        ack = m.resume("the haymaker one")
        assert ack == "Right away, sir — picking up haymaker digest where we left off."
        assert not m.pending_question()
        join_runners(m)
        assert "--resume sid-hay" in rec.launches[0]
    finally:
        m.close()


def test_resume_nothing_found(work):
    m = make_manager(work, Recorder())
    try:
        assert m.resume("pick up where we left off") == cs.NO_SESSION_LINE
    finally:
        m.close()


# ----------------------------------------------------------------- terminal
def test_open_terminal_attaches_read_only(work):
    """The pop-out is READ-ONLY (`tmux attach -r`): keystrokes in that
    window cannot drive the session — Jarvis stays the input path."""
    rec = Recorder()
    m = make_manager(work, rec)
    try:
        m.work_on("alpha")
        rec.calls.clear()
        assert m.open_terminal() is True
        assert rec.calls == [
            ["tmux", "has-session", "-t", "jarvis-alpha"],
            ["tmux", "new-session", "-d", "-s", "jarvis-alpha", "-c", str(work.alpha),
             "-x", "200", "-y", "50"],
            ["tmux", "list-clients", "-t", "jarvis-alpha", "-F", "#{client_name}"],
            ["tmux", "set-option", "-t", "jarvis-alpha", "window-size", "manual"],
            ["tmux", "resize-window", "-t", "jarvis-alpha", "-x", "200", "-y", "50"],
            ["xdotool", "search", "--name", "^Jarvis\\ ·\\ alpha$"],
            ["tmux", "set-option", "-t", "jarvis-alpha", "window-size", "latest"],
            ["gnome-terminal", "--title", "Jarvis · alpha", "--geometry=200x50",
             "--", "tmux", "attach", "-r", "-t", "jarvis-alpha"],
        ]
    finally:
        m.close()


def test_open_terminal_raises_existing_window(work):
    rec = Recorder(window_ids=["77594631"], tmux_exists=True)
    m = make_manager(work, rec)
    try:
        assert m.open_terminal("beta") is True
        assert rec.calls == [
            ["tmux", "has-session", "-t", "jarvis-beta-tools"],
            ["tmux", "list-clients", "-t", "jarvis-beta-tools", "-F", "#{client_name}"],
            ["tmux", "set-option", "-t", "jarvis-beta-tools", "window-size", "manual"],
            ["tmux", "resize-window", "-t", "jarvis-beta-tools", "-x", "200", "-y", "50"],
            ["xdotool", "search", "--name", "^Jarvis\\ ·\\ beta\\-tools$"],
            ["xdotool", "windowactivate", "77594631"],
        ]
        assert not any(c[0] == "gnome-terminal" for c in rec.calls)
    finally:
        m.close()


def test_open_terminal_without_projects(tmp_path):
    cfg = AssistantConfig({"claude": {"allowed_dirs": []}}, path=tmp_path / "a.json")
    m = cs.ClaudeSessionManager(cfg, SimpleNamespace(), None, tmp_path / "s.json",
                                tmp_path / "t", run=Recorder(), claude_bin="/x",
                                projects_dir=tmp_path / "none")
    try:
        assert m.open_terminal() is False
        assert m.submit("x") == cs.NO_PROJECT_LINE
    finally:
        m.close()


# ------------------------------------------- the live transcript (7.4b)
def test_encode_project_dir_matches_claude_code():
    """Claude Code's ~/.claude/projects/<dir> encoding, checked against the
    directory names on this machine."""
    assert cs.encode_project_dir("/home/hunterp/Jarvis") == "-home-hunterp-Jarvis"
    assert cs.encode_project_dir("/home/hunterp/vss_env") == "-home-hunterp-vss-env"
    assert cs.encode_project_dir(
        "/home/hunterp/.claude/projects/-home-hunterp/memory") == \
        "-home-hunterp--claude-projects--home-hunterp-memory"
    assert cs.encode_project_dir("") == ""


def test_find_transcript_prefers_the_named_session_then_mtime(tmp_path):
    root = tmp_path / "projects"
    d = root / cs.encode_project_dir("/w/proj")
    d.mkdir(parents=True)
    old, new = d / "aaa.jsonl", d / "bbb.jsonl"
    old.write_text("{}\n")
    new.write_text("{}\n")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    assert cs.find_transcript("/w/proj", root, "aaa") == old      # named wins
    assert cs.find_transcript("/w/proj", root) == new             # else newest
    assert cs.find_transcript("/w/proj", root, "gone") == new     # named absent
    assert cs.find_transcript("/w/proj", root, since=3000) is None
    assert cs.find_transcript("/w/other", root) is None           # no such dir


def test_sanitize_keys_never_sends_control_sequences():
    assert cs.sanitize_keys("fix the router") == "fix the router"
    # a multi-line prompt goes in as ONE line: Enter submits
    assert cs.sanitize_keys("do this\nthen that\r\nand this") == \
        "do this then that and this"
    assert cs.sanitize_keys("tab\there") == "tab here"
    assert cs.sanitize_keys("esc\x1b[2Jclear\x03") == "esc[2Jclear"
    assert cs.sanitize_keys("  spaced   out  ") == "spaced out"
    assert cs.sanitize_keys(None) == "" and cs.sanitize_keys("   ") == ""


def test_pane_state_reads_the_real_tui():
    assert cs.pane_state(Recorder.READY_PANE) == "ready"
    assert cs.pane_state(Recorder.TRUST_PANE) == "trust"
    assert cs.pane_state(Recorder.WORKING_PANE) == "ready"
    assert cs.pane_state("") == "starting"
    assert cs.pane_state("$ claude\n") == "starting"
    assert cs.pane_working(Recorder.WORKING_PANE) is True
    assert cs.pane_working(Recorder.READY_PANE) is False


def test_entry_turn_text_only_fires_on_the_message_that_ends_the_turn():
    def entry(stop, text="done", side=False):
        return {"type": "assistant", "isSidechain": side,
                "message": {"stop_reason": stop,
                            "content": [{"type": "text", "text": text}]}}

    assert cs.entry_turn_text(entry("end_turn")) == "done"
    assert cs.entry_turn_text(entry("stop_sequence")) == "done"
    assert cs.entry_turn_text(entry("tool_use")) is None
    assert cs.entry_turn_text(entry("end_turn", side=True)) is None   # a subagent
    assert cs.entry_turn_text({"type": "user", "message": {}}) is None
    assert cs.entry_turn_text({"type": "attachment"}) is None
    assert cs.entry_turn_text(None) is None
    # thinking blocks are not the spoken answer
    thinking = {"type": "assistant", "message": {
        "stop_reason": "end_turn",
        "content": [{"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "Two sentences. Right here."}]}}
    assert cs.entry_turn_text(thinking) == "Two sentences. Right here."


def test_parse_transcript_entry_derives_the_same_milestones():
    """Transcript records carry the print-mode `message` shape, so the
    milestone rules are the same code — including the failing-test verdict,
    which the app speaks."""
    task = cs.Task("t1", "w7-smoke", "p", "opus",
                   cwd="/home/hunterp/projects/w7-smoke")
    lines, now = [], 1000.0
    for entry in transcript_events():
        now += 1.0
        lines += cs.parse_transcript_entry(entry, task, now=now)
    texts = [(p.milestone, p.line) for p in lines]
    assert (False, "Read jarvis/router.py") in texts
    assert (True, "Editing router.py, sir.") in texts
    assert (True, "Tests passed, sir.") in texts
    assert (False, "Error: cat: /nonexistent: No such file or directory") in texts
    # Claude Code's own bookkeeping records and subagent turns are silent
    task2 = cs.Task("t2", "p", "x", "opus")
    for entry in ({"type": "attachment", "content": "x"},
                  {"type": "last-prompt", "prompt": "x"},
                  {"type": "queue-operation"},
                  {"type": "assistant", "isSidechain": True, "message": {
                      "content": [{"type": "tool_use", "id": "s", "name": "Edit",
                                   "input": {"file_path": "/w/sub.py"}}]}},
                  None, 7, "text"):
        assert cs.parse_transcript_entry(entry, task2, now=1.0) == []
    assert task2.files_touched == set()


def test_a_failing_suite_is_still_announced_from_the_transcript(work, sink):
    """The 2026-08-27 fix must survive the move off stdout: a red suite is
    spoken, not swallowed."""
    entries = [
        {"type": "assistant", "isSidechain": False, "message": {
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "b1", "name": "Bash",
                         "input": {"command": "python -m pytest -q"}}]}},
        {"type": "user", "isSidechain": False, "message": {"content": [
            {"type": "tool_result", "tool_use_id": "b1", "is_error": True,
             "content": "3 passed, 2 failed in 1.20s"}]}},
        {"type": "assistant", "isSidechain": False, "message": {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Two tests fail. I stopped there."}]}},
    ]
    rec = Recorder(stream=entries)
    m = make_manager(work, rec)
    try:
        task = m.submit("run the tests", project="alpha")
        join_runners(m)
        spoken = [p.line for p in sink.of(ClaudeProgress) if p.milestone]
        assert "Running the tests, sir." in spoken
        assert "2 tests failed, sir." in spoken
        assert task.state == "done"
        assert task.result_text == "Two tests fail. I stopped there."
    finally:
        m.close()


def test_a_resumed_session_replays_nothing(work, sink):
    """The tail starts at the END of an existing transcript, so yesterday's
    edits are never spoken as tonight's."""
    rec = Recorder(stream=transcript_events())
    m = make_manager(work, rec)
    try:
        m.submit("first", project="alpha")
        join_runners(m)
        first = [p.line for p in sink.of(ClaudeProgress)]
        assert "I'll start by reading the router." in first
        # the recorder APPENDS the same turn again, as a resumed session's
        # transcript really does; the tail must start where task one stopped
        rec.panes["alpha"]["alive"] = False       # the pane is restarted
        sink.events.clear()
        m.submit("second", project="alpha")
        join_runners(m)
        second = [p.line for p in sink.of(ClaudeProgress)]
        assert second == first                    # one turn's worth, not two
        assert rec.transcripts["alpha"].read_text().count(
            "I\'ll start by reading the router.") == 2
    finally:
        m.close()


def test_the_workspace_trust_dialog_is_answered_only_for_a_cleared_folder(work, sink):
    """A read-only pane could never answer it — and Jarvis only answers it
    for a folder the user has already cleared for Claude."""
    rec = Recorder(stream=transcript_events(), trust=True)
    m = make_manager(work, rec)
    try:
        task = m.submit("go", project="alpha")
        join_runners(m)
        assert task.state == "done"
        assert "alpha" in rec.trusted
        # a bare Enter, with no -l before it: the dialog's default is
        # "1. Yes, I trust this folder"
        assert ["tmux", "send-keys", "-t", "jarvis-alpha", "Enter"] in rec.calls
    finally:
        m.close()

    work.cfg.set("claude.auto_approve_anywhere", False)
    stray = work.tmp / "stray"
    stray.mkdir()
    rec2 = Recorder(stream=transcript_events(), trust=True)
    m2 = make_manager(work, rec2)
    try:
        m2._projects["stray"] = cs.Project(slug="stray", path=str(stray))
        m2.cfg.set("claude.permission_prompt_tool", True)
        task = m2.submit("go", project="stray")
        join_runners(m2)
        assert task.state == "failed"
        assert "stray" not in rec2.trusted
        assert rec2.typed == {}          # nothing was typed at that pane
        # Esc dismisses the dialog so the refused Claude does not sit there
        assert ["tmux", "send-keys", "-t", "jarvis-stray", "Escape"] in rec2.calls
    finally:
        m2.close()


def test_a_prompt_typed_at_a_warm_session_is_confirmed_before_enter(work):
    """The second task types into a live TUI — and Enter is pressed only
    once the text is visibly in the input box, because a busy TUI drops
    keys and a dropped prompt is a task that waits for a turn nobody
    asked for."""
    rec = Recorder(stream=transcript_events())
    m = make_manager(work, rec)
    try:
        m.submit("first", project="alpha")
        join_runners(m)
        m.submit("second one, sir", project="alpha")
        join_runners(m)
        assert len(rec.launches) == 1
        assert rec.prompts == [("alpha", "first"), ("alpha", "second one, sir")]
        typed = [c for c in rec.calls
                 if c[:2] == ["tmux", "send-keys"] and c[4:5] == ["-l"]
                 and c[5] == "second one, sir"]
        assert len(typed) == 1
        assert typed[0] == ["tmux", "send-keys", "-t", "jarvis-alpha", "-l",
                            "second one, sir"]
    finally:
        m.close()


def test_dropped_keys_are_retyped_not_silently_lost(work):
    rec = Recorder(stream=transcript_events(), drop_keys=2)
    m = make_manager(work, rec)
    try:
        m.submit("first", project="alpha")
        join_runners(m)
        task = m.submit("do the thing", project="alpha")
        join_runners(m)
        typed = [c for c in rec.calls
                 if c[:2] == ["tmux", "send-keys"] and c[4:5] == ["-l"]
                 and c[5] == "do the thing"]
        assert len(typed) == 3                     # two swallowed, one landed
        # the input box is cleared before each retype, so no half prompt
        assert ["tmux", "send-keys", "-t", "jarvis-alpha", "C-u"] in rec.calls
        assert rec.prompts[-1] == ("alpha", "do the thing")
        assert task.state == "done"
    finally:
        m.close()


def test_a_prompt_that_never_lands_fails_rather_than_hanging(work, sink):
    rec = Recorder(stream=transcript_events(), drop_keys=99)
    m = make_manager(work, rec)
    try:
        m.submit("first", project="alpha")
        join_runners(m)
        task = m.submit("do the thing", project="alpha")
        join_runners(m)
        assert task.state == "failed"
        assert sink.of(ClaudeTaskState)[-1].text == "tmux wouldn't take the prompt, sir."
    finally:
        m.close()


def test_a_multi_line_prompt_goes_in_as_one_line(work):
    rec = Recorder(stream=transcript_events())
    m = make_manager(work, rec)
    try:
        m.submit("do this\nthen that\ttoo", project="alpha")
        join_runners(m)
        # in the argv of the launch line, on ONE line: Enter submits
        assert rec.prompts == [("alpha", "do this then that too")]
        assert "\n" not in rec.launches[0] and "\t" not in rec.launches[0]
    finally:
        m.close()


# ----------------------------------------------------- terminal_open (7.7)
def test_terminal_open_is_true_only_while_a_client_is_attached(work):
    rec = Recorder()
    m = make_manager(work, rec, clients_ttl_s=0.0)
    try:
        m.work_on("alpha")
        assert m.terminal_open() is False
        assert m.terminal_open("alpha") is False
        rec.clients.add("alpha")
        assert m.terminal_open() is True
        assert m.terminal_open("alpha") is True
        assert m.terminal_open("beta") is False        # a different session
        assert m.open_projects() == ["alpha"]
        assert ["tmux", "list-clients", "-t", "jarvis-alpha", "-F",
                "#{client_name}"] in rec.calls
        rec.clients.clear()
        assert m.terminal_open() is False
        assert m.open_projects() == []
        # an unknown project never claims a terminal
        assert m.terminal_open("gamma") is False
    finally:
        m.close()


def test_terminal_open_is_cached_so_the_router_can_ask_freely(work):
    rec = Recorder(clients=["alpha"])
    m = make_manager(work, rec, clients_ttl_s=60.0)
    try:
        m.work_on("alpha")
        assert m.terminal_open() is True
        before = sum(1 for c in rec.calls if c[:2] == ["tmux", "list-clients"])
        for _ in range(20):
            assert m.terminal_open() is True
        after = sum(1 for c in rec.calls if c[:2] == ["tmux", "list-clients"])
        assert after == before                       # not one extra subprocess
    finally:
        m.close()


def test_an_attached_terminal_keeps_its_own_geometry(work):
    """While somebody is watching, the client drives the window size; only
    a detached session is pinned to 200x50 (a detached tmux would otherwise
    stay 80x24 and wrap the TUI)."""
    rec = Recorder(clients=["alpha"], tmux_exists=True)
    m = make_manager(work, rec, clients_ttl_s=0.0)
    try:
        m.work_on("alpha")
        rec.calls.clear()
        assert m._ensure_tmux(m.project_for("alpha")) is True
        assert not any(c[:2] == ["tmux", "resize-window"] for c in rec.calls)
        rec.clients.clear()
        rec.calls.clear()
        assert m._ensure_tmux(m.project_for("alpha")) is True
        assert ["tmux", "resize-window", "-t", "jarvis-alpha", "-x", "200",
                "-y", "50"] in rec.calls
    finally:
        m.close()


# ------------------------------------------------- unsafe dirs (C2, 7.3)
def test_unsafe_dir_reason_names_the_dirs_claude_never_gets(tmp_path):
    """$HOME's own .claude/settings.local.json is Claude Code's USER-LEVEL
    allowlist, so home / config / system dirs are never projects."""
    home = tmp_path / "home"
    (home / ".config" / "jarvis").mkdir(parents=True)
    (home / "repo").mkdir()
    reason = lambda p: cs.unsafe_dir_reason(str(p), str(home))    # noqa: E731
    assert reason(home) == "your home folder itself"
    assert reason(home / ".claude") == "a configuration folder"
    assert reason(home / ".config" / "jarvis") == "a configuration folder"
    assert reason(tmp_path) == "a folder your home folder lives in"
    assert reason("/") == "a folder your home folder lives in"
    assert reason("/etc") == "a system folder"
    assert reason("/usr/share") == "a system folder"
    assert reason("/tmp")          # the fake home lives under it
    assert cs.unsafe_dir_reason("/tmp", "/home/nobody") == "a scratch folder"
    assert reason("") == "an empty path"
    # a real project dir, anywhere on disk, is fine (auto_approve_anywhere)
    assert reason(home / "repo") == ""
    assert reason(tmp_path / "elsewhere" / "some-repo") == ""   # not created yet


def test_ensure_project_settings_refuses_config_dirs(work):
    """The user's hand-written ~/.claude/settings.local.json is never
    rewritten, even when a Project points straight at it."""
    m = make_manager(work, Recorder())
    try:
        cfg_dir = work.tmp / ".claude"
        cfg_dir.mkdir()
        settings = cfg_dir / "settings.local.json"
        settings.write_text('{"permissions": {"allow": ["Bash(ls)"]}}\n')
        before = settings.read_bytes()
        out = m.ensure_project_settings(cs.Project(slug="dot", path=str(cfg_dir)))
        assert out == cfg_dir / ".claude" / "settings.local.json"
        assert not out.exists()
        assert settings.read_bytes() == before
        # and the same dir cannot be worked in at all
        assert m.submit("do it", project=cs.Project(slug="dot", path=str(cfg_dir))) \
            == cs.UNSAFE_DIR_LINE
    finally:
        m.close()
