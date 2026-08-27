"""Tests for the permission path: jarvis/mcp_permissions.py + jarvis/approvals.py.

Spec: docs/specs/2026-08-26-jarvis-personal-assistant.md, section 7.3.

Three layers, all offline (UNIX sockets under tmp only; tests/conftest.py
firewalls /tmp/vss_voice):

1. `PermissionServer` — the dependency-free JSON-RPC stdio server, driven
   through pipes with a stubbed `ask` and, once, as a real subprocess.
2. `ApprovalBroker` — the socket round trip: auto-policy, the spoken
   question, allow / deny / timeout, and the bus events.
3. Both together — the CLI's MCP subprocess talking to a live broker.

NOTE (7.3): the CLI half — the `--permission-prompt-tool` / `--mcp-config`
flags — is still not exercised here, because it costs real Claude credits.
It was verified by hand against CLI 2.1.247 on 2026-08-27 (allow, deny and
no-answer paths all live); the findings are recorded at the top of
jarvis/claude_session.py and the gate is now open by default. What these
tests pin down is that the Jarvis side is complete and correct.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import jarvis.approvals as ap
import jarvis.mcp_permissions as mp
from jarvis.assistant_config import AssistantConfig
from jarvis.events import ApprovalRequested, ApprovalResolved, bus

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


# ------------------------------------------------------------------ helpers
class Sink:
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


@pytest.fixture
def sink():
    s = Sink(ApprovalRequested, ApprovalResolved)
    yield s
    s.close()


@pytest.fixture
def work(tmp_path):
    proj = tmp_path / "alpha"
    (proj / "jarvis").mkdir(parents=True)
    cfg = AssistantConfig({"claude": {"allowed_dirs": [str(proj)]}},
                          path=tmp_path / "assistant.json")
    return SimpleNamespace(tmp=tmp_path, proj=proj, cfg=cfg,
                           sock=tmp_path / "approvals.sock")


@pytest.fixture
def broker(work):
    b = ap.ApprovalBroker(work.sock, work.cfg, timeout_s=5.0)
    assert b.start() is True
    yield b
    b.stop()


def wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return bool(pred())


def rpc(server, msgs):
    """Drive PermissionServer.serve over pipes; returns the parsed replies."""
    import io
    stdin = io.BytesIO(b"".join(json.dumps(m).encode() + b"\n" for m in msgs))
    stdout = io.BytesIO()
    server.serve(stdin, stdout)
    return [json.loads(ln) for ln in stdout.getvalue().splitlines() if ln.strip()]


# ============================================================ 1. the server
def test_initialize_tools_list_and_ping():
    server = mp.PermissionServer(ask=lambda *a, **k: {"behavior": "deny"})
    out = rpc(server, [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18",
                    "clientInfo": {"name": "claude-code", "version": "2.1.246"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
    ])
    # the notification produces no reply
    assert [r.get("id") for r in out] == [1, 2, 3]
    init = out[0]["result"]
    assert init["protocolVersion"] == "2025-06-18"        # echoed
    assert init["capabilities"] == {"tools": {}}
    assert init["serverInfo"]["name"] == "jarvis-permissions"
    assert out[1]["result"] == {}
    tools = out[2]["result"]["tools"]
    assert [t["name"] for t in tools] == ["approve"]
    schema = tools[0]["inputSchema"]
    assert schema["properties"]["tool_name"]["type"] == "string"
    assert schema["properties"]["input"]["type"] == "object"
    assert "tool_use_id" in schema["properties"]
    assert schema["required"] == ["tool_name", "input"]
    assert all(r["jsonrpc"] == "2.0" for r in out)


def test_tools_call_allow_and_deny_shapes():
    answers = {"Read": {"behavior": "allow"},
               "Bash": {"behavior": "deny", "message": "Declined, sir."}}
    seen = []

    def ask(sock, tool_name, tool_input, project="", cwd="", timeout=0, rid=""):
        seen.append((sock, tool_name, tool_input, project, cwd, rid))
        return answers[tool_name]

    server = mp.PermissionServer(sock_path="/s.sock", project="alpha",
                                 cwd="/p", ask=ask)
    out = rpc(server, [
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
         "params": {"name": "approve",
                    "arguments": {"tool_name": "Read",
                                  "input": {"file_path": "/etc/hosts"},
                                  "tool_use_id": "toolu_9"}}},
        {"jsonrpc": "2.0", "id": 8, "method": "tools/call",
         "params": {"name": "approve",
                    "arguments": {"tool_name": "Bash",
                                  "input": {"command": "git push"}}}},
    ])
    allow = json.loads(out[0]["result"]["content"][0]["text"])
    assert allow == {"behavior": "allow", "updatedInput": {"file_path": "/etc/hosts"}}
    assert out[0]["result"]["content"][0]["type"] == "text"
    assert out[0]["result"]["isError"] is False
    deny = json.loads(out[1]["result"]["content"][0]["text"])
    assert deny == {"behavior": "deny", "message": "Declined, sir."}
    assert seen[0][:2] == ("/s.sock", "Read")
    assert seen[0][3:] == ("alpha", "/p", "toolu_9")


def test_server_is_tolerant_and_never_leaves_stdout():
    calls = []

    def ask(*a, **k):
        calls.append(a)
        raise RuntimeError("socket exploded")

    server = mp.PermissionServer(ask=ask)
    out = rpc(server, [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "nope", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 2, "method": "resources/list"},
        {"jsonrpc": "2.0", "method": "notifications/cancelled"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "approve", "arguments": {"tool_name": "Bash"}}},
    ])
    assert out[0]["error"]["code"] == -32602            # unknown tool
    assert out[1]["error"]["code"] == -32601            # unknown method
    assert [r.get("id") for r in out] == [1, 2, 3]      # notification silent
    # a crashing ask still returns a well-formed DENY, never an exception
    body = json.loads(out[2]["result"]["content"][0]["text"])
    assert body["behavior"] == "deny" and "socket exploded" in body["message"]
    assert calls


def test_server_handles_bad_json_lines():
    import io
    server = mp.PermissionServer(ask=lambda *a, **k: {"behavior": "deny"})
    stdout = io.BytesIO()
    server.serve(io.BytesIO(b"not json\n\n[1,2]\n"), stdout)
    out = [json.loads(ln) for ln in stdout.getvalue().splitlines() if ln.strip()]
    assert out[0]["error"]["code"] == -32700            # parse error
    assert out                                           # and it kept serving


def test_missing_socket_denies_with_not_running(tmp_path):
    reply = mp.ask_jarvis(str(tmp_path / "nope.sock"), "Bash", {"command": "ls"})
    assert reply["behavior"] == "deny"
    assert reply["message"] == mp.NOT_RUNNING
    assert mp.ask_jarvis("", "Bash", {})["behavior"] == "deny"
    result = mp.permission_result(reply, {"command": "ls"})
    assert result == {"behavior": "deny", "message": mp.NOT_RUNNING}


def test_permission_result_shape():
    assert mp.permission_result({"behavior": "allow"}, {"a": 1}) == \
        {"behavior": "allow", "updatedInput": {"a": 1}}
    assert mp.permission_result({"behavior": "ALLOW"}, {})["behavior"] == "allow"
    assert mp.permission_result({}, {})["message"] == "Denied by Jarvis"


def test_module_is_dependency_free():
    """It runs as a CLI subprocess with the project as cwd — stdlib only."""
    src = (REPO_ROOT / "jarvis" / "mcp_permissions.py").read_text()
    assert "import jarvis" not in src and "from jarvis" not in src


# =========================================================== 2. the broker
def test_describe_request_persona_lines(work):
    def q(tool, inp, cwd=""):
        req = ap.ApprovalRequest(request_id="r", tool_name=tool, input=inp, cwd=cwd)
        req.detail = ap.request_detail(tool, inp)
        return ap.describe_request(req, work.cfg)

    assert q("Bash", {"command": "git push origin main"}) == \
        "Claude wants to run “git push origin main”, sir; shall I allow it?"
    outside = q("Edit", {"file_path": str(Path.home() / ".bashrc")})
    assert outside.startswith("Claude wants to edit ~/.bashrc, sir")
    assert "outside the project" in outside and outside.endswith("allow it?")
    assert q("WebFetch", {"url": "https://example.com/x"}) == \
        "Claude wants to fetch example.com, sir; allow it?"
    inside = q("Edit", {"file_path": str(work.proj / "jarvis" / "router.py")})
    assert "outside the project" not in inside
    # every question is ONE sentence and keeps the persona
    for line in (q("Bash", {"command": "rm -rf build"}),
                 q("Task", {"description": "audit the config"}),
                 q("mcp__spotify__play", {"uri": "x"}),
                 q("Frobnicate", {"thing": "x"})):
        assert line.count(".") == 0 and line.endswith("?")
        assert ", sir" in line


def test_paths_in_input_finds_bash_and_key_paths():
    assert ap.paths_in_input("Edit", {"file_path": "/a/b.py"}) == ["/a/b.py"]
    found = ap.paths_in_input("Bash", {"command": "cp ~/.bashrc /tmp/x && ls -l"})
    assert "~/.bashrc" in found and "/tmp/x" in found
    assert ap.paths_in_input("Bash", {"command": "pytest -q"}) == []
    assert ap.paths_in_input("Read", {}) == []


def test_auto_policy_allows_only_inside_allowed_dirs(work, broker):
    def req(tool, inp, cwd=""):
        return ap.ApprovalRequest(request_id="r", tool_name=tool, input=inp, cwd=cwd)

    inside = str(work.proj / "jarvis" / "router.py")
    assert broker.auto_policy(req("Read", {"file_path": inside})) == "allow"
    assert broker.auto_policy(req("Write", {"file_path": str(work.proj / "new.py")})) == "allow"
    assert broker.auto_policy(req("Edit", {"file_path": "/etc/hosts"})) is None
    # relative paths resolve against the task cwd
    assert broker.auto_policy(req("Read", {"file_path": "jarvis/router.py"},
                                  cwd=str(work.proj))) == "allow"
    # Bash is never auto-allowed; its paths are free text
    assert broker.auto_policy(req("Bash", {"command": f"cat {inside}"})) is None
    # a tool with no path at all is always asked about
    assert broker.auto_policy(req("Grep", {"pattern": "def route"})) is None


def test_socket_round_trip_auto_allow(work, broker, sink):
    inside = str(work.proj / "jarvis" / "router.py")
    reply = ap.ask(work.sock, "Read", {"file_path": inside}, "alpha",
                   str(work.proj), request_id="r1")
    assert reply == {"id": "r1", "behavior": "allow", "message": ""}
    assert sink.of(ApprovalRequested) == []              # nobody was bothered
    resolved = sink.of(ApprovalResolved)[-1]
    assert (resolved.request_id, resolved.allowed, resolved.source) == ("r1", True, "policy")
    assert broker.pending() == []


@pytest.mark.parametrize("allowed,source", [(True, "typed"), (False, "voice")])
def test_socket_round_trip_asks_and_answers(work, broker, sink, allowed, source):
    asked = []
    broker.on_request = asked.append
    replies = []

    t = threading.Thread(target=lambda: replies.append(
        ap.ask(work.sock, "Bash", {"command": "git push origin main"},
               "alpha", str(work.proj), request_id="r2")), daemon=True)
    t.start()
    assert wait_for(lambda: bool(broker.pending()))
    req = broker.pending()[0]
    assert req.request_id == "r2" and req.project == "alpha"
    assert req.detail == "git push origin main"
    assert req.question == \
        "Claude wants to run “git push origin main”, sir; shall I allow it?"
    assert asked and asked[0] is req
    ev = sink.of(ApprovalRequested)[-1]
    assert (ev.request_id, ev.tool_name, ev.project) == ("r2", "Bash", "alpha")
    assert ev.question == req.question

    assert broker.answer(allowed, source=source) is True
    t.join(5)
    assert replies and replies[0]["behavior"] == ("allow" if allowed else "deny")
    assert replies[0]["id"] == "r2"
    if not allowed:
        assert source in replies[0]["message"]
    done = sink.of(ApprovalResolved)[-1]
    assert (done.request_id, done.allowed, done.source) == ("r2", allowed, source)
    assert broker.pending() == []
    assert broker.answer(True) is False                  # nothing left to answer
    assert broker.history[-1].source == source


def test_answer_targets_the_named_request_else_the_oldest(work, broker):
    broker.timeout_s = 5.0
    out = {}

    def fire(rid, cmd):
        out[rid] = ap.ask(work.sock, "Bash", {"command": cmd}, "alpha",
                          str(work.proj), request_id=rid)

    threads = [threading.Thread(target=fire, args=("a", "one"), daemon=True),
               threading.Thread(target=fire, args=("b", "two"), daemon=True)]
    threads[0].start()
    assert wait_for(lambda: len(broker.pending()) == 1)
    threads[1].start()
    assert wait_for(lambda: len(broker.pending()) == 2)
    assert [r.request_id for r in broker.pending()] == ["a", "b"]   # oldest first
    assert broker.answer(True, request_id="b") is True
    threads[1].join(5)
    assert out["b"]["behavior"] == "allow"
    assert [r.request_id for r in broker.pending()] == ["a"]
    assert broker.answer(False) is True                  # the oldest
    threads[0].join(5)
    assert out["a"]["behavior"] == "deny"


def test_timeout_denies_with_the_persona_line(work, sink):
    broker = ap.ApprovalBroker(work.sock, work.cfg, timeout_s=0.2)
    assert broker.start() is True
    try:
        reply = ap.ask(work.sock, "Bash", {"command": "curl evil.example"},
                       "alpha", str(work.proj), request_id="r3", timeout=10)
        assert reply["behavior"] == "deny"
        assert reply["message"] == ap.TIMEOUT_LINE
        assert ap.TIMEOUT_LINE == "No answer in two minutes, sir; I've declined it."
        resolved = sink.of(ApprovalResolved)[-1]
        assert (resolved.allowed, resolved.source) == (False, "timeout")
        assert broker.pending() == []
    finally:
        broker.stop()


def test_malformed_request_is_denied_not_fatal(work, broker):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        s.connect(str(work.sock))
        s.sendall(b"{not json\n")
        line = s.recv(4096).decode().strip()
    assert json.loads(line)["behavior"] == "deny"
    # the broker is still serving afterwards
    inside = str(work.proj / "jarvis" / "router.py")
    assert ap.ask(work.sock, "Read", {"file_path": inside},
                  cwd=str(work.proj))["behavior"] == "allow"


def test_stop_releases_waiting_clients(work, sink):
    broker = ap.ApprovalBroker(work.sock, work.cfg, timeout_s=30)
    broker.start()
    got = []
    t = threading.Thread(target=lambda: got.append(
        ap.ask(work.sock, "Bash", {"command": "sleep 1"}, request_id="r4",
               timeout=10)), daemon=True)
    t.start()
    assert wait_for(lambda: bool(broker.pending()))
    broker.stop()
    t.join(5)
    assert got and got[0]["behavior"] == "deny"
    assert not work.sock.exists()
    assert broker.running is False


def test_start_reports_failure_on_a_bad_path(tmp_path, work):
    # AF_UNIX paths are capped at ~108 bytes: bind fails, start() says so
    bad = ap.ApprovalBroker(tmp_path / ("x" * 120 + ".sock"), work.cfg)
    assert bad.start() is False
    assert bad.running is False
    assert bad.pending() == []


# ================================================= 3. both halves together
def test_mcp_subprocess_talks_to_a_live_broker(work, broker, sink):
    """The real CLI shape: the MCP server is a subprocess reading JSON-RPC on
    stdin, and it reaches this process's broker over the socket."""
    answered = threading.Event()

    def on_request(req):
        # stand in for the app: Hunter says yes
        threading.Timer(0.05, lambda: (broker.answer(True, req.request_id,
                                                     source="voice"),
                                       answered.set())).start()

    broker.on_request = on_request
    env = dict(os.environ, JARVIS_APPROVAL_SOCK=str(work.sock),
               JARVIS_PROJECT="alpha", PYTHONPATH=str(REPO_ROOT))
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "claude-code", "version": "2.1.246"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "approve",
                    "arguments": {"tool_name": "Bash",
                                  "input": {"command": "git push origin main"},
                                  "tool_use_id": "toolu_1"}}},
    ]
    proc = subprocess.run(
        [PY, "-m", "jarvis.mcp_permissions"],
        input=b"".join(json.dumps(m).encode() + b"\n" for m in msgs),
        capture_output=True, cwd=str(work.proj), env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode()
    out = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    assert [r["id"] for r in out] == [1, 2, 3]           # stdout is protocol only
    assert out[1]["result"]["tools"][0]["name"] == "approve"
    body = json.loads(out[2]["result"]["content"][0]["text"])
    assert body == {"behavior": "allow",
                    "updatedInput": {"command": "git push origin main"}}
    assert answered.wait(5)
    asked = sink.of(ApprovalRequested)[-1]
    assert asked.tool_name == "Bash" and asked.project == "alpha"
    assert asked.question == \
        "Claude wants to run “git push origin main”, sir; shall I allow it?"
    assert sink.of(ApprovalResolved)[-1].source == "voice"
    assert b"jarvis-permissions" in proc.stderr          # diagnostics on stderr


def test_mcp_subprocess_denies_when_jarvis_is_down(work):
    env = dict(os.environ, JARVIS_APPROVAL_SOCK=str(work.tmp / "dead.sock"),
               JARVIS_PROJECT="alpha", PYTHONPATH=str(REPO_ROOT))
    msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": "approve",
                      "arguments": {"tool_name": "Write",
                                    "input": {"file_path": "/etc/passwd"}}}}
    proc = subprocess.run([PY, "-m", "jarvis.mcp_permissions"],
                          input=json.dumps(msg).encode() + b"\n",
                          capture_output=True, cwd=str(work.proj), env=env,
                          timeout=60)
    assert proc.returncode == 0
    body = json.loads(json.loads(proc.stdout.splitlines()[0])
                      ["result"]["content"][0]["text"])
    assert body == {"behavior": "deny", "message": mp.NOT_RUNNING}
