"""Command socket (jarvis/cmdsock.py) + the `python -m jarvis.ask` client.

Real UNIX sockets under tmp, a fake app that answers the way JarvisApp does
(sync JarvisReply from _emit_result; async answers from a worker thread via
BrainState + JarvisReply), and one round trip through the real JarvisApp
built by tests/test_app_wiring's fixtures."""
from __future__ import annotations

import os
import stat
import threading
import time
from types import SimpleNamespace

import pytest

import jarvis.ask as ask_mod
import jarvis.cmdsock as cs
from jarvis.commander import CommandResult
from jarvis.events import BrainState, JarvisReply, Status, UserUtterance, bus
from tests.test_app_wiring import build, paths, seams  # noqa: F401 - fixtures


class FakeApp:
    """dispatch_text publishes like app._emit_result; `plan` says what the
    async side does afterwards."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.result = CommandResult(handled=True, reply="It is ten, sir.", speak=True,
                                    status="Clock")
        self.plan = None            # "brain" | "ack+brain" | "claude" | "hang"
        self.answer = "Two items on Tuesday, sir."

    def diagnostics_text(self):
        return "All systems nominal, sir."

    def dispatch_text(self, text, source="typed", quiet=False, turn_id=""):
        self.calls.append((text, source, quiet))
        self.turn_id = turn_id            # the real app stamps replies with it
        res = self.result
        if res.reply:
            bus.publish(JarvisReply(text=res.reply, speak=res.speak))
        if res.status:
            bus.publish(Status(text=res.status, kind="info"))
        if self.plan in ("brain", "ack+brain"):
            def _later():
                time.sleep(0.15)
                bus.publish(BrainState(state="thinking"))
                time.sleep(0.15)
                bus.publish(JarvisReply(text=self.answer, speak=False))
                bus.publish(BrainState(state="idle"))
            threading.Thread(target=_later, daemon=True).start()
        elif self.plan == "hang":
            def _hang():
                time.sleep(0.1)
                bus.publish(BrainState(state="thinking"))
            threading.Thread(target=_hang, daemon=True).start()
        return res


@pytest.fixture
def server(tmp_path):
    app = FakeApp()
    srv = cs.CommandSocket(tmp_path / "command.sock", app, timeout_s=5.0, idle_grace_s=0.4)
    assert srv.start()
    yield SimpleNamespace(app=app, sock=srv.sock_path, srv=srv)
    srv.stop()


def _run(sock, text, **kw):
    return list(cs.ask(sock, text, **kw))


# ------------------------------------------------------------- basics
def test_the_socket_is_private_and_goes_away_on_stop(server):
    mode = stat.S_IMODE(os.stat(server.sock).st_mode)
    assert mode == 0o600
    server.srv.stop()
    assert not server.sock.exists()
    with pytest.raises(ConnectionError) as exc:
        _run(server.sock, "hello")
    assert "not running" in str(exc.value)


def test_status_answers_with_diagnostics_and_no_dispatch(server):
    msgs = _run(server.sock, "status")
    assert msgs == [{"kind": "reply", "text": "All systems nominal, sir.", "speak": False},
                    {"kind": "end", "reason": "done"}]
    assert server.app.calls == []


def test_a_tier_one_answer_streams_reply_status_end(server):
    seen = []
    bus.subscribe(UserUtterance, seen.append)
    try:
        msgs = _run(server.sock, "what time is it")
    finally:
        bus.unsubscribe(UserUtterance, seen.append)
    assert server.app.calls == [("what time is it", "cli", False)]
    kinds = [m["kind"] for m in msgs]
    assert kinds == ["reply", "status", "end"]
    assert msgs[0] == {"kind": "reply", "text": "It is ten, sir.", "speak": True}
    assert msgs[1] == {"kind": "status", "text": "Clock", "level": "info"}
    assert msgs[-1]["reason"] == "done"
    # the transcript sees the question like a typed or Discord one
    assert [(e.text, e.source) for e in seen] == [("what time is it", "cli")]


def test_quiet_is_plumbed_through(server):
    _run(server.sock, "what time is it", quiet=True)
    assert server.app.calls[-1] == ("what time is it", "cli", True)


def test_a_brain_answer_closes_on_idle_after_thinking(server):
    server.app.result = CommandResult(handled=True, status="Thinking…", done=False)
    server.app.plan = "brain"
    t0 = time.monotonic()
    msgs = _run(server.sock, "what's on my calendar")
    assert time.monotonic() - t0 < 3.0
    replies = [m["text"] for m in msgs if m["kind"] == "reply"]
    assert replies == ["Two items on Tuesday, sir."]
    assert msgs[-1] == {"kind": "end", "reason": "answered"}
    assert "state" not in {m["kind"] for m in msgs}     # BrainState stays internal


def test_an_ack_then_the_answer_arrive_in_order(server):
    server.app.result = CommandResult(handled=True, reply="Looking that up, sir.",
                                      speak=True, ack=True, done=False, status="Looking it up…")
    server.app.plan = "ack+brain"
    msgs = _run(server.sock, "look up the f1 result")
    replies = [m["text"] for m in msgs if m["kind"] == "reply"]
    assert replies == ["Looking that up, sir.", "Two items on Tuesday, sir."]
    assert msgs[-1]["reason"] == "answered"


def test_a_claude_task_returns_the_ack_and_lets_the_client_go(server):
    server.app.result = CommandResult(handled=True, reply="On it, sir.", speak=True,
                                      ack=True, done=False, status="Working…")
    t0 = time.monotonic()
    msgs = _run(server.sock, "fix the tests in jarvis")
    elapsed = time.monotonic() - t0
    assert 0.3 < elapsed < 2.5
    assert [m["text"] for m in msgs if m["kind"] == "reply"] == ["On it, sir."]
    assert msgs[-1] == {"kind": "end", "reason": "idle"}


def test_a_reply_that_never_comes_ends_on_the_timeout(server):
    server.app.result = CommandResult(handled=True, status="Thinking…", done=False)
    server.app.plan = "hang"
    t0 = time.monotonic()
    msgs = _run(server.sock, "hello", timeout=1.0)
    assert 0.9 < time.monotonic() - t0 < 3.0
    assert msgs[-1] == {"kind": "end", "reason": "timeout"}


def test_a_sync_answer_with_no_reply_still_ends(server):
    server.app.result = CommandResult(handled=True, status="Target: firefox")
    msgs = _run(server.sock, "target firefox")
    assert [m["kind"] for m in msgs] == ["status", "end"] and msgs[-1]["reason"] == "done"


def test_malformed_and_empty_requests_get_an_error_line(server):
    import json
    import socket
    for raw in (b"not json\n", json.dumps({"text": ""}).encode() + b"\n", b"[1,2]\n"):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(str(server.sock))
        s.sendall(raw)
        data = b""
        while b'"end"' not in data:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
        s.close()
        lines = [json.loads(ln) for ln in data.decode().splitlines() if ln.strip()]
        assert lines[0]["kind"] == "error" and lines[-1] == {"kind": "end", "reason": "error"}
    assert server.app.calls == []


def test_a_crashing_dispatch_is_reported_not_swallowed(server):
    def boom(text, source="typed", quiet=False):
        raise RuntimeError("commander exploded")
    server.app.dispatch_text = boom
    msgs = _run(server.sock, "hello")
    assert msgs[0]["kind"] == "error" and msgs[-1]["reason"] == "error"


def test_the_collector_unsubscribes_after_the_turn(server):
    before = {k: len(v) for k, v in bus._subs.items()}
    _run(server.sock, "what time is it")
    after = {k: len(v) for k, v in bus._subs.items()}
    assert after == before


# ------------------------------------------------------------- the CLI
def test_the_cli_prints_the_reply_and_exits_zero(server, capsys):
    rc = ask_mod.main(["--sock", str(server.sock), "what", "time", "is", "it"])
    out, err = capsys.readouterr()
    assert rc == 0 and out.strip() == "It is ten, sir."
    assert "[info] Clock" in err


def test_the_cli_status_and_quiet_flags(server, capsys):
    assert ask_mod.main(["--sock", str(server.sock), "--status"]) == 0
    assert capsys.readouterr().out.strip() == "All systems nominal, sir."
    assert ask_mod.main(["--sock", str(server.sock), "-q", "hello"]) == 0
    assert server.app.calls[-1] == ("hello", "cli", True)


def test_the_cli_json_mode(server, capsys):
    import json
    assert ask_mod.main(["--sock", str(server.sock), "--json", "hello"]) == 0
    lines = [json.loads(ln) for ln in capsys.readouterr().out.splitlines()]
    assert [ln["kind"] for ln in lines] == ["reply", "status", "end"]


def test_the_cli_says_when_jarvis_is_not_running(tmp_path, capsys):
    rc = ask_mod.main(["--sock", str(tmp_path / "nope.sock"), "hello"])
    assert rc == 2 and "not running" in capsys.readouterr().err


def test_the_cli_exits_three_when_no_reply_came(server, capsys):
    server.app.result = CommandResult(handled=True, status="Working…", done=False)
    rc = ask_mod.main(["--sock", str(server.sock), "fix", "it"])
    assert rc == 3 and "no reply: idle" in capsys.readouterr().err


def test_the_cli_needs_some_text(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    assert ask_mod.main([]) == 1


# ------------------------------------------------------ the real app
def test_the_real_app_answers_over_the_socket(build, monkeypatch):  # noqa: F811
    from jarvis.config import CONFIG, PATHS
    monkeypatch.setattr(CONFIG, "talkback", True)
    app = build()
    assert type(app.cmdsock).__name__ == "CommandSocket"
    app.start_assistant(residency=False)
    assert app.cmdsock.running and PATHS.COMMAND_SOCK.exists()
    # a prefixed Tier-1 command answers synchronously and, by default, aloud
    msgs = _run(PATHS.COMMAND_SOCK, "jarvis what time is it")
    replies = [m["text"] for m in msgs if m["kind"] == "reply"]
    assert len(replies) == 1 and msgs[-1]["reason"] == "done"
    assert app.tts.spoken and app.tts.spoken[-1] == replies[0]
    # -q keeps the soundbar silent for THAT turn only
    spoken_before = list(app.tts.spoken)
    msgs = _run(PATHS.COMMAND_SOCK, "jarvis what time is it", quiet=True)
    assert [m["text"] for m in msgs if m["kind"] == "reply"]
    assert app.tts.spoken == spoken_before
    # the mute is per turn: a sync answer clears it on the way out, so a
    # reminder firing a minute later is not silently swallowed
    assert not app._quiet_turn
    # the next typed turn speaks again
    app.dispatch_text("jarvis what time is it", source="typed")
    assert len(app.tts.spoken) == len(spoken_before) + 1 and not app._quiet_turn
    # the CLI never lands in the typed-box history
    assert "jarvis what time is it" not in getattr(app.history, "items", lambda: [])()
    app.stop_assistant()
    assert not PATHS.COMMAND_SOCK.exists()


def test_a_foreign_turns_tagged_reply_never_answers_this_one(server):
    """Two CLI clients at once: the other turn's stamped reply must not
    close this stream with the wrong text. Untagged (voice/typed) events
    keep today's behaviour."""
    app = server.app

    def dispatch(text, source="typed", quiet=False, turn_id=""):
        app.calls.append((text, source, quiet))

        def _later():
            time.sleep(0.1)
            bus.publish(JarvisReply(text="the other client's answer",
                                    speak=False, turn_id="deadbeef0000"))
            time.sleep(0.1)
            bus.publish(JarvisReply(text="two items on Tuesday, sir.",
                                    speak=False, turn_id=turn_id))
        threading.Thread(target=_later, daemon=True).start()
        return CommandResult(handled=True, reply="", speak=False,
                             status="Thinking…", done=False)
    app.dispatch_text = dispatch
    msgs = _run(server.sock, "what's due this week")
    texts = [m["text"] for m in msgs if m["kind"] == "reply"]
    assert "the other client's answer" not in texts
    assert "two items on Tuesday, sir." in texts
    assert msgs[-1]["reason"] == "answered"
