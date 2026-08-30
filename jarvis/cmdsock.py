"""Command socket: ask Jarvis something from a shell, over SSH, from tmux
or from a script -- no cloud, no Tk.

speak_queue.txt lets any process make Jarvis TALK; approvals.sock carries
yes/no; the only ways to ASK him were the Tk box and Discord. This is a
UNIX SOCK_STREAM server beside approvals.sock (PATHS.COMMAND_SOCK, mode
0600, one thread per client, modelled on approvals.ApprovalBroker) that
takes one JSON line

    {"text": "what's due this week", "quiet": false, "timeout": 90}

dispatches it through JarvisApp.dispatch_text(text, source="cli") on the
client's thread and streams back the turn as JSON lines --
{"kind": "reply", "text", "speak"} for every JarvisReply / BriefingReady,
{"kind": "status", "text", "level"} for every Status -- until the turn is
over, then {"kind": "end", "reason"}. `text == "status"` answers with
diagnostics_text() and no dispatch.

Knowing when a turn is over is the whole difficulty. There is no turn id:
JarvisReply is {text, speak} and the app's _turn_busy is cleared at once
for every non-voice source, even when the CommandResult says done=False
(brain chat, the briefing, mail and web lookups all answer later from a
worker thread). So the collector subscribes to the bus BEFORE dispatching
and closes on the first of:

* the synchronous result was done -- after the reply line it published;
* a JarvisReply arrives while the model is NOT thinking (a Tier-1 answer
  that came back asynchronously);
* BrainState(idle) after a BrainState(thinking) -- the model answered
  (brain.chat / web_answer publish both);
* nothing at all for `idle_grace_s` after the ack of a done=False result
  that never started the model -- a Claude task, a desktop chain: the
  ack IS the answer for a shell client;
* the hard timeout.

Bus subscribers run on the Tk thread (a 30 ms pump), so the callbacks only
push into a queue.Queue; the client thread drains it to the socket.
"""
from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Optional

from jarvis.events import (BrainState, BriefingReady, JarvisReply, Status,
                           UserUtterance, bus)
from jarvis.logs import get_logger

log = get_logger("cmdsock")

MAX_LINE = 65536
DEFAULT_TIMEOUT_S = 90.0
MAX_TIMEOUT_S = 600.0
IDLE_GRACE_S = 3.0            # a done=False turn that never woke the model
SYNC_REPLY_WAIT_S = 1.0       # the Tk pump delivers the sync reply within 30 ms
NOT_RUNNING_LINE = "Jarvis is not running (no command socket)."


class ReplyCollector:
    """Queue the bus events of one turn; the socket thread drains it."""

    def __init__(self):
        self.q: "queue.Queue[tuple]" = queue.Queue()
        self._subs = ((JarvisReply, self._on_reply), (Status, self._on_status),
                      (BrainState, self._on_state), (BriefingReady, self._on_briefing))

    def __enter__(self):
        for etype, fn in self._subs:
            bus.subscribe(etype, fn)
        return self

    def __exit__(self, *exc):
        for etype, fn in self._subs:
            bus.unsubscribe(etype, fn)

    def _on_reply(self, ev):
        self.q.put(("reply", {"kind": "reply", "text": ev.text, "speak": bool(ev.speak)}))

    def _on_briefing(self, ev):
        text = ev.spoken or json.dumps(ev.sections)
        self.q.put(("reply", {"kind": "reply", "text": text, "speak": bool(ev.spoken),
                              "sections": ev.sections}))

    def _on_status(self, ev):
        self.q.put(("status", {"kind": "status", "text": ev.text, "level": ev.kind}))

    def _on_state(self, ev):
        self.q.put(("state", {"kind": "state", "state": ev.state}))


class CommandSocket:
    """UNIX SOCK_STREAM server; one thread per client."""

    def __init__(self, sock_path, app, timeout_s: float = DEFAULT_TIMEOUT_S,
                 idle_grace_s: float = IDLE_GRACE_S):
        self.sock_path = Path(sock_path)
        self.app = app
        self.timeout_s = float(timeout_s)
        self.idle_grace_s = float(idle_grace_s)
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.served = 0

    # ------------------------------------------------------------ lifecycle
    @property
    def running(self) -> bool:
        return self._server is not None and not self._stop.is_set()

    def start(self) -> bool:
        if self.running:
            return True
        self._stop.clear()
        try:
            self.sock_path.parent.mkdir(parents=True, exist_ok=True)
            if self.sock_path.exists():
                self.sock_path.unlink()
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(str(self.sock_path))
            os.chmod(self.sock_path, 0o600)
            srv.listen(8)
            srv.settimeout(0.5)
        except OSError:
            log.exception("command socket could not bind %s", self.sock_path)
            return False
        self._server = srv
        self._thread = threading.Thread(target=self._serve, name="cmdsock", daemon=True)
        self._thread.start()
        log.info("command socket listening on %s", self.sock_path)
        return True

    def stop(self):
        self._stop.set()
        srv, self._server = self._server, None
        if srv is not None:
            try:
                srv.close()
            except OSError:
                pass
        try:
            if self.sock_path.exists():
                self.sock_path.unlink()
        except OSError:
            pass
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def _serve(self):
        while not self._stop.is_set():
            srv = self._server
            if srv is None:
                return
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                if not self._stop.is_set():
                    log.exception("command socket accept failed")
                return
            threading.Thread(target=self._client, args=(conn,),
                             name="cmdsock-client", daemon=True).start()

    # --------------------------------------------------------------- client
    def _client(self, conn: socket.socket):
        with conn:
            try:
                conn.settimeout(10)
                line = _read_line(conn)
            except (OSError, ValueError) as exc:
                log.warning("command client dropped before sending: %s", exc)
                return
            if not line:
                return
            try:
                msg = json.loads(line)
                if not isinstance(msg, dict):
                    raise ValueError("not an object")
            except ValueError as exc:
                _send(conn, {"kind": "error", "text": f"malformed request: {exc}"})
                _send(conn, {"kind": "end", "reason": "error"})
                return
            self.served += 1
            try:
                self.serve_request(conn, msg)
            except Exception:                       # noqa: BLE001 - reply anyway
                log.exception("command request failed")
                _send(conn, {"kind": "error", "text": "Jarvis could not handle that"})
                _send(conn, {"kind": "end", "reason": "error"})

    def serve_request(self, conn, msg: dict):
        text = str(msg.get("text") or "").strip()
        if not text:
            _send(conn, {"kind": "error", "text": "empty request"})
            _send(conn, {"kind": "end", "reason": "error"})
            return
        quiet = bool(msg.get("quiet", False))
        try:
            timeout = float(msg.get("timeout") or self.timeout_s)
        except (TypeError, ValueError):
            timeout = self.timeout_s
        timeout = max(1.0, min(timeout, MAX_TIMEOUT_S))
        conn.settimeout(None)
        if text.lower() in ("status", "diagnostics"):
            fn = getattr(self.app, "diagnostics_text", None)
            line = fn() if callable(fn) else "diagnostics unavailable"
            _send(conn, {"kind": "reply", "text": line, "speak": False})
            _send(conn, {"kind": "end", "reason": "done"})
            return
        log.info("cli: %r%s", text, " (quiet)" if quiet else "")
        with ReplyCollector() as col:
            bus.publish(UserUtterance(text=text, source="cli"))
            result = self.app.dispatch_text(text, source="cli", quiet=quiet)
            reason = self._stream(conn, col, result, timeout)
        _send(conn, {"kind": "end", "reason": reason})

    def _stream(self, conn, col: ReplyCollector, result, timeout: float) -> str:
        """Forward queued events until the turn is over; returns why."""
        done = result is None or getattr(result, "done", True) is not False
        sync_reply = getattr(result, "reply", None) or ""
        sync_seen = not sync_reply
        thinking = False
        deadline = time.monotonic() + timeout
        last_event = time.monotonic()
        while True:
            now = time.monotonic()
            if now >= deadline:
                return "timeout"
            if done and sync_seen:
                # The sync reply landed; anything else queued goes with it.
                self._flush(conn, col)
                return "done"
            if done and now - last_event > SYNC_REPLY_WAIT_S:
                return "done"          # the bus never delivered it (no pump)
            if not done and not thinking and now - last_event > self.idle_grace_s:
                return "idle"          # Claude task / desktop chain: the ack is it
            try:
                kind, payload = col.q.get(timeout=0.25)
            except queue.Empty:
                continue
            last_event = time.monotonic()
            if kind == "state":
                if payload["state"] == "thinking":
                    thinking = True
                elif thinking and payload["state"] == "idle":
                    self._flush(conn, col)
                    return "answered"
                continue
            _send(conn, payload)
            if kind == "reply":
                if not sync_seen and payload["text"] == sync_reply:
                    sync_seen = True
                elif not done and not thinking:
                    self._flush(conn, col)
                    return "answered"

    @staticmethod
    def _flush(conn, col: ReplyCollector, settle_s: float = 0.1):
        end = time.monotonic() + settle_s
        while True:
            try:
                kind, payload = col.q.get(timeout=max(0.0, end - time.monotonic()))
            except queue.Empty:
                return
            if kind != "state":
                _send(conn, payload)


# ------------------------------------------------------------------ wire
def _read_line(conn: socket.socket, limit: int = MAX_LINE) -> str:
    buf = bytearray()
    while len(buf) < limit:
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf.extend(chunk)
        if b"\n" in chunk:
            break
    line, _, _ = bytes(buf).partition(b"\n")
    return line.decode("utf-8", "replace").strip()


def _send(conn: socket.socket, obj: dict):
    try:
        conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))
    except OSError as exc:
        log.warning("command reply not delivered: %s", exc)


def ask(sock_path, text: str, quiet: bool = False, timeout: float = DEFAULT_TIMEOUT_S):
    """Client generator: yields each JSON message from the server, ending
    with the {"kind": "end"} line. Raises ConnectionError when Jarvis is
    not running (no socket, or nobody listening)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(5.0)
        try:
            s.connect(str(sock_path))
        except OSError as exc:
            raise ConnectionError(NOT_RUNNING_LINE) from exc
        s.settimeout(float(timeout) + 5.0)
        s.sendall((json.dumps({"text": text, "quiet": bool(quiet),
                               "timeout": float(timeout)}) + "\n").encode("utf-8"))
        buf = b""
        while True:
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                yield {"kind": "end", "reason": "client-timeout"}
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue
                yield msg
                if isinstance(msg, dict) and msg.get("kind") == "end":
                    return
    finally:
        try:
            s.close()
        except OSError:
            pass
