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

A request may carry a recorded clip instead of text --

    {"audio_b64": "<wav/ogg/opus/flac>", "speak": false, "timeout": 90}

-- which jarvis/intercom.py decodes to the pipeline's own 16 kHz mono
float32, transcribes with the resident Whisper and dispatches with
source="intercom". The transcript comes back first as
{"kind": "heard", "text"}, so a misheard clip is visible as such at the
far end, and nothing is spoken in the room unless "speak" is true. That
is the phone intercom: `ssh spark 'jarvis --send-audio -' < clip.ogg`.

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

import base64
import json
import os
import queue
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from jarvis import intercom
from jarvis.events import (BrainState, BriefingReady, JarvisReply, Status,
                           UserUtterance, bus)
from jarvis.logs import get_logger

log = get_logger("cmdsock")

# One request is one line. A text ask is a few hundred bytes; an intercom
# clip (jarvis/intercom.py) is base64, so 10 MB of audio is ~13.4 MB of JSON
# -- two hundred times the 64 KB this used to allow, which silently truncated
# the request into a JSON parse error. The cap is the audio cap plus the
# base64 expansion plus room for the rest of the object.
MAX_LINE = intercom.MAX_B64_CHARS + 65536
DEFAULT_TIMEOUT_S = 90.0
MAX_TIMEOUT_S = 600.0
IDLE_GRACE_S = 3.0            # a done=False turn that never woke the model
SYNC_REPLY_WAIT_S = 1.0       # the Tk pump delivers the sync reply within 30 ms
NOT_RUNNING_LINE = "Jarvis is not running (no command socket)."
INTERCOM_OFF_LINE = intercom.OFF_LINE


class ReplyCollector:
    """Queue the bus events of one turn; the socket thread drains it.

    ``turn_id`` filters out replies STAMPED for a different turn (another
    CLI client's answer landing on the shared bus must not close this
    stream with the wrong text). Untagged events pass through: voice and
    typed turns do not stamp, and dropping them would break the idle-edge
    heuristics this stream already relies on."""

    def __init__(self, turn_id: str = ""):
        self.turn_id = turn_id
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

    def _foreign(self, ev) -> bool:
        tid = getattr(ev, "turn_id", "")
        return bool(self.turn_id and tid and tid != self.turn_id)

    def _on_reply(self, ev):
        if self._foreign(ev):
            return
        self.q.put(("reply", {"kind": "reply", "text": ev.text, "speak": bool(ev.speak)}))

    def _on_briefing(self, ev):
        if self._foreign(ev):
            return
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
            except RequestTooLarge as exc:
                # An oversized clip used to arrive here as a truncated line
                # and be reported as "malformed request", which sends the
                # phone hunting for a JSON bug that is not there.
                _send(conn, {"kind": "error", "text": str(exc)})
                _send(conn, {"kind": "end", "reason": "error"})
                return
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
        source, quiet = "cli", bool(msg.get("quiet", False))
        if msg.get("audio_b64"):
            # The intercom: a clip from the phone. It is transcribed FIRST
            # (on this client's thread, so the resident Whisper serialises
            # with the microphone path on its own model lock) and the
            # transcript then travels the ordinary turn. Silent unless
            # asked: nobody sending a clip from bed wants the soundbar
            # answering the room.
            source, quiet = intercom.SOURCE, not bool(msg.get("speak", False))
            try:
                text = self._intercom_text(msg)
            except intercom.IntercomError as exc:
                log.info("intercom: %s (%s)", exc.text, exc.kind)
                _send(conn, {"kind": "error", "text": exc.text})
                _send(conn, {"kind": "end", "reason": "error"})
                return
            # What he heard, before the answer: a misheard clip is otherwise
            # indistinguishable from a wrong answer at the far end.
            _send(conn, {"kind": "heard", "text": text})
        else:
            text = str(msg.get("text") or "").strip()
        if not text:
            _send(conn, {"kind": "error", "text": "empty request"})
            _send(conn, {"kind": "end", "reason": "error"})
            return
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
        # `jarvis board`: the mission-control panel as plain text, printed
        # rather than spoken. Beside the status case for the same reason --
        # it is a READ of app state, not a turn, so it must not travel the
        # dispatch path, be remembered as an exchange, or wake the speaker.
        # `jarvis "people reload"`: re-read people.json without a restart.
        # A READ of a file the owner just edited at this same keyboard, not
        # a turn -- so it sits beside status and board rather than
        # travelling the dispatch path. It changes WHO Jarvis recognises
        # and changes nothing about what anybody may do: enrolling is still
        # scripts/jarvis_people.py at a terminal, with consent, and this
        # cannot create, promote or address a person.
        if text.lower() in ("people reload", "reload people"):
            _send(conn, {"kind": "reply", "text": _reload_people(self.app),
                         "speak": False})
            _send(conn, {"kind": "end", "reason": "done"})
            return
        if text.lower() in ("board", "the board"):
            fn = getattr(self.app, "board_text", None)
            line = fn() if callable(fn) else "board unavailable"
            _send(conn, {"kind": "reply", "text": line, "speak": False})
            _send(conn, {"kind": "end", "reason": "done"})
            return
        log.info("%s: %r%s", source, text, " (quiet)" if quiet else "")
        turn_id = uuid.uuid4().hex[:12]
        with ReplyCollector(turn_id) as col:
            bus.publish(UserUtterance(text=text, source=source))
            result = self.app.dispatch_text(text, source=source, quiet=quiet,
                                            turn_id=turn_id)
            reason = self._stream(conn, col, result, timeout)
        _send(conn, {"kind": "end", "reason": reason})

    def _intercom_text(self, msg: dict) -> str:
        """Decode + transcribe one ``{"audio_b64": ...}`` request.

        The switches, the cap and the decode live in jarvis/intercom.py so
        the phone client (jarvis/webapp.py) reads them from the same place.
        Raises IntercomError carrying the line to send back; anything else
        is left to _client, which answers with the generic error rather
        than leaking a traceback down the socket."""
        return intercom.text_from_b64(self.app, msg["audio_b64"])

    def _stream(self, conn, col: ReplyCollector, result, timeout: float) -> str:
        """Forward queued events until the turn is over; returns why."""
        return stream_turn(lambda payload: _send(conn, payload), col, result,
                           timeout, self.idle_grace_s)


# ------------------------------------------------------- the turn's end
def stream_turn(send, col: ReplyCollector, result, timeout: float,
                idle_grace_s: float = IDLE_GRACE_S) -> str:
    """Hand every event of one turn to ``send`` until the turn is over;
    returns why it ended (done / answered / idle / timeout).

    The five close conditions are the module docstring's, and they are the
    hard-won part of this file -- so the transport is a callable rather
    than a socket, and jarvis/webapp.py (the phone) drives the SAME loop
    with ``send=lines.append`` instead of writing a second one that would
    drift out of step with this one."""
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
            _flush(send, col)
            return "done"
        if done and now - last_event > SYNC_REPLY_WAIT_S:
            return "done"          # the bus never delivered it (no pump)
        if not done and not thinking and now - last_event > idle_grace_s:
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
                _flush(send, col)
                return "answered"
            continue
        send(payload)
        if kind == "reply":
            if not sync_seen and payload["text"] == sync_reply:
                sync_seen = True
            elif not done and not thinking:
                _flush(send, col)
                return "answered"


def _flush(send, col: ReplyCollector, settle_s: float = 0.1):
    end = time.monotonic() + settle_s
    while True:
        try:
            kind, payload = col.q.get(timeout=max(0.0, end - time.monotonic()))
        except queue.Empty:
            return
        if kind != "state":
            send(payload)


# ------------------------------------------------------------------ wire
class RequestTooLarge(ValueError):
    """The request ran past MAX_LINE without a newline."""

    def __init__(self, limit: int):
        super().__init__("request too large (over %d MB)" % (limit // 1048576))


def _reload_people(app) -> str:
    """Re-read the owner registry and hand back the startup line.

    Never raises: this is a convenience, and a convenience that can kill
    the socket thread is not one. The line it returns is the SAME one the
    app logs at boot, so what he reads back is what the gate now believes.
    """
    gate = getattr(app, "gate", None)
    if gate is None:
        return "owner-gate: OFF -- there is no gate on this process."
    try:
        gate.reload()
    except Exception:  # noqa: BLE001 - a reload that failed changes nothing
        log.exception("people reload failed")
        return "the registry could not be re-read; nothing changed"
    try:
        return gate.startup_line()
    except Exception:  # noqa: BLE001
        return "the registry was re-read"


def _read_line(conn: socket.socket, limit: int = MAX_LINE) -> str:
    buf = bytearray()
    while len(buf) < limit:
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf.extend(chunk)
        if b"\n" in chunk:
            break
    if len(buf) >= limit and b"\n" not in buf:
        raise RequestTooLarge(limit)
    line, _, _ = bytes(buf).partition(b"\n")
    return line.decode("utf-8", "replace").strip()


def _send(conn: socket.socket, obj: dict):
    try:
        conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))
    except OSError as exc:
        log.warning("command reply not delivered: %s", exc)


def ask(sock_path, text: str = "", quiet: bool = False,
        timeout: float = DEFAULT_TIMEOUT_S, audio: bytes = None,
        speak: bool = False):
    """Client generator: yields each JSON message from the server, ending
    with the {"kind": "end"} line. Raises ConnectionError when Jarvis is
    not running (no socket, or nobody listening).

    `audio` sends a recorded clip instead of text (the intercom): the
    bytes of a wav / ogg / opus / flac file, base64 on the wire. `speak`
    asks for the answer aloud on the Spark as well -- off by default,
    because the point of the intercom is a silent house."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(5.0)
        try:
            s.connect(str(sock_path))
        except OSError as exc:
            raise ConnectionError(NOT_RUNNING_LINE) from exc
        s.settimeout(float(timeout) + 5.0)
        req = {"quiet": bool(quiet), "timeout": float(timeout)}
        if audio:
            req["audio_b64"] = base64.b64encode(bytes(audio)).decode("ascii")
            req["speak"] = bool(speak)
        else:
            req["text"] = text
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
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
