"""Approval broker — the Jarvis side of Claude Code's permission prompts.

Spec: docs/specs/2026-08-26-jarvis-personal-assistant.md, section 7.3.

When a `claude -p` task asks for a permission Claude Code cannot settle from
the project's settings.local.json (a path outside the allowed dirs, a tool
not in the allow list), the CLI calls the MCP tool `mcp__jarvis__approve`
(jarvis/mcp_permissions.py, a subprocess of the CLI).  That server connects
to this broker's UNIX socket, sends ONE JSON line and waits for ONE JSON
line back:

    -> {"id": "...", "tool_name": "Write", "input": {...}, "project": "x", "cwd": "/p"}
    <- {"id": "...", "behavior": "allow" | "deny", "message": "..."}

The broker either settles the request itself (`auto_policy`: read-only /
edit tools whose every path is under an allowed dir), or publishes
`ApprovalRequested` (the app speaks the question, the UI shows ALLOW /
DENY) and blocks the client until `answer()` arrives from typed / voice /
UI / Discord, or `timeout_s` passes (deny, with the timeout line).

Nothing here speaks or touches Tk; the app owns speech.
"""
from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from jarvis.events import ApprovalRequested, ApprovalResolved, bus
from jarvis.logs import get_logger

log = get_logger("approvals")

# --------------------------------------------------------------- lines (3.4)
TIMEOUT_LINE = "No answer in two minutes, sir; I've declined it."
ALLOWED_LINE = "Allowed, sir."
DECLINED_LINE = "Declined, sir."
NOT_RUNNING_MESSAGE = "Jarvis is not running"

# Tools the policy may settle on its own when every path is inside an
# allowed dir (7.3).  Bash is never auto-allowed: its paths are free text.
POLICY_TOOLS = frozenset({"Read", "Glob", "Grep", "Edit", "Write",
                          "MultiEdit", "NotebookEdit"})
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
READ_TOOLS = frozenset({"Read", "Glob", "Grep"})
FETCH_TOOLS = frozenset({"WebFetch", "WebSearch"})

# Keys whose value is a path (never a free-text pattern).
_PATH_KEYS = ("file_path", "path", "notebook_path", "filePath", "directory",
              "cwd")
# An absolute or ~-relative path token inside free text (Bash commands).
_PATH_TOKEN_RX = re.compile(r"(?<![\w@:.-])(?:~(?=[/\s]|$)|/)[^\s\"'`;|&<>()]*")

MAX_LINE = 1_000_000          # bytes per request line


# ------------------------------------------------------------------ request
@dataclass
class ApprovalRequest:
    request_id: str
    tool_name: str
    input: dict
    project: str = ""
    cwd: str = ""
    created: float = 0.0
    question: str = ""
    detail: str = ""
    # -- internal (not part of the spec's dataclass, defaults keep it compatible)
    _event: threading.Event = field(default_factory=threading.Event, repr=False)
    _allowed: Optional[bool] = field(default=None, repr=False)
    _source: str = field(default="", repr=False)
    _message: str = field(default="", repr=False)

    @property
    def allowed(self) -> Optional[bool]:
        return self._allowed

    @property
    def source(self) -> str:
        return self._source


# ------------------------------------------------------------- describing
def _short_path(path: str) -> str:
    """~/x for paths under home, the last three components otherwise."""
    path = str(path or "")
    home = str(Path.home())
    if path == home:
        return "~"
    if path.startswith(home + "/"):
        return "~/" + path[len(home) + 1:]
    parts = [p for p in path.split("/") if p]
    if len(parts) > 3:
        return ".../" + "/".join(parts[-3:])
    return path


def _first_path(inp: dict) -> str:
    for key in _PATH_KEYS:
        value = inp.get(key) if isinstance(inp, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _host(url: str) -> str:
    m = re.match(r"^\s*(?:[a-z][a-z0-9+.-]*://)?([^/\s?#]+)", str(url or ""), re.I)
    return m.group(1) if m else str(url or "")[:60]


def request_detail(tool_name: str, inp: dict) -> str:
    """The command / path / url the question is about (UI `detail`)."""
    inp = inp if isinstance(inp, dict) else {}
    if tool_name == "Bash":
        return str(inp.get("command") or "").strip()
    if tool_name in FETCH_TOOLS:
        return str(inp.get("url") or inp.get("query") or "").strip()
    path = _first_path(inp)
    if path:
        return path
    if tool_name in ("Glob", "Grep"):
        return str(inp.get("pattern") or "").strip()
    if tool_name == "Task":
        return str(inp.get("description") or inp.get("prompt") or "")[:80].strip()
    for value in inp.values():
        if isinstance(value, str) and value.strip():
            return value.strip()[:120]
    return ""


def describe_request(req: ApprovalRequest, cfg=None) -> str:
    """The spoken persona question (7.3 templates).

    'Claude wants to run "git push origin main", sir; shall I allow it?'
    'Claude wants to edit ~/.bashrc, sir, which is outside the project; allow it?'
    'Claude wants to fetch example.com, sir; allow it?'
    """
    tool = req.tool_name or "use a tool"
    inp = req.input if isinstance(req.input, dict) else {}
    detail = req.detail or request_detail(tool, inp)
    if tool == "Bash":
        cmd = re.sub(r"\s+", " ", detail).strip()
        if len(cmd) > 80:
            cmd = cmd[:77].rstrip() + "…"
        return f"Claude wants to run “{cmd}”, sir; shall I allow it?"
    if tool in FETCH_TOOLS:
        what = _host(detail) if tool == "WebFetch" else f"the web for “{detail[:60]}”"
        return f"Claude wants to {'fetch' if tool == 'WebFetch' else 'search'} {what}, sir; allow it?"
    if tool in EDIT_TOOLS or tool in READ_TOOLS:
        verb = "edit" if tool in EDIT_TOOLS else ("read" if tool == "Read" else "search")
        if tool == "Write" and detail and not os.path.exists(os.path.expanduser(detail)):
            verb = "create"
        shown = _short_path(detail) if detail else "a file"
        outside = ""
        if detail and cfg is not None and not _is_allowed(cfg, detail, req.cwd):
            outside = ", which is outside the project"
        return f"Claude wants to {verb} {shown}, sir{outside}; allow it?"
    if tool == "Task":
        return f"Claude wants to start an agent for “{detail[:60]}”, sir; allow it?"
    if tool.startswith("mcp__"):
        pretty = tool[5:].replace("__", " ")
        return f"Claude wants to use the {pretty} tool, sir; allow it?"
    shown = f" on {_short_path(detail)}" if detail else ""
    return f"Claude wants to use {tool}{shown}, sir; allow it?"


# ------------------------------------------------------------------ policy
def _allowed_dirs(cfg) -> list[str]:
    dirs = getattr(cfg, "allowed_dirs", None)
    if callable(dirs):
        dirs = dirs()
    if not dirs and hasattr(cfg, "get"):
        try:
            dirs = cfg.get("claude.allowed_dirs") or []
        except Exception:
            dirs = []
    return [os.path.abspath(os.path.expanduser(str(d))) for d in (dirs or [])
            if isinstance(d, str) and d.strip()]


def _is_allowed(cfg, path: str, base: str = "") -> bool:
    check = getattr(cfg, "is_allowed_path", None)
    if callable(check):
        try:
            return bool(check(path, base or None))
        except TypeError:
            return bool(check(path))
        except Exception:
            log.exception("is_allowed_path failed for %r", path)
            return False
    raw = os.path.expanduser(str(path))
    if base and not os.path.isabs(raw):
        raw = os.path.join(os.path.expanduser(base), raw)
    real = os.path.realpath(raw)
    for allowed in _allowed_dirs(cfg):
        root = os.path.realpath(allowed)
        if real == root or real.startswith(root.rstrip("/") + "/"):
            return True
    return False


def paths_in_input(tool_name: str, inp: dict) -> list[str]:
    """Every path a request touches: path-valued keys, plus absolute / ~
    tokens inside free text (Bash commands, patterns)."""
    found: list[str] = []

    def walk(value):
        if isinstance(value, dict):
            for key, sub in value.items():
                if key in _PATH_KEYS and isinstance(sub, str) and sub.strip():
                    found.append(sub.strip())
                else:
                    walk(sub)
        elif isinstance(value, list):
            for sub in value:
                walk(sub)
        elif isinstance(value, str):
            for m in _PATH_TOKEN_RX.finditer(value):
                tok = m.group(0).rstrip(".,:")
                if tok in ("/", "~") or len(tok) < 2:
                    continue
                found.append(tok)

    walk(inp if isinstance(inp, dict) else {})
    seen, out = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ------------------------------------------------------------------- broker
class ApprovalBroker:
    """UNIX SOCK_STREAM server; one thread per client (7.3)."""

    def __init__(self, sock_path, cfg, timeout_s: float = 120.0,
                 on_request: Optional[Callable[[ApprovalRequest], None]] = None,
                 on_resolved: Optional[Callable[[ApprovalRequest, bool, str], None]] = None):
        self.sock_path = Path(sock_path)
        self.cfg = cfg
        self.timeout_s = float(timeout_s)
        self.on_request = on_request
        self.on_resolved = on_resolved
        self._pending: dict[str, ApprovalRequest] = {}
        self._lock = threading.Lock()
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.history: list[ApprovalRequest] = []      # last 50, newest last

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
            log.exception("approval broker could not bind %s", self.sock_path)
            return False
        self._server = srv
        self._thread = threading.Thread(target=self._serve, name="approvals",
                                        daemon=True)
        self._thread.start()
        log.info("approval broker listening on %s", self.sock_path)
        return True

    def stop(self):
        self._stop.set()
        srv, self._server = self._server, None
        if srv is not None:
            try:
                srv.close()
            except OSError:
                pass
        with self._lock:
            pending = list(self._pending.values())
        for req in pending:
            self._resolve(req, False, "shutdown", "Jarvis is shutting down")
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
                    log.exception("approval accept failed")
                return
            threading.Thread(target=self._client, args=(conn,),
                             name="approval-client", daemon=True).start()

    # --------------------------------------------------------------- client
    def _client(self, conn: socket.socket):
        with conn:
            try:
                conn.settimeout(10)
                line = _read_line(conn)
            except (OSError, ValueError) as exc:
                log.warning("approval client dropped before sending: %s", exc)
                return
            if not line:
                return
            try:
                msg = json.loads(line)
                if not isinstance(msg, dict):
                    raise ValueError("not an object")
            except ValueError as exc:
                log.warning("approval request unparsable: %s", exc)
                _send(conn, {"id": "", "behavior": "deny",
                             "message": "malformed request"})
                return
            req = self._make_request(msg)
            try:
                allowed, message = self._handle(req)
            except Exception:                       # noqa: BLE001 - reply anyway
                log.exception("approval handling failed for %s", req.request_id)
                allowed, message = False, "Jarvis could not handle the request"
            _send(conn, {"id": req.request_id,
                         "behavior": "allow" if allowed else "deny",
                         "message": message})

    def _make_request(self, msg: dict) -> ApprovalRequest:
        tool = str(msg.get("tool_name") or msg.get("tool") or "")
        inp = msg.get("input")
        if not isinstance(inp, dict):
            inp = {"value": inp} if inp is not None else {}
        req = ApprovalRequest(
            request_id=str(msg.get("id") or uuid.uuid4().hex[:12]),
            tool_name=tool, input=inp,
            project=str(msg.get("project") or ""),
            cwd=str(msg.get("cwd") or ""),
            created=time.time())
        req.detail = request_detail(tool, inp)
        req.question = describe_request(req, self.cfg)
        return req

    def _handle(self, req: ApprovalRequest) -> tuple[bool, str]:
        policy = self.auto_policy(req)
        if policy == "allow":
            log.info("approval %s auto-allowed by policy: %s %s",
                     req.request_id, req.tool_name, req.detail[:80])
            req._allowed, req._source = True, "policy"
            self._remember(req)
            bus.publish(ApprovalResolved(request_id=req.request_id,
                                         allowed=True, source="policy"))
            return True, ""
        with self._lock:
            self._pending[req.request_id] = req
        log.info("approval %s asked: %s", req.request_id, req.question)
        bus.publish(ApprovalRequested(request_id=req.request_id,
                                      question=req.question,
                                      tool_name=req.tool_name,
                                      detail=req.detail, project=req.project))
        if self.on_request is not None:
            try:
                self.on_request(req)
            except Exception:                       # noqa: BLE001 - app hook
                log.exception("on_request hook failed")
        if not req._event.wait(self.timeout_s):
            self._resolve(req, False, "timeout", TIMEOUT_LINE)
        return bool(req._allowed), req._message

    # ---------------------------------------------------------------- api
    def pending(self) -> list[ApprovalRequest]:
        with self._lock:
            return sorted(self._pending.values(), key=lambda r: r.created)

    def answer(self, allowed: bool, request_id: Optional[str] = None,
               source: str = "typed") -> bool:
        """Settle one pending request (the named one, else the oldest)."""
        with self._lock:
            if request_id:
                req = self._pending.get(request_id)
            else:
                items = sorted(self._pending.values(), key=lambda r: r.created)
                req = items[0] if items else None
        if req is None:
            return False
        message = "" if allowed else f"Hunter declined via Jarvis ({source})"
        self._resolve(req, bool(allowed), source, message)
        return True

    def auto_policy(self, req: ApprovalRequest) -> Optional[str]:
        """'allow' when the tool is a read/edit tool and every path in its
        input is under an allowed dir; None otherwise (ask)."""
        if req.tool_name not in POLICY_TOOLS:
            return None
        paths = paths_in_input(req.tool_name, req.input)
        if not paths:
            return None
        for p in paths:
            if not _is_allowed(self.cfg, p, req.cwd):
                return None
        return "allow"

    # ------------------------------------------------------------ internals
    def _resolve(self, req: ApprovalRequest, allowed: bool, source: str,
                 message: str):
        with self._lock:
            live = self._pending.pop(req.request_id, None)
        if live is None and req._event.is_set():
            return                                  # already settled
        req._allowed, req._source, req._message = allowed, source, message
        self._remember(req)
        req._event.set()
        log.info("approval %s %s (%s)", req.request_id,
                 "allowed" if allowed else "declined", source)
        bus.publish(ApprovalResolved(request_id=req.request_id,
                                     allowed=allowed, source=source))
        if self.on_resolved is not None:
            try:
                self.on_resolved(req, allowed, source)
            except Exception:                       # noqa: BLE001 - app hook
                log.exception("on_resolved hook failed")

    def _remember(self, req: ApprovalRequest):
        self.history.append(req)
        del self.history[:-50]


# ------------------------------------------------------------- wire helpers
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
        log.warning("approval reply not delivered: %s", exc)


def ask(sock_path, tool_name: str, inp: dict, project: str = "",
        cwd: str = "", timeout: float = 125.0, request_id: str = "") -> dict:
    """Client side of the protocol (used by tests and by the MCP server's
    twin in mcp_permissions.py, which must stay dependency-free)."""
    msg = {"id": request_id or uuid.uuid4().hex[:12], "tool_name": tool_name,
           "input": inp, "project": project, "cwd": cwd}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(sock_path))
        s.sendall((json.dumps(msg) + "\n").encode("utf-8"))
        return json.loads(_read_line(s) or "{}")
