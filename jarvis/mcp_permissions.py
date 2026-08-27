"""Jarvis permission-prompt MCP server (stdio, dependency-free).

Spec: docs/specs/2026-08-26-jarvis-personal-assistant.md, section 7.3.

Claude Code spawns this process per `claude -p` task (via the generated
`--mcp-config`) and calls its single tool `approve` for every permission
it cannot settle from settings.local.json (`--permission-prompt-tool
mcp__jarvis__approve`).  The tool forwards the request to the running
Jarvis app over the UNIX socket named by `JARVIS_APPROVAL_SOCK`
(jarvis/approvals.py) and returns Claude Code's permission-result shape
as the tool's text content:

    {"behavior": "allow", "updatedInput": <input>}
    {"behavior": "deny",  "message": "..."}

Transport: JSON-RPC 2.0, one JSON object per line on stdin/stdout.  Nothing
but protocol goes to stdout; diagnostics go to stderr.  No jarvis imports —
the CLI runs this with the project as cwd, so only PYTHONPATH-free stdlib
is safe here.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import uuid

SERVER_NAME = "jarvis-permissions"
SERVER_VERSION = "1.0.0"
TOOL_NAME = "approve"
DEFAULT_PROTOCOL = "2024-11-05"
WAIT_S = 125.0                  # broker denies at 120 s; we outlast it
NOT_RUNNING = "Jarvis is not running"

TOOL_SPEC = {
    "name": TOOL_NAME,
    "description": ("Ask Hunter, through Jarvis, whether Claude may use a "
                    "tool. Returns a permission result."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "tool_name": {"type": "string",
                          "description": "The tool Claude wants to use."},
            "input": {"type": "object",
                      "description": "The tool's input."},
            "tool_use_id": {"type": "string",
                            "description": "The tool use id (optional)."},
        },
        "required": ["tool_name", "input"],
    },
}


def _log(msg: str):
    try:
        sys.stderr.write(f"[{SERVER_NAME}] {msg}\n")
        sys.stderr.flush()
    except OSError:
        pass


# ------------------------------------------------------------- the socket
def ask_jarvis(sock_path: str, tool_name: str, tool_input, project: str = "",
               cwd: str = "", timeout: float = WAIT_S,
               request_id: str = "") -> dict:
    """One request / one reply over the broker socket.  Returns the reply
    object; a missing or dead socket returns a deny with NOT_RUNNING."""
    if not sock_path or not os.path.exists(sock_path):
        return {"id": request_id, "behavior": "deny", "message": NOT_RUNNING}
    msg = {"id": request_id or uuid.uuid4().hex[:12], "tool_name": tool_name,
           "input": tool_input if isinstance(tool_input, dict) else {"value": tool_input},
           "project": project, "cwd": cwd}
    deadline = time.monotonic() + timeout
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(sock_path)
            s.sendall((json.dumps(msg) + "\n").encode("utf-8"))
            buf = bytearray()
            while b"\n" not in buf:
                left = deadline - time.monotonic()
                if left <= 0:
                    return {"id": msg["id"], "behavior": "deny",
                            "message": "Jarvis did not answer in time"}
                s.settimeout(left)
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf.extend(chunk)
    except (OSError, socket.timeout) as exc:
        _log(f"socket error: {exc}")
        return {"id": msg["id"], "behavior": "deny",
                "message": f"{NOT_RUNNING} ({exc.__class__.__name__})"}
    line = bytes(buf).partition(b"\n")[0].decode("utf-8", "replace").strip()
    if not line:
        return {"id": msg["id"], "behavior": "deny",
                "message": "Jarvis closed the connection without answering"}
    try:
        reply = json.loads(line)
    except ValueError:
        return {"id": msg["id"], "behavior": "deny",
                "message": "Jarvis sent an unreadable answer"}
    if not isinstance(reply, dict):
        reply = {}
    reply.setdefault("behavior", "deny")
    reply.setdefault("message", "")
    return reply


def permission_result(reply: dict, tool_input) -> dict:
    """Claude Code's expected shape for the tool's text content."""
    if str(reply.get("behavior", "")).lower() == "allow":
        return {"behavior": "allow", "updatedInput": tool_input}
    message = reply.get("message") or "Denied by Jarvis"
    return {"behavior": "deny", "message": str(message)}


# ------------------------------------------------------------------ server
class PermissionServer:
    def __init__(self, sock_path: str = "", project: str = "", cwd: str = "",
                 ask=ask_jarvis, timeout: float = WAIT_S):
        self.sock_path = sock_path
        self.project = project
        self.cwd = cwd
        self.ask = ask
        self.timeout = timeout
        self.protocol = DEFAULT_PROTOCOL

    # -- dispatch -----------------------------------------------------
    def handle(self, msg: dict):
        """Returns the response object, or None for notifications."""
        method = msg.get("method")
        msg_id = msg.get("id")
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        if method == "initialize":
            self.protocol = str(params.get("protocolVersion") or DEFAULT_PROTOCOL)
            return self._ok(msg_id, {
                "protocolVersion": self.protocol,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })
        if method == "notifications/initialized" or (
                isinstance(method, str) and method.startswith("notifications/")):
            return None
        if method == "ping":
            return self._ok(msg_id, {})
        if method == "tools/list":
            return self._ok(msg_id, {"tools": [TOOL_SPEC]})
        if method == "tools/call":
            return self._call(msg_id, params)
        if msg_id is None:
            return None                              # unknown notification
        return self._err(msg_id, -32601, f"Method not found: {method}")

    def _call(self, msg_id, params: dict):
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        if name != TOOL_NAME:
            return self._err(msg_id, -32602, f"Unknown tool: {name}")
        tool_name = str(args.get("tool_name") or "")
        tool_input = args.get("input")
        if tool_input is None:
            tool_input = {}
        _log(f"approve? {tool_name} {json.dumps(tool_input)[:200]}")
        try:
            reply = self.ask(self.sock_path, tool_name, tool_input,
                             self.project, self.cwd, self.timeout,
                             str(args.get("tool_use_id") or ""))
        except Exception as exc:                     # noqa: BLE001 - never crash the CLI
            _log(f"ask failed: {exc!r}")
            reply = {"behavior": "deny", "message": f"Jarvis error: {exc}"}
        result = permission_result(reply, tool_input)
        _log(f"-> {result['behavior']}")
        return self._ok(msg_id, {
            "content": [{"type": "text", "text": json.dumps(result)}],
            "isError": False,
        })

    @staticmethod
    def _ok(msg_id, result):
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _err(msg_id, code, message):
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": code, "message": message}}

    # -- stdio loop ---------------------------------------------------
    def serve(self, stdin=None, stdout=None):
        stdin = stdin or sys.stdin.buffer
        stdout = stdout or sys.stdout.buffer
        _log(f"serving; sock={self.sock_path or '(unset)'} project={self.project}")
        for raw in iter(stdin.readline, b""):
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self._write(stdout, self._err(None, -32700, "Parse error"))
                continue
            if isinstance(msg, list):                # batch
                out = [r for r in (self.handle(m) for m in msg
                                   if isinstance(m, dict)) if r is not None]
                if out:
                    self._write(stdout, out)
                continue
            if not isinstance(msg, dict):
                self._write(stdout, self._err(None, -32600, "Invalid request"))
                continue
            try:
                resp = self.handle(msg)
            except Exception as exc:                 # noqa: BLE001 - keep serving
                _log(f"handler crashed: {exc!r}")
                resp = self._err(msg.get("id"), -32603, f"Internal error: {exc}")
            if resp is not None:
                self._write(stdout, resp)
        _log("stdin closed; exiting")

    @staticmethod
    def _write(stdout, obj):
        try:
            stdout.write((json.dumps(obj) + "\n").encode("utf-8"))
            stdout.flush()
        except OSError as exc:
            _log(f"stdout write failed: {exc}")


def main(argv=None) -> int:
    server = PermissionServer(
        sock_path=os.environ.get("JARVIS_APPROVAL_SOCK", ""),
        project=os.environ.get("JARVIS_PROJECT", ""),
        cwd=os.getcwd())
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
