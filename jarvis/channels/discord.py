"""Two-way Discord channel (spec section 8.2).

A bot account, configured in `~/.config/jarvis/assistant.json` under
`discord.{bot_token, channel_id, user_id}`, lets Jarvis reach Hunter when he
is away from the desk and lets Hunter answer back:

- out: `post(text)` -> `POST /channels/<id>/messages` in 2000-char chunks,
  honouring 429 `retry_after` (the Alerts hub calls this);
- in: a gateway thread (`websockets` 15 with asyncio inside the thread)
  does HELLO -> heartbeat -> IDENTIFY (intents GUILDS | GUILD_MESSAGES |
  MESSAGE_CONTENT | DIRECT_MESSAGES), then hands every MESSAGE_CREATE in
  the configured channel (or a DM from the configured user) that was not
  written by the bot to `on_message(text, author_id)`; drops are RESUMEd
  with 1-60 s backoff; after three failed gateway attempts it falls back
  to REST polling `GET /channels/<id>/messages?after=<last_id>&limit=20`
  every 5 s.

Filtering: with `user_id` set, only that user is obeyed — in the channel
as well as by DM — because a "yes" in a shared channel can approve a
`git push`; with it unset, anyone in the channel counts. Other bots and
the bot itself are always ignored.

Secrets: the token is NEVER logged. Every log line and every exception
text that leaves this module passes through `_redact`, `redacted()`
describes the channel without it, and the third-party `websockets` logger
is pinned to INFO so its DEBUG frame dumps (which would carry the IDENTIFY
payload) can never reach a handler.

Transport: `UrllibTransport` (stdlib `urllib` for REST, `websockets` for
the gateway) is the only thing that touches the network; tests inject a
fake with scripted gateway frames and recorded REST calls (`transport=`),
so unit tests never open a socket.

Threading: `start()` spawns one daemon thread (`discord-gateway`); the
`on_message` callback runs ON that thread — the app must hand the text to
its own worker (it does: `dispatch_text` is asynchronous). `post()` is
safe from any thread (a lock keeps chunks of concurrent posts in order).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

from jarvis.channels import cfg_get, is_placeholder, redact
from jarvis.logs import get_logger

log = get_logger("channels.discord")

API_BASE = "https://discord.com/api/v10"
GATEWAY_QUERY = "v=10&encoding=json"
USER_AGENT = "DiscordBot (https://github.com/hunterp/Jarvis, 3.0)"
MESSAGE_LIMIT = 2000

INTENT_GUILDS = 1 << 0
INTENT_GUILD_MESSAGES = 1 << 9
INTENT_DIRECT_MESSAGES = 1 << 12
INTENT_MESSAGE_CONTENT = 1 << 15
INTENTS = (INTENT_GUILDS | INTENT_GUILD_MESSAGES
           | INTENT_MESSAGE_CONTENT | INTENT_DIRECT_MESSAGES)      # 37377

# Gateway opcodes
OP_DISPATCH = 0
OP_HEARTBEAT = 1
OP_IDENTIFY = 2
OP_RESUME = 6
OP_RECONNECT = 7
OP_INVALID_SESSION = 9
OP_HELLO = 10
OP_HEARTBEAT_ACK = 11

# Close codes after which retrying is pointless (Discord gateway docs).
FATAL_CLOSE_CODES = {
    4004: "authentication failed — check discord.bot_token",
    4010: "invalid shard",
    4011: "sharding required",
    4012: "invalid gateway version",
    4013: "invalid intents",
}
# 4014: a privileged intent (Message Content) is not enabled in the
# Developer Portal. The gateway is unusable but REST polling still works.
CLOSE_DISALLOWED_INTENTS = 4014
# Codes after which the session cannot be resumed (must IDENTIFY afresh).
NO_RESUME_CLOSE_CODES = {4007, 4009, CLOSE_DISALLOWED_INTENTS}

DEFAULT_ACTIVE_WINDOW_S = 600.0


# ---------------------------------------------------------------- transport
@dataclass
class HttpResponse:
    status: int
    headers: dict
    body: bytes = b""

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8")) if self.body else None
        except Exception:
            return None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class TransportError(Exception):
    """Network failure surfaced by a transport (message already redacted
    by the channel before it is logged)."""


class UrllibTransport:
    """The real thing: stdlib urllib for REST; `websockets` for the gateway.
    Never raises on an HTTP status (the channel reads 429/401 bodies);
    raises TransportError on connection-level failures."""

    def __init__(self, open_timeout: float = 15.0):
        self.open_timeout = open_timeout

    def request(self, method: str, url: str, headers: dict,
                body: Optional[bytes] = None, timeout: float = 10.0) -> HttpResponse:
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return HttpResponse(resp.status, dict(resp.headers.items()), resp.read())
        except urllib.error.HTTPError as err:
            try:
                payload = err.read()
            except Exception:
                payload = b""
            return HttpResponse(err.code, dict(err.headers.items()) if err.headers else {},
                                payload)
        except urllib.error.URLError as err:
            raise TransportError(f"{method} {_path_only(url)}: {err.reason}") from None
        except Exception as err:
            raise TransportError(f"{method} {_path_only(url)}: {err}") from None

    def connect(self, url: str):
        """Return an object usable as `async with transport.connect(url) as ws`
        with `await ws.send(str)`, `await ws.recv() -> str`, `await ws.close()`."""
        # Pin the library logger so DEBUG frame dumps (the IDENTIFY payload
        # carries the token) can never be emitted, whatever the root level.
        for name in ("websockets", "websockets.client", "websockets.asyncio.client"):
            lg = logging.getLogger(name)
            if lg.level == logging.NOTSET or lg.level < logging.INFO:
                lg.setLevel(logging.INFO)
        from websockets.asyncio.client import connect   # lazy: tests never need it
        return connect(url, open_timeout=self.open_timeout, close_timeout=5,
                       max_size=4 * 1024 * 1024, user_agent_header=USER_AGENT)


def _path_only(url: str) -> str:
    try:
        parts = urllib.parse.urlsplit(url)
        return parts.path
    except Exception:
        return "<url>"


# ------------------------------------------------------------- pure helpers
def chunk_text(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Split a message into <= `limit`-char pieces, preferring newline then
    space boundaries in the second half of each piece."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        piece = text[:cut].rstrip()
        if not piece:                        # pathological whitespace run
            piece, cut = text[:limit], limit
        chunks.append(piece)
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def gateway_url(base: str) -> str:
    """`wss://gateway.discord.gg` -> `wss://gateway.discord.gg/?v=10&encoding=json`."""
    base = (base or "").rstrip("/")
    if "?" in base:
        return base
    return f"{base}/?{GATEWAY_QUERY}"


def _close_code(exc: BaseException) -> Optional[int]:
    """Close code from a websockets ConnectionClosed (or a fake with .code)."""
    for attr in ("code",):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code
    rcvd = getattr(exc, "rcvd", None)
    code = getattr(rcvd, "code", None)
    return code if isinstance(code, int) else None


def _mask_id(value: str) -> str:
    value = str(value or "")
    return f"…{value[-4:]}" if len(value) > 4 else ("set" if value else "unset")


def message_wanted(msg: dict, channel_id: str, user_id: str, bot_id: str) -> bool:
    """The filter (pure): our channel, or a DM from the owner; never the bot
    itself or another bot; with `user_id` set, only the owner counts."""
    author = msg.get("author") or {}
    aid = str(author.get("id") or "")
    if not aid or (bot_id and aid == bot_id) or author.get("bot"):
        return False
    if msg.get("webhook_id"):
        return False
    cid = str(msg.get("channel_id") or "")
    is_dm = "guild_id" not in msg or msg.get("guild_id") in (None, "")
    if channel_id and cid == channel_id:
        return not user_id or aid == user_id
    if is_dm and user_id and aid == user_id:
        return True
    return False


# ---------------------------------------------------------------- channel
class DiscordChannel:
    """See the module docstring. `on_message(text, author_id)` is called on
    the gateway thread for every accepted message."""

    INTENTS = INTENTS

    def __init__(self, cfg: Any, on_message: Callable[[str, str], Any],
                 transport=None, *, poll_interval_s: float = 5.0,
                 backoff_s: tuple[float, float] = (1.0, 60.0),
                 recv_poll_s: float = 1.0,
                 active_window_s: float = DEFAULT_ACTIVE_WINDOW_S,
                 max_gateway_failures: int = 3,
                 request_timeout_s: float = 10.0,
                 now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self._cfg = cfg
        self._on_message = on_message
        self._transport = transport or UrllibTransport()
        self._poll_interval = float(poll_interval_s)
        self._backoff_min, self._backoff_max = float(backoff_s[0]), float(backoff_s[1])
        self._recv_poll = float(recv_poll_s)
        self._active_window = float(active_window_s)
        self._max_failures = int(max_gateway_failures)
        self._request_timeout = float(request_timeout_s)
        self._now = now
        self._sleep = sleep

        token = cfg_get(cfg, "discord.bot_token", "")
        channel_id = cfg_get(cfg, "discord.channel_id", "")
        user_id = cfg_get(cfg, "discord.user_id", "")
        self._token = "" if is_placeholder(token) else str(token).strip()
        self._channel_id = "" if is_placeholder(channel_id) else str(channel_id).strip()
        self._user_id = "" if is_placeholder(user_id) else str(user_id).strip()

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._post_lock = threading.Lock()
        self._mode = "off"                      # off | starting | gateway | polling | stopped
        self._active_until = 0.0
        self._content_hint_logged = False

        # Gateway state (readable by tests / status lines).
        self.bot_id = ""
        self.session_id = ""
        self.resume_url = ""
        self.last_seq: Optional[int] = None
        self.last_message_id = ""
        self.gateway_failures = 0
        self.heartbeat_interval_s = 0.0
        self.heartbeats_sent = 0
        self.connections = 0
        self.messages_in = 0
        self.messages_out = 0
        self._session_ready = False
        self._awaiting_ack = False

        try:   # let the Alerts hub mask our token in its own log lines too
            from jarvis.channels import notify as _notify
            _notify.register_secret(lambda: self._token)
        except Exception:
            pass

    # -- introspection ---------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self._token and self._channel_id)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def channel_id(self) -> str:
        return self._channel_id

    @property
    def user_id(self) -> str:
        return self._user_id

    def redacted(self) -> str:
        return (f"DiscordChannel(configured={self.configured}, mode={self._mode}, "
                f"channel={_mask_id(self._channel_id)}, user={_mask_id(self._user_id)}, "
                f"token={'set' if self._token else 'unset'})")

    __repr__ = redacted

    def _redact(self, text: Any) -> str:
        return redact(text, self._token)

    def status_text(self) -> str:
        """One persona-neutral clause for status lines (no secrets)."""
        if not self.configured:
            return "Discord is not set up"
        return {"gateway": "Discord connected", "polling": "Discord polling",
                "starting": "Discord connecting", "stopped": "Discord stopped",
                }.get(self._mode, "Discord off")

    # -- active window ----------------------------------------------------
    def mark_active(self, now: Optional[float] = None) -> None:
        """An exchange over Discord just happened: echo replies there for
        the next `active_window_s` (10 min)."""
        self._active_until = (self._now() if now is None else now) + self._active_window

    def is_active(self, now: Optional[float] = None) -> bool:
        return (self._now() if now is None else now) < self._active_until

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> bool:
        if not self.configured:
            log.info("discord: not configured; channel off")
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()
        self._mode = "starting"
        self._thread = threading.Thread(target=self._run_thread,
                                        name="discord-gateway", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and \
                thread is not threading.current_thread():
            thread.join(timeout)
        if self._mode not in ("off",):
            self._mode = "stopped"

    def join(self, timeout: float = 5.0) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    # -- outbound ------------------------------------------------------------
    def post(self, text: str, timeout_s: float = 10.0) -> bool:
        """Post `text` to the configured channel in <= 2000-char chunks.
        Returns True when every chunk was accepted within `timeout_s`."""
        if not self.configured:
            return False
        chunks = chunk_text(text)
        if not chunks:
            return False
        deadline = time.monotonic() + float(timeout_s)
        ok = True
        with self._post_lock:
            for chunk in chunks:
                resp = self._api("POST", f"/channels/{self._channel_id}/messages",
                                 body={"content": chunk}, deadline=deadline)
                if resp is None or not resp.ok:
                    ok = False
                    status = getattr(resp, "status", "no response")
                    log.warning("discord: post failed (%s)", status)
                    if resp is not None and resp.status in (401, 403, 404):
                        break                    # the rest would fail the same way
                else:
                    self.messages_out += 1
        return ok

    # -- REST --------------------------------------------------------------------
    def _headers(self, with_body: bool) -> dict:
        headers = {"Authorization": f"Bot {self._token}", "User-Agent": USER_AGENT,
                   "Accept": "application/json"}
        if with_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _api(self, method: str, path: str, body: Optional[dict] = None,
             params: Optional[dict] = None, deadline: Optional[float] = None,
             timeout: Optional[float] = None, quiet: bool = False) -> Optional[HttpResponse]:
        """One REST call with 429 handling. Returns the response (any
        status) or None on a transport failure / budget exhaustion.
        `quiet` demotes the failure line to DEBUG (repeated poll failures)."""
        url = f"{API_BASE}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        per_call = self._request_timeout if timeout is None else float(timeout)
        if deadline is None:
            deadline = time.monotonic() + per_call
        attempts = 0
        while True:
            attempts += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning("discord: %s %s gave up (budget exhausted)", method, path)
                return None
            try:
                resp = self._transport.request(method, url, self._headers(data is not None),
                                               body=data, timeout=min(per_call, remaining))
            except Exception as exc:
                log.log(logging.DEBUG if quiet else logging.WARNING,
                        "discord: %s %s failed: %s", method, path, self._redact(exc))
                return None
            if resp.status != 429:
                return resp
            wait = _retry_after(resp)
            if attempts >= 5 or wait > (deadline - time.monotonic()):
                log.warning("discord: rate limited on %s %s; retry_after %.2fs exceeds budget",
                            method, path, wait)
                return resp
            log.info("discord: rate limited; waiting %.2fs", wait)
            self._sleep(wait)

    # -- inbound -----------------------------------------------------------------
    def _note_message_id(self, mid: Any) -> None:
        mid = str(mid or "")
        if not mid.isdigit():
            return
        if not self.last_message_id or int(mid) > int(self.last_message_id):
            self.last_message_id = mid

    def _handle_message(self, msg: dict) -> bool:
        """Filter + deliver one message object (gateway or REST shape)."""
        if not isinstance(msg, dict):
            return False
        self._note_message_id(msg.get("id"))
        if not message_wanted(msg, self._channel_id, self._user_id, self.bot_id):
            return False
        text = (msg.get("content") or "").strip()
        if not text:
            if not msg.get("attachments") and not msg.get("embeds") and \
                    not self._content_hint_logged:
                self._content_hint_logged = True
                log.warning("discord: a message arrived with empty content — enable the "
                            "Message Content intent for the bot in the Developer Portal")
            return False
        self.messages_in += 1
        self.mark_active()
        author = str((msg.get("author") or {}).get("id") or "")
        try:
            self._on_message(text, author)
        except Exception as exc:
            log.error("discord: on_message failed: %s", self._redact(exc))
        return True

    # -- thread body -------------------------------------------------------------
    def _run_thread(self) -> None:
        outcome = "poll"
        try:
            outcome = asyncio.run(self._gateway_loop())
        except Exception as exc:
            log.error("discord: gateway thread crashed: %s", self._redact(exc))
        if outcome == "poll" and not self._stop.is_set():
            try:
                self._poll_loop()
            except Exception as exc:
                log.error("discord: polling crashed: %s", self._redact(exc))
        self._mode = "stopped"

    # -- gateway -----------------------------------------------------------------
    def _fetch_gateway_url(self) -> str:
        resp = self._api("GET", "/gateway/bot")
        if resp is None:
            raise TransportError("gateway lookup failed")
        if resp.status == 401:
            raise FatalGateway("authentication failed — check discord.bot_token")
        if not resp.ok:
            raise TransportError(f"gateway lookup returned {resp.status}")
        data = resp.json() or {}
        url = data.get("url") if isinstance(data, dict) else None
        if not url:
            raise TransportError("gateway lookup returned no url")
        return gateway_url(url)

    async def _async_wait(self, seconds: float) -> bool:
        """Sleep in slices so stop() is honoured; True when stopped."""
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end:
            if self._stop.is_set():
                return True
            await asyncio.sleep(min(0.25, max(0.0, end - time.monotonic())))
        return self._stop.is_set()

    async def _gateway_loop(self) -> str:
        """Returns 'stopped' | 'poll' | 'fatal'."""
        backoff = self._backoff_min
        while not self._stop.is_set():
            self._session_ready = False
            try:
                if self.session_id and self.resume_url:
                    url = self.resume_url
                else:
                    url = self._fetch_gateway_url()
                outcome = await self._session(url)
            except FatalGateway as exc:
                log.error("discord: %s; giving up", self._redact(exc))
                return "fatal"
            except Exception as exc:
                outcome = self._classify_drop(exc)
                if outcome == "fatal":
                    return "fatal"
            if outcome == "stopped" or self._stop.is_set():
                return "stopped"
            if self.gateway_failures >= self._max_failures:
                log.warning("discord: gateway unavailable after %d attempts; "
                            "falling back to REST polling every %.0fs",
                            self.gateway_failures, self._poll_interval)
                return "poll"
            if self._session_ready:
                backoff = self._backoff_min
            if await self._async_wait(backoff):
                return "stopped"
            backoff = min(backoff * 2, self._backoff_max)
        return "stopped"

    def _classify_drop(self, exc: BaseException) -> str:
        """Bookkeeping for a session that ended by exception."""
        code = _close_code(exc)
        if code in FATAL_CLOSE_CODES:
            log.error("discord: gateway closed %d: %s; giving up",
                      code, FATAL_CLOSE_CODES[code])
            return "fatal"
        if code == CLOSE_DISALLOWED_INTENTS:
            log.error("discord: gateway closed 4014 — enable the Message Content "
                      "intent for the bot in the Developer Portal; using REST polling")
            self.gateway_failures = max(self.gateway_failures, self._max_failures)
        if code in NO_RESUME_CLOSE_CODES:
            self._clear_session()
        if self._session_ready:
            log.info("discord: gateway dropped (%s); resuming",
                     code if code is not None else self._redact(exc))
        else:
            self.gateway_failures += 1
            log.warning("discord: gateway attempt %d failed: %s", self.gateway_failures,
                        f"close {code}" if code is not None else self._redact(exc))
        return "dropped"

    def _clear_session(self) -> None:
        self.session_id = ""
        self.resume_url = ""
        self.last_seq = None

    async def _send(self, ws, payload: dict) -> None:
        await ws.send(json.dumps(payload))

    async def _send_heartbeat(self, ws) -> None:
        await self._send(ws, {"op": OP_HEARTBEAT, "d": self.last_seq})
        self.heartbeats_sent += 1
        self._awaiting_ack = True

    async def _heartbeat_loop(self, ws, interval: float) -> None:
        await asyncio.sleep(interval * random.random())      # jitter, per the docs
        while True:
            if self._awaiting_ack:
                log.warning("discord: no heartbeat ACK; reconnecting")
                try:
                    await ws.close(code=4000, reason="zombie")
                except Exception:
                    pass
                return
            await self._send_heartbeat(ws)
            await asyncio.sleep(interval)

    async def _session(self, url: str) -> str:
        """One gateway connection: 'stopped' | 'reconnect'. Any close
        surfaces as an exception (classified by the caller)."""
        self.connections += 1
        self._awaiting_ack = False
        async with self._transport.connect(url) as ws:
            hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if hello.get("op") != OP_HELLO:
                raise TransportError(f"expected HELLO, got op {hello.get('op')}")
            interval = float((hello.get("d") or {}).get("heartbeat_interval", 41250)) / 1000.0
            self.heartbeat_interval_s = interval
            hb = asyncio.create_task(self._heartbeat_loop(ws, interval))
            try:
                if self.session_id and self.last_seq is not None:
                    await self._send(ws, {"op": OP_RESUME, "d": {
                        "token": self._token, "session_id": self.session_id,
                        "seq": self.last_seq}})
                else:
                    await self._send(ws, {"op": OP_IDENTIFY, "d": {
                        "token": self._token, "intents": self.INTENTS,
                        "properties": {"os": "linux", "browser": "jarvis",
                                       "device": "jarvis"}}})
                while not self._stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=self._recv_poll)
                    except asyncio.TimeoutError:
                        continue
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        log.warning("discord: undecodable gateway frame")
                        continue
                    if not isinstance(msg, dict):
                        continue
                    seq = msg.get("s")
                    if isinstance(seq, int):
                        self.last_seq = seq
                    op = msg.get("op")
                    if op == OP_DISPATCH:
                        self._dispatch(msg.get("t") or "", msg.get("d") or {})
                    elif op == OP_HEARTBEAT:
                        await self._send_heartbeat(ws)
                    elif op == OP_HEARTBEAT_ACK:
                        self._awaiting_ack = False
                    elif op == OP_RECONNECT:
                        log.info("discord: gateway asked us to reconnect")
                        return "reconnect"
                    elif op == OP_INVALID_SESSION:
                        resumable = bool(msg.get("d"))
                        if not resumable:
                            self._clear_session()
                        log.info("discord: invalid session (resumable=%s); reconnecting",
                                 resumable)
                        await self._async_wait(min(self._backoff_min, 1.0)
                                               + random.random() * min(self._backoff_min, 4.0))
                        return "reconnect"
                return "stopped"
            finally:
                hb.cancel()
                try:
                    await hb
                except (asyncio.CancelledError, Exception):
                    pass

    def _dispatch(self, event: str, data: dict) -> None:
        if event == "READY":
            self.session_id = str(data.get("session_id") or "")
            resume = data.get("resume_gateway_url")
            self.resume_url = gateway_url(resume) if resume else ""
            self.bot_id = str((data.get("user") or {}).get("id") or "")
            self._on_ready("connected")
        elif event == "RESUMED":
            self._on_ready("resumed")
        elif event == "MESSAGE_CREATE":
            self._handle_message(data)

    def _on_ready(self, how: str) -> None:
        self._session_ready = True
        self.gateway_failures = 0
        self._mode = "gateway"
        log.info("discord: gateway %s (bot %s, channel %s)", how,
                 _mask_id(self.bot_id), _mask_id(self._channel_id))

    # -- polling fallback ---------------------------------------------------------
    def _poll_loop(self) -> None:
        self._mode = "polling"
        path = f"/channels/{self._channel_id}/messages"
        first = True
        fails = 0                      # consecutive transport failures
        while not self._stop.is_set():
            # Back off while the network is down (5 s -> 10 -> ... -> 60 s).
            wait = min(self._poll_interval * (2 ** min(fails, 6)), max(self._poll_interval, 60.0))
            if not first and self._stop.wait(wait):
                return
            first = False
            if not self.last_message_id:
                # Establish the cursor without replaying history: an old
                # "yes" must never answer a new approval.
                resp = self._api("GET", path, params={"limit": 1}, quiet=fails > 0)
                fails = fails + 1 if resp is None else 0
                if resp is not None and resp.ok:
                    data = resp.json()
                    if isinstance(data, list) and data:
                        self._note_message_id(data[0].get("id"))
                    elif isinstance(data, list):
                        self.last_message_id = "0"
                elif resp is not None and resp.status in (401, 403):
                    log.error("discord: polling refused (%d); check the token and the "
                              "bot's channel permissions", resp.status)
                    return
                continue
            params = {"after": self.last_message_id, "limit": 20}
            resp = self._api("GET", path, params=params, quiet=fails > 0)
            if resp is None:
                fails += 1
                continue
            fails = 0
            if resp.status in (401, 403):
                log.error("discord: polling refused (%d); check the token and the "
                          "bot's channel permissions", resp.status)
                return
            if not resp.ok:
                log.warning("discord: poll returned %d", resp.status)
                continue
            data = resp.json()
            if not isinstance(data, list):
                continue
            for msg in sorted(data, key=lambda m: int(str(m.get("id", "0")) or 0)):
                self._handle_message(msg)


class FatalGateway(Exception):
    """Retrying cannot help (bad token); the channel gives up."""


def _retry_after(resp: HttpResponse) -> float:
    data = resp.json()
    value = None
    if isinstance(data, dict):
        value = data.get("retry_after")
    if value is None:
        for key, val in (resp.headers or {}).items():
            if str(key).lower() == "retry-after":
                value = val
                break
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 1.0


__all__ = ["DiscordChannel", "UrllibTransport", "HttpResponse", "TransportError",
           "FatalGateway", "chunk_text", "gateway_url", "message_wanted", "INTENTS",
           "MESSAGE_LIMIT", "API_BASE"]
