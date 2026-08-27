"""Tests for jarvis.channels — the Alerts hub and the two-way Discord channel.

Everything below runs against a fake transport: scripted gateway frames
(HELLO / READY / MESSAGE_CREATE / closes) and recorded REST calls, so no
socket is ever opened and no `notify-send` is ever spawned. The single live
test (JARVIS_LIVE=1) performs the unauthenticated gateway HELLO handshake —
read-only, no token — to prove the real transport against Discord.

Firewall: tests/conftest.py redirects JARVIS_LOG_DIR / JARVIS_ASSISTANT_CONFIG
before any jarvis import; nothing here touches /tmp/vss_voice, the live
app, or ~/.config/jarvis.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import types

import pytest

from jarvis.channels import cfg_get, is_placeholder, redact
from jarvis.channels import discord as dmod
from jarvis.channels import notify as nmod
from jarvis.channels.discord import (
    DiscordChannel,
    HttpResponse,
    INTENTS,
    chunk_text,
    gateway_url,
    message_wanted,
)
from jarvis.channels.notify import Alerts, discord_text, notify_argv

TOKEN = "bot-token-for-tests-only-0123456789abcdef"
CHANNEL = "111222333444555666"
OWNER = "777888999000111222"
BOT = "999000111222333444"
STRANGER = "123123123123123123"


# ------------------------------------------------------------------ fakes
class FakeCfg:
    """Looks like AssistantConfig.get(dotted, default) over a nested dict."""

    def __init__(self, data):
        self.data = data

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def cfg(token=TOKEN, channel=CHANNEL, user=OWNER, **extra):
    data = {"discord": {"bot_token": token, "channel_id": channel, "user_id": user}}
    data.update(extra)
    return FakeCfg(data)


class FakeClosed(Exception):
    def __init__(self, code):
        super().__init__(f"connection closed {code}")
        self.code = code


class FakeSocket:
    """Scripted gateway connection. Script items: a dict frame; ("sleep", s);
    ("close", code) -> recv raises FakeClosed(code); an Exception instance ->
    raised from recv. With auto_ack every heartbeat is answered with op 11."""

    def __init__(self, script, auto_ack=True):
        self.script = list(script)
        self.sent = []
        self.closed = None
        self.auto_ack = auto_ack
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, data):
        msg = json.loads(data)
        self.sent.append(msg)
        if msg.get("op") == 1 and self.auto_ack:
            self.script.insert(0, {"op": 11})

    async def recv(self):
        while True:
            if self.closed is not None:
                raise FakeClosed(self.closed)
            if not self.script:
                await asyncio.sleep(0.005)
                continue
            item = self.script.pop(0)
            if isinstance(item, tuple) and item[0] == "sleep":
                await asyncio.sleep(item[1])
                continue
            if isinstance(item, tuple) and item[0] == "close":
                raise FakeClosed(item[1])
            if isinstance(item, BaseException):
                raise item
            return json.dumps(item)

    async def close(self, code=1000, reason=""):
        self.closed = code


def resp(status, payload=None, headers=None):
    body = b"" if payload is None else json.dumps(payload).encode()
    return HttpResponse(status, headers or {}, body)


class FakeTransport:
    """Records REST calls; hands out scripted sockets (or raises) per connect."""

    def __init__(self, sockets=(), responder=None):
        self.sockets = list(sockets)
        self.connects = []
        self.calls = []                         # (method, url, headers, body)
        self.responder = responder or self.default_responder

    def request(self, method, url, headers, body=None, timeout=10.0):
        self.calls.append((method, url, dict(headers), body))
        return self.responder(method, url, body)

    def connect(self, url):
        self.connects.append(url)
        if not self.sockets:
            return FakeSocket([("sleep", 3600)])
        sock = self.sockets.pop(0)
        if isinstance(sock, BaseException):
            raise sock
        return sock

    @staticmethod
    def default_responder(method, url, body):
        if url.endswith("/gateway/bot"):
            return resp(200, {"url": "wss://fake.gateway"})
        if method == "POST" and "/messages" in url:
            return resp(200, {"id": "424242"})
        if method == "GET" and "/messages" in url:
            return resp(200, [])
        return resp(404, {"message": "unknown"})

    def posts(self):
        return [json.loads(b)["content"] for m, u, h, b in self.calls
                if m == "POST" and u.endswith("/messages")]


HELLO = {"op": 10, "d": {"heartbeat_interval": 50}}


def ready(session="sess-1", resume="wss://resume.fake", bot=BOT):
    return {"op": 0, "t": "READY", "s": 1,
            "d": {"session_id": session, "resume_gateway_url": resume,
                  "user": {"id": bot, "bot": True}}}


def message(text, author=OWNER, channel=CHANNEL, mid="1000", seq=None,
            guild="guild-1", bot=False, extra=None):
    d = {"id": mid, "channel_id": channel, "content": text,
         "author": {"id": author, "bot": bot}}
    if guild is not None:
        d["guild_id"] = guild
    if extra:
        d.update(extra)
    frame = {"op": 0, "t": "MESSAGE_CREATE", "d": d}
    if seq is not None:
        frame["s"] = seq
    return frame


def wait_for(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


@pytest.fixture
def make_channel():
    """Build a channel with fast timings; every channel is stopped at teardown."""
    made = []

    def _make(transport, config=None, on_message=None, **kw):
        received = []
        cb = on_message or (lambda text, author: received.append((text, author)))
        opts = dict(poll_interval_s=0.01, backoff_s=(0.01, 0.02), recv_poll_s=0.02,
                    request_timeout_s=1.0)
        opts.update(kw)
        ch = DiscordChannel(config or cfg(), cb, transport=transport, **opts)
        ch.received = received
        made.append(ch)
        return ch

    yield _make
    for ch in made:
        ch.stop(timeout=3.0)
        ch.join(3.0)


# ---------------------------------------------------------------- helpers
def test_cfg_get_reads_config_dict_and_namespace():
    assert cfg_get(cfg(), "discord.channel_id") == CHANNEL
    assert cfg_get({"discord": {"user_id": "u"}}, "discord.user_id") == "u"
    ns = types.SimpleNamespace(discord=types.SimpleNamespace(bot_token="t"))
    assert cfg_get(ns, "discord.bot_token") == "t"
    assert cfg_get(ns, "discord.missing", "dflt") == "dflt"
    assert cfg_get(None, "x.y", 3) == 3


@pytest.mark.parametrize("value,expected", [
    ("", True), (None, True), ("<your bot token>", True), ("YOUR_TOKEN", True),
    ("•••", True), ("changeme", True), (TOKEN, False), ("123", False), (0, False),
])
def test_is_placeholder(value, expected):
    assert is_placeholder(value) is expected


def test_redact_masks_secret_only_when_long_enough():
    assert redact(f"Bot {TOKEN} failed", TOKEN) == "Bot ••• failed"
    assert redact("short key ab", "ab") == "short key ab"
    assert redact(RuntimeError(TOKEN), TOKEN) == "•••"


# --------------------------------------------------------- pure functions
def test_chunk_text_respects_limit_and_boundaries():
    words = " ".join(f"w{i:04d}" for i in range(900))       # ~5400 chars
    chunks = chunk_text(words)
    assert len(chunks) == 3
    assert all(len(c) <= 2000 for c in chunks)
    assert all(not c.startswith(" ") and not c.endswith(" ") for c in chunks)
    assert " ".join(chunks).split() == words.split()
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    assert chunk_text("x" * 4001) == ["x" * 2000, "x" * 2000, "x"]
    para = ("line\n" * 500).strip()
    for c in chunk_text(para):
        assert len(c) <= 2000 and c.endswith("line")


def test_gateway_url_appends_version_and_encoding():
    assert gateway_url("wss://gateway.discord.gg") == "wss://gateway.discord.gg/?v=10&encoding=json"
    assert gateway_url("wss://x.gg/") == "wss://x.gg/?v=10&encoding=json"
    assert gateway_url("wss://x.gg/?v=10&encoding=json") == "wss://x.gg/?v=10&encoding=json"


def test_intents_are_the_four_required_bits():
    assert INTENTS == (1 << 0) | (1 << 9) | (1 << 15) | (1 << 12) == 37377


def test_message_wanted_filter_table():
    def m(**kw):
        return message("hi", **kw)["d"]
    # our channel, from the owner
    assert message_wanted(m(), CHANNEL, OWNER, BOT)
    # the bot's own echo
    assert not message_wanted(m(author=BOT, bot=True), CHANNEL, OWNER, BOT)
    # another bot in the channel
    assert not message_wanted(m(author=STRANGER, bot=True), CHANNEL, OWNER, BOT)
    # a stranger in the channel while an owner is configured
    assert not message_wanted(m(author=STRANGER), CHANNEL, OWNER, BOT)
    # no owner configured: anyone in the channel counts
    assert message_wanted(m(author=STRANGER), CHANNEL, "", BOT)
    # other channel
    assert not message_wanted(m(channel="555"), CHANNEL, OWNER, BOT)
    # DM from the owner (no guild_id) / DM from a stranger
    assert message_wanted(m(channel="dm-1", guild=None), CHANNEL, OWNER, BOT)
    assert not message_wanted(m(channel="dm-1", guild=None, author=STRANGER), CHANNEL, OWNER, BOT)
    # webhook posts never count
    assert not message_wanted(m(extra={"webhook_id": "9"}), CHANNEL, OWNER, BOT)
    # missing author
    assert not message_wanted({"channel_id": CHANNEL, "content": "x"}, CHANNEL, OWNER, BOT)


# ----------------------------------------------------------- unconfigured
def test_unconfigured_channel_is_a_no_op(make_channel, caplog):
    tr = FakeTransport()
    ch = make_channel(tr, cfg(token=""))
    assert ch.configured is False
    assert ch.start() is False
    assert ch.mode == "off"
    assert ch.post("hello") is False
    assert tr.calls == [] and tr.connects == []
    assert ch.status_text() == "Discord is not set up"
    assert "token" not in repr(ch).lower() or "token=unset" in repr(ch)


def test_placeholder_values_count_as_unconfigured(make_channel):
    assert make_channel(FakeTransport(), cfg(token="<your bot token>")).configured is False
    assert make_channel(FakeTransport(), cfg(channel="")).configured is False
    assert make_channel(FakeTransport(), cfg(user="")).configured is True   # user_id optional


def test_start_is_idempotent_and_stop_ends_thread(make_channel):
    tr = FakeTransport([FakeSocket([HELLO, ready(), ("sleep", 3600)])])
    ch = make_channel(tr)
    assert ch.start() is True
    assert ch.start() is True
    assert wait_for(lambda: ch.mode == "gateway")
    ch.stop(timeout=3.0)
    assert ch.join(3.0)
    assert ch.mode == "stopped"


# -------------------------------------------------------------- identify
def test_identify_payload_and_gateway_lookup(make_channel):
    sock = FakeSocket([HELLO, ready(), ("sleep", 3600)])
    tr = FakeTransport([sock])
    ch = make_channel(tr)
    ch.start()
    assert wait_for(lambda: ch.session_id == "sess-1")
    # REST lookup carried the bot token in the Authorization header only.
    method, url, headers, body = tr.calls[0]
    assert (method, url) == ("GET", "https://discord.com/api/v10/gateway/bot")
    assert headers["Authorization"] == f"Bot {TOKEN}"
    assert headers["User-Agent"].startswith("DiscordBot (")
    assert tr.connects == ["wss://fake.gateway/?v=10&encoding=json"]
    identify = sock.sent[0]
    assert identify["op"] == 2
    assert identify["d"]["token"] == TOKEN
    assert identify["d"]["intents"] == INTENTS
    assert identify["d"]["properties"]["os"] == "linux"
    assert ch.bot_id == BOT
    assert ch.resume_url == "wss://resume.fake/?v=10&encoding=json"
    assert ch.mode == "gateway"
    assert ch.status_text() == "Discord connected"


def test_heartbeat_uses_hello_interval_and_last_seq(make_channel):
    sock = FakeSocket([{"op": 10, "d": {"heartbeat_interval": 30}},
                       ready(), message("ping", mid="5", seq=7), ("sleep", 3600)])
    tr = FakeTransport([sock])
    ch = make_channel(tr)
    ch.start()
    assert wait_for(lambda: ch.heartbeats_sent >= 3, timeout=3.0)
    assert ch.heartbeat_interval_s == pytest.approx(0.03)
    beats = [m for m in sock.sent if m["op"] == 1]
    assert len(beats) >= 3
    assert beats[-1]["d"] == 7                      # last sequence number
    assert ch.last_seq == 7
    assert sock.closed is None                      # ACKed: never declared a zombie


def test_server_heartbeat_request_is_answered(make_channel):
    sock = FakeSocket([{"op": 10, "d": {"heartbeat_interval": 60000}}, ready(),
                       {"op": 1, "d": None}, ("sleep", 3600)], auto_ack=False)
    ch = make_channel(FakeTransport([sock]))
    ch.start()
    assert wait_for(lambda: any(m["op"] == 1 for m in sock.sent))


def test_missing_ack_closes_and_reconnects(make_channel):
    first = FakeSocket([{"op": 10, "d": {"heartbeat_interval": 20}}, ready(), ("sleep", 3600)],
                       auto_ack=False)
    second = FakeSocket([HELLO, {"op": 0, "t": "RESUMED", "d": {}}, ("sleep", 3600)])
    tr = FakeTransport([first, second])
    ch = make_channel(tr)
    ch.start()
    assert wait_for(lambda: second.entered, timeout=3.0)
    assert first.closed == 4000
    assert len(tr.connects) == 2
    assert ch.gateway_failures == 0                 # a drop after READY is not a failure


# --------------------------------------------------------- message filter
def test_message_create_filtering_and_delivery(make_channel):
    frames = [HELLO, ready(),
              message("echo", author=BOT, bot=True, mid="10"),
              message("elsewhere", channel="555", mid="11"),
              message("stranger", author=STRANGER, mid="12"),
              message("  yes  ", mid="13"),
              message("dm hello", channel="dm-9", guild=None, mid="14"),
              message("dm stranger", channel="dm-9", guild=None, author=STRANGER, mid="15"),
              message("", mid="16"),
              message("attachment only", mid="17", extra={"content": "", "attachments": [{}]}),
              ("sleep", 3600)]
    ch = make_channel(FakeTransport([FakeSocket(frames)]))
    assert ch.is_active() is False
    ch.start()
    assert wait_for(lambda: len(ch.received) == 2)
    time.sleep(0.05)
    assert ch.received == [("yes", OWNER), ("dm hello", OWNER)]
    assert ch.last_message_id == "17"
    assert ch.messages_in == 2
    assert ch.is_active() is True


def test_channel_open_to_anyone_when_no_user_id(make_channel):
    frames = [HELLO, ready(), message("hi from a friend", author=STRANGER, mid="3"),
              ("sleep", 3600)]
    ch = make_channel(FakeTransport([FakeSocket(frames)]), cfg(user=""))
    ch.start()
    assert wait_for(lambda: ch.received == [("hi from a friend", STRANGER)])


def test_on_message_exception_is_logged_not_fatal(make_channel, caplog):
    def boom(text, author):
        raise RuntimeError("handler failed " + TOKEN)
    frames = [HELLO, ready(), message("one", mid="1"), message("two", mid="2"),
              ("sleep", 3600)]
    ch = make_channel(FakeTransport([FakeSocket(frames)]), on_message=boom)
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        ch.start()
        assert wait_for(lambda: ch.messages_in == 2)
    assert ch.mode == "gateway"
    assert any("on_message failed" in r.getMessage() for r in caplog.records)
    assert all(TOKEN not in r.getMessage() for r in caplog.records)


def test_empty_content_hints_at_missing_intent_once(make_channel, caplog):
    frames = [HELLO, ready(), message("", mid="1"), message("", mid="2"), ("sleep", 3600)]
    ch = make_channel(FakeTransport([FakeSocket(frames)]))
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        ch.start()
        assert wait_for(lambda: ch.last_message_id == "2")
        time.sleep(0.03)
    hints = [r for r in caplog.records if "Message Content intent" in r.getMessage()]
    assert len(hints) == 1


# ---------------------------------------------------------- reconnection
def test_drop_after_ready_resumes_with_session_and_seq(make_channel):
    first = FakeSocket([HELLO, ready(session="S-42", resume="wss://resume.one"),
                        message("first", mid="1", seq=5), ("close", 1006)])
    second = FakeSocket([HELLO, {"op": 0, "t": "RESUMED", "d": {}},
                         message("second", mid="2", seq=6), ("sleep", 3600)])
    tr = FakeTransport([first, second])
    ch = make_channel(tr)
    ch.start()
    assert wait_for(lambda: len(ch.received) == 2, timeout=3.0)
    assert tr.connects[1] == "wss://resume.one/?v=10&encoding=json"
    resume = second.sent[0]
    assert resume["op"] == 6
    assert resume["d"] == {"token": TOKEN, "session_id": "S-42", "seq": 5}
    assert ch.last_seq == 6
    assert ch.gateway_failures == 0
    assert ch.mode == "gateway"
    # The gateway URL was looked up once; the resume URL needs no lookup.
    assert sum(1 for c in tr.calls if c[1].endswith("/gateway/bot")) == 1


def test_reconnect_opcode_triggers_resume(make_channel):
    first = FakeSocket([HELLO, ready(session="S-7"), {"op": 7, "d": None}, ("sleep", 3600)])
    second = FakeSocket([HELLO, {"op": 0, "t": "RESUMED", "d": {}}, ("sleep", 3600)])
    ch = make_channel(FakeTransport([first, second]))
    ch.start()
    assert wait_for(lambda: second.entered and len(second.sent) >= 1, timeout=3.0)
    assert second.sent[0]["op"] == 6 and second.sent[0]["d"]["session_id"] == "S-7"


def test_invalid_session_reidentifies(make_channel):
    first = FakeSocket([HELLO, ready(session="S-1"), {"op": 9, "d": False}])
    second = FakeSocket([HELLO, ready(session="S-2"), ("sleep", 3600)])
    ch = make_channel(FakeTransport([first, second]))
    ch.start()
    assert wait_for(lambda: ch.session_id == "S-2", timeout=3.0)
    assert second.sent[0]["op"] == 2                # IDENTIFY, not RESUME


def test_fatal_close_code_stops_without_polling(make_channel, caplog):
    sock = FakeSocket([HELLO, ("close", 4004)])
    tr = FakeTransport([sock])
    ch = make_channel(tr)
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        ch.start()
        assert ch.join(3.0)
    assert ch.mode == "stopped"
    assert len(tr.connects) == 1
    assert not any(c[0] == "GET" and "/messages" in c[1] for c in tr.calls)
    assert any("authentication failed" in r.getMessage() for r in caplog.records)


def test_disallowed_intents_close_falls_back_to_polling(make_channel, caplog):
    sock = FakeSocket([HELLO, ("close", 4014)])
    tr = FakeTransport([sock])
    ch = make_channel(tr)
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        ch.start()
        assert wait_for(lambda: ch.mode == "polling", timeout=3.0)
    assert any("Message Content" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------- polling fallback
def poll_responder(script):
    """GET /messages responder: the first (limit=1) call establishes the
    cursor; later calls pop from `script`."""
    state = {"n": 0}

    def responder(method, url, body):
        if url.endswith("/gateway/bot"):
            return resp(200, {"url": "wss://fake.gateway"})
        if method == "GET" and "/messages" in url:
            state["n"] += 1
            if "limit=1" in url and "after=" not in url:
                return resp(200, [{"id": "100", "channel_id": CHANNEL, "content": "old",
                                   "author": {"id": OWNER}}])
            return resp(200, script.pop(0) if script else [])
        return resp(200, {"id": "1"})
    return responder


def test_polling_after_three_gateway_failures(make_channel, caplog):
    batch = [  # newest first, as Discord returns them
        {"id": "103", "channel_id": CHANNEL, "content": "bot echo",
         "author": {"id": BOT, "bot": True}, "guild_id": "g"},
        {"id": "102", "channel_id": CHANNEL, "content": "no",
         "author": {"id": OWNER}, "guild_id": "g"},
        {"id": "101", "channel_id": CHANNEL, "content": "yes",
         "author": {"id": OWNER}, "guild_id": "g"},
    ]
    tr = FakeTransport([ConnectionError("refused"), ConnectionError("refused"),
                        ConnectionError("refused")], responder=poll_responder([batch]))
    ch = make_channel(tr)
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        ch.start()
        assert wait_for(lambda: len(ch.received) == 2, timeout=3.0)
    assert ch.mode == "polling"
    assert ch.status_text() == "Discord polling"
    assert ch.gateway_failures == 3
    assert ch.received == [("yes", OWNER), ("no", OWNER)]      # oldest first
    assert ch.last_message_id == "103"
    gets = [u for m, u, h, b in tr.calls if m == "GET" and "/messages" in u]
    assert gets[0].endswith(f"/channels/{CHANNEL}/messages?limit=1")
    assert any(u.endswith(f"/channels/{CHANNEL}/messages?after=100&limit=20") for u in gets)
    assert wait_for(lambda: any(u.endswith("after=103&limit=20") for m, u, h, b in tr.calls
                                if m == "GET"), timeout=2.0)
    assert any("falling back to REST polling" in r.getMessage() for r in caplog.records)


def test_polling_never_replays_history_without_cursor(make_channel):
    """The cursor is established with limit=1 and that message is NOT
    delivered — an hour-old 'yes' must never answer a new approval."""
    tr = FakeTransport([ConnectionError("x")] * 3, responder=poll_responder([]))
    ch = make_channel(tr)
    ch.start()
    assert wait_for(lambda: ch.last_message_id == "100", timeout=3.0)
    time.sleep(0.05)
    assert ch.received == []


def test_polling_stops_on_401(make_channel, caplog):
    def responder(method, url, body):
        if url.endswith("/gateway/bot"):
            return resp(200, {"url": "wss://fake.gateway"})
        return resp(401, {"message": "401: Unauthorized"})
    tr = FakeTransport([ConnectionError("x")] * 3, responder=responder)
    ch = make_channel(tr)
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        ch.start()
        assert ch.join(3.0)
    assert ch.mode == "stopped"
    assert any("polling refused (401)" in r.getMessage() for r in caplog.records)


def test_bad_token_on_gateway_lookup_is_fatal(make_channel):
    def responder(method, url, body):
        return resp(401, {"message": "401: Unauthorized", "code": 0})
    tr = FakeTransport([], responder=responder)
    ch = make_channel(tr)
    ch.start()
    assert ch.join(3.0)
    assert ch.mode == "stopped"
    assert tr.connects == []


# ---------------------------------------------------------------- posting
def test_post_chunks_at_2000_chars(make_channel):
    tr = FakeTransport()
    ch = make_channel(tr)
    text = " ".join(f"token{i:05d}" for i in range(500))      # ~5500 chars
    assert ch.post(text) is True
    posts = tr.posts()
    assert len(posts) == 3
    assert all(len(p) <= 2000 for p in posts)
    assert " ".join(posts).split() == text.split()
    for m, u, h, b in tr.calls:
        assert m == "POST" and u == f"https://discord.com/api/v10/channels/{CHANNEL}/messages"
        assert h["Authorization"] == f"Bot {TOKEN}"
        assert h["Content-Type"] == "application/json"
    assert ch.messages_out == 3
    assert ch.post("") is False


def test_post_honours_429_retry_after(make_channel):
    seen = {"n": 0}

    def responder(method, url, body):
        seen["n"] += 1
        if seen["n"] == 1:
            return resp(429, {"message": "You are being rate limited.", "retry_after": 0.01,
                              "global": False})
        return resp(200, {"id": "1"})
    slept = []
    tr = FakeTransport(responder=responder)
    ch = make_channel(tr, sleep=slept.append)
    assert ch.post("hello") is True
    assert seen["n"] == 2
    assert slept == [0.01]
    assert tr.posts() == ["hello", "hello"]


def test_post_429_header_fallback_and_budget(make_channel):
    def responder(method, url, body):
        return resp(429, None, headers={"Retry-After": "30"})
    slept = []
    ch = make_channel(FakeTransport(responder=responder), sleep=slept.append)
    assert ch.post("hello", timeout_s=0.5) is False       # 30 s exceeds the 0.5 s budget
    assert slept == []


def test_post_failure_is_logged_without_the_token(make_channel, caplog):
    def responder(method, url, body):
        raise RuntimeError(f"socket error for Bot {TOKEN}")
    ch = make_channel(FakeTransport(responder=responder))
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        assert ch.post("hello") is False
    assert any("post failed" in r.getMessage() for r in caplog.records)
    assert all(TOKEN not in r.getMessage() for r in caplog.records)


def test_post_stops_after_403(make_channel):
    calls = {"n": 0}

    def responder(method, url, body):
        calls["n"] += 1
        return resp(403, {"message": "Missing Access"})
    ch = make_channel(FakeTransport(responder=responder))
    assert ch.post("x" * 4500) is False
    assert calls["n"] == 1


# ------------------------------------------------------------ active window
def test_mark_active_window_is_ten_minutes(make_channel):
    ch = make_channel(FakeTransport())
    assert ch.is_active(now=1000.0) is False
    ch.mark_active(now=1000.0)
    assert ch.is_active(now=1000.0 + 599.0) is True
    assert ch.is_active(now=1000.0 + 600.0) is False
    assert ch.is_active(now=1000.0 + 601.0) is False


def test_active_window_uses_injected_clock(make_channel):
    clock = {"t": 50.0}
    ch = make_channel(FakeTransport(), now=lambda: clock["t"], active_window_s=60.0)
    ch.mark_active()
    clock["t"] = 109.0
    assert ch.is_active() is True
    clock["t"] = 111.0
    assert ch.is_active() is False


# ------------------------------------------------------------- token safety
def test_token_never_appears_in_logs_or_exceptions(make_channel, caplog):
    """Every failure path that can mention the token is exercised: gateway
    lookup, connect, recv, on_message and post."""
    class LeakyTransport(FakeTransport):
        def request(self, method, url, headers, body=None, timeout=10.0):
            self.calls.append((method, url, dict(headers), body))
            raise RuntimeError(f"DNS failed while sending Authorization: Bot {TOKEN}")

        def connect(self, url):
            self.connects.append(url)
            raise RuntimeError(f"connect refused with token {TOKEN}")

    tr = LeakyTransport()
    ch = make_channel(tr)
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        ch.start()
        # Three gateway lookups fail -> polling, whose GETs fail too.
        assert wait_for(lambda: ch.mode == "polling" and
                        sum(1 for c in tr.calls if c[0] == "GET" and "/messages" in c[1]) >= 3,
                        timeout=3.0)
        ch.post("hello")
        ch.stop(timeout=3.0)
    assert caplog.records, "expected failure log lines"
    # Repeated poll failures back off and drop to DEBUG after the first.
    poll_fails = [r for r in caplog.records if "/messages failed" in r.getMessage()]
    assert poll_fails[0].levelno == logging.WARNING
    assert any(r.levelno == logging.DEBUG for r in poll_fails[1:])
    for rec in caplog.records:
        assert TOKEN not in rec.getMessage()
        assert TOKEN not in str(rec.exc_info)
        assert TOKEN not in str(rec.exc_text)
    assert TOKEN not in repr(ch) and TOKEN not in ch.redacted()
    assert "token=set" in ch.redacted()
    assert TOKEN not in ch.status_text()


def test_close_code_helper_reads_websockets_shape():
    class Rcvd:
        code = 4009
    class Closed(Exception):
        rcvd = Rcvd()
    assert dmod._close_code(Closed()) == 4009
    assert dmod._close_code(FakeClosed(1006)) == 1006
    assert dmod._close_code(RuntimeError("x")) is None


def test_retry_after_prefers_body_then_header():
    assert dmod._retry_after(resp(429, {"retry_after": 2.5})) == 2.5
    assert dmod._retry_after(resp(429, None, {"Retry-After": "3"})) == 3.0
    assert dmod._retry_after(resp(429, {"retry_after": "bogus"})) == 1.0
    assert dmod._retry_after(resp(429)) == 1.0


def test_urllib_transport_maps_http_errors_to_responses(monkeypatch):
    import urllib.error
    import urllib.request

    class FakeHeaders(dict):
        def items(self):
            return super().items()

    def fake_urlopen(req, timeout=10.0):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many", FakeHeaders({"Retry-After": "1"}),
                                     __import__("io").BytesIO(b'{"retry_after": 1.0}'))
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    tr = dmod.UrllibTransport()
    r = tr.request("GET", "https://discord.com/api/v10/x", {"Authorization": f"Bot {TOKEN}"})
    assert r.status == 429 and r.json() == {"retry_after": 1.0}

    def fake_urlopen_down(req, timeout=10.0):
        raise urllib.error.URLError("Name or service not known")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen_down)
    with pytest.raises(dmod.TransportError) as ei:
        tr.request("GET", "https://discord.com/api/v10/gateway/bot", {"Authorization": f"Bot {TOKEN}"})
    assert TOKEN not in str(ei.value) and "/api/v10/gateway/bot" in str(ei.value)


# =========================================================== Alerts hub
class FakeDiscord:
    def __init__(self, configured=True, fail=False, delay=0.0):
        self.configured = configured
        self.posts = []
        self.fail = fail
        self.delay = delay

    def post(self, text, timeout_s=10.0):
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("discord down")
        self.posts.append((text, timeout_s))
        return True


class Recorder:
    def __init__(self, delay=0.0, rc=0, raise_exc=None):
        self.argv = []
        self.delay = delay
        self.rc = rc
        self.raise_exc = raise_exc

    def __call__(self, argv, timeout=5.0):
        self.argv.append((list(argv), timeout))
        if self.raise_exc:
            raise self.raise_exc
        if self.delay:
            time.sleep(self.delay)
        return types.SimpleNamespace(returncode=self.rc, stderr="", stdout="")


@pytest.mark.parametrize("kind,urgency,expire", [
    ("milestone", "normal", "8000"), ("done", "normal", "8000"),
    ("reminder", "normal", "8000"), ("blocked", "critical", "8000"),
    ("alarm", "critical", "8000"), ("question", "critical", "0"),
])
def test_notify_argv_per_kind(kind, urgency, expire):
    argv = notify_argv(kind, "Jarvis", "text here")
    assert argv == ["notify-send", "-a", "Jarvis", "-u", urgency, "-t", expire,
                    "--", "Jarvis", "text here"]


def test_discord_text_format():
    assert discord_text("done", "Jarvis task", "All tests pass, sir.") == \
        "**Jarvis task** — All tests pass, sir."
    q = discord_text("question", "Approval", "Claude wants to push to origin main, sir; shall I?")
    assert q == "**Approval** — Claude wants to push to origin main, sir; shall I? Reply yes or no."
    assert discord_text("question", "A", "Already asked. Reply yes or no.").count("Reply yes or no.") == 1


def test_alerts_fan_out_to_notify_send_and_discord():
    run = Recorder()
    alerts = Alerts(cfg(), run=run)
    fake = FakeDiscord()
    alerts.attach(fake)
    rec = alerts.alert("question", "Approval",
                       "Claude wants to run git push, sir; shall I allow it?", request_id="r1")
    assert alerts.flush(2.0)
    assert run.argv == [(["notify-send", "-a", "Jarvis", "-u", "critical", "-t", "0", "--",
                          "Approval", "Claude wants to run git push, sir; shall I allow it?"], 5.0)]
    assert fake.posts == [("**Approval** — Claude wants to run git push, sir; shall I allow it? "
                           "Reply yes or no.", 5.0)]
    assert rec.request_id == "r1" and rec.toast_ok is True and rec.discord_ok is True
    assert alerts.recent[-1] is rec


def test_alerts_skip_discord_when_unconfigured_or_detached():
    run = Recorder()
    alerts = Alerts(cfg(), run=run)
    alerts.alert("done", "Jarvis", "Finished, sir.")
    assert alerts.flush(2.0)
    assert len(run.argv) == 1
    fake = FakeDiscord(configured=False)
    alerts.attach(fake)
    rec = alerts.alert("milestone", "Jarvis", "Running the tests.")
    assert alerts.flush(2.0)
    assert fake.posts == [] and rec.discord_ok is None and rec.toast_ok is True
    assert len(run.argv) == 2


def test_alert_never_blocks_the_caller():
    run = Recorder(delay=0.3)
    alerts = Alerts(cfg(), run=run)
    alerts.attach(FakeDiscord(delay=0.2))
    t0 = time.monotonic()
    alerts.alert("done", "Jarvis", "slow toast")
    assert time.monotonic() - t0 < 0.05
    assert alerts.flush(3.0)
    assert len(run.argv) == 1


def test_alert_failures_are_logged_and_isolated(caplog):
    run = Recorder(raise_exc=RuntimeError("no dbus " + TOKEN))
    alerts = Alerts(cfg(), run=run)
    fake = FakeDiscord(fail=True)
    alerts.attach(fake)
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        rec = alerts.alert("blocked", "Jarvis", "Claude hit an error.")
        assert alerts.flush(2.0)
        # The worker survives: a second alert still goes out.
        fake.fail = False
        rec2 = alerts.alert("done", "Jarvis", "Recovered.")
        assert alerts.flush(2.0)
    assert rec.toast_ok is False and rec.discord_ok is False
    assert rec2.discord_ok is True
    assert fake.posts == [("**Jarvis** — Recovered.", 5.0)]
    msgs = [r.getMessage() for r in caplog.records]
    assert any("notify-send failed" in m for m in msgs)
    assert any("discord post failed" in m for m in msgs)
    assert all(TOKEN not in m for m in msgs)


def test_alert_nonzero_rc_marks_toast_failed(caplog):
    run = Recorder(rc=1)
    alerts = Alerts(cfg(), run=run)
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        rec = alerts.alert("reminder", "Jarvis", "Sir, this is your reminder.")
        assert alerts.flush(2.0)
    assert rec.toast_ok is False
    assert any("rc=1" in r.getMessage() for r in caplog.records)


def test_unknown_kind_becomes_milestone_and_switches_are_honoured():
    run = Recorder()
    alerts = Alerts(cfg(alerts={"desktop": False, "discord": True}), run=run)
    fake = FakeDiscord()
    alerts.attach(fake)
    rec = alerts.alert("bogus", "Jarvis", "text")
    assert rec.kind == "milestone"
    assert alerts.flush(2.0)
    assert run.argv == []                      # desktop toasts switched off
    assert fake.posts == [("**Jarvis** — text", 5.0)]


def test_default_run_seam_handles_missing_binary(monkeypatch):
    monkeypatch.setattr(nmod, "_missing_warned", False)
    assert nmod._run(["definitely-not-a-binary-jarvis", "x"]) is None


def test_alerts_with_real_channel_object_uses_configured_flag():
    """Alerts + DiscordChannel end-to-end through the fake transport."""
    tr = FakeTransport()
    ch = DiscordChannel(cfg(), lambda t, a: None, transport=tr)
    run = Recorder()
    alerts = Alerts(cfg(), run=run)
    alerts.attach(ch)
    alerts.alert("done", "Jarvis task", "The router tests pass, sir.")
    assert alerts.flush(2.0)
    assert tr.posts() == ["**Jarvis task** — The router tests pass, sir."]
    assert run.argv[0][0][-2:] == ["Jarvis task", "The router tests pass, sir."]


def test_real_assistant_config_drives_configured(tmp_path):
    """The channel reads the real AssistantConfig (W1) through cfg_get:
    placeholders -> unconfigured; filled in -> configured; redacted() masks."""
    from jarvis.assistant_config import AssistantConfig
    acfg = AssistantConfig.load(tmp_path / "assistant.json")
    assert acfg.is_configured("discord") is False
    ch = DiscordChannel(acfg, lambda t, a: None, transport=FakeTransport())
    assert ch.configured is False and ch.start() is False
    acfg.set("discord.bot_token", TOKEN)
    acfg.set("discord.channel_id", CHANNEL)
    acfg.set("discord.user_id", OWNER)
    ch = DiscordChannel(acfg, lambda t, a: None, transport=FakeTransport())
    assert ch.configured is True and acfg.is_configured("discord") is True
    assert ch.channel_id == CHANNEL and ch.user_id == OWNER
    assert TOKEN not in json.dumps(acfg.redacted())
    assert TOKEN not in ch.redacted()
    alerts = Alerts(acfg, run=Recorder())
    assert alerts.desktop_enabled and alerts.discord_enabled


# ================================================================= live
@pytest.mark.skipif(not os.environ.get("JARVIS_LIVE"), reason="JARVIS_LIVE=1 only")
def test_live_gateway_hello_unauthenticated():
    """Read-only, token-free: the public gateway endpoint and a real
    websocket HELLO through UrllibTransport. Never identifies."""
    tr = dmod.UrllibTransport()
    r = tr.request("GET", f"{dmod.API_BASE}/gateway", {"User-Agent": dmod.USER_AGENT})
    assert r.status == 200
    url = gateway_url(r.json()["url"])
    assert url.startswith("wss://")

    async def hello():
        async with tr.connect(url) as ws:
            frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            await ws.close()
            return frame
    frame = asyncio.run(hello())
    assert frame["op"] == 10
    assert frame["d"]["heartbeat_interval"] > 1000
    # A placeholder token is refused cleanly (401 -> FatalGateway path).
    r2 = tr.request("GET", f"{dmod.API_BASE}/gateway/bot",
                    {"Authorization": "Bot placeholder", "User-Agent": dmod.USER_AGENT})
    assert r2.status == 401
