"""The phone client: Jarvis in a browser on the home Wi-Fi, and nowhere else.

He asked for "an app that gets me to him". This is that app -- a single
page served by the Spark to his phone, off the same `dispatch_text` the
typed box and the `jarvis` CLI use. There is no cloud leg, no account, no
push service and no tunnel: the page is 40 KB of HTML that the Spark hands
to a device on the same LAN, and every answer comes back over the one TCP
connection that asked for it.

Why an http.server and not a framework: this is stdlib only by rule, and
the shape is genuinely small -- five endpoints, one of which is static.
`ThreadingHTTPServer` gives one thread per request, which is exactly what
the command socket already does (a turn is dispatched on the client's own
thread), so the concurrency story is the one this codebase already has.

The turn itself is NOT re-implemented here. `jarvis/cmdsock.py` worked out
the hard part -- there is no turn-completion signal on the bus, so a turn's
end has to be inferred from five conditions -- and this module drives that
same loop through ``cmdsock.stream_turn(send=lines.append, ...)``. A second
copy of that heuristic would drift out of step with the first one within a
month. Likewise the push-to-talk clip goes to ``intercom.text_from_clip``,
which already decodes, resamples, speaker-gates and transcribes.

``source="phone"`` is one of app.SOCKET_SOURCES, which buys three rules
that matter here and are not the typed box's:

* replies are stamped with a turn_id, so two devices asking at once do not
  read each other's answers;
* ``quiet`` mutes THAT turn's speech -- a question asked from bed does not
  make the soundbar answer the house. Speaking aloud is a toggle on the
  page, off by default, exactly like the intercom's ``--speak``;
* it never barges in on a reply he is being spoken in the room.

SECURITY -- the reason this file is careful rather than short
------------------------------------------------------------
It is off until he turns it on (``phone.enabled``, false in DEFAULTS), and
when it starts:

* **It binds to ONE private address.** Not 0.0.0.0, not "::" -- a single
  LAN address, and ``start()`` refuses outright if that address is not in a
  private range (RFC1918 / CGNAT / link-local / loopback). A wildcard bind
  is the difference between "my phone can reach it" and "the coffee shop
  can", and the check is not skippable by configuration.
* **Every acting endpoint needs a bearer token**, compared with
  ``hmac.compare_digest``. The token is 32 bytes of ``secrets`` generated on
  first enable and stored in the assistant config, which is already 0600,
  with ``phone.token`` in SECRET_KEYS so it is masked out of logs and
  ``repr(cfg)``. No ``/api/*`` route does anything without it.
* **Only private peers.** A connection whose remote address is not itself
  private is refused before routing -- belt to the bind's braces.
* **Rate limited and capped.** A sliding window per client address (and a
  tighter one for audio), a hard cap on the request body, and a
  ``Content-Length`` requirement so a body can be refused before it is read.
* **LAN only, by design.** There is deliberately no tunnel, no
  port-forward, no UPnP and no public-hostname support anywhere in this
  file, and the docs do not describe one. Reaching Jarvis from outside the
  house is a separate decision with a separate threat model, and it is his
  to make, not this module's to quietly enable.

The page shell (``GET /``) is static markup and is served without a token
on purpose: it holds no data and cannot act, and requiring one there would
break "Add to Home Screen" on Android, where the launcher opens the
manifest's ``start_url`` and drops the query string. The page finds its key
in ``?t=`` or in localStorage and puts it in an ``Authorization`` header.

THE MICROPHONE, HONESTLY
------------------------
``getUserMedia`` is gated on a secure context. ``http://192.168.x.x`` is not
one -- in Safari, in Chrome, in Firefox -- so on his iPhone the browser does
not merely refuse the microphone, ``navigator.mediaDevices`` is not even
defined. No amount of code here changes that.

So the TEXT path is the product: it is first in the layout, it is what the
quick buttons drive, and it works on every device on the LAN today. The
push-to-talk button ships wired to a working ``/api/voice`` endpoint, and
when the page is not in a secure context it renders visibly disabled with
the reason written under it and what would fix it -- never a button that
looks alive and silently does nothing. It comes alive by itself the day the
page is reached over https (a cert he trusts) or from the Spark's own
``http://127.0.0.1:8765``, which IS a secure context.

One more format note for that day: ``MediaRecorder`` prefers
``audio/webm;codecs=opus``, and libsndfile cannot read a WebM container.
The page asks for ``audio/ogg;codecs=opus`` first (Firefox gives it) and
falls back; a container Jarvis cannot read comes back as intercom's own
"send wav, ogg, opus or flac" line rather than a silent failure.
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import math
import os
import secrets
import socket
import struct
import threading
import time
import uuid
import zlib
from collections import deque
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from jarvis import cmdsock, intercom
from jarvis.events import Status, UserUtterance, bus
from jarvis.logs import get_logger

log = get_logger("webapp")

SOURCE = "phone"                  # one of jarvis/app.py's SOCKET_SOURCES
DEFAULT_PORT = 8765
DEFAULT_TIMEOUT_S = 60.0          # shorter than the CLI's 90: a phone waiting
MAX_TIMEOUT_S = 180.0             # on a spinner is a worse place to wait
MAX_TEXT_BYTES = 16 * 1024        # a typed question, with room to spare
DEFAULT_MAX_AUDIO_MB = 8
# A sliding window per client address. Sized for a person tapping, not for a
# script: the point is that a stray loop (or a wedged page) cannot pin the
# resident Whisper or the local model, not to police his own thumbs.
RATE_WINDOW_S = 60.0
RATE_MAX = 60
RATE_MAX_AUDIO = 12
RATE_MAX_CLIENTS = 64             # the LAN is small; do not grow unbounded
SHUTDOWN_POLL_S = 0.4
# How far past the cap an over-long body is read-and-dropped so that the
# 413 can actually reach the sender (see _Handler._refuse). Nothing is
# retained; this is a ceiling on wasted socket reads, not on memory.
DRAIN_SLACK_BYTES = 2 * 1024 * 1024

NOT_PRIVATE_LINE = ("the phone client refuses to bind %r: it is not a "
                    "private address. This is a LAN-only feature.")
LINK_FILE_DEFAULT = "~/jarvis-phone.txt"
QR_FILE_DEFAULT = "~/jarvis-phone.svg"


# ------------------------------------------------------------- addresses
def is_private_host(host: str) -> bool:
    """True for an address a home network can own.

    The rule is ``not is_global`` rather than a hand-written list of
    ranges, because the hand-written list gets it wrong: ``is_private`` is
    False for 100.64.0.0/10 (carrier-grade NAT, which some home routers do
    hand out) and True for 203.0.113.0/24. "Not routable on the public
    internet" is the property actually wanted here, and that is exactly
    what ``is_global`` answers, from IANA's own registry.

    A hostname is refused rather than resolved: "jarvis.local" that
    happens to resolve to a public record would sail straight past a check
    that resolved first and asked afterwards. So would a wildcard, which
    is the whole difference between "my phone can reach it" and "the
    coffee shop can"."""
    text = (host or "").strip()
    if not text or text in ("0.0.0.0", "::", "*"):        # noqa: S104 - refused
        return False
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        return False
    if addr.is_unspecified or addr.is_multicast:
        return False
    return not addr.is_global


def lan_address() -> str:
    """This box's address on the network its default route leaves by.

    A UDP socket that is *connected* and never written asks the kernel
    which source address it would use; no packet is sent and nothing is
    resolved. TEST-NET-1 (RFC 5737) is the target because it is guaranteed
    never to be a real host. The answer is only returned when it is
    private -- on a box whose default route is a public interface this
    returns "" and start() refuses, which is the correct outcome."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))
        addr = s.getsockname()[0]
    except OSError:
        log.debug("no default route; cannot guess the LAN address",
                  exc_info=True)
        return ""
    finally:
        s.close()
    return addr if is_private_host(addr) else ""


def new_token() -> str:
    """32 bytes of urlsafe randomness: fine in a URL and in a QR."""
    return secrets.token_urlsafe(32)


# ----------------------------------------------------------- rate limits
class RateLimiter:
    """Sliding window per client address, two buckets (all / audio).

    Not a security boundary -- the token is -- but it is what keeps a page
    stuck in a retry loop from queueing forty Whisper decodes."""

    def __init__(self, window_s: float = RATE_WINDOW_S, limit: int = RATE_MAX,
                 audio_limit: int = RATE_MAX_AUDIO):
        self.window_s = float(window_s)
        self.limit = int(limit)
        self.audio_limit = int(audio_limit)
        self._lock = threading.Lock()
        self._hits: dict[str, deque] = {}
        self._audio: dict[str, deque] = {}

    def _bucket(self, store: dict, key: str, limit: int, now: float) -> bool:
        dq = store.get(key)
        if dq is None:
            if len(store) >= RATE_MAX_CLIENTS:
                store.clear()          # a LAN cannot legitimately have 64
            dq = store[key] = deque()
        while dq and now - dq[0] > self.window_s:
            dq.popleft()
        if len(dq) >= limit:
            return False
        dq.append(now)
        return True

    def allow(self, key: str, audio: bool = False) -> bool:
        now = time.monotonic()
        with self._lock:
            if not self._bucket(self._hits, key, self.limit, now):
                return False
            if audio and not self._bucket(self._audio, key, self.audio_limit,
                                          now):
                return False
        return True


# ------------------------------------------------------------ the server
class _Server(ThreadingHTTPServer):
    """ThreadingHTTPServer without the reverse-DNS lookup at bind time.

    ``HTTPServer.server_bind`` calls ``socket.getfqdn(host)`` purely to
    fill in ``server_name``, which nothing here reads (it is for CGI). On
    a box whose resolver is slow or absent that lookup blocks -- and this
    binds from ``JarvisApp.start_assistant``, on the startup path, so a
    stalled resolver would be a stalled boot. The address is the name."""

    daemon_threads = True

    def server_bind(self):
        import socketserver

        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class PhoneServer:
    """The LAN HTTP server. start()/stop() like every other app thread."""

    def __init__(self, app, cfg=None, timeout_s: float = DEFAULT_TIMEOUT_S,
                 idle_grace_s: float = cmdsock.IDLE_GRACE_S):
        self.app = app
        self.cfg = cfg if cfg is not None else getattr(app, "assistant", None)
        self.timeout_s = float(timeout_s)
        self.idle_grace_s = float(idle_grace_s)
        self.token = ""
        self.host = ""
        self.port = 0
        self.served = 0
        self.limiter = RateLimiter()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -------------------------------------------------------- config reads
    def _get(self, key: str, default=None):
        if self.cfg is None:
            return default
        try:
            return self.cfg.get(f"phone.{key}", default)
        except Exception:                 # noqa: BLE001 - config boundary
            log.debug("phone config %s unreadable", key, exc_info=True)
            return default

    @property
    def enabled(self) -> bool:
        return bool(self._get("enabled", False))

    @property
    def max_audio_bytes(self) -> int:
        """The page's own cap, never above the intercom's own."""
        try:
            mb = float(self._get("max_audio_mb", DEFAULT_MAX_AUDIO_MB))
        except (TypeError, ValueError):
            mb = DEFAULT_MAX_AUDIO_MB
        return max(1, min(int(mb * 1048576), intercom.MAX_AUDIO_BYTES))

    @property
    def url(self) -> str:
        if not (self.host and self.port):
            return ""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}/"

    @property
    def link(self) -> str:
        """The URL with the key in it -- what goes in the file and the QR."""
        return f"{self.url}?t={self.token}" if self.url and self.token else ""

    @property
    def running(self) -> bool:
        return self._httpd is not None

    # ---------------------------------------------------------- lifecycle
    def start(self) -> bool:
        """Bind and serve, or explain in the log why not. Idempotent.

        Every refusal is a log line at warning or above: a phone client
        that silently did not come up is indistinguishable from a phone
        with bad Wi-Fi, and he would debug the wrong end of the house."""
        if self.running:
            return True
        if not self.enabled:
            log.info("phone client is off (phone.enabled false)")
            return False
        token = str(self._get("token", "") or "")
        if not token:
            token = new_token()
            if self.cfg is None or not self.cfg.set("phone.token", token):
                log.warning("phone client: the key could not be saved; it "
                            "will change on the next restart")
            else:
                log.info("phone client: generated a new key")
        self.token = token
        host = str(self._get("bind", "") or "").strip() or lan_address()
        if not host:
            log.warning("phone client: no private LAN address found on this "
                        "box; not starting")
            return False
        if not is_private_host(host):
            # The one refusal that is never overridable from the config.
            log.error(NOT_PRIVATE_LINE, host)
            bus.publish(Status(text="phone client refused a public bind "
                                    "address", kind="error"))
            return False
        # Port 0 is meaningful (let the kernel choose, which is how the tests
        # bind), so it cannot go through an `or DEFAULT_PORT` -- that reads
        # a deliberate 0 as "unset" and would quietly bind 8765 instead.
        raw = self._get("port", DEFAULT_PORT)
        try:
            port = DEFAULT_PORT if raw is None or raw == "" else int(raw)
        except (TypeError, ValueError):
            port = DEFAULT_PORT
        if not 0 <= port <= 65535:
            log.warning("phone client: port %r is out of range; using %d",
                        raw, DEFAULT_PORT)
            port = DEFAULT_PORT
        try:
            httpd = _Server((host, port), _Handler)
        except OSError as exc:
            log.warning("phone client could not bind %s:%s (%s)", host, port,
                        exc)
            return False
        httpd.phone = self                       # the handler's way back
        self.host, self.port = host, int(httpd.server_address[1])
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": SHUTDOWN_POLL_S},
            name="webapp", daemon=True)
        self._thread.start()
        log.info("phone client listening on %s (LAN only)", self.url)
        self.write_link_file()
        return True

    def stop(self):
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            try:
                httpd.shutdown()             # stops serve_forever's loop
            except Exception:                # noqa: BLE001 - teardown
                log.debug("phone client shutdown raised", exc_info=True)
            try:
                httpd.server_close()
            except OSError:
                pass
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
        self._thread = None

    # ---------------------------------------------------------- the link
    def write_link_file(self) -> Optional[Path]:
        """Leave the URL, the key and a QR where only he can read them.

        0600 under his home directory, because the URL carries the key.
        Best effort: a home directory that cannot be written is a reason to
        log, not a reason to refuse the phone client."""
        raw = str(self._get("link_file", LINK_FILE_DEFAULT) or "")
        if not raw:
            return None
        path = Path(raw).expanduser()
        qr_path = self._write_qr()
        body = [
            "Jarvis on your phone",
            "=" * 20,
            "",
            f"  {self.link}",
            "",
        ]
        if qr_path is not None:
            body += [f"QR (open it and point the camera): {qr_path}", ""]
        body += [
            "Open that in Safari on the phone, then Share -> Add to Home",
            "Screen. It only answers on the home Wi-Fi.",
            "",
            "The link contains the key, so this file is yours alone (0600).",
            "To rotate it: delete phone.token from",
            "  ~/.config/jarvis/assistant.json",
            "and restart Jarvis.",
            "",
            "Push-to-talk is off over plain http -- browsers only hand a page",
            "the microphone on a secure origin. The text box is the way in.",
            "",
        ]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(body), encoding="utf-8")
            os.chmod(path, 0o600)
        except OSError:
            log.warning("phone client: could not write %s", path, exc_info=True)
            return None
        log.info("phone client: the link is in %s", path)
        return path

    def _write_qr(self) -> Optional[Path]:
        raw = str(self._get("qr_file", QR_FILE_DEFAULT) or "")
        if not raw or not self.link:
            return None
        path = Path(raw).expanduser()
        try:
            svg = qr_svg(self.link)
        except ValueError:
            log.info("phone client: the link is too long for a QR; the URL "
                     "in the link file is the way in")
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(svg, encoding="utf-8")
            os.chmod(path, 0o600)
        except OSError:
            log.warning("phone client: could not write %s", path, exc_info=True)
            return None
        return path

    # ----------------------------------------------------------- the turn
    def authorized(self, given: str) -> bool:
        """Constant-time compare against the live key."""
        if not self.token or not given:
            return False
        return hmac.compare_digest(str(given).encode("utf-8"),
                                   self.token.encode("utf-8"))

    def ask(self, text: str, speak: bool = False,
            timeout: Optional[float] = None) -> dict:
        """One phone turn -> ``{"messages": [...], "reason": ...}``.

        The bus events of the turn are collected by cmdsock's own
        ReplyCollector (stamped with this turn's id) and drained by
        cmdsock's own stream_turn -- the heuristics that know when a turn
        is over live in exactly one place, and that place is not here."""
        text = (text or "").strip()
        if not text:
            return {"messages": [], "reason": "empty"}
        limit = max(1.0, min(float(timeout or self.timeout_s), MAX_TIMEOUT_S))
        # The two READS, kept off the dispatch path for the reason the
        # command socket keeps them off it: they are a look at app state,
        # not a turn, so they are not remembered as an exchange and do not
        # wake the speaker.
        low = text.lower()
        if low in ("status", "diagnostics"):
            return self._read("diagnostics_text", "diagnostics unavailable")
        if low in ("board", "the board"):
            return self._read("board_text", "board unavailable")
        turn_id = uuid.uuid4().hex[:12]
        lines: list[dict] = []
        log.info("phone: %r%s", text, "" if speak else " (quiet)")
        with cmdsock.ReplyCollector(turn_id) as col:
            bus.publish(UserUtterance(text=text, source=SOURCE))
            result = self.app.dispatch_text(text, source=SOURCE,
                                            quiet=not speak, turn_id=turn_id)
            reason = cmdsock.stream_turn(lines.append, col, result, limit,
                                         self.idle_grace_s)
        self.served += 1
        return {"messages": lines, "reason": reason}

    def _read(self, attr: str, fallback: str) -> dict:
        fn = getattr(self.app, attr, None)
        line = fn() if callable(fn) else fallback
        self.served += 1
        return {"messages": [{"kind": "reply", "text": line, "speak": False}],
                "reason": "done"}

    def voice(self, raw: bytes, speak: bool = False,
              timeout: Optional[float] = None) -> dict:
        """A push-to-talk clip -> the transcript, then the same turn.

        The decode, the resample, the speaker gate and Whisper are
        intercom's; nothing about audio is re-implemented here."""
        text = intercom.text_from_clip(self.app, raw)   # IntercomError -> caller
        out = self.ask(text, speak=speak, timeout=timeout)
        out["heard"] = text
        return out


# ---------------------------------------------------------- the handler
class _Handler(BaseHTTPRequestHandler):
    server_version = "JarvisPhone/1"
    protocol_version = "HTTP/1.1"       # keep-alive: every reply sets a length
    # HTTP/1.1 means connections are held open between taps, and a thread
    # is held with each. This is the ceiling on an IDLE one: a phone that
    # walks out of range must not park a thread until the app restarts.
    # It is a per-read timeout, not a budget for the turn, so a 60 s answer
    # is unaffected.
    timeout = 30

    # -------------------------------------------------------- plumbing
    @property
    def phone(self) -> PhoneServer:
        return self.server.phone         # set in PhoneServer.start

    def log_message(self, fmt, *args):
        """BaseHTTPRequestHandler writes to stderr; this app has a log."""
        log.debug("phone %s - %s", self.address_string(), fmt % args)

    def _peer(self) -> str:
        try:
            return str(self.client_address[0])
        except Exception:                 # noqa: BLE001 - odd address families
            return ""

    def _send(self, code: int, body: bytes, ctype: str,
              cache: str = "no-store", extra: Optional[dict] = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        # No third-party anything is loaded by this page, so the strictest
        # policy that still allows the inline script and style is free.
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; script-src 'unsafe-inline'; "
                         "style-src 'unsafe-inline'; img-src 'self' data:; "
                         "connect-src 'self'; manifest-src 'self'; "
                         "form-action 'none'; base-uri 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except OSError:
                log.debug("phone client went away mid-reply", exc_info=True)

    def _json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, code: int, text: str):
        self._json(code, {"error": text})

    def _refuse(self, code: int, text: str, drain: int = 0):
        """An error raised BEFORE the request body was read.

        Two things have to happen and they pull against each other. The
        unread body is still on the socket, so the connection cannot be
        reused -- on keep-alive the next request would parse somebody's
        audio as a request line; hence "Connection: close", which also
        sets close_connection on the handler.

        But closing on a client that is still uploading breaks its pipe
        BEFORE it reads the reply, so the phone shows "network error"
        where it should say "that clip is too large". So a bounded amount
        of the body is read and thrown away first (`drain`), which costs
        no memory -- the chunks are never kept -- and buys an honest
        message for the realistic case of a slightly-too-long clip. A body
        past that budget is still cut off unread, which is the point of
        having a cap at all."""
        self._discard(drain)
        self._send(code, json.dumps({"error": text}).encode("utf-8"),
                   "application/json; charset=utf-8",
                   extra={"Connection": "close"})

    def _discard(self, budget: int):
        """Read and drop up to `budget` bytes of the pending body."""
        left = int(budget)
        while left > 0:
            try:
                chunk = self.rfile.read(min(65536, left))
            except OSError:
                return
            if not chunk:
                return
            left -= len(chunk)

    def _token(self) -> str:
        header = self.headers.get("Authorization", "") or ""
        if header.startswith("Bearer "):
            return header[7:].strip()
        query = parse_qs(urlparse(self.path).query)
        return (query.get("t") or [""])[0]

    def _gate(self, audio: bool = False) -> bool:
        """Peer / token / rate: every acting endpoint starts here."""
        peer = self._peer()
        if not is_private_host(peer):
            log.warning("phone client refused a non-private peer %r", peer)
            self._refuse(403, "forbidden")
            return False
        if not self.phone.authorized(self._token()):
            log.info("phone client: bad or missing key from %s", peer)
            self._refuse(401, "a key is required")
            return False
        if not self.phone.limiter.allow(peer, audio=audio):
            log.info("phone client: rate limit hit by %s", peer)
            self._refuse(429, "too many requests; give it a moment")
            return False
        return True

    def _body(self, cap: int) -> Optional[bytes]:
        """Read the body, or answer and return None.

        The length is checked BEFORE the read: refusing a 200 MB upload
        after receiving it has already cost the memory it was refused
        for."""
        raw = self.headers.get("Content-Length")
        if raw is None:
            self._refuse(411, "a Content-Length is required")
            return None
        try:
            length = int(raw)
        except ValueError:
            self._refuse(400, "bad Content-Length")
            return None
        if length < 0:
            self._refuse(400, "bad Content-Length")
            return None
        if length > cap:
            # Drain a little past the cap so a clip that is merely a bit
            # too long still gets the 413 read back to it (see _refuse).
            self._refuse(413, f"too large (the cap is {cap // 1024} KB)",
                         drain=min(length, cap + DRAIN_SLACK_BYTES))
            return None
        try:
            return self.rfile.read(length)
        except OSError:
            log.debug("phone client dropped mid-body", exc_info=True)
            return None

    # ----------------------------------------------------------- routes
    def do_GET(self):                                       # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            # Static markup: no data, and it cannot act -- every /api route
            # below asks for the key. Serving it unauthenticated is what
            # makes "Add to Home Screen" work on Android, where the launcher
            # opens the manifest's start_url and drops the query string.
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/manifest.webmanifest":
            self._send(200, MANIFEST.encode("utf-8"),
                       "application/manifest+json; charset=utf-8",
                       cache="public, max-age=86400")
            return
        if path in ("/icon-180.png", "/apple-touch-icon.png"):
            self._send(200, icon_png(180), "image/png",
                       cache="public, max-age=86400")
            return
        if path == "/icon-512.png":
            self._send(200, icon_png(512), "image/png",
                       cache="public, max-age=86400")
            return
        if path == "/api/ping":
            if not self._gate():
                return
            self._json(200, {"ok": True, "mic_note": MIC_NOTE})
            return
        self._error(404, "no such page")

    def do_HEAD(self):                                      # noqa: N802
        self.do_GET()

    def do_POST(self):                                      # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/ask":
            self._ask()
        elif path == "/api/voice":
            self._voice()
        else:
            self._error(404, "no such endpoint")

    def _ask(self):
        if not self._gate():
            return
        body = self._body(MAX_TEXT_BYTES)
        if body is None:
            return
        try:
            msg = json.loads(body.decode("utf-8", "replace"))
            if not isinstance(msg, dict):
                raise ValueError("not an object")
        except ValueError as exc:
            self._error(400, f"malformed request: {exc}")
            return
        text = str(msg.get("text") or "").strip()
        if not text:
            self._error(400, "empty request")
            return
        try:
            out = self.phone.ask(text, speak=bool(msg.get("speak", False)),
                                 timeout=_timeout(msg.get("timeout")))
        except Exception:                 # noqa: BLE001 - answer, never 500-silent
            log.exception("phone ask failed")
            self._error(500, "Jarvis could not handle that")
            return
        self._json(200, out)

    def _voice(self):
        if not self._gate(audio=True):
            return
        # The clip arrives as the raw body (MediaRecorder's Blob), not as
        # base64: the phone is on Wi-Fi and a third more bytes for nothing
        # is a third more seconds before he hears anything.
        body = self._body(self.phone.max_audio_bytes)
        if body is None:
            return
        query = parse_qs(urlparse(self.path).query)
        speak = (query.get("speak") or ["0"])[0] in ("1", "true", "yes")
        try:
            out = self.phone.voice(body, speak=speak)
        except intercom.IntercomError as exc:
            log.info("phone intercom: %s (%s)", exc.text, exc.kind)
            self._error(400, exc.text)
            return
        except Exception:                 # noqa: BLE001 - answer, never 500-silent
            log.exception("phone voice failed")
            self._error(500, "Jarvis could not handle that clip")
            return
        self._json(200, out)


def _timeout(value) -> float:
    try:
        return max(1.0, min(float(value), MAX_TIMEOUT_S))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S


# ------------------------------------------------------------- the icon
@lru_cache(maxsize=4)
def icon_png(size: int = 180) -> bytes:
    """An arc-reactor roundel as a PNG, drawn here rather than shipped.

    Pure stdlib (zlib + struct): an 8-bit RGB image, one unfiltered
    scanline per row. iOS masks an apple-touch-icon into a rounded square
    itself, so the art is a full bleed square and there is no alpha to get
    wrong."""
    size = max(16, min(int(size), 1024))
    centre = (size - 1) / 2.0
    radius = size / 2.0
    rows = bytearray()
    for y in range(size):
        rows.append(0)                                   # filter type: none
        dy = y - centre
        for x in range(size):
            d = math.hypot(x - centre, dy) / radius
            if d < 0.14:
                rows += b"\xe8\xfb\xff"                  # the core
            elif 0.30 <= d < 0.38:
                rows += b"\x2f\xa8\xc4"                  # inner ring
            elif 0.56 <= d < 0.70:
                rows += b"\x4d\xe0\xf5"                  # outer ring
            elif d < 0.94:
                rows += b"\x07\x1a\x26"                  # the glass
            else:
                rows += b"\x04\x0d\x14"                  # the bezel
    return _png_chunks(size, bytes(rows))


def _png_chunks(size: int, raw: bytes) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)   # 8-bit RGB
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


# --------------------------------------------------------------- the QR
# A byte-mode QR, error correction L, versions 1-6 -- enough for a LAN URL
# with a 43-character key (~70 bytes; version 4 holds 78). Written out
# rather than imported because the rule for this feature is no new
# dependency, and a QR is the difference between typing a 43-character key
# on a phone keyboard and pointing the camera at the screen.
_QR_CAPACITY_L = {1: 17, 2: 32, 3: 53, 4: 78, 5: 106, 6: 134}
_QR_TOTAL_CODEWORDS = {1: 26, 2: 44, 3: 70, 4: 100, 5: 134, 6: 172}
_QR_EC_CODEWORDS_L = {1: 7, 2: 10, 3: 15, 4: 20, 5: 26, 6: 18}
_QR_EC_BLOCKS_L = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2}
_QR_ALIGN = {1: (), 2: (6, 18), 3: (6, 22), 4: (6, 26), 5: (6, 30), 6: (6, 34)}
# Mask 0 with EC level L, already BCH-encoded and XOR-masked (Table C.1).
_QR_FORMAT_L0 = 0b111011111000100


def _gf_tables():
    exp, logt = [0] * 512, [0] * 256
    x = 1
    for i in range(255):
        exp[i] = x
        logt[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11D
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, logt


_GF_EXP, _GF_LOG = _gf_tables()


def _rs_generator(n: int) -> list:
    poly = [1]
    for i in range(n):
        nxt = [0] * (len(poly) + 1)
        for j, coef in enumerate(poly):
            nxt[j] ^= coef
            nxt[j + 1] ^= _GF_EXP[(_GF_LOG[coef] + i) % 255] if coef else 0
        poly = nxt
    return poly


def _rs_remainder(data: bytes, n: int) -> list:
    gen = _rs_generator(n)
    rem = [0] * n
    for byte in data:
        factor = byte ^ rem[0]
        rem = rem[1:] + [0]
        if factor:
            lf = _GF_LOG[factor]
            for i, g in enumerate(gen[1:]):
                rem[i] ^= _GF_EXP[(_GF_LOG[g] + lf) % 255] if g else 0
    return rem


def _qr_matrix(text: str):
    """The module grid for `text` as a list of lists of 0/1."""
    data = text.encode("utf-8")
    version = next((v for v in sorted(_QR_CAPACITY_L)
                    if len(data) <= _QR_CAPACITY_L[v]), None)
    if version is None:
        raise ValueError("too long for a version-6 QR")
    size = 17 + 4 * version
    total = _QR_TOTAL_CODEWORDS[version]
    ec_per_block = _QR_EC_CODEWORDS_L[version]
    blocks = _QR_EC_BLOCKS_L[version]
    data_total = total - ec_per_block * blocks

    # -- bitstream: mode 0100, 8-bit length, the bytes, terminator, pad
    bits: list[int] = [0, 1, 0, 0]
    for i in range(7, -1, -1):
        bits.append((len(data) >> i) & 1)
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    bits += [0] * min(4, data_total * 8 - len(bits))
    bits += [0] * (-len(bits) % 8)
    pad = (0xEC, 0x11)
    words = bytearray(int("".join(str(b) for b in bits[i:i + 8]), 2)
                      for i in range(0, len(bits), 8))
    while len(words) < data_total:
        words.append(pad[(len(words) - len(bits) // 8) % 2])

    # -- interleave the blocks (one block for v1-5; two for v6-L)
    per, extra = divmod(data_total, blocks)
    chunks, at = [], 0
    for b in range(blocks):
        n = per + (1 if b >= blocks - extra else 0)
        chunks.append(bytes(words[at:at + n]))
        at += n
    ecs = [_rs_remainder(c, ec_per_block) for c in chunks]
    stream = bytearray()
    for i in range(max(len(c) for c in chunks)):
        for c in chunks:
            if i < len(c):
                stream.append(c[i])
    for i in range(ec_per_block):
        for e in ecs:
            stream.append(e[i])

    grid = [[None] * size for _ in range(size)]

    def finder(row: int, col: int):
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                r, c = row + dr, col + dc
                if not (0 <= r < size and 0 <= c < size):
                    continue
                edge = dr in (0, 6) or dc in (0, 6)
                core = 2 <= dr <= 4 and 2 <= dc <= 4
                inside = 0 <= dr <= 6 and 0 <= dc <= 6
                grid[r][c] = 1 if inside and (edge or core) else 0

    finder(0, 0)
    finder(0, size - 7)
    finder(size - 7, 0)
    for i in range(8, size - 8):                          # timing patterns
        grid[6][i] = grid[i][6] = 1 - (i % 2)
    for r in _QR_ALIGN[version]:                          # alignment patterns
        for c in _QR_ALIGN[version]:
            if grid[r][c] is not None:
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    grid[r + dr][c + dc] = 1 if max(abs(dr), abs(dc)) != 1 else 0
    grid[size - 8][8] = 1                                 # the dark module
    fmt = _QR_FORMAT_L0
    for i in range(15):                                   # format information
        bit = (fmt >> i) & 1
        if i < 6:
            grid[i][8] = bit
        elif i == 6:
            grid[7][8] = bit
        elif i == 7:
            grid[8][8] = bit
        elif i == 8:
            grid[8][7] = bit
        else:
            grid[8][14 - i] = bit
        if i < 8:
            grid[8][size - 1 - i] = bit
        else:
            grid[size - 15 + i][8] = bit

    # -- the data, up the two-module columns, mask 0 ((r + c) % 2 == 0)
    bit_at = 0
    stream_bits = len(stream) * 8
    col = size - 1
    upward = True
    while col > 0:
        if col == 6:                                      # skip the timing col
            col -= 1
        for step in range(size):
            row = size - 1 - step if upward else step
            for c in (col, col - 1):
                if grid[row][c] is not None:
                    continue
                bit = 0
                if bit_at < stream_bits:
                    bit = (stream[bit_at >> 3] >> (7 - (bit_at & 7))) & 1
                    bit_at += 1
                grid[row][c] = bit ^ (1 if (row + c) % 2 == 0 else 0)
        upward = not upward
        col -= 2
    return [[int(v or 0) for v in row] for row in grid]


def qr_svg(text: str, scale: int = 8, quiet: int = 4) -> str:
    """`text` as a scannable QR, as an SVG document.

    SVG rather than PNG because it is exact at any zoom -- he will point a
    camera at whatever size the file happens to open."""
    grid = _qr_matrix(text)
    n = len(grid)
    side = (n + quiet * 2) * scale
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" '
             f'height="{side}" viewBox="0 0 {side} {side}" '
             f'shape-rendering="crispEdges">',
             f'<rect width="{side}" height="{side}" fill="#ffffff"/>',
             '<g fill="#000000">']
    for r, row in enumerate(grid):
        c = 0
        while c < n:
            if not row[c]:
                c += 1
                continue
            run = c
            while run < n and row[run]:
                run += 1
            x = (c + quiet) * scale
            y = (r + quiet) * scale
            parts.append(f'<rect x="{x}" y="{y}" width="{(run - c) * scale}" '
                         f'height="{scale}"/>')
            c = run
    parts += ["</g>", "</svg>", ""]
    return "\n".join(parts)


# ------------------------------------------------------------ the page
MIC_NOTE = ("Push-to-talk needs a secure origin. This page is plain http on "
            "your Wi-Fi, so the browser will not hand it the microphone -- "
            "on any browser, not just Safari. The text box works everywhere.")

MANIFEST = json.dumps({
    "name": "Jarvis",
    "short_name": "Jarvis",
    "description": "Jarvis, on the home network.",
    "start_url": ".",
    "scope": "/",
    "display": "standalone",
    "orientation": "portrait",
    "background_color": "#050d14",
    "theme_color": "#050d14",
    "icons": [
        {"src": "/icon-180.png", "sizes": "180x180", "type": "image/png"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png",
         "purpose": "any maskable"},
    ],
}, indent=2)

# The quick taps. Each is (label, the sentence actually dispatched) -- the
# sentence is a real phrase off docs/capabilities.md, so a button and the
# spoken form take exactly the same path through the commander.
QUICK_TAPS = (
    ("What's due", "what's due this week"),
    ("Next exam", "when's my next exam"),
    ("Calendar", "what's on my calendar today"),
    ("Briefing", "brief me"),
    ("Timer 10m", "set a timer for 10 minutes"),
    ("Diagnostics", "diagnostics"),
    ("Board", "board"),
    ("Lights up", "lights up"),
    ("Lights down", "lights down"),
)

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1,
      viewport-fit=cover, maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Jarvis">
<meta name="theme-color" content="#050d14">
<meta name="referrer" content="no-referrer">
<title>Jarvis</title>
<link rel="manifest" href="/manifest.webmanifest">
<link rel="apple-touch-icon" href="/icon-180.png">
<link rel="icon" href="/icon-180.png">
<style>
  :root {
    --bg: #050d14; --slab: #0b1a25; --line: #17364a;
    --ink: #dbeaf2; --dim: #7b98aa; --cyan: #4de0f5; --warn: #f0b849;
    --bad: #e8695f;
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg); color: var(--ink);
    font: 16px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
          sans-serif;
    display: flex; flex-direction: column;
    padding: env(safe-area-inset-top) env(safe-area-inset-right)
             env(safe-area-inset-bottom) env(safe-area-inset-left);
  }
  header {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px 10px; border-bottom: 1px solid var(--line);
  }
  .mark {
    width: 22px; height: 22px; border-radius: 50%;
    border: 2px solid var(--cyan); position: relative; flex: none;
    box-shadow: 0 0 12px rgba(77, 224, 245, .45);
  }
  .mark::after {
    content: ""; position: absolute; inset: 5px; border-radius: 50%;
    background: var(--cyan);
  }
  h1 { font-size: 15px; letter-spacing: .16em; text-transform: uppercase;
       margin: 0; font-weight: 600; flex: 1; }
  #conn { font-size: 11px; color: var(--dim); letter-spacing: .08em; }
  #conn.bad { color: var(--bad); }
  #log { flex: 1; overflow-y: auto; padding: 14px 16px 4px;
         -webkit-overflow-scrolling: touch; }
  .msg { max-width: 86%; margin: 0 0 10px; padding: 9px 13px;
         border-radius: 14px; white-space: pre-wrap; word-wrap: break-word; }
  .me { margin-left: auto; background: #123249; border-bottom-right-radius: 4px; }
  .him { background: var(--slab); border: 1px solid var(--line);
         border-bottom-left-radius: 4px; }
  .note { font-size: 12px; color: var(--dim); margin: 0 0 10px;
          letter-spacing: .02em; }
  .note.err { color: var(--bad); }
  .heard { font-size: 12px; color: var(--dim); font-style: italic; }
  #taps { display: flex; gap: 8px; overflow-x: auto; padding: 8px 16px;
          border-top: 1px solid var(--line); }
  #taps button {
    flex: none; background: var(--slab); color: var(--ink);
    border: 1px solid var(--line); border-radius: 999px;
    padding: 8px 14px; font-size: 13px; font-family: inherit;
  }
  #taps button:active { background: #163b52; }
  form { display: flex; gap: 8px; padding: 8px 16px 4px; align-items: flex-end; }
  #text {
    flex: 1; background: var(--slab); color: var(--ink); font: inherit;
    border: 1px solid var(--line); border-radius: 12px; padding: 11px 13px;
    resize: none; max-height: 120px;
  }
  #text:focus { outline: none; border-color: var(--cyan); }
  button.go {
    background: var(--cyan); color: #041017; border: 0; border-radius: 12px;
    padding: 11px 18px; font: 600 15px inherit; font-family: inherit;
  }
  button.go:disabled { opacity: .45; }
  footer { padding: 4px 16px 14px; display: flex; align-items: center;
           gap: 12px; }
  #ptt {
    flex: 1; border-radius: 12px; padding: 12px; font-family: inherit;
    font-size: 14px; border: 1px solid var(--line); background: var(--slab);
    color: var(--ink); letter-spacing: .04em;
  }
  #ptt.off { color: var(--dim); border-style: dashed; }
  #ptt.rec { background: var(--bad); color: #180605; border-color: var(--bad); }
  .toggle { font-size: 12px; color: var(--dim); display: flex; gap: 6px;
            align-items: center; white-space: nowrap; }
  #micnote { font-size: 12px; color: var(--warn); padding: 0 16px 12px;
             line-height: 1.5; }
  #keybox { padding: 16px; }
  #keybox input { width: 100%; padding: 11px; font: inherit; border-radius: 10px;
                  background: var(--slab); color: var(--ink);
                  border: 1px solid var(--line); }
  [hidden] { display: none !important; }
</style>
</head>
<body>
<header>
  <div class="mark"></div>
  <h1>Jarvis</h1>
  <span id="conn">on the home network</span>
</header>

<div id="keybox" hidden>
  <p class="note">This page needs its key. It is in
  <code>~/jarvis-phone.txt</code> on the Spark.</p>
  <input id="key" type="password" autocomplete="off" autocapitalize="off"
         autocorrect="off" spellcheck="false" placeholder="paste the key">
</div>

<div id="log" aria-live="polite"></div>

<div id="taps"></div>

<form id="form" autocomplete="off">
  <textarea id="text" rows="1" placeholder="Ask him something"
            enterkeyhint="send" autocapitalize="sentences"></textarea>
  <button class="go" id="send" type="submit">Ask</button>
</form>

<footer>
  <button id="ptt" type="button">Hold to talk</button>
  <label class="toggle"><input type="checkbox" id="aloud"> aloud</label>
</footer>
<p id="micnote" hidden></p>

<script>
(function () {
  "use strict";
  var KEY_STORE = "jarvis.phone.key";
  var log = document.getElementById("log");
  var form = document.getElementById("form");
  var text = document.getElementById("text");
  var send = document.getElementById("send");
  var taps = document.getElementById("taps");
  var ptt = document.getElementById("ptt");
  var aloud = document.getElementById("aloud");
  var conn = document.getElementById("conn");
  var micnote = document.getElementById("micnote");
  var keybox = document.getElementById("keybox");
  var keyInput = document.getElementById("key");
  var busy = false;

  /* The key rides in the URL so that "Add to Home Screen" bookmarks a
     working link; localStorage is the fallback for a launcher that drops
     the query string (Android opens the manifest's start_url). */
  function readKey() {
    var q = new URLSearchParams(location.search).get("t");
    if (q) { try { localStorage.setItem(KEY_STORE, q); } catch (e) {} return q; }
    try { return localStorage.getItem(KEY_STORE) || ""; } catch (e) { return ""; }
  }
  var key = readKey();

  function el(cls, txt) {
    var d = document.createElement("div");
    d.className = cls;
    d.textContent = txt;
    log.appendChild(d);
    log.scrollTop = log.scrollHeight;
    return d;
  }
  function say(who, txt) { return el("msg " + who, txt); }
  function note(txt, bad) { return el("note" + (bad ? " err" : ""), txt); }

  function render(data) {
    var msgs = (data && data.messages) || [];
    var answered = false;
    for (var i = 0; i < msgs.length; i++) {
      var m = msgs[i];
      if (m.kind === "reply" && m.text) { say("him", m.text); answered = true; }
      else if (m.kind === "status" && m.text) { note(m.text); }
    }
    if (!answered) {
      if (data.reason === "timeout") note("No answer inside the window, sir.", 1);
      else if (data.reason === "idle") note("He has taken that on.");
      else note("Nothing came back.", 1);
    }
  }

  function post(url, body, ctype) {
    return fetch(url, {
      method: "POST", body: body, cache: "no-store",
      headers: Object.assign({ "Authorization": "Bearer " + key },
                             ctype ? { "Content-Type": ctype } : {})
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok) { throw new Error(j.error || ("HTTP " + r.status)); }
        return j;
      });
    });
  }

  function working(on) {
    busy = on;
    send.disabled = on;
    send.textContent = on ? "…" : "Ask";
  }

  function ask(q) {
    if (busy || !q.trim()) { return; }
    say("me", q);
    working(true);
    post("/api/ask", JSON.stringify({ text: q, speak: aloud.checked }),
         "application/json")
      .then(render)
      .catch(function (e) { note(String(e.message || e), 1); })
      .then(function () { working(false); });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var q = text.value;
    text.value = "";
    text.style.height = "auto";
    ask(q);
  });
  text.addEventListener("input", function () {
    text.style.height = "auto";
    text.style.height = Math.min(text.scrollHeight, 120) + "px";
  });
  text.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });

  __TAPS__.forEach(function (pair) {
    var b = document.createElement("button");
    b.type = "button";
    b.textContent = pair[0];
    b.addEventListener("click", function () { ask(pair[1]); });
    taps.appendChild(b);
  });

  /* ---- push to talk ------------------------------------------------
     getUserMedia only exists in a secure context. Over http on the LAN
     there is nothing to try, so the button says so instead of failing
     silently when it is pressed. */
  var canMic = !!(window.isSecureContext && navigator.mediaDevices &&
                  navigator.mediaDevices.getUserMedia && window.MediaRecorder);
  var rec = null, chunks = [], stream = null;

  function micType() {
    var want = ["audio/ogg;codecs=opus", "audio/ogg", "audio/mp4",
                "audio/webm;codecs=opus", "audio/webm"];
    for (var i = 0; i < want.length; i++) {
      if (MediaRecorder.isTypeSupported(want[i])) { return want[i]; }
    }
    return "";
  }

  function stopTracks() {
    if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); }
    stream = null;
  }

  function startRec() {
    if (!canMic || rec || busy) { return; }
    navigator.mediaDevices.getUserMedia({ audio: true }).then(function (s) {
      stream = s;
      chunks = [];
      var type = micType();
      rec = new MediaRecorder(s, type ? { mimeType: type } : undefined);
      rec.ondataavailable = function (e) {
        if (e.data && e.data.size) { chunks.push(e.data); }
      };
      rec.onstop = function () {
        var blob = new Blob(chunks, { type: rec.mimeType || "audio/ogg" });
        rec = null;
        stopTracks();
        ptt.classList.remove("rec");
        ptt.textContent = "Hold to talk";
        if (blob.size < 1200) { note("Too short to hear, sir."); return; }
        working(true);
        post("/api/voice?speak=" + (aloud.checked ? "1" : "0"), blob,
             blob.type)
          .then(function (data) {
            if (data.heard) { say("me", data.heard); }
            render(data);
          })
          .catch(function (e) { note(String(e.message || e), 1); })
          .then(function () { working(false); });
      };
      rec.start();
      ptt.classList.add("rec");
      ptt.textContent = "Listening — let go to send";
    }).catch(function (e) {
      note("The microphone was refused: " + (e.message || e), 1);
    });
  }

  function stopRec() {
    if (rec && rec.state !== "inactive") { rec.stop(); } else { stopTracks(); }
  }

  if (canMic) {
    ["mousedown", "touchstart"].forEach(function (ev) {
      ptt.addEventListener(ev, function (e) { e.preventDefault(); startRec(); });
    });
    ["mouseup", "mouseleave", "touchend", "touchcancel"].forEach(function (ev) {
      ptt.addEventListener(ev, function (e) { e.preventDefault(); stopRec(); });
    });
  } else {
    ptt.classList.add("off");
    ptt.textContent = "Hold to talk — unavailable here";
    micnote.hidden = false;
    micnote.textContent = "__MIC_NOTE__ To use it: open this page over https " +
      "with a certificate this phone trusts, or open it on the Spark itself " +
      "at http://127.0.0.1:8765/ — localhost counts as secure.";
    ptt.addEventListener("click", function () {
      micnote.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }

  /* ---- the key ----------------------------------------------------- */
  keyInput.addEventListener("change", function () {
    key = keyInput.value.trim();
    try { localStorage.setItem(KEY_STORE, key); } catch (e) {}
    keyInput.value = "";
    hello();
  });

  function hello() {
    if (!key) {
      keybox.hidden = false;
      conn.textContent = "no key";
      conn.className = "bad";
      return;
    }
    fetch("/api/ping", { headers: { "Authorization": "Bearer " + key },
                         cache: "no-store" })
      .then(function (r) {
        if (r.status === 401) { throw new Error("that key was refused"); }
        if (!r.ok) { throw new Error("HTTP " + r.status); }
        keybox.hidden = true;
        conn.textContent = "on the home network";
        conn.className = "";
        text.focus();
      })
      .catch(function (e) {
        keybox.hidden = false;
        conn.textContent = String(e.message || e);
        conn.className = "bad";
      });
  }
  hello();
})();
</script>
</body>
</html>
"""
PAGE = (PAGE.replace("__TAPS__", json.dumps([list(t) for t in QUICK_TAPS]))
            .replace("__MIC_NOTE__", MIC_NOTE))
