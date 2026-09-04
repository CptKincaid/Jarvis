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

HIS VOICE, OUT OF THE PHONE
---------------------------
A reply he cannot hear is half an assistant, and the Spark's speaker is no
use from a lecture theatre. ``POST /api/say`` hands one reply to
``tts.Rendition`` and streams back what it renders: the SAME F5 voice, the
same reference clip, the same sentence split and, above all, the same
speech cache. A canned line ("Always, sir.") or a repeat is already on disk
under that key, so it is a file read and a Content-Length -- no synthesis at
all. A miss is streamed chunked as each sentence lands, so the phone is
playing the first one while the second is still on the GPU.

The two rooms are two switches on the page and they are not the same
switch. **Voice** plays the answer out of the phone and is the new one;
**Aloud** is the old ``speak`` flag and makes the Spark answer the house as
well. Both are off by default -- Voice because he uses this in lectures,
Aloud because a question asked from bed should not wake anyone. Voice
never touches the room: ``Rendition`` reaches the engine and the cache and
reaches none of ``TTS``'s playback, so the room's silence during a phone
turn is a property of the code path rather than of a flag being clear.

iOS will not start audio without a user gesture, so the AudioContext is
created and unlocked on the TAP that enables Voice (and re-armed on each
send, for a switch remembered in localStorage from last time). iOS also
decides whether a page may be HEARD before it decides how loud: a bare
AudioContext plays in the "ambient" category, which the side switch on the
phone silences outright -- the fetch succeeds, the samples are scheduled,
``state`` says "running", and nothing comes out. The page claims
``navigator.audioSession.type = "playback"`` (Safari 16.4+) to leave that
category, and where it cannot it SAYS so rather than going quietly silent.

Silence is the failure mode that costs a bug report, so every clip ends by
saying what became of it -- nothing to say, nothing scheduled, a context
that is not running, or how many seconds went to the speaker -- and the
**Test** switch plays ``SOUND_CHECK`` down the identical path (same
endpoint, voice, cache, RIFF walk and scheduler as a reply) so the audio
path can be proved in one tap without asking a question first.

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
from jarvis import contacts as contacts_mod
from jarvis.logs import get_logger

log = get_logger("webapp")

SOURCE = "phone"                  # one of jarvis/app.py's SOCKET_SOURCES
DEFAULT_PORT = 8765
DEFAULT_TIMEOUT_S = 60.0          # shorter than the CLI's 90: a phone waiting
MAX_TIMEOUT_S = 180.0             # on a spinner is a worse place to wait
MAX_TEXT_BYTES = 16 * 1024        # a typed question, with room to spare
DEFAULT_MAX_AUDIO_MB = 8
# What /api/say sends back. PCM wav rather than anything cleverer because it
# is what the engines already write, what the speech cache already holds, and
# what every browser decodes without a codec question -- transcoding a reply
# would cost more time than it saved bytes on a home link.
SAY_TYPE = "audio/wav"
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

# No third-party anything is loaded by this page, so the strictest policy
# that still allows the inline script and style is free. ``media-src`` is
# here for the voice: the reply audio arrives through fetch (connect-src),
# and the fallback for a browser with no Web Audio plays it from a blob.
CSP = ("default-src 'none'; script-src 'unsafe-inline'; "
       "style-src 'unsafe-inline'; img-src 'self' data:; "
       "media-src 'self' blob:; connect-src 'self'; manifest-src 'self'; "
       "form-action 'none'; base-uri 'none'")

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
        self.spoken = 0                  # replies rendered for a phone's ear
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
        started = time.monotonic()
        with cmdsock.ReplyCollector(turn_id) as col:
            bus.publish(UserUtterance(text=text, source=SOURCE))
            result = self.app.dispatch_text(text, source=SOURCE,
                                            quiet=not speak, turn_id=turn_id)
            reason = cmdsock.stream_turn(lines.append, col, result, limit,
                                         self.idle_grace_s)
        self.served += 1
        ms = _elapsed_ms(started)
        log.info("phone turn %s: %s in %d ms (server side)", turn_id, reason, ms)
        return {"messages": lines, "reason": reason, "ms": ms}

    def _read(self, attr: str, fallback: str) -> dict:
        started = time.monotonic()
        fn = getattr(self.app, attr, None)
        line = fn() if callable(fn) else fallback
        self.served += 1
        return {"messages": [{"kind": "reply", "text": line, "speak": False}],
                "reason": "done", "ms": _elapsed_ms(started)}

    # ------------------------------------------------------------ his voice
    def can_speak(self) -> bool:
        """True when this box can hand the phone Jarvis's actual voice.

        A plain ``hasattr`` rather than a try-render: the page asks this on
        every ping to decide whether the Voice switch is offered at all, and
        a switch that is present but dead is worse than one that is absent.
        """
        return hasattr(getattr(self.app, "tts", None), "render")

    def rendition(self, text: str):
        """Jarvis's voice for one reply, or None when there is no engine.

        This is the whole reason the feature is honest: it goes to
        ``TTS.render``, which shares the room's engine, reference clip and
        speech cache and reaches NONE of its playback -- so a reply that
        comes out of the phone never also comes out of the Spark. The room's
        silence is a property of the code path, not of a flag.
        """
        tts = getattr(self.app, "tts", None)
        if tts is None or not hasattr(tts, "render"):
            return None
        rend = tts.render(text)
        self.spoken += 1
        return rend

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
        self.send_header("Content-Security-Policy", CSP)
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
        if path == "/contacts":
            # The ADDRESS BOOK's shell (jarvis/contacts.py). Static, like
            # "/": no name and no address is baked in; the rows arrive
            # only through the gated /api/contacts below.
            self._send(200, CONTACTS_PAGE.encode("utf-8"),
                       "text/html; charset=utf-8")
            return
        if path == "/api/contacts":
            if not self._gate():
                return
            self._contacts_get()
            return
        if path == "/api/ping":
            if not self._gate():
                return
            # The page beats on this every 20 s while it is in front of him,
            # which is also what holds the connection (and the tailnet's
            # direct path) open between questions -- see the heartbeat in
            # PAGE. It must therefore stay the cheapest thing here.
            self._json(200, {"ok": True, "mic_note": MIC_NOTE,
                             "voice": self.phone.can_speak()})
            return
        self._error(404, "no such page")

    def do_HEAD(self):                                      # noqa: N802
        self.do_GET()

    def do_POST(self):                                      # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/ask":
            self._ask()
        elif path == "/api/say":
            self._say()
        elif path == "/api/voice":
            self._voice()
        elif path == "/api/contacts":
            self._contacts_post()
        else:
            self._error(404, "no such endpoint")

    def _ask(self):
        if not self._gate():
            return
        asked = self._json_text()
        if asked is None:
            return
        text, msg = asked
        try:
            out = self.phone.ask(text, speak=bool(msg.get("speak", False)),
                                 timeout=_timeout(msg.get("timeout")))
        except Exception:                 # noqa: BLE001 - answer, never 500-silent
            log.exception("phone ask failed")
            self._error(500, "Jarvis could not handle that")
            return
        self._json(200, out)

    # ------------------------------------------------------- his voice
    def _say(self):
        """``POST {"text": ...}`` -> that reply, in Jarvis's own voice.

        Deliberately a separate call from ``/api/ask`` rather than audio
        bolted onto the answer: the page holds the Voice switch off by
        default (he is often in a lecture), and "off" has to mean nothing
        is fetched and nothing is rendered, not that a clip was made and
        thrown away. It also means the audio for a line he can already read
        is fetched while he is reading it.
        """
        if not self._gate():
            return
        asked = self._json_text()
        if asked is None:
            return
        # Before the cache lookups, not after: they are part of what the
        # phone waited for, and a hit that reports itself as free would be
        # measuring the wrong thing.
        started = time.monotonic()
        try:
            rend = self.phone.rendition(asked[0])
        except Exception:               # noqa: BLE001 - answer, never 500-silent
            log.exception("phone say could not be prepared")
            self._error(503, "his voice is not available just now")
            return
        if rend is None:
            self._error(503, "no speech engine on this box")
            return
        if not rend:
            # A reply that is all markup or all punctuation cleans down to
            # nothing sayable. That is silence, not a fault: an error here
            # would put a red line on his screen for a reply he can read
            # perfectly well.
            log.info("phone say: %r has nothing to speak", asked[0][:60])
            self._send(200, b"", SAY_TYPE)
            return
        if rend.cached:
            # Every chunk was already on disk, so the clip is a few file
            # reads: send it whole, with a real length and byte ranges, so
            # a media element (or Safari's own probe) is happy too.
            try:
                body = rend.body()
            except Exception:           # noqa: BLE001 - answer, never 500-silent
                log.exception("phone say: a cached chunk would not read")
                self._error(503, "his voice is not available just now")
                return
            log.info("phone say: %d chunk(s), all cached, %d bytes in %d ms",
                     len(rend), len(body), _elapsed_ms(started))
            self._send_clip(body)
            return
        self._stream_clip(rend, started)

    # ---------------------------------------------------- the address book
    # A read/write of ONE FILE (jarvis/contacts.py), never a turn: like
    # /api/status and /api/board these do not go through app.dispatch_text,
    # so an edit is not remembered as an exchange, never wakes the speaker
    # and never barges in. The validator and the uniqueness rules are the
    # CLI's -- contacts.validate_row / check_unique through Book.add -- so
    # the page cannot accept a row the CLI would refuse, and a refusal
    # writes nothing.
    def _contacts_get(self):
        try:
            self._json(200, contacts_mod.current().public())
        except Exception:                 # noqa: BLE001 - a file read
            log.exception("contacts page: the book could not be read")
            self._error(500, "the address book could not be read")

    def _contacts_post(self):
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
        op = str(msg.get("op") or "").strip().lower()
        book = contacts_mod.current()
        try:
            if op == "add":
                row = {k: msg.get(k) for k in
                       ("name", "email", "honorific", "aliases", "note")}
                contact, why = book.add(row)
            elif op == "remove":
                # BOTH the exact name and the exact address, so a stale
                # tab cannot remove the wrong row after a hand edit.
                name = str(msg.get("name") or "").strip()
                email = str(msg.get("email") or "").strip()
                if not name or not email:
                    self._error(400, "remove needs the name and the address")
                    return
                contact, why = book.remove(name, email)
            else:
                self._error(400, "op must be add or remove")
                return
        except OSError:
            log.exception("contacts page: the book could not be written")
            self._error(500, "the address book could not be written")
            return
        if contact is None:
            self._error(400, why)
            return
        log.info("contacts page: %s %s", op, contact.name)
        out = {"ok": True}
        out.update(book.public())
        self._json(200, out)

    def _json_text(self) -> Optional[tuple[str, dict]]:
        """``(text, the whole message)`` from a JSON body, or None having
        already answered. Shared by /api/ask and /api/say, which take the
        same shape and owe the same four errors."""
        body = self._body(MAX_TEXT_BYTES)
        if body is None:
            return None
        try:
            msg = json.loads(body.decode("utf-8", "replace"))
            if not isinstance(msg, dict):
                raise ValueError("not an object")
        except ValueError as exc:
            self._error(400, f"malformed request: {exc}")
            return None
        text = str(msg.get("text") or "").strip()
        if not text:
            self._error(400, "empty request")
            return None
        return text, msg

    def _send_clip(self, body: bytes):
        """A clip whose length is known: Content-Length, and byte ranges.

        The page itself never asks for a range -- it reads the body as a
        stream and feeds Web Audio. Ranges are here for the media element
        underneath: Safari probes an audio source with ``Range: bytes=0-1``
        before it will play it, and a server that answers 200 to that has
        been the reason for a silent <audio> tag more than once."""
        total = len(body)
        span = self._range(total)
        extra = {"Accept-Ranges": "bytes"}
        if span is None:
            self._send(200, body, SAY_TYPE, extra=extra)
            return
        start, end = span
        extra["Content-Range"] = f"bytes {start}-{end}/{total}"
        self._send(206, body[start:end + 1], SAY_TYPE, extra=extra)

    def _range(self, total: int) -> Optional[tuple[int, int]]:
        """``(start, end)`` for a single byte range, or None to send it all.

        Only ``bytes=a-b`` / ``bytes=a-`` / ``bytes=-n`` are understood.
        Anything else -- a multipart range, a malformed one, one that starts
        past the end -- returns None, and the whole body goes out with a
        200. Ignoring a Range is allowed and is the safe way to be wrong."""
        raw = (self.headers.get("Range") or "").strip()
        if not raw.startswith("bytes=") or "," in raw or total <= 0:
            return None
        spec = raw[6:].strip()
        first, _, last = spec.partition("-")
        try:
            if not first:                      # bytes=-n : the final n bytes
                n = int(last)
                if n <= 0:
                    return None
                return max(0, total - n), total - 1
            start = int(first)
            end = int(last) if last else total - 1
        except ValueError:
            return None
        if start < 0 or start >= total or end < start:
            return None
        return start, min(end, total - 1)

    def _stream_clip(self, rend, started: float):
        """Chunked, so the phone is playing while the Spark is still
        rendering: on a streaming engine that is the first BLOCK of the
        first chunk (~0.3 s in, whatever the reply's length), and on one
        that renders whole chunks it is the first chunk. See tts.Rendition.

        The generator's first item is the wav header, and it does not exist
        until that first audio does -- which is why it is pulled BEFORE the
        response line goes out. A render that is going to fail then fails
        while an error can still be sent, instead of committing a 200 and
        hanging up in the middle of a clip. It is also why the wait for it
        has to stay short: nothing at all reaches the phone until it
        returns.
        """
        blocks = rend.stream()
        try:
            head = next(blocks, None)
        except Exception:               # noqa: BLE001 - answer, never 500-silent
            log.exception("phone say: nothing would render")
            self._error(503, "his voice is not available just now")
            return
        if not head:
            self._error(503, "his voice produced nothing")
            return
        first_ms = _elapsed_ms(started)
        self.send_response(200)
        self.send_header("Content-Type", SAY_TYPE)
        # HTTP/1.1 has exactly two framings and the length is not known
        # here, so the chunks are framed by hand below.
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Accept-Ranges", "none")
        self.end_headers()
        sent = 0
        try:
            for block in _chain(head, blocks):
                if not block:
                    continue
                self.wfile.write(b"%x\r\n" % len(block) + block + b"\r\n")
                sent += len(block)
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            # He locked the phone, or walked out of range, mid-sentence.
            log.debug("phone left mid-clip after %d bytes", sent, exc_info=True)
            self.close_connection = True
            return
        except Exception:               # noqa: BLE001 - a half-sent clip
            log.exception("phone say failed after %d bytes", sent)
            self.close_connection = True
            return
        log.info("phone say: %d chunk(s), %d cached, first byte %d ms, "
                 "%d bytes in %d ms", len(rend),
                 sum(1 for _, p in rend.plan if p), first_ms, sent,
                 _elapsed_ms(started))

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


def _elapsed_ms(started: float) -> int:
    """Whole milliseconds since a ``time.monotonic()`` mark.

    Every turn carries one back to the page as ``ms``. The point is not
    vanity: the first question after the phone has been asleep arrives over
    a relay while the tailnet renegotiates a direct path, and without the
    server's own half of the clock the page cannot tell "he was slow" from
    "the link was slow" -- and neither can he."""
    return int(round((time.monotonic() - started) * 1000))


def _chain(first, rest):
    """``first`` then everything in ``rest``: the streaming clip's header
    has to be pulled before the response line goes out, and then put back
    in front of the body it belongs to."""
    yield first
    yield from rest


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

# What the Test switch plays. A real sentence rather than a synthetic tone
# on purpose: it goes down the identical path a reply does -- render, speech
# cache, chunked wav, RIFF walk, Web Audio -- so hearing it proves that path
# end to end, and it is a cache hit from the second tap onward.
SOUND_CHECK = "Sound check, sir."

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
  footer { padding: 4px 16px 8px; display: flex; align-items: center;
           gap: 12px; }
  #ptt {
    flex: 1; border-radius: 12px; padding: 12px; font-family: inherit;
    font-size: 14px; border: 1px solid var(--line); background: var(--slab);
    color: var(--ink); letter-spacing: .04em;
  }
  #ptt.off { color: var(--dim); border-style: dashed; }
  #book { color: var(--dim); font-size: 13px; text-decoration: none;
          white-space: nowrap; letter-spacing: .02em; }
  #ptt.rec { background: var(--bad); color: #180605; border-color: var(--bad); }
  #switches { display: flex; gap: 8px; padding: 0 16px 14px; }
  .sw {
    flex: 1; display: flex; align-items: center; gap: 8px; min-width: 0;
    background: var(--slab); color: var(--dim); font: inherit; font-size: 13px;
    border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px;
    letter-spacing: .02em; white-space: nowrap;
  }
  .sw span.txt { overflow: hidden; text-overflow: ellipsis; }
  .sw input { accent-color: var(--cyan); margin: 0; flex: none; }
  .sw:disabled { opacity: .45; }
  .dot { width: 9px; height: 9px; border-radius: 50%; flex: none;
         background: var(--line); }
  #voice.on { color: #041017; background: var(--cyan);
              border-color: var(--cyan); }
  #voice.on .dot { background: #041017; }
  /* Both audio controls take only the width of their own words, so the
     third switch costs the room switch nothing: "Aloud in the room" still
     fits whole on a 375 px phone, which is why it is not sharing thirds. */
  #voice, #check { flex: none; }
  #check:active { background: #163b52; }
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
  <a id="book" href="/contacts" title="Who Jarvis can email a file to">Address book</a>
</footer>

<!-- Two different rooms, and the labels say which. "Voice" plays his answer
     out of THIS phone; "Aloud" makes the Spark answer the house as well.
     Both are off to begin with. -->
<div id="switches">
  <button id="voice" type="button" class="sw" aria-pressed="false"
          title="Play his answers through this phone">
    <span class="dot"></span><span class="txt" id="voicelbl">Voice off</span>
  </button>
  <label class="sw" title="Answer out loud on the Spark, in the room">
    <input type="checkbox" id="aloud">
    <span class="txt">Aloud in the room</span>
  </label>
  <!-- One tap that proves the whole audio path without having to ask a
       question first, and prints what happened either way. -->
  <button id="check" type="button" class="sw"
          title="Play one line through this phone, to prove the sound works">
    <span class="txt">Test</span>
  </button>
</div>
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
  var voiceBtn = document.getElementById("voice");
  var voiceLbl = document.getElementById("voicelbl");
  var checkBtn = document.getElementById("check");
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
  /* The address book page reads the same localStorage key; the query form
     is for a launcher that dropped it. */
  var book = document.getElementById("book");
  function linkBook() {
    if (book) { book.href = "/contacts" + (key ? "?t=" + encodeURIComponent(key) : ""); }
  }
  linkBook();

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

  function render(data, t0) {
    var msgs = (data && data.messages) || [];
    var answered = false;
    for (var i = 0; i < msgs.length; i++) {
      var m = msgs[i];
      if (m.kind === "reply" && m.text) {
        say("him", m.text);
        /* One /api/say per reply MESSAGE, not per turn: the room caches
           speech per sentence chunk of one reply, and asking for the same
           string he said aloud is what turns a repeat into a cache hit. */
        speak(m.text);
        answered = true;
      } else if (m.kind === "status" && m.text) { note(m.text); }
    }
    if (!answered) {
      if (data.reason === "timeout") note("No answer inside the window, sir.", 1);
      else if (data.reason === "idle") note("He has taken that on.");
      else note("Nothing came back.", 1);
    }
    linkNote(data, t0);
  }

  /* The first question after the phone has been away is slow, and it is
     not him: the tailnet sends those packets through a relay while it
     renegotiates a direct path. The server times its own half of the turn
     and sends it back as `ms`, so the difference is the link's, and saying
     which half was slow beats letting him wonder. Silent when it is fine. */
  function now() {
    return (window.performance && performance.now) ? performance.now()
                                                   : Date.now();
  }
  function linkNote(data, t0) {
    if (!t0 || !data || typeof data.ms !== "number") { return; }
    var lag = Math.round(now() - t0 - data.ms);
    if (lag < 700) { return; }
    note("He answered in " + (data.ms / 1000).toFixed(1) + " s; the link " +
         "added " + (lag / 1000).toFixed(1) + " s. The tailnet is still " +
         "relaying — it settles once a direct path is up.");
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
    /* Every caller of ask() is a tap or a keypress, so this runs inside a
       user gesture -- which is the only place iOS will let an AudioContext
       out of "suspended". A remembered Voice switch is armed here, on the
       send, rather than when the answer arrives, which is too late. */
    if (voiceOn) { hush(); unlock(); }
    say("me", q);
    working(true);
    var t0 = now();
    post("/api/ask", JSON.stringify({ text: q, speak: aloud.checked }),
         "application/json")
      .then(function (data) { render(data, t0); })
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

  /* ---- his voice, in his ear ---------------------------------------
     The answer is spoken by the SAME engine, reference clip and speech
     cache the Spark uses (POST /api/say -> tts.Rendition), never by the
     browser's own speech synthesis: a stock robot reading his assistant's
     lines would be a different assistant, which defeats the point.

     Web Audio rather than an <audio> element, for one reason that decides
     it on this phone: a reply that has to be rendered arrives as a chunked
     stream of unknown length, and iOS Safari's media element wants a
     length and a byte range before it will commit to one. Reading the body
     with fetch and scheduling the PCM here plays the first sentence the
     moment it lands and never asks the question.

     OFF by default and remembered, because half of these turns happen in a
     lecture theatre. Off means nothing is fetched and nothing is rendered
     — there is no clip made and discarded. */
  var VOICE_STORE = "jarvis.phone.voice";
  var LEAD_S = 0.06;       /* schedule a hair ahead of now: a block that
                              arrives late must not be told to start in the
                              past, which plays it at once and stacks it */
  var MIN_BLOCK_S = 0.15;  /* and don't schedule slivers — a buffer seam
                              every few KB is audible as a tick */
  var CHECK_LINE = __CHECK__;
  var actx = null, voiceOn = false, playAt = 0, playing = [], inflight = null;
  var chain = Promise.resolve();
  var session = "no";      /* whether this browser let the page out of the
                              "ambient" category -- see claimSession */
  var told = false;        /* the side-switch line is said once per load */

  /* iOS decides whether a page may be HEARD before it decides how loud.
     A bare AudioContext plays in the "ambient" category, and the side
     switch on the phone silences that category outright: the fetch
     succeeds, the PCM is scheduled, state says "running", and nothing
     comes out of the speaker. Claiming "playback" (Safari 16.4+) is what
     leaves that category, so the switch stops mattering.

     Older WebKit gives script no such control at all. The only escape
     there is to play through an <audio> element, which would cost this
     page the thing it is built on -- a chunked reply of unknown length
     that Safari's media element will not commit to (see the note above
     fetchSay). So where the claim cannot be made, the page says which
     case he is in instead of pretending, and the Test switch prints it. */
  function claimSession() {
    var s = navigator.audioSession;
    if (!s) { session = "no"; return; }
    try { s.type = "playback"; } catch (e) { session = "no"; return; }
    session = (s.type === "playback") ? "yes" : "no";
  }

  function unlock() {
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) { return false; }
    claimSession();                 /* before the context makes a sound */
    if (actx && actx.state === "closed") { actx = null; playAt = 0; }
    if (!actx) { try { actx = new Ctx(); } catch (e) { return false; } }
    /* Not "suspended" alone: iOS parks a backgrounded or interrupted
       context in WebKit's own "interrupted" state, which that test walked
       straight past -- leaving a dead context to swallow the next reply
       in silence. Resume anything that is not already running. */
    if (actx.state !== "running" && actx.resume) {
      try { actx.resume(); } catch (e) {}
    }
    /* Starting one silent frame is what actually flips iOS out of
       "suspended"; resume() alone is not always enough. */
    try {
      var s = actx.createBufferSource();
      s.buffer = actx.createBuffer(1, 1, 22050);
      s.connect(actx.destination);
      s.start(0);
    } catch (e) {}
    return true;
  }

  function hush() {
    if (inflight) { try { inflight.abort(); } catch (e) {} inflight = null; }
    playing.forEach(function (s) { try { s.stop(); } catch (e) {} });
    playing = [];
    playAt = 0;
    chain = Promise.resolve();
  }

  /* ---- did any of it actually reach the speaker? ---------------------
     The bug this page had was not that audio broke; it was that audio
     broke IN SILENCE, which on a phone looks exactly like a switch that
     did not take -- and leaves him nothing to report but "it doesn't
     talk". Every clip now ends by saying what became of it, in terms he
     can act on. */
  function report(probe, got, secs) {
    var state = actx ? actx.state : "gone";
    if (!got) {
      note("There was nothing in that one to say aloud.");
      return;
    }
    if (!secs) {
      note("His voice arrived — " + (got >= 2048 ? Math.round(got / 1024) +
           " KB" : got + " bytes") + " — and this phone played none of it.", 1);
      return;
    }
    if (state !== "running") {
      note("This phone is holding audio back: it is '" + state + "'. " +
           "Tap Voice once more.", 1);
      return;
    }
    if (!probe && told) { return; }
    told = true;
    note("Handed " + secs.toFixed(1) + " s of his voice to this phone's " +
         "speaker. " + (session === "yes"
           ? "If you heard nothing, turn the volume up with this page open, " +
             "or check what it is paired to — the side switch is not what " +
             "is stopping it."
           : "If you heard nothing, your phone is muted: the side switch " +
             "silences this kind of audio. Flip it off silent, turn the " +
             "volume up, and tap Test."));
  }

  function speak(t) {
    if (!voiceOn || !t) { return; }
    /* Serialised: two replies in one turn must not race each other into
       the speakers, and the fetches are cheap enough to take in order. */
    chain = chain.then(function () { return fetchSay(t); })
                 .catch(function (e) {
                   if (e && e.name === "AbortError") { return; }
                   note("His voice did not reach this phone: " +
                        ((e && e.message) || e), 1);
                 });
  }

  function fetchSay(text, probe) {
    if (!unlock()) {
      return Promise.reject(new Error("this browser has no Web Audio"));
    }
    var ctrl = ("AbortController" in window) ? new AbortController() : null;
    inflight = ctrl;
    var head = null, pending = new Uint8Array(0);
    /* What report() needs to tell "the server sent nothing" from "the
       phone played nothing": bytes off the wire, and seconds handed to
       the sound card. */
    var got = 0, secs = 0;

    function grow(extra) {
      got += extra.length;
      var out = new Uint8Array(pending.length + extra.length);
      out.set(pending);
      out.set(extra, pending.length);
      pending = out;
    }
    function tag(d, at) {
      return String.fromCharCode(d[at], d[at + 1], d[at + 2], d[at + 3]);
    }
    function header() {
      /* Walk the RIFF chunks to "data" instead of assuming 44 bytes, and
         read the rate out of "fmt " instead of assuming 24 kHz — true of
         F5 and XTTS today, and a silent pitch shift the day it is not. */
      var d = pending;
      if (d.length < 12) { return false; }
      if (tag(d, 0) !== "RIFF" || tag(d, 8) !== "WAVE") {
        throw new Error("that was not audio");
      }
      var dv = new DataView(d.buffer, d.byteOffset, d.byteLength);
      var at = 12, fmt = null;
      while (at + 8 <= d.length) {
        var id = tag(d, at), size = dv.getUint32(at + 4, true);
        if (id === "fmt ") {
          if (at + 24 > d.length) { return false; }
          fmt = { channels: dv.getUint16(at + 10, true),
                  rate: dv.getUint32(at + 12, true),
                  bits: dv.getUint16(at + 22, true) };
        } else if (id === "data") {
          if (!fmt || fmt.bits !== 16 || !fmt.channels || !fmt.rate) {
            throw new Error("unexpected audio format");
          }
          head = fmt;
          pending = d.subarray(at + 8);
          return true;
        }
        at += 8 + size + (size % 2);
      }
      return false;
    }
    function emit(last) {
      if (!head && !header()) { return; }
      var frame = head.channels * 2;
      var want = last ? frame : Math.ceil(head.rate * MIN_BLOCK_S) * frame;
      if (pending.length < want) { return; }
      var n = pending.length - (pending.length % frame);
      if (!n) { return; }
      var pcm = pending.subarray(0, n);
      pending = pending.subarray(n);
      var frames = n / frame;
      var buf = actx.createBuffer(head.channels, frames, head.rate);
      var dv = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
      for (var c = 0; c < head.channels; c++) {
        var out = buf.getChannelData(c);
        for (var i = 0; i < frames; i++) {
          out[i] = dv.getInt16((i * head.channels + c) * 2, true) / 32768;
        }
      }
      var src = actx.createBufferSource();
      src.buffer = buf;
      src.connect(actx.destination);
      if (playAt < actx.currentTime + LEAD_S) {
        playAt = actx.currentTime + LEAD_S;
      }
      src.start(playAt);
      playAt += buf.duration;
      secs += buf.duration;
      playing.push(src);
      src.onended = function () {
        var i = playing.indexOf(src);
        if (i >= 0) { playing.splice(i, 1); }
      };
    }
    function pump(reader) {
      return reader.read().then(function (r) {
        if (r.done) { emit(true); return; }
        grow(new Uint8Array(r.value));
        emit(false);
        return pump(reader);
      });
    }

    return fetch("/api/say", {
      method: "POST", cache: "no-store",
      signal: ctrl ? ctrl.signal : undefined,
      headers: { "Authorization": "Bearer " + key,
                 "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    }).then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; })
                .then(function (j) {
                  throw new Error(j.error || ("HTTP " + r.status));
                });
      }
      if (r.body && r.body.getReader) { return pump(r.body.getReader()); }
      /* No streaming body (an older WebKit): take the whole clip and play
         it in one piece rather than not at all. */
      return r.arrayBuffer().then(function (b) {
        grow(new Uint8Array(b));
        emit(true);
      });
    }).then(function () {
      if (inflight === ctrl) { inflight = null; }
      /* Blocked audio must SAY it is blocked, and so must audio that was
         never scheduled: silence is exactly what a broken toggle, a muted
         phone and an empty reply all look like from here. */
      report(probe, got, secs);
    });
  }

  function setVoice(on, announce) {
    voiceOn = !!on;
    voiceBtn.classList.toggle("on", voiceOn);
    voiceBtn.setAttribute("aria-pressed", voiceOn ? "true" : "false");
    voiceLbl.textContent = voiceOn ? "Voice on" : "Voice off";
    try { localStorage.setItem(VOICE_STORE, voiceOn ? "1" : "0"); } catch (e) {}
    if (!voiceOn) { hush(); return; }
    if (!announce) { return; }
    if (!unlock()) {
      note("This browser will not play audio here, sir.", 1);
      return;
    }
    /* Say where the switch has got to, and point at the one control that
       settles it -- a toggle whose only feedback is its own colour is how
       this went unreported twice. */
    note("Voice on. Tap Test to hear him now, without asking anything.");
  }

  /* The sound check: the SAME endpoint, voice, speech cache, RIFF walk and
     scheduler a reply uses, off one tap. If this is heard and a reply is
     not, the fault is not in the audio path; if neither is heard while the
     page says it handed seconds to the speaker, the phone is muted. The
     line is cached after the first tap, so it comes back as a file read. */
  checkBtn.addEventListener("click", function () {
    if (!unlock()) {
      note("This browser will not play audio here, sir.", 1);
      return;
    }
    hush();
    note(session === "yes"
      ? "Sound check: this page holds the media session, so the side " +
        "switch cannot mute it."
      : "Sound check: this browser will not let a page hold the media " +
        "session, so the side switch CAN mute it.");
    chain = chain.then(function () { return fetchSay(CHECK_LINE, true); })
                 .catch(function (e) {
                   if (e && e.name === "AbortError") { return; }
                   note("The sound check did not reach this phone: " +
                        ((e && e.message) || e), 1);
                 });
  });

  /* The enabling TAP is the gesture iOS wants, so the AudioContext is
     built and unlocked right here — never lazily on the first reply, which
     is precisely the case the phone refuses. */
  voiceBtn.addEventListener("click", function () { setVoice(!voiceOn, true); });
  try { setVoice(localStorage.getItem(VOICE_STORE) === "1", false); }
  catch (e) { setVoice(false, false); }

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
    if (voiceOn) { unlock(); }        /* a press is a gesture; use it */
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
        if (voiceOn) { hush(); }
        var t0 = now();
        post("/api/voice?speak=" + (aloud.checked ? "1" : "0"), blob,
             blob.type)
          .then(function (data) {
            if (data.heard) { say("me", data.heard); }
            render(data, t0);
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
    linkBook();
    hello();
  });

  function ping() {
    return fetch("/api/ping", { headers: { "Authorization": "Bearer " + key },
                                cache: "no-store" })
      .then(function (r) {
        if (r.status === 401) { throw new Error("that key was refused"); }
        if (!r.ok) { throw new Error("HTTP " + r.status); }
        return r.json().catch(function () { return {}; });
      });
  }

  function hello() {
    if (!key) {
      keybox.hidden = false;
      conn.textContent = "no key";
      conn.className = "bad";
      return;
    }
    ping()
      .then(function (j) {
        keybox.hidden = true;
        conn.textContent = "on the home network";
        conn.className = "";
        /* A box with no speech engine gets a dead switch that SAYS it is
           dead, rather than one that looks alive and answers in silence. */
        if (j && j.voice === false) {
          setVoice(false, false);
          voiceBtn.disabled = true;
          voiceLbl.textContent = "Voice unavailable";
        }
        text.focus();
      })
      .catch(function (e) {
        keybox.hidden = false;
        conn.textContent = String(e.message || e);
        conn.className = "bad";
      });
  }
  hello();

  /* ---- holding the path open ---------------------------------------
     The first question after a spell away is slow for a reason that is
     not this app: the tailnet's packets go through a relay while NAT
     traversal renegotiates a direct path, and this server drops an idle
     keep-alive connection after 30 s. One cheap ping every 20 s while the
     page is actually in front of him keeps both alive, and one on the way
     back from the lock screen warms the path BEFORE he types rather than
     making his first question pay for it. Nothing is reconfigured
     anywhere; this is only traffic. */
  var BEAT_MS = 20000;
  function beat() {
    if (!key || document.visibilityState === "hidden") { return; }
    ping().catch(function () {});
  }
  setInterval(beat, BEAT_MS);
  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState !== "visible") { return; }
    beat();
    /* iOS parks the AudioContext when the page goes away -- sometimes
       "suspended", sometimes WebKit's own "interrupted" -- and until now
       nothing woke it, so the first reply after the lock screen was
       silent with no gesture in sight to blame. Returning to a tab is not
       a user gesture everywhere, so this may not take; when it does not,
       report() says which state it is stuck in on the next clip. */
    if (voiceOn) { unlock(); }
  });
  /* Restored from the back-forward cache: the sources scheduled against
     the old context will never fire, so drop them and re-arm. */
  window.addEventListener("pageshow", function (e) {
    if (e.persisted && voiceOn) { hush(); unlock(); }
  });
})();
</script>
</body>
</html>
"""
PAGE = (PAGE.replace("__TAPS__", json.dumps([list(t) for t in QUICK_TAPS]))
            .replace("__CHECK__", json.dumps(SOUND_CHECK))
            .replace("__MIC_NOTE__", MIC_NOTE))


# ---------------------------------------------------------- the address book
# The third way into ~/.config/jarvis/contacts.json (the file and the CLI
# are the other two). Same conventions as PAGE: no <form> (CSP form-action
# is 'none' and a submit would fail in silence), every call is a fetch()
# with the Bearer key, and the shell carries no data -- the list arrives
# through the gated API. No manifest, no Add-to-Home-Screen: it is a page
# he opens from the phone page's footer, not an app.
CONTACTS_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1,
      viewport-fit=cover, maximum-scale=1">
<meta name="theme-color" content="#050d14">
<meta name="referrer" content="no-referrer">
<title>Jarvis address book</title>
<link rel="icon" href="/icon-180.png">
<style>
  :root {
    --bg: #050d14; --slab: #0b1a25; --line: #17364a;
    --ink: #dbeaf2; --dim: #7b98aa; --cyan: #4de0f5; --warn: #f0b849;
    --bad: #e8695f;
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html, body { margin: 0; background: var(--bg); color: var(--ink);
               font: 15px/1.45 -apple-system, "Segoe UI", system-ui, sans-serif; }
  main { max-width: 640px; margin: 0 auto; padding: 14px 16px 40px; }
  h1 { font-size: 17px; letter-spacing: .06em; text-transform: uppercase;
       color: var(--cyan); margin: 8px 0 4px; }
  h1 a { color: var(--dim); font-size: 13px; text-transform: none;
         letter-spacing: .02em; text-decoration: none; margin-left: 10px; }
  p.hint { color: var(--dim); font-size: 13px; margin: 0 0 12px; }
  #status { color: var(--warn); font-size: 13px; min-height: 1.4em; margin: 6px 0; }
  #status.bad { color: var(--bad); }
  #status.ok { color: var(--cyan); }
  ul { list-style: none; padding: 0; margin: 0 0 18px; }
  li { background: var(--slab); border: 1px solid var(--line); border-radius: 12px;
       padding: 10px 12px; margin: 0 0 8px; display: flex; gap: 10px;
       align-items: flex-start; }
  li .who { flex: 1; min-width: 0; overflow-wrap: anywhere; }
  li .name { font-weight: 600; }
  li .addr { color: var(--dim); font-size: 13px; }
  li .meta { color: var(--dim); font-size: 12px; }
  li.bad { border-color: var(--bad); }
  .badge { display: inline-block; font-size: 11px; color: #180605; background: var(--bad);
           border-radius: 6px; padding: 1px 6px; margin-left: 6px; vertical-align: middle; }
  button { font: inherit; font-size: 13px; border-radius: 10px; padding: 8px 12px;
           border: 1px solid var(--line); background: var(--slab); color: var(--ink); }
  button.rm.armed { background: var(--bad); color: #180605; border-color: var(--bad); }
  button.go { background: var(--cyan); color: #041017; border: 0; font-weight: 600;
              padding: 10px 16px; }
  button:disabled { opacity: .45; }
  .add { background: var(--slab); border: 1px solid var(--line); border-radius: 12px;
         padding: 12px; display: grid; gap: 8px; }
  .add label { display: grid; gap: 3px; font-size: 12px; color: var(--dim); }
  input { font: inherit; font-size: 15px; padding: 9px 10px; border-radius: 10px;
          border: 1px solid var(--line); background: var(--bg); color: var(--ink); }
  input:focus { outline: none; border-color: var(--cyan); }
  input:invalid { border-color: var(--bad); }
  #path { color: var(--dim); font-size: 12px; margin: 10px 0 0; overflow-wrap: anywhere; }
  #keybox { margin: 12px 0; }
</style>
</head>
<body>
<main>
<h1>Address book <a href="/">&larr; Jarvis</a></h1>
<p class="hint">Who Jarvis can email a file to. Exact names only: two people who
share a first name are a question, never a guess. The address is shown to you and
never spoken.</p>
<div id="status" aria-live="polite"></div>
<div id="keybox" hidden>
  <label>Key <input id="key" type="password" autocomplete="off"
                    placeholder="the key from jarvis-phone.txt"></label>
</div>
<ul id="list"></ul>
<div class="add">
  <label>Name (first and last) <input id="name" type="text" maxlength="80"
                     autocomplete="off" autocapitalize="words"></label>
  <label>Email <input id="email" type="email" maxlength="254" autocomplete="off"
                      autocapitalize="none" inputmode="email"></label>
  <label>Honorific (optional: Dr, Prof) <input id="honorific" type="text"
                                     maxlength="12" autocomplete="off"></label>
  <label>Also answers to (optional, comma separated: my advisor, mum)
    <input id="aliases" type="text" autocomplete="off"></label>
  <label>Note (optional, never spoken) <input id="note" type="text" maxlength="200"
                                              autocomplete="off"></label>
  <button class="go" id="addbtn" type="button">Add to the book</button>
</div>
<p id="path"></p>
</main>
<script>
(function () {
  "use strict";
  var KEY_STORE = "jarvis.phone.key";
  var status = document.getElementById("status");
  var list = document.getElementById("list");
  var keybox = document.getElementById("keybox");
  var keyInput = document.getElementById("key");
  var addbtn = document.getElementById("addbtn");
  var pathEl = document.getElementById("path");
  var fields = ["name", "email", "honorific", "aliases", "note"];

  function readKey() {
    var q = new URLSearchParams(location.search).get("t");
    if (q) { try { localStorage.setItem(KEY_STORE, q); } catch (e) {} return q; }
    try { return localStorage.getItem(KEY_STORE) || ""; } catch (e) { return ""; }
  }
  var key = readKey();

  function say(text, cls) {
    status.textContent = text || "";
    status.className = cls || "";
  }

  function api(method, body) {
    var opts = { method: method, cache: "no-store",
                 headers: { "Authorization": "Bearer " + key } };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch("/api/contacts", opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (out) {
        if (r.status === 401) { keybox.hidden = false; throw new Error("that key was refused"); }
        if (!r.ok) { throw new Error(out.error || ("HTTP " + r.status)); }
        return out;
      });
    });
  }

  function text(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) { e.className = cls; }
    e.textContent = txt;
    return e;
  }

  function render(out) {
    list.textContent = "";
    var rows = out.contacts || [];
    rows.forEach(function (c) {
      var li = document.createElement("li");
      var who = text("div", "who", "");
      var name = text("div", "name", (c.honorific ? c.honorific + " " : "") + c.name);
      who.appendChild(name);
      who.appendChild(text("div", "addr", c.email));
      var meta = [];
      if (c.aliases && c.aliases.length) { meta.push("also: " + c.aliases.join(", ")); }
      if (c.note) { meta.push(c.note); }
      if (meta.length) { who.appendChild(text("div", "meta", meta.join(" \u00b7 "))); }
      li.appendChild(who);
      var rm = text("button", "rm", "Remove");
      rm.type = "button";
      rm.addEventListener("click", function () {
        /* Two taps, no confirm(): the first arms the button and says who
           it would remove; the second removes. Both the name and the
           address go up, so a stale list cannot remove the wrong row. */
        if (!rm.classList.contains("armed")) {
          rm.classList.add("armed");
          rm.textContent = "Really remove " + c.name + "?";
          setTimeout(function () { rm.classList.remove("armed"); rm.textContent = "Remove"; }, 6000);
          return;
        }
        rm.disabled = true;
        api("POST", { op: "remove", name: c.name, email: c.email })
          .then(function (o) { say("removed " + c.name, "ok"); render(o); })
          .catch(function (e) { say(e.message, "bad"); rm.disabled = false; });
      });
      li.appendChild(rm);
      list.appendChild(li);
    });
    (out.skipped || []).forEach(function (s) {
      var li = document.createElement("li");
      li.className = "bad";
      var who = text("div", "who", "");
      var name = text("div", "name", "row " + (s.index + 1) + (s.name ? ": " + s.name : ""));
      name.appendChild(text("span", "badge", "bad address, not used"));
      who.appendChild(name);
      who.appendChild(text("div", "meta", s.why + " \u2014 fix it in the file"));
      li.appendChild(who);
      list.appendChild(li);
    });
    if (!rows.length && !(out.skipped || []).length) {
      list.appendChild(text("li", "", "Nobody yet. Add someone below, or edit the file."));
    }
    pathEl.textContent = out.path ? "The book is " + out.path + " \u2014 plain JSON, edit it by hand if you like; Jarvis re-reads it on the next send." : "";
  }

  function load() {
    if (!key) { keybox.hidden = false; say("This page needs the phone key.", "bad"); return; }
    api("GET").then(function (o) { say(""); render(o); })
              .catch(function (e) { say(e.message, "bad"); });
  }

  keyInput.addEventListener("change", function () {
    key = keyInput.value.trim();
    try { localStorage.setItem(KEY_STORE, key); } catch (e) {}
    keyInput.value = "";
    keybox.hidden = true;
    load();
  });

  addbtn.addEventListener("click", function () {
    var email = document.getElementById("email");
    if (!email.checkValidity() || !email.value.trim()) {
      say("That is not an address: one at-sign, a dot in the domain, no spaces.", "bad");
      return;
    }
    var row = { op: "add" };
    fields.forEach(function (f) { row[f] = document.getElementById(f).value.trim(); });
    row.aliases = row.aliases ? row.aliases.split(",").map(function (a) { return a.trim(); })
                                         .filter(function (a) { return a; }) : [];
    addbtn.disabled = true;
    api("POST", row).then(function (o) {
      say("added " + row.name, "ok");
      fields.forEach(function (f) { document.getElementById(f).value = ""; });
      render(o);
    }).catch(function (e) { say(e.message, "bad"); })
      .then(function () { addbtn.disabled = false; });
  });

  load();
})();
</script>
</body>
</html>
"""
