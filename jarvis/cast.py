"""What a throw carries, and where it can honestly land.

He asked, verbatim: "reach out and grab at the screen (in the air) where the
camera is and then gesture towards almost throwing the cast onto the
HPCOMPUTER." The hand part -- reach, fist, fling -- lives in jarvis/gesture.py
and knows nothing about payloads or targets. THIS module is the other half:
what a closed fist is holding (a :class:`CastSubject`), where an open hand
sends it (a sink), and what is said when it cannot get there.

A grab carries no noun, so the subject is resolved from ambient state, in a
fixed order, first hit wins:

    HELD      already carrying: a second grab is a no-op
    DOCUMENT  the file he last had explained, inside commander's own
              900 s window (DOCUMENT_MAX_AGE_S mirrors LAST_DOCUMENT_S)
    TRACK     Spotify, only while it is actually playing
    SCREEN    the focused window's title -- the floor, it nearly always hits
    EMPTY     only when X itself is gone

and it is resolved on the REACH, not the grab: at the measured ~7.5 fps a
reach is 2-3 frames (~300-400 ms) of warning, and spending them on a Spotify
call is what lets the fist closing feel instant. The screen subject carries
its NAME on that path and takes its BYTES (``materialise``) only when a sink
actually wants them: a full 3840x2160 grab on every reach would put a
visible hole in the metaphor.

THE FIVE PROMISES the suite pins (tests/test_cast.py), because none of them
survives on good intentions:

* A held cast never reports success. ``CastResult.landed`` and ``.held``
  are separate booleans, never inferred from each other, and the
  constructor refuses both at once. The worst outcome this feature has is
  Jarvis saying a thing arrived somewhere it did not.
* The board sink cannot open a window. The 2026-08-26 desktop freeze came
  from window churn on :1, and the ruling that a board cast fires at once
  with no read-back rests entirely on a wrong one being cheap: a panel
  change on Jarvis's own surface. A source test forbids the vocabulary.
* An irreversible sink is unreachable by gesture. A fling is a much weaker
  signal of intent than a sentence, so ``cast()`` only ever PROPOSES such a
  send through the ``propose`` callable (commander.stash_destructive, the
  same read-back-then-yes machinery outbox.py uses on the file-and-remote
  branch); with no ``propose`` wired it refuses, never "does it anyway".
* No URL is ever built from an mDNS name. Measured: ``avahi-resolve -n
  spark-509f.local`` answers 172.17.0.1, the docker0 bridge, so a handoff
  URL built from the .local name is unreachable from the machine it is for.
  The handoff sink takes a numeric LAN address (webapp.lan_address asks the
  kernel for the default-route source and resolves nothing).
* Nothing in the suite touches the network. Every probe, clock, capture and
  publisher is injected; the live TCP probe is the default only, and a test
  asserts that constructing a sink does not call it.

WHAT CAN CATCH IT TODAY (measured 2026-09-03, ~00:45, network + filesystem
only; no camera device opened, no frame viewed):

* ``board`` -- his own console surface on the Spark. WORKS. Reversible, so
  it fires immediately and asks nothing.
* ``handoff`` -- the Spark SERVES a page (the way jarvis/webapp.py already
  does, one private address, token in the URL) and HPCOMPUTER FETCHES it.
  WORKS in principle today because a Windows firewall blocks inbound and
  leaves outbound open; it needs him to open a URL there, so it is honest
  to call it a handoff rather than a cast, and it leaves the box, so it is
  read back.
* ``hpcomputer`` -- DOES NOT WORK. 192.168.50.114 answers ARP (REACHABLE
  under probe at 00:45: powered on, on the LAN), ping is 100% loss, and
  none of 22/445/3389/5900/8008 (nor 2343, the one port it advertises)
  answers; this module's own probe of 22/445/3389 timed out again at 02:40
  and 07:21. It is not on the tailnet. So the cast is
  HELD, Jarvis says so out loud with the LIVE reason, and the payload falls
  back to the board. The one payload that does reach it is a TRACK: its
  Spotify client makes an OUTBOUND connection, which the firewall does not
  block, and "play it on hpcomputer" already works by voice. OpenSSH Server
  is being enabled on that machine by Hunter himself; ``HpcomputerSink``
  keeps a ``transport`` seam for it and implements no transport, because
  one that cannot be tested against the real host would only be a guess.

Identity is CONSULTED, never required: SFace scores 0.85 mean cosine on
uniform noise against a 0.363 same-person bar, and camera.identity ships
off, so requiring it would ship a feature that never fires. A DIFFERENT
name is a veto; an empty one is no opinion; a positive match is demanded
by ``needs_identity``, which is ON for the one sink that would push bytes
off this box under Jarvis's own hand (hpcomputer, the "scp" case) and OFF
for the handoff, where nothing moves until he fetches it himself from the
other chair, the URL is token-locked to the private LAN, expires and can
be revoked. Both are constructor flags, so the wiring can tighten either.

Directions: the sector map (``gesture.sinks``) SHIPS EMPTY, so every fling
lands on the board until he says which side HPCOMPUTER is on -- that is his
open question, nobody else can see the room, and a direction map for a set
of size one is wrong-target risk for no benefit. ``parse_side_teaching`` and
``teach_sink`` make it teachable by voice ("HPCOMPUTER is on my right").
Degrees arriving here are already in HIS frame -- 0 his right, 90 up -- the
single sign flip lives in gesture.to_his_frame, and nothing here may flip
another.

Clocks: the sinks and the outbox take a monotonic ``now``; ``resolve_subject``
takes WALL-CLOCK seconds because commander stamps ``_last_document`` with
``datetime.now().timestamp()``. Mixing them ages every document 15 minutes
in one go, so the parameter names say which.
"""
from __future__ import annotations

import errno
import functools
import ipaddress
import re
import secrets
import select
import socket
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Protocol

from jarvis.logs import get_logger

log = get_logger("cast")

# ---------------------------------------------------------------- numbers
# Mirrors commander.LAST_DOCUMENT_S (900.0): the window inside which "read
# it to me" still means the document he last had explained. One staleness
# rule, not two -- tests/test_cast.py asserts the two agree.
DOCUMENT_MAX_AGE_S = 900.0
# The live LAN probe. 0.35 s is the whole budget for every port at once
# (they are tried concurrently, one deadline); a cast is a physical metaphor
# and a two-second stall inside it reads as the feature being broken. The
# answer is cached for PROBE_CACHE_S so a grab-throw-grab-throw does not
# probe four times. Both GUESSED; neither has been tuned against his hand.
PROBE_TIMEOUT_S = 0.35
PROBE_CACHE_S = 15.0
# A second refusal inside this window is shortened to a few words: the
# first one carried the reason, and repeating the whole sentence every
# throw would teach him to stop listening to it.
REFUSAL_REPEAT_S = 60.0
# How long a held cast is kept before it is let go (GUESSED: a cast held
# before lunch should not be offered at dinner), and how many are kept.
HELD_TTL_S = 1800.0
HELD_CAPACITY = 3
# How long a handoff URL stays served (GUESSED; long enough to walk to the
# other desk and open it, short enough that a forgotten one goes dark).
HANDOFF_TTL_S = 900.0

# HPCOMPUTER, by NUMBER. Measured 2026-09-03: hpcomputer.local -> 192.168.50.114,
# ARP REACHABLE, no port open. Never the .local name (see the docstring).
HPCOMPUTER_HOST = "192.168.50.114"
# 22 because OpenSSH Server is the unblock in progress; 445 and 3389 because
# either answering means the Windows firewall profile has changed. Probed
# concurrently inside one PROBE_TIMEOUT_S, so the count does not add delay.
HPCOMPUTER_PORTS = (22, 445, 3389)
# The Spotify Connect device name, exactly as tools/spotify.DEFAULT_DEVICE.
HPCOMPUTER_DEVICE = "HPCOMPUTER"

# Mirrors webapp.DEFAULT_PORT; the test asserts they agree.
HANDOFF_PORT = 8765
HANDOFF_PATH = "/cast"

# The config key for the direction -> sink map. Ships EMPTY.
OPTION_SINKS = "gesture.sinks"
DEFAULT_SINK = "board"
# Only the horizontal axis carries a target: at reach distance the frame is
# 4.6 hand-units wide and 2.6 tall (gesture.py's measurement), DOWN is the
# cancel, and UP is reported but is not a target. A map entry for either
# vertical word is ignored, not honoured.
ROUTABLE_DIRECTIONS = ("left", "right")

# The enrolled name eye.resolve_wake accepts. Empty = no opinion.
OWNER_IDENTITY = "hunter"

# Which tone each outcome plays. Four DIFFERENT names on purpose:
# earcons.py drops a repeat of the SAME tone inside its 4 s cooldown, so a
# grab and a throw 1.5 s apart sharing a tone would silence the second.
EARCONS = {"grab": "heard-you", "landed": "done", "held": "warning",
           "drop": "held-back"}

# --------------------------------------------------------------- wording
HOLDING_LINE = "Holding {what}, sir."
NOTHING_LINE = "I've nothing in hand, sir."
BOARD_LINE = "On the board, sir."
BOARD_DARK_LINE = "The board wouldn't take it, sir; I've kept it."
BOARD_NO_SURFACE = "the board has no surface to write to"
# HPCOMPUTER. {reason} is the LIVE probe's own words, never a constant: the
# day Windows opens a port, this sentence must change on its own.
HELD_LINE = "HPCOMPUTER isn't answering, sir — {reason}. I've kept it here."
HELD_AGAIN_LINE = "Still nothing listening, sir."
OPEN_PORT_LINE = ("HPCOMPUTER has port {port} open, sir, but I've no way in "
                  "yet. I've kept it here.")
OPEN_PORT_REASON = "port {port} is open but I've no way to hand it anything yet"
NO_TRANSPORT_REASON = "I've no way to hand it anything yet"
NO_BYTES_LINE = "I've nothing of {what} to send, sir; I've kept the name."
SENT_LINE = "{What} is on HPCOMPUTER, sir."
REFUSED_LINE = "HPCOMPUTER wouldn't take it, sir; I've kept it here."
TRACK_MOVED_LINE = "{What}, on HPCOMPUTER, sir."
TRACK_NO_ROUTE_LINE = "I can't move the music from here, sir."
TRACK_OFF_LINE = "The music stays where it is, sir; moving it by hand is off."
TRACK_FAILED_LINE = "Spotify wouldn't move it, sir."
# The handoff page.
HANDOFF_LINE = "It's waiting on the Spark's page, sir — the address is on the board."
HANDOFF_NO_BYTES_LINE = "There's nothing of {what} I can serve, sir."
HANDOFF_DARK = "the page isn't being served"
HANDOFF_NO_ADDRESS = "this box has no private address to serve from"
HANDOFF_NO_TOKEN = "there's no token to lock the page with"
HANDOFF_NOWHERE = "there's nowhere to show the address"
HANDOFF_PUBLISH_FAILED_LINE = ("I couldn't put the address up, sir, so I've "
                               "taken the page down.")
# cast()
READBACK_LINE = "{What} to {target}, sir. Shall I send it?"
NO_PROPOSE_LINE = "I won't send that on a wave alone, sir; say the word and I will."
IDENTITY_LINE = ("I'd want to be sure it was you before I sent that, sir, "
                 "and I can't see well enough to say.")
SINK_RAISED_LINE = "{Target} wouldn't take it, sir; I've kept it."
UNTAUGHT_LINE = ("That went to the board, sir. Tell me which side HPCOMPUTER "
                 "is on and I'll throw there next time.")
# Teaching a side by voice.
TAUGHT_LINE = "{Direction} is {target} from now on, sir."
TAUGHT_UNKNOWN_LINE = "I don't know a target called {name}, sir."
TAUGHT_FAILED_LINE = "I couldn't save that, sir."
# The held register.
READY_LINE = "{Target} is listening now, sir — shall I send {what} over?"
EXPIRED_LINE = "I've let {what} go, sir; {target} never came back."
EVICTED_LINE = "I've let {what} go to make room, sir."

SINK_LABELS = {"board": "the board", "hpcomputer": "HPCOMPUTER",
               "handoff": "the handoff page"}


def _cap(text: str) -> str:
    text = str(text or "")
    return text[:1].upper() + text[1:] if text else text


def sink_label(name: str) -> str:
    return SINK_LABELS.get(str(name or "").strip().lower(), str(name or ""))


# ---------------------------------------------------------------- subject
@dataclass(frozen=True)
class CastSubject:
    """What the fist is holding. Strings, a path and a stamp -- never bytes.

    ``kind`` is one of document | track | screen | empty. ``spoken`` is what
    Jarvis says he picked up and is fixed at resolution: ``materialise``
    may add a path later but must never change the name he already heard.
    """
    kind: str
    spoken: str
    path: Optional[Path] = None      # the document, or the screen PNG
    uri: str = ""                    # spotify:track:... when known
    mime: str = ""
    at: float = 0.0

    @property
    def holdable(self) -> bool:
        return self.kind != "empty"


def empty_subject(at: float = 0.0) -> CastSubject:
    return CastSubject(kind="empty", spoken="", at=float(at))


_WINDOW_SEPARATORS = (" - ", " — ", " – ", " | ", " :: ", " · ")
_TRAILING_PAREN_RX = re.compile(r"\s*\([^()]*\)\s*$")
_EXTENSION_RX = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,4}$")
_EDGE_JUNK_RX = re.compile(r"^[\s\-—–•●*:|]+|[\s\-—–•●*:|]+$")


def window_spoken_name(title, max_words: int = 6) -> str:
    """The noun in an X11 title, fit to be said and shown on a chip.

    A raw title like ``Good evening Ali and Heather.txt (~/) - Text Editor``
    is unspeakable. Keep the segment before the first separator (the
    application name follows it), drop a trailing path parenthetical and
    a file extension, then cap the word count. Never a trailing separator,
    never None.
    """
    text = " ".join(str(title or "").split())
    if not text:
        return ""
    cut = len(text)
    for sep in _WINDOW_SEPARATORS:
        idx = text.find(sep)
        if idx >= 0:
            cut = min(cut, idx)
    head = text[:cut]
    head = _TRAILING_PAREN_RX.sub("", head)
    head = _EXTENSION_RX.sub("", head)
    head = _EDGE_JUNK_RX.sub("", head)
    words = head.split()
    if max_words and max_words > 0:
        words = words[:int(max_words)]
    return " ".join(words)


def document_spoken_name(path) -> str:
    """Mirror of commander._spoken_name: the stem with _ and - as spaces."""
    stem = getattr(path, "stem", None)
    if stem is None:
        stem = Path(str(path)).stem
    text = re.sub(r"[_\-]+", " ", str(stem))
    return " ".join(text.split())


_TRACK_TAILS = (", sir", " — on ", " - on ", ", paused on ", " on HPCOMPUTER")


def track_spoken_name(line) -> str:
    """'Kashmir by Led Zeppelin, sir — on HPCOMPUTER.' -> the track and
    artist only. tools/spotify's NOW_LINE puts ', sir' straight after them,
    so that is the first cut; the others cover a line shaped differently."""
    text = " ".join(str(line or "").split())
    for tail in _TRACK_TAILS:
        idx = text.find(tail)
        if idx >= 0:
            text = text[:idx]
    return text.strip().rstrip(".,;:— -").strip()


def subject_line(subject: CastSubject) -> str:
    """What is said when the fist closes: 'Holding the lab report, sir.'"""
    if subject is None or not subject.holdable or not subject.spoken:
        return NOTHING_LINE
    return HOLDING_LINE.format(what=subject.spoken)


# ---------------------------------------------------------- the providers
def _document_from(commander, now: float) -> Optional[CastSubject]:
    """commander._last_document = (path, wall-clock stamp), if fresh."""
    last = getattr(commander, "_last_document", None)
    if not last:
        return None
    try:
        path, when = last
    except (TypeError, ValueError):
        return None
    if path is None:
        return None
    try:
        age = float(now) - float(when)
    except (TypeError, ValueError):
        return None
    if age > DOCUMENT_MAX_AGE_S:
        return None
    path = Path(str(path)) if not isinstance(path, Path) else path
    spoken = document_spoken_name(path) or path.name
    return CastSubject(kind="document", spoken=spoken, path=path,
                       mime=_mime_of(path), at=float(now))


def _track_from(spotify, now: float) -> Optional[CastSubject]:
    """The playing track, or None. ``music_playing()`` is a cache read and
    is consulted first so a paused deck costs no API call."""
    if spotify is None:
        return None
    playing = getattr(spotify, "music_playing", None)
    if not callable(playing):
        spotify = getattr(spotify, "spotify", None)      # a services bag
        playing = getattr(spotify, "music_playing", None)
        if not callable(playing):
            return None
    if not playing():
        return None
    now_playing = getattr(spotify, "now_playing", None)
    if not callable(now_playing):
        return None
    res = now_playing()
    line = getattr(res, "speak", None) or getattr(res, "text", None) or res
    spoken = track_spoken_name(line)
    if not spoken or spoken.lower().startswith("nothing"):
        return None
    uri = str(getattr(res, "uri", "") or "")
    return CastSubject(kind="track", spoken=spoken, uri=uri, at=float(now))


def focused_window_title(timeout_s: float = 0.5) -> Optional[str]:
    """The focused X11 window's title via xdotool; '' for a window with no
    name, None when X (or xdotool) is not there. Bounded by ``timeout_s``
    because this runs on the reach path."""
    import subprocess                                    # noqa: PLC0415
    try:
        proc = subprocess.run(["xdotool", "getactivewindow", "getwindowname"],
                              capture_output=True, text=True, errors="replace",
                              timeout=float(timeout_s))
    except Exception:  # noqa: BLE001 - no X, no xdotool, or it hung
        log.debug("focused window title unavailable", exc_info=True)
        return None
    if proc.returncode != 0:
        return None
    return " ".join(proc.stdout.split())[:200]


def screen_subject(title: Optional[str], now: float = 0.0) -> Optional[CastSubject]:
    """A screen subject from a title, or None when there is no screen. An
    empty title with X present is still a screen ('the screen')."""
    if title is None:
        return None
    spoken = window_spoken_name(title) or "the screen"
    return CastSubject(kind="screen", spoken=spoken, at=float(now))


def focused_screen_subject(now: float = 0.0) -> Optional[CastSubject]:
    """The default SCREEN provider: the name only, never the bytes."""
    return screen_subject(focused_window_title(), now)


def resolve_subject(commander=None, spotify=None, *, now: Optional[float] = None,
                    providers: Optional[dict] = None,
                    held: Optional[CastSubject] = None) -> CastSubject:
    """HELD -> DOCUMENT -> TRACK -> SCREEN -> EMPTY. First hit wins.

    Runs on the REACH, so it must be cheap and must never raise: every rung
    is guarded and a rung that fails falls through to the next. ``now`` is
    WALL-CLOCK seconds (commander stamps documents with
    ``datetime.now().timestamp()``); ``providers`` overrides any rung by
    name ("document" | "track" | "screen") with a zero-argument callable
    returning a CastSubject or None -- that is how the app wires the real
    sources in and how the suite keeps the network and the display out.
    """
    stamp = time.time() if now is None else float(now)
    if held is not None and getattr(held, "holdable", False):
        return held
    providers = providers if isinstance(providers, dict) else {}
    rungs = (
        ("document", providers.get("document"),
         lambda: _document_from(commander, stamp)),
        ("track", providers.get("track"), lambda: _track_from(spotify, stamp)),
        ("screen", providers.get("screen"),
         lambda: focused_screen_subject(stamp)),
    )
    for name, override, default in rungs:
        fn = override if callable(override) else default
        try:
            got = fn()
        except Exception:  # noqa: BLE001 - a rung may not break the ladder
            log.debug("cast: %s provider failed", name, exc_info=True)
            continue
        if isinstance(got, CastSubject) and got.holdable:
            return got
    return empty_subject(stamp)


def _mime_of(path: Optional[Path]) -> str:
    if path is None:
        return ""
    ext = path.suffix.lower()
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".pdf": "application/pdf", ".txt": "text/plain",
            ".md": "text/markdown"}.get(ext, "application/octet-stream")


def materialise(subject: CastSubject,
                capture: Optional[Callable[[], Optional[dict]]] = None) -> CastSubject:
    """Give a screen subject its BYTES, lazily.

    ``capture`` is context.capture_screen: it writes latest.png and answers
    {'active_window', 'geometry', 'screenshot'}. Only a screen subject with
    no path is touched; the spoken name he already heard is never changed;
    a capture that fails or raises leaves the subject as it was, without
    bytes, so the sink can say so rather than crash.
    """
    if subject is None or subject.kind != "screen" or subject.path is not None:
        return subject
    if capture is None:
        return subject
    try:
        got = capture()
    except Exception:  # noqa: BLE001 - the display may be gone
        log.debug("cast: screen capture failed", exc_info=True)
        return subject
    if not isinstance(got, dict):
        return subject
    raw = got.get("screenshot") or got.get("path") or ""
    if not raw:
        return subject
    path = Path(str(raw))
    return replace(subject, path=path, mime=_mime_of(path) or "image/png")


# --------------------------------------------------------------- results
@dataclass(frozen=True)
class CastResult:
    """What became of a delivery. ``landed`` = it is THERE. ``held`` = kept,
    deliberately, and he is told. Independent on purpose, never inferred
    from each other, and never both -- the constructor refuses that."""
    landed: bool
    held: bool
    spoken: str
    detail: str = ""
    sink: str = ""
    fallback: str = ""           # where a held payload should go instead

    def __post_init__(self):
        if self.landed and self.held:
            raise ValueError("a cast cannot be both landed and held")
        if self.held and not self.spoken:
            raise ValueError("a held cast must say so")


def held_result(spoken: str, *, sink: str = "", detail: str = "",
                fallback: str = "") -> CastResult:
    return CastResult(landed=False, held=True, spoken=spoken, detail=detail,
                      sink=sink, fallback=fallback)


def landed_result(spoken: str, *, sink: str = "", detail: str = "") -> CastResult:
    return CastResult(landed=True, held=False, spoken=spoken, detail=detail,
                      sink=sink)


class Sink(Protocol):
    """Where a throw can land. Each one must be able to say it is
    unavailable and WHY, in words that can be spoken."""
    name: str
    label: str
    reversible: bool
    needs_identity: bool
    wants_bytes: bool

    def available(self) -> tuple[bool, str]: ...
    def needs_readback(self, subject: Optional[CastSubject] = None) -> bool: ...
    def deliver(self, subject: CastSubject) -> CastResult: ...


# ------------------------------------------------------------- the board
class BoardSink:
    """Jarvis's own console surface. Reversible, so it fires at once.

    ``publish`` is handed the CastSubject and writes it into the EXISTING
    board (BoardUpdate / a panel row). It must never create a window, raise
    one, or steal focus: the no-read-back ruling depends on a wrong cast
    costing three seconds of his attention and nothing more. ``console_visible``
    (optional) says whether the console is on top; when it is, the thing
    appearing IS the feedback and nothing is said -- "On the board, sir."
    every time would wear out in a day.
    """
    name = "board"
    label = SINK_LABELS["board"]
    reversible = True
    needs_identity = False
    wants_bytes = False

    def __init__(self, publish: Optional[Callable[[CastSubject], None]] = None,
                 *, console_visible: Optional[Callable[[], bool]] = None):
        self._publish = publish
        self._visible = console_visible

    def available(self) -> tuple[bool, str]:
        if not callable(self._publish):
            return False, BOARD_NO_SURFACE
        return True, ""

    def needs_readback(self, subject: Optional[CastSubject] = None) -> bool:
        return False

    def deliver(self, subject: CastSubject) -> CastResult:
        ok, why = self.available()
        if not ok:
            return held_result(BOARD_DARK_LINE, sink=self.name, detail=why)
        try:
            self._publish(subject)
        except Exception as exc:  # noqa: BLE001 - the surface may be gone
            log.warning("cast: board publish failed: %s", exc)
            return held_result(BOARD_DARK_LINE, sink=self.name, detail=str(exc))
        spoken = BOARD_LINE
        if callable(self._visible):
            try:
                if self._visible():
                    spoken = ""
            except Exception:  # noqa: BLE001 - visibility is a nicety
                pass
        return landed_result(spoken, sink=self.name)


def board_card(subject: CastSubject) -> dict:
    """Strings and numbers only, for whatever draws the board row."""
    return {"kind": subject.kind, "spoken": subject.spoken,
            "path": str(subject.path) if subject.path else "",
            "at": round(float(subject.at), 3)}


# ----------------------------------------------------------- the probe
def _numeric_host(host: str) -> Optional[str]:
    """The host as an IP literal, or None. A name is refused rather than
    resolved: that is the whole .local rule."""
    text = str(host or "").strip()
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return None


def probe_tcp(host: str, ports, timeout: float = PROBE_TIMEOUT_S) -> tuple[bool, str, str]:
    """Is anything on ``host`` accepting TCP on any of ``ports``?

    -> (ok, reason, detail). ``detail`` is the port that answered; ``reason``
    is the spoken explanation when none did. All ports are tried at once
    against ONE deadline, so the worst case is ``timeout`` however many
    there are. A refusal (RST) means the host is awake and answering IP
    with nothing on that port; a timeout on every port is what a firewall
    DROP -- or a machine that is off -- looks like, and TCP alone cannot
    tell those apart, so the reason does not pretend to.
    """
    ports = tuple(int(p) for p in (ports or ()))
    if not ports:
        return False, "no port was tried", ""
    addr = _numeric_host(host)
    if addr is None:
        return False, "I only probe numeric addresses", ""
    family = socket.AF_INET6 if ":" in addr else socket.AF_INET
    pending: dict = {}
    refused = False
    try:
        for port in ports:
            try:
                s = socket.socket(family, socket.SOCK_STREAM)
            except OSError:
                continue
            s.setblocking(False)
            err = s.connect_ex((addr, port))
            if err == 0:
                s.close()
                return True, "", str(port)
            if err in (errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EAGAIN):
                pending[s] = port
                continue
            if err == errno.ECONNREFUSED:
                refused = True
            s.close()
        deadline = time.monotonic() + max(float(timeout), 0.0)
        while pending:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            try:
                _, writable, _ = select.select([], list(pending), [], left)
            except (OSError, ValueError):
                break
            if not writable:
                break
            for s in writable:
                port = pending.pop(s)
                err = s.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                s.close()
                if err == 0:
                    return True, "", str(port)
                if err == errno.ECONNREFUSED:
                    refused = True
    finally:
        for s in pending:
            try:
                s.close()
            except OSError:
                pass
    if refused:
        return False, "it's awake, but nothing on it is listening", ""
    return False, "nothing on it is listening", ""


# -------------------------------------------------------------- HPCOMPUTER
class HpcomputerSink:
    """The machine he actually threw at. HELD today, honestly.

    ``probe()`` -> (ok, reason, detail) is the live reachability reading,
    cached for ``cache_s``; the default is :func:`probe_tcp` against
    HPCOMPUTER_HOST and is never called by construction. ``transfer(device)``
    is the Spotify Connect route (tools/spotify control('transfer',
    'HPCOMPUTER')) -- the one payload that reaches that box today, because
    its Spotify client connects OUTBOUND. ``transport(subject)`` is the seam
    for the SSH push Hunter is enabling on his side (h2pey@192.168.50.114,
    key ~/.ssh/hpcomputer; on the file-and-remote branch, remote.push is the
    shape). It is None here on purpose: a transport that could not be run
    against the real host would be a guess dressed as a feature.

    ``needs_identity`` is True because a file leaving the box is the
    irreversible case the safety lane named ("scp"); it only bites on the
    read-back path, which today never opens because ``available()`` is
    False until a transport exists AND the probe answers.
    """
    name = "hpcomputer"
    label = SINK_LABELS["hpcomputer"]
    reversible = False
    wants_bytes = True

    def __init__(self, *, probe: Optional[Callable[[], tuple]] = None,
                 now: Optional[Callable[[], float]] = None,
                 cache_s: float = PROBE_CACHE_S,
                 transfer: Optional[Callable[[str], object]] = None,
                 allow_track: bool = True,
                 transport: Optional[Callable[[CastSubject], object]] = None,
                 needs_identity: bool = True,
                 host: str = HPCOMPUTER_HOST, ports=HPCOMPUTER_PORTS):
        self._probe = probe if callable(probe) else functools.partial(
            probe_tcp, host, tuple(ports), timeout=PROBE_TIMEOUT_S)
        self._now = now if callable(now) else time.monotonic
        self._cache_s = float(cache_s)
        self._transfer = transfer if callable(transfer) else None
        self._allow_track = bool(allow_track)
        self._transport = transport if callable(transport) else None
        self.needs_identity = bool(needs_identity)
        self._cached: Optional[tuple[float, tuple]] = None
        self._last_refusal_at: Optional[float] = None
        self.host = host

    # -- reachability -------------------------------------------------
    def reach(self) -> tuple[bool, str, str]:
        """The cached live reading; a probe that raises reads as unreachable
        with its own words, never as reachable."""
        now = self._now()
        if self._cached is not None and now - self._cached[0] <= self._cache_s:
            return self._cached[1]
        try:
            got = self._probe()
            ok, reason, detail = bool(got[0]), str(got[1] or ""), str(got[2] or "")
        except Exception as exc:  # noqa: BLE001 - a probe boundary
            log.debug("cast: hpcomputer probe failed", exc_info=True)
            ok, reason, detail = False, f"I couldn't reach it ({exc})", ""
        if not ok and not reason:
            reason = "nothing on it is listening"
        self._cached = (now, (ok, reason, detail))
        return ok, reason, detail

    @property
    def track_route(self) -> bool:
        return self._allow_track and self._transfer is not None

    def available(self) -> tuple[bool, str]:
        """Can a FILE land there? Needs a transport AND a live answer."""
        ok, reason, detail = self.reach()
        if not ok:
            return False, reason
        if self._transport is None:
            return False, OPEN_PORT_REASON.format(port=detail or "?")
        return True, ""

    def needs_readback(self, subject: Optional[CastSubject] = None) -> bool:
        if subject is not None and subject.kind == "track" and self.track_route:
            return False
        return True

    # -- delivery -----------------------------------------------------
    def deliver(self, subject: CastSubject) -> CastResult:
        if subject.kind == "track":
            return self._deliver_track(subject)
        ok, reason, detail = self.reach()
        if not ok:
            return held_result(self._refusal(reason), sink=self.name,
                               detail=reason, fallback=DEFAULT_SINK)
        if self._transport is None:
            return held_result(OPEN_PORT_LINE.format(port=detail or "?"),
                               sink=self.name, detail=f"open:{detail}",
                               fallback=DEFAULT_SINK)
        if subject.path is None:
            return held_result(NO_BYTES_LINE.format(what=subject.spoken),
                               sink=self.name, detail="no-bytes",
                               fallback=DEFAULT_SINK)
        try:
            out = self._transport(subject)
        except Exception as exc:  # noqa: BLE001 - the transport boundary
            log.warning("cast: hpcomputer transport failed: %s", exc)
            return held_result(REFUSED_LINE, sink=self.name, detail=str(exc),
                               fallback=DEFAULT_SINK)
        return landed_result(SENT_LINE.format(What=_cap(subject.spoken)),
                             sink=self.name, detail=str(out or ""))

    def _deliver_track(self, subject: CastSubject) -> CastResult:
        if not self._allow_track:
            return held_result(TRACK_OFF_LINE, sink=self.name, detail="track-off")
        if self._transfer is None:
            return held_result(TRACK_NO_ROUTE_LINE, sink=self.name,
                               detail="no-transfer")
        try:
            self._transfer(HPCOMPUTER_DEVICE)
        except Exception as exc:  # noqa: BLE001 - Spotify's boundary
            log.warning("cast: spotify transfer failed: %s", exc)
            return held_result(TRACK_FAILED_LINE, sink=self.name, detail=str(exc))
        return landed_result(TRACK_MOVED_LINE.format(What=_cap(subject.spoken)),
                             sink=self.name, detail=HPCOMPUTER_DEVICE)

    def _refusal(self, reason: str) -> str:
        now = self._now()
        last = self._last_refusal_at
        self._last_refusal_at = now
        if last is not None and now - last <= REFUSAL_REPEAT_S:
            return HELD_AGAIN_LINE
        return HELD_LINE.format(reason=reason)


# ------------------------------------------------------------ the handoff
def _private_address(host: str) -> bool:
    """Mirror of webapp.is_private_host: an IP literal that is not global.
    A name is refused, not resolved."""
    addr = _numeric_host(host)
    if addr is None:
        return False
    ip = ipaddress.ip_address(addr)
    if ip.is_unspecified or ip.is_multicast:
        return False
    return not ip.is_global


def _lan_address_default() -> str:
    from jarvis import webapp                            # noqa: PLC0415
    return webapp.lan_address()


class HandoffSink:
    """The Spark serves, HPCOMPUTER fetches.

    ``address()`` is the numeric LAN address (webapp.lan_address by default,
    which asks the kernel and resolves nothing); ``port()`` the served port;
    ``token()`` the phone client's bearer token; ``serving()`` whether the
    page is actually up; ``publish_url(url, subject)`` puts the address
    somewhere he can read it from the other chair (the board). The sink
    registers the file under a short id -- ``path_for(ident, token)`` is
    what the page endpoint (a later step) answers from -- and the URL it
    builds is ``http://<ip>:<port>/cast/<id>?t=<token>``. Never a name: an
    ``address()`` that hands back one is refused as no address at all.

    ``reversible`` is False because a fetched file cannot be un-fetched, so
    a fling only PROPOSES this and he answers the read-back. ``needs_identity``
    defaults False: nothing leaves until he fetches it himself, the page is
    token-locked, private-only, time-limited and revocable, and with
    camera.identity shipping off True would make the one LAN sink that
    works today unreachable by gesture. His to tighten.
    """
    name = "handoff"
    label = SINK_LABELS["handoff"]
    reversible = False
    wants_bytes = True

    def __init__(self, *, address: Optional[Callable[[], str]] = None,
                 port: Optional[Callable[[], int]] = None,
                 token: Optional[Callable[[], str]] = None,
                 serving: Optional[Callable[[], bool]] = None,
                 publish_url: Optional[Callable[[str, CastSubject], None]] = None,
                 now: Optional[Callable[[], float]] = None,
                 ttl_s: float = HANDOFF_TTL_S, needs_identity: bool = False):
        self._address = address if callable(address) else _lan_address_default
        self._port = port if callable(port) else (lambda: HANDOFF_PORT)
        self._token = token if callable(token) else (lambda: "")
        self._serving = serving if callable(serving) else (lambda: False)
        self._publish_url = publish_url if callable(publish_url) else None
        self._now = now if callable(now) else time.monotonic
        self._ttl_s = float(ttl_s)
        self.needs_identity = bool(needs_identity)
        self._served: dict[str, tuple[Path, CastSubject, float]] = {}

    def needs_readback(self, subject: Optional[CastSubject] = None) -> bool:
        return True

    def available(self) -> tuple[bool, str]:
        try:
            if not self._serving():
                return False, HANDOFF_DARK
            addr = str(self._address() or "")
        except Exception as exc:  # noqa: BLE001 - the server's boundary
            return False, f"the page can't be reached ({exc})"
        if not addr or not _private_address(addr):
            return False, HANDOFF_NO_ADDRESS
        if not str(self._token() or ""):
            return False, HANDOFF_NO_TOKEN
        if self._publish_url is None:
            return False, HANDOFF_NOWHERE
        return True, ""

    def url_for(self, ident: str) -> str:
        addr = _numeric_host(self._address()) or ""
        if not addr:
            return ""
        host = f"[{addr}]" if ":" in addr else addr
        return f"http://{host}:{int(self._port())}{HANDOFF_PATH}/{ident}?t={self._token()}"

    def deliver(self, subject: CastSubject) -> CastResult:
        ok, why = self.available()
        if not ok:
            return held_result(f"{_cap(why)}, sir; I've kept it.", sink=self.name,
                               detail=why, fallback=DEFAULT_SINK)
        # A path is bytes-in-waiting; whether the file is still there is
        # checked at fetch time (path_for), where a vanished file is an
        # honest 404 rather than a stale promise here.
        path = subject.path
        if path is None:
            return held_result(HANDOFF_NO_BYTES_LINE.format(what=subject.spoken),
                               sink=self.name, detail="no-bytes",
                               fallback=DEFAULT_SINK)
        self.sweep()
        ident = secrets.token_urlsafe(6)
        url = self.url_for(ident)
        if not url or ".local" in url:
            return held_result(f"{_cap(HANDOFF_NO_ADDRESS)}, sir; I've kept it.",
                               sink=self.name, detail="no-url", fallback=DEFAULT_SINK)
        self._served[ident] = (Path(path), subject, self._now())
        try:
            self._publish_url(url, subject)
        except Exception as exc:  # noqa: BLE001 - the board's boundary
            self._served.pop(ident, None)
            log.warning("cast: handoff publish failed: %s", exc)
            return held_result(HANDOFF_PUBLISH_FAILED_LINE, sink=self.name,
                               detail=str(exc), fallback=DEFAULT_SINK)
        return landed_result(HANDOFF_LINE, sink=self.name, detail=url)

    # -- what the page endpoint reads ---------------------------------
    def sweep(self) -> None:
        now = self._now()
        for ident in [k for k, v in self._served.items() if now - v[2] > self._ttl_s]:
            self._served.pop(ident, None)

    def served(self) -> dict:
        self.sweep()
        return {k: (str(p), s.spoken) for k, (p, s, _) in self._served.items()}

    def path_for(self, ident: str, token: str) -> Optional[Path]:
        """The file behind a served id, only with the right token."""
        self.sweep()
        import hmac                                        # noqa: PLC0415
        want = str(self._token() or "")
        if not want or not hmac.compare_digest(want.encode(), str(token or "").encode()):
            return None
        got = self._served.get(str(ident))
        if got is None or not got[0].is_file():
            return None
        return got[0]

    def revoke(self, ident: str) -> bool:
        return self._served.pop(str(ident), None) is not None


# ------------------------------------------------------- choosing a sink
def direction_word(deg: float) -> str:
    """0 -> right, 90 -> up, 180 -> left, 270 -> down, in HIS frame.

    A quadrant, not an angle: two or three samples at 7.5 fps cannot
    honestly carry more. The sign flip that puts image coordinates into his
    frame lives in gesture.to_his_frame and nowhere else."""
    try:
        d = float(deg)
    except (TypeError, ValueError):
        return ""
    if d != d:
        return ""
    return ("right", "up", "left", "down")[int(((d + 45.0) % 360.0) // 90.0)]


def sink_map(get_option: Optional[Callable] = None) -> dict:
    """The configured direction -> sink-name map, lower-cased, or {}."""
    if not callable(get_option):
        return {}
    try:
        raw = get_option(OPTION_SINKS, {})
    except Exception:  # noqa: BLE001 - config is a nicety here
        return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k.strip().lower()] = v.strip().lower()
    return out


def is_taught(get_option: Optional[Callable] = None) -> bool:
    """Has he told Jarvis which side anything is on yet?"""
    return any(d in ROUTABLE_DIRECTIONS for d in sink_map(get_option))


def pick_sink(direction: str, *, get_option: Optional[Callable] = None,
              registry: dict):
    """The sink for a spoken direction. Unmapped, unknown, unroutable or
    misconfigured all fall back to the board, which is the cheap outcome."""
    board = registry[DEFAULT_SINK]
    word = str(direction or "").strip().lower()
    if word not in ROUTABLE_DIRECTIONS:
        return board
    name = sink_map(get_option).get(word, "")
    if not name:
        return board
    sink = registry.get(name)
    if sink is None:
        log.info("cast: %s is mapped to unknown sink %r; using the board", word, name)
        return board
    return sink


def resolve_sink(deg: float, *, get_option: Optional[Callable] = None,
                 registry: dict):
    return pick_sink(direction_word(deg), get_option=get_option, registry=registry)


# ------------------------------------------------------ teaching a side
_SINK_ALIASES = {
    "hpcomputer": "hpcomputer", "hp computer": "hpcomputer", "the hp": "hpcomputer",
    "hp": "hpcomputer", "the pc": "hpcomputer", "my pc": "hpcomputer",
    # "the desktop" / "my desktop" are deliberately absent: a folder here, not the host.
    "the windows machine": "hpcomputer", "the windows box": "hpcomputer",
    "the other computer": "hpcomputer", "the other machine": "hpcomputer",
    "board": "board", "the board": "board", "spark": "board", "the spark": "board",
    "here": "board", "this screen": "board", "my screen": "board",
    "the console": "board", "handoff": "handoff", "the handoff": "handoff",
    "the page": "handoff", "the handoff page": "handoff",
}
_KNOWN_SINKS = frozenset(_SINK_ALIASES.values())
_SIDE_RX = r"(?P<side>left|right)"
_TEACH_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    # "HPCOMPUTER is on my right" / "the pc is to the left" / "hp is right of me"
    r"^(?:the\s+)?(?P<sink>.+?)\s+(?:is|sits|'s)\s+(?:on|to|at)?\s*(?:my|the|your)?\s*"
    + _SIDE_RX + r"(?:\s+(?:of|side))?(?:\s+(?:me|you|us|the camera|the lens|the screen|my chair))?\s*[.!]?$",
    # "right is HPCOMPUTER" / "the left is the board"
    r"^(?:the\s+|my\s+)?" + _SIDE_RX + r"(?:\s+side)?\s+(?:is|goes to|means)\s+(?P<sink>.+?)\s*[.!]?$",
    # "throw right to HPCOMPUTER" / "throwing left goes to the board"
    r"^(?:a\s+)?throw(?:ing)?\s+(?:to\s+(?:the\s+|my\s+)?)?" + _SIDE_RX
    + r"\s+(?:goes\s+to|is|means|to|at|for)\s+(?P<sink>.+?)\s*[.!]?$",
))


def sink_alias(name: str) -> str:
    """'the pc' -> 'hpcomputer'; unknown names come back lower-cased."""
    text = " ".join(str(name or "").lower().replace("-", " ").split())
    text = re.sub(r"[.!?,]+$", "", text).strip()
    # The teaching patterns may have eaten a leading "the"; try both.
    return _SINK_ALIASES.get(text) or _SINK_ALIASES.get("the " + text, text)


def parse_side_teaching(text: str) -> Optional[tuple[str, str]]:
    """'HPCOMPUTER is on my right' -> ('right', 'hpcomputer'), or None.

    Only a KNOWN target parses. "Right is fine", "what is left" and "the
    light is on the left" all have the shape of a teaching and are not one;
    answering them with "I don't know a target called fine" would be a
    false fire on the voice path. ``teach_sink`` still refuses an unknown
    name in its own words when it is called directly.
    """
    body = " ".join(str(text or "").split())
    if not body:
        return None
    for rx in _TEACH_PATTERNS:
        m = rx.match(body)
        if not m:
            continue
        side = m.group("side").lower()
        sink = sink_alias(m.group("sink"))
        if sink in _KNOWN_SINKS and side in ROUTABLE_DIRECTIONS:
            return side, sink
    return None


def teach_sink(direction: str, sink_name: str, *, set_option: Callable,
               get_option: Optional[Callable] = None,
               registry: Optional[dict] = None) -> str:
    """Write one side of the map and say what changed.

    One machine, one side: any other direction that pointed at the same
    sink is dropped, so "HPCOMPUTER is on my right" after "on my left"
    moves it rather than doubling it. ``set_option(key, value)`` is
    AssistantConfig.set; a False return is reported, not swallowed.
    """
    side = str(direction or "").strip().lower()
    name = sink_alias(sink_name)
    if side not in ROUTABLE_DIRECTIONS or not name:
        return TAUGHT_FAILED_LINE
    if registry is not None and name not in registry:
        return TAUGHT_UNKNOWN_LINE.format(name=sink_name)
    current = sink_map(get_option)
    new = {d: s for d, s in current.items() if s != name}
    new[side] = name
    try:
        ok = set_option(OPTION_SINKS, new)
    except Exception:  # noqa: BLE001 - the config's boundary
        log.exception("cast: saving %s failed", OPTION_SINKS)
        return TAUGHT_FAILED_LINE
    if ok is False:
        return TAUGHT_FAILED_LINE
    return TAUGHT_LINE.format(Direction=_cap(side), target=sink_label(name))


# --------------------------------------------------------------- casting
def _identity_opinion(identity) -> str:
    """'' no opinion | 'owner' | 'other'."""
    name = str(identity or "").strip().lower()
    if not name:
        return ""
    return "owner" if name == OWNER_IDENTITY else "other"


def _deliver(sink, subject: CastSubject) -> CastResult:
    """deliver() with the boundary guarded: a sink that raises is HELD."""
    try:
        res = sink.deliver(subject)
    except Exception as exc:  # noqa: BLE001 - the sink's boundary
        log.warning("cast: %s raised: %s", getattr(sink, "name", "?"), exc)
        return held_result(SINK_RAISED_LINE.format(
            Target=_cap(getattr(sink, "label", getattr(sink, "name", "it")))),
            sink=str(getattr(sink, "name", "")), detail=str(exc))
    if not isinstance(res, CastResult):
        return held_result(SINK_RAISED_LINE.format(
            Target=_cap(getattr(sink, "label", "it"))),
            sink=str(getattr(sink, "name", "")), detail="bad-result")
    return res


def _available(sink) -> tuple[bool, str]:
    try:
        ok, why = sink.available()
        return bool(ok), str(why or "")
    except Exception as exc:  # noqa: BLE001 - the sink's boundary
        return False, str(exc)


def earcon_for(status: str) -> str:
    """The tone for a cast() status, or '' for the outcomes that speak
    instead (a refusal, a proposal, an empty hand)."""
    return EARCONS.get(str(status or ""), "")


def cast(sink, subject: CastSubject, *, speak: Callable[[str], None],
         propose: Optional[Callable[[Callable[[], CastResult], str], None]] = None,
         capture: Optional[Callable[[], Optional[dict]]] = None,
         identity: Optional[str] = None, fallback=None) -> str:
    """Send ``subject`` to ``sink`` and say what happened.

    Returns one word: empty | vetoed | held | landed | proposed | refused.

    * A reversible sink ACTS at once and its line is spoken.
    * A sink that needs a read-back for this subject is checked for a route
      first (an unreachable target HOLDS with its own live reason -- there
      is nothing to propose to a deaf machine), then requires a positive
      identity if the sink demands one, then is PROPOSED through
      ``propose(run, line)``: nothing happens until the spoken yes, when
      ``run()`` delivers and returns the CastResult (it does not speak; the
      confirm path speaks the reply it returns -- wrap it as
      CommandResult(handled=True, reply=res.spoken, speak=True) for
      commander.stash_destructive). With no ``propose`` wired the answer is
      no, out loud, never "do it anyway".
    * ``capture`` gives a screen subject its bytes only when the sink wants
      them; ``identity`` is the eye's name for whoever is in frame ('' is
      no opinion, another name is a silent veto); ``fallback`` is the board
      sink, handed a HELD payload so the name lands somewhere he can see.

    Blocks for the capture and the probe; call it off the capture thread.
    """
    if subject is None or not subject.holdable:
        return "empty"
    if _identity_opinion(identity) == "other":
        log.info("cast: vetoed, %r is not %s", identity, OWNER_IDENTITY)
        return "vetoed"
    try:
        needs = bool(sink.needs_readback(subject))
    except Exception:  # noqa: BLE001 - a sink of the wrong shape
        needs = not bool(getattr(sink, "reversible", False))
    if needs:
        ok, _why = _available(sink)
        if not ok:
            res = _deliver(sink, subject)
            return _finish(res, subject, speak, fallback)
        if getattr(sink, "needs_identity", False) and \
                _identity_opinion(identity) != "owner":
            speak(IDENTITY_LINE)
            return "refused"
        if not callable(propose):
            speak(NO_PROPOSE_LINE)
            return "refused"
        if getattr(sink, "wants_bytes", False):
            subject = materialise(subject, capture)
        target = getattr(sink, "label", getattr(sink, "name", "there"))
        line = READBACK_LINE.format(What=_cap(subject.spoken), target=target)

        def run() -> CastResult:
            return _deliver(sink, subject)

        propose(run, line)
        return "proposed"
    if getattr(sink, "wants_bytes", False):
        subject = materialise(subject, capture)
    res = _deliver(sink, subject)
    return _finish(res, subject, speak, fallback)


def _finish(res: CastResult, subject: CastSubject, speak, fallback) -> str:
    if res.held and fallback is not None and res.fallback and \
            res.fallback == getattr(fallback, "name", "") and \
            getattr(fallback, "name", "") != res.sink:
        try:
            fallback.deliver(subject)                 # silent; the line is the sink's
        except Exception:  # noqa: BLE001 - the fallback is best-effort
            log.debug("cast: fallback delivery failed", exc_info=True)
    if res.spoken:
        speak(res.spoken)
    if res.landed:
        return "landed"
    return "held"


# ------------------------------------------------------- the held register
@dataclass
class HeldCast:
    ident: str
    subject: CastSubject
    sink: str
    reason: str
    at: float
    offered: bool = False


@dataclass(frozen=True)
class OutboxEvent:
    """Something he can be told about a held cast: ready | expired | evicted."""
    kind: str
    held: HeldCast
    spoken: str


class CastOutbox:
    """What a held cast does while it waits.

    It is kept, it is visible (``pending()``), it expires after ``ttl_s``
    and says so once, and when the target comes back it OFFERS -- the same
    ruling he made about the morning briefing: offer, do not deliver. A
    cast is only ever spent by ``take()`` after his spoken yes. Nothing
    vanishes unannounced: an eviction over ``capacity`` is an event too.
    """

    def __init__(self, *, now: Optional[Callable[[], float]] = None,
                 ttl_s: float = HELD_TTL_S, capacity: int = HELD_CAPACITY):
        self._now = now if callable(now) else time.monotonic
        self._ttl_s = float(ttl_s)
        self._capacity = max(1, int(capacity))
        self._held: list[HeldCast] = []
        self._events: list[OutboxEvent] = []
        self._seq = 0

    def hold(self, subject: CastSubject, sink: str, reason: str) -> Optional[HeldCast]:
        if subject is None or not subject.holdable:
            return None
        self._seq += 1
        held = HeldCast(ident=f"c{self._seq}", subject=subject,
                        sink=str(sink or ""), reason=str(reason or ""),
                        at=self._now())
        self._held.append(held)
        while len(self._held) > self._capacity:
            old = self._held.pop(0)
            self._events.append(OutboxEvent(
                "evicted", old, EVICTED_LINE.format(what=old.subject.spoken)))
        return held

    def pending(self) -> tuple:
        return tuple(self._held)

    def get(self, ident: str) -> Optional[HeldCast]:
        for h in self._held:
            if h.ident == ident:
                return h
        return None

    def drop(self, ident: str) -> bool:
        before = len(self._held)
        self._held = [h for h in self._held if h.ident != ident]
        return len(self._held) != before

    def take(self, ident: str) -> Optional[HeldCast]:
        """Remove and return one: the spoken yes lands here."""
        held = self.get(ident)
        if held is not None:
            self.drop(ident)
        return held

    def clear(self) -> None:
        self._held = []
        self._events = []

    def poll(self, available: Optional[Callable[[str], tuple]] = None) -> list:
        """Expire, then ask ``available(sink_name)`` -> (ok, why) about each
        held cast; a target that has come back yields ONE 'ready' offer
        until it goes away again. A check that raises keeps the cast."""
        now = self._now()
        events, self._events = list(self._events), []
        keep = []
        for h in self._held:
            if now - h.at > self._ttl_s:
                events.append(OutboxEvent("expired", h, EXPIRED_LINE.format(
                    what=h.subject.spoken, target=sink_label(h.sink))))
            else:
                keep.append(h)
        self._held = keep
        if callable(available):
            for h in self._held:
                try:
                    ok, _why = available(h.sink)
                except Exception:  # noqa: BLE001 - the probe's boundary
                    log.debug("cast: availability check failed", exc_info=True)
                    continue
                if ok and not h.offered:
                    h.offered = True
                    events.append(OutboxEvent("ready", h, READY_LINE.format(
                        Target=_cap(sink_label(h.sink)), what=h.subject.spoken)))
                elif not ok:
                    h.offered = False
        return events

    def status(self) -> dict:
        return {"pending": len(self._held), "capacity": self._capacity,
                "ttl_s": self._ttl_s, "queued_events": len(self._events)}


__all__ = [
    "BoardSink", "CastOutbox", "CastResult", "CastSubject", "HandoffSink",
    "HeldCast", "HpcomputerSink", "OutboxEvent", "Sink", "board_card", "cast",
    "direction_word", "document_spoken_name", "earcon_for", "empty_subject",
    "focused_screen_subject", "focused_window_title", "is_taught",
    "materialise", "parse_side_teaching", "pick_sink", "probe_tcp",
    "resolve_sink", "resolve_subject", "screen_subject", "sink_alias",
    "sink_label", "sink_map", "subject_line", "teach_sink",
    "track_spoken_name", "window_spoken_name",
]
