"""Spotify tool — search & play, Liked Songs, transport, queue and radio
through spotipy (spec: docs/specs/2026-08-26-jarvis-personal-assistant.md,
sections 4.1 / 6 / 10 and the SPOTIFY work item).

Output is never the Spark's own speakers: playback is steered to a Spotify
Connect device — the one named in the utterance ("on my phone"), else
``spotify.default_device`` (HPCOMPUTER), else whatever is currently active,
else the spoken "nothing's listening" excuse.  Every reply is a fixed
film-JARVIS line returned as ``ToolResult.speak`` so the model turn is
skipped and the confirmation says exactly what actually started.

Auth: Authorization Code flow via ``spotipy.SpotifyOAuth`` with the loopback
redirect ``http://127.0.0.1:8888/callback``; the token lives next to
``assistant.json`` as ``spotify_token.json`` (0600, auto-refresh).  One-time
link: ``python -m jarvis.tools.spotify --login`` (or ``python -m jarvis.app
--spotify-login`` once the wiring item adds the flag).

Liked Songs: two strategies, both exercised by the tests against a fake
client.  ``uris`` (default, documented API): page /me/tracks, shuffle
client-side, start the first 100 as ``uris`` and queue the next few lazily
on a worker thread.  ``collection``: ``spotify:user:<id>:collection`` as
the context with server-side shuffle — undocumented and reported to fail
(403/404) for many third-party apps, so it is the fallback; flip
``spotify.liked_strategy`` to "collection" if the live test shows it works.

Radio: the /recommendations and related-artists endpoints are deprecated for
apps created after 27 Nov 2024, so "something like this" = a "This Is
<artist>" / "<artist> Radio" playlist when search still returns one, else
the artist's top tracks — and the spoken line says which.

Test seams: ``SpotifyTool(client=FakeSpotify(...), auth_factory=...,
token_path=tmp, spawn=sync, rng=random.Random(0), settle=lambda s: None)``.
No network in unit tests; the live check runs only with ``JARVIS_LIVE=1``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from jarvis.logs import get_logger
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.spotify")

try:                                   # pure python, pip-installed into ~/vss_env
    import spotipy
    from spotipy.cache_handler import CacheHandler as _CacheHandler
    from spotipy.exceptions import SpotifyException, SpotifyOauthError
except ImportError:                    # the app must still boot without it
    spotipy = None
    _CacheHandler = object

    class SpotifyException(Exception):          # duck-typed stand-in
        def __init__(self, http_status=None, code=-1, msg="", reason=None,
                     headers=None):
            super().__init__(msg)
            self.http_status, self.code = http_status, code
            self.msg, self.reason, self.headers = msg, reason, headers or {}

    class SpotifyOauthError(Exception):
        pass

try:
    from jarvis.assistant_config import _is_placeholder
except ImportError:                    # W1 not landed: same rule, locally
    _PLACEHOLDER = re.compile(
        r"^\s*$|^<.*>$|^(paste|your|change[ -_]?me|replace|todo|example)[-_ :]|"
        r"^x{3,}$", re.I)

    def _is_placeholder(value) -> bool:
        if value is None:
            return True
        if not isinstance(value, str):
            return not value
        return bool(_PLACEHOLDER.search(value.strip()))


# ------------------------------------------------------------- constants
SECTION = "spotify"
REDIRECT_URI = "http://127.0.0.1:8888/callback"
SCOPES = ("user-read-playback-state user-modify-playback-state "
          "user-read-currently-playing user-library-read user-library-modify "
          "playlist-read-private playlist-read-collaborative user-top-read")
DEFAULT_DEVICE = "HPCOMPUTER"
TOKEN_FILE = "spotify_token.json"
TOKEN_ENV = "JARVIS_SPOTIFY_TOKEN"
DOCS_HINT = "docs/assistant-setup.md"

PAGE = 50                  # /me/tracks and /me/playlists page size
LIKED_CAP = 500            # most Liked Songs paged per request
LIKED_CHUNK = 100          # uris per start_playback (Spotify's practical limit)
# 0, deliberately.  This used to be 20: after "play my liked songs" a worker
# pushed 20 more tracks into Spotify's USER QUEUE, so a later "queue Save Your
# Tears next" landed 21st and never played next -- "Queueing does not work"
# (#70).  The 100-uri chunk above is the session; the queue belongs to him.
LIKED_QUEUE_AHEAD = 0      # add_to_queue calls made lazily after the chunk
OWN_PLAYLIST_CAP = 200
PLAYLIST_CACHE_S = 300.0
SETTLE_AFTER_TRANSFER_S = 0.4
# Remote duck (#72): percentage OF the Connect device's current volume, not an
# absolute level.  A Connect device's 0-100 is its own hardware slider, so
# "set it to 30" on a laptop already sitting at 40 is not a duck at all.
DUCK_TO_PCT_OF_CURRENT = 30
# The Spark's own librespot advertises under this name (LIBRESPOT_NAME in
# scripts/jarvis-spotify-device; ``spotify.local_device`` overrides it).  When
# IT is the active device the music is a PipeWire stream on this box and the
# Room Mixer already ducks it, so the remote duck stands aside rather than
# lowering the same music twice.
LIBRESPOT_NAME = "Spark"
# The music-state cache behind ``music_playing()`` (the wake gate's relaxed
# threshold reads it).  A "playing" older than PLAYING_TTL_S is not evidence:
# an album ends, a phone leaves the house, and a gate relaxed on stale news
# is a gate relaxed for a stranger.  The poller keeps it fresh -- every
# POLL_PLAYING_S while playing, POLL_IDLE_S once paused.
#
# ONLY /me/player may write "playing" into that cache.  The obvious cheaper
# signal -- "Spotify lists an active Connect device", which the remote duck
# already fetches on every hold -- is NOT the same question: Spotify leaves
# the last device active long after a pause (HPCOMPUTER was still listed
# active all evening on 2026-09-01), so reading it as music would arm the
# relaxed wake bar after every single turn, in a silent room, for as long as
# Spotify stays linked.  An active device is only a REASON TO ASK: it starts
# the poller (POLL_AFTER_DEVICE_S), and the poller's honest answer is what
# lands in the cache.  That also covers music he started on his phone
# without telling Jarvis -- one hold, one poll, and the gate knows.
PLAYING_TTL_S = 150.0
POLL_PLAYING_S = 30.0
POLL_IDLE_S = 60.0
POLL_AFTER_PLAY_S = 600.0
POLL_AFTER_DEVICE_S = 600.0
POLL_BACKOFF_MAX_S = 300.0
API_TIMEOUT_S = 8
MARKET = "from_token"      # relative to the user token; overridable with spotify.market
FALLBACK_COUNTRY = "US"    # artist_top_tracks needs a real country code

# ---------------------------------------------------------- persona lines
SETUP_LINE = f"I'll need Spotify set up, sir; the notes are in {DOCS_HINT}."
MISSING_LIB_LINE = ("The Spotify library isn't installed, sir; "
                    "pip install spotipy into the venv.")
NOT_LINKED_LINE = "Spotify isn't linked yet, sir; run me with --spotify-login once."
LAPSED_LINE = "Spotify's link has lapsed, sir; run me with --spotify-login once."
LINKED_LINE = "Spotify's linked, sir."
PREMIUM_LINE = "I'm afraid that needs Spotify Premium, sir."
NO_DEVICE_LINE = "Nothing's listening, sir — open Spotify on {default} or your phone."
UNKNOWN_DEVICE_LINE = "I can't see {name} on Spotify, sir; I can see {seen}."
UNKNOWN_DEVICE_NONE_LINE = "I can't see {name} on Spotify, sir; nothing's listening."
NO_RESULTS_LINE = "I couldn't find {query} on Spotify, sir."
NOTHING_PLAYING_LINE = "Nothing's playing, sir."
NO_SEED_LINE = "Nothing's playing to go on, sir; name an artist and I'll oblige."
LIKED_EMPTY_LINE = "Your Liked Songs are empty, sir."
RESTRICTED_LINE = "Spotify won't allow that just now, sir."
RATE_LIMIT_LINE = "Spotify's rate-limiting me, sir; give it a moment."
UNREACHABLE_LINE = "I can't reach Spotify, sir."
API_DOWN_LINE = "Spotify isn't answering, sir."
BAD_ACTION_LINE = "I don't know that Spotify control, sir."
NEED_VALUE_LINE = "I'll need a {what} for that, sir."
PLAY_LINE = "{what}, sir — on {device}."
ARTIST_LINE = "{artist}'s top tracks, sir — on {device}."
ALBUM_LINE = "{album} by {artist}, sir — the whole album on {device}."
LIKED_URIS_LINE = "Your Liked Songs on shuffle, sir — {n} of them, on {device}."
# The DEFAULT wording now: "the default should be last added to liked songs
# and then go down, only shuffle if i sp[ecify]" (#66).
LIKED_ORDER_LINE = "Your Liked Songs, sir — newest first, {n} of them, on {device}."
LIKED_COLLECTION_LINE = "Your Liked Songs on shuffle, sir — on {device}."
NOW_LINE = "{track} by {artist}, sir — on {device}."
NOW_PAUSED_LINE = "{track} by {artist}, sir, paused on {device}."
QUEUE_LINE = "{track} by {artist} is up next, sir."
RADIO_PLAYLIST_LINE = "{playlist}, sir — on {device}."
RADIO_TOP_LINE = "No radio for {artist}, sir, so it's their top tracks on {device}."
RADIO_NOTHING_LINE = "I've nothing to go on for {artist}, sir."
NO_ARTIST_TRACKS_LINE = "I can't find anything of {artist}'s to play, sir."
LIKE_LINE = "Saved {track} to your Liked Songs, sir."
LIKED_ALREADY_LINE = "{track} is already in your Liked Songs, sir."
LIKE_EPISODE_LINE = "That's a podcast, sir; I only save songs."
TRANSFER_LINE = "Moved to {device}, sir."
CONTROL_LINES = {"pause": "Paused, sir.", "resume": "Resumed, sir.",
                 "next": "Skipped, sir.", "previous": "Back one, sir."}
VOLUME_LINE = "Volume {n}, sir."
SHUFFLE_LINE = "Shuffle {state}, sir."
REPEAT_LINE = "Repeat {state}, sir."
SEEK_LINE = "Jumped to {pos}, sir."

# Fixed lines worth prewarming in the speech cache.
PERSONA_LINES = [SETUP_LINE, NOT_LINKED_LINE, LAPSED_LINE, LINKED_LINE,
                 PREMIUM_LINE, NO_DEVICE_LINE.format(default=DEFAULT_DEVICE),
                 NOTHING_PLAYING_LINE, NO_SEED_LINE, LIKED_EMPTY_LINE,
                 RESTRICTED_LINE, RATE_LIMIT_LINE, UNREACHABLE_LINE,
                 API_DOWN_LINE, BAD_ACTION_LINE, LIKE_EPISODE_LINE] + \
    list(CONTROL_LINES.values())

KINDS = ("auto", "track", "artist", "album", "playlist")
ACTIONS = ("pause", "resume", "next", "previous", "volume", "shuffle",
           "repeat", "seek", "like", "transfer")

_OWN_PLAYLIST_HINTS = ("discover weekly", "release radar", "daily mix",
                       "on repeat", "repeat rewind", "time capsule",
                       "your top songs", "liked songs")
_DEVICE_TYPES = {  # utterance word -> Spotify device type
    "phone": "smartphone", "iphone": "smartphone", "mobile": "smartphone",
    "android": "smartphone", "computer": "computer", "pc": "computer",
    "desktop": "computer", "laptop": "computer", "mac": "computer",
    "tv": "tv", "television": "tv", "speaker": "speaker", "echo": "speaker",
    "alexa": "speaker", "sonos": "speaker", "tablet": "tablet", "ipad": "tablet",
    "car": "automobile", "console": "game_console", "playstation": "game_console",
    "xbox": "game_console"}
_DEVICE_SUFFIX = re.compile(
    r"\s+(on|onto|to|over to)\s+(?:my\s+|the\s+)?([\w' -]{2,40})$", re.I)
# "play it on my phone" is a Connect handover, not a search for a song called
# "it".  Without this the tool searched Spotify for the pronoun and the model
# learned to answer "I have no way of reaching it" instead of calling at all
# (#69).  Deliberately narrow: "something"/"anything" are real play queries.
# The lead verb is optional because this is checked on BOTH the raw query and
# the one infer_kind has already cleaned.
_PRONOUN_RX = re.compile(
    r"^(?:(?:please\s+)?(?:play|put on|start|move|switch|send|transfer)\s+)?"
    r"(?:it|this|that|music|the music|the song|the track|"
    r"this song|this track|whatever\'?s playing)$", re.I)
# Order matters where these are used: "I don't want them shuffled" contains
# the word "shuffled", so the in-order phrasings must be tested FIRST.
_INORDER_RX = re.compile(
    r"\b(?:in order|in sequence|in the order|newest first|latest first|"
    r"most recent(?:ly)?(?: added)?|last added|recently added|chronological(?:ly)?|"
    r"unshuffled|not shuffled|no shuffle|without shuffl(?:e|ing)|"
    r"stop shuffling|"
    # "I don't want them shuffled" -- his exact words, and the reason the
    # in-order patterns are tested BEFORE the shuffle ones: this phrase
    # contains "shuffled" and would otherwise read as a request for it.
    r"(?:don'?t|do not|never|no longer)\s+(?:want\s+)?(?:\w+\s+){0,2}shuffl\w*)\b",
    re.I)
_SHUFFLE_RX = re.compile(
    r"\b(?:shuffle|shuffled|shuffling|random|randomly|randomised|randomized|"
    r"on shuffle|mixed up)\b", re.I)


# ------------------------------------------------------------ pure helpers
def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _loose(text: Any) -> str:
    """Lower-case, single-spaced, punctuation trimmed but digits, '.' and
    ':' kept (for "40%", "0.5", "1:30")."""
    v = re.sub(r"\s+", " ", str(text or "").lower()).strip(" .!?,")
    return re.sub(r"[^a-z0-9.: ]+", "", v).strip()


def _cfg_get(cfg, dotted: str, default=None):
    """Read ``a.b.c`` from an AssistantConfig, a plain dict or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        node: Any = cfg
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
    get = getattr(cfg, "get", None)
    if callable(get):
        try:
            return get(dotted, default)
        except Exception:                # noqa: BLE001 - odd config object
            log.debug("config get %s failed", dotted, exc_info=True)
    return default


def token_path(cfg=None, override=None) -> Path:
    """Where the OAuth token lives: explicit arg > $JARVIS_SPOTIFY_TOKEN >
    next to the assistant config (so the test firewall's tmp config keeps
    the token in tmp too) > ~/.config/jarvis/spotify_token.json."""
    if override:
        return Path(os.path.expanduser(str(override)))
    env = os.environ.get(TOKEN_ENV)
    if env:
        return Path(os.path.expanduser(env))
    cfg_path = getattr(cfg, "path", None)
    if cfg_path:
        return Path(cfg_path).parent / TOKEN_FILE
    try:
        from jarvis.config import PATHS
        base = getattr(PATHS, "ASSISTANT_CONFIG", None)
        if base:
            return Path(base).parent / TOKEN_FILE
    except Exception:                    # noqa: BLE001 - config import optional
        log.debug("PATHS unavailable for the token path", exc_info=True)
    env_cfg = os.environ.get("JARVIS_ASSISTANT_CONFIG")
    if env_cfg:
        return Path(os.path.expanduser(env_cfg)).parent / TOKEN_FILE
    return Path.home() / ".config" / "jarvis" / TOKEN_FILE


_token_path = token_path        # the SpotifyTool kwarg shadows the name


def coerce_kind(kind: Any) -> str:
    k = _norm(kind)
    if k in ("track", "song", "songs", "tune"):
        return "track"
    if k in ("artist", "band", "artists", "musician"):
        return "artist"
    if k in ("album", "record", "lp", "ep"):
        return "album"
    if k in ("playlist", "mix", "playlists"):
        return "playlist"
    if k in ("liked", "likes", "library", "saved", "favourites", "favorites"):
        return "liked"
    return "auto"


def infer_kind(query: Any) -> tuple[str, str]:
    """(kind, cleaned query) from the utterance when the model gave no
    kind.  Pure; table-tested."""
    q = re.sub(r"\s+", " ", str(query or "")).strip()
    q = re.sub(r"^(?:please\s+)?(?:play|put on|start|queue up|stick on)\s+", "", q, flags=re.I)
    q = re.sub(r"\s+on spotify$", "", q, flags=re.I).strip(" .!?,")
    low = q.lower()
    if not low:
        return "auto", ""
    # "liked?" because Whisper drops the d: his 21:14:37 utterance came out as
    # "Play my like songs playlist in order", which missed this test, fell into
    # the "my <x> playlist" branch below and started a stranger's public
    # playlist called "All my liked songs" instead of his library (#66).
    if re.search(r"\b(?:my\s+)?(?:liked?\s+songs|likes|library|saved\s+songs|"
                 r"favou?rites?)\b", low):
        return "liked", q
    for hint in _OWN_PLAYLIST_HINTS:
        if hint in low:
            return "playlist", q
    m = re.match(r"^(?:my|our)\s+(.+?)(?:\s+playlist)?$", low)
    if m:
        return "playlist", m.group(1)
    m = re.match(r"^(?:the\s+)?(.+?)\s+playlist$", low) or \
        re.match(r"^(?:the\s+)?playlist\s+(?:called\s+)?(.+)$", low)
    if m:
        return "playlist", m.group(1)
    m = re.match(r"^(?:the\s+)?album\s+(.+)$", low) or \
        re.match(r"^(?:the\s+)?(.+?)\s+album$", low)
    if m:
        return "album", m.group(1)
    m = re.match(r"^(?:some\s+|a bit of\s+|any\s+)?(?:songs?|music|tracks?|stuff|"
                 r"something|anything|hits)\s+(?:by|from)\s+(.+)$", low)
    if m:
        return "artist", m.group(1)
    m = re.match(r"^(?:the\s+)?(?:artist|band)\s+(.+)$", low) or \
        re.match(r"^some\s+(.+)$", low)
    if m:
        return "artist", m.group(1)
    if re.search(r"\s+by\s+", low):
        return "track", q
    return "auto", q


def wants_shuffle(text: Any, default: Optional[bool] = None) -> Optional[bool]:
    """Did the utterance ASK for shuffle?  True / False / ``default``.

    Liked Songs used to shuffle unconditionally; he wants "the default should
    be last added to liked songs and then go down, only shuffle if i
    sp[ecify]" (#66).  Pure; table-tested."""
    t = str(text or "")
    if _INORDER_RX.search(t):
        return False
    if _SHUFFLE_RX.search(t):
        return True
    return default


def split_device(query: Any) -> tuple[str, Optional[str]]:
    """Peel a trailing 'on my phone' / 'on HPCOMPUTER' off a query when the
    model left it in.  Only known device words or a device-looking token."""
    q = str(query or "").strip()
    m = _DEVICE_SUFFIX.search(q)
    if not m:
        return q, None
    prep, cand = m.group(1).lower(), m.group(2).strip()
    words = _norm(cand).split()
    if not words:
        return q, None
    if prep != "on":
        # "move it TO my phone" (#69).  Stricter than "on": only the last word
        # may name the device, so "Fly Me To The Moon" and "add it to the car
        # playlist" are still song titles, not handovers.
        if words[-1] in _DEVICE_TYPES or cand.isupper():
            return q[:m.start()].strip(), cand
        return q, None
    if words[-1] in _DEVICE_TYPES or any(w in _DEVICE_TYPES for w in words) \
            or cand.isupper():
        return q[:m.start()].strip(), cand
    return q, None


def clean_device_name(value: Any) -> Optional[str]:
    v = re.sub(r"\s+", " ", str(value or "")).strip(" .!?,")
    v = re.sub(r"^(?:on\s+)?(?:my|the)\s+", "", v, flags=re.I)
    return v or None


def parse_volume(value: Any, current: Optional[int] = None) -> Optional[int]:
    """"40" / "40%" / 40 / "up" / "down" / "max" / "mute" -> 0..100."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0, min(100, int(round(value))))
    v = _loose(value)
    if not v:
        return None
    m = re.search(r"\d+(?:\.\d+)?", v)
    if m and not re.search(r"\b(up|down|louder|quieter|softer)\b", v):
        n = float(m.group())
        if 0 < n <= 1 and "." in m.group():
            n *= 100
        return max(0, min(100, int(round(n))))
    if v in ("max", "full", "maximum", "loudest", "all the way up"):
        return 100
    if v in ("mute", "silent", "off", "zero"):
        return 0
    if v in ("half", "halfway", "medium"):
        return 50
    step = int(float(m.group())) if m else 10
    if re.search(r"\b(up|louder|more|higher)\b", v):
        return max(0, min(100, (current if current is not None else 50) + step))
    if re.search(r"\b(down|quieter|softer|less|lower)\b", v):
        return max(0, min(100, (current if current is not None else 50) - step))
    return None


def parse_seek(value: Any) -> Optional[int]:
    """"1:30" / "90" / "90 seconds" / "2 minutes" / "start" -> position ms."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0, int(value * 1000))
    v = _loose(value)
    if not v:
        return None
    if v in ("start", "beginning", "top", "restart", "0"):
        return 0
    m = re.match(r"^(\d+):(\d{1,2})$", v)
    if m:
        return (int(m.group(1)) * 60 + int(m.group(2))) * 1000
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(m|min|mins|minutes?)$", v)
    if m:
        return int(float(m.group(1)) * 60_000)
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds?)?$", v)
    if m:
        return int(float(m.group(1)) * 1000)
    return None


def parse_onoff(value: Any, default: Optional[bool] = True) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    v = _norm(value)
    if not v:
        return default
    if v in ("on", "true", "yes", "1", "enable", "enabled", "start"):
        return True
    if v in ("off", "false", "no", "0", "disable", "disabled", "stop", "none"):
        return False
    return None


def parse_repeat(value: Any) -> Optional[str]:
    v = _norm(value)
    if not v or v in ("on", "all", "context", "yes", "true", "playlist", "album"):
        return "context"
    if v in ("track", "one", "this", "this track", "this song", "song", "single"):
        return "track"
    if v in ("off", "no", "false", "none", "stop"):
        return "off"
    return None


def fmt_ms(ms: int) -> str:
    s = max(0, int(ms)) // 1000
    return f"{s // 60}:{s % 60:02d}"


def _artist_names(item: Optional[dict], limit: int = 2) -> str:
    names = [a.get("name", "") for a in (item or {}).get("artists", []) or []
             if isinstance(a, dict) and a.get("name")]
    if not names:
        return "an unknown artist"
    if len(names) > limit:
        names = names[:limit]
    return " and ".join(names)


def _items(result: Any, key: Optional[str] = None) -> list[dict]:
    """The item list of a search/page response with the None entries
    Spotify now returns for restricted playlists dropped."""
    node = result or {}
    if key and isinstance(node, dict):
        node = node.get(key) or {}
    items = node.get("items") if isinstance(node, dict) else node
    return [i for i in (items or []) if isinstance(i, dict)]


# --------------------------------------------------------------- errors
class SpotifyError(Exception):
    """A failure with its spoken persona line.  ``kind`` steers fallbacks:
    setup | auth | premium | device | network | restricted | api."""

    def __init__(self, line: str, kind: str = "api", text: Optional[str] = None,
                 status: Optional[int] = None):
        super().__init__(line)
        self.line, self.kind, self.text = line, kind, text or line
        self.status = status

    @property
    def fatal(self) -> bool:
        """Errors that no other strategy can route around."""
        return self.kind in ("setup", "auth", "premium", "device", "network")


# ----------------------------------------------------------- token cache
class TokenCache(_CacheHandler):
    """spotipy CacheHandler that keeps the token 0600 in a 0700 directory
    and writes atomically."""

    def __init__(self, path):
        self.path = Path(path)

    def get_cached_token(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            log.warning("spotify token %s unreadable (%s)", self.path, exc)
            return None

    def save_token_to_cache(self, token_info):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.path.parent, 0o700)
            except OSError:
                pass
            fd, tmp = tempfile.mkstemp(prefix=".spotify-", suffix=".tmp",
                                       dir=str(self.path.parent))
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(token_info, fh)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
            os.chmod(self.path, 0o600)
        except OSError:
            log.exception("could not write the spotify token to %s", self.path)

    def linked(self) -> bool:
        tok = self.get_cached_token()
        return bool(isinstance(tok, dict) and tok.get("refresh_token"))

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


@dataclass
class Device:
    id: str
    name: str
    type: str = ""
    is_active: bool = False
    volume: Optional[int] = None

    @classmethod
    def from_api(cls, d: dict) -> "Device":
        return cls(id=str(d.get("id") or ""), name=str(d.get("name") or "?"),
                   type=str(d.get("type") or "").lower(),
                   is_active=bool(d.get("is_active")),
                   volume=d.get("volume_percent"))


@dataclass
class Started:
    """What actually began playing, for the confirmation."""
    kind: str
    name: str
    artist: str = ""
    uri: str = ""
    how: str = ""


# ------------------------------------------------------------- the tool
class SpotifyTool:
    def __init__(self, cfg=None, client=None, token_path: Optional[os.PathLike | str] = None,
                 auth_factory: Optional[Callable] = None, spawn: Optional[Callable] = None,
                 rng: Optional[random.Random] = None, now: Callable[[], float] = time.time,
                 settle: Optional[Callable[[float], None]] = None,
                 poll: bool = True):
        self.cfg = cfg
        self._client = client
        self._injected = client is not None
        self._auth = None
        self._auth_factory = auth_factory
        self._spawn = spawn or self._thread
        self._rng = rng or random.Random()
        self._now = now
        self._settle = settle or time.sleep
        self.cache = TokenCache(_token_path(cfg, token_path))
        self._lock = threading.RLock()
        self._playlists: list[dict] = []
        self._playlists_at = 0.0
        self._user_id: Optional[str] = None
        self._market_bad = False
        # (device_id, volume_percent, name) of a remote duck in flight -- see
        # duck().
        self._ducked_from: Optional[tuple[str, int, str]] = None
        # The music-state cache (music_playing) and its poller.  Its own lock,
        # not self._lock: the wake gate reads it on the hotword thread and must
        # never queue behind a Web API call.
        self._music_lock = threading.Lock()
        self._music_playing = False
        self._music_at = -1e9          # when the cache was last written
        self._played_at = -1e9         # his last play/resume command
        self._device_at = -1e9         # last time a Connect device was active
        self._poll_enabled = bool(poll)
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()
        self._poll_backoff = 0.0

    # ---------------------------------------------------------- config
    @property
    def default_device(self) -> str:
        return str(_cfg_get(self.cfg, "spotify.default_device") or DEFAULT_DEVICE)

    @property
    def local_device(self) -> str:
        """The Connect name of THIS box's librespot (see LIBRESPOT_NAME)."""
        return str(_cfg_get(self.cfg, "spotify.local_device") or LIBRESPOT_NAME)

    @property
    def liked_shuffle(self) -> bool:
        """Whether Liked Songs shuffle when he does not say.  FALSE by
        default: /me/tracks pages newest-added first, which is exactly the
        order he asked for (#66).  ``spotify.liked_shuffle: true`` restores
        the old behaviour."""
        return bool(_cfg_get(self.cfg, "spotify.liked_shuffle", False))

    @property
    def liked_strategy(self) -> str:
        s = _norm(_cfg_get(self.cfg, "spotify.liked_strategy", "uris"))
        return "collection" if s == "collection" else "uris"

    @property
    def market(self) -> str:
        """Market for search / saved tracks.  "from_token" is the legacy
        value spotipy documents; some apps now get a 400 for it, so
        ``_api_market`` drops it and remembers."""
        return str(_cfg_get(self.cfg, "spotify.market", MARKET) or MARKET).strip()

    @property
    def country(self) -> str:
        """artist_top_tracks wants a country code, never "from_token"."""
        m = self.market
        return m.upper() if re.fullmatch(r"[A-Za-z]{2}", m) else FALLBACK_COUNTRY

    def _opt(self, key: str, default: int) -> int:
        try:
            return int(_cfg_get(self.cfg, f"spotify.{key}", default) or default)
        except (TypeError, ValueError):
            return default

    def configured(self) -> bool:
        return not _is_placeholder(_cfg_get(self.cfg, "spotify.client_id")) and \
            not _is_placeholder(_cfg_get(self.cfg, "spotify.client_secret"))

    def setup_line(self) -> str:
        fn = getattr(self.cfg, "setup_line", None)
        if callable(fn):
            try:
                line = fn(SECTION)
                if line:
                    return str(line)
            except Exception:            # noqa: BLE001 - odd config object
                log.debug("setup_line failed", exc_info=True)
        return SETUP_LINE

    def linked(self) -> bool:
        return self._client is not None or self.cache.linked()

    def ensure_ready(self) -> None:
        if spotipy is None and self._client is None:
            raise SpotifyError(MISSING_LIB_LINE, "setup")
        if not self.configured():
            raise SpotifyError(self.setup_line(), "setup")
        if not self.linked():
            raise SpotifyError(NOT_LINKED_LINE, "auth")

    # ------------------------------------------------------------ auth
    def _make_auth(self, open_browser: bool = False):
        cid = _cfg_get(self.cfg, "spotify.client_id")
        secret = _cfg_get(self.cfg, "spotify.client_secret")
        if self._auth_factory is not None:
            return self._auth_factory(cid, secret, self.cache, open_browser)
        if spotipy is None:
            raise SpotifyError(MISSING_LIB_LINE, "setup")
        return spotipy.SpotifyOAuth(
            client_id=cid, client_secret=secret, redirect_uri=REDIRECT_URI,
            scope=SCOPES, cache_handler=self.cache, open_browser=open_browser,
            show_dialog=False, requests_timeout=API_TIMEOUT_S)

    def _get_auth(self):
        if self._auth is None:
            self._auth = self._make_auth()
        return self._auth

    def client(self):
        with self._lock:
            if self._client is None:
                self.ensure_ready()
                self._client = spotipy.Spotify(
                    auth_manager=self._get_auth(), requests_timeout=API_TIMEOUT_S,
                    retries=1)
            return self._client

    def _refresh(self) -> bool:
        """Expired access token: refresh through the auth manager and save
        (0600).  False when there is nothing to refresh with."""
        tok = self.cache.get_cached_token()
        refresh = (tok or {}).get("refresh_token") if isinstance(tok, dict) else None
        if not refresh:
            return False
        try:
            new = self._get_auth().refresh_access_token(refresh)
        except Exception as exc:         # noqa: BLE001 - oauth boundary
            log.warning("spotify token refresh failed: %s", type(exc).__name__)
            return False
        if isinstance(new, dict) and new.get("access_token"):
            if "refresh_token" not in new:
                new["refresh_token"] = refresh
            self.cache.save_token_to_cache(new)
            log.info("spotify token refreshed")
            return True
        return False

    def login(self, open_browser: bool = True) -> str:
        """Interactive Authorization Code flow (loopback server on :8888,
        browser opened by spotipy).  Returns the spoken confirmation."""
        if spotipy is None and self._auth_factory is None:
            raise SpotifyError(MISSING_LIB_LINE, "setup")
        if not self.configured():
            raise SpotifyError(self.setup_line(), "setup")
        auth = self._make_auth(open_browser=open_browser)
        self._auth = auth
        try:
            url = auth.get_authorize_url()
            print(f"If the browser does not open, visit:\n  {url}", file=sys.stderr)
            code = auth.get_auth_response(open_browser=open_browser)
            token = auth.get_access_token(code, as_dict=True, check_cache=False)
        except SpotifyError:
            raise
        except Exception as exc:         # noqa: BLE001 - oauth boundary
            log.warning("spotify login failed: %s", type(exc).__name__)
            raise SpotifyError(f"The Spotify link didn't go through, sir; "
                               f"{type(exc).__name__}.", "auth") from exc
        if not isinstance(token, dict) or not token.get("access_token"):
            raise SpotifyError("The Spotify link didn't go through, sir.", "auth")
        self.cache.save_token_to_cache(token)
        if not self._injected:
            self._client = None          # rebuild on the new token
        try:
            me = self._api("me") or {}
            log.info("spotify linked as %s", me.get("display_name") or me.get("id"))
        except SpotifyError as exc:
            log.warning("spotify linked but /me failed: %s", exc.text)
        return LINKED_LINE

    # ------------------------------------------------------------- api
    def _api(self, method: str, *args, **kwargs):
        """One Web API call with the errors translated; a 401 refreshes the
        token once and retries."""
        fn = getattr(self.client(), method)
        try:
            return fn(*args, **kwargs)
        except SpotifyError:
            raise
        except Exception as exc:         # noqa: BLE001 - translated below
            if getattr(exc, "http_status", None) == 401 and self._refresh():
                try:
                    return fn(*args, **kwargs)
                except Exception as exc2:    # noqa: BLE001
                    raise self._translate(exc2) from exc2
            raise self._translate(exc) from exc

    def _translate(self, exc: Exception) -> SpotifyError:
        status = getattr(exc, "http_status", None)
        reason = str(getattr(exc, "reason", "") or "")
        msg = str(getattr(exc, "msg", "") or exc)
        low = (reason + " " + msg).lower()
        name = type(exc).__name__
        if isinstance(exc, SpotifyOauthError) or status == 401:
            return SpotifyError(LAPSED_LINE, "auth", f"spotify auth failed: {msg[:80]}")
        if status == 403 and ("premium" in low):
            return SpotifyError(PREMIUM_LINE, "premium", "spotify: premium required")
        if status == 404 and ("device" in low or "no active" in low):
            return SpotifyError(NO_DEVICE_LINE.format(default=self.default_device),
                                "device", "spotify: no active device")
        if status == 403 and "restriction" in low:
            return SpotifyError(RESTRICTED_LINE, "restricted",
                                "spotify: restriction violated")
        if status == 403 and "scope" in low:
            # Classified HERE, on the full message: spotipy's text starts with
            # the request URL, so "Insufficient client scope" sits past the
            # 80-char cut below (index 157 of 162 on a real search) and a
            # substring check on the stored text never sees it.
            return SpotifyError(API_DOWN_LINE, "scope",
                                "spotify: insufficient client scope", status=403)
        if status == 429:
            return SpotifyError(RATE_LIMIT_LINE, "api", "spotify: rate limited")
        if status is None and name in ("ConnectionError", "Timeout", "ReadTimeout",
                                       "ConnectTimeout", "SSLError", "OSError",
                                       "RetryError", "socket.gaierror"):
            return SpotifyError(UNREACHABLE_LINE, "network", f"spotify unreachable: {name}")
        return SpotifyError(API_DOWN_LINE, "api",
                            f"spotify error {status or name}: {msg[:80]}",
                            status=status)

    def _api_market(self, method: str, *args, **kwargs):
        """``_api`` for the calls that carry a market/country: a 400 means
        Spotify rejected the market, so retry once without it and stop
        sending it for the rest of this session.

        A 403 naming a scope means the same thing for a different reason:
        ``market=from_token`` resolves the country from the user, so Spotify
        requires the ``user-read-private`` scope for it. Without that scope
        every search 403s and the user hears "Spotify isn't answering, sir"
        -- a network-shaped line for a permissions problem. The scope cannot
        simply be added to SCOPES: spotipy rejects a cached token whose scopes
        do not cover the request, so widening them silently unlinks a working
        account until the user re-authorises. Dropping the market is the fix
        that costs the user nothing."""
        if self._market_bad:
            kwargs = {k: v for k, v in kwargs.items() if k not in ("market", "country")}
            return self._api(method, *args, **kwargs)
        try:
            return self._api(method, *args, **kwargs)
        except SpotifyError as exc:
            stripped = {k: v for k, v in kwargs.items()
                        if k not in ("market", "country")}
            scope_denied = exc.kind == "scope"
            if (exc.status != 400 and not scope_denied) or stripped == kwargs:
                raise
            log.warning("spotify: %s rejected market=%r; retrying without it",
                        method, kwargs.get("market") or kwargs.get("country"))
            self._market_bad = True
            return self._api(method, *args, **stripped)

    # ---------------------------------------------------------- devices
    def devices(self) -> list[Device]:
        raw = self._api("devices") or {}
        return [Device.from_api(d) for d in (raw.get("devices") or [])
                if isinstance(d, dict) and d.get("id")]

    @staticmethod
    def _match_device(devs: list[Device], name: str) -> Optional[Device]:
        want = _norm(name)
        if not want:
            return None
        for d in devs:                                   # exact
            if _norm(d.name) == want:
                return d
        for d in devs:                                   # substring either way
            n = _norm(d.name)
            if want in n or (n and n in want):
                return d
        words = want.split()
        types = {_DEVICE_TYPES[w] for w in words if w in _DEVICE_TYPES}
        for d in devs:                                   # by device type
            if d.type in types:
                return d
        return None

    def resolve_device(self, named: Optional[str] = None,
                       prefer_active: bool = False) -> Device:
        """named > default_device > active (play); named > active > default
        (transport, so 'pause' never drags playback off the phone)."""
        devs = self.devices()
        named = clean_device_name(named)
        if named:
            d = self._match_device(devs, named)
            if d is not None:
                return d
            seen = ", ".join(x.name for x in devs[:3])
            line = UNKNOWN_DEVICE_LINE.format(name=named, seen=seen) if seen \
                else UNKNOWN_DEVICE_NONE_LINE.format(name=named)
            raise SpotifyError(line, "device", f"spotify: no device {named!r}")
        active = next((d for d in devs if d.is_active), None)
        default = self._match_device(devs, self.default_device)
        order = (active, default) if prefer_active else (default, active)
        for d in order:
            if d is not None:
                return d
        raise SpotifyError(NO_DEVICE_LINE.format(default=self.default_device),
                           "device", "spotify: no device available")

    def _ensure_on(self, device: Device) -> Device:
        """Transfer playback to the device when it is not the active one."""
        if not device.is_active:
            self._api("transfer_playback", device.id, force_play=False)
            self._settle(SETTLE_AFTER_TRANSFER_S)
            device.is_active = True
        return device

    # ------------------------------------------------------- remote duck
    def duck(self, to_pct: int = DUCK_TO_PCT_OF_CURRENT) -> bool:
        """Lower the ACTIVE Connect device's volume through the Web API.

        The Room Mixer can only reach PipeWire streams on this box.  When the
        music is on HPCOMPUTER or his phone there is no local sink-input at
        all, so `pactl` moves nothing and the duck is theatre -- which is why
        "he cant discern my voice from the vocalists in the music" (#72) with
        "mixer: ducked 1 stream(s) to 30%" sitting in the log next to it (the
        one stream it found was the idle librespot pipe).  Spotify's own
        volume endpoint is the only handle on a remote device there is.

        Stands aside when the active device is this box's own librespot
        (``local_device``): that music is a PipeWire stream the Room Mixer
        ducks itself, and lowering it here too would put it at 30 % of 30 %.

        Returns True when a volume was actually moved.  Never raises: a
        failed duck must cost him nothing but the duck."""
        with self._lock:
            if self._ducked_from is not None:
                return False
        try:
            self.ensure_ready()
            dev = next((d for d in self.devices() if d.is_active), None)
            # The device lookup is a REASON TO ASK, never an answer.  An
            # active Connect device is not playing music: Spotify keeps the
            # last one active long after a pause, and this duck runs on every
            # hold, so calling it music would leave the wake gate's relaxed
            # bar armed after every turn with nothing coming out of any
            # speaker.  Arm the poller instead and let /me/player say.
            if dev is not None:
                self._note_device_active()
            if dev is None or dev.id is None or not isinstance(dev.volume, int):
                return False
            if _norm(dev.name) == _norm(self.local_device):
                log.debug("spotify: remote duck skipped: %s is this box's "
                          "librespot, the Room Mixer has its stream", dev.name)
                return False
            floor = max(0, min(100, int(to_pct)))
            want = int(round(dev.volume * floor / 100.0))
            if want >= dev.volume:
                return False
            self._api("volume", want, device_id=dev.id)
        except SpotifyError as exc:
            log.debug("spotify: remote duck skipped (%s)", exc.text)
            return False
        with self._lock:
            self._ducked_from = (dev.id, dev.volume, dev.name)
        return True

    def unduck(self) -> bool:
        """Put the remote device back where he had it.  Idempotent.

        His original volume is kept until the write CONFIRMS.  It used to be
        popped first, so one transient 5xx on the way back discarded the only
        record of where he had it and left the device at 30 % of his volume
        for good; that was unreachable while the remote duck never fired, and
        this branch fires it on every hold.  The Room Mixer retries on its
        next pump and once more at stop(), and what is still held down is in
        its state file for the next start to heal."""
        with self._lock:
            saved = self._ducked_from
        if saved is None:
            return False
        try:
            self._api("volume", int(saved[1]), device_id=saved[0])
        except SpotifyError as exc:
            log.debug("spotify: remote unduck failed (%s); keeping %s at %d%% "
                      "for a retry", exc.text, saved[2], saved[1])
            return False
        with self._lock:
            if self._ducked_from == saved:
                self._ducked_from = None
        return True

    @property
    def ducked_device(self) -> Optional[str]:
        """Name of the Connect device a duck is holding down, else None."""
        with self._lock:
            return self._ducked_from[2] if self._ducked_from else None

    @property
    def ducked_state(self) -> Optional[dict]:
        """What a duck is holding down, as the Room Mixer persists it: the
        device, HIS volume, and the name for the log.  The remote has no
        stream-restore database and no pactl, so that file is the only thing
        that survives a crash between the duck and the restore."""
        with self._lock:
            saved = self._ducked_from
        if saved is None:
            return None
        return {"device": saved[0], "volume_pct": int(saved[1]), "name": saved[2]}

    def restore_volume(self, device_id: str, volume_pct: int,
                       at_or_below: Optional[int] = None) -> Optional[bool]:
        """Put a device back to a volume a previous run left it away from.

        The heal path, called by the Room Mixer from its state file.  Returns
        True when the volume was written, False when the device is there but
        must be left alone, and None when it is not listed at all -- the
        caller keeps the record and asks again, exactly as heal() keeps a
        sink-input entry until its stream reappears.

        ``at_or_below`` is the local heal's rule in remote form: only a device
        still sitting at (or under) the duck floor can still be ours.  Anything
        louder is a volume he has since chosen himself, and moving it would be
        the mixer overruling him."""
        try:
            self.ensure_ready()
            dev = next((d for d in self.devices() if d.id == device_id), None)
            if dev is None:
                return None
            if at_or_below is not None and isinstance(dev.volume, int) and \
                    dev.volume > at_or_below:
                log.debug("spotify: %s is at %d%%, above the %d%% floor -- his "
                          "own volume, left alone", dev.name, dev.volume,
                          at_or_below)
                return False
            self._api("volume", max(0, min(100, int(volume_pct))),
                      device_id=device_id)
        except SpotifyError as exc:
            log.debug("spotify: remote heal failed (%s)", exc.text)
            return None
        return True

    # ------------------------------------------------------- music state
    def music_playing(self) -> bool:
        """Is music KNOWN to be playing?  A cache read -- no request, no
        lock shared with the API -- so the wake gate may call it on every
        candidate.  False on anything stale or unknown: this only ever
        relaxes a convenience gate, so it must fail strict."""
        with self._music_lock:
            return self._music_playing and \
                (self._now() - self._music_at) <= PLAYING_TTL_S

    def _note_music(self, playing: bool) -> None:
        """Record what a request just learned about the speakers."""
        with self._music_lock:
            self._music_playing = bool(playing)
            self._music_at = self._now()
        if playing:
            self._ensure_poll()

    def _note_playback(self, pb) -> None:
        """Note a /me/player answer: playing means an item AND is_playing."""
        pb = pb if isinstance(pb, dict) else {}
        self._note_music(bool(pb.get("item")) and bool(pb.get("is_playing")))

    def _note_play_command(self) -> None:
        """He asked for music: it is playing, and worth polling for a while
        even if the first refresh says otherwise (Connect takes a moment)."""
        with self._music_lock:
            self._played_at = self._now()
        self._note_music(True)

    def _note_device_active(self) -> None:
        """A Connect device is active.  NOT music -- Spotify keeps the last
        device active long after a pause -- so this writes no state the wake
        gate reads.  It only buys the poller a reason to ask /me/player,
        which is the one source allowed to say "playing"."""
        with self._music_lock:
            self._device_at = self._now()
        self._ensure_poll()

    def _poll_wanted(self) -> bool:
        """Under _music_lock.  The poller runs only while it has a reason:
        the cache says playing, he pressed play within POLL_AFTER_PLAY_S, or
        a Connect device was active within POLL_AFTER_DEVICE_S.  A Spotify
        with nothing linked and nothing active still costs no requests."""
        now = self._now()
        if self._music_playing and now - self._music_at <= PLAYING_TTL_S:
            return True
        if now - self._played_at <= POLL_AFTER_PLAY_S:
            return True
        return now - self._device_at <= POLL_AFTER_DEVICE_S

    def _poll_once(self) -> Optional[float]:
        """One refresh of the cache: the seconds until the next, or None
        when polling should stop.  Errors back off (doubling to
        POLL_BACKOFF_MAX_S) rather than stopping outright -- a Wi-Fi blip
        must not leave the gate strict for the rest of the album -- but the
        TTL means a poller that keeps failing ages the cache out anyway."""
        with self._music_lock:
            if not self._poll_wanted():
                return None
        try:
            state = self.playback_state()          # notes the cache itself
        except Exception as exc:  # noqa: BLE001 - never let the poller die
            text = getattr(exc, "text", None) or type(exc).__name__
            self._poll_backoff = min(max(POLL_PLAYING_S, self._poll_backoff * 2),
                                     POLL_BACKOFF_MAX_S)
            log.debug("spotify: music poll failed (%s); next in %.0f s",
                      text, self._poll_backoff)
            return self._poll_backoff
        self._poll_backoff = 0.0
        return POLL_PLAYING_S if state == "playing" else POLL_IDLE_S

    def _poll_loop(self) -> None:
        # Refresh FIRST, then wait.  A leading sleep made the poller useless
        # for the thing it exists to do: the duck's device probe starts it,
        # and a 30 s wait before the first read left that guess standing for
        # longer than most conversations.  It is also how music he started on
        # his phone becomes known within one interval instead of two.
        while not self._poll_stop.is_set():
            nxt = self._poll_once()
            if nxt is None:
                # Hand the slot back under the lock, so a play command that
                # lands right now sees no live thread and starts a new one
                # instead of relying on this one, which is leaving.
                with self._music_lock:
                    if not self._poll_wanted():
                        self._poll_thread = None
                        return
                nxt = POLL_IDLE_S
            if self._poll_stop.wait(nxt):
                return

    def _ensure_poll(self) -> None:
        if not self._poll_enabled:
            return
        with self._music_lock:
            t = self._poll_thread
            if t is not None and t.is_alive():
                return
            t = threading.Thread(target=self._poll_loop, name="spotify-poll",
                                 daemon=True)
            self._poll_thread = t
        t.start()

    def close(self) -> None:
        """Stop the poller for good (quit, and the tests' teardown); the
        cache itself stays readable.  Idempotent.

        Named close, and this class must never gain a ``stop()``:
        app.stop_assistant resolves ``stop`` before ``close`` on everything
        in its list, so a transport stop() here would pause his music every
        time Jarvis quit.  Transport goes through control("pause")."""
        self._poll_enabled = False
        self._poll_stop.set()
        with self._music_lock:
            t = self._poll_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)

    # ----------------------------------------------------------- search
    def _user_id_(self) -> str:
        if not self._user_id:
            me = self._api("me") or {}
            self._user_id = str(me.get("id") or "")
        return self._user_id

    def _own_playlists(self) -> list[dict]:
        with self._lock:
            if self._playlists and self._now() - self._playlists_at < PLAYLIST_CACHE_S:
                return self._playlists
        out: list[dict] = []
        offset = 0
        while offset < OWN_PLAYLIST_CAP:
            page = self._api("current_user_playlists", limit=PAGE, offset=offset) or {}
            items = _items(page)
            out.extend(items)
            if not page.get("next") or len(items) < PAGE:
                break
            offset += PAGE
        with self._lock:
            self._playlists, self._playlists_at = out, self._now()
        return out

    def _find_own_playlist(self, name: str) -> Optional[dict]:
        want = _norm(name)
        if not want:
            return None
        lists = self._own_playlists()
        for p in lists:
            if _norm(p.get("name")) == want:
                return p
        for p in lists:
            if _norm(p.get("name")).startswith(want):
                return p
        for p in lists:
            if want in _norm(p.get("name")):
                return p
        return None

    def _search(self, q: str, kind: str, limit: int = 5) -> dict:
        return self._api_market("search", q=q, limit=limit, type=kind,
                                market=self.market) or {}

    def _find(self, kind: str, q: str) -> Optional[dict]:
        """One item of one kind; own playlists first for playlists."""
        if kind == "playlist":
            own = self._find_own_playlist(q)
            if own is not None:
                return own
        res = self._search(q, kind)
        items = _items(res, kind + "s")
        if not items:
            return None
        want = _norm(q)
        for it in items:
            if _norm(it.get("name")) == want:
                return it
        if kind == "playlist":
            for it in items:
                if str((it.get("owner") or {}).get("id", "")).lower() == "spotify":
                    return it
        return items[0]

    def _find_auto(self, q: str) -> tuple[str, Optional[dict]]:
        own = self._find_own_playlist(q)
        if own is not None and _norm(own.get("name")) == _norm(q):
            return "playlist", own
        res = self._search(q, "track,artist,album,playlist")
        want = _norm(q)
        for kind in ("artist", "album", "playlist"):
            for it in _items(res, kind + "s"):
                if _norm(it.get("name")) == want:
                    return kind, it
        tracks = _items(res, "tracks")
        if tracks:
            return "track", tracks[0]
        if own is not None:
            return "playlist", own
        for kind in ("artist", "album", "playlist"):
            items = _items(res, kind + "s")
            if items:
                return kind, items[0]
        return "auto", None

    # ------------------------------------------------------------ play
    def _start(self, device: Device, **what) -> None:
        self._ensure_on(device)
        self._api("start_playback", device_id=device.id, **what)
        self._note_play_command()

    def _play_item(self, kind: str, item: dict, device: Device) -> Started:
        name = str(item.get("name") or "?")
        if kind == "track":
            self._start(device, uris=[item["uri"]])
            return Started("track", name, _artist_names(item), item.get("uri", ""))
        if kind == "artist":
            top = _items(self._api_market("artist_top_tracks", item["id"],
                                          country=self.country), "tracks")
            uris = [t["uri"] for t in top if t.get("uri")]
            if not uris:
                this_is = self._find("playlist", f"This Is {name}")
                if this_is is None:
                    raise SpotifyError(NO_ARTIST_TRACKS_LINE.format(artist=name), "api",
                                       f"spotify: no top tracks for {name}")
                self._start(device, context_uri=this_is["uri"])
                return Started("playlist", str(this_is.get("name")), "", this_is["uri"])
            self._start(device, uris=uris[:LIKED_CHUNK])
            return Started("artist", name, name, item.get("uri", ""))
        self._start(device, context_uri=item["uri"])
        return Started(kind, name, _artist_names(item) if kind == "album" else "",
                       item.get("uri", ""))

    def _confirm(self, s: Started, device: Device) -> str:
        if s.kind == "track":
            return PLAY_LINE.format(what=f"{s.name} by {s.artist}", device=device.name)
        if s.kind == "artist":
            return ARTIST_LINE.format(artist=s.name, device=device.name)
        if s.kind == "album":
            return ALBUM_LINE.format(album=s.name, artist=s.artist, device=device.name)
        return PLAY_LINE.format(what=s.name, device=device.name)

    def play(self, query: Any = "", kind: Any = "auto", device: Any = None) -> ToolResult:
        query = str(query or "").strip()
        spoken = query                 # kept for the shuffle hint below (#66)
        device = clean_device_name(device)
        if not device:
            query, device = split_device(query)
        k = coerce_kind(kind)
        if k == "auto":
            k, query = infer_kind(query)
        if k == "liked":
            return self.liked(device, shuffle=wants_shuffle(spoken))
        if device and _PRONOUN_RX.match(query.strip(" .!?,")):
            # "play it on my phone" -> a Connect handover, not a search for a
            # song called "it".  Spotify Connect supports the handover; the
            # search is what made it look like it did not (#69).
            query = ""
        if not query:
            if device:
                # He named a device and nothing to play: move the music there.
                return self.control("transfer", value=device)
            return self.control("resume", device=device)
        dev = self.resolve_device(device)
        if k == "auto":
            k, item = self._find_auto(query)
        else:
            item = self._find(k, query)
        if item is None:
            raise SpotifyError(NO_RESULTS_LINE.format(query=query), "api",
                               f"spotify: no results for {query!r}")
        started = self._play_item(k, item, dev)
        line = self._confirm(started, dev)
        return ToolResult(text=line, speak=line)

    # ------------------------------------------------------ liked songs
    def _liked_uris(self, cap: Optional[int] = None) -> list[str]:
        cap = cap or self._opt("liked_cap", LIKED_CAP)
        out: list[str] = []
        offset = 0
        while len(out) < cap:
            page = self._api_market("current_user_saved_tracks", limit=PAGE,
                                    offset=offset, market=self.market) or {}
            items = _items(page)
            for it in items:
                tr = it.get("track") or {}
                if tr.get("uri") and not tr.get("is_local"):
                    out.append(tr["uri"])
            if not page.get("next") or len(items) < PAGE:
                break
            offset += PAGE
        return out[:cap]

    def _set_shuffle(self, device: Device, state: bool) -> None:
        """Best effort: a device that refuses the toggle must not cost him
        the music, so this warns and carries on."""
        try:
            self._api("shuffle", bool(state), device_id=device.id)
        except SpotifyError as exc:
            log.warning("liked songs: could not set shuffle=%s (%s)", state, exc.text)

    def _liked_by_uris(self, device: Device, uris: list[str],
                       shuffled: bool = True) -> int:
        chunk = max(1, min(LIKED_CHUNK, self._opt("liked_chunk", LIKED_CHUNK)))
        first, rest = uris[:chunk], uris[chunk:]
        self._ensure_on(device)
        if not shuffled:
            # Spotify's shuffle flag is STICKY per device and outranks the
            # order of `uris`: an ordered list started while the flag is on
            # comes back out random, which is exactly what he heard (#66).
            self._set_shuffle(device, False)
        self._api("start_playback", device_id=device.id, uris=first)
        self._note_play_command()
        ahead = self._opt("liked_queue_ahead", LIKED_QUEUE_AHEAD)
        if rest and ahead > 0:
            self._spawn(lambda: self._queue_rest(device, rest[:ahead]))
        return len(uris)

    def _queue_rest(self, device: Device, uris: list[str]) -> int:
        n = 0
        for uri in uris:
            try:
                self._api("add_to_queue", uri, device_id=device.id)
                n += 1
            except SpotifyError as exc:
                log.warning("liked songs: queueing stopped after %d (%s)", n, exc.text)
                break
        return n

    def _liked_by_collection(self, device: Device) -> None:
        uid = self._user_id_()
        if not uid:
            raise SpotifyError(API_DOWN_LINE, "api", "spotify: no user id for the collection")
        self._ensure_on(device)
        self._api("shuffle", True, device_id=device.id)
        self._api("start_playback", device_id=device.id,
                  context_uri=f"spotify:user:{uid}:collection")
        self._note_play_command()

    def liked(self, device: Any = None, shuffle: Any = None) -> ToolResult:
        """Liked Songs.  NEWEST ADDED FIRST unless he asks for shuffle.

        ``shuffle=None`` means "he did not say", which falls to
        ``spotify.liked_shuffle`` (False).  /me/tracks already pages
        most-recently-added first, so the untouched order IS "last added to
        liked songs and then go down" (#66)."""
        dev = self.resolve_device(clean_device_name(device))
        # An empty string is the model declining to fill the field, not a
        # request: it must read as "he did not say", never as shuffle=on.
        said = shuffle is not None and not (isinstance(shuffle, str)
                                            and not shuffle.strip())
        shuffled = bool(parse_onoff(shuffle, default=True)) if said \
            else self.liked_shuffle
        uris = self._liked_uris()
        if not uris:
            raise SpotifyError(LIKED_EMPTY_LINE, "api", "spotify: no saved tracks")
        if shuffled:
            self._rng.shuffle(uris)
        order = ("uris", "collection") if self.liked_strategy == "uris" \
            else ("collection", "uris")
        if not shuffled:
            # spotify:user:<id>:collection is a server-SHUFFLED context by
            # construction, so it can never honour an order: it is not a
            # fallback when he asked for newest-first.
            order = ("uris",)
        last: Optional[SpotifyError] = None
        for how in order:
            try:
                if how == "collection":
                    self._liked_by_collection(dev)
                    line = LIKED_COLLECTION_LINE.format(device=dev.name)
                else:
                    n = self._liked_by_uris(dev, uris, shuffled)
                    line = (LIKED_URIS_LINE if shuffled else
                            LIKED_ORDER_LINE).format(n=n, device=dev.name)
                if last is not None:
                    log.info("liked songs: %s failed (%s); %s worked", order[0],
                             last.text, how)
                return ToolResult(text=line, speak=line)
            except SpotifyError as exc:
                if exc.fatal:
                    raise
                log.warning("liked songs: %s strategy failed: %s", how, exc.text)
                last = exc
        raise last or SpotifyError(API_DOWN_LINE)

    # ------------------------------------------------------- transport
    def now_playing(self) -> ToolResult:
        pb = self._api("current_playback", additional_types="track,episode") or {}
        self._note_playback(pb)
        item = pb.get("item") or {}
        if not item:
            return ToolResult(text=NOTHING_PLAYING_LINE, speak=NOTHING_PLAYING_LINE)
        track = str(item.get("name") or "?")
        artist = _artist_names(item) if item.get("artists") else \
            str((item.get("show") or {}).get("name") or "an unknown artist")
        device = str((pb.get("device") or {}).get("name") or self.default_device)
        line = (NOW_LINE if pb.get("is_playing") else NOW_PAUSED_LINE).format(
            track=track, artist=artist, device=device)
        pos = fmt_ms(pb.get("progress_ms") or 0)
        dur = fmt_ms(item.get("duration_ms") or 0)
        album = str((item.get("album") or {}).get("name") or "")
        text = f"{line} {pos} of {dur}" + (f", album {album}." if album else ".")
        return ToolResult(text=text, speak=line)

    def playback_state(self) -> str:
        """"playing" / "paused" / "idle" — the one machine-readable read of
        what the speakers are doing. ``now_playing`` answers a sentence, and
        a caller that must decide whether pausing is worth doing (the class
        stager, which has to know whether it owes a resume) cannot parse
        prose. Raises SpotifyError like every other call here."""
        pb = self._api("current_playback", additional_types="track,episode") or {}
        self._note_playback(pb)
        if not pb.get("item"):
            return "idle"
        return "playing" if pb.get("is_playing") else "paused"

    def control(self, action: Any = "", value: Any = None, device: Any = None) -> ToolResult:
        act = _norm(action).replace(" ", "_")
        aliases = {"play": "resume", "unpause": "resume", "continue": "resume",
                   "stop": "pause", "skip": "next", "forward": "next",
                   "back": "previous", "prev": "previous", "previous_track": "previous",
                   "next_track": "next", "save": "like", "heart": "like",
                   "move": "transfer", "switch": "transfer", "louder": "volume",
                   "quieter": "volume", "mute": "volume", "vol": "volume",
                   "loop": "repeat", "jump": "seek", "rewind": "seek"}
        raw, act = act, aliases.get(act, act)
        if raw in ("louder", "quieter", "mute") and value is None:
            value = raw                  # "louder" is both the action and the amount
        if act not in ACTIONS:
            raise SpotifyError(BAD_ACTION_LINE, "api", f"spotify: unknown action {action!r}")
        named = clean_device_name(device)
        if act == "like":
            return self._like()
        if act == "transfer":
            target = clean_device_name(value) or named
            if not target:
                raise SpotifyError(NEED_VALUE_LINE.format(what="device name"), "api")
            dev = self.resolve_device(target)
            if not dev.is_active:
                self._api("transfer_playback", dev.id, force_play=True)
                self._note_play_command()        # force_play: it IS playing now
            line = TRANSFER_LINE.format(device=dev.name)
            return ToolResult(text=line, speak=line)
        dev = self.resolve_device(named, prefer_active=True)
        if act == "resume":
            self._start(dev)
            line = CONTROL_LINES["resume"]
        elif act == "pause":
            self._api("pause_playback", device_id=dev.id)
            self._note_music(False)
            line = CONTROL_LINES["pause"]
        elif act == "next":
            self._api("next_track", device_id=dev.id)
            line = CONTROL_LINES["next"]
        elif act == "previous":
            self._api("previous_track", device_id=dev.id)
            line = CONTROL_LINES["previous"]
        elif act == "volume":
            n = parse_volume(value, dev.volume)
            if n is None:
                raise SpotifyError(NEED_VALUE_LINE.format(what="volume level"), "api")
            self._api("volume", n, device_id=dev.id)
            line = VOLUME_LINE.format(n=n)
        elif act == "shuffle":
            state = parse_onoff(value, default=True)
            if state is None:
                raise SpotifyError(NEED_VALUE_LINE.format(what="shuffle on or off"), "api")
            self._api("shuffle", state, device_id=dev.id)
            line = SHUFFLE_LINE.format(state="on" if state else "off")
        elif act == "repeat":
            state = parse_repeat(value)
            if state is None:
                raise SpotifyError(NEED_VALUE_LINE.format(what="repeat mode"), "api")
            self._api("repeat", state, device_id=dev.id)
            word = {"context": "on", "track": "on this track", "off": "off"}[state]
            line = REPEAT_LINE.format(state=word)
        else:                                                   # seek
            ms = parse_seek(value)
            if ms is None:
                raise SpotifyError(NEED_VALUE_LINE.format(what="position"), "api")
            self._api("seek_track", ms, device_id=dev.id)
            line = SEEK_LINE.format(pos=fmt_ms(ms))
        return ToolResult(text=line, speak=line)

    def _like(self) -> ToolResult:
        pb = self._api("current_playback") or {}
        self._note_playback(pb)
        item = pb.get("item") or {}
        tid = item.get("id")
        if not tid:
            return ToolResult(text=NOTHING_PLAYING_LINE, ok=False, speak=NOTHING_PLAYING_LINE)
        if _norm(item.get("type")) == "episode" or item.get("show"):
            return ToolResult(text=LIKE_EPISODE_LINE, ok=False, speak=LIKE_EPISODE_LINE)
        name = str(item.get("name") or "That one")
        have = self._api("current_user_saved_tracks_contains", [tid]) or [False]
        if have and have[0]:
            line = LIKED_ALREADY_LINE.format(track=name)
        else:
            self._api("current_user_saved_tracks_add", [tid])
            line = LIKE_LINE.format(track=name)
        return ToolResult(text=line, speak=line)

    # ------------------------------------------------------------ queue
    def queue(self, query: Any = "", device: Any = None) -> ToolResult:
        query = str(query or "").strip()
        named = clean_device_name(device)
        if not named:
            query, named = split_device(query)
        kind, query = infer_kind(query)
        if not query:
            raise SpotifyError(NEED_VALUE_LINE.format(what="song name"), "api")
        item = self._find("track", query)
        if item is None:
            raise SpotifyError(NO_RESULTS_LINE.format(query=query), "api",
                               f"spotify: no results for {query!r}")
        dev = self.resolve_device(named, prefer_active=True)
        self._api("add_to_queue", item["uri"], device_id=dev.id)
        line = QUEUE_LINE.format(track=item.get("name", "?"), artist=_artist_names(item))
        return ToolResult(text=line, speak=line)

    # ------------------------------------------------------------ radio
    def _seed_artist(self, seed: str) -> tuple[Optional[dict], Optional[str]]:
        """(artist item, current track uri) — from the seed text or from
        whatever is playing."""
        if seed:
            art = self._find("artist", seed)
            if art is None:
                tr = self._find("track", seed)
                arts = (tr or {}).get("artists") or []
                art = arts[0] if arts else None
            return art, None
        pb = self._api("current_playback") or {}
        self._note_playback(pb)
        item = pb.get("item") or {}
        arts = item.get("artists") or []
        return (arts[0] if arts else None), item.get("uri")

    def radio(self, seed: Any = "", device: Any = None) -> ToolResult:
        seed = str(seed or "").strip()
        named = clean_device_name(device)
        if not named:
            seed, named = split_device(seed)
        art, current_uri = self._seed_artist(seed)
        if not art or not art.get("name"):
            raise SpotifyError(NO_SEED_LINE, "api", "spotify: no seed for radio")
        name = str(art["name"])
        dev = self.resolve_device(named)
        want = {_norm(f"this is {name}"), _norm(f"{name} radio")}
        found: list[dict] = []
        for q in (f"This Is {name}", f"{name} Radio"):
            found.extend(_items(self._search(q, "playlist"), "playlists"))
        exact = [p for p in found if _norm(p.get("name")) in want]
        exact.sort(key=lambda p: str((p.get("owner") or {}).get("id", "")).lower() != "spotify")
        if exact:
            pl = exact[0]
            self._start(dev, context_uri=pl["uri"])
            line = RADIO_PLAYLIST_LINE.format(playlist=pl.get("name"), device=dev.name)
            return ToolResult(text=line, speak=line)
        art_id = art.get("id")
        top = _items(self._api_market("artist_top_tracks", art_id,
                                      country=self.country), "tracks") if art_id else []
        uris = [t["uri"] for t in top if t.get("uri") and t.get("uri") != current_uri]
        if not uris:
            raise SpotifyError(RADIO_NOTHING_LINE.format(artist=name), "api",
                               f"spotify: nothing for radio on {name}")
        self._rng.shuffle(uris)
        self._start(dev, uris=uris)
        line = RADIO_TOP_LINE.format(artist=name, device=dev.name)
        return ToolResult(text=line, speak=line)

    # -------------------------------------------------------- registry
    def _guard(self, fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        def handler(**kwargs) -> ToolResult:
            try:
                self.ensure_ready()
                return fn(**kwargs)
            except SpotifyError as exc:
                if exc.kind != "api":
                    log.warning("spotify: %s", exc.text)
                return ToolResult(text=exc.text, ok=False, speak=exc.line)
            except Exception as exc:     # noqa: BLE001 - tool boundary
                log.exception("spotify tool failed")
                return ToolResult(text=f"spotify failed: {str(exc)[:80]}", ok=False,
                                  speak=API_DOWN_LINE)
        return handler

    def tools(self) -> list[ToolSpec]:
        def play(query="", kind="auto", device="", **_):
            return self.play(query, kind, device)

        def liked(device="", shuffle=None, **_):
            return self.liked(device, shuffle)

        def control(action="", value=None, device="", **_):
            return self.control(action, value, device)

        def now_playing(**_):
            return self.now_playing()

        def queue(query="", **_):
            return self.queue(query)

        def radio(seed="", device="", **_):
            return self.radio(seed, device)

        return [
            ToolSpec("spotify_play",
                     "Play a song, artist, album or playlist on Spotify; own playlists first.",
                     {"type": "object",
                      "properties": {"query": {"type": "string"},
                                     "kind": {"type": "string", "enum": list(KINDS)},
                                     "device": {"type": "string"}},
                      "required": ["query"]},
                     self._guard(play)),
            # The old description was "Shuffle Hunter's Liked Songs on Spotify."
            # -- the model read that as an instruction and every request came
            # back shuffled, which is half of #66.  Order is the default now
            # and shuffle is opt-in, in the wording as well as the code.
            ToolSpec("spotify_liked",
                     "Play Hunter's Liked Songs on Spotify, newest added first; "
                     "shuffle=true only if he asks.",
                     {"type": "object",
                      "properties": {"device": {"type": "string"},
                                     "shuffle": {"type": "boolean"}}},
                     self._guard(liked)),
            # "device" was missing from this schema, so the model had no way
            # to see that moving playback was on offer; asked to "play it on my
            # phone" it invented "I have no way of reaching it" instead of
            # calling transfer, which Spotify Connect supports fine (#69).
            ToolSpec("spotify_control",
                     "Spotify transport: pause, resume, next, previous, volume, shuffle, "
                     "repeat, seek, like, and transfer (move playback to another device).",
                     {"type": "object",
                      "properties": {"action": {"type": "string", "enum": list(ACTIONS)},
                                     "value": {"type": "string"},
                                     "device": {"type": "string"}},
                      "required": ["action"]},
                     self._guard(control)),
            ToolSpec("spotify_now_playing", "What is playing on Spotify right now.",
                     {"type": "object", "properties": {}},
                     self._guard(now_playing)),
            ToolSpec("spotify_queue", "Queue a song to play next on Spotify.",
                     {"type": "object", "properties": {"query": {"type": "string"}},
                      "required": ["query"]},
                     self._guard(queue)),
            ToolSpec("spotify_radio",
                     "Play more like the current song, or like a named artist, on Spotify.",
                     {"type": "object", "properties": {"seed": {"type": "string"}}},
                     self._guard(radio)),
        ]

    @staticmethod
    def _thread(fn: Callable[[], Any]) -> None:
        threading.Thread(target=fn, name="spotify-queue", daemon=True).start()


def make_tools(cfg, services) -> list[ToolSpec]:
    """Registry contract (spec 4.1)."""
    tool = SpotifyTool(cfg)
    try:
        if services is not None and getattr(services, "spotify", None) is None:
            services.spotify = tool
    except Exception:                    # noqa: BLE001 - services may be frozen
        log.debug("services has no room for the spotify tool", exc_info=True)
    return tool.tools()


# ------------------------------------------------------------------ CLI
def login(cfg=None, token_path=None, open_browser: bool = True,
          auth_factory=None) -> str:
    """One-time link.  Returns the spoken line; raises SpotifyError."""
    if cfg is None:
        from jarvis.assistant_config import AssistantConfig
        cfg = AssistantConfig.load()
    return SpotifyTool(cfg, token_path=token_path,
                       auth_factory=auth_factory).login(open_browser=open_browser)


def _announce(line: str) -> None:
    """Have the running app say the line (speak-queue IPC) when it is up."""
    try:
        from jarvis.config import PATHS
        if (PATHS.LOG_DIR / "jarvis.pid").exists():
            from jarvis import speak_queue
            speak_queue.say(line)
    except Exception:                    # noqa: BLE001 - best effort
        log.debug("could not announce through the speak queue", exc_info=True)


def login_cli(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="jarvis.tools.spotify",
                                 description="Link Jarvis to Spotify (one-time OAuth).")
    ap.add_argument("--login", action="store_true", help="run the browser login")
    ap.add_argument("--status", action="store_true", help="show link status")
    ap.add_argument("--no-browser", action="store_true",
                    help="print the URL instead of opening a browser")
    ns = ap.parse_args(argv)
    from jarvis.assistant_config import AssistantConfig
    cfg = AssistantConfig.load()
    tool = SpotifyTool(cfg)
    if ns.status or not ns.login:
        print(f"configured: {tool.configured()}  linked: {tool.linked()}  "
              f"token: {tool.cache.path}")
        return 0
    os.environ.setdefault("DISPLAY", ":1")
    try:
        line = tool.login(open_browser=not ns.no_browser)
    except SpotifyError as exc:
        print(exc.line, file=sys.stderr)
        return 1
    print(line)
    _announce(line)
    return 0


if __name__ == "__main__":             # python -m jarvis.tools.spotify --login
    sys.exit(login_cli())
