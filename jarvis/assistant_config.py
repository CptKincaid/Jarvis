"""Assistant config: ``~/.config/jarvis/assistant.json`` (spec section 10).

One file holds every personal-assistant setting and secret: home location,
Google secret-iCal URLs, the iCloud / Gmail app passwords, the Discord bot
token, the Claude session rules (allowed dirs, models, skill phrases) and
the briefing / alarm / autostart options. It is created from DEFAULTS with
placeholders and mode 0600 on first load, saved atomically (0600), and a
corrupt file is moved aside as ``assistant.json.bad`` and recreated.

Loading never raises: with an unwritable directory the config lives in
memory and every ``save()`` logs the failure.

Two ways to read a value::

    cfg.get("alarms.snooze_min", 10)     # dotted, default when missing
    cfg.alarms.snooze_min                # live attribute view (spec 4-9 use this)

Every consumer of a section with placeholders missing asks
``cfg.is_configured("gmail")`` and speaks ``cfg.setup_line("gmail")``.
Secrets never reach a log: ``redacted()`` / ``repr(cfg)`` mask them and
``scrub(text)`` strips their values out of arbitrary text.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from jarvis.logs import get_logger

log = get_logger("assistant_config")

ENV_VAR = "JARVIS_ASSISTANT_CONFIG"
DEFAULT_PATH = Path.home() / ".config" / "jarvis" / "assistant.json"
DOCS_HINT = "docs/assistant-setup.md"
MASK = "•••"          # "•••"

# Spec 10.1 — kept literally; the file is created from this.
DEFAULTS: dict = {
    "version": 1,
    "user": {"name": "Hunter"},
    "units": "us",
    "local_model": "gemma4:26b",
    "home_location": {"city": "", "region": "", "lat": None, "lon": None},
    "location_lookup": True,
    "google_ical_urls": [],
    "icloud": {"apple_id": "", "app_password": "",
               "url": "https://caldav.icloud.com"},
    "gmail": {"address": "", "app_password": "", "imap_host": "imap.gmail.com",
              "accounts": []},
    "claude": {
        "allowed_dirs": ["/home/hunterp/Jarvis", "/home/hunterp/haymaker-digest"],
        "projects_root": "/home/hunterp/projects",
        "permission_mode": "acceptEdits",
        "web_model": "haiku",         # one-shot web lookups; fast beats big here
        "dangerously_skip_permissions": False,
        # User decision 2026-08-26: work is auto-approved ANYWHERE, not only
        # under allowed_dirs ("even if its not in the base project directory
        # it can be auto approved").  Set false to restore the old behaviour
        # of refusing any project outside claude.allowed_dirs.
        "auto_approve_anywhere": True,
        # Verified live against Claude CLI 2.1.247 on 2026-08-27 (findings are
        # recorded at the top of jarvis/claude_session.py).  With this on, an
        # action outside the project asks Hunter aloud through the broker; set
        # it false to drop --mcp-config / --permission-prompt-tool, which also
        # makes work outside allowed_dirs refuse outright again.
        "permission_prompt_tool": True,
        "model": "opus", "big_model": "fable", "fast_mode": False, "effort": "",
        "skill_phrases": {
            "^review (this|my|the) code$": "/code-review",
            "^commit (this|it|that)$": "/commit",
            "^simplify (this|it|that)$": "/simplify",
            "^security review$": "/security-review",
            "^run a ralph loop on (.+)$": "/ralph-loop $1",
            "^plan a feature (.+)$": "/feature-dev $1",
        },
    },
    # Every desktop banner (the hub AND the direct notify-send sites) obeys
    # alerts.desktop; alerts.discord gates the Discord fan-out;
    # alerts.claude_hooks gates the spoken lines from the user's OWN Claude
    # Code sessions (scripts/claude_hooks/narrate.py reads this file with
    # plain json, so the key lives here rather than in a jarvis module).
    "alerts": {"desktop": True, "discord": True, "claude_hooks": True},
    # User-defined spoken shortcuts -> a tool call, matched ahead of the
    # classifier and the model. See docs/assistant-setup.md "Custom phrases".
    "phrases": [],
    # on_first_wake: deliver the briefing after the first thing you say to
    # Jarvis each day, once it is past `after` (24 h clock, local time).
    "calendar": {"heads_up_min": 10},
    # Canvas LMS: Account > Settings > New Access Token (read-only use).
    # heads_up_hours: the deadline heads-up (jarvis/deadlines.py) speaks
    # this long before each due time; the meeting heads-up's ten minutes
    # is no use for an 11:59 pm deadline.
    "canvas": {"base_url": "https://canvas.tamu.edu", "token": "", "heads_up_hours": 3},
    # Local document Q&A: drop PDFs / notes here; indexed with nomic-embed-text
    "docs": {"paths": ["~/Documents/Jarvis Docs"], "index_dir": "~/.aiws_trainer/docs_index",
             "max_files": 500, "embed_model": "nomic-embed-text",
             "ollama_url": "http://localhost:11434"},
    # Local code Q&A over his OWN repos (jarvis/tools/docs.py CodeIndex).
    # paths empty means "the folders in claude.allowed_dirs" -- those already
    # ARE his repos, and a second list would drift out of sync. index_dir
    # empty means PATHS.MEMORY_DIR/code_index, a chroma collection separate
    # from the documents one so quiz mode never draws a flashcard from app.py.
    "code": {"paths": [], "index_dir": "", "max_files": 3000},
    "screen": {"model": "llama3.2-vision:latest", "max_width": 1280},
    # Study / focus sessions (jarvis/focus.py): block and break lengths in
    # minutes, "Halfway, sir" for blocks of 10+ min, the session ends itself
    # after max_blocks (0 = until "end the session"). music: "pause" pauses
    # Spotify for the block and resumes it for the break, "playlist" plays
    # `playlist` for the block and pauses it for the break, "off" leaves it.
    # dnd: while a block is running, proactive lines (heads-ups, watchdog
    # warnings, the hooks narrator) are held by jarvis/quiet.py and read
    # back as the catch-up digest at the break -- not mid-pomodoro.
    "focus": {"block_min": 25, "break_min": 5, "halfway": True, "max_blocks": 4,
              "music": "pause", "playlist": "", "dnd": True},
    # Lecture notes (jarvis/lecture.py): after each noted line the mic
    # re-opens without a wake word for window_s seconds (capped at 30 by
    # the recorder; the 60 s hard cap on a capture must leave room to talk).
    "lecture": {"window_s": 20},
    # Quiz mode over the documents index: questions per round, chunks of
    # study text handed to the model per round (jarvis/tools/quiz.py).
    # window_s: the mic stays open this long for the ANSWER (and for a
    # yes/no read-back) instead of CONFIG.followup_window's 4 s, which is
    # sized for "...and Tuesday?" and not for thinking about a flashcard.
    # Capped at 30 by the recorder, like lecture.window_s.
    "quiz": {"questions": 5, "chunks": 6, "window_s": 15},
    # yield_to_trainer: when a training process appears, unload the local
    # model (brain.release) and speak the lent line; reload once the trainer
    # is gone for two ticks. Off by default: every tool answer runs through
    # the local model, so a multi-hour run leaves Jarvis with Tier 1 only.
    "health": {"warn_gb": 16, "critical_gb": 8, "hog_gb": 20, "interval_s": 30,
               "yield_to_trainer": False},
    # Long-term memory (jarvis/memory.py): facts are also indexed with
    # nomic-embed-text so "who's my dentist" finds "my dentist is Dr Patel";
    # semantic=false keeps the substring store only.
    "memory": {"semantic": True},
    # Weekly memory garden (jarvis/garden.py): once the ISO week closes,
    # the week's activity journal is read by the local model and up to
    # max_facts durable facts are filed as long-term memory, tagged
    # source="garden" so "forget the last garden pass" can take them back.
    # run_before_hour keeps the 26B extraction in the small hours; a week
    # still ungardened by Wednesday runs at any hour instead.
    "garden": {"enabled": True, "max_facts": 4, "run_before_hour": 6},
    # Activity journal (jarvis/context.py + tools/journal.py): the focused
    # window is sampled every window_interval_s; day files older than
    # keep_days are pruned. enabled=false stops the sampler only -- exchanges
    # and tool calls are always journaled.
    "journal": {"enabled": True, "window_interval_s": 60, "keep_days": 90},
    "briefing": {"enabled": False, "on_first_wake": True, "after": "06:00", "hn_items": 3,
                 "news_feeds": ["https://www.theverge.com/rss/index.xml",
                                "https://feeds.arstechnica.com/arstechnica/index"],
                 "sports_feeds": [], "stock_symbols": [],
                 # Per-section switches, set by voice ("no news in the
                 # morning"); a missing name counts as on. verbosity "brief"
                 # halves every briefing view's sentence allowance.
                 "sections": {"weather": True, "calendar": True, "news": True,
                              "sports": True, "stocks": True, "canvas": True,
                              "todos": True, "alarms": True, "reminders": True,
                              # exam-week study: the flashcard deck for the
                              # course whose exam is inside study_days, and
                              # the offer to run some now
                              "study": True},
                 "study_days": 5, "study_offer": True, "study_offer_n": 10,
                 "verbosity": "normal",
                 # The good-night preview offers a wake-up alarm when
                 # tomorrow's first event starts by early_before and no alarm
                 # already covers it: wake_lead_min before the event.
                 "wake_offer": True, "early_before": "09:00", "wake_lead_min": 60},
    "alarms": {"sound": "", "volume": 0.8, "escalate": True,
               "max_ring_s": 300, "snooze_min": 10},
    # Destructive read-back (commander._try_destructive_confirm): "cancel
    # all alarms" / "clear my list" with more than one item is read back
    # and waits for a yes. shaky_logprob: a transcript whose Whisper
    # avg_logprob is below this also gets a read-back for a one-item
    # whole-list cancel. Calibrated 2026-08-30 from the live log (n=32
    # accepted turns: p10 -0.61, median -0.41; the garbled ones sat at
    # -0.94/-0.95), so -0.7 flags the doubtful tail without nagging.
    "confirm": {"read_back": True, "shaky_logprob": -0.7},
    # "No, I said X": re-dispatch X, drop the misheard exchange, log the
    # pair to corrections.json. learn_vocab additionally appends new
    # capitalised words from X to the Whisper vocabulary prompt -- off by
    # default because Whisper's casing on a misheard name is itself a guess.
    "corrections": {"learn_vocab": False},
    "discord": {"bot_token": "", "channel_id": "", "user_id": ""},
    # Spotify (jarvis/tools/spotify.py): the developer-app credentials and the
    # speaker Jarvis reaches for when nothing else is playing. The OAuth token
    # itself lives beside this file in spotify_token.json (0600), never here.
    "spotify": {"client_id": "", "client_secret": "",
                "default_device": "HPCOMPUTER", "liked_strategy": "uris",
                "market": "from_token"},
    "autostart": {"enabled": False},
    # Quiet hours / do-not-disturb (jarvis/quiet.py). hours: "HH:MM" 24 h,
    # overnight windows wrap ("23:00" -> "07:00"); dnd_until / free_until are
    # timestamps he sets himself ("do not disturb for an hour", "I am free");
    # calendar: a running timed event whose title contains one of the
    # keywords also holds proactive speech; hold_when_away needs presence.
    "quiet": {"hours": {"start": "", "end": ""}, "dnd_until": 0, "free_until": 0,
              "calendar": True, "calendar_keywords": ["class", "exam", "meeting", "busy"],
              "hold_when_away": True},
    # Background watchers (jarvis/grades.py, mailwatch.py, keyword_watch.py):
    # unprompted lines, all held by the quiet policy. grades diffs the Canvas
    # course totals every 15 min; people_mail speaks unread mail from someone
    # in the people book every 10; keywords is a list of words or phrases to
    # watch for across unread mail and Canvas announcements -- empty means the
    # keyword watch never runs. All three are silent without their credentials.
    "watch": {"grades": True, "people_mail": True, "keywords": []},
    # Bedtime wind-down (jarvis/winddown.py): "good night" fades Spotify to
    # nothing over fade_s and pauses it, warms the screen (GNOME night
    # light) and dims it to `brightness`, and arms do-not-disturb until
    # quiet hours close (or `morning` when they are not configured).
    # "Good morning" -- or the next app start after the window -- puts it
    # all back. Off by default; every half has its own switch, and
    # brightness is floored well above black.
    "wind_down": {"enabled": False, "fade_s": 60, "brightness": 0.5,
                  "night_light": True, "music": True, "dnd": True,
                  "morning": "07:00"},
    # Presence (jarvis/presence.py): the phone's Wi-Fi address and/or MAC.
    # away_after_min is the grace before "out" -- iPhones nap off Wi-Fi for
    # minutes at a time, so anything under ~10 flaps.
    "presence": {"enabled": True, "phone_ip": "", "phone_mac": "",
                 "away_after_min": 12, "poll_s": 60},
    # The voice path's two latency/feedback policies (jarvis/app.py):
    # speculative_stt decodes the clip during the endpoint silence and reuses
    # the result when no more speech followed; nudge is the short spoken /
    # earcon cue after a wake-word turn that produced nothing to answer,
    # at most once per nudge_cooldown_s.
    "listening": {"speculative_stt": True, "nudge": True, "nudge_cooldown_s": 30},
    # Phone intercom (jarvis/intercom.py): a recorded clip sent over the
    # command socket instead of a wake word. verify_speaker runs the same
    # ECAPA gate the microphone path uses -- off by default because the
    # 0600 socket behind the user's own SSH session is already the
    # authentication, and a phone codec moves the embedding far enough that
    # the gate (which fails SHUT) would reject his own voice. max_mb caps
    # one clip; the socket framing allows a little more than this.
    "intercom": {"enabled": True, "verify_speaker": False, "max_mb": 10},
}

SECRET_KEYS = ("icloud.app_password", "gmail.app_password", "discord.bot_token",
               "spotify.client_secret", "canvas.token")
# Secrets that live inside a LIST of sections rather than at a dotted path:
# (list key, field). gmail.accounts[].app_password was invisible to redacted()
# and scrub(), so repr(cfg) printed three real app passwords in full.
SECRET_LIST_FIELDS = (("gmail.accounts", "app_password"),)

# is_configured() / setup_line() sections and the film-JARVIS excuse for each.
SETUP_LINES: dict[str, str] = {
    "home_location": "I'll need your home location set up, sir; "
                     f"the notes are in {DOCS_HINT}.",
    "google_ical": "I'll need your Google calendar link set up, sir; "
                   f"the notes are in {DOCS_HINT}.",
    "icloud": "I'll need your iCloud calendar set up, sir; "
              f"the notes are in {DOCS_HINT}.",
    "gmail": "I'll need your Gmail app password set up, sir; "
             f"the notes are in {DOCS_HINT}.",
    "discord": "I'll need the Discord bot set up, sir; "
               f"the notes are in {DOCS_HINT}.",
    "claude": "I'll need the Claude command line set up, sir; "
              f"the notes are in {DOCS_HINT}.",
}
SECTIONS = tuple(SETUP_LINES)

# Values that mean "not filled in yet": empty, "<paste here>", "PASTE-…",
# "your-…", "changeme", "xxxx…".  Real secrets never look like these.
_PLACEHOLDER = re.compile(
    r"^\s*$|^<.*>$|^(paste|your|replace|todo|example)[-_ :]|"
    r"^change[ -_]?me\b|^x{3,}$", re.I)


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return False
    if not isinstance(value, str):
        return not value
    return bool(_PLACEHOLDER.search(value.strip()))


def _deep_merge(base: dict, override: dict) -> dict:
    """Return ``base`` with ``override`` laid over it; dicts merge
    recursively, lists and scalars replace.  Unknown keys survive."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def config_path(path: Optional[os.PathLike | str] = None) -> Path:
    """Resolve the config file: explicit arg, else $JARVIS_ASSISTANT_CONFIG,
    else ``~/.config/jarvis/assistant.json``."""
    raw = path or os.environ.get(ENV_VAR) or ""
    if raw:
        return Path(os.path.expanduser(str(raw)))
    return Path.home() / ".config" / "jarvis" / "assistant.json"


def _claude_bin() -> str:
    """Seam for is_configured('claude'); mirrors MachineProfile.claude_bin
    without importing jarvis.config (keeps this module import-light).  A
    GNOME autostart launch may lack ~/.local/bin on PATH, so that install
    location is the last resort."""
    found = os.environ.get("JARVIS_CLAUDE_BIN") or shutil.which("claude")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "claude"
    return str(local) if os.access(local, os.X_OK) else ""


def _write_private(path: Path, text: str) -> None:
    """Atomic write with mode 0600: temp file in the same directory
    (created 0600 by mkstemp), fsync, os.replace, chmod to be sure."""
    # mode applies only when the directory is created (never widened later)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix=".assistant-", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.chmod(path, 0o600)


def _file_stamp(path: Path):
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


class _Section:
    """Live attribute view over one dict of the config: ``cfg.alarms.volume``
    reads through ``cfg.get("alarms.volume")`` every time, so it never goes
    stale after ``set()`` or ``reload_if_changed()``.  Assignment saves:
    ``cfg.alarms.volume = 0.5``."""
    __slots__ = ("_cfg", "_prefix")

    def __init__(self, cfg: "AssistantConfig", prefix: str):
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_prefix", prefix)

    def _key(self, name: str) -> str:
        return f"{self._prefix}.{name}"

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        sentinel = object()
        value = self._cfg.get(self._key(name), sentinel)
        if value is sentinel:
            raise AttributeError(
                f"assistant.json has no key {self._key(name)!r}")
        if isinstance(value, dict):
            return _Section(self._cfg, self._key(name))
        return value

    def __setattr__(self, name: str, value) -> None:
        self._cfg.set(self._key(name), value)

    def __getitem__(self, name: str):
        sentinel = object()
        value = self._cfg.get(self._key(name), sentinel)
        if value is sentinel:
            raise KeyError(self._key(name))
        return value

    def get(self, name: str, default=None):
        return self._cfg.get(self._key(name), default)

    def as_dict(self) -> dict:
        return self._cfg.get(self._prefix, {}) or {}

    def keys(self):
        return self.as_dict().keys()

    def items(self):
        return self.as_dict().items()

    def __contains__(self, name: str) -> bool:
        return name in self.as_dict()

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())

    def __eq__(self, other) -> bool:
        if isinstance(other, _Section):
            return self.as_dict() == other.as_dict()
        return self.as_dict() == other

    def __repr__(self) -> str:
        shown = self._cfg.redacted()
        for part in self._prefix.split("."):
            shown = shown.get(part, {}) if isinstance(shown, dict) else {}
        return f"<{self._prefix} {shown!r}>"


class AssistantConfig:
    """See the module docstring and spec 10.2."""

    DEFAULTS = DEFAULTS
    SECRET_KEYS = SECRET_KEYS
    SECTIONS = SECTIONS

    def __init__(self, data: Optional[dict] = None,
                 path: Optional[os.PathLike | str] = None):
        self._lock = threading.RLock()
        self.path: Optional[Path] = Path(path) if path else None
        self._data: dict = _deep_merge(DEFAULTS, data or {})
        self._stamp = _file_stamp(self.path) if self.path else None

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Optional[os.PathLike | str] = None) -> "AssistantConfig":
        """Never raises.  Creates the file (0600, placeholders) when missing;
        moves a corrupt one to ``<name>.bad`` and recreates it; fills in any
        keys added since the file was written."""
        p = config_path(path)
        raw: dict = {}
        need_write = False
        try:
            if p.exists():
                try:
                    loaded = json.loads(p.read_text(encoding="utf-8"))
                    if not isinstance(loaded, dict):
                        raise ValueError(f"top level is {type(loaded).__name__}")
                    raw = loaded
                except (ValueError, UnicodeDecodeError) as exc:
                    bad = p.with_name(p.name + ".bad")
                    log.warning("assistant config %s is corrupt (%s); "
                                "moved to %s and recreated", p, exc, bad)
                    os.replace(p, bad)
                    raw, need_write = {}, True
                else:
                    try:
                        mode = p.stat().st_mode & 0o777
                        if mode & 0o077:
                            os.chmod(p, 0o600)
                            log.warning("assistant config had mode %o; "
                                        "tightened to 600", mode)
                    except OSError:
                        log.warning("could not check mode of %s", p)
            else:
                need_write = True
                log.info("assistant config missing; creating %s with placeholders", p)
        except OSError:
            log.exception("assistant config %s unreadable; using defaults in memory", p)
        cfg = cls(raw, p)
        if not need_write and cfg._data != raw:
            need_write = True          # new keys since the file was written
            log.info("assistant config %s gained new default keys", p)
        if need_write:
            cfg.save()
        return cfg

    # ------------------------------------------------------------- access
    def get(self, dotted: str, default: Any = None) -> Any:
        """Dotted lookup (``"claude.model"``).  Dicts and lists come back as
        copies: mutate-and-forget can't silently change the config."""
        node: Any = self._data
        with self._lock:
            for part in dotted.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return default
            if isinstance(node, (dict, list)):
                return copy.deepcopy(node)
            return node

    def set(self, dotted: str, value: Any) -> bool:
        """Set a dotted key (intermediate dicts are created) and save
        atomically.  Returns True when the file was written."""
        parts = dotted.split(".")
        with self._lock:
            node = self._data
            for part in parts[:-1]:
                child = node.get(part)
                if not isinstance(child, dict):
                    if child is not None:
                        log.warning("assistant config: %s was %r; now a section",
                                    part, child)
                    child = node[part] = {}
                node = child
            node[parts[-1]] = copy.deepcopy(value)
        return self.save()

    def update(self, values: dict) -> bool:
        """Several dotted keys, one save."""
        with self._lock:
            for dotted, value in values.items():
                parts = dotted.split(".")
                node = self._data
                for part in parts[:-1]:
                    child = node.get(part)
                    if not isinstance(child, dict):
                        child = node[part] = {}
                    node = child
                node[parts[-1]] = copy.deepcopy(value)
        return self.save()

    def as_dict(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._data)

    def __getattr__(self, name: str):
        # Only reached for names that are not real attributes: top-level
        # config keys.  cfg.claude -> section view, cfg.units -> "us".
        if name.startswith("_") or name in ("path",):
            raise AttributeError(name)
        data = self.__dict__.get("_data")
        if data is None or name not in data:
            raise AttributeError(f"{type(self).__name__} has no attribute "
                                 f"or config key {name!r}")
        value = self.get(name)
        if isinstance(value, dict):
            return _Section(self, name)
        return value

    # -------------------------------------------------------- persistence
    def save(self) -> bool:
        """Atomic 0600 write of the whole config.  Never raises."""
        if self.path is None:
            return False
        # Serialize AND replace under the lock: two concurrent saves must
        # land in snapshot order, or an older snapshot could win the race.
        with self._lock:
            text = json.dumps(self._data, indent=2, ensure_ascii=False) + "\n"
            try:
                _write_private(self.path, text)
            except OSError:
                log.exception("assistant config save failed: %s", self.path)
                return False
            self._stamp = _file_stamp(self.path)
        return True

    def reload_if_changed(self) -> bool:
        """Re-read the file when its mtime/size moved (the user edited it
        by hand).  A corrupt edit keeps the in-memory copy and logs.
        Returns True when the data was reloaded."""
        if self.path is None:
            return False
        stamp = _file_stamp(self.path)
        if stamp == self._stamp:
            return False
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("top level is not an object")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            log.warning("assistant config changed on disk but is unreadable "
                        "(%s); keeping the loaded copy", exc)
            self._stamp = stamp
            return False
        with self._lock:
            self._data = _deep_merge(DEFAULTS, loaded)
        self._stamp = stamp
        log.info("assistant config reloaded from %s", self.path)
        return True

    # ------------------------------------------------------ configured?
    def is_configured(self, section: str) -> bool:
        """True when the section has what its tool needs (placeholders and
        empty values do not count).  Unknown sections are False."""
        if section == "home_location":
            lat, lon = self.get("home_location.lat"), self.get("home_location.lon")
            return _is_number(lat) and _is_number(lon)
        if section == "google_ical":
            urls = self.get("google_ical_urls") or []
            return any(isinstance(u, str) and u.strip().lower().startswith(
                ("http://", "https://", "webcal://")) and not _is_placeholder(u)
                for u in urls)
        if section == "icloud":
            return not _is_placeholder(self.get("icloud.apple_id")) and \
                not _is_placeholder(self.get("icloud.app_password"))
        if section == "gmail":
            # Multi-account configs carry gmail.accounts and no top-level
            # pair; without this the setup line kept telling the user to
            # configure mail that was already working.
            for entry in (self.get("gmail.accounts") or []):
                if isinstance(entry, dict) and \
                        not _is_placeholder(entry.get("address")) and \
                        not _is_placeholder(entry.get("app_password")):
                    return True
            return not _is_placeholder(self.get("gmail.address")) and \
                not _is_placeholder(self.get("gmail.app_password"))
        if section == "discord":
            return not _is_placeholder(self.get("discord.bot_token")) and \
                not _is_placeholder(self.get("discord.channel_id"))
        if section == "claude":
            return bool(_claude_bin()) and bool(self.allowed_dirs)
        return False

    def missing_sections(self) -> list[str]:
        return [s for s in SECTIONS if not self.is_configured(s)]

    @staticmethod
    def setup_line(section: str) -> str:
        """The spoken film-JARVIS excuse for a section that is not set up."""
        line = SETUP_LINES.get(section)
        if line:
            return line
        what = section.replace("_", " ").strip() or "that"
        return f"I'll need {what} set up, sir; the notes are in {DOCS_HINT}."

    @staticmethod
    def setup_lines() -> list[str]:
        """Every fixed excuse, for the speech-cache prewarm."""
        return list(SETUP_LINES.values())

    # ------------------------------------------------------------ secrets
    def redacted(self) -> dict:
        """A deep copy with every SECRET_KEYS value masked as "•••" (an
        empty placeholder stays "" so a settings view can show it is unset)."""
        data = self.as_dict()
        for dotted in SECRET_KEYS:
            parts = dotted.split(".")
            node = data
            for part in parts[:-1]:
                node = node.get(part) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, dict) and parts[-1] in node:
                value = node[parts[-1]]
                if not _is_placeholder(value):
                    node[parts[-1]] = MASK
        for list_key, field in SECRET_LIST_FIELDS:
            node = data
            for part in list_key.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            for entry in (node if isinstance(node, list) else []):
                if isinstance(entry, dict) and not _is_placeholder(entry.get(field)):
                    entry[field] = MASK
        return data

    def secret_values(self) -> list[str]:
        values = [v for v in (self.get(k) for k in SECRET_KEYS)
                  if isinstance(v, str) and not _is_placeholder(v)]
        for list_key, field in SECRET_LIST_FIELDS:
            for entry in (self.get(list_key) or []):
                v = entry.get(field) if isinstance(entry, dict) else None
                if isinstance(v, str) and not _is_placeholder(v):
                    values.append(v)
        return values

    def scrub(self, text: Any) -> str:
        """Replace every configured secret value inside ``text`` with "•••"
        (for log lines, subprocess argv echoes, error messages)."""
        text = "" if text is None else str(text)
        for value in sorted(self.secret_values(), key=len, reverse=True):
            text = text.replace(value, MASK)
        return text

    def __repr__(self) -> str:
        return f"AssistantConfig(path={str(self.path)!r}, {self.redacted()!r})"

    __str__ = __repr__

    # --------------------------------------------------------- properties
    @property
    def home_location(self) -> Optional[dict]:
        """``{"city","region","lat","lon"}`` or None until lat/lon are set
        (then the location tool falls back to the IP lookup)."""
        if not self.is_configured("home_location"):
            return None
        loc = self.get("home_location") or {}
        return {"city": loc.get("city", "") or "", "region": loc.get("region", "") or "",
                "lat": float(loc["lat"]), "lon": float(loc["lon"])}

    @property
    def user_name(self) -> str:
        return str(self.get("user.name") or "Hunter")

    @property
    def units(self) -> str:
        return str(self.get("units") or "us")

    @property
    def local_model(self) -> str:
        return str(self.get("local_model") or DEFAULTS["local_model"])

    @property
    def allowed_dirs(self) -> list[str]:
        dirs = self.get("claude.allowed_dirs") or []
        return [os.path.abspath(os.path.expanduser(str(d)))
                for d in dirs if isinstance(d, str) and d.strip()]

    @property
    def projects_root(self) -> str:
        root = self.get("claude.projects_root") or DEFAULTS["claude"]["projects_root"]
        return os.path.abspath(os.path.expanduser(str(root)))

    @property
    def skill_phrases(self) -> dict:
        phrases = self.get("claude.skill_phrases") or {}
        return {str(k): str(v) for k, v in phrases.items()}

    # ------------------------------------------------------------ paths
    def is_allowed_path(self, path: os.PathLike | str,
                        base: Optional[os.PathLike | str] = None) -> bool:
        """True when the real path (symlinks and ``..`` resolved) is one of
        the allowed dirs or under one.  A relative ``path`` is taken against
        ``base`` (the project cwd) when given, else the process cwd."""
        try:
            raw = os.path.expanduser(str(path))
            if base is not None and not os.path.isabs(raw):
                raw = os.path.join(os.path.expanduser(str(base)), raw)
            real = os.path.realpath(raw)
        except (OSError, ValueError):
            return False
        for allowed in self.allowed_dirs:
            root = os.path.realpath(allowed)
            if real == root or real.startswith(root.rstrip(os.sep) + os.sep):
                return True
        return False

    def add_allowed_dir(self, path: os.PathLike | str) -> bool:
        """Add a directory (absolute, ``~`` expanded) and save.  Returns
        True when it was new."""
        new = os.path.abspath(os.path.expanduser(str(path)))
        with self._lock:
            current = [os.path.abspath(os.path.expanduser(str(d)))
                       for d in (self.get("claude.allowed_dirs") or [])]
            real_new = os.path.realpath(new)
            if any(os.path.realpath(d) == real_new for d in current):
                return False
            current.append(new)
        self.set("claude.allowed_dirs", current)
        return True

    def remove_allowed_dir(self, path: os.PathLike | str) -> bool:
        target = os.path.realpath(os.path.expanduser(str(path)))
        current = self.get("claude.allowed_dirs") or []
        kept = [d for d in current
                if os.path.realpath(os.path.expanduser(str(d))) != target]
        if len(kept) == len(current):
            return False
        self.set("claude.allowed_dirs", kept)
        return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and value == value      # not NaN
