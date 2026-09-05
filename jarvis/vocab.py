"""The Whisper initial_prompt, built from what Hunter actually says.

The transcriber used to prime every utterance with the warehouse list the
V3 port carried over from VSS ("AGV, forklift, pallet, conveyor, ...") --
jargon he never speaks, biasing transcripts toward it ("Formula One ring",
"system, yes." at -0.95 in the live log). This module builds the prompt
from HIS world instead, layered so what he taught explicitly survives the
cap and harvested material falls off the tail first:

1. ``~/.aiws_trainer/voice_vocab.txt`` -- the manual vocabulary file (the
   Tk editor writes one term per line) plus the words the corrections
   feature learns (``corrections.learn_vocab`` appends comma-joined).
2. ``~/.aiws_trainer/voice_names.txt`` -- names taught by voice ("my
   advisor's name is spelled P-E-Y-R-O-V-I"), newest first.
3. His TTS pronunciation entries (jarvis/pronounce.py user file): a name
   he taught the voice to SAY is a name Whisper should HEAR. User entries
   only -- the ~100 shipped defaults would eat the whole budget, and the
   shipped ones that matter for hearing are in the seed already.
4. ``DEFAULT_VOCAB`` (jarvis/transcriber.py) -- the assistant seed.
5. Calendar event titles from the read-only disk cache
   (``~/.cache/jarvis/calendar_cache.json``) -- course names such as
   "BIOSENSORS" arrive here with no network and no CalendarService.
   RECURRING titles only (MIN_TITLE_RECURRENCE events within ONE
   source): a course is a name he says out loud, a one-off appointment,
   visit or delivery is a name he never says to Jarvis AND private
   detail. 2026-09-04 20:58:45: a surname from a one-off appointment
   title, cut by title[:48] to end in "<surname>,", was echoed four
   times by Whisper on unclear audio and became a "Was that for me?"
   card showing the name on screen and out loud. The live cache held
   14 distinct titles, 11 of them one-offs (3 sources, 30 events, no
   title on more than one source; the 3 courses have 6, 6 and 7
   events each). Counted per source so that the same appointment
   mirrored on two calendars is still a one-off.
6. Canvas course names via ``canvas.cached_course_names()`` -- a snapshot
   of that tool's module cache, NEVER a fetch.
7. The buildings in those same events, normalised through
   ``leavetime.building_key`` -- "Wisenbaker", "Emerging Technologies".
   He says these out loud to teach a walk, and they are proper nouns no
   language model expects.

Whisper's prompt window is ~224 tokens and which end a backend trims
differs, so the prompt is capped HERE (PROMPT_CHAR_CAP) and the cap drops
OUR tail -- the harvested, lowest-priority terms. Per-term caps (titles
48, buildings 32) cut at a WORD boundary and strip trailing punctuation
(clip_term), and build_prompt() strips a trailing comma from EVERY term:
a comma-separated prompt whose term ends in "," reads as "X,, Y", and a
term that ends in a comma is exactly the shape a greedy decoder continues
as "X, X, X, ..." (the 20:58:45 echo above). build_prompt() is called
on the hot voice path (every partial() too), so the result is cached
PROMPT_TTL_S seconds; add_name()/save_vocab() bust the cache so a freshly
taught name reaches the very next attempt.

Nothing here writes outside ``PATHS.NAMES_FILE``; the calendar cache and
the Canvas course cache are read-only inputs.
"""
from __future__ import annotations

import json
import re
import threading
import time

from jarvis.config import PATHS
from jarvis.logs import get_logger
from jarvis.transcriber import DEFAULT_VOCAB

log = get_logger("vocab")

# ~220 tokens at the usual ~4 chars/token -- under Whisper's ~224-token
# prompt window, same budget as commander.VOCAB_CHAR_CAP (900).
PROMPT_CHAR_CAP = 880
PROMPT_TTL_S = 60.0
MAX_CALENDAR_TITLES = 20
MAX_BUILDINGS = 8
MAX_COURSES = 12
# Events a title must appear in (whitespace-collapsed, lowercased, within
# ONE source of the cache) before it is a name worth priming Whisper
# with. 2 is the smallest count that separates "his courses" (6+ each in
# the live cache) from the 11 one-off appointments there; a once-a-term
# seminar with a single event stays out, and it is the seminar's NAME
# that stays out, never the event. Known cost: a weekly course whose
# only meeting inside calendar.WINDOW_DAYS falls on a holiday week or a
# term boundary loses its name for that fortnight.
MIN_TITLE_RECURRENCE = 2
TITLE_CHARS = 48
BUILDING_CHARS = 32

_clock = time.monotonic        # test seam
_lock = threading.Lock()
_cached: tuple[float, str] | None = None
_calendar_cached: tuple[float, list, list] | None = None
# (file mtime, event titles, building names)


def clear_cache() -> None:
    """Forget the built prompt (and the calendar-title parse) so the next
    build_prompt() re-reads everything. Called by add_name() and by
    transcriber.save_vocab() -- a just-taught word must reach the very
    next transcription, not the one after the TTL."""
    global _cached, _calendar_cached
    with _lock:
        _cached = None
        _calendar_cached = None


# ------------------------------------------------------------- names file
def load_names() -> list:
    """Names from ``PATHS.NAMES_FILE`` (one per line, ``#`` comments),
    newest FIRST: the file is append-only, and when the cap bites, the
    name he taught this week should outlive one from last term."""
    try:
        if not PATHS.NAMES_FILE.exists():
            return []
        lines = PATHS.NAMES_FILE.read_text(encoding="utf-8").splitlines()
    except Exception:
        log.exception("names file unreadable: %s", PATHS.NAMES_FILE)
        return []
    out, seen = [], set()
    for line in reversed(lines):
        name = " ".join(line.split())
        if not name or name.startswith("#"):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def add_name(name: str) -> bool:
    """Append one name to the names file (case-insensitively deduped).
    Returns True when it was new. The same file the spelled-name command
    writes; corrections keep appending to voice_vocab.txt and both feed
    build_prompt(), so the two teaching paths are one mechanism."""
    name = " ".join(str(name or "").split())
    if not name:
        return False
    if name.lower() in {n.lower() for n in load_names()}:
        return False
    PATHS.NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PATHS.NAMES_FILE.open("a", encoding="utf-8") as fh:
        fh.write(name + "\n")
    clear_cache()
    log.info("vocabulary name added: %r", name)
    return True


# ------------------------------------------------------------- harvesters
_TRAILING_PUNCT = ",:;.- \t"


def clip_term(term: str, cap: int) -> str:
    """Clip one prompt term to ``cap`` chars at a WORD boundary -- the last
    whitespace at or before the cap; a first word that alone exceeds the
    cap keeps the hard cut -- and then strip trailing punctuation, so no
    term can end in a dangling comma, colon or dash. title[:48] used to
    cut "... <initial>. <surname>, <suffix>" to "... <surname>,"
    (2026-09-04)."""
    term = " ".join(str(term or "").split())
    if len(term) > cap:
        cut = term.rfind(" ", 0, cap + 1)
        term = term[:cut] if cut > 0 else term[:cap]
    return term.rstrip(_TRAILING_PUNCT)


def _user_vocab() -> str:
    """The raw manual/corrections vocabulary file. Read directly rather
    than through transcriber.load_vocab(): that helper substitutes
    DEFAULT_VOCAB when the file is missing, and here the seed is its own
    layer further down the prompt."""
    try:
        if PATHS.VOCAB_FILE.exists():
            return PATHS.VOCAB_FILE.read_text(encoding="utf-8")
    except Exception:
        log.exception("vocab file unreadable: %s", PATHS.VOCAB_FILE)
    return ""


def _pronounce_keys() -> list:
    try:
        from jarvis import pronounce
        return list(pronounce.get().user_items().keys())
    except Exception:
        log.exception("pronunciation keys unavailable for the prompt")
        return []


def _building_name(location) -> str:
    """The spoken building in a calendar location string, or "".

    Reuses jarvis/leavetime.py's normaliser so the prompt and the learned
    walks agree on what a building is called: the two ETB rooms are one
    "Emerging Technologies", the Zoom URL and the empty locations are
    nothing. Imported here rather than at module scope -- build_prompt runs
    on the hot voice path and this file's imports stay small."""
    try:
        from jarvis.leavetime import building_key, speech_name
    except Exception:  # noqa: BLE001 - the prompt is best-effort
        return ""
    key = building_key(location)
    return speech_name(key) if key else ""


def _parse_calendar_cache() -> tuple:
    """(titles, buildings) from the calendar's disk cache, parsed once per
    file mtime. Read-only on purpose: no CalendarService, no refresh, no
    lock shared with the tool -- a stale title still biases Whisper right."""
    global _calendar_cached
    path = PATHS.CACHE_DIR / "calendar_cache.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return [], []
    hit = _calendar_cached
    if hit is not None and hit[0] == mtime:
        return list(hit[1]), list(hit[2])
    order, counts = [], {}          # first spelling seen; key -> {source: n}
    buildings, seen_b = [], set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        sources = data.get("sources") if isinstance(data, dict) else None
        for name, entry in (sources or {}).items():
            events = entry.get("events") if isinstance(entry, dict) else None
            for ev in events if isinstance(events, list) else []:
                if not isinstance(ev, dict):
                    continue
                title = " ".join(str(ev.get("title") or "").split())
                key = title.lower()
                if title and key != "untitled":
                    if key not in counts:
                        counts[key] = {}
                        order.append(title)
                    per = counts[key]
                    per[name] = per.get(name, 0) + 1
                place = clip_term(_building_name(ev.get("location")),
                                  BUILDING_CHARS)
                if place and place.lower() not in seen_b:
                    seen_b.add(place.lower())
                    buildings.append(place)
    except Exception:
        log.exception("calendar cache unreadable for the prompt")
        return [], []
    # The cache carries no recurrence field, so recurrence is COUNTED,
    # and counted WITHIN one source: a one-off appointment mirrored on a
    # second calendar is the same event twice, not a course, and a
    # cross-source sum would have put its surname straight back into the
    # prompt. A course recurs on the calendar that holds it (see the
    # module docstring, item 5).
    titles = [clip_term(t, TITLE_CHARS) for t in order
              if max(counts[t.lower()].values()) >= MIN_TITLE_RECURRENCE]
    titles = [t for t in titles if t][:MAX_CALENDAR_TITLES]
    buildings = buildings[:MAX_BUILDINGS]
    _calendar_cached = (mtime, titles, buildings)
    return list(titles), list(buildings)


def _calendar_titles() -> list:
    return _parse_calendar_cache()[0]


def _calendar_buildings() -> list:
    """The buildings he actually walks to. He says these names out loud --
    "how long to Wisenbaker" teaches jarvis/leavetime.py a walk -- and a
    proper noun no language model expects is exactly what an initial_prompt
    is for. Last in the priority order: the cap should eat a building
    before it eats a name he taught by hand."""
    return _parse_calendar_cache()[1]


def _course_names() -> list:
    """Canvas course names already sitting in the tool's module cache --
    cached_course_names() never fetches, so an empty list before the
    first Canvas question is the correct price."""
    try:
        from jarvis.tools import canvas
        return canvas.cached_course_names()[:MAX_COURSES]
    except Exception:
        log.exception("canvas course names unavailable for the prompt")
        return []


# ------------------------------------------------------------- the prompt
_SPLIT_RX = re.compile(r"[,\n]")


def _terms(text: str) -> list:
    return [t for t in (" ".join(p.split()) for p in _SPLIT_RX.split(text or ""))
            if t]


def build_prompt() -> str:
    """The initial_prompt for the next transcription (cached PROMPT_TTL_S).
    Wired into the Transcriber as ``prompt_provider`` by app.py; must be
    cheap, must never raise on the voice path (each harvester eats its
    own failures), and must never touch the network."""
    global _cached
    now = _clock()
    with _lock:
        if _cached is not None and now - _cached[0] < PROMPT_TTL_S:
            return _cached[1]

    picked, seen, total = [], set(), 0
    capped = False
    # Buildings before titles and courses: they are short tokens he says
    # OUT LOUD ("how long to Wisenbaker") and were measured to carry the
    # domain-term win (hard-set WER 3.66% -> 1.08%); a 48-char lecture
    # title is the worst use of the same budget.
    for term in (_terms(_user_vocab()) + load_names() + _pronounce_keys()
                 + _terms(DEFAULT_VOCAB) + _calendar_buildings()
                 + _calendar_titles() + _course_names()):
        # No term may begin or end in a comma, whichever layer it came
        # from: the prompt is comma-joined, so "X," becomes "X,, Y" and
        # ",X" becomes "Y, ,X" -- an empty list item either way -- and a
        # term that ends in a comma is the shape Whisper continued as
        # "<surname>, <surname>, <surname>, <surname>," on 2026-09-04
        # (a one-off appointment title cut at the cap; see clip_term).
        # Names, pronunciation keys and titles are never split on commas
        # before they get here, so both ends are stripped here.
        term = str(term or "").strip(", \t")
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        cost = len(term) + (2 if picked else 0)
        if total + cost > PROMPT_CHAR_CAP:
            capped = True
            continue                  # skip the long term, keep the short ones
        seen.add(key)
        picked.append(term)
        total += cost

    prompt = ", ".join(picked)
    with _lock:
        _cached = (now, prompt)
    log.info("whisper prompt built: %d terms, %d chars%s",
             len(picked), len(prompt), " (capped)" if capped else "")
    return prompt
