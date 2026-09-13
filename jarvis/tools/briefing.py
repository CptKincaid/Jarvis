"""Morning briefing data (spec section 6.7).

``build_briefing(cfg, registry, fetch, now, cache_path)`` returns
``(sections, fact_sheet)``: weather and calendar through the tool
registry (``get_weather`` / ``get_calendar``), Tech & AI news from Hacker
News (top stories by score) plus the first tech item of each configured
RSS/Atom feed (defaults The Verge + Ars Technica), and sports / stocks
only when the config lists feeds / symbols. The brain renders the spoken
briefing from the fact sheet; the app publishes ``BriefingReady``.

All HTTP goes through the module-level ``_fetch`` seam (tests pass a
fake). Feeds are fetched in parallel with a bounded pool so a slow
source cannot stall the tool loop; any source failing degrades to the
sections that worked. News is cached 15 min at ``cache_path``.

Two more views share the tool (``get_briefing(when=...)``):
``build_preview`` is tomorrow in one breath for "good night" (first event
and where, tomorrow's weather, Canvas due within a day, open to-dos, the
alarm that is set -- and a wake-up offer when the first event is early and
no alarm covers it); ``build_week`` is the workload forecast, day by day
(calendar, Canvas, reminders) with the heavy and the clear days named.
Neither fetches news: a preview or a forecast is not the morning paper.

Every section obeys ``briefing.sections.<name>`` (True unless switched
off by voice: "no news in the morning"), and ``briefing.verbosity``
("brief") halves the sentence allowance the model is given.
"""
from __future__ import annotations

import csv
import io
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.tools.registry import ToolResult, ToolSpec
from jarvis.tools.timekeeper import count_words

log = get_logger("tools.briefing")

NOTHING_LINE = "I couldn't reach any of the briefing sources, sir."
PREVIEW_NOTHING_LINE = "I can't see anything for tomorrow, sir; the sources are out of reach."
WEEK_NOTHING_LINE = "I can't see your week, sir; the sources are out of reach."
WHENS = ("today", "tomorrow", "week")
# Every section a briefing view can carry. briefing.sections.<name> false
# drops one (voice: "no news in the morning"); unknown names are ignored
# so a typo in the file cannot blank the card.
SECTIONS = ("weather", "calendar", "news", "sports", "stocks",
            "canvas", "todos", "alarms", "reminders", "study")
# Spoken allowance per view; "brief" verbosity halves it (never below 2).
VIEW_SENTENCES = {"today": 6, "tomorrow": 5, "week": 8}
WAKE_LABEL = "wake up"
OFFER_LINE = "Shall I wake you at {time}, sir?"
OFFER_TTL_S = 180.0            # a bedtime yes is quick; anything later is a new subject
WEEK_DAYS = 7
# Exam-week study. Canvas already finds the next exam and the flashcard
# store already filters by topic; the briefing is where the two meet.
STUDY_DAYS = 5                     # an exam further out is not this week's problem
STUDY_OFFER_N = 10                 # cards the "shall we run ten now" offer starts
STUDY_DUE_CAP = 200                # rows read to size the deck; not a session length
STUDY_CARDS_LINE = "{cards} due on your {course} deck{weak}."
STUDY_OFFER_LINE = "Shall we run {n} now, sir?"
STUDY_NO_DECK_LINE = ("Nothing on your {course} deck yet, sir; say quiz me on "
                      "{course} and I'll build one.")
DEFAULT_NEWS_FEEDS = ["https://www.theverge.com/rss/index.xml",
                      "https://feeds.arstechnica.com/arstechnica/index"]
HN_TOP_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{id}.json"
HN_SOURCE = "Hacker News"
STOOQ_URL = "https://stooq.com/q/l/?s={sym}.us&f=sd2t2ohlcv&h&e=csv"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=1d&interval=1d"
NEWS_ITEMS = 3                     # total news items in the briefing
HN_SCAN = 30                       # top-story ids ranked by score
FEED_SCAN = 6                      # entries examined per feed for a tech item
NEWS_CACHE_S = 15 * 60
FETCH_TIMEOUT = 6.0
POOL_WORKERS = 8
USER_AGENT = "Jarvis/1.0 (personal assistant; +https://github.com/hunterp)"
# Feed categories that mark a non-tech story (the Ars "all content" feed
# carries culture/obituaries; the briefing is Tech & AI only).
NON_TECH_CATEGORIES = {"culture", "obituaries", "obituary", "entertainment",
                       "film", "movies", "tv", "television", "music",
                       "sports", "sport", "food", "lifestyle", "celebrity",
                       "books", "fashion", "travel", "health", "politics"}


# ----------------------------------------------------------------- http
def _fetch(url: str, timeout: float = FETCH_TIMEOUT, headers: Optional[dict] = None) -> bytes:
    """The ONE network seam: GET url -> bytes (raises on failure)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


Fetch = Callable[..., bytes]


def _safe_fetch(fetch: Fetch, url: str, timeout: float = FETCH_TIMEOUT) -> Optional[bytes]:
    try:
        return fetch(url, timeout=timeout)
    except TypeError:
        try:
            return fetch(url)
        except Exception as exc:              # noqa: BLE001 - source boundary
            log.warning("fetch failed %s: %s", url[:80], type(exc).__name__)
            return None
    except Exception as exc:                  # noqa: BLE001 - source boundary
        log.warning("fetch failed %s: %s", url[:80], type(exc).__name__)
        return None


def _cfg_get(cfg, dotted: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        cur = cfg
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return default if cur is None else cur
    get = getattr(cfg, "get", None)
    if callable(get):
        try:
            val = get(dotted, default)
            return default if val is None else val
        except Exception:
            log.debug("cfg.get(%s) failed", dotted, exc_info=True)
    cur = cfg
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "enabled")
    return bool(value)


def _int_cfg(cfg, dotted: str, default: int) -> int:
    """A hand-edited assistant.json can hold "5" or a typo; neither may
    take the briefing down."""
    try:
        return int(_cfg_get(cfg, dotted, default))
    except (TypeError, ValueError):
        return default


def section_on(cfg, name: str) -> bool:
    """``briefing.sections.<name>``: True unless explicitly switched off."""
    return _truthy(_cfg_get(cfg, f"briefing.sections.{name}", True))


def verbosity_cap(cfg, normal: int) -> int:
    """The sentence allowance for a view: ``normal``, or half of it (at
    least two) when ``briefing.verbosity`` is "brief" ("shorter
    briefings", "be briefer")."""
    mode = str(_cfg_get(cfg, "briefing.verbosity", "normal") or "normal").strip().lower()
    if mode in ("brief", "short", "shorter", "terse"):
        return max(2, int(normal) // 2)
    return int(normal)


def coerce_when(value) -> str:
    """Loose model values -> one of WHENS ("tonight" -> "tomorrow")."""
    text = " ".join(str(value or "").lower().split()).strip(" .?!")
    if not text:
        return "today"
    if text in WHENS:
        return text
    if "tomorrow" in text or "tonight" in text or "evening" in text or \
            text in ("tmrw", "tmr", "preview", "next day"):
        return "tomorrow"
    if "week" in text or "7 day" in text or "seven day" in text:
        return "week"
    return "today"


# --------------------------------------------------------- hacker news
def hn_top(fetch: Fetch, n: int = NEWS_ITEMS, scan: int = HN_SCAN,
           pool: Optional[ThreadPoolExecutor] = None) -> Optional[list[dict]]:
    """Top ``n`` stories by score among the first ``scan`` top-story ids.
    None when the id list is unreachable (a failed source), [] when it
    was empty."""
    raw = _safe_fetch(fetch, HN_TOP_URL)
    if raw is None:
        return None
    try:
        ids = [int(i) for i in json.loads(raw)][:scan]
    except (ValueError, TypeError):
        log.warning("hn: bad topstories payload")
        return None

    def item(i):
        blob = _safe_fetch(fetch, HN_ITEM_URL.format(id=i))
        if blob is None:
            return None
        try:
            return json.loads(blob)
        except ValueError:
            return None

    if pool is not None:
        items = list(pool.map(item, ids))
    else:
        items = [item(i) for i in ids]
    stories = []
    for it in items:
        if not isinstance(it, dict) or it.get("type", "story") != "story":
            continue
        title = " ".join(str(it.get("title") or "").split())
        if not title or it.get("dead") or it.get("deleted"):
            continue
        stories.append({"title": title, "source": HN_SOURCE,
                        "score": int(it.get("score") or 0),
                        "url": it.get("url") or ""})
    stories.sort(key=lambda s: -s["score"])
    return stories[:n]


# ------------------------------------------------------------ rss/atom
def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _text(el) -> str:
    if el is None:
        return ""
    txt = el.text or ""
    if not txt.strip() and len(el):
        txt = "".join(el.itertext())
    return " ".join(re.sub(r"<[^>]+>", " ", txt).split())


def parse_feed(xml_bytes: bytes) -> tuple[str, list[dict]]:
    """RSS 2.0 or Atom -> (feed_title, [{title, link, categories}])."""
    root = ET.fromstring(xml_bytes)
    feed_title = ""
    entries: list[dict] = []
    kind = _strip_ns(root.tag)
    if kind == "rss" or kind == "rdf":
        channel = next((c for c in root if _strip_ns(c.tag) == "channel"), root)
        for child in channel:
            name = _strip_ns(child.tag)
            if name == "title" and not feed_title:
                feed_title = _text(child)
            elif name == "item":
                entries.append(_entry(child))
        if kind == "rdf":                # RSS 1.0 puts items beside channel
            for child in root:
                if _strip_ns(child.tag) == "item":
                    entries.append(_entry(child))
    elif kind == "feed":
        for child in root:
            name = _strip_ns(child.tag)
            if name == "title" and not feed_title:
                feed_title = _text(child)
            elif name == "entry":
                entries.append(_entry(child))
    else:
        raise ValueError(f"not a feed: <{kind}>")
    return feed_title, entries


def _entry(node) -> dict:
    title = link = ""
    cats: list[str] = []
    for child in node:
        name = _strip_ns(child.tag)
        if name == "title":
            title = _text(child)
        elif name == "link":
            href = child.get("href")
            if href and child.get("rel", "alternate") == "alternate":
                link = href
            elif not href and not link:
                link = _text(child)
        elif name == "category":
            term = child.get("term") or _text(child)
            if term:
                cats.append(term.strip().lower())
    return {"title": title, "link": link, "categories": cats}


def tidy_source(feed_title: str, url: str = "") -> str:
    name = feed_title.split(" - ")[0].split(" | ")[0].strip()
    name = re.sub(r"\s*[:-]\s*all (content|stories)$", "", name, flags=re.I)
    if name:
        return name
    host = re.sub(r"^https?://(www\.|feeds\.)?", "", url).split("/")[0]
    return host or "the feed"


def feed_first_item(xml_bytes: bytes, url: str = "", scan: int = FEED_SCAN,
                    tech_only: bool = True) -> Optional[dict]:
    """The first entry that is not tagged non-tech (falls back to the very
    first entry when every scanned entry is tagged)."""
    feed_title, entries = parse_feed(xml_bytes)
    source = tidy_source(feed_title, url)
    entries = [e for e in entries if e["title"]]
    if not entries:
        return None
    pick = entries[0]
    if tech_only:
        for e in entries[:scan]:
            if not (set(e["categories"]) & NON_TECH_CATEGORIES):
                pick = e
                break
    return {"title": pick["title"], "source": source, "url": pick["link"]}


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", title.lower()).strip()


def dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        key = _norm_title(it.get("title", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def interleave(*groups: list[dict]) -> list[dict]:
    """Round-robin merge so every source gets a voice before any repeats."""
    out = []
    for i in range(max((len(g) for g in groups), default=0)):
        for g in groups:
            if i < len(g):
                out.append(g[i])
    return out


# ---------------------------------------------------------------- news
def _read_cache(path: Optional[Path], now: float) -> Optional[list[dict]]:
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text())
        if now - float(data.get("fetched_at", 0)) < NEWS_CACHE_S:
            items = data.get("items")
            if isinstance(items, list) and items:
                return items
    except (OSError, ValueError, TypeError):
        pass
    return None


def _write_cache(path: Optional[Path], now: float, items: list[dict]) -> None:
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fetched_at": now, "items": items}))
        tmp.replace(p)
    except OSError:
        log.warning("news cache write failed: %s", path)


def fetch_news(cfg, fetch: Fetch, cache_path=None, now: Optional[float] = None,
               pool: Optional[ThreadPoolExecutor] = None) -> tuple[list[dict], bool]:
    """-> (items [{title, source}], complete). ``complete`` is False when a
    source failed (the partial result is returned but not cached)."""
    now = time.time() if now is None else now
    cached = _read_cache(cache_path, now)
    if cached is not None:
        return [{"title": i["title"], "source": i["source"]} for i in cached], True
    feeds = _cfg_get(cfg, "briefing.news_feeds", None)
    if feeds is None:
        feeds = DEFAULT_NEWS_FEEDS
    feeds = [str(u) for u in feeds if str(u).strip()]
    hn_n = int(_cfg_get(cfg, "briefing.hn_items", NEWS_ITEMS) or NEWS_ITEMS)
    total = int(_cfg_get(cfg, "briefing.items", NEWS_ITEMS) or NEWS_ITEMS)
    complete = True

    def feed_item(url):
        raw = _safe_fetch(fetch, url)
        if raw is None:
            return None
        try:
            return feed_first_item(raw, url)
        except (ET.ParseError, ValueError) as exc:
            log.warning("feed parse failed %s: %s", url[:80], type(exc).__name__)
            return None

    if pool is not None:
        hn_future = pool.submit(hn_top, fetch, hn_n, HN_SCAN, pool)
        feed_futures = [pool.submit(feed_item, u) for u in feeds]
        hn = hn_future.result()
        feed_items = [f.result() for f in feed_futures]
    else:
        hn = hn_top(fetch, hn_n, HN_SCAN)
        feed_items = [feed_item(u) for u in feeds]
    if hn is None:
        complete = False
        hn = []
    groups = [hn]
    for it in feed_items:
        if it is None:
            complete = False
        else:
            groups.append([it])
    merged = dedupe(interleave(*groups))[:total]
    items = [{"title": i["title"], "source": i["source"]} for i in merged]
    if complete and items:
        _write_cache(cache_path, now, items)
    return items, complete


# --------------------------------------------------------- sports/stocks
def fetch_sports(feeds: list[str], fetch: Fetch,
                 pool: Optional[ThreadPoolExecutor] = None) -> list[str]:
    def one(url):
        raw = _safe_fetch(fetch, url)
        if raw is None:
            return None
        try:
            it = feed_first_item(raw, url, tech_only=False)
        except (ET.ParseError, ValueError):
            return None
        return f"{it['title']} — {it['source']}" if it else None

    results = list(pool.map(one, feeds)) if pool is not None else [one(u) for u in feeds]
    return [r for r in results if r]


def parse_stooq_csv(raw: bytes) -> Optional[dict]:
    """Symbol,Date,Time,Open,High,Low,Close,Volume -> {symbol, close, open}."""
    try:
        text = raw.decode("utf-8", errors="replace")
    except AttributeError:
        text = str(raw)
    if "<" in text[:20]:                    # HTML: blocked / not found
        return None
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        return None
    head = [h.strip().lower() for h in rows[0]]
    row = rows[1]
    try:
        rec = dict(zip(head, row))
        close = float(rec.get("close", "nan"))
        open_ = float(rec.get("open", "nan"))
    except (ValueError, TypeError):
        return None
    if close != close:                      # NaN: N/D from stooq
        return None
    sym = (rec.get("symbol") or "").split(".")[0].upper()
    return {"symbol": sym, "close": close, "open": open_ if open_ == open_ else None,
            "prev": None}


def parse_yahoo_json(raw: bytes) -> Optional[dict]:
    try:
        meta = json.loads(raw)["chart"]["result"][0]["meta"]
        return {"symbol": str(meta.get("symbol", "")).upper(),
                "close": float(meta["regularMarketPrice"]),
                "open": None,
                "prev": float(meta["chartPreviousClose"])
                if meta.get("chartPreviousClose") is not None else None}
    except (ValueError, TypeError, KeyError, IndexError):
        return None


def stock_line(q: dict) -> str:
    price = f"{q['close']:.2f}".rstrip("0").rstrip(".")
    base = q.get("prev") if q.get("prev") else q.get("open")
    if base:
        pct = (q["close"] - base) / base * 100
        if abs(pct) < 0.05:
            move = "flat"
        else:
            move = f"{'up' if pct > 0 else 'down'} {abs(pct):.1f}%"
        return f"{q['symbol']} {price}, {move}"
    return f"{q['symbol']} {price}"


def fetch_stocks(symbols: list[str], fetch: Fetch,
                 pool: Optional[ThreadPoolExecutor] = None) -> list[str]:
    """stooq CSV per symbol; Yahoo chart JSON when stooq answers with
    HTML (it sits behind a browser check from some networks)."""
    def one(sym):
        sym = str(sym).strip().upper()
        if not re.fullmatch(r"[A-Z0-9.\-]{1,10}", sym):
            return None
        q = None
        raw = _safe_fetch(fetch, STOOQ_URL.format(sym=sym.lower()))
        if raw is not None:
            q = parse_stooq_csv(raw)
        if q is None:
            raw = _safe_fetch(fetch, YAHOO_URL.format(sym=sym))
            if raw is not None:
                q = parse_yahoo_json(raw)
        if q is None:
            return None
        q["symbol"] = q.get("symbol") or sym
        return stock_line(q)

    results = list(pool.map(one, symbols)) if pool is not None else [one(s) for s in symbols]
    return [r for r in results if r]


# ------------------------------------------------------------ assemble
def _registry_result(registry, name: str, args: dict) -> tuple[bool, str]:
    """(ok, raw text) from one tool call; never raises."""
    if registry is None or not hasattr(registry, "call"):
        return False, "not available"
    try:
        has = getattr(registry, "has", None)
        if callable(has) and not has(name):
            return False, "not available"
        res = registry.call(name, args)
    except Exception as exc:                  # noqa: BLE001 - tool boundary
        log.warning("briefing: %s failed: %s", name, type(exc).__name__)
        return False, "failed"
    return bool(getattr(res, "ok", True)), str(getattr(res, "text", "") or "")


def _registry_text(registry, name: str, args: dict) -> tuple[bool, str]:
    ok, text = _registry_result(registry, name, args)
    return ok, " ".join(text.split())


DUE_DAYS = 2            # at 7 am, "today" plus tomorrow morning's items
_DUE_ITEM_RX = re.compile(r"^\d+\)\s*")


def _due_lines(registry) -> list[str]:
    """Canvas items due within DUE_DAYS, one per line, from the canvas_due
    fact sheet ('Due in the next 2 days (2):' then '1) COURSE - title,
    when'). Empty when the token is unset (canvas_due answers ok=False with
    its setup line -- the briefing must not nag about a missing token every
    morning), on any failure, and when nothing is due."""
    ok, text = _registry_result(registry, "canvas_due", {"days": DUE_DAYS})
    if not ok:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2 or not lines[0].lower().startswith("due"):
        return []                         # "nothing due in the next 2 days"
    return [_DUE_ITEM_RX.sub("", ln) for ln in lines[1:]]


def _exam_call(exam_lookup) -> Optional[dict]:
    """The next exam dict, or None. Split out of _exam_line because the
    study section needs the COURSE off the same lookup and must not pay
    for a second Canvas round trip to get it."""
    if not callable(exam_lookup):
        return None
    try:
        exam = exam_lookup()
    except Exception as exc:                  # noqa: BLE001 - source boundary
        log.warning("briefing: exam lookup failed: %s", type(exc).__name__)
        return None
    return exam or None


def _exam_words(exam: Optional[dict], now: datetime) -> str:
    if not exam:
        return ""
    from jarvis.tools.canvas import exam_words
    try:
        return exam_words(exam, now)
    except (KeyError, TypeError, AttributeError):
        return ""


def _exam_line(exam_lookup, now: datetime) -> str:
    """'Midterm 1 for BIOSENSORS, in 6 days, Friday at 9:00 am' or ''."""
    return _exam_words(_exam_call(exam_lookup), now)


def _exam_days(exam: Optional[dict], now: datetime) -> Optional[int]:
    """Whole days from now to the exam, or None when it has no usable
    date. Day granularity on purpose: an exam at 9 am on Friday and one at
    5 pm on Friday are the same nights of revision."""
    when = (exam or {}).get("when")
    if not isinstance(when, datetime):
        return None
    # The same conversion the spoken line uses. now.tzinfo is build_briefing's
    # datetime.now().astimezone() -- a fixed offset, not a zone -- so applying
    # it to an exam past the next DST change counted 6 days to an exam 5 days
    # out (now 2026-10-27 07:05 CDT, exam 2026-11-01 23:59 CST): one day over
    # briefing.study_days, and the deck line AND the offer went silent on the
    # morning they should first be spoken.
    from jarvis.tools.canvas import in_local
    try:
        return (in_local(when, now).date() - now.date()).days
    except (ValueError, OverflowError, TypeError):
        return None


def _study_section(cfg, exam: Optional[dict], store, now: datetime) -> tuple:
    """('14 cards due on your BIOSENSORS deck, 9 of them in box one. Shall
    we run ten now, sir?', offer) when an exam is close, else ('', {}).

    Silent unless there IS an exam inside study_days carrying a course
    name: a deck line with no exam behind it is nagging, and with no
    course there is nothing to filter the deck by -- every card of every
    subject would be counted as revision for this one.
    """
    if store is None or not exam:
        return "", {}
    days = _exam_days(exam, now)
    near = _int_cfg(cfg, "briefing.study_days", STUDY_DAYS)
    if days is None or days < 0 or days > near:
        return "", {}
    course = " ".join(str(exam.get("course") or "").split())
    if not course:
        return "", {}
    try:
        cards = store.due(limit=STUDY_DUE_CAP, now=now.timestamp(), topic=course)
    except Exception:                          # noqa: BLE001 - store boundary
        log.exception("briefing: flashcard deck unreadable")
        return "", {}
    if not cards:
        # An empty deck on the eve of an exam earns one line: it is the
        # moment he would want cards and has none. (Building them from his
        # lecture notes overnight is the follow-up, not this.)
        return STUDY_NO_DECK_LINE.format(course=course), {}
    weak = sum(1 for c in cards if int(c.get("box") or 0) == 1)
    n = len(cards)
    line = STUDY_CARDS_LINE.format(
        cards="One card" if n == 1 else f"{n} cards", course=course,
        weak=f", {weak} of them in box one" if weak else "")
    if not _truthy(_cfg_get(cfg, "briefing.study_offer", True)):
        return line, {}
    want = max(1, min(_int_cfg(cfg, "briefing.study_offer_n", STUDY_OFFER_N), n))
    offer = {"course": course, "n": want, "made_at": time.time()}
    return f"{line} {STUDY_OFFER_LINE.format(n=count_words(want))}", offer


def _day_line(now: datetime) -> str:
    clock = now.strftime("%I:%M %p").lstrip("0").lower()
    return f"Briefing for {now.strftime('%A')} {now.day} {now.strftime('%B')}, {clock}."


def build_briefing(cfg, registry, fetch: Optional[Fetch] = None, now=None,
                   cache_path=None, exam_lookup: Optional[Callable] = None,
                   flashcards=None,
                   park_offer: Optional[Callable] = None) -> tuple[dict, str]:
    """-> (sections, fact_sheet). sections = {weather: str, calendar: [str],
    due: [str], exam: str, study: str, news: [{title, source}], sports:
    [str], stocks: [str]}. ``fetch`` defaults to the module's ``_fetch``
    looked up at call time (tests monkeypatch it). ``exam_lookup`` () ->
    the next exam dict (tools/canvas.find_next_exam) or None; the countdown
    line and the study section both come off it. ``flashcards`` is a
    quiz.FlashcardStore; without one there is no study section.
    ``park_offer(offer)`` receives the "shall we run ten now" offer for
    Commander._try_study_offer to answer -- a callback rather than a third
    return value so every existing caller keeps working."""
    fetch = fetch or _fetch
    if now is None:
        now_dt = datetime.now().astimezone()
    elif isinstance(now, datetime):
        now_dt = now
    else:
        now_dt = datetime.fromtimestamp(float(now)).astimezone()
    now_ts = now_dt.timestamp()
    if cache_path is None:
        cache_path = Path.home() / ".cache" / "jarvis" / "news_cache.json"

    sections = {"weather": "", "calendar": [], "due": [], "exam": "",
                "study": "", "news": [], "sports": [], "stocks": []}
    notes = {}
    # A section switched off by voice is neither fetched nor mentioned:
    # "no news in the morning" must not leave a "News: unavailable" line
    # for the model to apologise about.
    on = {name: section_on(cfg, name) for name in ("weather", "calendar", "news",
                                                    "sports", "stocks", "canvas",
                                                    "study")}

    sports_feeds = [str(u) for u in (_cfg_get(cfg, "briefing.sports_feeds", []) or [])
                    if str(u).strip()] if on["sports"] else []
    symbols = [str(s) for s in (_cfg_get(cfg, "briefing.stock_symbols", []) or [])
               if str(s).strip()] if on["stocks"] else []

    with ThreadPoolExecutor(max_workers=POOL_WORKERS) as pool:
        news_future = pool.submit(fetch_news, cfg, fetch, cache_path, now_ts, pool) \
            if on["news"] else None
        sports_future = pool.submit(fetch_sports, sports_feeds, fetch, pool) \
            if sports_feeds else None
        stocks_future = pool.submit(fetch_stocks, symbols, fetch, pool) \
            if symbols else None
        # Canvas can spend its whole 6 s budget; on the pool, not inline,
        # so a slow LMS overlaps the feeds instead of adding to the wait.
        # The canvas switch (briefing.sections.canvas) covers both Canvas-fed
        # lines: off means no call at all, not an empty result.
        canvas_on = on.get("canvas", True)
        due_future = pool.submit(_due_lines, registry) if canvas_on else None
        # The DICT, not the line: the study section needs the course off
        # this same lookup (_exam_words turns it into the spoken line).
        exam_future = pool.submit(_exam_call, exam_lookup) if canvas_on else None

        if on["weather"]:
            ok, text = _registry_text(registry, "get_weather", {"when": "today"})
            if ok and text:
                sections["weather"] = text
            else:
                notes["weather"] = text or "unavailable"
        if on["calendar"]:
            ok, text = _registry_text(registry, "get_calendar", {"range": "today"})
            if ok and text:
                sections["calendar"] = [ln.strip() for ln in text.splitlines() if ln.strip()]
            else:
                notes["calendar"] = text or "unavailable"
        if canvas_on:
            # Silent when Canvas is unconfigured or down: no note, no line.
            try:
                sections["due"] = due_future.result()
            except Exception as exc:              # noqa: BLE001 - source boundary
                log.warning("briefing: due failed: %s", type(exc).__name__)
            exam = None
            try:
                exam = exam_future.result()
            except Exception as exc:              # noqa: BLE001
                log.warning("briefing: exam failed: %s", type(exc).__name__)
            sections["exam"] = _exam_words(exam, now_dt)
            if on.get("study", True):
                line, offer = _study_section(cfg, exam, flashcards, now_dt)
                sections["study"] = line
                if offer and callable(park_offer):
                    try:
                        park_offer(offer)
                    except Exception:             # noqa: BLE001 - caller boundary
                        log.debug("could not park the study offer", exc_info=True)

        news, complete = [], True
        if news_future is not None:
            try:
                news, complete = news_future.result()
            except Exception as exc:          # noqa: BLE001 - source boundary
                log.warning("briefing: news failed: %s", type(exc).__name__)
                news, complete = [], False
        sections["news"] = news
        if on["news"]:
            if not news:
                notes["news"] = "unavailable"
            elif not complete:
                notes["news"] = "partial"
        if sports_future is not None:
            try:
                sections["sports"] = sports_future.result()
            except Exception as exc:          # noqa: BLE001
                log.warning("briefing: sports failed: %s", type(exc).__name__)
            if not sections["sports"]:
                notes["sports"] = "unavailable"
        if stocks_future is not None:
            try:
                sections["stocks"] = stocks_future.result()
            except Exception as exc:          # noqa: BLE001
                log.warning("briefing: stocks failed: %s", type(exc).__name__)
            if not sections["stocks"]:
                notes["stocks"] = "unavailable"

    lines = [_day_line(now_dt)]
    if on["weather"]:
        lines.append(f"Weather: {sections['weather'] or notes.get('weather', 'unavailable')}")
    if on["calendar"]:
        if sections["calendar"]:
            lines.append("Calendar: " + " ".join(sections["calendar"]))
        else:
            lines.append(f"Calendar: {notes.get('calendar', 'nothing')}")
    if on.get("canvas", True):
        if sections["due"]:
            lines.append("Due: " + " ".join(f"{i}) {d}" for i, d in enumerate(sections["due"], 1)))
        if sections["exam"]:
            lines.append(f"Exam: {sections['exam']}")
        if sections["study"]:
            lines.append(f"Study: {sections['study']}")
    if on["news"]:
        if sections["news"]:
            items = " ".join(f"{i}) {n['title']} ({n['source']})"
                             for i, n in enumerate(sections["news"], 1))
            lines.append(f"News: {items}")
        else:
            lines.append("News: unavailable")
    if sports_feeds:
        lines.append("Sports: " + ("; ".join(sections["sports"]) or "unavailable"))
    if symbols:
        lines.append("Stocks: " + ("; ".join(sections["stocks"]) or "unavailable"))
    return sections, "\n".join(lines)


def news_follow_up(cfg) -> bool:
    """``briefing.news_follow_up``: after a briefing that summarised the
    news, park the stories so "let's hear it" reads them. On by default."""
    return _truthy(_cfg_get(cfg, "briefing.news_follow_up", True))


_ORDINALS = ("One", "Two", "Three", "Four", "Five", "Six")


def news_line(items) -> str:
    """The stories, composed here and not by the model: "Here they are,
    sir. One: <title>, from <source>. Two: ..." -- one sir, a full stop
    between stories so the voice breathes (list-composition rule)."""
    parts = []
    for i, n in enumerate(items[:len(_ORDINALS)]):
        title = str((n or {}).get("title") or "").strip().rstrip(".")
        source = str((n or {}).get("source") or "").strip()
        if not title:
            continue
        tail = f", from {source}" if source else ""
        parts.append(f"{_ORDINALS[len(parts)]}: {title}{tail}.")
    if not parts:
        return "There is nothing to read, sir."
    return "Here they are, sir. " + " ".join(parts)


def briefing_enabled(cfg) -> bool:
    return _truthy(_cfg_get(cfg, "briefing.enabled", False))


# ------------------------------------------------- shared by the views
_CLOCK_RX = re.compile(r"\b(\d{1,2}):(\d{2})\s*(am|pm)\b", re.I)
_CANVAS_ITEM_RX = re.compile(
    r"^(?P<body>.+?),\s+(?P<when>(?:today|tomorrow|yesterday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
    r"(?:\s+(?P<dom>\d{1,2})\s+(?P<mon>[A-Za-z]{3}))?\s+\d{1,2}:\d{2}\s+[ap]m)$", re.I)
_WEEKDAY_ABBR = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _now_dt(now) -> datetime:
    if now is None:
        return datetime.now().astimezone()
    if isinstance(now, datetime):
        return now
    return datetime.fromtimestamp(float(now)).astimezone()


def _clock(d: datetime) -> str:
    h = d.hour % 12 or 12
    return f"{h}:{d.minute:02d} {'am' if d.hour < 12 else 'pm'}"


def _svc(services, name: str):
    return getattr(services, name, None) if services is not None else None


def _day_word(day: date, today: date) -> str:
    if day == today:
        return "today"
    if day == today + timedelta(days=1):
        return "tomorrow"
    return day.strftime("%A")


def _first_clock(text: str, day: date, tz) -> Optional[datetime]:
    """The earliest 'H:MM am/pm' in a calendar line, on ``day``. The line is
    the calendar tool's own wording ("Tomorrow: 8:00 am Biosensors for an
    hour at ZACH 350, ..."), so its clocks are the events' starts."""
    best = None
    for m in _CLOCK_RX.finditer(text or ""):
        hour, minute = int(m.group(1)) % 12, int(m.group(2))
        if m.group(3).lower() == "pm":
            hour += 12
        try:
            dt = datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz)
        except ValueError:
            continue
        if best is None or dt < best:
            best = dt
    return best


def _calendar_events(services) -> Optional[list]:
    """Structured events from the parked CalendarSource, or None when
    there is no usable source (then the tool's text is all we have)."""
    cal = _svc(services, "calendar")
    if cal is None:
        return None
    try:
        conf = getattr(cal, "configured", True)
        if callable(conf):
            conf = conf()
        if not conf:
            return None
        return list(cal.events())
    except Exception:
        log.debug("briefing: calendar events unavailable", exc_info=True)
        return None


def _canvas_source(cfg, services) -> bool:
    """True when there is ANY Canvas coursework to read: a REST token, or a
    Canvas calendar feed among the subscriptions.

    The preview and the week forecast used to gate on ``canvas.token``
    alone, so on a box whose university blocks personal tokens (his does)
    they stayed dark while the coursework sat in a calendar he had already
    subscribed to. The gate is about a SOURCE, not about the token."""
    if _cfg_get(cfg, "canvas.token", ""):
        return True
    from jarvis.tools import canvas_ical
    return canvas_ical.has_coursework(_calendar_events(services) or [])


def _event_line(ev) -> str:
    title = (getattr(ev, "title", "") or "").strip() or "an event"
    if getattr(ev, "all_day", False):
        text = f"all day: {title}"
    else:
        text = f"{_clock(ev.start)} {title}"
    loc = (getattr(ev, "location", "") or "").strip()
    return f"{text} at {loc}" if loc else text


def _canvas_items(text: str, today: date) -> list[tuple[date, str]]:
    """The canvas_due sheet ("Due this week (2): 1) CSCE - Lab 3, tomorrow
    11:59 pm 2) ...", whitespace already collapsed) -> [(day, 'CSCE - Lab
    3, tomorrow 11:59 pm')]. Lines that carry no date ("and 2 more") are
    dropped."""
    out = []
    for part in re.split(r"\s(?=\d+\)\s)", text or ""):
        m = re.match(r"^\d+\)\s+(?P<rest>.+)$", part.strip())
        if not m:
            continue
        # the sheet's "and 2 more" tail rides on the last item once the
        # registry has collapsed the newlines
        item = re.sub(r"\s+and \d+ more$", "", m.group("rest").strip())
        w = _CANVAS_ITEM_RX.match(item)
        if not w:
            continue
        when = w.group("when").lower()
        day = None
        if when.startswith("today"):
            day = today
        elif when.startswith("tomorrow"):
            day = today + timedelta(days=1)
        elif when.startswith("yesterday"):
            day = today - timedelta(days=1)
        elif w.group("dom"):
            try:
                parsed = datetime.strptime(
                    f"{w.group('dom')} {w.group('mon')} {today.year}", "%d %b %Y").date()
                # "Tue 5 Jan" asked in late December is next year's
                if parsed < today - timedelta(days=7):
                    parsed = parsed.replace(year=today.year + 1)
                day = parsed
            except ValueError:
                day = None
        else:
            abbr = when[:3]
            if abbr in _WEEKDAY_ABBR:
                want = _WEEKDAY_ABBR.index(abbr)
                day = today + timedelta(days=(want - today.weekday()) % 7)
        if day is not None:
            out.append((day, item))
    return out


def _open_todos(services, limit: int = 5) -> tuple[list[str], int]:
    notes = _svc(services, "notes")
    if notes is None or not hasattr(notes, "list"):
        return [], 0
    try:
        items = notes.list("todo", limit=limit)
        texts = [str(i.get("text") if isinstance(i, dict) else i).strip() for i in items]
        texts = [t for t in texts if t]
        count = getattr(notes, "count", None)
        total = int(count("todo")) if callable(count) else len(texts)
    except Exception:
        log.debug("briefing: todos unavailable", exc_info=True)
        return [], 0
    return texts, max(total, len(texts))


def _tk_items(services, kind: str) -> list:
    tk = _svc(services, "timekeeper")
    if tk is None or not hasattr(tk, "list"):
        return []
    try:
        return list(tk.list(kind))
    except Exception:
        log.debug("briefing: %s list unavailable", kind, exc_info=True)
        return []


def _item_due(item, tz) -> Optional[datetime]:
    due = getattr(item, "effective_due", None)
    if due is None:
        due = getattr(item, "due", None)
    try:
        return datetime.fromtimestamp(float(due), tz)
    except (TypeError, ValueError, OSError):
        return None


# -------------------------------------------------------------- preview
def wake_offer(cfg, first_start: Optional[datetime], alarms: list, now: datetime) -> Optional[dict]:
    """An alarm to offer for tomorrow's first event, or None.

    Only when the event starts at or before ``briefing.early_before``
    (09:00) and no alarm already rings tomorrow before it. The offered
    time is ``briefing.wake_lead_min`` (60) before the event, on the
    quarter hour, and must still be in the future."""
    if first_start is None or not _truthy(_cfg_get(cfg, "briefing.wake_offer", True)):
        return None
    if not section_on(cfg, "alarms"):
        return None
    try:
        hh, mm = (int(x) for x in str(_cfg_get(cfg, "briefing.early_before", "09:00")
                                        or "09:00").split(":")[:2])
        lead = int(_cfg_get(cfg, "briefing.wake_lead_min", 60) or 60)
    except (TypeError, ValueError):
        hh, mm, lead = 9, 0, 60
    if (first_start.hour, first_start.minute) > (hh, mm):
        return None
    day = first_start.date()
    for it in alarms:
        due = _item_due(it, first_start.tzinfo)
        if due is not None and due.date() == day and due <= first_start:
            return None                           # an alarm already covers it
    wake = first_start - timedelta(minutes=max(0, lead))
    wake = wake.replace(minute=wake.minute - wake.minute % 15, second=0, microsecond=0)
    if wake <= now:
        return None
    time_words = _clock(wake)
    return {"due": wake.timestamp(), "time": time_words, "label": WAKE_LABEL,
            "line": OFFER_LINE.format(time=time_words), "made_at": now.timestamp()}


def build_preview(cfg, registry, services=None, now=None) -> tuple[dict, str, Optional[dict]]:
    """Tomorrow in one breath -> (sections, fact_sheet, offer).

    sections = {weather: str, calendar: [str], canvas: [str], todos: [str],
    alarm: str, offer?: str}. ``offer`` is the wake-up alarm to propose
    (see wake_offer) or None. Weather, calendar and Canvas come through the
    registry the way the morning briefing's do; to-dos and alarms are read
    from the services directly (they are local stores, not tools)."""
    now_dt = _now_dt(now)
    today = now_dt.date()
    tomorrow = today + timedelta(days=1)
    sections: dict = {"weather": "", "calendar": [], "canvas": [], "todos": [], "alarm": ""}
    notes: dict = {}
    first_start = None

    if section_on(cfg, "weather"):
        ok, text = _registry_text(registry, "get_weather", {"when": "tomorrow"})
        if ok and text:
            sections["weather"] = text
        else:
            notes["weather"] = text or "unavailable"
    if section_on(cfg, "calendar"):
        events = _calendar_events(services)
        if events is not None:
            todays = [e for e in events if getattr(e, "on", None) and e.on(tomorrow)]
            todays.sort(key=lambda e: (not getattr(e, "all_day", False), e.start))
            sections["calendar"] = [_event_line(e) for e in todays] or \
                ["Nothing on tomorrow, sir."]
            timed = [e for e in todays if not getattr(e, "all_day", False)]
            first_start = min((e.start for e in timed), default=None)
        else:
            ok, text = _registry_text(registry, "get_calendar", {"range": "tomorrow"})
            if ok and text:
                sections["calendar"] = [ln.strip() for ln in text.splitlines() if ln.strip()]
                first_start = _first_clock(text, tomorrow, now_dt.tzinfo)
            else:
                notes["calendar"] = text or "unavailable"
    if section_on(cfg, "canvas") and _canvas_source(cfg, services):
        ok, text = _registry_text(registry, "canvas_due", {"days": 1})
        if ok and text:
            sections["canvas"] = [item for _day, item in _canvas_items(text, today)]
            if not sections["canvas"] and not text.lower().startswith("nothing due"):
                notes["canvas"] = "nothing parsed"
        else:
            notes["canvas"] = text or "unavailable"
    todos_total = 0
    if section_on(cfg, "todos"):
        sections["todos"], todos_total = _open_todos(services)
    alarms = _tk_items(services, "alarm") if section_on(cfg, "alarms") else []
    tomorrows = []
    for it in alarms:
        due = _item_due(it, now_dt.tzinfo)
        if due is not None and due.date() == tomorrow:
            label = (getattr(it, "label", "") or "").strip()
            tomorrows.append(f"{_clock(due)}" + (f" ({label})" if label else ""))
    if section_on(cfg, "alarms"):
        sections["alarm"] = "; ".join(tomorrows) or "none set"
    offer = wake_offer(cfg, first_start, alarms, now_dt) if section_on(cfg, "alarms") else None
    if offer:
        sections["offer"] = offer["line"]

    lines = [f"Preview for tomorrow, {tomorrow.strftime('%A')} {tomorrow.day} "
             f"{tomorrow.strftime('%B')}."]
    if section_on(cfg, "weather"):
        lines.append(f"Weather tomorrow: {sections['weather'] or notes.get('weather', 'unavailable')}")
    if section_on(cfg, "calendar"):
        if sections["calendar"]:
            lines.append("First up: " + " ".join(sections["calendar"]))
        else:
            lines.append(f"Calendar: {notes.get('calendar', 'nothing')}")
    if section_on(cfg, "canvas") and _canvas_source(cfg, services):
        if sections["canvas"]:
            lines.append("Canvas due tomorrow: " + "; ".join(sections["canvas"]))
        else:
            lines.append(f"Canvas: {notes.get('canvas', 'nothing due tomorrow')}")
    if section_on(cfg, "todos"):
        if sections["todos"]:
            head = f"{todos_total} open" if todos_total > len(sections["todos"]) else "open"
            lines.append(f"To-dos ({head}): " + "; ".join(sections["todos"]))
        else:
            lines.append("To-dos: none open")
    if section_on(cfg, "alarms"):
        lines.append(f"Alarm tomorrow: {sections['alarm']}")
    return sections, "\n".join(lines), offer


# ----------------------------------------------------------------- week
def build_week(cfg, registry, services=None, now=None, days: int = WEEK_DAYS) -> tuple[dict, str]:
    """The workload forecast -> (sections, fact_sheet). sections = {summary:
    str, days: [{label, date, items: [str]}], todos: [str]}. Calendar
    events (structured when the source is parked on services, else the
    calendar tool's week text), Canvas due within the week and reminders
    are bucketed day by day; the heaviest and the clear days are named so
    the model can say "Heavy Tuesday, sir"."""
    now_dt = _now_dt(now)
    today = now_dt.date()
    days = max(1, int(days or WEEK_DAYS))
    span = [today + timedelta(days=i) for i in range(days)]
    buckets: dict[date, list[str]] = {d: [] for d in span}
    notes: dict = {}

    if section_on(cfg, "calendar"):
        events = _calendar_events(services)
        if events is not None:
            for d in span:
                todays = [e for e in events if getattr(e, "on", None) and e.on(d)]
                todays.sort(key=lambda e: (not getattr(e, "all_day", False), e.start))
                buckets[d].extend(_event_line(e) for e in todays)
        else:
            ok, text = _registry_text(registry, "get_calendar", {"range": "week"})
            if ok and text:
                for d, item in _week_text_items(text, today):
                    if d in buckets:
                        buckets[d].append(item)
            else:
                notes["calendar"] = text or "unavailable"
    if section_on(cfg, "canvas") and _canvas_source(cfg, services):
        ok, text = _registry_text(registry, "canvas_due", {"days": days})
        if ok and text:
            for d, item in _canvas_items(text, today):
                if d in buckets:
                    buckets[d].append(f"due: {item}")
        else:
            notes["canvas"] = text or "unavailable"
    if section_on(cfg, "reminders"):
        for it in _tk_items(services, "reminder"):
            due = _item_due(it, now_dt.tzinfo)
            if due is None or due.date() not in buckets:
                continue
            label = (getattr(it, "label", "") or "").strip() or "a reminder"
            buckets[due.date()].append(f"{_clock(due)} reminder: {label}")
    todos, todos_total = _open_todos(services) if section_on(cfg, "todos") else ([], 0)

    day_rows = [{"label": _day_word(d, today).capitalize(), "date": d.isoformat(),
                 "items": list(buckets[d])} for d in span]
    counts = {d: len(buckets[d]) for d in span}
    clear = [_day_word(d, today) for d in span if counts[d] == 0]
    busiest = max(span, key=lambda d: counts[d])
    total = sum(counts.values())
    if total == 0:
        summary = "Nothing on the week at all."
    else:
        heavy = [d for d in span if counts[d] >= 3] or ([busiest] if counts[busiest] >= 2 else [])
        parts = []
        if heavy:
            parts.append("Heaviest: " + ", ".join(
                f"{_day_word(d, today)} ({counts[d]})" for d in heavy))
        else:
            parts.append("A light week: nothing over one item a day")
        if clear:
            parts.append("Clear: " + ", ".join(clear))
        summary = "; ".join(parts) + "."
    sections = {"summary": summary, "days": day_rows, "todos": todos,
                "unavailable": sorted(notes)}

    lines = [f"Week ahead from {today.strftime('%A')} {today.day} {today.strftime('%B')}: "
             f"{total} item{'s' if total != 1 else ''} over {days} days."]
    lines.append(summary)
    for note_name in ("calendar", "canvas"):
        if note_name in notes:
            lines.append(f"{note_name.capitalize()}: {notes[note_name]}")
    for row in day_rows:
        lines.append(f"{row['label']}: " + ("; ".join(row["items"]) or "clear"))
    if section_on(cfg, "todos"):
        if todos:
            head = f"{todos_total} open" if todos_total > len(todos) else "open"
            lines.append(f"To-dos ({head}): " + "; ".join(todos))
        else:
            lines.append("To-dos: none open")
    return sections, "\n".join(lines)


def _week_text_items(text: str, today: date) -> list[tuple[date, str]]:
    """The calendar tool's week line ("This week: today 8:00 am X, 2:30 pm
    Y; Wednesday all day: Z; nothing else.") -> [(day, item)]."""
    body = re.sub(r"^this week:\s*", "", text.strip(), flags=re.I)
    body = re.sub(r"\.?\s*that's as of .*$", "", body, flags=re.I)
    out = []
    for part in body.split(";"):
        part = part.strip().rstrip(".")
        if not part or part.lower() == "nothing else":
            continue
        m = re.match(r"^(today|tomorrow|monday|tuesday|wednesday|thursday|friday|"
                     r"saturday|sunday)\s+(.+)$", part, re.I)
        if not m:
            continue
        word = m.group(1).lower()
        if word == "today":
            day = today
        elif word == "tomorrow":
            day = today + timedelta(days=1)
        else:
            want = ("monday", "tuesday", "wednesday", "thursday", "friday",
                    "saturday", "sunday").index(word)
            day = today + timedelta(days=(want - today.weekday()) % 7)
        for item in m.group(2).split(", "):
            item = item.strip()
            if item:
                out.append((day, item))
    return out


# ---------------------------------------------------------------- tool
def make_tools(cfg, services) -> list[ToolSpec]:
    cache_path = getattr(services, "news_cache_path", None) if services is not None else None

    def _registry():
        reg = getattr(services, "tools", None) if services is not None else None
        return reg

    def _next_exam():
        # Canvas (when the token is set) merged with the calendar's cache;
        # silent None when neither has an exam or Canvas is unconfigured.
        from jarvis.tools.canvas import find_next_exam
        cal = getattr(services, "calendar", None) if services is not None else None
        exam, _checked = find_next_exam(cfg, cal)
        return exam

    def _park_study_offer(offer):
        # Answered by Commander._try_study_offer, the same way the wake-up
        # offer below is answered by _try_alarm_offer.
        if services is None:
            return
        try:
            services.study_offer = offer
        except Exception:
            log.debug("could not park the study offer", exc_info=True)

    def _retire_offer():
        if services is None:
            return
        try:
            services.briefing_offer = None
        except Exception:
            log.debug("could not retire the briefing offer", exc_info=True)

    def _park_news(items):
        # Answered by Commander._try_briefing_offer -- end-anchored yes/no,
        # the 60 s TTL, anything else a new subject with the offer dropped.
        if services is None:
            return

        def deliver() -> bool:
            speak = getattr(services, "speak", None)
            if not callable(speak):
                return False
            try:
                # An ANSWER to his yes, not a proactive line: quiet hours
                # must not hold it for the digest.
                speak(news_line(items), proactive=False)
            except Exception:                        # noqa: BLE001 - the rung says "held"
                log.exception("news follow-up could not be spoken")
                return False
            return True
        try:
            services.briefing_offer = {"made_at": time.time(), "deliver": deliver,
                                       "kind": "news"}
        except Exception:
            log.debug("could not park the news follow-up", exc_info=True)

    def _park_offer(offer):
        # The commander answers the "shall I wake you" yes/no from here
        # (Commander._try_alarm_offer), the way a calendar add waits on
        # calendar.pending_event. Parked on services, not on the tool: the
        # tool never sees the reply.
        if services is None:
            return
        try:
            services.alarm_offer = offer
        except Exception:
            log.debug("could not park the alarm offer", exc_info=True)

    def get_briefing(when="today", **_) -> ToolResult:
        # No `briefing.enabled` check here. That flag means "let a plain
        # 'good morning' trigger a briefing", and commander._h_briefing
        # already enforces exactly that -- it lets an EXPLICIT request past
        # regardless. The tool cannot tell the two apart, so refusing here
        # only ever broke the explicit ask: the user said "give me the
        # briefing" and was told the briefing is switched off.
        when = coerce_when(when)
        registry = _registry()
        if when == "tomorrow":
            sections, sheet, offer = build_preview(cfg, registry, services)
            _park_offer(offer)
            # "none set" is the timekeeper's honest answer, not a source
            # that came through: with everything else dark it is no preview
            got_any = any(sections.get(k) for k in ("weather", "calendar", "canvas",
                                                    "todos")) or \
                sections.get("alarm") not in ("", "none set")
            if not got_any:
                return ToolResult(text="preview sources unreachable", ok=False,
                                  speak=PREVIEW_NOTHING_LINE)
            return ToolResult(text=sheet, card=sections,
                              max_sentences=verbosity_cap(cfg, VIEW_SENTENCES["tomorrow"]))
        if when == "week":
            sections, sheet = build_week(cfg, registry, services)
            got_any = any(row["items"] for row in sections["days"]) or \
                bool(sections["todos"]) or not sections["unavailable"]
            if not got_any:
                return ToolResult(text="week sources unreachable", ok=False,
                                  speak=WEEK_NOTHING_LINE)
            return ToolResult(text=sheet, card=sections,
                              max_sentences=verbosity_cap(cfg, VIEW_SENTENCES["week"]))
        sections, sheet = build_briefing(
            cfg, registry, cache_path=cache_path, exam_lookup=_next_exam,
            flashcards=getattr(services, "flashcards", None) if services else None,
            park_offer=_park_study_offer)
        got_any = bool(sections["weather"] or sections["calendar"] or
                       sections["due"] or sections["exam"] or sections["study"] or
                       sections["news"] or sections["sports"] or sections["stocks"])
        if not got_any:
            return ToolResult(text="briefing sources unreachable", ok=False,
                              speak=NOTHING_LINE)
        # A briefing that is being DELIVERED retires any parked offer of
        # one -- HERE, because this tool is the one door every delivery
        # goes through (the explicit rung, the model answering inside a
        # compound question, the first wake). 2026-09-04 15:06: the model
        # delivered it inline, the arrival offer from 14:44 then asked
        # "Shall I run your briefing, sir?", he said yes, and the calendar
        # and the lab were read a second time. Only the first-wake path
        # cleared the offer; this covers the other two doors.
        _retire_offer()
        # ... and if the news was summarised rather than read, the stories
        # go on the SAME offer protocol, so "let's hear it" / "yes" reads
        # them and anything else drops them without a word (15:07:02 that
        # day: "Let's hear it" was a generic yes with nothing parked, and
        # routed to the model, which answered about the inbox).
        if sections.get("news") and news_follow_up(cfg):
            _park_news(sections["news"])
        return ToolResult(text=sheet, card=sections,
                          max_sentences=verbosity_cap(cfg, VIEW_SENTENCES["today"]))

    spec = ToolSpec(
        name="get_briefing",
        # <= 20 words: the descriptions ride in every prompt.
        # "today's" is deliberately gone. It used to open this description
        # as "today's weather, calendar, ...", and gemma4 read a plain
        # "What's on today?" as a briefing request 3 of 5 live tries,
        # stealing it from get_calendar. This tool is the COMPOSED summary
        # of several sources; the day-by-day event list is get_calendar.
        description=("The composed summary of weather, calendar, coursework, "
                     "exam and news: morning briefing, tomorrow's preview, "
                     "or the week ahead."),
        parameters={"type": "object", "properties": {
            "when": {"type": "string", "enum": list(WHENS),
                     "description": "today (the morning briefing), tomorrow "
                                    "(the evening preview) or week (the forecast)"}}},
        handler=get_briefing,
    )
    return [spec]
