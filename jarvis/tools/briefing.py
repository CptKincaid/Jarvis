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
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from jarvis.logs import get_logger
from jarvis.tools.registry import ToolResult, ToolSpec

log = get_logger("tools.briefing")

BRIEFING_OFF_LINE = ("The morning briefing is switched off, sir; "
                     "the toggle is in settings under Briefing.")
NOTHING_LINE = "I couldn't reach any of the briefing sources, sir."
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
def _registry_text(registry, name: str, args: dict) -> tuple[bool, str]:
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
    text = " ".join(str(getattr(res, "text", "") or "").split())
    return bool(getattr(res, "ok", True)), text


def _day_line(now: datetime) -> str:
    clock = now.strftime("%I:%M %p").lstrip("0").lower()
    return f"Briefing for {now.strftime('%A')} {now.day} {now.strftime('%B')}, {clock}."


def build_briefing(cfg, registry, fetch: Optional[Fetch] = None, now=None,
                   cache_path=None) -> tuple[dict, str]:
    """-> (sections, fact_sheet). sections = {weather: str, calendar: [str],
    news: [{title, source}], sports: [str], stocks: [str]}. ``fetch``
    defaults to the module's ``_fetch`` looked up at call time (tests
    monkeypatch it)."""
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

    sections = {"weather": "", "calendar": [], "news": [], "sports": [], "stocks": []}
    notes = {}

    sports_feeds = [str(u) for u in (_cfg_get(cfg, "briefing.sports_feeds", []) or [])
                    if str(u).strip()]
    symbols = [str(s) for s in (_cfg_get(cfg, "briefing.stock_symbols", []) or [])
               if str(s).strip()]

    with ThreadPoolExecutor(max_workers=POOL_WORKERS) as pool:
        news_future = pool.submit(fetch_news, cfg, fetch, cache_path, now_ts, pool)
        sports_future = pool.submit(fetch_sports, sports_feeds, fetch, pool) \
            if sports_feeds else None
        stocks_future = pool.submit(fetch_stocks, symbols, fetch, pool) \
            if symbols else None

        ok, text = _registry_text(registry, "get_weather", {"when": "today"})
        if ok and text:
            sections["weather"] = text
        else:
            notes["weather"] = text or "unavailable"
        ok, text = _registry_text(registry, "get_calendar", {"range": "today"})
        if ok and text:
            sections["calendar"] = [ln.strip() for ln in text.splitlines() if ln.strip()]
        else:
            notes["calendar"] = text or "unavailable"

        try:
            news, complete = news_future.result()
        except Exception as exc:              # noqa: BLE001 - source boundary
            log.warning("briefing: news failed: %s", type(exc).__name__)
            news, complete = [], False
        sections["news"] = news
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
    lines.append(f"Weather: {sections['weather'] or notes.get('weather', 'unavailable')}")
    if sections["calendar"]:
        lines.append("Calendar: " + " ".join(sections["calendar"]))
    else:
        lines.append(f"Calendar: {notes.get('calendar', 'nothing')}")
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


def briefing_enabled(cfg) -> bool:
    return _truthy(_cfg_get(cfg, "briefing.enabled", False))


# ---------------------------------------------------------------- tool
def make_tools(cfg, services) -> list[ToolSpec]:
    cache_path = getattr(services, "news_cache_path", None) if services is not None else None

    def _registry():
        reg = getattr(services, "tools", None) if services is not None else None
        return reg

    def get_briefing(**_) -> ToolResult:
        if not briefing_enabled(cfg):
            return ToolResult(text=BRIEFING_OFF_LINE, ok=False, speak=BRIEFING_OFF_LINE)
        sections, sheet = build_briefing(cfg, _registry(), cache_path=cache_path)
        got_any = bool(sections["weather"] or sections["calendar"] or
                       sections["news"] or sections["sports"] or sections["stocks"])
        if not got_any:
            return ToolResult(text="briefing sources unreachable", ok=False,
                              speak=NOTHING_LINE)
        return ToolResult(text=sheet, max_sentences=6, card=sections)

    spec = ToolSpec(
        name="get_briefing",
        description="Morning briefing: today's weather, calendar and three tech news items.",
        parameters={"type": "object", "properties": {}},
        handler=get_briefing,
    )
    return [spec]
