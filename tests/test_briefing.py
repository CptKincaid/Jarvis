"""Briefing tool (spec 6.7): canned Hacker News / Atom / RSS / stooq CSV /
Yahoo JSON through a fake ``fetch`` -> exact sections and fact sheet; the
disabled path; partial feed failure degrading to what worked; the 15-min
news cache. Live feeds only with JARVIS_LIVE=1.

Firewall: tmp cache path, tmp JARVIS_LOG_DIR (conftest) and a tmp
JARVIS_ASSISTANT_CONFIG; no network in the unit tests.
"""
import json
import os
from datetime import datetime
from types import SimpleNamespace

import pytest

from jarvis.tools import briefing as br
from jarvis.tools.briefing import (BRIEFING_OFF_LINE, build_briefing, dedupe,
                                   feed_first_item, fetch_news, fetch_stocks,
                                   interleave, parse_feed, tidy_source)
from jarvis.tools.registry import ToolRegistry, ToolResult, ToolSpec


@pytest.fixture(autouse=True)
def _firewall(tmp_path, monkeypatch, request):
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")
    if "live" not in request.node.name:          # unit tests: no network, ever
        import urllib.request

        def _no_network(*a, **k):
            raise AssertionError("unit test tried to reach the network")
        monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    yield


NOW = datetime(2026, 8, 26, 7, 5).astimezone()
VERGE = "https://www.theverge.com/rss/index.xml"
ARS = "https://feeds.arstechnica.com/arstechnica/index"

HN_IDS = json.dumps([101, 102, 103, 104, 105]).encode()


def _hn_item(i, title, score, **extra):
    return json.dumps({"id": i, "type": "story", "title": title, "score": score,
                       "url": f"https://x/{i}", **extra}).encode()


ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title type="text">The Verge</title>
  <entry>
    <title type="html"><![CDATA[Nvidia is about to be a hundred-billion-dollar company]]></title>
    <link rel="alternate" type="text/html" href="https://www.theverge.com/tech/1"/>
    <category scheme="https://www.theverge.com" term="AI"/>
    <category scheme="https://www.theverge.com" term="Tech"/>
  </entry>
  <entry>
    <title>Second Verge story</title>
    <link rel="alternate" href="https://www.theverge.com/tech/2"/>
  </entry>
</feed>"""

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0">
  <channel>
    <title>Ars Technica - All content</title>
    <link>https://arstechnica.com</link>
    <item>
      <title>RIP, Tim Curry: Ars remembers his top 10 performances</title>
      <link>https://arstechnica.com/culture/1</link>
      <category><![CDATA[Culture]]></category>
      <category><![CDATA[Obituaries]]></category>
    </item>
    <item>
      <title>AWS acquires DuckLabs</title>
      <link>https://arstechnica.com/tech/2</link>
      <category><![CDATA[Tech]]></category>
      <category><![CDATA[Cloud]]></category>
    </item>
  </channel>
</rss>"""

STOOQ_CSV = b"Symbol,Date,Time,Open,High,Low,Close,Volume\r\nAAPL.US,2026-08-26,22:00:00,230.10,233.5,229.9,232.14,33571543\r\n"
STOOQ_HTML = b"<meta charset=utf-8><title>Stooq</title>The page you requested does not exist"
YAHOO_JSON = json.dumps({"chart": {"result": [{"meta": {
    "symbol": "NVDA", "regularMarketPrice": 191.5, "chartPreviousClose": 189.0}}]}}).encode()
SPORTS_RSS = b"""<rss version="2.0"><channel><title>BBC Sport - Football</title>
<item><title>City win the derby</title><link>https://bbc/1</link></item></channel></rss>"""


def canned():
    return {
        br.HN_TOP_URL: HN_IDS,
        br.HN_ITEM_URL.format(id=101): _hn_item(101, "Low score story", 10),
        br.HN_ITEM_URL.format(id=102): _hn_item(102, "GLM-5.3-Flash", 500),
        br.HN_ITEM_URL.format(id=103): _hn_item(103, "Tailcat – netcat over Tailscale", 300),
        br.HN_ITEM_URL.format(id=104): _hn_item(104, "Dead story", 900, dead=True),
        br.HN_ITEM_URL.format(id=105): _hn_item(105, "AWS Acquires DuckLabs", 700),
        VERGE: ATOM,
        ARS: RSS,
    }


class FakeFetch:
    def __init__(self, table=None):
        self.table = dict(canned() if table is None else table)
        self.calls = []

    def __call__(self, url, timeout=None, headers=None):
        self.calls.append(url)
        val = self.table.get(url)
        if val is None:
            raise OSError(f"no route to {url}")
        if isinstance(val, Exception):
            raise val
        return val


class FakeRegistry(ToolRegistry):
    def __init__(self, weather="72°F and partly cloudy, high 85, low 64, 10% chance of rain.",
                 calendar="Today: 10:00 am dentist for an hour, 2:30 pm standup; nothing else.",
                 weather_ok=True, calendar_ok=True):
        super().__init__()
        self.calls = []

        def w(**kw):
            self.calls.append(("get_weather", kw))
            return ToolResult(text=weather, ok=weather_ok)

        def c(**kw):
            self.calls.append(("get_calendar", kw))
            return ToolResult(text=calendar, ok=calendar_ok)
        self.register(ToolSpec("get_weather", "w", handler=w))
        self.register(ToolSpec("get_calendar", "c", handler=c))


class Cfg:
    def __init__(self, **briefing):
        self.data = {"briefing": {"enabled": False, "hn_items": 3,
                                  "news_feeds": [VERGE, ARS],
                                  "sports_feeds": [], "stock_symbols": [],
                                  **briefing}}

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


# ------------------------------------------------------------- parsing
def test_parse_atom_and_rss():
    title, entries = parse_feed(ATOM)
    assert title == "The Verge"
    assert entries[0]["title"] == "Nvidia is about to be a hundred-billion-dollar company"
    assert entries[0]["link"] == "https://www.theverge.com/tech/1"
    assert entries[0]["categories"] == ["ai", "tech"]
    title, entries = parse_feed(RSS)
    assert title == "Ars Technica - All content"
    assert [e["title"] for e in entries] == [
        "RIP, Tim Curry: Ars remembers his top 10 performances", "AWS acquires DuckLabs"]
    assert entries[0]["categories"] == ["culture", "obituaries"]
    with pytest.raises(ValueError):
        parse_feed(b"<html><body>nope</body></html>")


def test_feed_first_item_skips_non_tech_and_tidies_source():
    assert feed_first_item(ATOM, VERGE) == {
        "title": "Nvidia is about to be a hundred-billion-dollar company",
        "source": "The Verge", "url": "https://www.theverge.com/tech/1"}
    assert feed_first_item(RSS, ARS) == {
        "title": "AWS acquires DuckLabs", "source": "Ars Technica",
        "url": "https://arstechnica.com/tech/2"}
    assert feed_first_item(RSS, ARS, tech_only=False)["title"].startswith("RIP, Tim Curry")
    only_culture = RSS.replace(b"<![CDATA[Tech]]>", b"<![CDATA[Culture]]>")
    assert feed_first_item(only_culture, ARS)["title"].startswith("RIP, Tim Curry")
    assert feed_first_item(b"<rss><channel><title>x</title></channel></rss>") is None
    assert tidy_source("", "https://feeds.example.org/a/b") == "example.org"
    assert tidy_source("", "https://www.example.org/rss") == "example.org"
    assert tidy_source("Hacker News: All Stories") == "Hacker News"


def test_dedupe_and_interleave():
    items = [{"title": "AWS Acquires DuckLabs"}, {"title": "aws acquires ducklabs!"},
             {"title": ""}, {"title": "Other"}]
    assert [i["title"] for i in dedupe(items)] == ["AWS Acquires DuckLabs", "Other"]
    assert interleave([1, 2, 3], [4], [5, 6]) == [1, 4, 5, 2, 6, 3]
    assert interleave() == []


def test_hn_top_by_score_skips_dead():
    fetch = FakeFetch()
    top = br.hn_top(fetch, n=3)
    assert [t["title"] for t in top] == \
        ["AWS Acquires DuckLabs", "GLM-5.3-Flash", "Tailcat – netcat over Tailscale"]
    assert all(t["source"] == "Hacker News" for t in top)
    assert br.hn_top(FakeFetch({}), n=3) is None
    assert br.hn_top(FakeFetch({br.HN_TOP_URL: b"[]"}), n=3) == []
    assert br.hn_top(FakeFetch({br.HN_TOP_URL: b"garbage"}), n=3) is None


# ---------------------------------------------------------------- news
def test_fetch_news_mix_dedupe_cut_and_cache(tmp_path):
    fetch = FakeFetch()
    cache = tmp_path / "news.json"
    items, complete = fetch_news(Cfg(), fetch, cache, now=1000.0)
    assert complete
    # round-robin HN / Verge / Ars; the Ars pick duplicates HN's top story
    # (same title, different case) so it drops and HN #2 fills the third slot
    assert items == [
        {"title": "AWS Acquires DuckLabs", "source": "Hacker News"},
        {"title": "Nvidia is about to be a hundred-billion-dollar company", "source": "The Verge"},
        {"title": "GLM-5.3-Flash", "source": "Hacker News"},
    ]
    assert json.loads(cache.read_text())["items"] == items
    n_calls = len(fetch.calls)
    again, complete = fetch_news(Cfg(), fetch, cache, now=1000.0 + 14 * 60)
    assert again == items and complete and len(fetch.calls) == n_calls   # cache hit
    fetch_news(Cfg(), fetch, cache, now=1000.0 + 16 * 60)
    assert len(fetch.calls) > n_calls                                     # expired


def test_fetch_news_defaults_and_explicit_empty_feeds(tmp_path):
    fetch = FakeFetch()
    items, _ = fetch_news(None, fetch, None, now=1.0)              # no cfg -> Verge + Ars
    assert VERGE in fetch.calls and ARS in fetch.calls and len(items) == 3
    fetch = FakeFetch()
    items, complete = fetch_news(Cfg(news_feeds=[]), fetch, None, now=1.0)
    assert complete and [i["source"] for i in items] == ["Hacker News"] * 3
    assert VERGE not in fetch.calls


def test_fetch_news_partial_failure_not_cached(tmp_path):
    table = canned()
    table[br.HN_TOP_URL] = OSError("hn down")
    fetch = FakeFetch(table)
    cache = tmp_path / "news.json"
    items, complete = fetch_news(Cfg(), fetch, cache, now=5.0)
    assert not complete
    assert [i["source"] for i in items] == ["The Verge", "Ars Technica"]
    assert not cache.exists()
    table = canned()
    table[ARS] = b"<html>not xml"
    items, complete = fetch_news(Cfg(), FakeFetch(table), cache, now=5.0)
    assert not complete and [i["source"] for i in items] == \
        ["Hacker News", "The Verge", "Hacker News"]
    table = canned()
    del table[br.HN_TOP_URL]
    del table[VERGE]
    del table[ARS]
    items, complete = fetch_news(Cfg(), FakeFetch(table), cache, now=5.0)
    assert items == [] and not complete


# ------------------------------------------------------ sports/stocks
def test_stocks_stooq_then_yahoo_fallback():
    fetch = FakeFetch({
        br.STOOQ_URL.format(sym="aapl"): STOOQ_CSV,
        br.STOOQ_URL.format(sym="nvda"): STOOQ_HTML,
        br.YAHOO_URL.format(sym="NVDA"): YAHOO_JSON,
        br.STOOQ_URL.format(sym="zzzz"): b"Symbol,Date,Time,Open,High,Low,Close,Volume\r\nZZZZ.US,N/D,N/D,N/D,N/D,N/D,N/D,N/D\r\n",
    })
    lines = fetch_stocks(["aapl", "NVDA", "zzzz", "bad sym!"], fetch)
    assert lines == ["AAPL 232.14, up 0.9%", "NVDA 191.5, up 1.3%"]
    assert br.YAHOO_URL.format(sym="AAPL") not in fetch.calls
    assert br.stock_line({"symbol": "X", "close": 10.0, "open": 10.0, "prev": None}) == "X 10, flat"
    assert br.stock_line({"symbol": "X", "close": 9.0, "prev": 10.0}) == "X 9, down 10.0%"
    assert br.stock_line({"symbol": "X", "close": 9.0}) == "X 9"
    assert br.parse_stooq_csv(b"") is None and br.parse_yahoo_json(b"{}") is None


def test_sports_first_item_per_feed():
    url = "https://feeds.bbci.co.uk/sport/football/rss.xml"
    lines = br.fetch_sports([url, "https://down.example/rss"], FakeFetch({url: SPORTS_RSS}))
    assert lines == ["City win the derby — BBC Sport"]


# ------------------------------------------------------------ assemble
def test_build_briefing_sections_and_fact_sheet(tmp_path):
    fetch = FakeFetch()
    reg = FakeRegistry()
    sections, sheet = build_briefing(Cfg(), reg, fetch, NOW, tmp_path / "news.json")
    assert sections == {
        "weather": "72°F and partly cloudy, high 85, low 64, 10% chance of rain.",
        "calendar": ["Today: 10:00 am dentist for an hour, 2:30 pm standup; nothing else."],
        "news": [
            {"title": "AWS Acquires DuckLabs", "source": "Hacker News"},
            {"title": "Nvidia is about to be a hundred-billion-dollar company", "source": "The Verge"},
            {"title": "GLM-5.3-Flash", "source": "Hacker News"},
        ],
        "sports": [], "stocks": [],
    }
    assert sheet == (
        "Briefing for Wednesday 26 August, 7:05 am.\n"
        "Weather: 72°F and partly cloudy, high 85, low 64, 10% chance of rain.\n"
        "Calendar: Today: 10:00 am dentist for an hour, 2:30 pm standup; nothing else.\n"
        "News: 1) AWS Acquires DuckLabs (Hacker News) "
        "2) Nvidia is about to be a hundred-billion-dollar company (The Verge) "
        "3) GLM-5.3-Flash (Hacker News)")
    assert reg.calls == [("get_weather", {"when": "today"}), ("get_calendar", {"range": "today"})]
    assert "Sports" not in sheet and "Stocks" not in sheet


def test_build_briefing_with_sports_and_stocks(tmp_path):
    table = canned()
    sport = "https://feeds.bbci.co.uk/sport/football/rss.xml"
    table[sport] = SPORTS_RSS
    table[br.STOOQ_URL.format(sym="aapl")] = STOOQ_CSV
    cfg = Cfg(sports_feeds=[sport], stock_symbols=["AAPL"])
    sections, sheet = build_briefing(cfg, FakeRegistry(), FakeFetch(table), NOW,
                                     tmp_path / "news.json")
    assert sections["sports"] == ["City win the derby — BBC Sport"]
    assert sections["stocks"] == ["AAPL 232.14, up 0.9%"]
    assert sheet.endswith("Sports: City win the derby — BBC Sport\nStocks: AAPL 232.14, up 0.9%")


def test_build_briefing_degrades_per_section(tmp_path):
    table = canned()
    table[br.HN_TOP_URL] = OSError("down")
    reg = FakeRegistry(weather="weather service unreachable", weather_ok=False,
                       calendar="I'll need your Google calendar link set up, sir.",
                       calendar_ok=False)
    sections, sheet = build_briefing(Cfg(stock_symbols=["AAPL"]), reg, FakeFetch(table),
                                     NOW, tmp_path / "news.json")
    assert sections["weather"] == "" and sections["calendar"] == []
    assert [n["source"] for n in sections["news"]] == ["The Verge", "Ars Technica"]
    assert sections["stocks"] == []
    assert sheet.splitlines() == [
        "Briefing for Wednesday 26 August, 7:05 am.",
        "Weather: weather service unreachable",
        "Calendar: I'll need your Google calendar link set up, sir.",
        "News: 1) Nvidia is about to be a hundred-billion-dollar company (The Verge) "
        "2) AWS acquires DuckLabs (Ars Technica)",
        "Stocks: unavailable",
    ]
    # no registry at all, every feed down -> everything unavailable, never raises
    sections, sheet = build_briefing(Cfg(), None, FakeFetch({}), NOW, tmp_path / "n2.json")
    assert sections == {"weather": "", "calendar": [], "news": [], "sports": [], "stocks": []}
    assert sheet.splitlines()[1:] == ["Weather: not available", "Calendar: not available",
                                      "News: unavailable"]
    # a registry without the weather tool registered
    reg = ToolRegistry()
    sections, sheet = build_briefing(Cfg(), reg, FakeFetch(), NOW, tmp_path / "n3.json")
    assert "Weather: not available" in sheet and len(sections["news"]) == 3


def test_build_briefing_accepts_epoch_now_and_naive_datetime(tmp_path):
    _, sheet = build_briefing(Cfg(), FakeRegistry(), FakeFetch(), NOW.timestamp(),
                              tmp_path / "n.json")
    assert sheet.startswith("Briefing for Wednesday 26 August, 7:05 am.")
    _, sheet = build_briefing(Cfg(), FakeRegistry(), FakeFetch(), datetime(2026, 8, 27, 18, 30),
                              tmp_path / "n.json")
    assert sheet.startswith("Briefing for Thursday 27 August, 6:30 pm.")


# ---------------------------------------------------------------- tool
def test_get_briefing_disabled_by_default(tmp_path, monkeypatch):
    fetch = FakeFetch()
    monkeypatch.setattr(br, "_fetch", fetch)
    reg = FakeRegistry()
    services = SimpleNamespace(tools=reg, news_cache_path=tmp_path / "news.json")
    for cfg in (Cfg(), Cfg(enabled="false"), None, {"briefing": {}}):
        (spec,) = br.make_tools(cfg, services)
        r = spec.handler(extra="ignored")
        assert r.text == r.speak == BRIEFING_OFF_LINE and not r.ok
    assert fetch.calls == [] and reg.calls == []
    assert spec.name == "get_briefing"
    assert len(spec.description.split()) <= 20
    assert spec.schema()["function"]["parameters"] == {"type": "object", "properties": {}}


def test_get_briefing_enabled(tmp_path, monkeypatch):
    fetch = FakeFetch()
    monkeypatch.setattr(br, "_fetch", fetch)
    reg = FakeRegistry()
    services = SimpleNamespace(tools=reg, news_cache_path=tmp_path / "news.json")
    reg.register_many(br.make_tools(Cfg(enabled=True), services))
    r = reg.call("get_briefing", {})
    assert r.ok and r.max_sentences == 6 and r.speak is None
    assert r.text.startswith("Briefing for ")
    assert r.card["weather"].startswith("72°F") and len(r.card["news"]) == 3
    assert (tmp_path / "news.json").exists()
    assert not str(tmp_path / "news.json").startswith("/tmp/vss_voice")
    for cfg in (Cfg(enabled="on"), {"briefing": {"enabled": 1}}):
        (spec,) = br.make_tools(cfg, services)
        assert spec.handler().ok


def test_get_briefing_nothing_reachable(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "_fetch", FakeFetch({}))
    services = SimpleNamespace(tools=None, news_cache_path=tmp_path / "news.json")
    (spec,) = br.make_tools(Cfg(enabled=True), services)
    r = spec.handler()
    assert not r.ok and r.speak == br.NOTHING_LINE


# ---------------------------------------------------------------- live
@pytest.mark.skipif(not os.environ.get("JARVIS_LIVE"), reason="JARVIS_LIVE=1 for live feeds")
def test_live_news_feeds(tmp_path):
    items, complete = fetch_news(None, br._fetch, tmp_path / "news.json", now=None)
    assert complete, "a live source failed"
    assert len(items) == 3
    assert {i["source"] for i in items} <= {"Hacker News", "The Verge", "Ars Technica"}
    assert all(i["title"] for i in items)
    print("\nLIVE NEWS:", json.dumps(items, ensure_ascii=False))


@pytest.mark.skipif(not os.environ.get("JARVIS_LIVE"), reason="JARVIS_LIVE=1 for live quotes")
def test_live_stock_quote():
    lines = fetch_stocks(["AAPL"], br._fetch)
    assert len(lines) == 1 and lines[0].startswith("AAPL ")
    print("\nLIVE QUOTE:", lines)


# ---------------------------------------------- through the tool loop
def test_disabled_by_default_in_the_real_assistant_config(tmp_path, monkeypatch):
    """Spec 10.1: a freshly created assistant.json has briefing.enabled
    false, so the tool refuses without touching a single source."""
    from jarvis.assistant_config import AssistantConfig

    cfg = AssistantConfig.load(tmp_path / "assistant.json")
    assert cfg.get("briefing.enabled") is False
    assert br.briefing_enabled(cfg) is False
    fetch = FakeFetch()
    monkeypatch.setattr(br, "_fetch", fetch)
    reg = FakeRegistry()
    (spec,) = br.make_tools(cfg, SimpleNamespace(tools=reg,
                                                 news_cache_path=tmp_path / "n.json"))
    r = spec.handler()
    assert not r.ok and r.speak == BRIEFING_OFF_LINE
    assert "switched off" in r.speak and "Briefing" in r.speak   # says how to enable
    assert fetch.calls == [] and reg.calls == []
    cfg.set("briefing.enabled", True)
    assert br.briefing_enabled(cfg) is True
    assert spec.handler().ok


def _brain_with_briefing(tmp_path, monkeypatch, enabled, reply):
    """A JarvisBrain whose registry holds get_briefing and whose Ollama is
    a canned single turn. Returns (brain, payloads)."""
    from jarvis import brain as brain_mod

    monkeypatch.setattr(br, "_fetch", FakeFetch())
    reg = FakeRegistry()
    services = SimpleNamespace(tools=reg, news_cache_path=tmp_path / "news.json")
    reg.register_many(br.make_tools(Cfg(enabled=enabled), services))
    payloads = []

    def fake_http(path, payload=None, timeout=None):
        payloads.append(payload)
        return {"message": {"content": reply}}

    monkeypatch.setattr(brain_mod, "_http", fake_http)
    return brain_mod.JarvisBrain(None, None, registry=reg), payloads


BRIEF_REPLY = (
    "Good morning, sir. Seventy-two and partly cloudy today, high of eighty-five "
    "with a ten per cent chance of rain. You have the dentist at ten and standup "
    "at half two. Hacker News reports AWS has acquired DuckLabs. The Verge says "
    "Nvidia is about to be a hundred-billion-dollar company. And GLM-5.3-Flash is "
    "the talk of the front page.")


def test_briefing_rendered_by_the_local_model(tmp_path, monkeypatch):
    """Spec 6.7 + 4.2: force_tool runs get_briefing first, then ONE model
    turn renders it; the brain emits BRIEFING(card) then SPEAK."""
    brain, payloads = _brain_with_briefing(tmp_path, monkeypatch, True, BRIEF_REPLY)
    tags = brain._chat_sync("brief me", force_tool="get_briefing")

    assert len(payloads) == 1, "exactly one model turn renders the briefing"
    tool_turn = payloads[0]["messages"][-1]
    assert tool_turn["role"] == "tool" and tool_turn["tool_name"] == "get_briefing"
    assert tool_turn["content"].startswith("Briefing for ")
    assert "GLM-5.3-Flash (Hacker News)" in tool_turn["content"]

    assert [t[0] for t in tags] == ["BRIEFING", "SPEAK"]
    card = json.loads(tags[0][1])
    assert card["weather"].startswith("72°F")
    # round-robin across sources, the duplicated Ars/HN AWS story deduped
    assert [n["source"] for n in card["news"]] == \
        ["Hacker News", "The Verge", "Hacker News"]
    spoken = tags[1][1]
    assert spoken == BRIEF_REPLY, "a briefing may run to six sentences"
    assert 1 <= spoken.count(". ") + 1 <= 6


def test_briefing_disabled_speaks_the_off_line_through_the_brain(tmp_path, monkeypatch):
    """Asked while off: the excuse is spoken verbatim, the model is never
    called and no source is fetched."""
    brain, payloads = _brain_with_briefing(tmp_path, monkeypatch, False, "unused")
    tags = brain._chat_sync("brief me", force_tool="get_briefing")
    assert tags == [("SPEAK", BRIEFING_OFF_LINE)]
    assert payloads == []
