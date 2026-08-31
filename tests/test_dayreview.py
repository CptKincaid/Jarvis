"""Nightly self-review (jarvis/dayreview.py): the day digest from jarvis.log
+ turns.jsonl, the spoken line, the filed reviews, the first-wake hook and
the "how did yesterday go" command."""
from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
import jarvis.dayreview as dr
from jarvis.commander import _DAYREVIEW_RX, _h_dayreview

TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)
LONG_AGO = TODAY - timedelta(days=5)


def _ts(day, h, m, s):
    return datetime(day.year, day.month, day.day, h, m, s)


def _line(dt, name, level, msg):
    return f"{dt.strftime('%H:%M:%S')}.000 jarvis.{name} {level} {msg}"


def _turn(outcome, wait):
    head = "turn" if outcome == "audio" else f"turn[{outcome}]"
    w = "—" if wait is None else f"{wait:.2f}s"
    return f"{head}: wake→mic 61ms · speech 1.9s · dead-air 0.8s · stt 0.37s · wait {w} (stop=vad)"


@pytest.fixture
def files(tmp_path):
    """A log with two day-slices (yesterday, then today after a midnight
    clock regression) and a ledger whose epochs anchor both. Yesterday
    starts with a fake test-run traceback BEFORE the app's boot marker."""
    y, t = YESTERDAY, TODAY
    boot = _ts(y, 14, 56, 14)
    records = [
        {"outcome": "audio", "wait": 1.4, "at": _ts(y, 15, 4, 11).timestamp()},
        {"outcome": "audio", "wait": 2.6, "at": _ts(y, 15, 30, 0).timestamp()},
        {"outcome": "abort", "wait": None, "at": _ts(y, 15, 30, 9).timestamp()},
        {"outcome": "uncertain", "wait": None, "at": _ts(y, 16, 0, 0).timestamp()},
        {"outcome": "audio", "wait": 0.9, "at": _ts(t, 0, 20, 0).timestamp()},
    ]
    lines = [
        # before the boot: another process (a pre-firewall test run) wrote here
        _line(_ts(y, 14, 38, 15), "config", "INFO", "machine: MachineProfile(...)"),
        _line(_ts(y, 14, 40, 0), "brain", "ERROR", "chat error"),
        "Traceback (most recent call last):",
        '  File "tests/test_x.py", line 9, in boom',
        '    raise RuntimeError("cuda gone")',
        "RuntimeError: cuda gone",
        _line(_ts(y, 14, 40, 1), "tools.registry", "ERROR", "tool get_weather failed"),
        # the app boots
        _line(boot, "app", "INFO", f"{dr.BOOT_MARKER} ['get_time', 'get_weather']"),
        _line(boot + timedelta(seconds=10), "brain", "INFO", "ollama: gemma4:26b resident (load 6.8 s)"),
        _line(_ts(y, 15, 4, 11), "turn", "INFO", _turn("audio", 1.4)),
        _line(_ts(y, 15, 10, 0), "hotword", "INFO", "wake suppressed: speaker score 0.036 < 0.25"),
        _line(_ts(y, 15, 10, 1), "hotword", "INFO", "wake suppressed: speaker score 0.100 < 0.25"),
        _line(_ts(y, 15, 10, 2), "hotword", "INFO", "wake suppressed: speaker score 0.200 < 0.25"),
        _line(_ts(y, 15, 12, 0), "speaker", "INFO", "speaker verify: score=0.210 threshold=0.3 REJECT"),
        _line(_ts(y, 15, 12, 5), "speaker", "INFO", "speaker verify: score=0.410 threshold=0.3 MATCH"),
        _line(_ts(y, 15, 13, 0), "speaker", "INFO", "speaker verify: score=0.150 threshold=0.3 REJECT"),
        _line(_ts(y, 15, 13, 1), "speaker", "INFO", "segment filter: 0/6 windows matched (scores: 0.12, 0.20)"),
        _line(_ts(y, 15, 13, 2), "speaker", "INFO", "segment filter: 4/6 windows matched (scores: 0.42)"),
        _line(_ts(y, 15, 13, 3), "speaker", "INFO", "segment filter: 0/3 windows matched (scores: 0.01)"),
        _line(_ts(y, 15, 14, 0), "speaker", "ERROR", "speaker verify FAILED SHUT (model) -- rejecting audio; typed input still works"),
        _line(_ts(y, 15, 30, 0), "turn", "INFO", _turn("audio", 2.6)),
        _line(_ts(y, 15, 30, 9), "turn", "INFO", _turn("abort", None)),
        _line(_ts(y, 15, 53, 32), "app", "WARNING", "turn watchdog fired after 60s; releasing the wake word"),
        _line(_ts(y, 15, 52, 32), "commander", "INFO", "Uncertain intent (conf=0.50): 'was had like'"),
        _line(_ts(y, 16, 0, 0), "turn", "INFO", _turn("uncertain", None)),
        _line(_ts(y, 16, 5, 0), "commander", "INFO", "Uncertain intent (conf=0.55): 'hmm'"),
        _line(_ts(y, 16, 6, 0), "commander", "INFO", "Ignored (background chat, conf=0.80): 'drop my needle'"),
        _line(_ts(y, 16, 30, 0), "tools.registry", "ERROR", "tool get_mail failed"),
        "Traceback (most recent call last):",
        "imaplib.IMAP4.error: LOGIN failed",
        _line(_ts(y, 16, 31, 0), "commander", "ERROR", "handler last mail failed"),
        _line(_ts(y, 17, 0, 0), "brain", "INFO", "ollama: gemma4:26b resident (load 5.2 s)"),
        _line(_ts(y, 17, 5, 0), "brain", "INFO", "ollama: gemma4:26b resident (load 0.0 s)"),
        _line(_ts(y, 18, 0, 0), "tts", "WARNING", "f5 sidecar unavailable — falling back to edge"),
        _line(_ts(y, 18, 1, 0), "tts", "WARNING", "fish retired (401) — switching to edge permanently"),
        _line(_ts(y, 21, 9, 8), "app", "WARNING", "turn watchdog fired after 60s; releasing the wake word"),
        # midnight: the clock runs backwards
        _line(_ts(t, 0, 10, 0), "app", "INFO", f"{dr.BOOT_MARKER} ['get_time']"),
        _line(_ts(t, 0, 20, 0), "turn", "INFO", _turn("audio", 0.9)),
    ]
    log = tmp_path / "jarvis.log"
    log.write_text("\n".join(lines) + "\n")
    turns = tmp_path / "turns.jsonl"
    turns.write_text("".join(json.dumps(r) + "\n" for r in records))
    return SimpleNamespace(log=log, turns=turns, dir=tmp_path, records=records)


# ------------------------------------------------------------ the digest
def test_yesterday_is_counted_from_the_boot_marker_on(files):
    d = dr.summarize_day(files.log, files.turns, YESTERDAY)
    assert d["has_data"] and d["has_log"]
    assert d["turns"] == 3 and d["aborts"] == 1 and d["uncertain_turns"] == 1
    assert d["median_wait_s"] == 2.0 and d["worst_wait_s"] == 2.6 and d["answered"] == 2
    assert d["speaker_rejections"] == 3       # two 0/N segment filters + the fail-shut
    assert d["verify_rejects"] == 2           # raw REJECT scores: table only
    assert d["wake_suppressed"] == 3
    assert d["watchdog_releases"] == 2
    assert d["tool_exceptions"] == 2          # get_mail (registry) + "last mail" (handler)
    assert d["tts_fallbacks"] == 2
    assert d["uncertain"] == 2 and d["ignored"] == 1
    assert d["residency_reloads"] == 1        # the 6.8 s boot load is not a reload
    assert d["boots"] == 1
    # the pre-boot test-run traceback is cut, not keyword-filtered
    assert d["errors"] == 3                   # FAILED SHUT, get_mail, timer
    assert not any("cuda gone" in n or "chat error" in n or "get_weather" in n
                   for n in d["notes"])
    assert any("get_mail" in n for n in d["notes"])


def test_today_is_dated_by_the_ledger_anchor_not_the_mtime(files):
    d = dr.summarize_day(files.log, files.turns, TODAY)
    assert d["turns"] == 1 and d["boots"] == 1 and d["watchdog_releases"] == 0
    assert d["median_wait_s"] == 0.9


def test_a_day_with_nothing_has_no_data_and_no_line(files):
    d = dr.summarize_day(files.log, files.turns, LONG_AGO)
    assert not d["has_data"] and d["turns"] == 0 and d["log_lines"] == 0
    assert dr.spoken_line(d) == ""


def test_missing_files_do_not_raise(tmp_path):
    d = dr.summarize_day(tmp_path / "nope.log", tmp_path / "nope.jsonl", YESTERDAY)
    assert not d["has_data"] and d["turns"] == 0


def test_a_day_without_a_boot_is_kept_whole():
    lines = ["10:00:00.000 jarvis.app WARNING turn watchdog fired after 60s; releasing the wake word",
             "11:00:00.000 jarvis.app INFO something"]
    assert dr.boot_cut(lines) == lines
    assert dr.count_events(lines)["watchdog_releases"] == 1


def test_split_days_ignores_small_out_of_order_stamps():
    lines = ["10:00:00.000 jarvis.a INFO x", "09:59:59.500 jarvis.b INFO y",   # thread jitter
             "Traceback (most recent call last):",
             "00:00:01.000 jarvis.a INFO z"]                                     # midnight
    days = dr.split_days(lines)
    assert [len(d) for d in days] == [3, 1]


def test_rotated_files_are_read_oldest_first(tmp_path):
    (tmp_path / "jarvis.log.2").write_text("a\n")
    (tmp_path / "jarvis.log.1").write_text("b\n")
    (tmp_path / "jarvis.log").write_text("c\n")
    assert dr.read_log_lines(tmp_path / "jarvis.log") == ["a", "b", "c"]


def test_an_unanchored_slice_is_counted_back_from_the_next():
    segs = [["10:00:00.000 jarvis.app INFO early"], ["09:00:00.000 jarvis.app INFO later"]]
    dated = dr.date_segments(segs, [], TODAY)
    assert [d for d, _ in dated] == [YESTERDAY, TODAY]


# ----------------------------------------------------------- the wording
def test_the_spoken_line_is_two_sentences_with_the_numbers(files):
    d = dr.summarize_day(files.log, files.turns, YESTERDAY)
    line = dr.spoken_line(d)
    assert line.startswith("Yesterday: 3 turns, median wait 2.0 seconds, worst 2.6. ")
    assert "I dropped 3 clips of yours at the speaker gate" in line
    assert "the turn watchdog let go twice" in line
    assert "2 tool calls failed" in line
    assert "the voice fell back twice" in line
    assert "reloaded once" in line
    assert line.endswith(", sir.")
    assert line.count(". ") == 1                  # exactly two sentences


def test_a_clean_day_says_so():
    d = {"has_data": True, "turns": 12, "median_wait_s": 1.3, "worst_wait_s": 1.31}
    assert dr.spoken_line(d, name="sir") == \
        "Yesterday: 12 turns, median wait 1.3 seconds. Nothing went wrong that I could see, sir."
    d = {"has_data": True, "turns": 0, "watchdog_releases": 1}
    assert dr.spoken_line(d, label="So far today") == \
        "So far today: no voice turns. The turn watchdog let go once, sir."


def test_the_table_has_every_row(files):
    d = dr.summarize_day(files.log, files.turns, YESTERDAY)
    text = dr.table(d)
    assert text.startswith("```\n") and text.endswith("\n```")
    for label in ("turns", "median wait", "worst wait", "speaker-gate rejections",
                  "watchdog releases", "tool-handler exceptions", "TTS fallbacks",
                  "model reloads", "app boots", "notes:"):
        assert label in text, label
    assert f"day review {YESTERDAY.isoformat()}" in text


# ------------------------------------------------------- filing / timer
def test_the_reviewer_files_yesterday_once_and_hands_it_on(files):
    posted = []
    r = dr.DayReviewer(files.log, files.turns, files.dir / "reviews",
                       on_filed=lambda day, d: posted.append((day, d["turns"])))
    filed = r.tick()
    assert YESTERDAY in filed and TODAY not in filed
    # empty days are filed too (so they are not re-checked) but not posted
    assert posted == [(YESTERDAY, 3)]
    assert (files.dir / "reviews" / f"{YESTERDAY.isoformat()}.json").exists()
    assert r.tick() == [] and posted == [(YESTERDAY, 3)]
    # a filed day is served from disk: the log changing later does not move it
    files.log.write_text("")
    assert r.review(YESTERDAY)["turns"] == 3
    assert r.review(YESTERDAY, refresh=True)["turns"] == 3      # ledger still there
    assert r.review(TODAY)["turns"] == 1                        # today: computed, never filed
    assert not (files.dir / "reviews" / f"{TODAY.isoformat()}.json").exists()


def test_a_failing_hook_does_not_stop_the_filing(files):
    def boom(day, d):
        raise RuntimeError("discord down")
    r = dr.DayReviewer(files.log, files.turns, files.dir / "reviews", on_filed=boom)
    assert YESTERDAY in r.tick()


def test_old_reviews_are_pruned(tmp_path):
    old = tmp_path / f"{(TODAY - timedelta(days=dr.REVIEW_KEEP_DAYS + 1)).isoformat()}.json"
    keep = tmp_path / f"{YESTERDAY.isoformat()}.json"
    old.write_text("{}"), keep.write_text("{}")
    (tmp_path / "notes.json").write_text("{}")
    assert dr.prune_reviews(tmp_path, today=TODAY) == 1
    assert keep.exists() and not old.exists() and (tmp_path / "notes.json").exists()


def test_the_thread_starts_and_stops(files, monkeypatch):
    monkeypatch.setattr(dr, "FIRST_TICK_S", 0.01)
    monkeypatch.setattr(dr, "TICK_S", 0.01)
    r = dr.DayReviewer(files.log, files.turns, files.dir / "reviews")
    r.start()
    done = threading.Event()
    deadline = datetime.now() + timedelta(seconds=3)
    while datetime.now() < deadline and not done.is_set():
        if (files.dir / "reviews" / f"{YESTERDAY.isoformat()}.json").exists():
            done.set()
        else:
            threading.Event().wait(0.02)
    r.stop()
    assert done.is_set()
    r._thread.join(timeout=2)
    assert not r._thread.is_alive()


# ------------------------------------------------------------- the app
def _app(monkeypatch, tmp_path, digest, busy=False):
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(get=lambda k, d=None: {"briefing.on_first_wake": True,
                                                         "briefing.after": "06:00"}.get(k, d),
                                  user_name="Hunter")
    a._init_assistant_state()
    a.brain = SimpleNamespace(is_busy=busy)
    a.said = []
    a._say = a.said.append
    a.chats = []
    a.services = SimpleNamespace(brain=SimpleNamespace(chat=lambda t, **kw: a.chats.append((t, kw))))
    a._briefing_state_path = lambda: tmp_path / "briefing.json"
    a.dayreviewer = SimpleNamespace(review=lambda day: digest)
    return a


def test_first_wake_speaks_the_review_before_the_briefing(monkeypatch, tmp_path):
    digest = {"has_data": True, "turns": 4, "median_wait_s": 1.2, "worst_wait_s": 3.0,
              "speaker_rejections": 2}
    a = _app(monkeypatch, tmp_path, digest)
    a._deliver_first_wake_briefing()
    assert a.said == ["Yesterday: 4 turns, median wait 1.2 seconds, worst 3.0. "
                      "I dropped 2 clips of yours at the speaker gate, sir.",
                      "Your briefing for today, sir."]
    assert len(a.chats) == 1 and a.chats[0][1] == {"force_tool": "get_briefing"}


def test_first_wake_says_nothing_about_a_day_with_no_data(monkeypatch, tmp_path):
    a = _app(monkeypatch, tmp_path, {"has_data": False})
    a._deliver_first_wake_briefing()
    assert a.said == ["Your briefing for today, sir."]


def test_a_broken_review_does_not_block_the_briefing(monkeypatch, tmp_path):
    a = _app(monkeypatch, tmp_path, {})
    def boom(day):
        raise RuntimeError("disk")
    a.dayreviewer = SimpleNamespace(review=boom)
    a._deliver_first_wake_briefing()
    assert a.said == ["Your briefing for today, sir."] and len(a.chats) == 1


def test_a_busy_model_still_leaves_the_review_unspoken(monkeypatch, tmp_path):
    a = _app(monkeypatch, tmp_path, {"has_data": True, "turns": 1}, busy=True)
    a._deliver_first_wake_briefing()
    assert a.said == [] and a.chats == []


def test_day_review_text_on_demand(monkeypatch, tmp_path, files):
    from jarvis.config import PATHS
    monkeypatch.setattr(PATHS, "LOG_DIR", files.dir)
    a = _app(monkeypatch, tmp_path, {"has_data": False})
    assert a.day_review_text("yesterday").startswith("I have no record of yesterday")
    a.dayreviewer = SimpleNamespace(review=lambda day: {"has_data": True, "turns": 2})
    assert a.day_review_text("yesterday") == \
        "Yesterday: 2 turns. Nothing went wrong that I could see, sir."
    # today is read live from the log dir, never from a filed digest
    assert a.day_review_text("today").startswith("So far today: 1 turn, median wait 0.9 seconds.")


def test_the_review_is_posted_through_the_alerts_hub(monkeypatch, tmp_path):
    a = _app(monkeypatch, tmp_path, {})
    sent = []
    a.alerts = SimpleNamespace(alert=lambda kind, title, text, request_id=None:
                               sent.append((kind, title, text)))
    a._on_review_filed(YESTERDAY, {"day": YESTERDAY.isoformat(), "turns": 3})
    assert sent[0][0] == "milestone" and YESTERDAY.isoformat() in sent[0][1]
    assert sent[0][2].startswith("```")


# ------------------------------------------------------------ the command
def test_the_phrasings():
    for said, which in (("how did yesterday go", "yesterday"), ("how was yesterday", "yesterday"),
                        ("how's today going", "today"), ("how is today going", "today"),
                        ("how did today go", "today"), ("yesterday's review", "yesterday"),
                        ("review yesterday", "yesterday"), ("what went wrong yesterday", "yesterday"),
                        ("how did things go yesterday", "yesterday"), ("how has today been", "today")):
        m = _DAYREVIEW_RX.match(said)
        assert m, said
        assert next(g for g in m.groups() if g) == which, said
    for said in ("how are you doing today", "how did the meeting go yesterday",
                 "how are your systems", "what did I do yesterday"):
        assert not _DAYREVIEW_RX.match(said), said


def test_the_handler_speaks_what_the_app_reports():
    asked = []
    c = SimpleNamespace(_svc=lambda n: (lambda w: asked.append(w) or "Yesterday: fine, sir.")
                        if n == "dayreview" else None)
    m = _DAYREVIEW_RX.match("how did today go")
    res = _h_dayreview(c, "how did today go", m)
    assert res.handled and res.speak and res.reply == "Yesterday: fine, sir."
    assert asked == ["today"] and res.status == "Day review"
    assert _h_dayreview(SimpleNamespace(_svc=lambda n: None), "how did yesterday go",
                        _DAYREVIEW_RX.match("how did yesterday go")) is None


def test_the_command_is_tier_one_without_the_prefix():
    from jarvis.commander import ASSISTANT_TIER1
    assert "day review" in [c.name for c in ASSISTANT_TIER1]


def test_the_reviewer_thread_is_joinable_and_restartable(files):
    """stop() used to be a bare Event.set(): a tick in flight (a Discord
    post) outlived stop_assistant into the teardown, and a stopped
    reviewer could never be started again."""
    r = dr.DayReviewer(files.log, files.turns, files.dir / "reviews")
    r.start()
    t1 = r._thread
    assert t1 is not None and t1.is_alive()
    r.stop()
    t1.join(timeout=2)
    assert not t1.is_alive()
    r.start()                                  # a fresh thread, cleared stop
    assert r._thread is not t1 and r._thread.is_alive()
    r.stop()


def test_the_asides_counter_matches_a_line_repr_quoted_with_doubles():
    """Regression: aside.py logs the spoken line with %r, and repr switches
    to double quotes as soon as the text holds an apostrophe -- so a
    counter anchored on `^aside: '` missed every possessive/contraction
    aside and deflated only the volunteered half of the ratio the day
    review prints beside "asides silenced"."""
    rx = dict(dr.COUNTERS)["asides"]
    plain = "aside: " + repr("the seminar is due at 5")
    apostrophe = "aside: " + repr("Newton's laws quiz is due at 11:59 pm")
    assert '"' in apostrophe and "'" in plain     # repr really does switch
    assert rx.search(plain) and rx.search(apostrophe)
    # ...and the sibling lines still belong to their own counters.
    silenced = "aside: silenced for the day by " + repr("no more asides")
    assert not rx.search(silenced)
    assert dict(dr.COUNTERS)["asides_silenced"].search(silenced)
    assert not rx.search("aside: quiet this turn (busy)")
