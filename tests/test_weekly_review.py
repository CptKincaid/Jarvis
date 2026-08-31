"""The weekly self-improvement report (jarvis/dayreview.py week_*): the
nightly digest now carries the day's warning clusters, the closed ISO week
is aggregated from the seven filed digests, and what got worse is spoken,
carded, and appended to feedback.jsonl as a standing bug list.

Pure arithmetic over JSON on disk: no log re-reading (the log is tmpfs and
gone by Sunday, which is why the clusters are filed nightly), no model, no
network.
"""
import json
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
import jarvis.dayreview as dr
from jarvis.commander import _WEEKREVIEW_RX, _WEEK_RX, _h_week_review

# A Monday: the week that has closed is the one before it.
MONDAY = date(2026, 8, 31)
KEY = "2026-W35"


def _line(h, name, level, msg):
    return f"{h:02d}:00:00.000 jarvis.{name} {level} {msg}"


def _digest(day, turns=4, median=1.0, worst=2.0, clusters=None, **counts):
    d = {"day": day.isoformat(), "has_data": True, "has_log": True,
         "turns": turns, "answered": turns, "median_wait_s": median,
         "worst_wait_s": worst, "clusters": clusters or []}
    for k in dr.WEEK_COUNTERS + dr.WEEK_TURN_KEYS:
        d.setdefault(k, 0)
    d["turns"], d["answered"] = turns, turns
    d.update(counts)
    return d


def _cluster(logger="tts", message="fish credentials missing", count=1,
             level="WARNING"):
    return {"logger": logger, "message": message, "count": count,
            "level": level, "example": message, "last_time": "03:00:00.000"}


def _week(days=7, **kw):
    return [_digest(MONDAY - timedelta(days=7 - i), **kw) for i in range(days)]


# ------------------------------------------------- clusters in the digest
def test_the_nightly_digest_carries_the_days_clusters():
    lines = [_line(3, "tts", "WARNING", "fish credentials missing; falling back to f5"),
             _line(4, "tts", "WARNING", "fish credentials missing; falling back to f5"),
             _line(5, "brain", "ERROR", "chat error: connection refused"),
             _line(6, "app", "INFO", "nothing to see here")]
    clusters = dr.day_clusters(lines)
    assert [c["logger"] for c in clusters] == ["tts", "brain"]
    assert clusters[0]["count"] == 2 and clusters[1]["level"] == "ERROR"
    # normalised, so tomorrow's copy of the same warning groups with it
    assert "<path>" not in clusters[0]["message"]
    assert clusters[0]["example"].startswith("fish credentials")


def test_clusters_survive_into_the_filed_digest(tmp_path):
    log = tmp_path / "jarvis.log"
    log.write_text("\n".join([
        _line(9, "app", "INFO", f"{dr.BOOT_MARKER} ['get_time']"),
        _line(10, "tts", "WARNING", "fish credentials missing")]))
    digest = dr.summarize_day(log, tmp_path / "turns.jsonl",
                              datetime.fromtimestamp(log.stat().st_mtime).date())
    assert digest["clusters"] and digest["clusters"][0]["logger"] == "tts"
    dr.file_review(tmp_path, digest)
    reloaded = dr.load_review(tmp_path, date.fromisoformat(digest["day"]))
    assert reloaded["clusters"] == digest["clusters"]


def test_a_broken_log_does_not_break_the_digest(monkeypatch):
    monkeypatch.setattr(dr.logtriage, "cluster_warnings",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert dr.day_clusters(["anything"]) == []


# --------------------------------------------------------- the week key
def test_the_week_key_and_days_name_the_closed_week():
    assert dr.week_key(MONDAY) == KEY
    assert dr.week_key(MONDAY + timedelta(days=4)) == KEY   # still that week
    assert dr.week_key(MONDAY + timedelta(days=7)) == "2026-W36"
    days = dr.week_days(MONDAY)
    assert len(days) == 7 and days[0].weekday() == 0
    assert days[-1] == MONDAY - timedelta(days=1)           # last Sunday


def test_the_garden_and_the_report_agree_about_which_week_it_is():
    from jarvis.garden import week_key as garden_key
    assert garden_key(datetime(2026, 8, 31, 2, 0)) == dr.week_key(MONDAY)


# ------------------------------------------------------- summarize_week
def test_totals_sum_and_the_median_is_of_the_daily_medians():
    days = [_digest(MONDAY - timedelta(days=i), turns=2, median=m,
                    speaker_rejections=1)
            for i, m in enumerate([1.0, 2.0, 3.0])]
    week = dr.summarize_week(days, key=KEY)
    assert week["turns"] == 6 and week["speaker_rejections"] == 3
    assert week["median_wait_s"] == 2.0 and week["days_with_data"] == 3
    assert week["week"] == KEY and week["has_data"] is True


def test_a_day_with_no_data_is_not_a_day():
    days = _week(3) + [{"day": "2026-08-28", "has_data": False}]
    assert dr.summarize_week(days, key=KEY)["days_with_data"] == 3


def test_an_empty_week_says_so_rather_than_dividing_by_zero():
    week = dr.summarize_week([], key=KEY)
    assert week["has_data"] is False and week["median_wait_s"] is None
    assert week["turns"] == 0 and dr.week_spoken(week) == ""


def test_clusters_regroup_across_days_and_count_the_days():
    days = [_digest(MONDAY - timedelta(days=i),
                    clusters=[_cluster(count=2)] + ([_cluster(logger="brain",
                                                              message="chat error",
                                                              level="ERROR")]
                                                    if i == 0 else []))
            for i in range(3)]
    clusters = dr.summarize_week(days, key=KEY)["clusters"]
    assert clusters[0]["days"] == 3 and clusters[0]["count"] == 6
    # a thing that broke once outranks nothing, but not the nightly one
    assert clusters[1]["days"] == 1 and clusters[1]["level"] == "ERROR"


# --------------------------------------------------------------- trends
def test_a_wait_that_rose_enough_is_a_trend():
    week = dr.summarize_week(_week(median=2.1), _week(median=1.4), key=KEY)
    assert dr.week_trends(week) == \
        ["the median wait rose from 1.4 to 2.1 seconds"]


def test_a_wait_that_barely_moved_is_not_a_trend():
    week = dr.summarize_week(_week(median=1.5), _week(median=1.4), key=KEY)
    assert dr.week_trends(week) == []


def test_a_counter_that_tripled_is_a_trend_and_a_falling_one_is_not():
    week = dr.summarize_week(_week(speaker_rejections=3),
                             _week(speaker_rejections=1), key=KEY)
    assert dr.week_trends(week) == \
        ["the speaker gate dropped you 21 times, against 7 last week"]
    calm = dr.summarize_week(_week(speaker_rejections=1),
                             _week(speaker_rejections=3), key=KEY)
    assert dr.week_trends(calm) == []


def test_a_first_week_has_nothing_to_compare_with():
    week = dr.summarize_week(_week(), key=KEY)
    assert week["prev"] == {} and dr.week_trends(week) == []


# --------------------------------------------------------------- spoken
def test_two_sentences_the_numbers_then_what_got_worse():
    week = dr.summarize_week(_week(median=2.1), _week(median=1.4), key=KEY)
    said = dr.week_spoken(week)
    assert said.startswith("Last week: 28 turns over 7 days, median wait 2.1 seconds.")
    assert said.endswith("The median wait rose from 1.4 to 2.1 seconds, sir.")


def test_with_no_trend_a_recurring_warning_is_the_second_sentence():
    week = dr.summarize_week(_week(clusters=[_cluster()]), key=KEY)
    said = dr.week_spoken(week)
    assert "tts complained on 7 days running, sir: fish credentials missing." in said


def test_a_clean_week_says_so():
    assert dr.week_spoken(dr.summarize_week(_week(), key=KEY)).endswith(
        "Nothing is getting worse that I can see, sir.")


def test_the_table_carries_the_numbers_the_trends_and_the_clusters():
    week = dr.summarize_week(_week(median=2.1, clusters=[_cluster(count=3)]),
                             _week(median=1.4), key=KEY)
    table = dr.week_table(week)
    assert table.startswith("```\nJarvis week review 2026-W35")
    assert "against last week:" in table and "recurring in the log:" in table
    assert "21x on 7 day(s) tts WARNING" in table


# ---------------------------------------------------------- regressions
def test_a_recurring_cluster_is_actionable_a_one_off_is_not():
    week = dr.summarize_week(
        [_digest(MONDAY - timedelta(days=1), clusters=[_cluster(count=1)]),
         _digest(MONDAY - timedelta(days=2), clusters=[_cluster(count=1)]),
         _digest(MONDAY - timedelta(days=3),
                 clusters=[_cluster(logger="mail", message="one off")])], key=KEY)
    rows = dr.week_regressions(week)
    assert [r["logger"] for r in rows] == ["tts"]
    assert rows[0]["kind"] == "cluster" and rows[0]["days"] == 2
    assert rows[0]["text"] == "tts: fish credentials missing (2x on 2 days)"


def test_a_single_day_burst_is_actionable_when_it_is_frequent():
    week = dr.summarize_week([_digest(MONDAY - timedelta(days=1),
                                      clusters=[_cluster(count=9)])], key=KEY)
    assert len(dr.week_regressions(week)) == 1


def test_worsened_metrics_join_the_list():
    week = dr.summarize_week(_week(median=2.1), _week(median=1.4), key=KEY)
    rows = dr.week_regressions(week)
    assert [r["kind"] for r in rows] == ["metric"]
    assert "median wait rose" in rows[0]["text"]


# --------------------------------------------------------- persistence
def test_the_weekly_report_lives_in_its_own_subdirectory(tmp_path):
    """prune_reviews dates every stem in reviews/*.json; a
    "2026-W35.json" beside the daily files would be an unparseable stem
    forever, so the weeks get a subdirectory of their own."""
    dr.file_review(tmp_path, _digest(MONDAY - timedelta(days=100)))
    week = dr.summarize_week(_week(), key=KEY)
    assert dr.file_week(tmp_path, week).parent.name == dr.WEEKS_DIRNAME
    assert dr.prune_reviews(tmp_path, today=MONDAY) == 1      # only the day file
    assert dr.load_week(tmp_path, KEY)["week"] == KEY


def test_pruning_keeps_the_newest_weeks(tmp_path):
    for i in range(6):
        dr.file_week(tmp_path, {"week": f"2026-W{30 + i:02d}", "has_data": True})
    assert dr.prune_weeks(tmp_path, keep=3) == 3
    assert dr.load_week(tmp_path, "2026-W30") is None
    assert dr.latest_week(tmp_path)["week"] == "2026-W35"


def test_the_pending_report_is_owed_once(tmp_path):
    dr.file_week(tmp_path, dict(dr.summarize_week(_week(), key=KEY), spoken=False))
    assert dr.pending_week(tmp_path)["week"] == KEY
    assert dr.mark_week_spoken(tmp_path, KEY) is True
    assert dr.pending_week(tmp_path) is None
    assert dr.mark_week_spoken(tmp_path, KEY) is False


def test_missing_files_are_None_not_a_traceback(tmp_path):
    assert dr.load_week(tmp_path, KEY) is None
    assert dr.latest_week(tmp_path / "gone") is None
    assert dr.pending_week(tmp_path) is None
    assert dr.prune_weeks(tmp_path / "gone") == 0


# ------------------------------------------------------------ the rung
def _reviewer(tmp_path, today=MONDAY, on_week=None):
    return dr.DayReviewer(tmp_path / "nolog", tmp_path / "noturns", tmp_path,
                          now=lambda: datetime(today.year, today.month, today.day, 2),
                          on_week=on_week)


def test_the_rung_files_the_week_once_and_calls_back(tmp_path):
    for day in dr.week_days(MONDAY):
        dr.file_review(tmp_path, _digest(day))
    seen = []
    r = _reviewer(tmp_path, on_week=seen.append)
    week = r.week_tick()
    assert week["week"] == KEY and week["turns"] == 28
    assert [w["week"] for w in seen] == [KEY]
    assert r.week_tick() is None and len(seen) == 1      # never twice


def test_the_rung_compares_with_the_week_before(tmp_path):
    for day in dr.week_days(MONDAY):
        dr.file_review(tmp_path, _digest(day, median=2.1))
        dr.file_review(tmp_path, _digest(day - timedelta(days=7), median=1.4))
    week = _reviewer(tmp_path).week_tick()
    assert week["prev"]["median_wait_s"] == 1.4
    assert dr.week_trends(week)


def test_a_week_the_box_was_off_files_nothing(tmp_path):
    assert _reviewer(tmp_path).week_tick() is None
    assert dr.load_week(tmp_path, KEY) is None


def test_a_week_with_only_empty_days_is_filed_but_owes_no_line(tmp_path):
    for day in dr.week_days(MONDAY):
        dr.file_review(tmp_path, {"day": day.isoformat(), "has_data": False})
    seen = []
    week = _reviewer(tmp_path, on_week=seen.append).week_tick()
    assert week["has_data"] is False and week["spoken"] is True
    assert seen == [] and dr.pending_week(tmp_path) is None


def test_the_daily_tick_runs_the_weekly_rung(tmp_path):
    for day in dr.week_days(MONDAY):
        dr.file_review(tmp_path, _digest(day))
    r = _reviewer(tmp_path)
    r.tick()
    assert dr.load_week(tmp_path, KEY) is not None


def test_a_weekly_failure_does_not_lose_the_daily_tick(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "summarize_week",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no")))
    for day in dr.week_days(MONDAY):
        dr.file_review(tmp_path, _digest(day))
    assert _reviewer(tmp_path).tick() == []          # the days were already filed


# ------------------------------------------------------------- commands
@pytest.mark.parametrize("said", [
    "weekly review", "my weekly report", "week in review",
    "how was my week", "how did my week go", "what went wrong last week",
    "review my last week", "weekly self review"])
def test_the_weekly_phrasings_match(said):
    assert _WEEKREVIEW_RX.match(said)


@pytest.mark.parametrize("said", [
    "how's my week looking", "weekly briefing", "what's the week ahead",
    "how did yesterday go", "recap my day"])
def test_the_calendar_forecast_keeps_its_own_words(said):
    assert _WEEKREVIEW_RX.match(said) is None


def test_the_forecast_and_the_self_review_never_both_match():
    for said in ("weekly review", "how was my week", "week in review"):
        assert _WEEK_RX.match(said) is None


def test_the_handler_speaks_the_line_and_cards_the_table():
    from jarvis.events import JarvisReply, bus
    cards = []
    bus.subscribe(JarvisReply, lambda ev: cards.append(ev.text))
    c = SimpleNamespace(_svc=lambda n: (lambda: ("two sentences", "the table"))
                        if n == "week_review" else None)
    res = _h_week_review(c, "weekly review", None)
    assert res.handled and res.speak and res.reply == "two sentences"
    assert cards and cards[-1] == "the table"


def test_a_handler_failure_is_an_excuse():
    c = SimpleNamespace(_svc=lambda n: (lambda: 1 / 0) if n == "week_review" else None)
    assert "didn't complete" in _h_week_review(c, "weekly review", None).reply


def test_without_the_service_it_falls_through():
    c = SimpleNamespace(_svc=lambda n: None)
    assert _h_week_review(c, "weekly review", None) is None


# ------------------------------------------------------------- the app
def _app(tmp_path):
    a = object.__new__(app_mod.JarvisApp)
    a.alerts_sent = []
    a._alert = lambda kind, title, text, **kw: a.alerts_sent.append((kind, title, text))
    return a


def test_the_app_posts_the_table_and_files_the_regressions(tmp_path, monkeypatch):
    from jarvis.commander import Commander
    monkeypatch.setattr(Commander, "FEEDBACK_LOG", tmp_path / "feedback.jsonl")
    week = dr.summarize_week(_week(median=2.1, clusters=[_cluster(count=2)]),
                             _week(median=1.4), key=KEY)
    a = _app(tmp_path)
    a._on_week_filed(week)
    kind, title, text = a.alerts_sent[0]
    assert kind == "milestone" and title == f"Week review {KEY}"
    assert "Jarvis week review" in text
    rows = [json.loads(ln) for ln in
            (tmp_path / "feedback.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in rows} == {"regression"}
    assert {r["week"] for r in rows} == {KEY}
    assert any("median wait rose" in r["text"] for r in rows)
    assert any(r["detail"]["kind"] == "cluster" for r in rows)


def test_the_regression_rows_do_not_look_like_intent_labels(tmp_path, monkeypatch):
    """The commander appends {text, label, how} rows to the same file; a
    reader must be able to tell the two apart at a glance."""
    from jarvis.commander import Commander
    monkeypatch.setattr(Commander, "FEEDBACK_LOG", tmp_path / "f.jsonl")
    a = _app(tmp_path)
    a._on_week_filed(dr.summarize_week(_week(median=2.1), _week(median=1.4), key=KEY))
    row = json.loads((tmp_path / "f.jsonl").read_text().splitlines()[0])
    assert row["kind"] == "regression" and "label" not in row
    assert row["how"] == "weekly-review"


def test_nothing_is_appended_when_nothing_regressed(tmp_path, monkeypatch):
    from jarvis.commander import Commander
    monkeypatch.setattr(Commander, "FEEDBACK_LOG", tmp_path / "f.jsonl")
    a = _app(tmp_path)
    a._on_week_filed(dr.summarize_week(_week(), key=KEY))
    assert not (tmp_path / "f.jsonl").exists()


def test_the_spoken_review_is_available_on_demand(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod.PATHS, "REVIEWS_DIR", tmp_path)
    dr.file_week(tmp_path, dr.summarize_week(_week(), key=KEY))
    a = _app(tmp_path)
    spoken, card = a.week_review_text()
    assert spoken.startswith("Last week:") and "Jarvis week review" in card


def test_before_the_first_week_he_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod.PATHS, "REVIEWS_DIR", tmp_path)
    a = _app(tmp_path)
    a.dayreviewer = None
    spoken, card = a.week_review_text()
    assert "no full week to review yet" in spoken and card == ""


def test_the_first_wake_owes_the_line_once(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod.PATHS, "REVIEWS_DIR", tmp_path)
    dr.file_week(tmp_path, dict(dr.summarize_week(_week(), key=KEY), spoken=False))
    a = _app(tmp_path)
    assert a._pending_week_line().startswith("Last week:")
    assert a._pending_week_line() == ""             # marked before speaking


def test_the_garden_line_is_silent_without_a_garden(tmp_path):
    assert _app(tmp_path)._pending_garden_line() == ""
