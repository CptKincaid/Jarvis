"""Syllabus ingestion (jarvis/syllabus.py + the commander's "scan the
syllabus" rung): retrieval from a real chromadb store through the fake
embedder, the parse guards that catch a hallucinated year, the spoken
read-back and its yes/no, and the THREE-source merge -- the accepted rows
must reach deadlines.tick AND canvas.find_next_exam, or he reminds Hunter
about an exam he then denies having.

No Ollama and no Canvas: the embedder is tests.test_docs.FakeEmbed, the
model call is a fake on services.brain, and the store is a tmp file.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.commander as commander
import jarvis.syllabus as syl
import jarvis.tools.docs as docs_mod
from jarvis.commander import Commander, IntentClassifier, syllabus_kind
from jarvis.config import CONFIG, PATHS
from jarvis.deadlines import DeadlineHeadsUp
from jarvis.tools.canvas import find_next_exam
from jarvis.tools.docs import EmbedError
from tests.test_docs import RECIPE, SYLLABUS, FakeEmbed

# A local-zone "now" every date in this file is measured from, taken off the
# REAL clock rather than frozen: parse_rows() is handed NOW by the unit tests
# below, but the commander's scan rung reads its own datetime.now()
# (commander.py:5643) and parse_rows drops rows already past. A frozen
# 2026-09-01 would have started dropping _iso(30) on 2026-10-01 09:00 --
# "Filed, sir; one on the books." instead of two -- and every row by
# 2026-12-10. The fixture must date off whichever clock the rung reads.
#
# Today's DATE at a pinned 09:00, not the bare wall clock: every date here is
# a whole day or more from NOW, so the hour buys nothing, and leaving it free
# breaks test_merge_items_lets_canvas_win_a_duplicate between 23:00 and
# midnight -- merge_items dedupes on due.date() (syllabus.py:275, "the same
# local day"), so its NOW + 30d + 1h would land on the following date.
NOW = datetime.now().astimezone().replace(hour=9, minute=0, second=0, microsecond=0)


def _iso(days, clock="09:00"):
    day = (NOW + timedelta(days=days)).date().isoformat()
    return f"{day}T{clock}" if clock else day


# The model's answer: two real rows, plus the three failures the guards
# exist for (no date, an impossible date, a year two decades out).
MODEL_ROWS = [
    {"title": "Midterm 1", "course": "CS 101", "due": _iso(30)},
    {"title": "Final exam", "course": "CS 101", "due": _iso(100, "")},
    {"title": "Reading week", "course": "CS 101", "due": ""},
    {"title": "Lab 2", "course": "CS 101", "due": "2026-02-31"},
    {"title": "Midterm 2", "course": "CS 101", "due": "2046-11-05"},
]


@pytest.fixture(autouse=True)
def _fresh_env_index(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_DOCS_INDEX_DIR", str(tmp_path / "env_index"))
    # The store default is PATHS.MEMORY_DIR, which conftest firewalls once
    # per SESSION; each test needs its own or they bleed.
    monkeypatch.setattr(syl, "state_path", lambda: tmp_path / syl.STATE_NAME)


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "Jarvis Docs"
    d.mkdir()
    (d / "syllabus.md").write_text(SYLLABUS)
    (d / "recipe.txt").write_text(RECIPE)
    return d


@pytest.fixture
def index(tmp_path, folder):
    embed = FakeEmbed()
    idx = docs_mod.DocsIndex([folder], tmp_path / "index", embed=embed)
    idx.reindex()
    idx.fake_embed = embed
    return idx


# ------------------------------------------------------------- retrieval
def test_gather_dedupes_across_the_three_topics(index):
    """Three queries into one small syllabus return the same chunks; a
    duplicate would be paid for twice in the model's context window."""
    chunks = syl.gather(index)
    assert chunks
    keys = [(c["name"], c["chunk"]) for c in chunks]
    assert len(keys) == len(set(keys))
    assert chunks == sorted(chunks, key=lambda h: (h["name"], h["chunk"]))


def test_gather_lets_an_ollama_outage_through(index, monkeypatch):
    """The caller says "my document index isn't answering"; swallowing it
    here would say "no syllabus" and send him to check the folder."""
    def _down(*a, **k):
        raise EmbedError("URLError: connection refused")
    monkeypatch.setattr(syl, "topic_chunks", _down)
    with pytest.raises(EmbedError):
        syl.gather(index)


def test_gather_survives_one_bad_topic(index, monkeypatch):
    calls = []

    def _flaky(idx, topic, k=6):
        calls.append(topic)
        if topic == syl.TOPICS[0]:
            raise RuntimeError("chroma blew up")
        return docs_mod.topic_chunks(idx, topic, k=k)

    monkeypatch.setattr(syl, "topic_chunks", _flaky)
    assert syl.gather(index)
    assert len(calls) == len(syl.TOPICS)


def test_source_text_names_the_file_and_respects_the_cap(index):
    chunks = syl.gather(index)
    text = syl.source_text(chunks)
    assert "From syllabus.md: " in text
    assert all(line.startswith("From ") for line in text.splitlines())
    # A chunk longer than the whole budget is truncated, not dropped: an
    # empty sheet asks the model to find dates in nothing.
    tight = syl.source_text(chunks, limit=200)
    assert tight.startswith("From ") and 0 < len(tight) <= 201


# --------------------------------------------------------------- parsing
def test_parse_rows_keeps_the_good_and_drops_the_rest():
    rows = syl.parse_rows(MODEL_ROWS, NOW)
    assert [r["title"] for r in rows] == ["Midterm 1", "Final exam"]
    # a time given -> that time; no time -> the 11:59 pm Canvas convention
    assert (rows[0]["due"].hour, rows[0]["due"].minute) == (9, 0)
    assert rows[0]["all_day"] is False
    assert (rows[1]["due"].hour, rows[1]["due"].minute) == (23, 59)
    assert rows[1]["all_day"] is True
    assert all(r["due"].tzinfo is not None for r in rows)


def test_parse_rows_drops_the_hallucinated_year():
    """The failure this whole read-back exists for: a syllabus writes
    "November 5" and the model supplies a year twenty years out."""
    far = [{"title": "Midterm 2", "course": "CS 101", "due": "2046-11-05"}]
    assert syl.parse_rows(far, NOW) == []


def test_parse_rows_drops_what_has_already_happened():
    """A syllabus scanned in October must not file September's midterm."""
    past = [{"title": "Quiz 1", "course": "CS 101", "due": _iso(-3)}]
    assert syl.parse_rows(past, NOW) == []


def test_parse_rows_dedupes_and_caps_and_sorts():
    same = [{"title": "Midterm 1", "course": "CS 101", "due": _iso(30)},
            {"title": "midterm 1", "course": "cs 101", "due": _iso(30, "10:00")}]
    assert len(syl.parse_rows(same, NOW)) == 1
    many = [{"title": f"Lab {i}", "course": "CS 101", "due": _iso(i + 1)}
            for i in range(syl.MAX_ROWS + 5)]
    rows = syl.parse_rows(many, NOW)
    assert len(rows) == syl.MAX_ROWS
    assert rows == sorted(rows, key=lambda r: r["due"])


def test_parse_rows_survives_rubbish():
    assert syl.parse_rows(None, NOW) == []
    assert syl.parse_rows(["a string", 7, {}], NOW) == []


def test_read_back_names_every_row():
    """"I found four dates, shall I add them?" hides exactly the thing
    being confirmed; this is the only moment a wrong date is catchable."""
    rows = syl.parse_rows(MODEL_ROWS, NOW)
    line = syl.read_back_line(rows, NOW)
    assert "2 dates" in line
    for row in rows:
        assert row["title"] in line
    assert "CS 101" in line and line.endswith("?")
    assert "one date" in syl.read_back_line(rows[:1], NOW)


# ----------------------------------------------------------------- store
def test_save_load_round_trip_and_idempotent_add(tmp_path):
    rows = syl.parse_rows(MODEL_ROWS, NOW)
    assert syl.add(rows) == 2
    assert syl.add(rows) == 0, "re-scanning the same syllabus doubled every date"
    back = syl.load()
    assert [r["title"] for r in back] == ["Midterm 1", "Final exam"]
    assert back[0]["due"] == rows[0]["due"] and back[1]["all_day"] is True
    assert not list(tmp_path.glob("*.tmp")), "the atomic temp file was left behind"


def test_load_of_a_missing_or_corrupt_file_is_empty(tmp_path):
    assert syl.load() == []
    (tmp_path / syl.STATE_NAME).write_text("{not json")
    assert syl.load() == []
    (tmp_path / syl.STATE_NAME).write_text('[{"title": "", "due": "nope"}, 7]')
    assert syl.load() == []


def test_stored_items_are_canvas_shaped_and_future_only():
    syl.save([{"title": "Old quiz", "course": "CS 101", "due": NOW - timedelta(days=2),
               "all_day": False},
              {"title": "Midterm 1", "course": "CS 101", "due": NOW + timedelta(days=30),
               "all_day": False}])
    items = syl.stored_items(NOW)
    assert [i["title"] for i in items] == ["Midterm 1"]
    assert set(items[0]) >= {"course", "title", "due"}   # what fetch_due returns
    assert items[0]["source"] == "syllabus"


def test_merge_items_lets_canvas_win_a_duplicate():
    """A professor who posts the midterm to Canvas AND lists it in the
    syllabus must not produce two reminders and two exam-eve calls."""
    canvas_item = {"course": "CS 101", "title": "Midterm 1",
                   "due": NOW + timedelta(days=30, hours=1)}
    rows = [{"course": "CS 101", "title": "midterm 1",
             "due": NOW + timedelta(days=30)},
            {"course": "CS 101", "title": "Final exam",
             "due": NOW + timedelta(days=100)}]
    merged = syl.merge_items([canvas_item], rows)
    assert [i["title"] for i in merged] == ["Midterm 1", "Final exam"]
    assert merged[0] is canvas_item                   # Canvas's due time wins
    assert syl.merge_items([], rows) == sorted(rows, key=lambda r: r["due"])


# ------------------------------------------------- the three-source merge
def test_the_accepted_rows_reach_the_next_exam_answer():
    """Vetting's honest correction: merging only into deadlines.tick would
    have him call an exam eve for a midterm that "when's my next exam"
    then denies. No Canvas token here at all."""
    syl.save([{"title": "Midterm 1", "course": "CS 101",
               "due": NOW + timedelta(days=30), "all_day": False}])
    exam, checked = find_next_exam({"canvas": {"token": ""}}, None, now=NOW)
    assert checked is False              # Canvas was never consulted...
    assert exam and exam["title"] == "Midterm 1"      # ...and he still answers


def test_a_syllabus_deadline_files_its_reminder(tmp_path):
    """deadlines.tick's third source, end to end through the real filer."""
    due = NOW + timedelta(hours=2)
    syl.save([{"title": "Lab 3 report", "course": "CS 101", "due": due,
               "all_day": False}], tmp_path / "s.json")
    tk = MagicMock()
    heads = DeadlineHeadsUp({"canvas": {"token": ""}}, tk, lead_hours=3,
                            state_path=tmp_path / "state.json", now=lambda: NOW,
                            syllabus_path=tmp_path / "s.json")
    assert heads.tick() == 1
    text = tk.add_reminder.call_args[0][1]
    assert text.startswith("Lab 3 report for CS 101 is due in")
    assert heads.tick() == 0, "the same deadline was filed twice"


def test_a_syllabus_exam_gets_the_evening_before_call(tmp_path):
    """The whole point: an exam the professor never put in Canvas."""
    tomorrow = (NOW + timedelta(days=1)).replace(hour=9, minute=0)
    syl.save([{"title": "Midterm 1", "course": "CS 101", "due": tomorrow,
               "all_day": False}], tmp_path / "s.json")
    tk = MagicMock()
    evening = NOW.replace(hour=19, minute=0)
    heads = DeadlineHeadsUp({"canvas": {"token": ""}}, tk, lead_hours=3,
                            state_path=tmp_path / "state.json", now=lambda: evening,
                            syllabus_path=tmp_path / "s.json")
    heads.tick()
    spoken = [c[0][1] for c in tk.add_reminder.call_args_list]
    assert any(t.startswith("Midterm 1 for CS 101 is tomorrow at") for t in spoken)


# ------------------------------------------------------- the commander rung
@pytest.fixture
def services(index):
    svc = SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), memory=MagicMock(),
        context=MagicMock(), tts=MagicMock(), docs=index,
        assistant=SimpleNamespace(get=lambda k, d=None: d),
    )
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.brain = SimpleNamespace(think=MagicMock(), syllabus_calls=[])
    svc.brain.read_syllabus = lambda text, today="": (
        svc.brain.syllabus_calls.append((text, today)) or list(MODEL_ROWS))
    svc.replies = []
    svc.reply = lambda text, speak=True: svc.replies.append((text, speak))
    return svc


@pytest.fixture
def cmdr(services, tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "auto_type", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    return Commander(services)


def test_the_phrasings_that_reach_the_rung():
    for text in ("scan the syllabus", "scan my syllabus", "read my syllabus",
                 "go through my syllabus", "check my syllabus for dates",
                 "add my syllabus dates", "what's on my syllabus",
                 "pull the dates out of my syllabus", "scan my syllabi"):
        assert syllabus_kind(text), text
    for text in ("quiz me on the syllabus", "explain the syllabus", "read the news"):
        assert not syllabus_kind(text), text


def test_it_is_tier_one_without_the_wake_word():
    """The hotword eats "jarvis", so spoken text never reaches the
    prefixed registry."""
    names = {c.name for c in commander.ASSISTANT_TIER1}
    assert "scan syllabus" in names


def test_a_scan_reads_every_date_back_before_filing_anything(cmdr, services):
    res = cmdr.handle("scan my syllabus", "typed")
    assert res.ack and res.done is False and res.speak
    assert res.reply == syl.SCANNING_LINE
    # the model saw the retrieved syllabus text and today's date
    text, today = services.brain.syllabus_calls[0]
    assert "syllabus.md" in text and today
    line = services.replies[-1][0]
    assert line.startswith("I found 2 dates in your syllabus, sir:")
    assert "Midterm 1" in line and "Final exam" in line
    assert syl.load() == [], "dates were filed before he said yes"
    assert cmdr._pending_destructive is not None


def test_yes_files_them_and_no_drops_them(cmdr, services):
    cmdr.handle("scan my syllabus", "typed")
    res = cmdr.handle("yes", "voice")
    assert res.handled and res.speak and res.reply == syl.FILED_LINE.format(n=2)
    assert [r["title"] for r in syl.load()] == ["Midterm 1", "Final exam"]

    cmdr.handle("scan my syllabus", "typed")
    res = cmdr.handle("no", "voice")
    assert res.handled and "Very good" in res.reply
    assert len(syl.load()) == 2, "a refusal filed something"


def test_a_second_yes_on_the_same_syllabus_files_nothing_new(cmdr, services):
    cmdr.handle("scan my syllabus", "typed")
    cmdr.handle("yes", "voice")
    cmdr.handle("scan my syllabus", "typed")
    res = cmdr.handle("yes", "voice")
    assert res.reply == "Already on the books, sir."
    assert len(syl.load()) == 2


def test_no_dates_in_the_documents_says_so(cmdr, services):
    services.brain.read_syllabus = lambda text, today="": []
    cmdr.handle("scan my syllabus", "typed")
    assert services.replies[-1][0] == syl.NO_DATES_LINE
    assert cmdr._pending_destructive is None


def test_a_model_that_falls_over_is_not_a_crash(cmdr, services):
    def _boom(text, today=""):
        raise RuntimeError("ollama said no")
    services.brain.read_syllabus = _boom
    cmdr.handle("scan my syllabus", "typed")
    assert services.replies[-1][0] == syl.NO_DATES_LINE


def test_an_empty_documents_folder_says_so(cmdr, services, tmp_path):
    services.docs = docs_mod.DocsIndex([tmp_path / "nothing"], tmp_path / "i2",
                                       embed=FakeEmbed())
    cmdr.handle("scan my syllabus", "typed")
    assert services.replies[-1][0] == syl.NO_SYLLABUS_LINE


def test_ollama_down_blames_the_index_not_the_folder(cmdr, services, monkeypatch):
    def _down(*a, **k):
        raise EmbedError("URLError: connection refused")
    monkeypatch.setattr(syl, "topic_chunks", _down)
    cmdr.handle("scan my syllabus", "typed")
    assert "index isn't answering" in services.replies[-1][0]


def test_no_documents_service_at_all(services, tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "l.json")
    monkeypatch.setattr(CONFIG, "talkback", True)
    services.docs = None
    res = Commander(services).handle("scan my syllabus", "typed")
    assert res.handled and res.reply == syl.NO_SYLLABUS_LINE


def test_the_store_lives_under_the_memory_dir(monkeypatch):
    monkeypatch.undo()
    assert syl.state_path().parent == PATHS.MEMORY_DIR
