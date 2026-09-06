"""Lecture notes by voice (jarvis/lecture.py + the commander mode).

The commander is real with a real NotesStore under tmp_path, the docs
folder is a tmp folder, the docs index is a recorder for the "end notes"
kick (and, in one test, a real DocsIndex over a fake embedder so the notes
file really lands in the store). Canvas is monkeypatched -- no network.
"""
from __future__ import annotations

import math
import re
import sqlite3
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
import jarvis.recorder as recorder_mod
from jarvis.commander import Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.lecture import (END_LINE, END_NONE_LINE, FAIL_LINE, START_LINE, LectureNotes,
                            _match_course, note_path, resolve_course, slugify)
from jarvis.tools.docs import DOC_PREFIX, DocsIndex
from jarvis.tools.notes import NotesStore


class Cfg:
    def __init__(self, folder, **more):
        self.data = {"docs": {"paths": [str(folder)]}, "lecture": {"window_s": 20},
                     "canvas": {"base_url": "https://canvas.example", "token": ""}}
        self.data.update(more)

    def get(self, dotted, default=None):
        obj = self.data
        for part in dotted.split("."):
            if not isinstance(obj, dict) or part not in obj:
                return default
            obj = obj[part]
        return obj


class FakeIndex:
    def __init__(self):
        self.kicks = 0

    def kick(self):
        self.kicks += 1
        return True


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "Jarvis Docs"
    d.mkdir()
    return d


@pytest.fixture
def notes(tmp_path):
    store = NotesStore(tmp_path / "notes.db")
    yield store
    store.close()


@pytest.fixture
def cmdr(folder, notes, tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    svc = SimpleNamespace(assistant=Cfg(folder), notes=notes, docs_index=FakeIndex(),
                          desktop=SimpleNamespace(parse_action=lambda part: None),
                          tts=None)
    c = Commander(svc)
    c.services = svc
    return c


# ---------------------------------------------------------------- paths
@pytest.mark.parametrize("course,slug", [
    ("Biosensors", "biosensors"), ("BMEN 420 BIOSENSORS", "bmen-420-biosensors"),
    ("Signals & Systems!", "signals-systems"), ("", "course")])
def test_slugify(course, slug):
    assert slugify(course) == slug


def test_note_path_is_under_the_docs_folder(folder):
    p = note_path(Cfg(folder), "Biosensors", datetime(2026, 9, 3, 10, 0))
    assert p == folder / "notes" / "biosensors-2026-09-03.md"


# --------------------------------------------------------------- canvas
def test_match_course_prefers_exact_then_substring_then_words():
    names = ["BIOSENSORS", "SIGNALS AND SYSTEMS", "SYSTEMS PHYSIOLOGY"]
    assert _match_course("biosensors", names) == "BIOSENSORS"
    assert _match_course("signals", names) == "SIGNALS AND SYSTEMS"
    assert _match_course("systems physiology", names) == "SYSTEMS PHYSIOLOGY"
    assert _match_course("physiology lecture", names) == "SYSTEMS PHYSIOLOGY"
    assert _match_course("thermodynamics", names) is None
    assert _match_course("", names) is None


def test_resolve_course_uses_the_canvas_roster_when_linked(folder, monkeypatch):
    from jarvis.tools import canvas
    cfg = Cfg(folder, canvas={"base_url": "https://canvas.example", "token": "tok"})
    monkeypatch.setattr(canvas, "active_courses",
                        lambda settings, fetch, budget: [
                            {"name": "BMEN 420 500 BIOSENSORS FA26"},
                            {"name": "ECEN 314 SIGNALS AND SYSTEMS FA26"}])
    assert resolve_course(cfg, "biosensors") == "BIOSENSORS"
    assert resolve_course(cfg, "signals") == "SIGNALS AND SYSTEMS"
    assert resolve_course(cfg, "thermo") == "thermo"        # not on the roster


def test_resolve_course_falls_back_when_canvas_is_unset_or_down(folder, monkeypatch):
    from jarvis.tools import canvas
    assert resolve_course(Cfg(folder), "biosensors") == "biosensors"   # no token
    cfg = Cfg(folder, canvas={"base_url": "https://canvas.example", "token": "tok"})

    def _down(*a, **k):
        raise canvas.CanvasError("network", "unreachable")
    monkeypatch.setattr(canvas, "active_courses", _down)
    assert resolve_course(cfg, "biosensors") == "biosensors"


# ------------------------------------------------------------- the file
def test_lecture_notes_writes_a_header_then_timestamped_lines(folder, notes):
    clock = [datetime(2026, 9, 3, 10, 5)]
    ln = LectureNotes(Cfg(folder), "Biosensors", notes=notes, now=lambda: clock[0])
    assert ln.path.exists() and ln.path.read_text() == "# Biosensors — 2026-09-03\n\n"
    assert ln.add("Impedance is the ratio of voltage to current") == 1
    clock[0] = datetime(2026, 9, 3, 10, 7)
    assert ln.add("  the   Randles   circuit  ") == 2
    lines = ln.path.read_text().splitlines()
    assert lines[2] == "- 10:05  Impedance is the ratio of voltage to current"
    assert lines[3] == "- 10:07  the Randles circuit"
    assert ln.add("   ") == 2                              # blank: not a line
    assert ln.close() == END_LINE.format(n="2 lines", course="Biosensors")
    rows = sqlite3.connect(str(notes.db_path)).execute(
        "SELECT text, tags FROM notes ORDER BY id").fetchall()
    assert rows == [("Impedance is the ratio of voltage to current", "course:biosensors"),
                    ("the Randles circuit", "course:biosensors")]


def test_reopening_the_same_day_appends(folder):
    clock = datetime(2026, 9, 3, 10, 5)
    first = LectureNotes(Cfg(folder), "Biosensors", now=lambda: clock)
    first.add("one")
    second = LectureNotes(Cfg(folder), "biosensors", now=lambda: clock)
    second.add("two")
    assert second.path == first.path
    assert first.path.read_text().count("\n- ") == 2


def test_an_unwritable_folder_fails_at_open(folder):
    (folder / "notes").write_text("a file where the folder should be")
    with pytest.raises(OSError):
        LectureNotes(Cfg(folder), "Biosensors")


# ------------------------------------------------------------ commander
@pytest.mark.parametrize("text", [
    "notes for biosensors", "take notes for biosensors", "lecture notes for biosensors",
    "jarvis, start lecture notes on biosensors", "class notes in biosensors"])
def test_start_phrasings_open_the_mode(cmdr, text):
    res = cmdr.handle(text, "voice")
    assert res.reply == START_LINE.format(course="biosensors") and res.speak
    assert res.status == "Lecture notes: biosensors"
    assert cmdr.lecture_course == "biosensors"


def test_the_course_keeps_its_casing(cmdr):
    cmdr.handle("Jarvis, notes for Biosensors", "voice")
    assert cmdr.lecture_course == "Biosensors"


def test_a_plain_note_is_still_a_note(cmdr, notes):
    res = cmdr.handle("jarvis, note that the demo is on friday", "voice")
    assert res.reply == "Noted, sir." and cmdr.lecture_course is None


def test_capture_files_every_utterance_until_end_notes(cmdr, folder, notes, monkeypatch):
    cmdr.handle("notes for biosensors", "typed")
    # The mode bypasses the intent classifier and every router: reaching
    # the classifier would be a failure, so make it loud.
    def _boom(*a, **k):
        raise AssertionError("reached the intent classifier")
    cmdr.intent = SimpleNamespace(classify=_boom)

    res = cmdr.handle("Impedance is the ratio of voltage to current", "voice")
    assert res.handled and res.speak is False
    assert res.reply == "Impedance is the ratio of voltage to current"
    assert res.status == "Noting: biosensors (1)"
    res = cmdr.handle("Jarvis, the Randles circuit models the interface", "voice")
    assert res.status == "Noting: biosensors (2)"
    path = folder / "notes" / f"biosensors-{datetime.now().date().isoformat()}.md"
    body = path.read_text()
    assert re.search(r"^- \d\d:\d\d  Impedance is the ratio", body, re.M)
    assert "the Randles circuit models the interface" in body
    assert cmdr.services.docs_index.kicks == 0

    res = cmdr.handle("end notes", "voice")
    assert res.reply == END_LINE.format(n="2 lines", course="biosensors") and res.speak
    assert res.status == "Notes closed" and cmdr.lecture_course is None
    assert cmdr.services.docs_index.kicks == 1, "the index must pick the file up"
    assert [n["text"] for n in notes.list("note")] == [
        "Impedance is the ratio of voltage to current",
        "the Randles circuit models the interface"]
    # ordinary handling is back
    res = cmdr.handle("jarvis, note that the demo is on friday", "voice")
    assert res.reply == "Noted, sir."


@pytest.mark.parametrize("text", ["end notes", "Jarvis, end the notes", "stop notes please",
                                  "close my lecture notes", "end note taking"])
def test_end_phrasings(cmdr, text):
    cmdr.handle("notes for biosensors", "typed")
    res = cmdr.handle(text, "voice")
    assert res.reply == END_NONE_LINE and cmdr.lecture_course is None


def test_no_notes_store_still_writes_the_file(cmdr, folder):
    cmdr.services.notes = None
    cmdr.handle("notes for biosensors", "typed")
    cmdr.handle("a line", "voice")          # the mode files spoken lines only
    assert cmdr.handle("end notes", "typed").reply == END_LINE.format(n="one line",
                                                                       course="biosensors")
    assert "a line" in next((folder / "notes").glob("biosensors-*.md")).read_text()


def test_unwritable_folder_is_refused_at_start(cmdr, folder):
    (folder / "notes").write_text("not a folder")
    res = cmdr.handle("notes for biosensors", "typed")
    assert res.reply == FAIL_LINE and cmdr.lecture_course is None


def test_the_lecture_command_is_tier_one():
    from jarvis.commander import ASSISTANT_TIER1
    assert "lecture notes" in {c.name for c in ASSISTANT_TIER1}


# ---------------------------------------------------------- docs index
DIM = 32


def _fake_embed(texts, model=None, base_url=None, timeout=None):
    out = []
    for text in texts:
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z]+", text.lower()):
            v[sum(ord(c) * (i + 1) for i, c in enumerate(tok)) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


def test_kick_reindexes_after_the_first_pass_and_finds_the_notes(folder, tmp_path,
                                                                 monkeypatch):
    import urllib.request

    def _no_network(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    (folder / "recipe.txt").write_text("Whisk flour with sugar and salt for pancakes.")
    index = DocsIndex([folder], tmp_path / "index", embed=_fake_embed)
    assert index.start_background() is True
    index.wait(20)
    assert index.start_background() is False            # the once-per-process latch
    ln = LectureNotes(Cfg(folder), "Biosensors")
    ln.add("Impedance spectroscopy measures the electrode interface")
    assert index.kick() is True                          # the latch is bypassed
    index.wait(20)
    hits = index.query("impedance spectroscopy electrode")
    assert hits and hits[0]["name"].startswith("biosensors-")
    assert DOC_PREFIX  # imported for the seam's contract; the fake ignores it


def test_kick_is_a_no_op_while_a_pass_runs(folder, tmp_path):
    gate = threading.Event()

    def slow_embed(texts, model=None, base_url=None, timeout=None):
        if any(t.startswith(DOC_PREFIX) for t in texts):
            gate.wait(5)
        return _fake_embed(texts)
    (folder / "a.txt").write_text("alpha beta gamma")
    index = DocsIndex([folder], tmp_path / "index", embed=slow_embed)
    index.start_background()
    time.sleep(0.2)
    assert index.busy() and index.kick() is False
    gate.set()
    index.wait(20)


# ------------------------------------------------------------- the app
def _app(monkeypatch, tmp_path, lecture_course="biosensors"):
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(get=lambda k, d=None: {"lecture.window_s": 20}.get(k, d),
                                  user_name="Hunter")
    a._init_assistant_state()
    a.starts = []
    a.recorder = SimpleNamespace(endpointer=object(), recording=False,
                                 start=lambda followup=False, **kw: a.starts.append((followup, kw)))
    a._audio_busy = threading.Event()
    a._turn_busy = threading.Event()
    a._turn_timer = a._turn_watchdog = None
    a._pending_uncertain = {}
    a._uncertain_lock = threading.Lock()
    a._thinking_i = 0
    a._thinking_delay_s = 5.0
    a._turn_timeout_s = 30.0
    a._tts_active = False
    a._turn_filler_pending = False
    a.tts = SimpleNamespace(is_speaking=False, pending=0)
    a.said = []
    a._say = a.said.append
    a.exchanges = []
    a.context = SimpleNamespace(add_exchange=lambda u, j: a.exchanges.append((u, j)))
    a.turns = SimpleNamespace(mark=lambda *x, **k: None, abandon=lambda r: None)
    a._briefing_state_path = lambda: tmp_path / "b.json"
    a.commander = SimpleNamespace(lecture_course=lecture_course)
    a._turn_after_result = lambda result: None
    return a


def test_a_noted_line_reopens_the_mic_with_the_lecture_window(monkeypatch, tmp_path):
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(CONFIG, "followup_window", 4.0)
    monkeypatch.setattr(app_mod.MACHINE, "has_mic", True, raising=False)
    a = _app(monkeypatch, tmp_path)
    res = SimpleNamespace(reply="a line", speak=False, done=True, ack=False,
                          status="Noting: biosensors (1)", handled=True)
    a.commander.handle = lambda text, source, **kw: res
    a._dispatch("a line", "voice")
    time.sleep(0.4)
    assert a.starts == [(True, {"window": 20.0})], a.starts
    assert a.exchanges == [], "a note is not a conversation exchange"
    assert not a._turn_busy.is_set() and not a._reopen_mic


def test_the_mic_waits_for_speech_to_finish_first(monkeypatch, tmp_path):
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(app_mod.MACHINE, "has_mic", True, raising=False)
    a = _app(monkeypatch, tmp_path)
    a.tts.is_speaking = True
    res = SimpleNamespace(reply="a line", speak=False, done=True, ack=False,
                          status="Noting: biosensors (1)", handled=True)
    a.commander.handle = lambda text, source, **kw: res
    a._dispatch("a line", "voice")
    time.sleep(0.3)
    assert a.starts == [] and a._followup_after_speech


def test_a_typed_note_does_not_touch_the_mic(monkeypatch, tmp_path):
    a = _app(monkeypatch, tmp_path)
    res = SimpleNamespace(reply="a line", speak=False, done=True, ack=False,
                          status="Noting: biosensors (1)", handled=True)
    a.commander.handle = lambda text, source, **kw: res
    a._dispatch("a line", "typed")
    time.sleep(0.3)
    assert a.starts == [] and not a._reopen_mic


def test_outside_lecture_mode_the_window_is_the_default(monkeypatch, tmp_path):
    monkeypatch.setattr(CONFIG, "followup_window", 4.0)
    a = _app(monkeypatch, tmp_path, lecture_course=None)
    assert a._capture_window() is None
    a.commander.lecture_course = "x"
    assert a._capture_window() == 20.0
    a.assistant = SimpleNamespace(get=lambda k, d=None: 2)     # below the default
    assert a._capture_window() == 4.0


# ------------------------------------------------------------ recorder
def test_recorder_clamps_the_window_to_half_the_hard_cap():
    clamp = recorder_mod.Recorder.clamp_window
    assert clamp(None) == 0.0 and clamp("junk") == 0.0
    assert clamp(20) == 20.0
    assert clamp(500) == recorder_mod.MAX_FOLLOWUP_WINDOW_S == 30.0


def _rec(endpointer, window):
    import numpy as np
    rec = object.__new__(recorder_mod.Recorder)
    rec.recording = True
    rec.endpointer = endpointer
    rec._ep_cursor = 0
    rec._record_rate = 16000
    rec._record_start_time = time.monotonic() - 10.0
    rec._audio_frames = [np.zeros((1600, 1), dtype=np.float32)] * 20
    rec._stop_endpoint, rec._stop_dead_air = "", None
    rec._voice_stopped = False
    rec._followup = True
    rec._followup_window = window
    rec.aborted = []
    rec.abort = lambda: rec.aborted.append(True) or setattr(rec, "recording", False)
    rec.stops = []
    rec.stop = lambda reason="manual", **kw: rec.stops.append(reason)
    return rec


def test_a_longer_window_keeps_the_follow_up_open(monkeypatch):
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_vad", True)
    monkeypatch.setattr(recorder_mod.CONFIG, "followup_window", 4.0)

    class Vad:
        silence_since_speech = None

        def feed(self, a, r):
            pass
    rec = _rec(Vad(), 20.0)                 # 10 s in, nothing said: still open
    assert rec._check_endpoint() is False and not rec.aborted
    rec = _rec(Vad(), 0.0)                  # the 4 s default: closed quietly
    assert rec._check_endpoint() is True and rec.aborted
