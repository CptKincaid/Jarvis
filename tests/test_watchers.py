"""The shared watcher bones (jarvis/watchers.py): dotted config reads over a
dict and an object, the seen-id file's pruning and hard cap, an atomic save,
and a thread that stops even when its tick raises every time.
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from jarvis.watchers import PollingWatcher, cfg_get, keyword_hit

TZ = ZoneInfo("America/Chicago")
NOW = datetime(2026, 8, 31, 20, 30, tzinfo=TZ)


@pytest.fixture(autouse=True)
def _firewall():
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")


# ------------------------------------------------------------- cfg_get
def test_cfg_get_reads_dicts_objects_and_missing_keys():
    assert cfg_get({"watch": {"keywords": ["a"]}}, "watch.keywords") == ["a"]
    assert cfg_get({"watch": {}}, "watch.keywords", []) == []
    assert cfg_get({"watch": {"keywords": None}}, "watch.keywords", []) == []
    assert cfg_get(None, "watch.keywords", "d") == "d"
    assert cfg_get({"a": 1}, "a.b", "d") == "d", "a scalar mid-path is not a dict"
    cfg = SimpleNamespace(get=lambda dotted, default=None: {"watch.grades": False}
                          .get(dotted, default))
    assert cfg_get(cfg, "watch.grades", True) is False
    assert cfg_get(cfg, "watch.people_mail", True) is True

    class Angry:
        def get(self, dotted, default=None):
            raise RuntimeError("config on fire")
    # a config whose get() throws falls through to the attribute walk
    assert cfg_get(Angry(), "watch.grades", "d") == "d"


def test_keyword_hit_returns_the_first_match_in_list_order():
    assert keyword_hit("REU and biosensors", ("biosensors", "reu")) == "biosensors"
    assert keyword_hit("REU and biosensors", ("reu", "biosensors")) == "reu"
    assert keyword_hit("x", ("", None)) == ""


# --------------------------------------------------------------- state
class _Probe(PollingWatcher):
    NAME = "probe"
    INTERVAL_S = 0.01
    FIRST_DELAY_S = 0.0
    KEEP_DAYS = 2
    MAX_STATE = 3

    def __init__(self, boom=False, **kw):
        super().__init__(**kw)
        self.ticks = 0
        self._boom = boom

    def tick(self) -> int:
        self.ticks += 1
        if self._boom:
            raise RuntimeError("tick on fire")
        return 0


def _iso(days_ago):
    return (NOW - timedelta(days=days_ago)).astimezone(timezone.utc).isoformat()


def test_seen_ids_older_than_keep_days_are_pruned(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"fresh": _iso(0), "old": _iso(9), "junk": 3}))
    w = _Probe(state_path=path, now=lambda: NOW)
    w._prune_seen()
    assert set(w._state) == {"fresh"}, "a non-string value is junk, not a date"


def test_the_state_file_is_hard_capped_oldest_first(tmp_path):
    w = _Probe(state_path=tmp_path / "s.json", now=lambda: NOW)
    for i in range(5):
        w._state[f"k{i}"] = _iso(i * 0.1)          # k0 newest, k4 oldest
    w._prune_seen()
    assert set(w._state) == {"k0", "k1", "k2"}


def test_the_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "deep" / "s.json"
    w = _Probe(state_path=path, now=lambda: NOW)
    w._mark("one")
    w._save()
    assert list(json.loads(path.read_text())) == ["one"]
    assert list(path.parent.iterdir()) == [path], "a .tmp file was left behind"


def test_an_unreadable_state_file_is_an_empty_start(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("not json at all")
    assert _Probe(state_path=path, now=lambda: NOW)._state == {}
    path.write_text('["a list"]')
    assert _Probe(state_path=path, now=lambda: NOW)._state == {}


def test_no_state_path_means_no_file(tmp_path):
    w = _Probe(now=lambda: NOW)
    w._mark("one")
    w._save()
    assert list(tmp_path.iterdir()) == []


# -------------------------------------------------------------- speak
def test_speak_needs_text_and_a_sink_and_never_raises():
    said = []
    w = _Probe(announce=lambda title, text: said.append((title, text)))
    assert w._speak("T", "a line") is True and said == [("T", "a line")]
    assert w._speak("T", "") is False
    assert _Probe()._speak("T", "no sink attached") is False

    def boom(title, text):
        raise RuntimeError("sink on fire")
    assert _Probe(announce=boom)._speak("T", "a line") is False


# -------------------------------------------------------------- thread
def test_a_tick_that_always_raises_does_not_kill_the_thread(tmp_path):
    w = _Probe(boom=True, state_path=tmp_path / "s.json", now=lambda: NOW)
    w.start()
    deadline = time.monotonic() + 3.0
    while w.ticks < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert w.ticks >= 2, "the loop died on the first exception"
    w.stop()
    assert not w._thread.is_alive()


def test_stop_from_inside_the_tick_does_not_join_itself(tmp_path):
    """stop() called on the watcher's own thread must not deadlock waiting
    for that thread to finish."""
    class _SelfStop(_Probe):
        def tick(self):
            self.ticks += 1
            self.stop()
            return 0

    w = _SelfStop(state_path=tmp_path / "s.json", now=lambda: NOW)
    w.start()
    w._thread.join(timeout=5)
    assert not w._thread.is_alive() and w.ticks == 1
