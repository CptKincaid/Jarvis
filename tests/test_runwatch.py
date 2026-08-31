"""The run ledger (jarvis/runwatch.py): a training run gets a start, a
finish and an honest duration, plus opt-in epoch narration.

Synthetic Proc lists and a fake /proc throughout (the suite never touches
the real one). The /proc/<pid>/stat parse IS exercised against real text,
because reading field 22 by splitting from the front is the classic way
to get the wrong number.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.commander as commander
import jarvis.runwatch as runwatch
from jarvis.commander import ASSISTANT_TIER1, REGISTRY, Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.events import RunProgress
from jarvis.runwatch import (
    ABSENT_TICKS,
    MIN_RUN_S,
    RunLedger,
    elapsed_words,
    parse_progress,
    read_proc_start,
)
from jarvis.tools.health import (LENT_LINE, RECLAIMED_ELAPSED_LINE, Proc,
                                 Snapshot, Watchdog, make_run_ledger)

# A real /proc/<pid>/stat line, trimmed after field 22. The comm here has a
# space AND a ')' in it on purpose: that is what breaks a naive split.
STAT = ("3506483 (python (train)) S 1 3506483 3506483 0 -1 4194304 1 0 0 0 "
        "12 3 0 0 20 0 1 0 14502013 0 0")


def _proc(pid, hint="train.py", gb=30.0):
    return Proc(pid=pid, name="python", rss_gb=gb, hint=hint)


def _ledger(**kw):
    kw.setdefault("start_time", lambda pid: None)
    return RunLedger(**kw)


# ------------------------------------------------------------- /proc reads
def test_proc_start_is_read_from_field_22_past_a_nasty_comm(tmp_path):
    (tmp_path / "3506483").mkdir()
    (tmp_path / "3506483" / "stat").write_text(STAT)
    (tmp_path / "stat").write_text("cpu 1 2 3\nbtime 1787973784\nprocesses 9\n")
    got = read_proc_start(3506483, proc_root=str(tmp_path))
    # 14502013 ticks / 100 = 145020.13 s after boot
    assert got == pytest.approx(1787973784 + 14502013 / runwatch.clock_ticks())


def test_proc_start_is_none_when_proc_will_not_say(tmp_path):
    assert read_proc_start(999_999, proc_root=str(tmp_path)) is None
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "stat").write_text(STAT)
    assert read_proc_start(1, proc_root=str(tmp_path)) is None   # no btime


def test_read_tail_returns_the_end_and_never_raises(tmp_path):
    path = tmp_path / "7.log"
    path.write_text("x" * 100 + "epoch 4\n")
    assert runwatch.read_tail(path, limit=20).endswith("epoch 4\n")
    assert runwatch.read_tail(tmp_path / "missing.log") == ""


# ------------------------------------------------------------------ words
@pytest.mark.parametrize("secs,words", [
    (0, "under a minute"), (29, "under a minute"), (60, "1 minute"),
    (1320, "22 minutes"), (3600, "an hour"), (4200, "an hour and 10 minutes"),
    (7200, "2 hours"),
])
def test_elapsed_words(secs, words):
    assert elapsed_words(secs) == words


@pytest.mark.parametrize("text,found", [
    ("Epoch 1 loss: 0.9\nEpoch 4 loss: 0.31\n", {"epoch": 4, "loss": 0.31}),
    ("epoch=12 | train_loss = 2.5", {"epoch": 12, "loss": None}),
    ("Epoch: 3\nloss 0.5\n", {"epoch": 3, "loss": 0.5}),
    ("Downloading shards 4/8", None),
    ("", None),
])
def test_parse_progress(text, found):
    got = parse_progress(text)
    if found is None:
        assert got is None
    else:
        assert got["epoch"] == found["epoch"] and got["loss"] == found["loss"]


# ------------------------------------------------------------- the ledger
def test_a_run_is_started_once_and_finished_once():
    led = _ledger()
    evs = led.apply([_proc(7)], now=1_000.0)
    assert [(e.kind, e.pid, e.label) for e in evs] == [("started", 7, "train.py")]
    assert evs[0].line() == \
        "Your train.py has started, sir; I'll tell you when it's done."
    # while it runs: silence, however many ticks
    for t in range(1, 5):
        assert led.apply([_proc(7)], now=1_000.0 + t * 30) == []
    # one absent tick is not a finish (an epoch restart must not narrate)
    assert led.apply([], now=1_200.0) == []
    assert ABSENT_TICKS == 2
    evs = led.apply([], now=1_230.0)
    assert [e.kind for e in evs] == ["finished"]
    assert evs[0].elapsed_s == pytest.approx(120.0)       # to the LAST sighting
    assert evs[0].line() == "That's your train.py done, sir; 2 minutes."
    assert led.apply([], now=1_260.0) == []               # and never again


def test_a_run_already_going_reports_its_true_age():
    """The restart case: /proc knows when it started, so the elapsed does
    not reset to when Jarvis happened to look."""
    led = _ledger(start_time=lambda pid: 1_000.0)
    evs = led.apply([_proc(7)], now=2_320.0)
    assert evs[0].started_at == 1_000.0
    assert evs[0].elapsed_s == pytest.approx(1_320.0)
    led.apply([], now=2_350.0)
    done = led.apply([], now=2_380.0)[0]
    assert done.elapsed_s == pytest.approx(1_320.0)       # cached past /proc


def test_a_start_time_from_the_future_is_not_believed():
    led = _ledger(start_time=lambda pid: 9_999.0)
    ev = led.apply([_proc(7)], now=1_000.0)[0]
    assert ev.started_at == 1_000.0 and ev.elapsed_s == 0.0


def test_a_two_second_run_is_recorded_but_not_narrated():
    led = _ledger()
    led.apply([_proc(7)], now=1_000.0)
    led.apply([], now=1_002.0)
    ev = led.apply([], now=1_004.0)[0]
    assert ev.brief is True and ev.line() == "" and MIN_RUN_S == 60.0


def test_two_runs_are_tracked_separately():
    led = _ledger()
    led.apply([_proc(7)], now=1_000.0)
    evs = led.apply([_proc(7), _proc(8, "finetune_piper.py")], now=1_030.0)
    assert [(e.kind, e.pid) for e in evs] == [("started", 8)]
    led.apply([_proc(8, "finetune_piper.py")], now=1_060.0)
    evs = led.apply([_proc(8, "finetune_piper.py")], now=1_090.0)
    assert [(e.kind, e.pid) for e in evs] == [("finished", 7)]
    assert len(led.active) == 1


def test_a_bad_proc_row_is_skipped_not_fatal():
    led = _ledger()
    assert led.apply([SimpleNamespace(pid="not a pid")], now=1_000.0) == []


# ----------------------------------------------------------- the progress
def _tailer(text):
    return lambda path: text


def test_progress_needs_a_log_dir_and_a_changed_epoch(tmp_path):
    led = _ledger(log_dir=tmp_path, tail=_tailer("epoch 1 loss: 0.9\n"),
                  progress_gap_s=300.0)
    led.apply([_proc(7)], now=1_000.0)
    evs = led.apply([_proc(7)], now=1_400.0)
    assert [(e.kind, e.epoch, e.loss) for e in evs] == [("progress", 1, 0.9)]
    assert evs[0].line() == "Epoch 1, sir."           # no previous loss yet
    # same epoch, gap passed: nothing (never a heartbeat)
    led._tail = _tailer("epoch 1 loss: 0.8\n")
    assert led.apply([_proc(7)], now=1_800.0) == []
    # new epoch, but inside the gap: still nothing
    led._tail = _tailer("epoch 2 loss: 0.8\n")
    assert led.apply([_proc(7)], now=1_500.0) == []


def test_progress_says_whether_the_loss_is_falling(tmp_path):
    led = _ledger(log_dir=tmp_path, tail=_tailer("epoch 1 loss: 0.9\n"))
    led.apply([_proc(7)], now=1_000.0)
    led.apply([_proc(7)], now=1_400.0)
    led._tail = _tailer("epoch 4 loss: 0.31\n")
    ev = led.apply([_proc(7)], now=1_800.0)[0]
    assert ev.line() == "Epoch 4, sir; the loss is still falling."
    led._tail = _tailer("epoch 5 loss: 0.44\n")
    ev = led.apply([_proc(7)], now=2_200.0)[0]
    assert ev.line() == "Epoch 5, sir; the loss has stopped falling."


def test_no_log_dir_means_no_progress_at_all():
    led = _ledger(tail=_tailer("epoch 9 loss: 0.1\n"))
    led.apply([_proc(7)], now=1_000.0)
    assert led.apply([_proc(7)], now=9_000.0) == []


def test_an_unreadable_run_log_is_silent(tmp_path):
    def boom(path):
        raise OSError("gone")
    led = _ledger(log_dir=tmp_path, tail=boom)
    led.apply([_proc(7)], now=1_000.0)
    assert led.apply([_proc(7)], now=1_400.0) == []


# --------------------------------------------------------- the watchdog
def _snap(trainers=()):
    return Snapshot(mem_total_gb=121.7, mem_avail_gb=60.0, load1=1.0,
                    trainers=[_proc(pid, hint) for pid, hint in trainers])


class _Clock:
    """The watchdog calls ledger.apply() without a `now`, so the ledger's
    own clock is what a test has to drive. 30 s per tick, the real
    health.DEFAULT_INTERVAL_S."""

    def __init__(self, start=1_000.0, step=30.0):
        self.t, self.step = float(start), float(step)

    def __call__(self):
        return self.t

    def tick(self, n=1):
        self.t += self.step * n


def _wd(ledger, **kw):
    spoken, published = [], []
    wd = Watchdog({}, speak=spoken.append, publish=published.append,
                  interval=0, runs=ledger, **kw)
    return wd, spoken, published


def test_the_watchdog_speaks_two_beats_per_run_and_publishes_the_lane():
    clock = _Clock()
    led = _ledger(clock=clock)
    wd, spoken, published = _wd(led)
    wd.check(_snap([(7, "train.py")]))
    for _ in range(44):                          # 22 minutes of ticks
        clock.tick()
        wd.check(_snap([(7, "train.py")]))
    clock.tick()
    wd.check(_snap())
    clock.tick()
    wd.check(_snap())
    assert spoken == [
        "Your train.py has started, sir; I'll tell you when it's done.",
        "That's your train.py done, sir; 22 minutes."]
    lane = [e for e in published if isinstance(e, RunProgress)]
    assert [(e.kind, e.label) for e in lane] == [("started", "train.py"),
                                                 ("finished", "train.py")]
    assert lane[0].line and lane[1].line


def test_the_gpu_yield_lines_absorb_the_beats_rather_than_doubling_them():
    """The spoken line count per run must not go up: with yield on, the
    lend/reclaim lines ARE the two beats and carry the duration."""
    fake = SimpleNamespace(release=lambda reason="": True, reclaim=lambda: True)
    clock = _Clock()
    # the run was already 10 minutes old when Jarvis first saw it
    led = _ledger(start_time=lambda pid: 400.0, clock=clock)
    wd, spoken, _ = _wd(led, brain=fake)
    wd.yield_to_trainer = True
    wd.check(_snap([(7, "train.py")]))
    assert spoken == [LENT_LINE]
    clock.tick()
    wd.check(_snap())
    clock.tick()
    wd.check(_snap())
    assert len(spoken) == 2
    assert spoken[1] == RECLAIMED_ELAPSED_LINE.format(elapsed="10 minutes")


def test_a_brief_run_does_not_change_the_reclaim_line():
    fake = SimpleNamespace(release=lambda reason="": True, reclaim=lambda: True)
    led = _ledger()
    wd, spoken, _ = _wd(led, brain=fake)
    wd.yield_to_trainer = True
    wd.check(_snap([(7, "train.py")]))
    wd.check(_snap())
    wd.check(_snap())
    from jarvis.tools.health import RECLAIMED_LINE
    assert spoken[1] == RECLAIMED_LINE


def test_narrate_false_keeps_the_lane_and_says_nothing():
    led = _ledger(narrate=False)
    wd, spoken, published = _wd(led)
    wd.check(_snap([(7, "train.py")]))
    assert spoken == []
    lane = [e for e in published if isinstance(e, RunProgress)]
    assert [e.kind for e in lane] == ["started"] and lane[0].line == ""


def test_a_broken_ledger_does_not_sink_the_tick():
    led = _ledger()
    led.apply = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("nope"))
    wd, spoken, _ = _wd(led)
    assert wd.check(_snap([(7, "train.py")])) == [] and spoken == []


def test_no_ledger_at_all_is_the_old_behaviour():
    wd, spoken, published = _wd(None)
    assert wd.check(_snap([(7, "train.py")])) == [] and spoken == []
    assert [e for e in published if isinstance(e, RunProgress)] == []


def test_make_run_ledger_leaves_progress_off_by_default():
    led = make_run_ledger({})
    assert led is not None and led.log_dir is None and led.narrate is True
    led = make_run_ledger({"runwatch": {"progress": True, "log_dir": "~/runs",
                                        "narrate": False, "min_run_s": 5}})
    assert led.log_dir is not None and led.narrate is False and led.min_run_s == 5


def test_the_shipped_defaults_carry_a_runwatch_block():
    from jarvis.assistant_config import DEFAULTS
    block = DEFAULTS["runwatch"]
    assert block["narrate"] is True and block["progress"] is False
    assert block["progress_gap_s"] == 300


# ------------------------------------------------------- "quietly, please"
@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    led = _ledger()
    svc = SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), memory=MagicMock(),
        context=MagicMock(), tts=MagicMock(), brain=MagicMock(),
        health_watchdog=SimpleNamespace(runs=led))
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    c = Commander(svc)
    c.ledger = led
    return c


def test_quietly_is_tier_one():
    assert "quietly" in {c.name for c in REGISTRY}
    assert "quietly" in {c.name for c in ASSISTANT_TIER1}


@pytest.mark.parametrize("phrase", [
    "quietly please", "quietly, please", "narrate quietly", "keep it down",
    "stop narrating", "quietly",
])
def test_quietly_phrases_match(phrase):
    assert commander._QUIETLY_RX.match(phrase)


def test_quietly_does_not_shadow_the_barge_in_words():
    for phrase in ("quiet", "be quiet", "hush", "pipe down", "that's enough"):
        assert not commander._QUIETLY_RX.match(phrase)


def test_quietly_holds_the_narration_for_the_running_job_only(cmdr):
    led = cmdr.ledger
    res = cmdr.handle("quietly please", source="voice")
    assert res.reply == commander.QUIETLY_IDLE_LINE and led.muted is False
    led.apply([_proc(7)], now=1_000.0)
    res = cmdr.handle("quietly please", source="voice")
    assert res.reply == commander.QUIETLY_LINE and led.muted is True
    # the mute is for THAT run: it lifts when the run ends, nothing to restore
    led.apply([], now=1_030.0)
    led.apply([], now=1_060.0)
    assert led.muted is False


def test_a_muted_run_still_publishes_the_lane():
    led = _ledger(clock=_Clock())
    wd, spoken, published = _wd(led)
    wd.check(_snap([(7, "train.py")]))
    spoken.clear()
    led.muted = True
    wd.check(_snap())
    wd.check(_snap())
    assert spoken == []
    assert [e.kind for e in published if isinstance(e, RunProgress)] == \
        ["started", "finished"]
