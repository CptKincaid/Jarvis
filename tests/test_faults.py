"""The fault lane (jarvis/faults.py): the board that HOLDS a fault, the
spoken-once state file, the watchdog's trainers rule, the engine card's
FAULT row and "what's wrong".

No Tk, no /proc, no nvidia-smi: the watchdog is driven through check()
with synthetic Snapshots (the tests/test_gpu_yield.py pattern) and the
card row is asserted through the pure telemetry contract, never a canvas.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.commander as commander
from jarvis.commander import ASSISTANT_TIER1, REGISTRY, Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.events import FaultRaised, Status
from jarvis.faults import (
    CLEAR_TOKEN,
    NOTHING_WRONG_LINE,
    SPOKEN_COOLDOWN_S,
    FaultBoard,
    FaultLog,
    token_for,
)
from jarvis.tools.health import (
    TRAINER_COUNT,
    Proc,
    Snapshot,
    Watchdog,
    distinct_runs,
    trainers_line,
)


def _snap(trainers=(), avail=60.0, top=()):
    return Snapshot(mem_total_gb=121.7, mem_avail_gb=avail, load1=1.0,
                    top=[Proc(pid=p, name=n, rss_gb=g) for p, n, g in top],
                    trainers=[Proc(pid=pid, name="python", rss_gb=gb, hint=hint)
                              for pid, gb, hint in trainers])


def _wd(**kw):
    spoken, published = [], []
    wd = Watchdog({}, speak=spoken.append, publish=published.append,
                  interval=0, **kw)
    return wd, spoken, published


# ------------------------------------------------------------ the token
@pytest.mark.parametrize("rule,text,token", [
    ("trainers", "2 trainers on the pool", "2 TRAINERS"),
    ("hogs", "2 processes over 20 GB", "2 HOGS"),
    ("memory", "Memory tight: 12 GB free", "MEM 12G"),
    ("memory", "Memory critical: 7.4 GB free", "MEM 7.4G"),
    ("memory", "Memory unreadable", "MEMORY"),
    ("disk", "", "disk"),
])
def test_token_for_fits_the_card(rule, text, token):
    got = token_for(rule, text)
    assert got == token and len(got) <= 10


# ------------------------------------------------------------- the board
def test_the_board_holds_a_fault_until_the_detector_clears_it():
    board = FaultBoard()
    assert board.token == CLEAR_TOKEN and board.current is None
    board.apply(FaultRaised(rule="trainers", kind="error", token="2 TRAINERS",
                            text="2 trainers on the pool", line="Two trainers, sir."))
    assert board.token == "2 TRAINERS" and board.kind == "error"
    # a second event on the same rule keeps ONE episode (since is preserved)
    since = board.current.since
    board.apply(FaultRaised(rule="trainers", kind="error", token="2 TRAINERS",
                            text="2 trainers on the pool"))
    assert board.current.since == since
    board.apply(FaultRaised(rule="trainers", cleared=True))
    assert board.current is None and board.token == CLEAR_TOKEN


def test_a_clear_on_another_rule_does_not_wipe_the_live_fault():
    board = FaultBoard()
    board.apply(FaultRaised(rule="trainers", kind="error", token="2 TRAINERS"))
    board.apply(FaultRaised(rule="memory", cleared=True))
    assert board.token == "2 TRAINERS"


def test_an_error_outranks_a_later_warning():
    board = FaultBoard()
    board.apply(FaultRaised(rule="memory", kind="error", token="MEM 7.4G"))
    board.apply(FaultRaised(rule="hogs", kind="warn", token="2 HOGS"))
    assert board.token == "MEM 7.4G"


def test_describe_says_the_line_and_how_long_it_has_stood():
    board = FaultBoard()
    board.apply(FaultRaised(rule="memory", kind="warn", token="MEM 12G",
                            text="Memory tight", line="Memory is tight, sir."))
    fault = board.current
    assert board.describe(now=fault.since + 5) == "Memory is tight, sir."
    assert board.describe(now=fault.since + 600) == \
        "Memory is tight, sir. That's been standing 10 minutes, sir."
    board.clear()
    assert board.describe() == ""


def test_the_board_survives_a_broken_event():
    board = FaultBoard()
    board.on_event(SimpleNamespace(cleared=False))     # no fields at all
    assert board.token == CLEAR_TOKEN


# ---------------------------------------------------------- the state file
def test_the_state_file_says_it_once_and_re_arms_after_the_cooldown(tmp_path):
    path = tmp_path / "faults.json"
    fl = FaultLog(path)
    assert fl.should_speak("trainers:2 TRAINERS", now=1_000.0) is True
    assert fl.should_speak("trainers:2 TRAINERS", now=1_100.0) is False
    # a RESTART must not re-announce the same episode: that is the point
    assert FaultLog(path).should_speak("trainers:2 TRAINERS", now=1_200.0) is False
    later = 1_000.0 + SPOKEN_COOLDOWN_S + 1
    assert FaultLog(path).should_speak("trainers:2 TRAINERS", now=later) is True


def test_clearing_makes_the_next_occurrence_news_again(tmp_path):
    fl = FaultLog(tmp_path / "faults.json")
    assert fl.should_speak("memory:MEM 12G", now=1_000.0) is True
    fl.clear("memory:MEM 12G")
    assert fl.should_speak("memory:MEM 12G", now=1_010.0) is True


def test_the_state_file_is_written_atomically_and_survives_corruption(tmp_path):
    path = tmp_path / "faults.json"
    FaultLog(path).should_speak("memory:MEM 12G", now=1_000.0)
    assert json.loads(path.read_text())["spoken"]["memory:MEM 12G"] == 1_000.0
    assert not list(tmp_path.glob("*.tmp"))
    path.write_text("{not json")
    assert FaultLog(path).should_speak("memory:MEM 12G", now=1_001.0) is True


def test_an_unwritable_state_file_costs_the_dedupe_not_the_warning(tmp_path):
    fl = FaultLog(tmp_path / "nope" / "deep" / "faults.json")
    fl.path = tmp_path / "file"          # a FILE where the parent dir must be
    fl.path.write_text("x")
    fl.path = fl.path / "faults.json"
    assert fl.should_speak("memory:MEM 12G", now=1_000.0) is True


# -------------------------------------------------------- the trainers rule
def test_distinct_runs_counts_jobs_not_ddp_workers():
    procs = [Proc(pid=i, name="python", rss_gb=float(10 - i), hint="train.py")
             for i in range(4)]
    assert len(distinct_runs(procs)) == 1
    procs.append(Proc(pid=9, name="python", rss_gb=3.0, hint="finetune_piper.py"))
    runs = distinct_runs(procs)
    assert [p.hint for p in runs] == ["train.py", "finetune_piper.py"]


def test_two_trainer_runs_raise_one_error_and_name_the_incident():
    wd, spoken, published = _wd()
    assert wd.check(_snap([(7, 30.0, "train.py")])) == []
    fired = wd.check(_snap([(7, 30.0, "train.py"), (8, 20.0, "finetune_piper.py")]))
    assert [(a.rule, a.kind) for a in fired] == [("trainers", "error")]
    assert spoken == ["2 trainers are on the pool at once, sir: train.py and "
                      "finetune_piper.py. The last time that happened the box "
                      "needed a hard power-off."]
    # once per episode
    assert wd.check(_snap([(7, 30.0, "train.py"), (8, 20.0, "finetune_piper.py")])) == []
    assert len(spoken) == 1
    faults = [e for e in published if isinstance(e, FaultRaised)]
    assert [(f.rule, f.token, f.cleared) for f in faults] == \
        [("trainers", "2 TRAINERS", False)]
    # one exits: the fault clears and the rule re-arms
    assert wd.check(_snap([(7, 30.0, "train.py")])) == []
    cleared = [e for e in published if isinstance(e, FaultRaised) and e.cleared]
    assert [c.rule for c in cleared] == ["trainers"]
    assert wd.check(_snap([(7, 30.0, "train.py"), (9, 20.0, "train_yolo.py")]))
    assert len(spoken) == 2


def test_four_ddp_workers_are_one_run_and_say_nothing():
    wd, spoken, _ = _wd()
    procs = [(pid, 20.0, "train.py") for pid in (7, 8, 9, 10)]
    assert wd.check(_snap(procs)) == [] and spoken == []
    assert TRAINER_COUNT == 2


def test_trainers_line_names_the_scripts_not_python():
    line = trainers_line([Proc(pid=1, name="python", rss_gb=30.0, hint="train.py"),
                          Proc(pid=2, name="python", rss_gb=20.0, hint="")])
    assert "train.py and python" in line


# --------------------------------------------------- the watchdog + the log
def test_a_fault_already_spoken_is_shown_but_not_repeated(tmp_path):
    """The restart case: the state file remembers, so the board lights and
    the room stays quiet."""
    path = tmp_path / "faults.json"
    wd, spoken, published = _wd(faults=FaultLog(path))
    two = [(7, 30.0, "train.py"), (8, 20.0, "finetune_piper.py")]
    wd.check(_snap(two))
    assert len(spoken) == 1
    wd2, spoken2, published2 = _wd(faults=FaultLog(path))
    wd2.check(_snap(two))
    assert spoken2 == []                                  # said once, not twice
    assert [f.token for f in published2 if isinstance(f, FaultRaised)] == \
        ["2 TRAINERS"]                                    # the board still lights


def test_a_recovery_clears_the_state_file_entry(tmp_path):
    path = tmp_path / "faults.json"
    wd, spoken, _ = _wd(faults=FaultLog(path))
    two = [(7, 30.0, "train.py"), (8, 20.0, "finetune_piper.py")]
    wd.check(_snap(two))
    wd.check(_snap([(7, 30.0, "train.py")]))              # recovered: cleared
    wd2, spoken2, _ = _wd(faults=FaultLog(path))
    wd2.check(_snap(two))
    assert spoken2 and len(spoken) == 1                   # a NEW episode speaks


def test_no_fault_log_keeps_the_old_behaviour():
    wd, spoken, _ = _wd()
    assert wd._faults is None
    wd.check(_snap(avail=7.0))
    assert len(spoken) == 1


def test_the_memory_recovery_still_ends_with_the_ok_chip():
    """FaultRaised(cleared) must be published BEFORE the recovery Status:
    tests and the UI both read the last event as the chip."""
    wd, _, published = _wd()
    wd.check(_snap(avail=7.0))
    published.clear()
    wd.check(_snap(avail=25.0))
    assert isinstance(published[-1], Status) and published[-1].kind == "ok"
    assert any(isinstance(e, FaultRaised) and e.cleared for e in published)


def test_the_lend_rule_is_not_a_fault():
    wd, _, published = _wd(brain=SimpleNamespace(release=lambda reason="": True,
                                                 reclaim=lambda: True))
    wd.yield_to_trainer = True
    wd.check(_snap([(7, 30.0, "train.py")]))
    assert [e for e in published if isinstance(e, FaultRaised)] == []


# ------------------------------------------------------------ the card row
def test_the_engine_card_has_a_fault_row():
    from jarvis.ui.reactor import CARD_ROWS
    assert ("FAULT", "fault") in CARD_ROWS


def test_the_window_telemetry_reports_the_fault_token():
    from jarvis.ui.main_window import MainWindow
    win = object.__new__(MainWindow)
    win._asr_text, win._tts_text, win._llm_text, win._dev_text = "S", "E", "L", "D"
    win._faults = FaultBoard()
    assert win._telemetry()["fault"] == CLEAR_TOKEN
    win._ev_fault(FaultRaised(rule="trainers", kind="error", token="2 TRAINERS"))
    assert win._telemetry()["fault"] == "2 TRAINERS"
    # and it outlives ERROR_HOLD_S, which is the whole point
    from jarvis.ui.main_window import ERROR_HOLD_S
    assert ERROR_HOLD_S == 6.0 and win._faults.current is not None


# --------------------------------------------------------- "what's wrong"
@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), memory=MagicMock(),
        context=MagicMock(), tts=MagicMock(), brain=MagicMock(),
        faults=FaultBoard(), log_triage=lambda: ("Nothing in the log, sir.", ""))
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    return Commander(svc)


def test_whats_wrong_is_tier_one():
    assert "whats wrong" in {c.name for c in REGISTRY}
    assert "whats wrong" in {c.name for c in ASSISTANT_TIER1}


@pytest.mark.parametrize("phrase", [
    "what's wrong", "what is wrong", "anything wrong", "what's the matter",
    "is something wrong", "what's wrong with the box",
])
def test_whats_wrong_phrases_match(phrase):
    assert commander._WHATS_WRONG_RX.match(phrase)


def test_whats_wrong_answers_with_the_live_fault(cmdr):
    cmdr.services.faults.apply(FaultRaised(
        rule="trainers", kind="error", token="2 TRAINERS",
        text="2 trainers on the pool", line="Two trainers on the pool, sir."))
    res = cmdr.handle("what's wrong", source="voice")
    assert res.handled and res.speak
    assert res.reply.startswith("Two trainers on the pool, sir.")


def test_whats_wrong_falls_through_to_the_log_triage_when_clear(cmdr):
    res = cmdr.handle("what's wrong", source="voice")
    assert res.reply == "Nothing in the log, sir." and res.status == "Log triage"


def test_whats_wrong_without_a_triage_service_still_answers(cmdr):
    cmdr.services.log_triage = None
    res = cmdr.handle("anything wrong", source="voice")
    assert res.reply == NOTHING_WRONG_LINE


def test_the_log_triage_command_is_not_shadowed(cmdr):
    assert not commander._WHATS_WRONG_RX.match("anything wrong in your log")
    assert commander._LOGTRIAGE_RX.match("anything wrong in your log")
