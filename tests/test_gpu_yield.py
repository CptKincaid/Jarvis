"""GPU yield-and-reclaim: lend the local model to a trainer, take it back.

Three layers, each against fakes:
- jarvis.brain release()/reclaim(): the keep_alive-0 unload through the
  _http seam (FakeOllama from tests/test_brain_tools.py), the lent flag
  that turns every local door into the MODEL_LENT_LINE before any HTTP,
  keep_alive 0 on any payload built while lent, residency that stops
  re-warming, and the persona helpers falling back to their text.
- jarvis.tools.health.Watchdog: the trainer rule as a state machine driven
  through check() with a FakeBrain -- lend once when a trainer appears,
  hold while it runs, reclaim after TRAINER_ABSENT_TICKS ticks without it,
  the manual "take the GPU back" hold, and the opt-in default (off).
- jarvis.commander: "take the GPU back" / "lend the GPU" at Tier 1.
"""
import types
from unittest.mock import MagicMock

import pytest

import jarvis.commander as commander
from jarvis.commander import Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.tools import health
from jarvis.tools.health import (LENT_LINE, RECLAIMED_LINE, Proc, Snapshot,
                                 Watchdog, find_trainers, is_trainer)
from tests.test_brain_tools import brain, setup, text_reply  # noqa: F401

GB_KB = 1024 * 1024


# =============================================================== brain
@pytest.fixture
def lent_off(brain, monkeypatch):  # noqa: F811
    monkeypatch.setitem(brain._RESIDENCY, "lent", False)
    monkeypatch.setitem(brain._RESIDENCY, "unloaded_once", True)


def test_release_unloads_with_keep_alive_zero_and_marks_lent(brain, setup, lent_off):  # noqa: F811
    b, fake, _ = setup
    assert not brain.is_lent()
    assert brain.release(reason="trainer pid 7") is True
    assert brain.is_lent() and b.is_lent()
    assert fake.calls == [("/api/generate",
                           {"model": brain.OLLAMA_MODEL, "keep_alive": 0}, 30)]
    # idempotent: a second release costs nothing
    assert brain.release() is True and len(fake.calls) == 1


def test_release_marks_lent_even_when_ollama_is_down(brain, setup, lent_off):  # noqa: F811
    b, fake, _ = setup
    fake.fail = brain.OllamaDown("refused")
    assert brain.release() is False
    assert brain.is_lent()                # nothing must reload it later
    fake.fail = TimeoutError("slow")
    monkeypatch_lent = brain._RESIDENCY
    monkeypatch_lent["lent"] = False
    assert brain.release() is False and brain.is_lent()


def test_a_local_turn_while_lent_is_the_excuse_with_no_http(brain, setup, lent_off):  # noqa: F811
    b, fake, record = setup
    brain.release()
    fake.calls.clear()
    fake.replies = [text_reply("should never be asked")]
    assert b._chat_sync("what's the weather like?") == \
        [("SPEAK", brain.MODEL_LENT_LINE)]
    # a forced tool (the "last mail" / briefing doors) does not run either:
    # its render turn would have loaded the model
    assert b._chat_sync("any mail?", force_tool="get_weather") == \
        [("SPEAK", brain.MODEL_LENT_LINE)]
    assert fake.calls == [] and record == []
    # the streamed path goes through the same gate
    spoken = []
    assert b._chat_sync("hello", on_sentence=spoken.append) == \
        [("SPEAK", brain.MODEL_LENT_LINE)]
    assert spoken == [] and fake.calls == []


def test_any_payload_built_while_lent_says_unload_after(brain, setup, lent_off):  # noqa: F811
    assert brain._chat_payload([])["keep_alive"] == -1
    brain.release()
    assert brain._chat_payload([])["keep_alive"] == 0
    brain.reclaim()
    assert brain._chat_payload([])["keep_alive"] == -1


def test_residency_never_rewarms_while_lent(brain, setup, lent_off):  # noqa: F811
    b, fake, _ = setup
    brain.release()
    fake.calls.clear()
    fake.ps = {"models": []}
    assert brain.ensure_resident() is False
    assert brain.ensure_resident(first=True) is False
    assert fake.calls == []               # not even /api/ps


def test_persona_helpers_and_the_route_classifier_fall_back_while_lent(brain, setup, lent_off):  # noqa: F811
    b, fake, _ = setup
    brain.release()
    fake.calls.clear()
    fake.replies = [text_reply("never")]
    assert brain.summarize("Three commits today. One file dirty.", 2) == \
        "Three commits today. One file dirty."
    assert brain.local_line("say hi", "x", fallback="fb") == "fb"
    assert brain.classify_route("what's the capital of peru") == ("local", 0.0)
    assert fake.calls == []


def test_reclaim_clears_the_flag_and_warms_the_model(brain, setup, lent_off):  # noqa: F811
    b, fake, _ = setup
    brain.release()
    fake.calls.clear()
    fake.ps = {"models": []}
    fake.replies = [text_reply("", load_duration=6_900_000_000)]
    assert b.reclaim() is True
    assert not brain.is_lent()
    paths = [path for path, _, _ in fake.calls]
    assert paths == ["/api/ps", "/api/chat"]
    warm = fake.calls[1][1]
    assert warm["keep_alive"] == -1 and warm["options"]["num_predict"] == 1
    # the model answers again
    fake.replies = [text_reply("Back with you, sir.")]
    assert b._chat_sync("you there?") == [("SPEAK", "Back with you, sir.")]


def test_reclaim_when_ollama_is_down_reports_false_but_unlends(brain, setup, lent_off):  # noqa: F811
    b, fake, _ = setup
    brain.release()
    fake.fail = brain.OllamaDown("refused")
    assert brain.reclaim() is False
    assert not brain.is_lent()
    # not lent: reclaim is just a residency check
    fake.fail = None
    fake.ps = {"models": [{"name": brain.OLLAMA_MODEL}]}
    assert brain.reclaim() is True


# ============================================================ watchdog
class FakeBrain:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok
        self.lent = False

    def release(self, reason=""):
        self.calls.append(("release", reason))
        self.lent = True
        return self.ok

    def reclaim(self):
        self.calls.append(("reclaim",))
        self.lent = False
        return self.ok

    def is_lent(self):
        return self.lent


def _snap(trainers=(), avail=60.0):
    return Snapshot(mem_total_gb=121.7, mem_avail_gb=avail, load1=1.0,
                    top=[Proc(pid=1, name="ollama", rss_gb=22.0)],
                    trainers=[Proc(pid=pid, name="python", rss_gb=gb, hint=hint)
                              for pid, gb, hint in trainers])


def _wd(cfg=None, brain_obj=None, **kw):
    spoken, published = [], []
    wd = Watchdog(cfg if cfg is not None else {"health": {"yield_to_trainer": True}},
                  speak=spoken.append, publish=published.append,
                  brain=brain_obj or FakeBrain(), **kw)
    return wd, spoken, published


@pytest.mark.parametrize("argv,expected", [
    (["python", "train.py", "--epochs", "3"], True),
    (["/home/h/vss_env/bin/python3", "-m", "aiws_trainer.train"], True),
    (["python", "finetune_piper.py"], True),
    (["python", "scripts/fine_tune.py"], True),
    (["torchrun", "--nproc_per_node", "1", "train_yolo.py"], True),
    (["python", "-m", "jarvis.app"], False),
    (["python", "constraint.py"], False),      # "train" inside a word
    (["python", "-c", "import torch"], False),
    (["/usr/bin/ollama", "runner"], False),
    ([], False),
])
def test_is_trainer(argv, expected):
    assert is_trainer(argv) is expected


def test_find_trainers_reads_cmdlines_of_interpreters_only(monkeypatch):
    table = {100: ("python", 38.0, ["python", "train.py"]),
             101: ("ollama", 22.0, ["/usr/bin/ollama", "runner"]),
             102: ("python3", 2.5, ["python3", "-m", "aiws_trainer.train"]),
             103: ("python", 1.0, ["python", "-m", "jarvis.app"]),
             104: ("Xorg", 1.2, ["Xorg", ":1"]),
             105: ("python", 0.1, ["python", "train.py"])}     # our own pid
    read = []
    monkeypatch.setattr(health, "iter_process_rss",
                        lambda proc_root=None: ((pid, n, int(g * GB_KB))
                                                for pid, (n, g, _) in table.items()))

    def cmdline(pid, proc_root=None):
        read.append(pid)
        return table[pid][2]
    monkeypatch.setattr(health, "read_cmdline", cmdline)
    got = find_trainers(self_pid=105)
    assert [(p.pid, p.hint) for p in got] == [(100, "train.py"),
                                              (102, "aiws_trainer.train")]
    assert got[0].rss_gb == pytest.approx(38.0)
    assert sorted(read) == [100, 102, 103]          # never ollama / Xorg / self


def test_watchdog_lends_once_holds_and_reclaims_after_two_absent_ticks():
    wd, spoken, published = _wd()
    fb = wd._brain_obj
    assert wd.check(_snap()) == []                    # nothing running
    fired = wd.check(_snap([(4242, 30.0, "train.py")]))
    assert [(a.rule, a.kind) for a in fired] == [("trainer", "warn")]
    assert fired[0].line == LENT_LINE and spoken == [LENT_LINE]
    assert fired[0].status == "GPU lent to train.py"
    assert fb.calls == [("release", "trainer pid 4242 train.py")]
    assert wd.lent_to == 4242
    # the run continues: nothing more is said, nothing more is called
    assert wd.check(_snap([(4242, 31.0, "train.py")])) == []
    assert wd.check(_snap([(4242, 31.0, "train.py")])) == []
    # gone for one tick: still lent (an epoch restart must not cost a reload)
    assert wd.check(_snap()) == [] and wd.lent_to == 4242
    # back for a tick: the absence counter starts over
    assert wd.check(_snap([(4242, 31.0, "train.py")])) == []
    assert wd.check(_snap()) == []
    fired = wd.check(_snap())
    assert [(a.rule, a.kind, a.line) for a in fired] == \
        [("trainer", "ok", RECLAIMED_LINE)]
    assert fb.calls[-1] == ("reclaim",) and wd.lent_to is None
    assert spoken == [LENT_LINE, RECLAIMED_LINE]
    assert [e.text for e in published] == ["GPU lent to train.py", "GPU reclaimed"]
    # a new run later lends again
    assert len(wd.check(_snap([(5000, 2.0, "finetune.py")]))) == 1
    assert wd.lent_to == 5000


def test_watchdog_follows_a_replacement_trainer_without_reloading():
    wd, spoken, _ = _wd()
    wd.check(_snap([(1, 30.0, "train.py")]))
    # one run ends and the next starts inside the same tick
    assert wd.check(_snap([(2, 5.0, "fine_tune.py")])) == []
    assert wd.lent_to == 2 and spoken == [LENT_LINE]
    assert wd._brain_obj.calls == [("release", "trainer pid 1 train.py")]


def test_watchdog_manual_reclaim_holds_off_the_running_trainer():
    wd, spoken, _ = _wd()
    fb = wd._brain_obj
    wd.check(_snap([(7, 30.0, "train.py")]))
    assert wd.manual_reclaim() is True
    assert fb.calls[-1] == ("reclaim",) and wd.lent_to is None
    # the same run is still going: the rule must not undo the order
    assert wd.check(_snap([(7, 30.0, "train.py")])) == []
    assert wd.check(_snap([(7, 30.0, "train.py")])) == []
    assert fb.calls == [("release", "trainer pid 7 train.py"), ("reclaim",)]
    # a different trainer alongside it IS lent to
    fired = wd.check(_snap([(7, 30.0, "train.py"), (8, 3.0, "finetune.py")]))
    assert len(fired) == 1 and wd.lent_to == 8
    # once the held run is gone and a new one appears later, the hold is over
    wd.check(_snap([(8, 3.0, "finetune.py")]))
    wd.manual_reclaim()
    assert wd.check(_snap()) == [] and wd.check(_snap()) == []
    assert len(wd.check(_snap([(7, 1.0, "train.py")]))) == 1   # pid reused: lent again
    # manual reclaim when nothing is lent is just a warm-up
    wd2, _, _ = _wd()
    assert wd2.manual_reclaim() is True and wd2._brain_obj.calls == [("reclaim",)]


def test_watchdog_yield_is_off_by_default_and_memory_rules_still_run():
    from jarvis.assistant_config import AssistantConfig
    wd, spoken, _ = _wd(cfg=AssistantConfig({}))
    assert wd.yield_to_trainer is False
    fired = wd.check(_snap([(4242, 30.0, "train.py")], avail=12.0))
    assert [a.rule for a in fired] == ["memory"]
    assert wd._brain_obj.calls == [] and wd.lent_to is None
    on = Watchdog(AssistantConfig({"health": {"yield_to_trainer": True}}),
                  brain=FakeBrain())
    assert on.yield_to_trainer is True


def test_watchdog_reports_a_failed_unload_and_a_failed_reload():
    wd, spoken, published = _wd(brain_obj=FakeBrain(ok=False))
    fired = wd.check(_snap([(1, 30.0, "train.py")]))
    assert fired[0].status == "GPU lent to train.py (unload failed)"
    assert wd.lent_to == 1                       # still lent: nothing reloads it
    wd.check(_snap()), wd.check(_snap())
    assert published[-1].kind == "warn" and "failed to load" in published[-1].text

    class Boom(FakeBrain):
        def release(self, reason=""):
            raise RuntimeError("no ollama")
    wd, spoken, _ = _wd(brain_obj=Boom())
    assert wd.check(_snap([(1, 30.0, "train.py")])) == []
    assert wd.lent_to is None and spoken == []     # try again next tick


def test_snapshot_carries_the_trainers_it_found(monkeypatch):
    from tests.test_health import _fake_probes
    _fake_probes(monkeypatch)
    snap = health.snapshot(gpu=False)
    assert [p.hint for p in snap.trainers] == ["train.py"]
    assert health.fact_sheet(snap).startswith("Memory 41 of 122 GB free")


def test_watchdog_resolves_the_real_brain_module_lazily(monkeypatch):
    wd = Watchdog({"health": {"yield_to_trainer": True}})
    import jarvis.brain as brain_mod
    assert wd._brain() is brain_mod


# =========================================================== commander
@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = types.SimpleNamespace(desktop=MagicMock(), workflows=MagicMock(),
                                memory=MagicMock(), context=MagicMock(),
                                tts=MagicMock(), brain=FakeBrain())
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    return Commander(svc)


@pytest.mark.parametrize("text", ["take the GPU back", "jarvis, take the gpu back",
                                  "take back the gpu", "reclaim the gpu",
                                  "load your model back", "get your brain back"])
def test_take_the_gpu_back(cmdr, text):
    cmdr.services.brain.lent = True
    res = cmdr.handle(text, source="typed")
    assert res.reply == commander.GPU_RECLAIMED_LINE and res.speak
    assert cmdr.services.brain.calls == [("reclaim",)]


def test_take_the_gpu_back_prefers_the_watchdog_hold(cmdr):
    wd = FakeBrain()               # anything with manual_reclaim()
    wd.manual_reclaim = lambda: wd.calls.append(("manual",)) or True
    cmdr.services.health_watchdog = wd
    cmdr.services.brain.lent = True
    # addressed: unprefixed voice goes through the intent classifier first
    res = cmdr.handle("jarvis, take the gpu back", source="voice")
    assert res.reply == commander.GPU_RECLAIMED_LINE
    assert wd.calls == [("manual",)] and cmdr.services.brain.calls == []


def test_take_the_gpu_back_when_it_was_never_lent(cmdr):
    res = cmdr.handle("take the gpu back", source="typed")
    assert res.reply == commander.GPU_NOT_LENT_LINE and res.speak
    assert cmdr.services.brain.calls == []


def test_reclaim_failure_is_spoken_after_the_acknowledgement(cmdr):
    cmdr.services.brain.lent = True
    cmdr.services.brain.ok = False
    res = cmdr.handle("take the gpu back", source="typed")
    assert res.reply == commander.GPU_RECLAIMED_LINE
    cmdr.services.tts.speak.assert_called_once_with(commander.GPU_RECLAIM_FAILED_LINE)


@pytest.mark.parametrize("text", ["lend the GPU", "lend the gpu to the trainer",
                                  "release the gpu", "free up the gpu",
                                  "unload your model"])
def test_lend_the_gpu(cmdr, text):
    res = cmdr.handle(text, source="typed")
    assert res.reply == commander.GPU_LENT_LINE and res.speak
    assert cmdr.services.brain.calls == [("release", "")]
    again = cmdr.handle("lend the gpu", source="typed")
    assert again.reply == commander.GPU_ALREADY_LENT_LINE
    assert len(cmdr.services.brain.calls) == 1


def test_gpu_commands_are_tier_one_and_need_the_brain_doors(cmdr):
    names = [c.name for c in commander.ASSISTANT_TIER1]
    assert "gpu reclaim" in names and "gpu lend" in names
    cmdr.services.brain = MagicMock(spec=["chat"])     # no release/reclaim
    res = cmdr.handle("take the gpu back", source="typed")
    assert res.reply != commander.GPU_RECLAIMED_LINE


# ------------------------------------------------- the lend/pin race (#12)
def test_unpin_reunloads_only_a_pinning_payload_that_raced_the_lend(brain, setup, lent_off):  # noqa: F811
    b, fake, _ = setup
    brain._RESIDENCY["lent"] = True
    brain._unpin_if_lent({"keep_alive": 0})          # built while lent: fine
    assert fake.calls == []
    brain._unpin_if_lent({"keep_alive": -1})         # built before, landed after
    assert fake.calls == [("/api/generate",
                           {"model": brain.OLLAMA_MODEL, "keep_alive": 0}, 30)]
    brain._RESIDENCY["lent"] = False
    brain._unpin_if_lent({"keep_alive": -1})         # not lent: nothing to do
    assert len(fake.calls) == 1


def test_unpin_swallows_an_unreachable_ollama(brain, setup, lent_off):  # noqa: F811
    b, fake, _ = setup
    brain._RESIDENCY["lent"] = True
    fake.fail = brain.OllamaDown("refused")
    brain._unpin_if_lent({"keep_alive": -1})         # must not raise


def test_the_residency_warm_up_compensates_when_release_races_it(brain, setup, lent_off, monkeypatch):  # noqa: F811
    """release() landing during the 300 s warm chat: the warm request pinned
    the model back. ensure_resident must hand its payload to the unpin."""
    b, fake, _ = setup
    fake.replies = [{"message": {"role": "assistant", "content": ""},
                     "done": True, "load_duration": 0}]
    seen = []
    monkeypatch.setattr(brain, "_unpin_if_lent", lambda payload: seen.append(payload))
    assert brain.ensure_resident(first=False) is True
    assert len(seen) == 1 and seen[0]["keep_alive"] == -1, \
        "the warm payload must reach the compensating unload"
