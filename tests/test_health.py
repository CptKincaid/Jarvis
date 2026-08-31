"""Spark system health (jarvis/tools/health.py): the fact sheet and the
memory watchdog, every probe faked through the module seams -- no
nvidia-smi, no /proc scan, no thread except the one start/stop test.

Covers: nvidia-smi parsing incl. the GB10's [N/A] fields, /proc parsing
against a fake proc tree, the fact sheet text, the tool contract (word
cap, never raises, ok=False + spoken line when nothing is readable),
and the watchdog rules: warn -> critical ladder, once per episode,
re-arm hysteresis at warn+4, straight-to-critical, the two-trainers
rule, unreadable counters, a speak callback that raises, config from a
dict AND the real AssistantConfig, and the daemon thread lifecycle.
"""
import os
import threading
from types import SimpleNamespace

import pytest

from jarvis.assistant_config import AssistantConfig
from jarvis.events import Status
from jarvis.tools import health
from jarvis.tools.health import (CRITICAL_LINE, HOGS_LINE, UNREADABLE_LINE,
                                 WARN_LINE, Proc, Snapshot, Watchdog,
                                 cmdline_hint, fact_sheet, iter_process_rss,
                                 make_tools, parse_meminfo, parse_nvidia_smi,
                                 read_cmdline)
from jarvis.tools.registry import DESCRIPTION_WORD_CAP, ToolRegistry, ToolResult

GB_KB = 1024 * 1024


@pytest.fixture(autouse=True)
def _firewall(monkeypatch):
    """No real probes: every seam raises unless a test replaces it."""
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")

    def _boom(*a, **k):
        raise AssertionError("unit test hit a real probe")
    for name in ("read_meminfo", "read_loadavg", "run_nvidia_smi",
                 "disk_free", "iter_process_rss", "read_cmdline"):
        monkeypatch.setattr(health, name, _boom)
    yield


def _fake_probes(monkeypatch, total_gb=121.7, avail_gb=41.0, load=3.1,
                 smi="42, 2418 MHz, 12.4 W, 2 %", disks=(210.0, 1500.0),
                 procs=(("python", 38.0, ["python", "train.py", "--epochs", "3"]),
                        ("ollama", 22.0, ["/usr/bin/ollama", "runner"]),
                        ("Xorg", 1.2, ["Xorg", ":1"]),
                        ("bash", 0.01, ["bash"]))):
    """Wire the seams to canned values; procs = (name, rss_gb, argv)."""
    monkeypatch.setattr(health, "read_meminfo", lambda: {
        "MemTotal": int(total_gb * GB_KB), "MemFree": 1,
        "MemAvailable": int(avail_gb * GB_KB)})
    monkeypatch.setattr(health, "read_loadavg", lambda: (load, 0.5, 0.3))
    monkeypatch.setattr(health, "run_nvidia_smi", lambda: smi)
    free = dict(zip(("/", "/home"), disks))
    monkeypatch.setattr(health, "disk_free", lambda path: free.get(path))
    table = {100 + i: p for i, p in enumerate(procs)}
    monkeypatch.setattr(health, "iter_process_rss",
                        lambda proc_root=None: ((pid, n, int(g * GB_KB))
                                                for pid, (n, g, _) in table.items()))
    monkeypatch.setattr(health, "read_cmdline",
                        lambda pid, proc_root=None: table[pid][2])


def _snap(avail, procs=(("python", 38.0), ("ollama", 19.0)), total=121.7):
    return Snapshot(mem_total_gb=total, mem_avail_gb=avail, load1=1.0,
                    top=[Proc(pid=i, name=n, rss_gb=g) for i, (n, g) in enumerate(procs)])


class _Sink:
    def __init__(self):
        self.spoken, self.events = [], []

    def speak(self, line):
        self.spoken.append(line)

    def publish(self, ev):
        self.events.append(ev)


# ------------------------------------------------------------ parsing
def test_parse_nvidia_smi_real_row():
    gpu = parse_nvidia_smi("43, 2418 MHz, 12.61 W, 2 %")
    assert gpu == {"temp_c": 43.0, "sm_mhz": 2418.0, "power_w": 12.61, "util_pct": 2.0}


def test_parse_nvidia_smi_na_fields_are_none_not_fatal():
    # The GB10 answers [N/A] for memory and, under a wedged driver, for
    # clocks too; a partial row still yields what it has.
    gpu = parse_nvidia_smi("43, [N/A], N/A, 2 %")
    assert gpu == {"temp_c": 43.0, "sm_mhz": None, "power_w": None, "util_pct": 2.0}
    assert parse_nvidia_smi("[N/A], [N/A], [N/A], [N/A]") is None
    assert parse_nvidia_smi("") is None
    assert parse_nvidia_smi(None) is None
    assert parse_nvidia_smi("NVIDIA-SMI has failed") is None


def test_parse_meminfo():
    mem = parse_meminfo("MemTotal:       127606644 kB\nMemFree:  33603196 kB\n"
                        "MemAvailable:   86291428 kB\nHugePages_Total:       0\n")
    assert mem["MemTotal"] == 127606644
    assert mem["MemAvailable"] == 86291428
    assert mem["HugePages_Total"] == 0


def test_iter_process_rss_reads_a_proc_tree(tmp_path, monkeypatch):
    # The real readers against a fake /proc: a process, a kernel thread
    # without VmRSS, a non-pid entry and a status that vanishes mid-scan.
    monkeypatch.undo()
    (tmp_path / "123").mkdir()
    (tmp_path / "123" / "status").write_text(
        "Name:\tpython3\nUmask:\t0022\nVmPeak:\t 1 kB\nVmRSS:\t 39845888 kB\n")
    (tmp_path / "123" / "cmdline").write_bytes(b"python3\0-m\0jarvis.app\0")
    (tmp_path / "7").mkdir()
    (tmp_path / "7" / "status").write_text("Name:\tkthreadd\n")
    (tmp_path / "self").mkdir()
    (tmp_path / "999").mkdir()                       # no status file at all
    rows = list(iter_process_rss(str(tmp_path)))
    assert rows == [(123, "python3", 39845888)]
    assert read_cmdline(123, str(tmp_path)) == ["python3", "-m", "jarvis.app"]
    assert read_cmdline(999, str(tmp_path)) == []
    assert list(iter_process_rss(str(tmp_path / "missing"))) == []


@pytest.mark.parametrize("argv, hint", [
    (["python", "-m", "jarvis.app"], "jarvis.app"),
    (["/home/x/vss_env/bin/python3.12", "-u", "/home/x/train_piper.py", "--a"], "train_piper.py"),
    (["python", "-c", "import x"], ""),
    (["/usr/bin/ollama", "runner", "--model"], ""),
    (["Xorg", ":1"], ""),
    ([], ""),
])
def test_cmdline_hint(argv, hint):
    assert cmdline_hint(argv) == hint


# --------------------------------------------------------- fact sheet
def test_fact_sheet_text(monkeypatch):
    _fake_probes(monkeypatch)
    snap = health.snapshot()
    assert fact_sheet(snap) == (
        "Memory 41 of 122 GB free; GPU 42 C at 2418 MHz, 12 W, 2% busy; "
        "load 3.1; disk free / 210 GB, /home 1500 GB; "
        "top: python 38 GB (train.py), ollama 22 GB, Xorg 1.2 GB; "
        "note: 2 processes over 20 GB each.")


def test_fact_sheet_notes_tight_memory_and_skips_gpu_na(monkeypatch):
    _fake_probes(monkeypatch, avail_gb=7.6, smi="43, [N/A], [N/A], [N/A]",
                 procs=(("python", 60.0, ["python", "big.py"]),))
    sheet = fact_sheet(health.snapshot())
    assert "Memory 7.6 of 122 GB free" in sheet
    assert "GPU 43 C;" in sheet                     # N/A fields simply absent
    assert "top: python 60 GB (big.py)" in sheet
    assert sheet.endswith("note: memory tight.")


def test_fact_sheet_degrades_per_probe(monkeypatch):
    # nvidia-smi dead, no disks, no loadavg: the sheet still reports memory.
    _fake_probes(monkeypatch, smi=None, disks=(None, None))
    monkeypatch.setattr(health, "read_loadavg", lambda: 1 / 0)
    sheet = fact_sheet(health.snapshot())
    assert sheet.startswith("Memory 41 of 122 GB free; GPU unavailable; load unknown; "
                            "disk unknown; top:")


# --------------------------------------------------------------- tool
def test_tool_contract(monkeypatch):
    _fake_probes(monkeypatch)
    services = SimpleNamespace(speak=lambda line: None)
    specs = make_tools({}, services)
    assert [s.name for s in specs] == ["system_health"]
    spec = specs[0]
    assert len(spec.description.split()) <= DESCRIPTION_WORD_CAP
    reg = ToolRegistry()
    reg.register_many(specs)
    assert reg.budget()["ok"]
    res = reg.call("system_health", {"unexpected": 1})
    assert isinstance(res, ToolResult) and res.ok
    assert res.max_sentences == 3
    assert res.text.startswith("Memory 41 of 122 GB free")
    assert res.speak is None
    # The watchdog is parked on services, not started.
    assert isinstance(services.health_watchdog, Watchdog)
    assert not services.health_watchdog.running


def test_tool_reports_the_gpu_it_probed_and_the_watchdog_never_asks(monkeypatch):
    """06d6e18 switched the TOOL to snapshot(gpu=False) along with the
    watchdog, so "how's the Spark doing" always said 'GPU unavailable'.
    The tool may probe (Popen + kill-without-wait, bounded); only the
    30 s watchdog tick must not."""
    _fake_probes(monkeypatch)
    probes = []
    monkeypatch.setattr(health, "run_nvidia_smi",
                        lambda: probes.append(1) or "42, 2418 MHz, 12.4 W, 2 %")
    res = make_tools({}, None)[0].handler()
    assert res.ok and probes == [1]
    assert "GPU 42 C at 2418 MHz, 12 W, 2% busy" in res.text
    Watchdog({}, services=None).tick()
    assert probes == [1], "the watchdog never runs nvidia-smi"


def test_tool_unreadable_counters_speak_the_excuse(monkeypatch):
    # Every seam raising (the autouse firewall) must not escape the handler.
    monkeypatch.setattr(health, "disk_free", lambda path: None)
    res = make_tools(None, None)[0].handler()
    assert res.ok is False
    assert res.speak == UNREADABLE_LINE
    assert res.speak.endswith(", sir.")


def test_tool_survives_services_without_setattr(monkeypatch):
    _fake_probes(monkeypatch)
    specs = make_tools({}, object())          # cannot park the watchdog
    assert specs[0].handler().ok


# ----------------------------------------------------------- watchdog
def test_watchdog_ladder_once_per_episode_and_rearm():
    sink = _Sink()
    wd = Watchdog({}, speak=sink.speak, publish=sink.publish, interval=0)
    assert wd.warn_gb == 16 and wd.critical_gb == 8
    assert wd.check(_snap(41)) == []
    fired = wd.check(_snap(12.4))
    assert [a.kind for a in fired] == ["warn"]
    assert sink.spoken == ["Memory is getting tight, sir: 12 gigabytes free, "
                           "with python at 38 gigabytes and ollama at 19."]
    assert sink.spoken[0] == WARN_LINE.format(
        free="12", hogs=", with python at 38 gigabytes and ollama at 19")
    assert [(e.kind, e.text) for e in sink.events if isinstance(e, Status)] == \
        [("warn", "Memory tight: 12 GB free")]
    # Still tight: silence.
    assert wd.check(_snap(11)) == [] and wd.check(_snap(15.9)) == []
    # Critical: one error.
    fired = wd.check(_snap(7.4))
    assert [a.kind for a in fired] == ["error"]
    assert sink.spoken[-1] == CRITICAL_LINE.format(
        free="7", hogs=", with python at 38 gigabytes and ollama at 19")
    assert sink.events[-1].kind == "error"
    assert sink.events[-1].text == "Memory critical: 7.4 GB free"
    assert wd.check(_snap(6)) == []
    # Back into the warn band and even just under warn+4: the episode
    # stays open, nothing re-fires.
    assert wd.check(_snap(12)) == [] and wd.check(_snap(19.9)) == []
    assert len(sink.spoken) == 2
    # Above warn + 4: re-armed (a quiet "ok" chip, nothing spoken).
    assert wd.check(_snap(20)) == []
    assert sink.events[-1].kind == "ok" and "recovered" in sink.events[-1].text
    assert len(sink.spoken) == 2
    fired = wd.check(_snap(12))
    assert [a.kind for a in fired] == ["warn"]
    assert len(sink.spoken) == 3


def test_watchdog_straight_to_critical_is_one_alert():
    sink = _Sink()
    wd = Watchdog({}, speak=sink.speak, publish=sink.publish)
    assert wd.check(_snap(60)) == []
    fired = wd.check(_snap(5))
    assert [a.kind for a in fired] == ["error"]
    assert len(sink.spoken) == 1
    assert wd.check(_snap(5)) == [] and wd.check(_snap(12)) == []


def test_watchdog_spoken_line_without_hogs():
    sink = _Sink()
    wd = Watchdog({}, speak=sink.speak, publish=sink.publish)
    wd.check(_snap(9, procs=(("Xorg", 3.2),)))
    assert sink.spoken == ["Memory is getting tight, sir: 9 gigabytes free, "
                           "with Xorg at 3 gigabytes."]
    wd2 = Watchdog({}, speak=sink.speak, publish=sink.publish)
    wd2.check(_snap(9, procs=()))
    assert sink.spoken[-1] == "Memory is getting tight, sir: 9 gigabytes free."


def test_watchdog_two_trainers_rule():
    sink = _Sink()
    wd = Watchdog({}, speak=sink.speak, publish=sink.publish)
    # One hog, plenty of memory: nothing.
    assert wd.check(_snap(70, procs=(("python", 60.0), ("ollama", 12.0)))) == []
    # Two over 20 GB each with memory still fine: warn ONCE.
    fired = wd.check(_snap(50, procs=(("python", 38.0), ("python", 22.5), ("ollama", 9.0))))
    assert [(a.kind, a.rule) for a in fired] == [("warn", "hogs")]
    assert sink.spoken == [HOGS_LINE.format(hog="20", hogs="python at 38 gigabytes and python at 22")]
    assert sink.events[-1].kind == "warn"
    assert sink.events[-1].text == "2 processes over 20 GB"
    assert wd.check(_snap(50, procs=(("python", 38.0), ("python", 22.5)))) == []
    # One of them exits: re-armed; both back: fires again.
    assert wd.check(_snap(70, procs=(("python", 38.0), ("ollama", 12.0)))) == []
    fired = wd.check(_snap(50, procs=(("python", 38.0), ("python", 22.5))))
    assert [a.rule for a in fired] == ["hogs"]
    assert len(sink.spoken) == 2


def test_watchdog_memory_and_hogs_fire_together_as_two_alerts():
    sink = _Sink()
    wd = Watchdog({}, speak=sink.speak, publish=sink.publish)
    fired = wd.check(_snap(12, procs=(("python", 38.0), ("python", 22.0))))
    assert [a.rule for a in fired] == ["memory", "hogs"]
    assert len(sink.spoken) == 2


def test_watchdog_config_from_dict_and_assistant_config():
    wd = Watchdog({"health": {"warn_gb": 30, "critical_gb": 20, "hog_gb": 10,
                              "interval_s": 5}})
    assert (wd.warn_gb, wd.critical_gb, wd.hog_gb, wd.interval) == (30, 20, 10, 5)
    cfg = AssistantConfig({"health": {"warn_gb": "24", "critical_gb": 40}})
    wd = Watchdog(cfg)
    assert wd.warn_gb == 24
    assert wd.critical_gb == 24          # clamped: critical never above warn
    assert wd.hog_gb == 20 and wd.interval == 30
    # Junk values fall back to the defaults rather than raising.
    wd = Watchdog({"health": {"warn_gb": "lots"}})
    assert wd.warn_gb == 16


def test_watchdog_unreadable_memory_is_idle_not_fatal():
    sink = _Sink()
    wd = Watchdog({}, speak=sink.speak, publish=sink.publish)
    for _ in range(3):
        assert wd.check(Snapshot()) == []
    assert sink.spoken == [] and sink.events == []
    # Counters coming back: the rules resume.
    assert [a.kind for a in wd.check(_snap(5))] == ["error"]


def test_watchdog_speak_and_publish_failures_do_not_escape():
    def bad_speak(line):
        raise RuntimeError("tts down")

    def bad_publish(ev):
        raise RuntimeError("bus down")
    wd = Watchdog({}, speak=bad_speak, publish=bad_publish)
    assert [a.kind for a in wd.check(_snap(5))] == ["error"]


def test_watchdog_resolves_services_speak_at_fire_time():
    # Built before services.speak exists (make_tools runs before the TTS
    # is wired); the callback is looked up when an alert fires.
    services = SimpleNamespace()
    sink = _Sink()
    wd = Watchdog({}, services=services, publish=sink.publish)
    assert [a.kind for a in wd.check(_snap(5))] == ["error"]
    services.speak = sink.speak
    wd.check(_snap(40))                  # re-arm
    wd.check(_snap(5))
    assert len(sink.spoken) == 1 and sink.spoken[0].startswith("Memory is critical, sir")


def test_watchdog_tick_survives_a_broken_probe(monkeypatch):
    # The autouse firewall makes every seam raise; tick() must not.
    monkeypatch.setattr(health, "disk_free", lambda path: None)
    wd = Watchdog({}, publish=lambda ev: None)
    assert wd.tick() == []
    assert wd.last is not None and wd.last.mem_avail_gb is None


def test_watchdog_thread_start_stop(monkeypatch):
    ticked = threading.Event()

    def meminfo():
        ticked.set()
        return {"MemTotal": 121 * GB_KB, "MemAvailable": 40 * GB_KB}
    _fake_probes(monkeypatch, procs=(("python", 3.0, ["python", "-m", "jarvis.app"]),))
    monkeypatch.setattr(health, "read_meminfo", meminfo)
    sink = _Sink()
    wd = Watchdog({}, speak=sink.speak, publish=sink.publish, interval=0.01)
    wd.start()
    thread = wd._thread
    assert thread.daemon and thread.name == "health-watchdog"
    wd.start()                           # idempotent: same thread
    assert wd._thread is thread
    assert ticked.wait(2.0)
    wd.stop()
    assert not wd.running
    assert sink.spoken == []             # 40 GB free: nothing to say
    assert wd.last.mem_avail_gb == 40


# ------------------------------------------------- GPU claimant detector
DIGEST_ARGV = ["/usr/bin/python3", "-u", "digest_llm.py",
               "--cache", "./digest_cache.json",
               "--forums", "./digest_forums.json",
               "--override-window", "7", "--out", "./themes_weekly.json",
               "--no-push"]


def test_tick_hands_the_configured_claimants_to_the_snapshot(monkeypatch):
    """The live path, end to end: the 30 s tick must carry health.yield_to
    into snapshot(), or the digest is invisible where it matters (it was:
    ~50 minutes starved behind a pinned gemma4:26b on 2026-08-30)."""
    _fake_probes(monkeypatch, procs=(("python3", 9.0, DIGEST_ARGV),))
    seen = []

    class FakeBrain:
        def release(self, reason=""):
            seen.append(reason)
            return True

        def reclaim(self):
            return True

    wd = Watchdog(AssistantConfig({}), publish=lambda ev: None,
                  brain=FakeBrain())
    wd.tick()
    assert wd.last is not None
    assert [(p.claim, p.hint) for p in wd.last.claimants] == \
        [("digest_llm", "digest_llm.py")]
    assert wd.last.trainers == []          # a digest is not a training run
    assert seen == ["digest_llm pid 100 digest_llm.py"]
    assert wd.lent_to == 100


def test_claimant_scan_reads_only_candidate_cmdlines(monkeypatch):
    """A named claimant may be a compiled job with no interpreter, but the
    scan must still stay a few dozen small reads, not the whole table."""
    table = {10: ("python3", 9.0, DIGEST_ARGV),
             11: ("render_job", 4.0, ["/opt/render_job", "--gpu"]),
             12: ("Xorg", 1.2, ["Xorg", ":1"]),
             13: ("chrome", 8.0, ["/opt/chrome"])}
    read = []
    monkeypatch.setattr(health, "iter_process_rss",
                        lambda proc_root=None: ((pid, n, int(g * GB_KB))
                                                for pid, (n, g, _) in table.items()))

    def cmdline(pid, proc_root=None):
        read.append(pid)
        return table[pid][2]
    monkeypatch.setattr(health, "read_cmdline", cmdline)
    got = health.find_claimants(names=("digest_llm", "render_job"))
    assert [(p.pid, p.claim, p.hint) for p in got] == \
        [(10, "digest_llm", "digest_llm.py"),
         (11, "render_job", "render_job")]      # not "--gpu", the last argv
    assert sorted(read) == [10, 11]      # never Xorg / chrome


def test_yield_to_survives_a_malformed_config_value():
    # A hand-edited assistant.json must never take the watchdog thread down.
    assert health.claimant_names(7) == ()
    assert health.claimant_names({"digest_llm": True}) == ("digest_llm",)
    wd = Watchdog({"health": {"yield_to": 7}}, publish=lambda ev: None)
    assert wd.yield_to == ()
