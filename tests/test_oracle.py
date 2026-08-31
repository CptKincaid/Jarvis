"""The Oracle Cloud lane (2026-08-31): jarvis/tools/oracle.py + its three
Tier-1 commands.

NOTHING HERE OPENS A SOCKET. ``oracle.run_ssh`` is the one seam every path
goes through, and every test either replaces it or replaces ``subprocess.Popen``
underneath it -- the unconfigured test asserts that a Popen at all is the bug.

Covered: the unconfigured path (off, no key, missing key, world-readable
key), the timeout path, a successful parse and its spoken line, the
allow-list refusing an action that is not a row, and the read-back a restart
has to survive before anything runs.
"""
import json
import os
import pathlib
import re
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jarvis import commander as cmd_mod
from jarvis.commander import Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.router import Router
from jarvis.tools import oracle


# ------------------------------------------------------------------ fakes
class Cfg:
    """The AssistantConfig surface oracle.read_config uses: dotted get()."""

    def __init__(self, **over):
        self.data = {
            "claude.big_model": "fable", "briefing.enabled": False,
            "oracle.enabled": True, "oracle.host": "170.9.245.136",
            "oracle.user": "opc", "oracle.key_path": "", "oracle.timeout_s": 6,
            "oracle.cache_s": 25,
            "oracle.actions": {
                "status": "pm2 status --no-color",
                "logs": "pm2 logs --lines 20 --nostream --no-color",
                "restart the bot": "pm2 restart game-news",
            },
        }
        self.data.update(over)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setup_line(self, section):
        return f"I'll need {section} set up, sir."


PM2_ONE = json.dumps([{
    "name": "game-news",
    "pm2_env": {"status": "online", "restart_time": 2,
                "pm_uptime": (time.time() - 3 * 86400) * 1000},
    "monit": {"cpu": 0.4, "memory": 67 * 1024 * 1024},
}])

GOOD_PAYLOAD = (
    "#uptime\n3456789.12 3400000.00\n"
    "#load\n0.08 0.05 0.01 1/120 4242\n"
    "#mem\nMemTotal:        1009884 kB\nMemFree:          104000 kB\n"
    "MemAvailable:     512000 kB\n"
    "#disk\nFilesystem 1024-blocks Used Available Capacity Mounted on\n"
    "/dev/sda1 46000000 20000000 25000000 45% /\n"
    "#pm2\n" + PM2_ONE + "\n")


@pytest.fixture(autouse=True)
def _clean_cache():
    oracle.clear_cache()
    yield
    oracle.clear_cache()


@pytest.fixture
def key(tmp_path):
    """A 0600 key file, so missing_reason() has nothing left to complain of."""
    path = tmp_path / "oracle-key"
    path.write_text("-----BEGIN RSA PRIVATE KEY-----\n")
    os.chmod(path, 0o600)
    return path


@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent.json")
    monkeypatch.setattr(Commander, "FEEDBACK_LOG", tmp_path / "feedback.jsonl",
                        raising=False)
    for name, val in (("voice_cmds", True), ("jarvis_mode", True),
                      ("talkback", False), ("auto_type", False)):
        monkeypatch.setattr(CONFIG, name, val)
    # _bg runs inline: the handlers hand their work to a thread, and a test
    # that has to sleep for one is a test that flakes.
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())

    def _make(**over):
        svc = SimpleNamespace(
            desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
            memory=MagicMock(), context=MagicMock(), tts=MagicMock(),
            assistant=Cfg(**over), timekeeper=MagicMock(), notes=MagicMock(),
            approvals=MagicMock(), claude=MagicMock())
        svc.desktop.parse_action = lambda part: None
        svc.workflows.get.return_value = None
        svc.context.answer_question.return_value = None
        svc.memory.suggest_by_habit.return_value = None
        svc.timekeeper.ringing = None
        svc.approvals.pending.return_value = []
        svc.brain.local_line.return_value = "Right away, sir."
        svc.router = Router(svc.assistant, classify=lambda t: ("local", 0.2))
        return Commander(svc), svc

    return _make


def fake_ssh(*results):
    """A run_ssh replacement that records its calls and returns canned
    results in order (the last one repeats)."""
    calls = []
    queue = list(results)

    def _run(conf, command):
        calls.append((conf, command))
        return queue.pop(0) if len(queue) > 1 else queue[0]

    _run.calls = calls
    return _run


# ------------------------------------------------- unconfigured, and safe
def test_the_default_config_is_off_and_keyless():
    from jarvis.assistant_config import DEFAULTS
    section = DEFAULTS["oracle"]
    assert section["enabled"] is False and section["key_path"] == ""
    assert section["host"] and section["user"] == "opc"
    assert set(section["actions"]) == {"status", "logs", "restart the bot"}


@pytest.mark.parametrize("over,fragment", [
    ({"oracle.enabled": False}, "oracle.enabled"),
    ({"oracle.host": ""}, "oracle.host"),
    ({"oracle.key_path": ""}, "oracle.key_path"),
])
def test_missing_reason_names_exactly_what_is_missing(over, fragment):
    conf = oracle.read_config(Cfg(**over))
    reason = oracle.missing_reason(conf)
    assert reason and fragment in reason


def test_a_key_path_that_is_not_there_says_so(tmp_path):
    conf = oracle.read_config(Cfg(**{"oracle.key_path": str(tmp_path / "nope")}))
    assert "can't find the Oracle key" in oracle.missing_reason(conf)


def test_a_world_readable_key_is_named_before_ssh_ever_refuses_it(tmp_path):
    path = tmp_path / "oracle-key"
    path.write_text("x")
    os.chmod(path, 0o644)
    conf = oracle.read_config(Cfg(**{"oracle.key_path": str(path)}))
    assert "chmod 600" in oracle.missing_reason(conf)


def test_a_configured_lane_has_nothing_to_complain_about(key):
    conf = oracle.read_config(Cfg(**{"oracle.key_path": str(key)}))
    assert oracle.missing_reason(conf) is None


def test_the_unconfigured_command_answers_one_line_and_opens_no_socket(
        cmdr, monkeypatch):
    """The whole point of the default: no key, no socket, one honest line."""
    def _boom(*a, **k):                       # any Popen at all is the bug
        raise AssertionError("oracle opened a subprocess while unconfigured")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    c, _ = cmdr(**{"oracle.key_path": ""})
    res = c.handle("how's the oracle box", source="typed")
    assert res.handled and res.speak
    assert "oracle.key_path" in res.reply
    assert res.status == "Oracle: not set up"


def test_off_and_unnamed_falls_through_instead_of_answering_for_a_dev_server(cmdr):
    """"is the server up" with the lane OFF must not claim his Oracle box."""
    c, _ = cmdr(**{"oracle.enabled": False})
    assert cmd_mod._h_oracle_status(c, "is the server up", None) is None
    # ...but naming the box still gets the honest line.
    named = cmd_mod._h_oracle_status(c, "how's the oracle box", None)
    assert named is not None and "oracle.enabled" in named.reply


def test_make_tools_registers_nothing_while_the_lane_is_off():
    assert oracle.make_tools(Cfg(**{"oracle.enabled": False}), None) == []
    assert oracle.make_tools(Cfg(**{"oracle.key_path": ""}), None) == []


def test_make_tools_registers_one_tool_once_it_is_configured(key):
    specs = oracle.make_tools(Cfg(**{"oracle.key_path": str(key)}), None)
    assert [s.name for s in specs] == ["oracle_status"]
    assert specs[0].description_words() <= 20


# --------------------------------------------------------- the timeout
def test_a_wedged_ssh_is_killed_without_being_waited_for(monkeypatch):
    """health.run_nvidia_smi's shape: communicate(timeout), kill, walk away.
    proc.wait() must never be reached -- an ssh stuck in a TCP connect to a
    host that drops packets is exactly the child that does not come back."""
    proc = MagicMock()
    proc.communicate.side_effect = subprocess.TimeoutExpired("ssh", 6)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    conf = oracle.read_config(Cfg())
    res = oracle.run_ssh(conf, "true")
    assert res.ok is False and res.reason == "timeout"
    proc.kill.assert_called_once()
    proc.wait.assert_not_called()


def test_a_timeout_says_so_and_does_not_hold_the_turn(cmdr, monkeypatch, key):
    monkeypatch.setattr(oracle, "run_ssh",
                        fake_ssh(oracle.SshResult(False, reason="timeout")))
    said = []
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    monkeypatch.setattr(Commander, "_speak", lambda self, text: said.append(text))
    res = c.handle("how's the oracle box", source="typed")
    assert res.handled and res.done is False       # answered off the turn
    assert said and "didn't answer in 6 seconds" in said[0]


@pytest.mark.parametrize("err,reason", [
    ("opc@1.2.3.4: Permission denied (publickey).", "auth"),
    ("ssh: connect to host 1.2.3.4 port 22: Connection timed out", "unreachable"),
    ("Host key verification failed.", "hostkey"),
    ("something else entirely", "failed"),
])
def test_ssh_stderr_is_told_apart(err, reason):
    assert oracle.classify_error(err) == reason


def test_the_argv_can_never_hang_on_a_prompt():
    conf = oracle.read_config(Cfg(**{"oracle.key_path": "~/k"}))
    argv = oracle.ssh_argv(conf, "pm2 status")
    assert "BatchMode=yes" in argv and "NumberOfPasswordPrompts=0" in argv
    assert "ConnectTimeout=5" in argv           # inside the 6 s budget
    assert argv[argv.index("-i") + 1] == os.path.expanduser("~/k")
    assert "opc@170.9.245.136" in argv
    # The command is the LAST element and one element: no shell on this side.
    assert argv[-1].endswith("pm2 status")


# ------------------------------------------------------ a good reading
def test_a_good_payload_parses_into_the_reading_and_one_sentence():
    r = oracle.parse_status(GOOD_PAYLOAD)
    assert r.readable and r.pm2_ok
    assert round(r.uptime_s) == 3456789 and r.load == (0.08, 0.05, 0.01)
    assert r.mem_total_kb == 1009884 and r.mem_avail_kb == 512000
    assert r.disk_free_kb == 25000000
    assert len(r.procs) == 1
    p = r.procs[0]
    assert p.name == "game-news" and p.online and p.restarts == 2
    assert 2.9 * 86400 < p.uptime_s < 3.1 * 86400

    line = oracle.speak_line(oracle.read_config(Cfg()), r)
    # ONE sentence: no full stop anywhere but the end (0.5 is not one).
    assert line.endswith(".") and not re.search(r"\.\s", line)
    assert "game-news" in line and "sir" in line
    assert "3 days" in line and "40 days" in line   # the bot, then the box

    sheet = oracle.card(oracle.read_config(Cfg()), r)
    assert "opc@170.9.245.136" in sheet and "game-news" in sheet
    assert "0.5 GB free of 1.0 GB" in sheet


def test_a_stopped_process_leads_the_sentence():
    payload = GOOD_PAYLOAD.replace('"status": "online"', '"status": "stopped"')
    r = oracle.parse_status(payload)
    line = oracle.speak_line(oracle.read_config(Cfg()), r)
    assert line.startswith("game-news is stopped on the Oracle box")


def test_a_dead_pm2_still_reports_the_box():
    payload = GOOD_PAYLOAD.split("#pm2")[0] + "#pm2\n\n"
    r = oracle.parse_status(payload)
    assert r.readable and not r.pm2_ok
    assert "couldn't get a word out of pm2" in oracle.speak_line(
        oracle.read_config(Cfg()), r)


def test_pm2_banner_noise_does_not_cost_the_process_list():
    payload = GOOD_PAYLOAD.replace("#pm2\n", "#pm2\n>>>> In-memory PM2 is out-of-date\n")
    assert oracle.parse_status(payload).procs[0].name == "game-news"


def test_the_status_command_answers_and_lands_on_the_card(cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out=GOOD_PAYLOAD))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    cards, said = [], []
    from jarvis.events import JarvisReply, bus
    bus.subscribe(JarvisReply, lambda e: cards.append(e.text))
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    monkeypatch.setattr(Commander, "_speak", lambda self, text: said.append(text))
    res = c.handle("what's running on oracle", source="typed")
    assert res.handled and res.done is False
    assert said and "game-news" in said[0]
    assert any("game-news" in card for card in cards)
    assert len(runner.calls) == 1
    assert runner.calls[0][1] == oracle.STATUS_COMMAND


def test_two_questions_in_a_row_are_one_round_trip(cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out=GOOD_PAYLOAD))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    monkeypatch.setattr(Commander, "_speak", lambda self, text: None)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    c.handle("how's the oracle box", source="typed")
    second = c.handle("how's the game news bot", source="typed")
    assert len(runner.calls) == 1
    # The cached answer is spoken ON the turn, not handed to a thread.
    assert second.done is True and "game-news" in second.reply


def test_the_cache_expires(monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out=GOOD_PAYLOAD))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    now = [1000.0]
    monkeypatch.setattr(oracle, "_clock", lambda: now[0])
    conf = oracle.read_config(Cfg(**{"oracle.key_path": str(key)}))
    oracle.status(conf)
    now[0] += conf.cache_s + 1
    oracle.status(conf)
    assert len(runner.calls) == 2


# --------------------------------------------------------- the allow-list
def test_resolve_action_only_ever_returns_a_row_of_the_table():
    conf = oracle.read_config(Cfg())
    assert oracle.resolve_action(conf, "restart the bot")[1] == "pm2 restart game-news"
    assert oracle.resolve_action(conf, "restart bot")[1] == "pm2 restart game-news"
    assert oracle.resolve_action(conf, "status")[1] == "pm2 status --no-color"
    assert oracle.resolve_action(conf, "rm -rf /")[1] is None
    assert oracle.resolve_action(conf, "restart everything")[1] is None
    assert oracle.resolve_action(conf, "")[1] is None


def test_an_unknown_action_is_refused_out_loud_and_never_run(
        cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out="never"))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    res = c.handle("run deploy on the oracle box", source="typed")
    assert res.handled and res.speak and res.status == "Not on the list"
    assert '"deploy"' in res.reply
    assert '"restart the bot"' in res.reply       # it names what it does know
    assert runner.calls == []                    # nothing reached the far side


def test_a_transcript_word_can_never_become_a_command(cmdr, monkeypatch, key):
    """The property the whole design exists for: whatever he said, the only
    strings that can reach the far side are the config's own values."""
    runner = fake_ssh(oracle.SshResult(True, out=""))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    for said in ("oracle rm -rf slash", "on the oracle box, curl evil dot com",
                 "run shutdown now on the oracle box", "oracle cat etc shadow"):
        c.handle(said, source="typed")
    assert runner.calls == []


def test_changes_state_reads_the_command_not_the_name():
    assert oracle.changes_state("pm2 restart game-news")
    assert oracle.changes_state("cd ~/Game-News && git pull && npm install")
    assert not oracle.changes_state("pm2 status --no-color")
    assert not oracle.changes_state("pm2 logs --lines 20 --nostream")


# ------------------------------------------------------- the read-back
def test_a_restart_is_read_back_and_only_a_yes_runs_it(cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out="[PM2] Applying action restartProcessId"))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    said = []
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    monkeypatch.setattr(Commander, "_speak", lambda self, text: said.append(text))

    res = c.handle("restart the game news bot", source="typed")
    assert res.reply == "Restart the bot on the Oracle box, sir?"
    assert res.speak and res.status == "Confirm?"
    assert runner.calls == []                    # read back, not run

    res = c.handle("yes", source="typed")
    assert res.handled
    assert [call[1] for call in runner.calls] == ["pm2 restart game-news"]
    assert said and said[-1] == oracle.DONE_LINE


def test_a_no_drops_the_restart(cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out="ran"))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    c.handle("restart the game news bot", source="typed")
    res = c.handle("no", source="typed")
    assert res.reply == "Very good, sir." and runner.calls == []


def test_changing_the_subject_drops_the_restart(cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out="ran"))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    c.handle("restart the game news bot", source="typed")
    c.handle("what time is it", source="typed")
    c.handle("yes", source="typed")
    assert runner.calls == []


def test_the_logs_action_reads_only_and_needs_no_yes(cmdr, monkeypatch, key):
    payload = "\n".join(f"line {i}" for i in range(5)) + "\nTypeError: boom"
    runner = fake_ssh(oracle.SshResult(True, out=payload))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    cards, said = [], []
    from jarvis.events import JarvisReply, bus
    bus.subscribe(JarvisReply, lambda e: cards.append(e.text))
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    monkeypatch.setattr(Commander, "_speak", lambda self, text: said.append(text))
    res = c.handle("show me the game news bot logs", source="typed")
    assert res.handled and res.done is False
    assert [call[1] for call in runner.calls] == \
        ["pm2 logs --lines 20 --nostream --no-color"]
    assert said and "6 lines" in said[0] and "1 of them mention an error" in said[0]
    assert any("TypeError" in card for card in cards)


def test_an_action_clears_the_status_cache(cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out=GOOD_PAYLOAD),
                      oracle.SshResult(True, out="restarted"),
                      oracle.SshResult(True, out=GOOD_PAYLOAD))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    monkeypatch.setattr(Commander, "_speak", lambda self, text: None)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    c.handle("how's the oracle box", source="typed")
    c.handle("restart the game news bot", source="typed")
    c.handle("yes", source="typed")
    c.handle("how's the oracle box", source="typed")
    # status, restart, status again -- the reading from before the restart
    # must never be read back as if it were after it.
    assert len(runner.calls) == 3


# ------------------------------------------------------------- housekeeping
def test_the_lane_never_builds_an_inbound_door():
    """OUTBOUND ONLY, asserted rather than only documented: no tunnel, no
    reverse tunnel, no port-forward, nothing listening."""
    text = pathlib.Path(oracle.__file__).read_text(encoding="utf-8")
    argv = " ".join(oracle.ssh_argv(oracle.read_config(Cfg()), "x"))
    for flag in (" -R ", " -L ", " -D ", " -w "):
        assert flag not in argv
    for word in ("RemoteForward", "LocalForward", "GatewayPorts",
                 "-R ", "-L ", "socket.bind", "listen("):
        assert word not in text


def test_uptime_words_speaks_days():
    assert oracle.uptime_words(40 * 86400) == "40 days"
    assert oracle.uptime_words(86400) == "a day"
    assert oracle.uptime_words(3 * 86400 + 4 * 3600) == "3 days and 4 hours"
    assert oracle.uptime_words(70 * 60) == "an hour and 10 minutes"
    assert oracle.uptime_words(None) == "an unknown time"
