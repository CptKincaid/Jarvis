"""The Oracle Cloud lane (rewritten 2026-08-31 against the REAL box):
jarvis/tools/oracle.py + its five Tier-1 commands.

NOTHING HERE OPENS A SOCKET. ``oracle.run_ssh`` is the one seam every path
goes through, and every test either replaces it or replaces ``subprocess.Popen``
underneath it -- the unconfigured test asserts that a Popen at all is the bug.

REAL_PAYLOAD below is `status_command`'s actual output, copied verbatim off
``opc@163.192.101.18`` on 2026-08-31, MemAvailable absurdity and all. Every
number the spoken lines are asserted against is that box's number.

Covered: the unconfigured path (off, no key, missing key, world-readable
key), the timeout path, the real reading and the two spoken lines it makes,
resolving nine spoken names onto nine units, the allow-list refusing
everything that is not one of three actions, and the read-back a restart has
to survive before anything runs.
"""
import json
import os
import pathlib
import re
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jarvis import commander as cmd_mod
from jarvis.assistant_config import DEFAULTS
from jarvis.commander import Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.router import Router
from jarvis.tools import oracle


# ------------------------------------------------------------------ fakes
class Cfg:
    """The AssistantConfig surface oracle.read_config uses: dotted get().
    Seeded from the SHIPPED defaults, so a test that passes here is a test
    about the table he actually gets."""

    def __init__(self, **over):
        self.data = {"claude.big_model": "fable", "briefing.enabled": False}
        for key, val in DEFAULTS["oracle"].items():
            self.data[f"oracle.{key}"] = val
        self.data["oracle.enabled"] = True
        self.data.update(over)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setup_line(self, section):
        return f"I'll need {section} set up, sir."


# The host off the stale PDF cheat sheet -- a different server entirely.
STALE_HOST = "170.9.245.136"

UNITS = ["haymaker-bot", "coa-bot", "exoshock-bot", "vrider-bot",
         "timecard-bot", "knightfall-web", "elevation-api",
         "monday-sheets-sync", "bot-dashboard"]


def payload(**states) -> str:
    """status_command's real output, with any unit's state overridden.

    Verbatim from demon-bot 2026-08-31 -- note MemAvailable (20512504 kB)
    exceeding MemTotal (5779324 kB), which is the reading that makes the
    clamp in _available_kb load-bearing rather than defensive."""
    roll = "\n".join(f"{u} {states.get(u, 'active')}" for u in UNITS)
    return (
        "#host\ndemon-bot\n"
        "#uptime\n6195909.46 12284858.41\n"
        "#load\n0.27 0.11 0.03 1/389 1834800\n"
        "#mem\n"
        "MemTotal:        5779324 kB\n"
        "MemFree:         1809620 kB\n"
        "MemAvailable:   20512504 kB\n"
        "Buffers:             836 kB\n"
        "Cached:          2090396 kB\n"
        "#disk\n"
        "Filesystem                 1024-blocks     Used Available Capacity Mounted on\n"
        "/dev/mapper/ocivolume-root    30867456 20462012  10405444      67% /\n"
        f"#services\n{roll}\n")


REAL_PAYLOAD = payload()


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
def conf(key):
    return oracle.read_config(Cfg(**{"oracle.key_path": str(key)}))


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


# ------------------------------------------------- the box he actually has
def test_the_shipped_config_is_the_verified_box():
    """The host, the user and the key that were checked by ssh, and the lane
    still OFF -- switching it on is his call, not the installer's."""
    section = DEFAULTS["oracle"]
    assert section["enabled"] is False
    assert section["host"] == "163.192.101.18"
    assert section["user"] == "opc"
    assert section["key_path"].endswith("ssh-key-2025-08-15.key")
    assert len(section["services"]) == 9


def test_the_shipped_config_matches_the_verified_table():
    """DEFAULTS ships literal config data and oracle.py holds the table the
    matchers are built from; they have to be the same nine rows."""
    shipped = DEFAULTS["oracle"]["services"]
    assert shipped == oracle.DEFAULT_SERVICES
    assert [row["unit"] for row in shipped.values()] == UNITS


def test_flipping_enabled_is_the_only_edit_needed(monkeypatch, tmp_path):
    """The promise in the docs: one line in assistant.json and the lane is
    live. Anything else still missing would show up here."""
    fake_key = tmp_path / "ssh-key-2025-08-15.key"
    fake_key.write_text("k")
    os.chmod(fake_key, 0o600)
    cfg = Cfg(**{"oracle.enabled": True, "oracle.key_path": str(fake_key)})
    assert oracle.missing_reason(oracle.read_config(cfg)) is None


# ------------------------------------------------- unconfigured, and safe
@pytest.mark.parametrize("over,fragment", [
    ({"oracle.enabled": False}, "oracle.enabled"),
    ({"oracle.host": ""}, "oracle.host"),
    ({"oracle.key_path": ""}, "oracle.key_path"),
    ({"oracle.services": {}}, "oracle.services"),
])
def test_missing_reason_names_exactly_what_is_missing(over, fragment, key):
    over.setdefault("oracle.key_path", str(key))
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


def test_the_unconfigured_command_answers_one_line_and_opens_no_socket(
        cmdr, monkeypatch):
    """The whole point of the default: off, no socket, one honest line."""
    def _boom(*a, **k):                       # any Popen at all is the bug
        raise AssertionError("oracle opened a subprocess while unconfigured")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    c, _ = cmdr(**{"oracle.enabled": False})
    res = c.handle("how's the oracle box", source="typed")
    assert res.handled and res.speak
    assert "oracle.enabled" in res.reply
    assert res.status == "Oracle: not set up"


def test_off_and_unnamed_falls_through_instead_of_answering_for_a_dev_server(cmdr):
    """"is the server up" with the lane OFF must not claim his Oracle box --
    and neither must "how's the haymaker", because ~/haymaker-digest is a
    job on THIS machine."""
    c, _ = cmdr(**{"oracle.enabled": False})
    assert cmd_mod._h_oracle_status(c, "is the server up", None) is None
    m = cmd_mod._ORACLE_SERVICE_RX.match("how's the haymaker")
    assert m and cmd_mod._h_oracle_service(c, "how's the haymaker", m) is None
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
def test_a_wedged_ssh_is_killed_without_being_waited_for(monkeypatch, conf):
    """health.run_nvidia_smi's shape: communicate(timeout), kill, walk away.
    proc.wait() must never be reached -- an ssh stuck in a TCP connect to a
    host that drops packets is exactly the child that does not come back."""
    proc = MagicMock()
    proc.communicate.side_effect = subprocess.TimeoutExpired("ssh", 6)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
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
    ("sudo: a password is required", "privilege"),
    ("Failed to restart haymaker-bot.service: Interactive authentication "
     "required.", "privilege"),
    ("something else entirely", "failed"),
])
def test_ssh_stderr_is_told_apart(err, reason):
    assert oracle.classify_error(err) == reason


def test_a_refused_sudo_is_not_blamed_on_the_ssh_key(conf):
    """The two failures have different fixes and live in different files."""
    line = oracle.fail_line(conf, "privilege")
    assert "sudo" in line and "key" not in line


def test_the_argv_can_never_hang_on_a_prompt(key):
    conf = oracle.read_config(Cfg(**{"oracle.key_path": str(key)}))
    argv = oracle.ssh_argv(conf, "uptime")
    assert "BatchMode=yes" in argv and "NumberOfPasswordPrompts=0" in argv
    assert "ConnectTimeout=5" in argv           # inside the 6 s budget
    assert argv[argv.index("-i") + 1] == str(key)
    assert "opc@163.192.101.18" in argv
    # The command is the LAST element and one element: no shell on this side.
    assert argv[-1] == "uptime"


# ------------------------------------------------------ the real reading
def test_the_real_payload_parses_into_the_reading(conf):
    r = oracle.parse_status(conf, REAL_PAYLOAD)
    assert r.readable and r.host == "demon-bot"
    assert round(r.uptime_s) == 6195909 and r.load == (0.27, 0.11, 0.03)
    assert r.disk_total_kb == 30867456 and r.disk_free_kb == 10405444
    assert [s.unit for s in r.services] == UNITS
    assert all(s.up for s in r.services)


def test_a_kernel_that_lies_about_memavailable_is_not_believed(conf):
    """demon-bot reports MemAvailable 20512504 kB on a 5779324 kB box --
    nineteen gigabytes free of five and a half. procps' `free` falls back to
    MemFree for exactly this, and so must we: without the clamp the spoken
    line is a confident absurdity."""
    r = oracle.parse_status(conf, REAL_PAYLOAD)
    assert r.mem_total_kb == 5779324
    assert r.mem_avail_kb == 1809620            # MemFree, not MemAvailable
    sheet = oracle.card(conf, r)
    assert "3.8 GB used of 5.5 GB" in sheet


def test_a_healthy_box_is_one_sentence_leading_with_the_roll_call(conf):
    r = oracle.parse_status(conf, REAL_PAYLOAD)
    line = oracle.speak_line(conf, r)
    assert line == ("All nine services are up on demon-bot, sir; ten weeks "
                    "and a day of uptime and a third of the disk free.")
    # ONE sentence: no full stop anywhere but the end.
    assert line.endswith(".") and not re.search(r"\.\s", line)


def test_anything_down_leads_the_sentence(conf):
    r = oracle.parse_status(conf, payload(**{"haymaker-bot": "inactive"}))
    assert oracle.speak_line(conf, r) == \
        "Haymaker is down, sir; the other eight are up."


@pytest.mark.parametrize("states,expected", [
    ({"haymaker-bot": "failed"},
     "Haymaker failed, sir; the other eight are up."),
    ({"haymaker-bot": "inactive", "coa-bot": "inactive"},
     "Haymaker and Court of Awe are down, sir; the other seven are up."),
    ({"haymaker-bot": "inactive", "coa-bot": "failed"},
     "Haymaker and Court of Awe are not running, sir; the other seven are up."),
    ({u: "inactive" for u in UNITS[:4]},
     "Four of the nine services are down on demon-bot, sir; the other five "
     "are up."),
])
def test_the_unhealthy_sentence_names_what_is_wrong(conf, states, expected):
    r = oracle.parse_status(conf, payload(**states))
    line = oracle.speak_line(conf, r)
    assert line == expected.replace("Haymaker failed",
                                    "Haymaker is failed")


def test_a_unit_systemd_never_heard_of_is_unaccounted_for_not_down(conf):
    """A typo in the config is a different thing from a bot falling over."""
    trimmed = REAL_PAYLOAD.replace("haymaker-bot active\n", "")
    r = oracle.parse_status(conf, trimmed)
    assert r.state_of("haymaker-bot").state == "unknown"
    assert "unaccounted for" in oracle.speak_line(conf, r)


def test_the_card_shows_every_service_and_marks_the_dead_one(conf):
    r = oracle.parse_status(conf, payload(**{"coa-bot": "failed"}))
    sheet = oracle.card(conf, r)
    assert "demon-bot" in sheet and "opc@163.192.101.18" in sheet
    for unit in UNITS:
        assert unit in sheet
    assert "! coa-bot" in sheet and "  haymaker-bot" in sheet
    assert "9.9 GB free of 29.4 GB" in sheet


def test_a_box_that_answers_nothing_readable_says_so(conf):
    assert oracle.speak_line(conf, oracle.parse_status(conf, "")) == \
        oracle.UNREADABLE_LINE


def test_uptime_words_goes_to_weeks_for_a_server(conf):
    assert oracle.uptime_words(6195909.46) == "ten weeks and a day"
    assert oracle.uptime_words(70 * 86400) == "ten weeks"
    assert oracle.uptime_words(40 * 86400) == "five weeks and five days"
    assert oracle.uptime_words(86400) == "a day"
    assert oracle.uptime_words(3 * 86400 + 4 * 3600) == "three days and four hours"
    assert oracle.uptime_words(70 * 60) == "an hour and 10 minutes"
    assert oracle.uptime_words(None) == "an unknown time"


def test_disk_words_answers_the_question_that_was_asked():
    assert oracle.disk_words(10405444, 30867456) == "a third of the disk free"
    assert oracle.disk_words(50, 100) == "half the disk free"
    assert oracle.disk_words(41, 100) == "41 percent of the disk free"
    assert oracle.disk_words(None, 100) == "" and oracle.disk_words(1, 0) == ""


# ------------------------------------------------------- the round trip
def test_the_status_command_is_one_round_trip_over_the_nine_units(
        cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out=REAL_PAYLOAD))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    cards, said = [], []
    from jarvis.events import JarvisReply, bus
    bus.subscribe(JarvisReply, lambda e: cards.append(e.text))
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    monkeypatch.setattr(Commander, "_speak", lambda self, text: said.append(text))
    res = c.handle("how are the bots", source="typed")
    assert res.handled and res.done is False
    assert said and said[0].startswith("All nine services are up")
    assert any("haymaker-bot" in card for card in cards)
    assert len(runner.calls) == 1
    sent = runner.calls[0][1]
    for unit in UNITS:
        assert unit in sent
    assert "pm2" not in sent and "systemctl is-active" in sent


def test_the_status_command_survives_an_empty_service_table(key):
    """`for u in ; do` is a syntax error on the far side, so the loop has to
    go away rather than be built empty."""
    conf = oracle.read_config(Cfg(**{"oracle.key_path": str(key),
                                     "oracle.services": {}}))
    assert "for u in" not in oracle.status_command(conf)


def test_two_questions_in_a_row_are_one_round_trip(cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out=REAL_PAYLOAD))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    monkeypatch.setattr(Commander, "_speak", lambda self, text: None)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    c.handle("how's the oracle box", source="typed")
    second = c.handle("how's the haymaker bot", source="typed")
    assert len(runner.calls) == 1
    # The cached answer is spoken ON the turn, not handed to a thread.
    assert second.done is True and second.reply.startswith("Haymaker is up")


def test_the_cache_expires(monkeypatch, conf):
    runner = fake_ssh(oracle.SshResult(True, out=REAL_PAYLOAD))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    now = [1000.0]
    monkeypatch.setattr(oracle, "_clock", lambda: now[0])
    oracle.status(conf)
    now[0] += conf.cache_s + 1
    oracle.status(conf)
    assert len(runner.calls) == 2


def test_a_reading_taken_against_a_different_table_is_not_reused(
        monkeypatch, conf, key):
    """He edited oracle.services between two questions: answering the new
    roll-call off the old one would be a wrong answer with a straight face."""
    monkeypatch.setattr(oracle, "run_ssh",
                        fake_ssh(oracle.SshResult(True, out=REAL_PAYLOAD)))
    oracle.status(conf)
    assert oracle.cached(conf) is not None
    fewer = oracle.read_config(Cfg(**{
        "oracle.key_path": str(key),
        "oracle.services": {"haymaker": {"unit": "haymaker-bot"}}}))
    assert oracle.cached(fewer) is None


# ------------------------------------------------------- one service
@pytest.mark.parametrize("spoken,unit", [
    ("haymaker", "haymaker-bot"),
    ("haymaker bot", "haymaker-bot"),
    ("the haymaker", "haymaker-bot"),
    ("the haymaker bot", "haymaker-bot"),
    ("coa", "coa-bot"),
    ("court of awe", "coa-bot"),
    ("exoshock", "exoshock-bot"),
    ("vrider", "vrider-bot"),
    ("v rider", "vrider-bot"),
    ("timecard", "timecard-bot"),
    ("time card", "timecard-bot"),
    ("knightfall", "knightfall-web"),
    ("knightfall web", "knightfall-web"),
    ("elevation api", "elevation-api"),
    ("the elevation api", "elevation-api"),
    ("ditch grade", "elevation-api"),
    ("monday sync", "monday-sheets-sync"),
    ("monday sheets sync", "monday-sheets-sync"),
    ("dashboard", "bot-dashboard"),
    ("bot dashboard", "bot-dashboard"),
])
def test_a_spoken_name_resolves_to_the_right_unit(conf, spoken, unit):
    svc = oracle.resolve_service(conf, spoken)
    assert svc is not None and svc.unit == unit


@pytest.mark.parametrize("spoken", [
    "", "the bot", "the server", "the oracle box", "everything",
    "haymaker digest", "the other one", "rm -rf /", "nginx", "docker",
])
def test_a_name_that_is_not_one_of_his_nine_resolves_to_nothing(conf, spoken):
    """Including "the bot": with nine of them that is not an answer, and
    picking whichever was listed first is how the wrong one gets restarted."""
    assert oracle.resolve_service(conf, spoken) is None


def test_a_bare_monday_never_opens_the_lane_on_a_scheduling_question(conf):
    """"what about Monday?" is asked while planning the week, and this lane
    answering it would be Jarvis talking over that conversation. The DOOR is
    what stays shut: no phrase in the shipped table is bare "monday", so the
    bare-name matcher never fires on it. The resolver is still generous once
    the box HAS been named ("on the oracle box, restart monday"), which is a
    sentence nobody says by accident."""
    assert cmd_mod._ORACLE_SERVICE_RX.match("what about monday") is None
    assert cmd_mod._ORACLE_SERVICE_RX.match("how's monday") is None
    assert cmd_mod._ORACLE_SERVICE_RX.match("what about monday sync")
    assert oracle.resolve_service(conf, "monday").unit == "monday-sheets-sync"


def test_one_service_answers_off_the_same_roll_call(cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out=payload(
        **{"knightfall-web": "inactive"})))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    said = []
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    monkeypatch.setattr(Commander, "_speak", lambda self, text: said.append(text))
    res = c.handle("is knightfall up", source="typed")
    assert res.handled and res.done is False
    assert said == ["Knightfall is down on demon-bot, sir; eight of the nine "
                    "are up."]
    assert len(runner.calls) == 1


def test_a_healthy_service_still_mentions_a_sick_neighbour(conf):
    r = oracle.parse_status(conf, payload(**{"coa-bot": "inactive"}))
    svc = oracle.resolve_service(conf, "haymaker")
    assert oracle.service_line(conf, r, svc) == \
        "Haymaker is up on demon-bot, sir, though Court of Awe is not."


def test_a_healthy_service_on_a_healthy_box_says_so(conf):
    r = oracle.parse_status(conf, REAL_PAYLOAD)
    svc = oracle.resolve_service(conf, "the elevation api")
    assert oracle.service_line(conf, r, svc) == \
        "The elevation API is up on demon-bot, sir, along with the other eight."


def test_a_service_question_about_something_else_is_not_answered_here(
        cmdr, monkeypatch, key):
    """"how's the weather" must not reach this lane at all, and a name the
    regex lets through but the table does not know falls to the model."""
    runner = fake_ssh(oracle.SshResult(True, out=REAL_PAYLOAD))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    assert cmd_mod._ORACLE_SERVICE_RX.match("how's the weather") is None
    assert cmd_mod._ORACLE_SERVICE_RX.match("how's the run") is None
    conf = oracle.read_config(Cfg(**{
        "oracle.key_path": str(key),
        "oracle.services": {"coa": {"unit": "coa-bot"}}}))
    assert oracle.resolve_service(conf, "haymaker") is None
    assert runner.calls == []


# --------------------------------------------------------- the allow-list
def test_action_command_builds_only_the_two_it_knows(conf):
    unit = "haymaker-bot"
    assert oracle.action_command(conf, "logs", unit) == \
        "journalctl -u haymaker-bot -n 20 --no-pager"
    assert oracle.action_command(conf, "restart", unit) == \
        "sudo -n systemctl restart haymaker-bot && systemctl is-active haymaker-bot"
    for action in ("stop", "start", "deploy", "", "rm", "status"):
        assert oracle.action_command(conf, action, unit) is None
    # ...and never against a unit this config did not load.
    assert oracle.action_command(conf, "restart", "sshd") is None
    assert oracle.action_command(conf, "restart", "haymaker-bot; rm -rf /") is None


def test_a_restart_needs_sudo_because_polkit_refuses_the_plain_one():
    """Verified on the box: `pkcheck --action-id
    org.freedesktop.systemd1.manage-units` answers "Authorization requires
    authentication" for opc over a non-interactive ssh, while `sudo -n true`
    succeeds. The -n is what keeps a locked-down sudoers from turning this
    into a password prompt that eats the whole budget."""
    cmd = oracle.restart_command("haymaker-bot")
    assert cmd.startswith("sudo -n systemctl restart ")
    assert "-n" in cmd.split()[:2]


def test_a_unit_name_that_is_not_a_unit_name_never_loads():
    """The config is not a transcript, but it is the only text that reaches
    a remote shell here, so it is checked like one."""
    conf = oracle.read_config(Cfg(**{"oracle.services": {
        "good": {"unit": "haymaker-bot"},
        "hostile": {"unit": "haymaker-bot; curl evil.example"},
        "spaced": {"unit": "haymaker bot"},
        "dashed": {"unit": "-rf"},
        "empty": {"unit": ""},
    }}))
    assert [s.unit for s in conf.services] == ["haymaker-bot"]
    assert "curl" not in oracle.status_command(conf)


def test_a_transcript_word_can_never_become_a_command(cmdr, monkeypatch, key):
    """The property the whole design exists for: whatever he said, the only
    strings that can reach the far side are ones this module built."""
    runner = fake_ssh(oracle.SshResult(True, out=""))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    for said in ("oracle rm dash rf slash", "on the oracle box, curl evil dot com",
                 "run shutdown now on the oracle box", "oracle cat etc shadow",
                 "restart haymaker semicolon reboot"):
        c.handle(said, source="typed")
    assert runner.calls == []


def test_an_unknown_thing_said_at_the_box_is_refused_out_loud(
        cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out="never"))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    res = c.handle("run deploy on the oracle box", source="typed")
    assert res.handled and res.speak and res.status == "Not on the list"
    assert '"deploy"' in res.reply
    assert runner.calls == []


def test_stopping_a_service_is_refused_out_loud_not_silently_dropped(
        cmdr, monkeypatch, key):
    """Silence would leave him believing the bot had been stopped."""
    runner = fake_ssh(oracle.SshResult(True, out="never"))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    res = c.handle("stop haymaker", source="typed")
    assert res.handled and res.speak and res.status == "Not on the list"
    assert res.reply == oracle.UNKNOWN_ACTION_LINE
    assert runner.calls == []


def test_restart_the_bot_asks_which_of_the_nine(cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out="never"))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    res = c.handle("restart the bot", source="typed")
    assert res.handled and res.reply == oracle.WHICH_RESTART_LINE
    assert res.status == "Which one?" and runner.calls == []


# ------------------------------------------------------- the read-back
def test_a_restart_is_read_back_by_name_and_only_a_yes_runs_it(
        cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out="active\n"))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    said = []
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    monkeypatch.setattr(Commander, "_speak", lambda self, text: said.append(text))

    res = c.handle("restart the haymaker bot", source="typed")
    assert res.reply == "Restart Haymaker on the Oracle box, sir?"
    assert res.speak and res.status == "Confirm?"
    assert runner.calls == []                    # read back, not run

    res = c.handle("yes", source="typed")
    assert res.handled
    assert [call[1] for call in runner.calls] == [
        "sudo -n systemctl restart haymaker-bot && "
        "systemctl is-active haymaker-bot"]
    assert said and said[-1] == "Haymaker is back up, sir."


def test_a_restart_that_does_not_come_back_up_is_not_called_done(conf):
    svc = oracle.resolve_service(conf, "haymaker")
    assert oracle.restart_line(svc, "active") == "Haymaker is back up, sir."
    assert oracle.restart_line(svc, "activating") == \
        "Haymaker restarted, sir, and it's still starting."
    assert "isn't up yet" in oracle.restart_line(svc, "")


def test_a_no_drops_the_restart(cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out="active"))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    c.handle("restart the haymaker bot", source="typed")
    res = c.handle("no", source="typed")
    assert res.reply == "Very good, sir." and runner.calls == []


def test_changing_the_subject_drops_the_restart(cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out="active"))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    c.handle("restart the haymaker bot", source="typed")
    c.handle("what time is it", source="typed")
    c.handle("yes", source="typed")
    assert runner.calls == []


def test_an_action_clears_the_roll_call_cache(cmdr, monkeypatch, key):
    runner = fake_ssh(oracle.SshResult(True, out=REAL_PAYLOAD),
                      oracle.SshResult(True, out="active"),
                      oracle.SshResult(True, out=REAL_PAYLOAD))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    monkeypatch.setattr(Commander, "_speak", lambda self, text: None)
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    c.handle("how's the oracle box", source="typed")
    c.handle("restart the haymaker bot", source="typed")
    c.handle("yes", source="typed")
    c.handle("how's the oracle box", source="typed")
    # status, restart, status again -- the reading from before the restart
    # must never be read back as if it were after it.
    assert len(runner.calls) == 3


# ------------------------------------------------------------- the logs
def test_the_logs_read_one_named_journal_and_need_no_yes(
        cmdr, monkeypatch, key):
    lines = [f"Aug 31 13:43:5{i} demon-bot python3[1324061]: line {i}"
             for i in range(5)]
    lines.append("Aug 31 13:44:00 demon-bot python3[1324061]: ERROR - boom")
    runner = fake_ssh(oracle.SshResult(True, out="\n".join(lines)))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    cards, said = [], []
    from jarvis.events import JarvisReply, bus
    bus.subscribe(JarvisReply, lambda e: cards.append(e.text))
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    monkeypatch.setattr(Commander, "_speak", lambda self, text: said.append(text))
    res = c.handle("show me the haymaker logs", source="typed")
    assert res.handled and res.done is False
    assert [call[1] for call in runner.calls] == \
        ["journalctl -u haymaker-bot -n 20 --no-pager"]
    assert said == ["The last 6 lines of Haymaker, sir; 1 of them mention "
                    "an error."]
    assert any("ERROR - boom" in card for card in cards)


def test_a_clean_log_says_so_without_calling_it_healthy(conf):
    svc = oracle.resolve_service(conf, "haymaker")
    line, sheet = oracle.log_summary(svc, "started\nconnected\nready")
    assert line == "The last 3 lines of Haymaker, sir; not one of them " \
                   "mentions an error."
    assert sheet == "started\nconnected\nready"
    assert oracle.log_summary(svc, "")[0] == "Haymaker's log came back empty, sir."


def test_asking_the_box_for_logs_asks_which_service(cmdr, monkeypatch, key):
    """Nine journals are not one answer."""
    runner = fake_ssh(oracle.SshResult(True, out="never"))
    monkeypatch.setattr(oracle, "run_ssh", runner)
    cards = []
    from jarvis.events import JarvisReply, bus
    bus.subscribe(JarvisReply, lambda e: cards.append(e.text))
    c, _ = cmdr(**{"oracle.key_path": str(key)})
    res = c.handle("show me the oracle logs", source="typed")
    assert res.handled and res.reply == oracle.WHICH_SERVICE_LINE
    assert any("haymaker-bot" in card for card in cards)
    assert runner.calls == []


# ------------------------------------------------------------- housekeeping
def test_the_lane_never_builds_an_inbound_door(conf):
    """OUTBOUND ONLY, asserted rather than only documented: no tunnel, no
    reverse tunnel, no port-forward, nothing listening."""
    text = pathlib.Path(oracle.__file__).read_text(encoding="utf-8")
    argv = " ".join(oracle.ssh_argv(conf, "x"))
    for flag in (" -R ", " -L ", " -D ", " -w "):
        assert flag not in argv
    for word in ("RemoteForward", "LocalForward", "GatewayPorts",
                 "-R ", "-L ", "socket.bind", "listen("):
        assert word not in text


def test_nothing_the_box_is_ever_asked_still_talks_about_pm2(conf):
    """The pm2/game-news cheat sheet described an OLDER server. It may
    survive in prose that says so; it may not survive anywhere it could be
    sent to demon-bot, which has no pm2 on it at all."""
    sent = [oracle.status_command(conf)]
    for unit in UNITS:
        sent += [oracle.action_command(conf, a, unit) for a in oracle.ACTIONS]
    for command in sent:
        assert command and "pm2" not in command


def test_the_shipped_config_holds_nothing_from_the_stale_cheat_sheet():
    text = json.dumps(DEFAULTS["oracle"]).lower()
    for stale in ("pm2", "game-news", "game news", STALE_HOST):
        assert stale not in text


def test_the_old_server_address_is_gone_from_the_source():
    """The cheat sheet's host was a DIFFERENT machine. There is no version
    of this code in which it is the right answer, so it survives nowhere."""
    for path in ("jarvis/tools/oracle.py", "jarvis/commander.py",
                 "jarvis/assistant_config.py"):
        assert STALE_HOST not in pathlib.Path(path).read_text(encoding="utf-8")
