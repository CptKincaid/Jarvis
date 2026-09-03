"""HPCOMPUTER: files both ways, the read-only question list, and the door
that refuses everything else -- jarvis/tools/remote.py + jarvis/tools/filepick.py.

NOTHING HERE OPENS A SOCKET, TOUCHES THE TAILNET OR WRITES OUTSIDE tmp_path.
``remote.run_ssh``, ``remote.run_copy`` and ``remote.tailnet_state`` are the
three module-level seams and every test replaces the ones it needs; the
unconfigured tests assert that reaching ``subprocess.Popen`` AT ALL is the
bug, because the whole point of ``missing_reason`` is to answer without
opening anything.

The ground truth these were written against, measured 2026-09-02: HPCOMPUTER
is NOT a tailnet peer (only ``spark`` and an offline iPhone are), it answers
no ping and has no port 22, there is no key for it in ``~/.ssh``, and
tailscaled here runs ``--tun=userspace-networking`` so tailnet traffic must
go through the SOCKS5 proxy on 127.0.0.1:1055.  So the lane ships OFF, and
what is tested is every decision made BEFORE the socket -- which is where
the irreversible mistakes live.
"""
import os
import subprocess

import pytest

from jarvis import commander as cmd_mod
from jarvis.assistant_config import DEFAULTS
from jarvis.tools import filepick, remote


# ------------------------------------------------------------------ fakes
class Cfg:
    """The AssistantConfig surface read_config uses: dotted get().  Seeded
    from the SHIPPED defaults, so a passing test is a test about the config
    he actually gets."""

    def __init__(self, **over):
        self.data = {}
        for key, val in DEFAULTS["remote"].items():
            self.data[f"remote.{key}"] = val
        self.data.update(over)

    def get(self, key, default=None):
        return self.data.get(key, default)


def ready_cfg(tmp_path, **over):
    """A config that passes missing_reason -- enabled, host, user, no key."""
    base = {"remote.enabled": True,
            "remote.host": "hpcomputer.tail5323b8.ts.net",
            "remote.user": "hunterp",
            "remote.key_path": "",
            "remote.local_roots": [str(tmp_path / "Desktop")]}
    base.update(over)
    return Cfg(**base)


@pytest.fixture
def desk(tmp_path):
    d = tmp_path / "Desktop"
    d.mkdir()
    return d


class Result:
    def __init__(self, ok=True, out="", err="", reason=""):
        self.ok, self.out, self.err, self.reason = ok, out, err, reason


class NoPopen:
    """Any Popen at all is the bug in an unconfigured test."""

    def __call__(self, *a, **k):
        raise AssertionError("opened a process with the lane unconfigured")


# ============================================================ the config
def test_the_shipped_config_is_off_and_empty():
    """It ships OFF with a blank host because as of 2026-09-02 HPCOMPUTER is
    not on the tailnet and has no key here.  A default that pretended
    otherwise would produce a ten-second timeout instead of a sentence."""
    row = DEFAULTS["remote"]
    assert row["enabled"] is False
    assert row["host"] == ""
    assert row["user"] == ""
    assert row["key_path"] == ""


def test_the_shipped_config_routes_through_the_userspace_socks_proxy():
    """The one setting that would otherwise cost an afternoon: tailscaled
    here has no tun device, so a direct ssh to a tailnet name cannot route."""
    assert DEFAULTS["remote"]["socks_proxy"] == "127.0.0.1:1055"


def test_remote_is_not_a_nagging_setup_section():
    """Like `oracle`, it stays out of SECTIONS: missing_sections() drives a
    spoken nag, and nagging about a machine he has not connected is noise."""
    from jarvis.assistant_config import SECTIONS
    assert "remote" not in SECTIONS


@pytest.mark.parametrize("over,expect", [
    ({}, "disabled"),
    ({"remote.enabled": True}, "no-host"),
    ({"remote.enabled": True, "remote.host": "h"}, "no-user"),
    ({"remote.enabled": True, "remote.host": "h", "remote.user": "u",
      "remote.key_path": "/nope/missing.key"}, "bad-key"),
    ({"remote.enabled": True, "remote.host": "h", "remote.user": "u"}, ""),
])
def test_missing_reason_names_exactly_what_is_absent(over, expect):
    """A present host with a missing user is a real state, and it must be
    named before a socket opens rather than becoming an auth failure."""
    assert remote.missing_reason(remote.read_config(Cfg(**over))) == expect


def test_a_hostile_config_leaves_the_lane_off():
    class Angry:
        def get(self, *a, **k):
            raise RuntimeError("boom")
    conf = remote.read_config(Angry())
    assert conf.enabled is False and remote.missing_reason(conf) == "disabled"


def test_timeouts_are_clamped_not_trusted():
    conf = remote.read_config(Cfg(**{"remote.timeout_s": 99999,
                                     "remote.transfer_timeout_s": -5}))
    assert conf.timeout_s == remote.MAX_TIMEOUT_S
    assert conf.transfer_timeout_s == remote.MIN_TIMEOUT_S


# ========================================================== the transport
def test_ssh_argv_cannot_sit_on_a_password_prompt(tmp_path):
    """BatchMode plus the three *Authentication=no options are what make the
    timeout mean anything -- without them a host that falls back to password
    auth burns the whole budget waiting for a human being."""
    argv = remote.ssh_argv(remote.read_config(ready_cfg(tmp_path)), "uptime")
    joined = " ".join(argv)
    assert "BatchMode=yes" in joined
    assert "PasswordAuthentication=no" in joined
    assert "KbdInteractiveAuthentication=no" in joined
    assert "NumberOfPasswordPrompts=0" in joined
    assert "-n" in argv                       # stdin cannot be inherited


def test_ssh_argv_accepts_a_first_key_but_never_a_changed_one(tmp_path):
    argv = remote.ssh_argv(remote.read_config(ready_cfg(tmp_path)), "uptime")
    assert "StrictHostKeyChecking=accept-new" in " ".join(argv)


def test_ssh_argv_carries_the_socks_proxy_because_there_is_no_tun(tmp_path):
    """The measured fact this lane is built on."""
    argv = remote.ssh_argv(remote.read_config(ready_cfg(tmp_path)), "uptime")
    joined = " ".join(argv)
    assert "ProxyCommand=" in joined and "-X 5" in joined and "1055" in joined


def test_the_proxy_can_be_switched_off_for_a_real_tun(tmp_path):
    conf = remote.read_config(ready_cfg(tmp_path, **{"remote.socks_proxy": ""}))
    assert "ProxyCommand" not in " ".join(remote.ssh_argv(conf, "uptime"))


def test_the_command_is_the_last_argument_after_a_double_dash(tmp_path):
    argv = remote.ssh_argv(remote.read_config(ready_cfg(tmp_path)), "uptime -p")
    assert argv[-3] == "--"
    assert argv[-2] == "hunterp@hpcomputer.tail5323b8.ts.net"
    assert argv[-1] == "uptime -p"


def test_scp_argv_puts_the_ends_in_the_right_order(tmp_path):
    conf = remote.read_config(ready_cfg(tmp_path))
    push = remote.scp_argv(conf, "/a/b.txt", "~/jarvis-inbox/b.txt", push=True)
    assert push[-2] == "/a/b.txt"
    assert push[-1].endswith(":~/jarvis-inbox/b.txt")
    pull = remote.scp_argv(conf, "/a/b.txt", "~/out/b.txt", push=False)
    assert pull[-2].endswith(":~/out/b.txt")
    assert pull[-1] == "/a/b.txt"


def test_a_local_name_starting_with_a_dash_cannot_become_an_scp_flag(tmp_path):
    conf = remote.read_config(ready_cfg(tmp_path))
    argv = remote.scp_argv(conf, "-rf", "~/jarvis-inbox/-rf", push=True)
    assert argv[-2] == os.path.join(".", "-rf")


def test_a_timeout_kills_the_child_and_does_not_wait_for_it(tmp_path, monkeypatch):
    """subprocess.run kills on timeout and then WAITS; an ssh wedged in a TCP
    connect to a sleeping host is exactly the child that never comes back."""
    killed = []

    class Wedged:
        returncode = None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired("ssh", timeout)

        def kill(self):
            killed.append(True)

        def wait(self, *a, **k):              # pragma: no cover - must not run
            raise AssertionError("waited on a wedged ssh")

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Wedged())
    res = remote.run_ssh(remote.read_config(ready_cfg(tmp_path)), "uptime")
    assert res.ok is False and res.reason == "timeout" and killed == [True]


def test_a_missing_ssh_binary_is_a_sentence_not_a_traceback(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError()))
    res = remote.run_ssh(remote.read_config(ready_cfg(tmp_path)), "uptime")
    assert res.reason == "no-ssh"


@pytest.mark.parametrize("err,expect", [
    ("REMOTE HOST IDENTIFICATION HAS CHANGED! Permission denied", "hostkey"),
    ("Permission denied (publickey).", "auth"),
    ("ssh: connect to host x port 22: No route to host", "unreachable"),
    ("scp: /x: No space left on device", "no-space"),
    ("scp: /x: Read-only file system", "denied"),
    ("scp: /x: No such file or directory", "not-there"),
    ("something else entirely", "failed"),
])
def test_classify_error_puts_the_host_key_first(err, expect):
    """A changed host key ALSO prints "Permission denied" further down, and
    it is the one failure he must look at himself rather than retry."""
    assert remote.classify_error(err) == expect


# ================================================== reachability, honestly
def _peers(*rows):
    return {"Peer": {str(i): r for i, r in enumerate(rows)}}


def test_tailnet_state_tells_absent_from_asleep(tmp_path, monkeypatch):
    """"Never joined" and "asleep" need different sentences from him -- one
    is "install Tailscale on it", the other is "wake it"."""
    import json as _json
    conf = remote.read_config(ready_cfg(tmp_path, **{"remote.host": "hpcomputer"}))

    def fake(payload):
        class P:
            returncode = 0

            def communicate(self, timeout=None):
                return _json.dumps(payload), ""
        return lambda *a, **k: P()

    monkeypatch.setattr(subprocess, "Popen",
                        fake(_peers({"HostName": "spark", "Online": True})))
    assert remote.tailnet_state(conf) == "absent"
    monkeypatch.setattr(subprocess, "Popen", fake(
        _peers({"HostName": "hpcomputer", "Online": False})))
    assert remote.tailnet_state(conf) == "offline"
    monkeypatch.setattr(subprocess, "Popen", fake(
        _peers({"HostName": "hpcomputer", "Online": True})))
    assert remote.tailnet_state(conf) == "online"


def test_a_daemon_that_cannot_be_asked_is_unknown_not_absent(tmp_path, monkeypatch):
    """`tailscale status` with the wrong socket fails, and that is NOT the
    same thing as the host being gone.  Reporting it as gone would be a
    confident wrong answer."""
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert remote.tailnet_state(remote.read_config(ready_cfg(tmp_path))) == "unknown"


def test_every_failure_reason_has_a_spoken_line(tmp_path):
    conf = remote.read_config(ready_cfg(tmp_path))
    for reason in ("disabled", "no-host", "no-user", "bad-key", "no-ssh",
                   "off-tailnet", "asleep", "timeout", "auth", "unreachable",
                   "hostkey", "no-space", "denied", "not-there", "exists",
                   "failed"):
        line = remote.fail_line(conf, reason)
        assert line and "{" not in line
        assert "sir" in line


# ============================================== tier 1: the question list
def test_a_question_never_interpolates_a_transcript(tmp_path):
    """The spoken words choose a ROW; the command is a constant."""
    conf = remote.read_config(ready_cfg(tmp_path))
    for key in remote.QUERIES:
        assert remote.query_command(conf, key)
    assert remote.query_command(conf, "rm -rf /") == ""


def test_the_inbox_question_quotes_the_configured_path(tmp_path):
    conf = remote.read_config(ready_cfg(
        tmp_path, **{"remote.inbox": "~/my inbox; rm -rf ~"}))
    cmd = remote.query_command(conf, "inbox")
    assert "'~/my inbox; rm -rf ~'" in cmd


def test_a_question_on_an_unconfigured_lane_opens_nothing(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", NoPopen())
    res = remote.ask(remote.read_config(Cfg()), "up")
    assert res.ok is False and res.reason == "disabled"


# ==================================================== tier 2: pushing files
def test_a_push_can_only_ever_land_in_the_configured_inbox(tmp_path):
    """Speech never names a remote path.  This is the function that makes
    that true, so it is tested against a name that tries to escape."""
    conf = remote.read_config(ready_cfg(tmp_path))
    assert remote.inbox_target(conf, "b.txt") == "~/jarvis-inbox/b.txt"
    # a basename is taken even from something path-shaped
    assert remote.inbox_target(conf, "../../.ssh/authorized_keys") == \
        "~/jarvis-inbox/authorized_keys"


def test_push_refuses_a_file_outside_his_folders(tmp_path, desk, monkeypatch):
    monkeypatch.setattr(remote, "run_copy", lambda *a, **k:
                        pytest.fail("copied a file from outside the roots"))
    conf = remote.read_config(ready_cfg(tmp_path))
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    assert remote.push(conf, outside).reason == "outside"


def test_push_refuses_a_directory_and_a_fifo(tmp_path, desk, monkeypatch):
    monkeypatch.setattr(remote, "run_copy", lambda *a, **k:
                        pytest.fail("copied something that is not a file"))
    conf = remote.read_config(ready_cfg(tmp_path))
    (desk / "folder").mkdir()
    assert remote.push(conf, desk / "folder").reason == "not-a-file"
    os.mkfifo(desk / "pipe")
    assert remote.push(conf, desk / "pipe").reason == "not-a-file"


def test_push_refuses_a_file_past_the_cap(tmp_path, desk, monkeypatch):
    monkeypatch.setattr(remote, "run_copy", lambda *a, **k:
                        pytest.fail("copied an oversized file"))
    conf = remote.read_config(ready_cfg(tmp_path, **{"remote.max_mb": 1}))
    big = desk / "big.bin"
    big.write_bytes(b"0" * (2 * 1024 * 1024))
    assert remote.push(conf, big).reason == "too-big"


def test_a_good_push_uses_the_inbox_and_the_local_basename(tmp_path, desk,
                                                           monkeypatch):
    seen = {}

    def fake_copy(conf, local, rem, push):
        seen.update(local=local, remote=rem, push=push)
        return remote.SshResult(True)

    monkeypatch.setattr(remote, "run_copy", fake_copy)
    conf = remote.read_config(ready_cfg(tmp_path))
    f = desk / "budget.xlsx"
    f.write_text("x")
    assert remote.push(conf, f).ok
    assert seen["push"] is True
    assert seen["remote"] == "~/jarvis-inbox/budget.xlsx"


# ==================================================== tier 2: pulling files
def test_a_pulled_name_comes_from_the_remote_listing_not_the_transcript(
        tmp_path, monkeypatch):
    """The string handed to scp was produced by the far side, so a misheard
    word cannot become a path there."""
    monkeypatch.setattr(remote, "run_ssh", lambda conf, cmd, **k:
                        remote.SshResult(True, out="a.txt\nb.txt\nsub/\n"))
    conf = remote.read_config(ready_cfg(tmp_path))
    names, why = remote.list_remote(conf, "outbox")
    assert why == "" and names == ["a.txt", "b.txt"]        # the dir is dropped


def test_an_unquotable_remote_name_is_dropped_not_escaped(tmp_path, monkeypatch):
    """A file with a backtick or a newline in its name is not worth the class
    of bug that quoting it would invite."""
    monkeypatch.setattr(remote, "run_ssh", lambda conf, cmd, **k:
                        remote.SshResult(
                            True, out="ok.txt\n$(rm -rf ~).txt\n`id`.txt\n"
                                      "--flag.txt\n"))
    conf = remote.read_config(ready_cfg(tmp_path))
    names, _ = remote.list_remote(conf, "outbox")
    assert names == ["ok.txt"]


def test_pull_refuses_a_folder_that_is_not_on_the_allow_list(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "run_ssh", lambda *a, **k:
                        pytest.fail("listed a folder that is not allow-listed"))
    conf = remote.read_config(ready_cfg(tmp_path))
    assert remote.list_remote(conf, "etc")[1] == "not-there"


def test_pull_re_checks_the_name_even_though_the_caller_should_have(
        tmp_path, monkeypatch):
    """A seam that is only safe when its caller behaves is not a safe seam."""
    monkeypatch.setattr(remote, "run_copy", lambda *a, **k:
                        pytest.fail("copied an unsafe remote name"))
    conf = remote.read_config(ready_cfg(tmp_path))
    assert remote.pull(conf, "outbox", "../../.ssh/id_ed25519").reason == \
        "not-there"


def test_pull_never_clobbers_something_of_his(tmp_path, desk, monkeypatch):
    monkeypatch.setattr(remote, "run_copy", lambda *a, **k:
                        pytest.fail("overwrote a local file"))
    conf = remote.read_config(ready_cfg(tmp_path))
    (desk / "notes.txt").write_text("mine")
    assert remote.pull(conf, "outbox", "notes.txt").reason == "exists"
    assert (desk / "notes.txt").read_text() == "mine"


def test_a_pull_lands_in_his_first_root(tmp_path, desk):
    conf = remote.read_config(ready_cfg(tmp_path))
    assert remote.pull_target(conf, "b.txt") == desk / "b.txt"
    # even if the remote reported something path-shaped
    assert remote.pull_target(conf, "x/../y.txt") == desk / "y.txt"


# ================================================= resolving a spoken name
def test_two_similar_names_ASK_and_never_guess(desk):
    """The normal case in a Downloads folder, and picking the newer one
    silently is how the wrong file gets sent to someone."""
    for n in ("budget-2026.xlsx", "budget-final.xlsx"):
        (desk / n).write_text("x")
    pick = filepick.pick("budget", roots=[str(desk)])
    assert pick.ambiguous and pick.path is None
    assert set(p.name for p in pick.candidates) == \
        {"budget-2026.xlsx", "budget-final.xlsx"}
    assert " or " in filepick.describe(pick.candidates)


def test_one_clear_name_resolves_without_the_extension(desk):
    (desk / "budget.xlsx").write_text("x")
    (desk / "unrelated.png").write_text("x")
    assert filepick.pick("budget", roots=[str(desk)]).path.name == "budget.xlsx"


def test_a_name_he_half_remembers_matches_nothing_rather_than_the_wrong_file(desk):
    """There is no threshold at which a wrong file becomes acceptable."""
    (desk / "gadget.png").write_text("x")
    assert filepick.pick("budget", roots=[str(desk)]).reason == "not-found"


def test_an_explicit_path_outside_the_roots_is_refused(desk):
    assert filepick.pick("/etc/shadow", roots=[str(desk)]).reason == "outside"


def test_a_symlink_out_of_the_roots_is_refused_both_ways(desk):
    """Two layers: the walk never offers a symlink, and an explicit path is
    checked after resolve().  Either alone would leave a hole."""
    os.symlink("/etc/passwd", desk / "evil.txt")
    assert filepick.pick("evil", roots=[str(desk)]).reason == "not-found"
    assert filepick.pick(str(desk / "evil.txt"),
                         roots=[str(desk)]).reason == "outside"


def test_dotfiles_and_key_folders_are_never_offered(desk):
    (desk / ".env").write_text("SECRET=1")
    ssh = desk / ".ssh"
    ssh.mkdir()
    (ssh / "id_ed25519").write_text("k")
    assert filepick.pick("env", roots=[str(desk)]).reason == "not-found"
    assert filepick.pick("id_ed25519", roots=[str(desk)]).reason == "not-found"


def test_a_folder_is_not_a_file(desk):
    """Two layers again: the walk only ever yields files, so a spoken folder
    name is not even a candidate ("send me the Desktop" must never become a
    recursive copy), and an EXPLICIT path to one is refused by check_file."""
    (desk / "Projects").mkdir()
    assert filepick.pick("Projects", roots=[str(desk)]).reason == "not-found"
    assert filepick.pick(str(desk / "Projects"),
                         roots=[str(desk)]).reason == "not-a-file"


def test_an_empty_name_asks_rather_than_scanning(desk):
    assert filepick.pick("  ", roots=[str(desk)]).reason == "empty"


def test_every_refusal_reason_has_a_line():
    for reason in filepick.REASONS:
        line = filepick.reason_line(reason, 100, 250)
        assert line and "{" not in line


# ================================================ the spoken commands
POSITIVE = [
    ("put the budget on HPCOMPUTER", "remote push"),
    ("send this file to hp computer", "remote push"),
    ("copy budget.xlsx over to the desktop machine", "remote push"),
    ("get the budget from HPCOMPUTER", "remote pull"),
    ("grab that file off hp computer", "remote pull"),
    ("fetch the report from HPCOMPUTER's desktop", "remote pull"),
    ("is HPCOMPUTER up", "remote status"),
    ("how's hp computer", "remote status"),
    ("is the hp awake", "remote status"),
    ("what's the disk on HPCOMPUTER", "remote query"),
    ("who's logged in on hp computer", "remote query"),
    ("what's in the inbox on HPCOMPUTER", "remote query"),
    ("run the build on HPCOMPUTER", "remote freeform"),
    ("delete the logs on the hp", "remote freeform"),
]


def _ungated():
    """REGISTRY as the dispatcher sees it with no optional services wired.

    `Command("workflow", lambda t: True, ...)` matches EVERY utterance and is
    held back only by ``needs=("workflows",)`` -- the dispatcher skips a
    command whose services are absent (commander.py, `self._svc(n) is None`).
    A naive walk of REGISTRY would therefore hand it every phrase here and
    prove nothing, so the service-gated entries are dropped exactly as the
    dispatcher drops them.  The remote commands declare no needs, so they
    are never dropped and this still tests them against every ungated rival
    -- oracle, "run shell", "find file" and the rest.
    """
    return [c for c in cmd_mod.REGISTRY if not getattr(c, "needs", ())]


@pytest.mark.parametrize("said,name", POSITIVE)
def test_the_spoken_forms_reach_the_command_they_should(said, name):
    for c in _ungated():
        if c.matcher(said):
            assert c.name == name, f"{said!r} went to {c.name}"
            return
    pytest.fail(f"{said!r} matched nothing")


# The NEGATIVE table.  Every line here is a phrase that must NOT reach this
# lane, and most of them are the reason a rule in the module exists.
NEGATIVE = [
    # "my desktop" is the FOLDER -- his own words for the mail lane are
    # "this file ... on my desktop".  Reading it as the host would put a
    # file on another machine when he asked to move it two inches.
    "put this on my desktop",
    "save it to my desktop",
    "move the file to my desktop",
    "what's on my desktop",
    # the mail lane's sentence, not this one
    "email this file to Dana",
    "send the budget to Dana",
    # the Oracle lane
    "is the oracle box up",
    "restart haymaker on the oracle box",
    # plain conversation and the rest of the app
    "put the kettle on",
    "how's the printer",
    "turn the lights on",
    "what's the weather on Thursday",
]


@pytest.mark.parametrize("said", NEGATIVE)
def test_the_negative_table_never_reaches_this_lane(said):
    for c in _ungated():
        if c.matcher(said):
            assert not c.name.startswith("remote "), \
                f"{said!r} was taken by {c.name}"
            return


def test_a_mention_of_the_host_is_not_an_order(monkeypatch):
    """"The music's playing on HPCOMPUTER" is conversation.  The refusal door
    only fires on words that read as an instruction; everything else goes to
    the model, because refusing a remark would be worse than answering it."""
    m = cmd_mod._REMOTE_FREEFORM_RX.match("the music's playing on HPCOMPUTER")
    assert m is not None
    said = cmd_mod._oracle_group(m, "cmd", "cmd2", "cmd3")
    assert not cmd_mod._REMOTE_ORDER_RX.search(said)


@pytest.mark.parametrize("said", [
    "run the build on HPCOMPUTER", "delete the logs on the hp",
    "sudo rm -rf on HPCOMPUTER", "shut down hp computer",
    "install docker on the desktop machine",
])
def test_an_order_aimed_at_the_host_is_refused_and_never_run(said):
    """The safety argument in one test: there is no path from speech to a
    shell on the other machine.  "delete the logs" and "delete the block"
    differ by one phoneme and only one of them is recoverable."""
    m = cmd_mod._REMOTE_FREEFORM_RX.match(said)
    assert m is not None
    spoken = cmd_mod._oracle_group(m, "cmd", "cmd2", "cmd3")
    assert cmd_mod._REMOTE_ORDER_RX.search(spoken)


def test_the_refusal_says_what_it_will_do_instead():
    line = remote.FREEFORM_REFUSAL.format(name="HPCOMPUTER")
    assert "don't run loose commands" in line
    assert "fetch files" in line and "{" not in line


# ======================================= coordination with the mail lane
def test_the_two_lanes_share_one_resolver_and_do_not_duplicate_it():
    """`filepick` is the shared base: it ranks names and vets paths for BOTH
    lanes.  The mail lane's `filephrase` is a PHRASE layer on top of it, not
    a second resolver -- two resolvers that disagreed about "that file" is
    the failure the shared module exists to prevent."""
    filephrase = pytest.importorskip("jarvis.filephrase")
    assert filephrase.filepick is filepick
    assert filephrase.SKIP_DIRS is filepick.SKIP_DIRS


def test_this_lane_understands_his_phrase_shapes_through_that_layer(desk, tmp_path):
    """"Put that file on my desktop on HPCOMPUTER" -- a folder is the handle
    and there is no name at all.  His own phrasing, so it resolves.

    Skipped rather than failed when the phrase layer is absent: this lane
    falls back to the shared resolver alone and still works on plain names,
    which is the contract _remote_resolve_local documents."""
    pytest.importorskip("jarvis.filephrase")
    (desk / "lab-report.pdf").write_text("x")
    conf = remote.read_config(ready_cfg(tmp_path))
    for said in ("lab report", "that file on my desktop", "the PDF on my desktop"):
        got = cmd_mod._remote_resolve_local(said, conf)
        assert got.ok and got.path.name == "lab-report.pdf", said


def test_the_push_regex_hands_the_phrase_over_whole():
    m = cmd_mod._REMOTE_PUSH_RX.match("put that file on my desktop on HPCOMPUTER")
    assert m is not None and m.group("what") == "file on my desktop"


def test_this_lane_keeps_hard_containment_where_the_mail_lane_relaxes_it(
        desk, tmp_path):
    """The mail lane lets him attach a path he names outright from anywhere;
    a push has a second machine's filesystem on the far end, so this lane
    passes allow_explicit_outside=False and "/etc/shadow" stays unsayable."""
    conf = remote.read_config(ready_cfg(tmp_path))
    assert cmd_mod._remote_resolve_local("/etc/shadow", conf).reason == "outside"
