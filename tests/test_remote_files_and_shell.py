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
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis import commander as cmd_mod
from jarvis.assistant_config import DEFAULTS
from jarvis.commander import Commander, IntentClassifier
from jarvis.config import CONFIG
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
    """A config that passes missing_reason -- enabled, host, user, and a key
    file that exists.  Until F11 (2026-09-03) this fixture said "no key" and
    the lane called that ready; a blank key is a refusal now, so the fixture
    writes one under tmp_path rather than lean on the box's ~/.ssh."""
    key = tmp_path / "hpcomputer.key"
    if not key.exists():
        key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    base = {"remote.enabled": True,
            "remote.host": "hpcomputer.tail5323b8.ts.net",
            "remote.user": "hunterp",
            "remote.key_path": str(key),
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
    # The shipped default.  This row used to expect "" (ready) -- F11.
    ({"remote.enabled": True, "remote.host": "h", "remote.user": "u"}, "no-key"),
])
def test_missing_reason_names_exactly_what_is_absent(over, expect):
    """A present host with a missing user is a real state, and it must be
    named before a socket opens rather than becoming an auth failure."""
    assert remote.missing_reason(remote.read_config(Cfg(**over))) == expect


def test_a_key_that_is_there_makes_the_lane_ready(tmp_path):
    assert remote.missing_reason(remote.read_config(ready_cfg(tmp_path))) == ""


def test_a_blank_key_path_is_refused_before_a_socket_opens(monkeypatch):
    """F11 (2026-09-03).  DEFAULTS and the docs example ship key_path "",
    and missing_reason called that ready.  ssh then ran with
    IdentitiesOnly=yes and no -i, which offers only the default-named
    identities -- and ~/.ssh here holds none (measured: no id_rsa, id_ecdsa
    or id_ed25519; the key is ~/.ssh/hpcomputer).  So the first thing he
    heard after enabling the lane was "HPCOMPUTER turned my key away, sir"
    -- the far side blamed for a blank line in his own settings, after a
    socket had opened.  A blank key is a refusal that names remote.key_path."""
    monkeypatch.setattr(subprocess, "Popen", NoPopen())
    conf = remote.read_config(Cfg(**{"remote.enabled": True,
                                     "remote.host": "192.168.50.114",
                                     "remote.user": "h2pey",
                                     "remote.key_path": ""}))
    assert remote.missing_reason(conf) == "no-key"
    assert remote.ask(conf, "up").reason == "no-key"
    assert remote.push(conf, Path("/x")).reason == "no-key"
    assert remote.pull(conf, "outbox", "a.txt").reason == "no-key"
    line = remote.fail_line(conf, "no-key")
    assert "remote.key_path" in line and "turned my key away" not in line


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
    for reason in ("disabled", "no-host", "no-user", "no-key", "bad-key",
                   "no-ssh", "off-tailnet", "asleep", "timeout", "auth",
                   "unreachable", "hostkey", "no-space", "denied", "not-there",
                   "exists", "failed"):
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
    """Quoted, but NOT the leading tilde.

    This test used to pin the bug: it asserted `'~/my inbox; rm -rf ~'`, and
    a quoted tilde is never expanded, so the whole thing was one literal
    directory name that could not exist.  Measured on this box 2026-09-03:
    `sh -c "ls -1p -- '~/x'"` fails "No such file or directory" (rc 2) while
    the $HOME form resolves -- and that stderr hits _MISSING_RX, so Jarvis
    reported a config bug as "I couldn't find that on HPCOMPUTER, sir."
    Every character that came from the config file is still inside single
    quotes; only the tilde is outside them."""
    conf = remote.read_config(ready_cfg(
        tmp_path, **{"remote.inbox": "~/my inbox; rm -rf ~"}))
    cmd = remote.query_command(conf, "inbox")
    assert '"$HOME"/\'my inbox; rm -rf ~\'' in cmd
    assert "'~/my inbox" not in cmd


@pytest.mark.parametrize("path,expect", [
    ("~/jarvis-outbox", '"$HOME"/jarvis-outbox'),
    ("~/my inbox; rm -rf ~", '"$HOME"/\'my inbox; rm -rf ~\''),
    ("~", '"$HOME"'),
    ("~/", '"$HOME"'),
    ("/srv/drop box", "'/srv/drop box'"),
    ("relative/dir", "relative/dir"),
])
def test_a_tilde_never_reaches_the_remote_shell_inside_quotes(path, expect):
    assert remote.shell_path(path) == expect


@pytest.mark.parametrize("path,expect", [
    ("~/jarvis-inbox", "jarvis-inbox"),
    ("~", "."),
    ("/srv/drop", "/srv/drop"),
])
def test_scp_gets_a_home_relative_path_because_it_is_not_a_shell(path, expect):
    """scp on OpenSSH 9.6 (this box) speaks SFTP, which has no tilde and
    resolves a relative path against the login home; the legacy -O protocol
    runs a shell whose cwd is that same home.  Relative is right in both."""
    assert remote.scp_path(path) == expect


def test_every_shipped_remote_directory_survives_the_round_trip():
    """The shipped config is all "~" paths, which is the state that made
    this a bug rather than a theory."""
    row = DEFAULTS["remote"]
    for path in [row["inbox"]] + list(row["pull_dirs"].values()):
        assert path.startswith("~"), path
        assert "'~" not in remote.shell_path(path)
        assert not remote.scp_path(path).startswith("~")


def test_a_question_on_an_unconfigured_lane_opens_nothing(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", NoPopen())
    res = remote.ask(remote.read_config(Cfg()), "up")
    assert res.ok is False and res.reason == "disabled"


# ==================================================== tier 2: pushing files
def test_a_push_can_only_ever_land_in_the_configured_inbox(tmp_path):
    """Speech never names a remote path.  This is the function that makes
    that true, so it is tested against a name that tries to escape."""
    conf = remote.read_config(ready_cfg(tmp_path))
    # Home-relative, not "~/...": see scp_path.  scp is not a shell.
    assert remote.inbox_target(conf, "b.txt") == "jarvis-inbox/b.txt"
    # a basename is taken even from something path-shaped
    assert remote.inbox_target(conf, "../../.ssh/authorized_keys") == \
        "jarvis-inbox/authorized_keys"


@pytest.mark.parametrize("name", [
    "note;rm -rf ~.txt", "back`id`.txt", "two\nlines.txt", "-rf.txt",
    "$(id).txt", "quote'.txt",
])
def test_a_local_name_a_remote_shell_could_read_is_refused_not_escaped(
        tmp_path, name):
    """The pull side has always vetted names with SAFE_REMOTE_NAME_RX and the
    push side did not: the LOCAL basename went into the remote argument
    unescaped.  Harmless while scp speaks SFTP (no remote shell); remote
    command execution under -O, an older scp, or a different SCP_BIN.  It is
    refused with a sentence, the way an unquotable remote name is dropped."""
    conf = remote.read_config(ready_cfg(tmp_path))
    assert remote.inbox_target(conf, name) == ""


def test_a_push_of_such_a_name_never_opens_a_transfer(tmp_path, desk,
                                                      monkeypatch):
    monkeypatch.setattr(remote, "run_copy", lambda *a, **k:
                        pytest.fail("copied a name I will not write remotely"))
    conf = remote.read_config(ready_cfg(tmp_path))
    f = desk / "back`id`.txt"
    f.write_text("x")
    assert remote.push(conf, f).reason == "odd-name"
    assert remote.fail_line(conf, "odd-name").startswith("That file's name")


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
    assert seen["remote"] == "jarvis-inbox/budget.xlsx"


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


# ==================================================================
# The spoken lane, end to end -- no socket, no host, no tailnet
# ==================================================================
# `remote.run_ssh`, `remote.run_copy` and `remote.tailnet_state` are the
# three seams and every test below replaces all three.  What is under test
# is the DECISIONS: which door a sentence reaches, what has to be said to
# spend a transfer read-back, and which channel may say it.
@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A Commander whose HPCOMPUTER lane is configured and whose transport
    records instead of connecting."""
    desk = tmp_path / "Desktop"
    desk.mkdir(exist_ok=True)
    for name, body in (("budget.xlsx", b"B" * 400),
                       ("lab_report.pdf", b"L" * 400),
                       ("lab_report_final.pdf", b"F" * 400)):
        (desk / name).write_bytes(body)

    copied = []
    monkeypatch.setattr(remote, "run_copy",
                        lambda conf, local, rem, push:
                        (copied.append((local, rem, push)),
                         remote.SshResult(True))[1])
    monkeypatch.setattr(remote, "run_ssh", lambda conf, cmd, **k:
                        remote.SshResult(True, out="report_v1.pdf\n"
                                                   "report_v2.pdf\n"))
    monkeypatch.setattr(remote, "tailnet_state", lambda conf: "online")
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "i.json")
    monkeypatch.setattr(Commander, "FEEDBACK_LOG", tmp_path / "fb.jsonl",
                        raising=False)
    for key, val in (("voice_cmds", True), ("jarvis_mode", True),
                     ("auto_type", False), ("talkback", False)):
        monkeypatch.setattr(CONFIG, key, val)
    # _bg runs inline, as in test_oracle: the four answering doors hand
    # their round trip to a thread (F10), and a test that sleeps for one
    # flakes.  What the worker SAYS is recorded from _speak_now, and what
    # it puts on the strip from the bus.
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    spoken, statuses = [], []
    monkeypatch.setattr(Commander, "_speak_now",
                        lambda self, text: spoken.append(text) or True)
    from jarvis.events import Status, bus
    bus.subscribe(Status, lambda e: statuses.append(e.text))
    svc = types.SimpleNamespace(
        assistant=ready_cfg(tmp_path), memory=MagicMock(), desktop=MagicMock(),
        workflows=MagicMock(), brain=MagicMock(), context=MagicMock(),
        tts=MagicMock(), timekeeper=MagicMock(), notes=MagicMock(),
        approvals=MagicMock(), claude=MagicMock(), router=MagicMock())
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.memory.resolve_person.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.timekeeper.ringing = None
    svc.approvals.pending.return_value = []
    svc.router.pending.return_value = None
    svc.claude.active_project = "jarvis"
    c = Commander(svc)
    c.copied, c.spoken, c.statuses = copied, spoken, statuses
    return c


# ---------------------------------------------- reachable by voice at all
@pytest.mark.parametrize("name", ["remote push", "remote pull",
                                  "remote status", "remote query",
                                  "remote freeform"])
def test_every_door_is_registered_for_unprefixed_speech(name):
    """The family copied the Oracle five's registry shape everywhere except
    here, and ASSISTANT_TIER1 is the list that makes a command reachable
    once the hotword has eaten the wake word.  Without it `cmd_text` is
    None, the registry pass is skipped, and the intent gate drops or
    mis-routes every phrasing."""
    assert name in [c.name for c in cmd_mod.ASSISTANT_TIER1]


@pytest.mark.parametrize("said,expect", [
    ("put the lab report on HPCOMPUTER", "Which one?"),
    ("is HPCOMPUTER up", "HPCOMPUTER: up"),
    ("what's the disk on HPCOMPUTER", "HPCOMPUTER: disk"),
    ("get the report from HPCOMPUTER", "Which one?"),
    ("run the build on HPCOMPUTER", "Refused"),
])
def test_the_doors_answer_bare_voice_with_no_wake_word(wired, said, expect):
    """The live path: the hotword consumed "jarvis", so this is what the
    commander actually receives.  A door that answers from its worker (F10)
    puts the answer's status on the strip rather than on the turn's result,
    so either place counts -- the refusal and a read-back are still on the
    turn, the round trips are not."""
    res = wired.handle(said, source="voice")
    assert res.handled
    assert expect in [res.status] + wired.statuses, \
        f"{said!r} -> {res.status!r} / {res.reply!r} / {wired.statuses!r}"


def test_a_push_reads_back_and_moves_nothing(wired):
    res = wired.handle("put the budget on HPCOMPUTER", source="voice")
    assert res.reply == "Send budget.xlsx to HPCOMPUTER's inbox, sir?"
    assert wired.copied == [] and wired.question_open()


# ------------------------------------------ the transfer read-back is strict
def test_a_stray_yeah_cannot_push_a_file(wired):
    """The exact line the mail lane pins as a regression
    (test_send_file.py::test_a_stray_yeah_to_something_else_cannot_send_the_file).
    It answered a live offer once already; parse_yes_no waives its
    overheard-speech guard whenever the first word is a yes word, and this
    lane's own design calls a transfer "a file on another machine" -- as
    irreversible as an email."""
    wired.handle("put the budget on HPCOMPUTER", source="voice")
    wired.handle("Yeah, so you should be able to look that up.", source="voice")
    assert wired.copied == []
    assert wired._pending_destructive is None, "and the offer is spent"


def test_sure_is_asked_again_rather_than_obeyed(wired):
    """"sure" is in _YES_WORDS and deliberately absent from the send lane's
    grammar: it is what a man says while still listening."""
    wired.handle("put the budget on HPCOMPUTER", source="voice")
    res = wired.handle("sure", source="voice")
    assert res.reply == outbox_unsure()
    assert wired.copied == []
    res = wired.handle("yes", source="voice")
    # The yes turn acknowledges; the outcome is spoken from the worker (F10).
    assert res.reply == "Sending it now, sir." and res.ack and res.done is False
    assert wired.spoken[-1] == "budget.xlsx is on HPCOMPUTER, sir."
    assert wired.copied[-1][2] is True


def outbox_unsure():
    from jarvis import outbox
    return outbox.UNSURE_LINE


@pytest.mark.parametrize("said", ["yes", "yes please", "go ahead", "send it"])
def test_a_real_yes_still_spends_it(wired, said):
    wired.handle("put the budget on HPCOMPUTER", source="voice")
    wired.handle(said, source="voice")
    assert len(wired.copied) == 1


@pytest.mark.parametrize("said", ["no", "cancel", "not that one"])
def test_a_no_drops_it(wired, said):
    wired.handle("put the budget on HPCOMPUTER", source="voice")
    res = wired.handle(said, source="voice")
    assert res.status == "Dropped" and wired.copied == []


@pytest.mark.parametrize("src", ["discord", "phone", "socket", "cli"])
def test_a_yes_from_another_room_moves_no_file(wired, src):
    wired.handle("put the budget on HPCOMPUTER", source="voice")
    wired.handle("yes", source=src)
    assert wired.copied == [], f"a {src} yes pushed the file"
    # left parked: that turn is not its answer, nor its cancellation
    assert wired._pending_destructive is not None
    wired.handle("yes", source="voice")
    assert len(wired.copied) == 1


def test_a_pull_is_just_as_strict(wired):
    wired.handle("get report v1 from HPCOMPUTER", source="voice")
    wired.handle("Yeah, so you should be able to look that up.", source="voice")
    assert wired.copied == []


# ------------------------------------------------ "Which one?" is answerable
@pytest.mark.parametrize("answer,expect", [
    ("the second one", "lab_report_final.pdf"),
    ("the final one", "lab_report_final.pdf"),
    ("the first one", "lab_report.pdf"),
])
def test_an_ambiguous_push_hears_its_answer(wired, answer, expect):
    res = wired.handle("put the lab report on HPCOMPUTER", source="voice")
    assert "Which one?" in res.reply and wired.question_open()
    res = wired.handle(answer, source="voice")
    assert res.reply == f"Send {expect} to HPCOMPUTER's inbox, sir?"
    assert wired.copied == [], "choosing a file confirms nothing"
    wired.handle("yes", source="voice")
    assert wired.copied[-1][1] == f"jarvis-inbox/{expect}"


def test_an_ambiguous_pull_hears_its_answer(wired):
    wired.handle("get the report from HPCOMPUTER", source="voice")
    # The listing is a round trip, so "Which one?" is spoken from the
    # worker (F10); the answer's read-back needs no round trip and is on
    # the turn.
    asked = wired.spoken[-1]
    assert "report_v1.pdf" in asked and "report_v2.pdf" in asked
    assert wired.question_open()
    res = wired.handle("the second one", source="voice")
    assert "report_v2.pdf" in res.reply
    assert wired.copied == []


# ------------------------------------------- the refusal door's manners
REFUSAL_MUST_NOT_CLAIM = [
    # questions and reports that merely CONTAIN an order word
    "did you install anything on the HP",
    "have you run the tests on the HP",
    "the build failed on the HP",
    "i need to update the HP",
    "i should install python on the HP",
]


@pytest.mark.parametrize("said", REFUSAL_MUST_NOT_CLAIM)
def test_a_question_about_the_host_is_not_an_order_to_it(wired, said):
    res = wired.handle(said, source="voice")
    assert res.status != "Refused", f"{said!r} was refused out loud"
    assert remote.FREEFORM_REFUSAL.format(name="HPCOMPUTER") != res.reply


@pytest.mark.parametrize("said,order", [
    ("run the build on HPCOMPUTER", True),
    ("delete the logs on the hp", True),
    ("please restart the hp computer", True),
    ("shut down HPCOMPUTER", True),
    ("did you install anything on the HP", False),
    ("the build failed on the HP", False),
    ("i need to update the HP", False),
    ("remind me to run the backup on the HP", False),
])
def test_an_order_is_a_verb_at_the_head_of_the_clause(said, order):
    """_REMOTE_ORDER_RX used to be an unanchored `search`, so any clause
    CONTAINING one of its words read as an instruction.  An order is an
    imperative: the verb comes first."""
    m = cmd_mod._REMOTE_FREEFORM_RX.match(said)
    assert m is not None
    spoken = cmd_mod._oracle_group(m, "cmd", "cmd2", "cmd3")
    assert bool(cmd_mod._REMOTE_ORDER_RX.match(spoken.strip())) is order


def test_the_refusal_door_no_longer_outranks_the_commands_it_shadowed():
    """It is the loosest matcher in the family, and at its old index it
    took "clear the shopping list on my desktop computer" (a list) and
    "remind me to run the backup on the HP" (a reminder) before either
    could be answered."""
    names = [c.name for c in cmd_mod.REGISTRY]
    assert names.index("remote freeform") > names.index("list add")
    assert names.index("remote freeform") > names.index("remind me")
    # ...and still after the four doors that DO something.
    for door in ("remote push", "remote pull", "remote status", "remote query"):
        assert names.index("remote freeform") > names.index(door)


def test_a_reminder_that_names_the_machine_is_a_reminder():
    """"Remind me to run the backup on the HP" was answered "I don't run
    loose commands on HPCOMPUTER, sir." -- a reminder he asked for and did
    not get.  Two independent fixes now stop it: the refusal door is below
    `remind me` in REGISTRY, and "run" in the middle of a clause is no
    longer an order.  Both are asserted, so removing either is a failure.
    ("workflow" is skipped: its matcher accepts every utterance and it is
    held back only by needs=("workflows",).)"""
    said = "remind me to run the backup on the hp"
    hits = [c.name for c in cmd_mod.REGISTRY
            if c.name != "workflow" and c.matcher(said)]
    assert hits and hits[0] == "remind me", hits
    m = cmd_mod._REMOTE_FREEFORM_RX.match(said)
    spoken = cmd_mod._oracle_group(m, "cmd", "cmd2", "cmd3")
    assert not cmd_mod._REMOTE_ORDER_RX.match(spoken.strip())


# ==================================================================
# F10 (2026-09-03): nothing that opens a socket runs on the voice turn
# ==================================================================
# A confirmed push ran remote.push INLINE from _try_destructive_confirm
# (`pend[0]()`), on the _process_audio thread with app._audio_busy set, for
# up to transfer_timeout_s (120 s default, 900 s clamp); the status, query
# and pull doors each held the turn for a full ssh budget (12 s) the same
# way.  For all of it Jarvis was silent and deaf -- no "stop", no "never
# mind", no wake word (app.py gates wake/recording on _audio_busy).  The
# Oracle lane this family says it copies hands its round trip to c._bg and
# returns done=False; so does the send lane.  These pin that shape: _bg is
# replaced with a QUEUE, and the transport must not have been touched by
# the time handle() returns.
@pytest.fixture
def queued(wired, monkeypatch):
    """The wired Commander with _bg parked instead of inlined."""
    jobs = []
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: jobs.append(fn))
    wired.jobs = jobs
    return wired


def test_a_confirmed_push_leaves_the_voice_turn_at_once(queued):
    queued.handle("put the budget on HPCOMPUTER", source="voice")
    res = queued.handle("yes", source="voice")
    assert queued.copied == [], "the copy ran on the voice turn"
    assert res.handled and res.ack and res.done is False
    assert res.reply == "Sending it now, sir." and res.speak
    assert len(queued.jobs) == 1
    queued.jobs[0]()
    assert queued.copied[-1][2] is True
    assert queued.spoken[-1] == "budget.xlsx is on HPCOMPUTER, sir."


def test_a_confirmed_pull_leaves_the_voice_turn_at_once(queued):
    queued.handle("get report v1 from HPCOMPUTER", source="voice")
    queued.jobs.pop()()                       # the listing, from its worker
    assert queued.spoken[-1].startswith("Bring report_v1.pdf from HPCOMPUTER")
    res = queued.handle("yes", source="voice")
    assert queued.copied == [], "the copy ran on the voice turn"
    assert res.ack and res.done is False and res.reply == "Fetching it now, sir."
    queued.jobs.pop()()
    assert queued.copied[-1][2] is False
    assert queued.spoken[-1] == "report_v1.pdf is on your Desktop, sir."


@pytest.mark.parametrize("said,expect", [
    ("is HPCOMPUTER up", "HPCOMPUTER: up"),
    ("what's the disk on HPCOMPUTER", "HPCOMPUTER: disk"),
    ("get the report from HPCOMPUTER", "Which one?"),
])
def test_a_remote_round_trip_never_runs_on_the_voice_turn(queued, monkeypatch,
                                                          said, expect):
    asked = []
    monkeypatch.setattr(remote, "run_ssh", lambda conf, cmd, **k:
                        asked.append(cmd) or
                        remote.SshResult(True, out="report_v1.pdf\n"
                                                   "report_v2.pdf\n"))
    res = queued.handle(said, source="voice")
    assert asked == [], f"{said!r} opened ssh on the voice turn"
    assert res.handled and res.done is False and not res.reply
    assert len(queued.jobs) == 1
    queued.jobs[0]()
    assert len(asked) == 1
    assert queued.statuses[-1] == expect
    assert queued.spoken, "the worker said nothing"


def test_a_worker_outcome_goes_through_the_app_door_when_there_is_one(queued):
    """The send lane's F20 lesson, kept here by name: bus + _speak_now show
    and speak but do not CLOSE a done=False turn, so after the outcome the
    wake word stayed dead until the 60 s watchdog.  services.reply
    (JarvisApp._async_reply) shows, speaks, closes the turn and arms the
    follow-up window; when it is wired, the worker must use it and nothing
    else."""
    delivered = []
    queued.services.reply = lambda text, speak=True: delivered.append((text, speak))
    queued.handle("is HPCOMPUTER up", source="voice")
    queued.jobs[0]()
    assert delivered and delivered[-1][0].startswith("HPCOMPUTER is up")
    assert delivered[-1][1] is True
    assert queued.spoken == [], "spoken twice: once per door"


def test_a_worker_that_blows_up_still_closes_the_turn(queued, monkeypatch):
    monkeypatch.setattr(remote, "ask", lambda conf, key:
                        (_ for _ in ()).throw(RuntimeError("boom")))
    queued.handle("is HPCOMPUTER up", source="voice")
    queued.jobs[0]()
    assert queued.spoken[-1] == remote.fail_line(
        remote.read_config(queued.services.assistant), "failed")
