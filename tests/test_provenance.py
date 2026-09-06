"""WHERE EVERY NUMBER IN THIS LANE CAME FROM, pinned so it cannot drift.

This project has twice been damaged by a confident number with no source, and
round 7 found a third: ``jarvis/tools/remote.py`` reported FIRST-PERSON
measurements against HPCOMPUTER with wire-level detail -- "Measured against
HPCOMPUTER itself (OpenSSH_for_Windows_9.5, SFTP protocol 3): a 22222-byte
file at the target name came back 1111 bytes, exit 0" -- and nothing in the
repository can show a session that produced them.  What is actually known:

  * docs/capabilities.md records HPCOMPUTER measured 2026-09-03 as ARP
    REACHABLE, ping 100% loss, 22/445/3389/5900/8008/2343 CLOSED.  Port 22
    shut means no SSH session was possible that day.
  * He set OpenSSH up on Windows on 2026-09-05 and the Inbox/Outbox link was
    proved with a real file, so a session became possible today.
  * The SAME FILE, around line 715, says the Windows rows were NOT measured,
    "by instruction the first live command is Hunter's".  It said both
    things at once, and that is the defect.
  * The claim landed in 36d26fe, written 2026-09-05.

THE FIX IS NOT TO DELETE THE NUMBERS.  They are right.  Every one of them was
RE-MEASURED HERE on 2026-09-05, against this box's own
/usr/lib/openssh/sftp-server driven over a pipe by ``scp -D`` and ``sftp -D``
-- no socket, no network, nothing of his touched:

    scp default (SFTP)   22222-byte target -> 1111 bytes, exit 0, nothing
                         on stdout or stderr
    scp -O (legacy RCP)  22222 -> 1111, exit 0, silent.  Measured through a
                         stand-in for ssh that runs the far-side command
                         locally, because -O does not use -D.
    sftp put             22222 -> 1111, exit 0
    bare `rename`        destroyed the file at the target name 10 of 10,
                         exit 0.  On the wire: the client uses
                         posix-rename@openssh.com when the server
                         advertises it, and this one does.
    `rename -l`          refused 10 of 10, exit 1,
                         `remote rename "...": Failure`, both files intact.
                         On the wire: SSH2_FXP_RENAME.
    concurrent claim     two sessions renaming their own temp onto ONE name,
                         12 rounds: 7 wins / 5 wins, ZERO rounds where both
                         moved, ZERO corrupt, the loser's temp intact 12/12.
    client flag set      `scp` here is OpenSSH_9.6p1 and its usage line is
                         [-346ABCOpqRrsTv] [-c] [-D] [-F] [-i] [-J] [-l]
                         [-o] [-P] [-S] [-X].  No no-clobber among them.

SO THE THREE PROVENANCES, THREE DIFFERENT WORDS, and this file is what keeps
them apart:

  MEASURED HERE   the protocol behaviour and the client's flags, run on this
                  box against its own sftp-server.  The CLIENT is ours and
                  runs here, so its flag set is a local fact outright.
  INFERRED        what the Windows SERVER does.  No session with HPCOMPUTER
                  can be shown, so `rename -l` refusing over there is read
                  off the protocol (SSH_FXP_RENAME, opcode 18, which the
                  draft says SHOULD fail when the target exists) and off
                  Windows' own MoveFile/CreateDirectory semantics.  It is
                  ALSO why the lane pins `-l`: that flag fixes the opcode
                  whatever the far server advertises, so the safety does not
                  rest on the inference.
  HIS WORD        the mkdir refusal.  He was asked directly and answered
                  YES.  Round 6 attributed that correctly and pinned it in
                  both directions; this file keeps doing that.

The tests below re-derive the local numbers rather than quoting them, so a
number here cannot become a claim nobody can check.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SFTP_SERVER = "/usr/lib/openssh/sftp-server"

# Every file in this lane that carries one of these numbers.
SOURCES = ("jarvis/tools/remote.py", "jarvis/foldersync.py",
           "tests/test_foldersync.py", "tests/test_remote_files_and_shell.py")

# A sentence that claims somebody ran a command against his Windows box and
# watched the result.  None may exist: no such session can be shown.
FIRST_PERSON_HPCOMPUTER = re.compile(
    r"[Mm]easured (against|on) HPCOMPUTER"
    r"|HPCOMPUTER itself"
    r"|OpenSSH_for_Windows[^\n]*\)\s*:\s*a \d+-byte"
    r"|measured on his box"
    r"|against the real box")


def _text(rel: str) -> str:
    return (REPO / rel).read_text()


def test_no_file_claims_a_first_person_measurement_against_his_windows_box():
    """THE DEFECT.  Port 22 was recorded CLOSED on 2026-09-03 and no probe of
    HPCOMPUTER is allowed from here at all, so a sentence reporting a
    wire-level result from it is a number with no source -- the third one
    this project has had.  Every such sentence must name where it really
    came from instead."""
    bad = []
    for rel in SOURCES:
        for i, line in enumerate(_text(rel).splitlines(), 1):
            if FIRST_PERSON_HPCOMPUTER.search(line):
                bad.append(f"{rel}:{i}: {line.strip()}")
    assert not bad, (
        "these sentences claim a session with HPCOMPUTER that nobody can "
        "show:\n  " + "\n  ".join(bad) +
        "\nSay MEASURED HERE (this box's own sftp-server), INFERRED (from "
        "the protocol), or HIS WORD -- three provenances, three different "
        "words.")


def test_the_truncation_numbers_name_the_box_they_were_run_on():
    """The numbers stay; what changes is that they say where they came
    from.  A bare "22222 -> 1111" with no attribution is how this went
    wrong."""
    for rel in SOURCES:
        text = _text(rel)
        if "22222" not in text:
            continue
        assert ("sftp-server" in text or "on this box" in text
                or "MEASURED HERE" in text), (
            f"{rel} carries the truncation numbers and does not say they "
            "were re-measured locally against this box's own sftp-server")


def test_the_mkdir_refusal_is_still_his_word_and_not_a_measurement():
    """Round 6 got this one right and pinned it in BOTH directions; round 7
    keeps the pin.  Upgrading his sentence to a measurement is as wrong as
    losing it."""
    text = _text("jarvis/tools/remote.py")
    assert "CONFIRMED BY HIM, 2026-09-05" in text
    assert "he\n# answered YES" in text or "answered YES" in text
    assert "THAT IS NOT A LOCAL MEASUREMENT" in text, (
        "the sentence that stops his word being read as a measurement has "
        "gone")


def test_the_windows_query_rows_are_still_marked_unmeasured():
    """The other half of the contradiction.  remote.py said the Windows rows
    were NOT measured 'by instruction' AND reported wire-level Windows
    measurements 160 lines later.  The honest half must survive."""
    text = _text("jarvis/tools/remote.py")
    assert "NOT\n# measured: the Windows rows against HPCOMPUTER" in text, (
        "the note saying the Windows rows are unmeasured has gone; if the "
        "far side really has been measured, that is a different sentence "
        "with a date and a transcript behind it")


# ---------------------------------------------------------------- re-derive
needs_sftp = pytest.mark.skipif(
    not Path(SFTP_SERVER).exists() or shutil.which("sftp") is None,
    reason="no local OpenSSH sftp-server to measure against")


@needs_sftp
def test_the_write_really_does_truncate_a_file_at_the_target_name(tmp_path):
    """RE-DERIVED, not quoted.  This is the whole reason the lane claims a
    name before it writes: nothing on this link refuses a write.

    Local only: ``-D`` runs the sftp-server as a child over a pipe, so there
    is no socket, no network and nothing of his anywhere near it.
    """
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"V" * 22222)
    src = tmp_path / "src.bin"
    src.write_bytes(b"S" * 1111)
    proc = subprocess.run(
        ["scp", "-D", SFTP_SERVER, str(src), f"localhost:{victim}"],
        capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "" and proc.stderr == "", (
        "the write said something; the lane's argument is that it is SILENT")
    assert victim.stat().st_size == 1111, (
        "the write did not truncate; if that is really true on this box the "
        "whole claim-before-write design needs re-reading")


@needs_sftp
def test_the_flag_that_pins_the_opcode_is_the_one_that_refuses(tmp_path):
    """RE-DERIVED.  Bare ``rename`` replaces, ``rename -l`` refuses -- and
    RENAME_FLAG is what makes the difference, which is why dropping it is
    asserted rather than left to a reader."""
    from jarvis.tools.remote import RENAME_FLAG
    assert RENAME_FLAG == "-l"

    def one(flag: str, tag: str):
        target = tmp_path / f"target-{tag}.bin"
        target.write_bytes(b"T" * 22222)
        temp = tmp_path / f"ours-{tag}.tmp"
        temp.write_bytes(b"O" * 100)
        line = f"rename {flag} {temp} {target}\n".replace("  ", " ")
        proc = subprocess.run(
            ["sftp", "-D", SFTP_SERVER, "-b", "-", "localhost"],
            input=line, capture_output=True, text=True, timeout=60,
            check=False)
        return proc, target, temp

    proc, target, temp = one("", "bare")
    assert proc.returncode == 0
    assert target.stat().st_size == 100 and not temp.exists(), (
        "bare rename did NOT replace; the lane's reason for -l rests on it")

    proc, target, temp = one(RENAME_FLAG, "flagged")
    assert proc.returncode != 0, "rename -l did not refuse a taken name"
    assert "Failure" in proc.stderr
    assert target.stat().st_size == 22222 and temp.stat().st_size == 100, (
        "rename -l moved bytes; both files must be left byte-intact")


@needs_sftp
def test_the_client_has_no_no_clobber_flag():
    """The reason the refusal cannot come from the write.  This is a fact
    about OUR OWN client, which runs on this box, so it is measurable here
    outright -- unlike anything about his server."""
    usage = subprocess.run(["scp"], capture_output=True, text=True,
                           check=False).stderr
    assert "usage: scp" in usage
    flags = re.search(r"usage: scp \[-([A-Za-z0-9]+)\]", usage).group(1)
    for forbidden in ("n", "k"):          # the usual spellings of no-clobber
        assert forbidden not in flags, (
            f"scp now has a -{forbidden} flag; if it is a no-clobber the "
            "claim-before-write design has a cheaper answer and this whole "
            "comment block should be re-read")
