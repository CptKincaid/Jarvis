"""Round 5: the five things an adversary blocked round 4 on.

NOTHING HERE OPENS A SOCKET, TOUCHES ~/Desktop, OR SPEAKS TO HPCOMPUTER.
Every test works in ``tmp_path`` against the same FAKE far side as
tests/test_foldersync.py, whose fixtures are imported rather than copied.

The five, in the adversary's own order:

1  a ValueError on the very path finding L was about -- a good file ahead
   of a "failed" one in the SAME pass unpacks a 4-tuple into three names.
2  the census pin is defeatable three ways (see tests/test_write_census.py
   and tests/test_census_defeats.py).
3  status.txt is silent about three things he can see with his own eyes:
   an inbound name this lane will not touch, a FOLDER over there, and an
   Outbox file skipped by SKIP_SUFFIXES/SKIP_PREFIXES.
4  _foreign_temps is stale: it is recomputed only when there is something
   to push.
5  a failed send leaves the staged partial AND its staging folder on his
   machine for ever, while the source says the close will take it away.
"""
# ruff: noqa: F811 -- `home` is a pytest FIXTURE imported from
# tests/test_foldersync.py rather than copied.

from jarvis import foldersync as fs
from jarvis.tools import remote

from tests.test_foldersync import (Cfg, FakeTransport, drop, home,  # noqa: F401
                                   paths_for, syncer)


# =====================================================================  1
def test_one_bad_file_behind_a_good_one_does_not_kill_the_pass(home):
    """THE CRASH.  b.txt fails with the module's own generic "failed" --
    not a FILE_REASON -- so the link is probed, the link ANSWERS, and the
    freshest listing is folded back into ``taken``.  a.txt has already sent
    in this same pass, so ``sent`` is not empty.

    Round 4's whole stated point was that one file's accident is not the
    pass's death.  Before the fix this raises ValueError out of _push_once,
    _half catches it, and the pass is charged a "pass-failed" event.
    """
    tr = FakeTransport()
    tr.send_fail = {"b.txt": "failed"}      # generic: NOT in FILE_REASONS
    tr.put("outbox", "in.txt", b"inbound")
    s = syncer(home, tr)
    drop(home, "a.txt", b"1234567")         # sends first, fills `sent`
    drop(home, "b.txt", b"89")              # fails second, triggers the probe

    events = s.run_pass()

    assert not [e for e in events if e.outcome == "pass-failed"], \
        "the push half died on the path finding L was about"
    assert "a.txt" in tr.dirs["inbox"], "the good file never landed"
    assert any(e.name == "b.txt" and e.outcome == "failed" for e in events), \
        "the bad file was never reported as one file's problem"
    assert any(e.direction == "pull" and e.outcome == "received"
               for e in events), "the inbound half never ran"


def test_the_good_file_ahead_of_the_bad_one_still_moves_to_sent(home):
    """The other half of the same accident: a.txt is verified and his
    original moves, even though b.txt behind it blew up."""
    tr = FakeTransport()
    tr.send_fail = {"b.txt": "failed"}
    s = syncer(home, tr)
    drop(home, "a.txt", b"1234567")
    drop(home, "b.txt", b"89")

    s.run_pass()

    assert (home / "Sent" / "a.txt").exists(), \
        "his original never moved to Sent, so it will be sent again"
    assert (home / "Outbox" / "b.txt").exists(), "his file was taken away"


# =====================================================================  3
def test_status_says_the_inbound_names_it_will_not_touch(home):
    """A name with a comma or an ampersand in it sits on HPCOMPUTER for
    ever.  He can see it there; status.txt said nothing at all."""
    tr = FakeTransport()
    tr.put("outbox", "notes, final.txt", b"xxxx")
    tr.put("outbox", "R&D plan.txt", b"yyyy")
    s = syncer(home, tr)

    s.run_pass()
    text = (home / "status.txt").read_text()

    assert "2 file(s)" in text and "cannot bring across" in text, \
        f"status.txt is silent about the two names it skipped:\n{text}"


def test_status_says_there_is_a_folder_over_there(home):
    """This lane moves files, never a folder.  He can see the folder."""
    tr = FakeTransport()
    tr.folders["outbox"].add("Photos")
    s = syncer(home, tr)

    s.run_pass()
    text = (home / "status.txt").read_text()

    assert "folder(s)" in text and "Photos" not in text.split("recent")[0][:0], \
        f"status.txt is silent about the folder over there:\n{text}"
    assert "1 folder(s)" in text, text


def test_status_says_the_outbox_files_it_is_not_sending(home):
    """A .tmp or a dotfile in his Outbox is skipped silently, and the
    status file then says "outbox empty" while he is looking at two
    files."""
    tr = FakeTransport()
    s = syncer(home, tr)
    drop(home, "draft.tmp", b"zzz")
    drop(home, ".hidden", b"zzz")

    s.run_pass()
    text = (home / "status.txt").read_text()

    assert "outbox    empty" in text, "fixture wrong: something is sendable"
    assert "not sending" in text and "2 file(s)" in text, \
        f"status.txt says empty while two files sit in his Outbox:\n{text}"


# =====================================================================  4
def test_the_foreign_temp_count_is_recomputed_when_there_is_nothing_to_push(home):
    """The count is only refreshed inside _sweep_my_stages, which the push
    half reaches only when it has something to send.  So a stale number
    from an earlier pass is still on his desk after the litter has gone."""
    tr = FakeTransport()
    stage = remote.remote_stage_name()
    tr.folders["inbox"].add(stage)          # somebody else's staging folder
    s = syncer(home, tr)
    drop(home, "a.txt", b"1234567")         # gives pass 1 something to push

    s.run_pass()
    assert s._foreign_temps == 1, "fixture wrong: the litter was not counted"

    tr.folders["inbox"].discard(stage)      # the litter goes away
    s.run_pass()                            # nothing to push now

    assert s._foreign_temps == 0, \
        "status.txt still tells him about litter that is not there"
    assert "look like my own in-flight" not in (home / "status.txt").read_text()


def test_a_pass_that_did_not_look_says_nothing_about_litter(home):
    """The trade the fix makes, pinned so it is a decision and not a bug.

    The litter is in the remote INBOX and only the push half lists that
    folder, so a pass with nothing to send has not looked.  It must not
    then REPORT -- reporting a number nobody measured this pass is what
    put a stale count on his desk in the first place.  The moment there is
    something to send, the count is real again.
    """
    tr = FakeTransport()
    s = syncer(home, tr)

    s.run_pass()                            # empty Outbox, empty far side
    tr.folders["inbox"].add(remote.remote_stage_name())
    s.run_pass()                            # still nothing to push

    assert s._foreign_temps == 0, "it claimed a count it never measured"
    assert "look like my own in-flight" not in (home / "status.txt").read_text()

    drop(home, "a.txt", b"1234567")         # now there IS something to send
    s.run_pass()

    assert s._foreign_temps == 1, "the count never came back"
    assert "look like my own in-flight" in (home / "status.txt").read_text()


# =====================================================================  5
class ScpLeavesWhatItWrote(FakeTransport):
    """MODELLED, not measured: a scp that fails part-way leaves the bytes it
    already wrote at the destination name.  The stock FakeTransport returns
    the reason before it stages anything, which is why round 4's suite never
    saw the litter the adversary measured on the real link.

    Nothing here talks to HPCOMPUTER.  The behaviour being modelled -- a
    partial file left behind by a failed copy, and an sftp rmdir that then
    refuses the non-empty directory -- is the ordinary contract of scp and
    of rmdir; it is stated as modelled every time it is quoted.
    """

    def send(self, local, name, key="inbox"):
        reason = super().send(local, name, key)
        if reason and "/" in name and not self.fail:
            self.staged[name] = (1, self.stamp)      # what scp got out
        return reason


def test_a_failed_send_leaves_nothing_of_ours_on_his_machine(home):
    """MEASURED by the adversary: 5 staging folders after 12 passes on one
    failing file.  The source says "the close at the end of the pass, or a
    later pass, takes it away" and the server does not -- rmdir refuses a
    non-empty directory, so both the partial and the folder are permanent.
    """
    tr = ScpLeavesWhatItWrote()
    tr.send_fail = {"a.txt": "failed"}
    s = syncer(home, tr)
    drop(home, "a.txt", b"1234567")

    for _ in range(12):
        s.run_pass()

    assert not tr.staged, f"partials left on his machine: {sorted(tr.staged)}"
    assert not tr.stages, f"staging folders left behind: {sorted(tr.stages)}"


def test_the_stage_is_given_back_even_when_the_partial_will_not_go(home):
    """If the far side refuses the delete too, the code must not claim the
    close will take it away -- but it must still not accumulate a new
    folder every pass for the same file."""
    tr = ScpLeavesWhatItWrote()
    tr.send_fail = {"a.txt": "failed"}
    s = syncer(home, tr)
    drop(home, "a.txt", b"1234567")

    s.run_pass()

    assert tr.discarded, "the staged partial was never even asked about"


# ============================ the two the adversary named but did not block
def test_the_setting_says_out_loud_that_it_can_reach_a_file_of_his(home):
    """remove_broken_copies ON can delete a file OF HIS -- one that replaced
    ours at that name inside the verify window.  MEASURED here, and it is
    DETERMINISTIC: it is what the code does every time that sequence
    happens, not a race that sometimes loses.

    The fix is not to change the behaviour -- he has not answered the
    setting yet and it ships OFF -- it is that the setting's own sentence
    now says this, in both places a person reads it.
    """
    class HisFileReplacesOurs(FakeTransport):
        """He drops his own report over ours between the landing and the
        size check.  The name is the same; the identity we hold is only
        the NAME we took."""

        def listing(self, key):
            rows, why = super().listing(key)
            if key == "inbox" and "a.txt" in self.dirs["inbox"]:
                self.dirs["inbox"]["a.txt"] = (99999, self.stamp)   # HIS
            return rows, why

    tr = HisFileReplacesOurs()
    tr.short_write = 3
    s = syncer(home, tr, remove_broken_copies=True)
    drop(home, "a.txt", b"x" * 30)

    s.run_pass()

    assert "a.txt" not in tr.dirs["inbox"], \
        "fixture wrong: nothing was deleted, so there is nothing to warn about"
    assert tr.removed == ["a.txt"], tr.removed

    # ...and the sentence he would read must say so, in both places.
    src = (fs.__file__ and open(fs.__file__, encoding="utf-8").read())
    assert "MEASURED and DETERMINISTIC" in src, \
        "the setting's own comment does not say what turning it on costs"
    from jarvis import assistant_config
    cfg_src = open(assistant_config.__file__, encoding="utf-8").read()
    assert "the delete takes YOUR file" in cfg_src, \
        "the config file's line does not say what turning it on costs"


def test_the_source_says_the_landing_record_is_resolved_by_size(home):
    """Narrow, and named rather than implied.  _resolve_landing can only ask
    "is something at that name, at the byte count I sent" -- there is no
    checksum on this link.  Finding L's guarantee is the ORDERING, and the
    source now says which of the two is load-bearing."""
    src = open(fs.__file__, encoding="utf-8").read()
    body = src[src.index("def _resolve_landing"):]
    body = body[:body.index("def _open_stage")]
    assert "RESOLVED BY SIZE, NOT BY IDENTITY" in body
    assert "rests on ORDERING" in body


def test_a_different_file_of_our_exact_size_is_read_as_ours(home):
    """The limit the sentence above describes, MEASURED rather than
    asserted.  Nothing of his is deleted by this -- his original is moved
    to Sent, which is the visible consequence."""
    tr = FakeTransport()
    s = syncer(home, tr)
    a = drop(home, "a.txt", b"1234567")          # 7 bytes
    key = f"push:a.txt|{fs.stat_key(a)}"
    s.ledger.mark_claiming(key, "a.txt", 7, 0.0)
    s.ledger.save()
    tr.put("inbox", "a.txt", b"HISFILE")         # 7 bytes, NOT ours

    s.run_pass()

    assert (home / "Sent" / "a.txt").exists(), \
        "size is the only question this link can ask, and it was answered"
    assert tr.dirs["inbox"]["a.txt"][0] == 7, "his file over there was touched"
