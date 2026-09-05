"""Desktop folder sync -- jarvis/foldersync.py.

NOTHING HERE OPENS A SOCKET OR TOUCHES ~/Desktop.  Every test works in
``tmp_path`` with a FAKE transport; the real one
(:class:`foldersync.SshTransport`) is exercised only for the argv and the
listing PARSE, both of which are decisions made before a socket opens.

The two measured constraints these were written against (2026-09-05):

* the ``jarvis`` account CANNOT create in the Windows Outbox
  (``dest open ".../Outbox/.writetest": Permission denied``), so the puller
  can never clear what it has taken and needs a LOCAL ledger instead;
* ``remote.push`` refuses a source outside ``remote.local_roots``, and
  ``~/Desktop/Jarvis/Outbox`` is inside the shipped ``~/Desktop`` -- asserted
  here rather than assumed, with the opposite case asserted too.

The sftp listing text parsed below is verbatim from the format measured on
this box in ``jarvis/tools/remote.py`` (``sftp -D`` to the local
sftp-server; no socket).
"""
import dataclasses
import errno
import json
import os
import threading
import time
from pathlib import Path

import pytest

from jarvis import foldersync as fs
from jarvis.assistant_config import DEFAULTS
from jarvis.tools import filepick, remote


# ------------------------------------------------------------------ fakes
class Cfg:
    """The AssistantConfig surface read_config uses: dotted get(), seeded
    from the SHIPPED defaults so a passing test is about the config he
    actually gets."""

    def __init__(self, **over):
        self.data = {}
        for section in ("remote", "foldersync"):
            for key, val in DEFAULTS[section].items():
                self.data[f"{section}.{key}"] = val
        self.data.update(over)

    def get(self, key, default=None):
        return self.data.get(key, default)


class FakeTransport:
    """A Windows far side in a dict.  ``inbox`` is writable, ``outbox`` is
    NOT -- the measured permission on the real box -- and this fake refuses
    a write there so a puller that ever tries to tidy up fails a test.

    The two write primitives behave the way they were MEASURED to behave
    against HPCOMPUTER on 2026-09-05 (OpenSSH_for_Windows_9.5, SFTP 3):

    * :meth:`send` is scp, and scp TRUNCATES.  It writes at the name it is
      given whatever is there -- 10 of 10 rounds destroyed a 22222-byte
      file, exit 0, no message.  There is no no-clobber flag on it.
    * :meth:`claim` is ``rename -l``, and it REFUSES.  10 of 10 rounds
      refused an existing name with both files byte-intact; two concurrent
      sessions renaming onto the same name, 12 rounds, gave exactly one
      winner every time.  That is the only operation on this link that can
      say no.
    """

    def __init__(self, inbox=None, outbox=None):
        self.dirs = {"inbox": dict(inbox or {}), "outbox": dict(outbox or {})}
        self.folders = {"inbox": set(), "outbox": set()}   # FOLDERS over there
        self.fail = ""            # a reason to return from every call
        self.listing_fail = {}    # {key: reason} -- ONE folder, not the link
        self.send_fail = {}       # {local name: reason} -- the FILE, not the link
        self.fetch_fail = {}      # {name: reason} -- likewise, inbound
        self.claim_fail = {}      # {final: reason} -- the rename itself
        self.calls = []
        self.claims = []          # every (temp, final) rename attempted
        self.discarded = []       # every temp of OURS taken away again
        self.short_write = 0      # land this many bytes instead of the file's
        self.stamp = "Sep  5 13:45"

    # -- helpers a test uses to describe the far side ------------------
    def put(self, key, name, data=b"x", stamp=None):
        self.dirs[key][name] = (len(data), stamp or self.stamp)

    def listing(self, key):
        self.calls.append(("listing", key))
        if self.fail:
            return [], self.fail
        if self.listing_fail.get(key):
            return [], self.listing_fail[key]
        rows = [fs.Entry(n, s, t) for n, (s, t) in sorted(self.dirs[key].items())]
        rows += [fs.Entry(n, 4096, self.stamp, True)
                 for n in sorted(self.folders[key])]
        return rows, ""

    def send(self, local, name, key="inbox"):
        self.calls.append(("send", name))
        if self.fail:
            return self.fail
        if key != "inbox":
            raise AssertionError("nothing may write outside the remote inbox")
        if local.name in self.send_fail or name in self.send_fail:
            return self.send_fail.get(local.name) or self.send_fail[name]
        if name in self.folders[key]:
            return "failed"       # scp onto a folder: what the regexes miss
        size = self.short_write or local.stat().st_size
        self.dirs[key][name] = (size, self.stamp)   # scp TRUNCATES
        return ""

    def claim(self, temp, final, key="inbox"):
        """``rename -l``: it takes the name or it says it could not."""
        self.calls.append(("claim", final))
        self.claims.append((temp, final))
        if self.fail:
            return self.fail
        if final in self.claim_fail:
            return self.claim_fail[final]
        if temp not in self.dirs[key]:
            return "taken"        # generic "Failure" is all the wire says
        if final in self.dirs[key] or final in self.folders[key]:
            return "taken"        # measured: refused, both files intact
        self.dirs[key][final] = self.dirs[key].pop(temp)
        return ""

    def discard(self, temp, key="inbox"):
        """Only ever a temp of OURS -- anything else is a test failure."""
        if not temp.startswith(remote.REMOTE_TEMP_PREFIX):
            raise AssertionError(f"tried to delete {temp!r} on HPCOMPUTER")
        self.calls.append(("discard", temp))
        self.discarded.append(temp)
        if self.fail:
            return self.fail
        self.dirs[key].pop(temp, None)
        return ""

    def fetch(self, key, name, dest):
        self.calls.append(("fetch", name))
        if self.fail:
            return self.fail
        if name in self.fetch_fail:
            return self.fetch_fail[name]
        size, _stamp = self.dirs[key][name]
        dest.write_bytes(b"y" * (self.short_write or size))
        return ""

    def delete(self, key, name):        # never called: asserted below
        raise AssertionError("foldersync must never delete on HPCOMPUTER")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway ~/Desktop/Jarvis with the three folders, plus a state dir."""
    desk = tmp_path / "Desktop" / "Jarvis"
    for sub in ("Inbox", "Outbox"):
        (desk / sub).mkdir(parents=True)
    (tmp_path / "state").mkdir()
    monkeypatch.setattr(fs.PATHS, "STATE_DIR", tmp_path / "state", raising=False)
    return desk


def paths_for(home):
    return fs.Paths(outbox=home / "Outbox", inbox=home / "Inbox",
                    sent=home / "Sent", status=home / "status.txt")


def syncer(home, transport=None, **over):
    conf = remote.read_config(Cfg(**{
        "remote.enabled": True, "remote.host": "192.168.50.114",
        "remote.user": "jarvis", "remote.key_path": "",
        "remote.local_roots": (str(home.parent),)}))
    sconf = fs.SyncConfig(**{**dict(enabled=True, paths=paths_for(home)), **over})
    return fs.Syncer(conf, sconf, transport or FakeTransport(),
                     ledger=fs.Ledger(home.parent.parent / "state" / "l.json"))


def drop(home, name, data=b"hello", age_s=60.0):
    """A file sitting quietly in the Outbox, old enough to be eligible."""
    p = home / "Outbox" / name
    p.write_bytes(data)
    old = time.time() - age_s
    os.utime(p, (old, old))
    return p


# ------------------------------------------------- names Windows refuses
@pytest.mark.parametrize("name,problem", [
    ("report.pdf", ""),
    ("Quarterly Report (final) v2.xlsx", ""),
    ("notes_2026-09-05.txt", ""),
    ("12:30 meeting.txt", "name-charset"),      # a colon
    ("what*now.txt", "name-charset"),
    ("pipe|name.txt", "name-charset"),
    ("q?.txt", "name-charset"),
    ('quote".txt', "name-charset"),
    ("plan\\b.txt", "name-charset"),
    ("résumé.pdf", "name-charset"),             # unicode
    ("party 🎉.txt", "name-charset"),            # emoji
    ("trailing.", "name-trailing"),
    ("trailing ", "name-trailing"),
    ("CON", "name-reserved"),
    ("con.txt", "name-reserved"),
    ("PRN.log", "name-reserved"),
    ("aux", "name-reserved"),
    ("NUL.dat", "name-reserved"),
    ("COM1.txt", "name-reserved"),
    ("lpt9", "name-reserved"),
    ("COM10.txt", ""),                          # not reserved: only 1-9
    ("CONTRACT.pdf", ""),                       # a prefix is not a device
])
def test_windows_name_problem(name, problem):
    assert fs.windows_name_problem(name) == problem


def test_a_path_too_long_for_windows_is_refused_by_length_not_by_name():
    folder = "/C:/Users/h2pey/Desktop/Jarvis/Inbox"
    ok = "a" * 100 + ".txt"
    assert fs.windows_name_problem(ok) == ""
    assert fs.path_problem(folder, ok) == ""
    # 121 characters is the most SAFE_REMOTE_NAME_RX allows, so the only way
    # to exceed Windows' 260 is a long FOLDER -- which is config, not speech.
    deep = "/C:/Users/h2pey/" + "/".join(["a-folder-with-a-long-name"] * 9)
    assert fs.path_problem(deep, ok) == "name-too-long"


def test_the_charset_rule_is_the_transport_s_own_rule():
    """foldersync must not be more permissive than the thing that does the
    copying, or a name it accepts is refused later with a worse message."""
    for name in ("12:30.txt", "résumé.pdf", "-rf.txt", ".hidden"):
        assert not remote.SAFE_REMOTE_NAME_RX.match(name)
        assert fs.windows_name_problem(name) == "name-charset"


# ------------------------------------------------------ name collisions
def test_dedupe_never_overwrites_and_keeps_the_extension():
    taken = {"report.pdf"}
    assert fs.dedupe_name("report.pdf", taken) == "report (2).pdf"
    taken.add("report (2).pdf")
    assert fs.dedupe_name("report.pdf", taken) == "report (3).pdf"
    assert fs.dedupe_name("fresh.pdf", taken) == "fresh.pdf"
    assert fs.dedupe_name("noext", {"noext"}) == "noext (2)"


def test_dedupe_gives_up_rather_than_looping():
    taken = {"r.txt"} | {f"r ({i}).txt" for i in range(2, fs.MAX_COPIES + 2)}
    assert fs.dedupe_name("r.txt", taken) == ""


# ---------------------------------------------------------- quiescence
def test_a_growing_file_is_never_quiescent_and_a_still_one_is(tmp_path):
    """MEASURED against a real slow writer on a real clock: the numbers this
    prints are the evidence the rule holds, not an argument that it should."""
    target = tmp_path / "big.bin"
    target.write_bytes(b"")
    stop = threading.Event()

    def slow_writer():
        with open(target, "ab", buffering=0) as fh:
            while not stop.is_set():
                fh.write(b"0" * 65536)
                time.sleep(0.05)

    t = threading.Thread(target=slow_writer, daemon=True)
    t.start()
    try:
        time.sleep(0.2)
        # While it grows: not once, across several independent judgements.
        for _ in range(3):
            assert fs.is_quiescent(target, samples=3, interval_s=0.15,
                                   min_quiet_s=0.4) is False
    finally:
        stop.set()
        t.join(timeout=5)
    grown = target.stat().st_size
    assert grown > 0
    # It must become sendable, and only AFTER the quiet window -- both
    # halves matter, so both are measured rather than assumed.
    assert fs.is_quiescent(target, samples=3, interval_s=0.15,
                           min_quiet_s=0.4) is False       # too soon
    time.sleep(0.5)
    assert fs.is_quiescent(target, samples=3, interval_s=0.15,
                           min_quiet_s=0.4) is True
    assert target.stat().st_size == grown       # nothing here wrote to it


def test_a_file_written_this_instant_is_held_back_by_the_age_gate(tmp_path):
    p = tmp_path / "just-made.txt"
    p.write_bytes(b"x")
    # Stable across every sample, but its mtime is NOW: a drag that has
    # written its first block and stalled looks exactly like this.
    assert fs.is_quiescent(p, samples=2, interval_s=0.01,
                           min_quiet_s=30.0) is False


def test_quiescence_costs_nothing_when_the_file_vanishes_mid_sample(tmp_path):
    p = tmp_path / "gone.txt"
    p.write_bytes(b"x")
    p.unlink()
    assert fs.is_quiescent(p, samples=2, interval_s=0.01,
                           min_quiet_s=0.0) is False


# ------------------------------------------------------------ the ledger
def test_the_ledger_never_fetches_the_same_file_twice(home):
    t = FakeTransport(outbox={"notes.txt": (5, "Sep  5 13:45")})
    s = syncer(home, t)
    assert [e.outcome for e in s.pull_once()] == ["received"]
    assert (home / "Inbox" / "notes.txt").read_bytes() == b"yyyyy"
    assert s.pull_once() == []                      # second pass: nothing
    assert [c for c in t.calls if c[0] == "fetch"] == [("fetch", "notes.txt")]


def test_the_same_name_with_different_contents_comes_across_again(home):
    t = FakeTransport(outbox={"notes.txt": (5, "Sep  5 13:45")})
    s = syncer(home, t)
    s.pull_once()
    # He rewrote it: same name, a new size and a new stamp.
    t.dirs["outbox"]["notes.txt"] = (9, "Sep  5 14:02")
    events = s.pull_once()
    assert [e.outcome for e in events] == ["received"]
    assert events[0].detail == "notes (2).txt"      # his first copy untouched
    assert (home / "Inbox" / "notes.txt").read_bytes() == b"yyyyy"
    assert (home / "Inbox" / "notes (2).txt").stat().st_size == 9


def test_a_byte_identical_file_put_back_does_not_loop(home):
    t = FakeTransport(outbox={"notes.txt": (5, "Sep  5 13:45")})
    s = syncer(home, t)
    s.pull_once()
    for _ in range(5):                              # he puts it back, unchanged
        t.dirs["outbox"]["notes.txt"] = (5, "Sep  5 13:45")
        assert s.pull_once() == []
    assert len([c for c in t.calls if c[0] == "fetch"]) == 1


def test_the_ledger_survives_a_restart(home):
    t = FakeTransport(outbox={"notes.txt": (5, "Sep  5 13:45")})
    s = syncer(home, t)
    s.pull_once()
    s2 = syncer(home, t)                            # a fresh process
    assert s2.pull_once() == []


def test_the_ledger_cannot_grow_without_bound(home):
    led = fs.Ledger(home.parent.parent / "state" / "l.json", cap=10)
    for i in range(50):
        led.mark_pulled(f"f{i}.txt|1|Sep  5 13:45", landed=f"f{i}.txt", now=i)
    led.save()
    assert len(led.pulled) <= 10
    # The newest survive: an entry still sitting in the remote Outbox is the
    # one that must not be forgotten, and it is the one most recently seen.
    assert "f49.txt|1|Sep  5 13:45" in led.pulled


def test_sweeping_keeps_what_is_still_on_the_far_side(home):
    led = fs.Ledger(home.parent.parent / "state" / "l.json", cap=3)
    for i in range(3):
        led.mark_pulled(f"f{i}|1|s", landed=f"f{i}", now=i)
    led.sweep({"f0|1|s"}, now=1000.0)               # only f0 is still there
    led.mark_pulled("new|1|s", landed="new", now=1001.0)
    assert "f0|1|s" in led.pulled                   # refreshed by the sweep


# --------------------------------------------------------------- pushing
def test_a_pushed_file_is_verified_then_moved_to_sent(home):
    t = FakeTransport()
    p = drop(home, "quarterly.pdf", b"a" * 4096)
    s = syncer(home, t)
    events = s.push_once()
    assert [e.outcome for e in events] == ["sent"]
    assert not p.exists()                            # the Outbox empties
    assert (home / "Sent" / "quarterly.pdf").read_bytes() == b"a" * 4096
    assert t.dirs["inbox"]["quarterly.pdf"][0] == 4096
    # The listing that VERIFIES came after the send, not instead of it --
    # and the bytes went at a temp name of ours, which a CLAIM then renamed
    # onto the name he will see.  Nothing writes at his name.
    assert [c[0] for c in t.calls] == ["listing", "send", "claim", "listing"]


def test_a_short_write_on_the_far_side_leaves_his_file_where_it_is(home):
    t = FakeTransport()
    t.short_write = 10                               # 10 of 4096 bytes land
    p = drop(home, "quarterly.pdf", b"a" * 4096)
    s = syncer(home, t)
    events = s.push_once()
    assert [e.outcome for e in events] == ["verify-failed"]
    assert p.exists()                                # NOT moved
    assert not (home / "Sent").exists()
    note = home / "Outbox" / ("quarterly.pdf" + fs.NOTE_SUFFIX)
    assert "4096" in note.read_text() and "10" in note.read_text()


def test_a_failing_push_stops_after_a_bounded_number_of_tries(home):
    t = FakeTransport()
    t.short_write = 10
    drop(home, "q.pdf", b"a" * 4096)
    s = syncer(home, t)
    for _ in range(fs.MAX_ATTEMPTS + 3):
        s.push_once()
    sends = [c for c in t.calls if c[0] == "send"]
    assert len(sends) == fs.MAX_ATTEMPTS
    assert (home / "Outbox" / "q.pdf").exists()


def test_a_name_already_on_the_far_side_lands_beside_it_never_over_it(home):
    t = FakeTransport(inbox={"report.pdf": (11, "Sep  4 09:00")})
    drop(home, "report.pdf", b"new content")
    s = syncer(home, t)
    events = s.push_once()
    assert [e.outcome for e in events] == ["sent"]
    assert events[0].detail == "report (2).pdf"
    assert t.dirs["inbox"]["report.pdf"] == (11, "Sep  4 09:00")   # untouched
    assert "report (2).pdf" in t.dirs["inbox"]


def test_a_file_over_the_cap_is_refused_visibly_and_left_alone(home):
    t = FakeTransport()
    p = drop(home, "movie.mkv", b"a" * 4096)
    s = syncer(home, t, max_mb=0.001)                # 1 KB cap
    events = s.push_once()
    assert [e.outcome for e in events] == ["too-big"]
    assert p.exists()
    note = (home / "Outbox" / ("movie.mkv" + fs.NOTE_SUFFIX)).read_text()
    assert "too big" in note.lower()
    assert not [c for c in t.calls if c[0] == "send"]


def test_a_name_windows_refuses_never_reaches_the_transport(home):
    t = FakeTransport()
    for name in ("CON.txt", "trailing.", "12:30 notes.txt"):
        drop(home, name, b"x")
    s = syncer(home, t)
    events = s.push_once()
    assert sorted(e.outcome for e in events) == \
        ["name-charset", "name-reserved", "name-trailing"]
    assert not [c for c in t.calls if c[0] == "send"]
    for name in ("CON.txt", "trailing.", "12:30 notes.txt"):
        assert (home / "Outbox" / name).exists()
        assert (home / "Outbox" / (name + fs.NOTE_SUFFIX)).exists()


def test_a_folder_dropped_in_the_outbox_is_refused_not_recursed(home):
    t = FakeTransport()
    (home / "Outbox" / "a whole folder").mkdir()
    (home / "Outbox" / "a whole folder" / "inner.txt").write_bytes(b"x")
    s = syncer(home, t)
    events = s.push_once()
    assert [e.outcome for e in events] == ["not-a-file"]
    assert not [c for c in t.calls if c[0] == "send"]


def test_our_own_notes_and_part_files_are_never_pushed(home):
    t = FakeTransport()
    (home / "Outbox" / ("x.txt" + fs.NOTE_SUFFIX)).write_bytes(b"why")
    (home / "Outbox" / (fs.PART_PREFIX + "y.txt")).write_bytes(b"half")
    (home / "Outbox" / ".hidden").write_bytes(b"h")
    (home / "Outbox" / "download.crdownload").write_bytes(b"h")
    for p in (home / "Outbox").iterdir():
        old = time.time() - 600
        os.utime(p, (old, old))
    s = syncer(home, t)
    assert s.push_once() == []
    assert not [c for c in t.calls if c[0] == "send"]


def test_a_file_still_being_written_is_left_for_the_next_pass(home):
    t = FakeTransport()
    p = home / "Outbox" / "growing.bin"
    p.write_bytes(b"a" * 100)                        # mtime is NOW
    s = syncer(home, t, min_quiet_s=3600.0)
    assert s.push_once() == []
    assert p.exists()
    assert not [c for c in t.calls if c[0] == "send"]


# --------------------------------------------------------- the local roots
def test_the_outbox_must_be_inside_remote_local_roots(home):
    """CHECKED, not assumed -- and a failure is reported, never fixed by
    widening the roots."""
    ok = remote.read_config(Cfg(**{"remote.local_roots": (str(home.parent),)}))
    assert fs.preflight(ok, paths_for(home)) == []
    narrow = remote.read_config(Cfg(**{"remote.local_roots": ("/nonexistent",)}))
    problems = fs.preflight(narrow, paths_for(home))
    assert any("outside" in p for p in problems)
    assert narrow.local_roots == ("/nonexistent",)   # NOT widened


def test_the_shipped_desktop_root_really_does_contain_the_outbox(home):
    """The one thing that would make the whole feature refuse every file."""
    roots = filepick.expand_roots((str(home.parent),))
    f = drop(home, "x.txt", b"x")
    assert filepick.check_file(f, roots, 100) == ""


# ---------------------------------------------------------- no recursion
def test_the_two_folders_may_not_be_the_same_folder(home):
    same = fs.Paths(outbox=home / "Inbox", inbox=home / "Inbox",
                    sent=home / "Sent", status=home / "status.txt")
    conf = remote.read_config(Cfg(**{"remote.local_roots": (str(home.parent),)}))
    assert any("same folder" in p for p in fs.preflight(conf, same))


def test_nothing_the_puller_writes_is_ever_seen_by_the_pusher(home):
    t = FakeTransport(outbox={"from-windows.txt": (5, "Sep  5 13:45")})
    s = syncer(home, t)
    s.pull_once()
    landed = home / "Inbox" / "from-windows.txt"
    assert landed.exists()
    old = time.time() - 600
    os.utime(landed, (old, old))
    assert s.push_once() == []                        # different folder
    assert not [c for c in t.calls if c[0] == "send"]


def test_sent_may_not_sit_inside_the_outbox(home):
    bad = fs.Paths(outbox=home / "Outbox", inbox=home / "Inbox",
                   sent=home / "Outbox" / "Sent", status=home / "status.txt")
    conf = remote.read_config(Cfg(**{"remote.local_roots": (str(home.parent),)}))
    assert any("inside the outbox" in p for p in fs.preflight(conf, bad))


# ----------------------------------------------------------- pulling detail
def test_a_pulled_file_is_never_visible_half_written(home):
    t = FakeTransport(outbox={"big.bin": (100, "Sep  5 13:45")})
    seen = []
    real_fetch = t.fetch

    def watching_fetch(key, name, dest):
        r = real_fetch(key, name, dest)
        seen.append(sorted(p.name for p in (home / "Inbox").iterdir()))
        return r

    t.fetch = watching_fetch
    s = syncer(home, t)
    s.pull_once()
    assert seen == [[fs.PART_PREFIX + "big.bin"]]     # hidden while in flight
    assert sorted(p.name for p in (home / "Inbox").iterdir()) == ["big.bin"]


def test_a_short_fetch_leaves_nothing_behind_and_is_retried(home):
    t = FakeTransport(outbox={"big.bin": (100, "Sep  5 13:45")})
    t.short_write = 7
    s = syncer(home, t)
    events = s.pull_once()
    assert [e.outcome for e in events] == ["verify-failed"]
    assert list((home / "Inbox").iterdir()) == []
    for _ in range(fs.MAX_ATTEMPTS + 3):
        s.pull_once()
    assert len([c for c in t.calls if c[0] == "fetch"]) == fs.MAX_ATTEMPTS


def test_a_remote_name_this_module_will_not_touch_is_skipped_not_escaped(home):
    t = FakeTransport(outbox={"ok.txt": (2, "Sep  5 13:45")})
    t.dirs["outbox"]["a;b`c.txt"] = (2, "Sep  5 13:45")
    s = syncer(home, t)
    events = s.pull_once()
    assert [e.name for e in events] == ["ok.txt"]
    assert sorted(p.name for p in (home / "Inbox").iterdir()) == ["ok.txt"]


def test_the_puller_never_writes_to_the_far_side(home):
    """The measured permission: the jarvis account cannot create in the
    Windows Outbox.  A puller that tries to tidy up is the bug."""
    t = FakeTransport(outbox={"notes.txt": (5, "Sep  5 13:45")})
    s = syncer(home, t)
    s.pull_once()
    assert not [c for c in t.calls if c[0] in ("send", "delete")]


# --------------------------------------------------------- link being down
def test_a_dead_link_loses_nothing_and_backs_off(home):
    t = FakeTransport()
    t.fail = "unreachable"
    p = drop(home, "waiting.pdf", b"a" * 100)
    s = syncer(home, t)
    first = s.interval_s
    outcomes = []
    for _ in range(6):
        outcomes += [e.outcome for e in s.run_pass()]
    assert p.exists()                                  # nothing lost
    assert set(outcomes) == {"link-down"}
    assert s.interval_s > first
    assert s.interval_s <= s.conf.max_backoff_s


def test_the_backoff_resets_the_moment_the_box_answers(home):
    t = FakeTransport()
    t.fail = "unreachable"
    s = syncer(home, t)
    for _ in range(4):
        s.run_pass()
    assert s.interval_s > s.conf.remote_interval_s
    t.fail = ""
    s.run_pass()
    assert s.interval_s == s.conf.remote_interval_s


def test_a_dead_link_logs_once_not_once_a_second(home, caplog):
    t = FakeTransport()
    t.fail = "unreachable"
    drop(home, "waiting.pdf", b"a")
    s = syncer(home, t)
    with caplog.at_level("WARNING", logger="jarvis.foldersync"):
        for _ in range(20):
            s.run_pass()
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_what_he_sees_while_the_link_is_down(home):
    t = FakeTransport()
    t.fail = "unreachable"
    drop(home, "waiting.pdf", b"a" * 100)
    s = syncer(home, t)
    s.run_pass()
    text = (home / "status.txt").read_text()
    assert "HPCOMPUTER" in text
    assert "waiting.pdf" in text
    assert "not answering" in text.lower() or "couldn't reach" in text.lower()


def test_the_status_file_is_rewritten_only_when_it_changes(home):
    t = FakeTransport()
    s = syncer(home, t)
    s.run_pass()
    first = (home / "status.txt").stat().st_mtime_ns
    s.status_clock = lambda: "the same clock reading"
    s.write_status()
    s.write_status()
    assert (home / "status.txt").stat().st_mtime_ns >= first


# ------------------------------------------------------------- the record
def test_every_outcome_is_recorded_by_name_and_size_and_never_by_bytes(home):
    t = FakeTransport(outbox={"in.txt": (3, "Sep  5 13:45")})
    drop(home, "out.txt", b"secret bytes here")
    s = syncer(home, t)
    s.run_pass()
    rows = [json.loads(x) for x in s.history_path.read_text().splitlines()]
    assert {r["name"] for r in rows} == {"out.txt", "in.txt"}
    assert {r["outcome"] for r in rows} == {"sent", "received"}
    assert all(r["size"] > 0 for r in rows)
    blob = s.history_path.read_text()
    assert "secret bytes here" not in blob
    assert all("content" not in r and "data" not in r for r in rows)


def test_the_record_cannot_grow_without_bound(home):
    s = syncer(home, FakeTransport())
    for i in range(fs.HISTORY_LINES + 50):
        s.record(fs.Event(time.time(), "push", f"f{i}.txt", 1, "sent", ""))
    assert len(s.history_path.read_text().splitlines()) <= fs.HISTORY_LINES


# -------------------------------------------------- the real transport
def test_the_sftp_listing_parse_is_the_measured_format():
    """Verbatim from jarvis/tools/remote.py, measured on this box with
    `sftp -q -b - -D /usr/lib/openssh/sftp-server` (no socket)."""
    out = ('sftp> ls -ln "Desktop/Jarvis/Outbox"\n'
           "-rw-rw-r--    ? hunterp  hunterp   1 Sep  3 12:21 "
           "jarvis-outbox/a.txt\n"
           "drwxrwxr-x    ? hunterp  hunterp 4096 Sep  3 12:21 "
           "jarvis-outbox/sub dir\n"
           "-rw-rw-r--    ? hunterp  hunterp 20481 Sep  3  2025 "
           "jarvis-outbox/old report.pdf\n")
    rows = fs.parse_sftp_entries(out)
    assert [(r.name, r.size, r.stamp, r.is_dir) for r in rows] == [
        ("a.txt", 1, "Sep 3 12:21", False),
        # KEPT, not dropped.  A folder over there is a name that is taken,
        # and dedupe_name has to see it -- see the poison case below.
        ("sub dir", 4096, "Sep 3 12:21", True),
        ("old report.pdf", 20481, "Sep 3 2025", False),
    ]


def test_the_transfer_budget_grows_with_the_file_and_is_still_capped():
    conf = remote.read_config(Cfg())
    assert fs.transfer_budget(conf, 1024) == conf.transfer_timeout_s
    assert fs.transfer_budget(conf, 500 * 1024 * 1024) > conf.transfer_timeout_s
    assert fs.transfer_budget(conf, 10**12) <= remote.MAX_TRANSFER_S


def test_the_real_transport_never_builds_a_path_outside_the_configured_inbox():
    conf = remote.read_config(Cfg(**{
        "remote.inbox": "/C:/Users/h2pey/Desktop/Jarvis/Inbox"}))
    tr = fs.SshTransport(conf)
    assert tr.target("report.pdf") == \
        "/C:/Users/h2pey/Desktop/Jarvis/Inbox/report.pdf"
    for bad in ("../escape.txt", "/etc/passwd", "a/b.txt"):
        assert tr.target(bad) == ""


def test_the_real_transport_is_not_reached_by_an_unconfigured_lane(monkeypatch):
    """Same rule as the voice lane: no socket opens before missing_reason."""
    def explode(*a, **k):
        raise AssertionError("a socket must not open here")

    monkeypatch.setattr(remote, "run_sftp", explode)
    monkeypatch.setattr(remote, "run_copy", explode)
    tr = fs.SshTransport(remote.read_config(Cfg(**{"remote.enabled": False})))
    assert tr.listing("outbox") == ([], "disabled")
    assert tr.send(Path("/nonexistent"), "x.txt") == "disabled"


# ------------------------------------------------------------- one at a time
def test_only_one_sync_runs_at_a_time(home, tmp_path):
    lock = tmp_path / "state" / "foldersync.lock"
    with fs.single_instance(lock) as got:
        assert got
        with fs.single_instance(lock) as second:
            assert second is False


# ------------------------------------------------------------------ config
def test_the_shipped_config_points_at_the_folders_he_asked_for():
    d = DEFAULTS["foldersync"]
    assert d["enabled"] is False          # he turns it on, like every lane
    assert d["outbox"] == "~/Desktop/Jarvis/Outbox"
    assert d["inbox"] == "~/Desktop/Jarvis/Inbox"
    assert d["sent"] == "~/Desktop/Jarvis/Sent"
    conf = fs.read_config(Cfg())
    assert conf.paths.outbox == Path.home() / "Desktop/Jarvis/Outbox"
    assert conf.paths.sent == Path.home() / "Desktop/Jarvis/Sent"


def test_the_cap_falls_back_to_the_remote_lane_s_own(home):
    rconf = remote.read_config(Cfg(**{"remote.max_mb": 250}))
    sconf = fs.read_config(Cfg(**{"foldersync.max_mb": 0}))
    assert fs.effective_max_mb(rconf, sconf) == 250
    sconf2 = fs.read_config(Cfg(**{"foldersync.max_mb": 900}))
    assert fs.effective_max_mb(rconf, sconf2) == 900


# ------------------------------------------------- the measurement itself
@pytest.mark.skipif(not os.environ.get("JARVIS_SLOW_MEASURE"),
                    reason="set JARVIS_SLOW_MEASURE=1: ~30 s of real clock")
def test_measure_the_shipped_quiescence_rule_against_a_slow_write(tmp_path,
                                                                  capsys):
    """The SHIPPED numbers (3 samples, 1 s apart, 4 s quiet) against a file
    written the way a drag writes one.  The suite's fast version proves the
    mechanism; this one proves the constants, and prints what it measured so
    the number in the report has a source.
    """
    d = DEFAULTS["foldersync"]
    kw = dict(samples=d["stable_samples"], interval_s=d["stable_interval_s"],
              min_quiet_s=d["min_quiet_s"])
    target = tmp_path / "drag.bin"
    target.write_bytes(b"")
    stop = threading.Event()
    written = [0]

    def slow_writer():                      # ~4 MB/s, flushing every 250 ms
        with open(target, "ab", buffering=0) as fh:
            while not stop.is_set():
                fh.write(b"0" * (1024 * 1024))
                written[0] += 1024 * 1024
                time.sleep(0.25)

    t = threading.Thread(target=slow_writer, daemon=True)
    t.start()
    verdicts = []
    t0 = time.time()
    try:
        while time.time() - t0 < 10.0:      # ten seconds of it growing
            verdicts.append(fs.is_quiescent(target, **kw))
    finally:
        stop.set()
        t.join(timeout=10)
    done = time.time()
    assert verdicts and not any(verdicts), \
        f"{sum(verdicts)} of {len(verdicts)} judgements said 'send it'"
    size = target.stat().st_size
    while not fs.is_quiescent(target, **kw):
        assert time.time() - done < 30, "it never became sendable"
        time.sleep(0.25)
    settled = time.time() - done
    assert target.stat().st_size == size
    with capsys.disabled():
        print(f"\n  MEASURED, shipped constants "
              f"(samples={kw['samples']}, interval={kw['interval_s']}s, "
              f"quiet={kw['min_quiet_s']}s):\n"
              f"    {len(verdicts)} judgements over {done - t0:.1f}s of a "
              f"file growing to {size / 1048576:.0f} MB: "
              f"{sum(verdicts)} said send it\n"
              f"    became sendable {settled:.2f}s after the last write")


# ----------------------------------------------- the thing that runs it
UNIT = Path(__file__).resolve().parents[1] / \
    "scripts" / "systemd" / "jarvis-foldersync.service"


def test_the_unit_survives_a_reboot_without_him_logging_in():
    live = [ln.strip() for ln in UNIT.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    assert "WantedBy=default.target" in live        # + lingering, already on
    assert "Restart=always" in live
    # It must NOT need the desktop session -- that is the whole reason it is
    # not a thread inside Jarvis, which is a Tk app.
    assert not [ln for ln in live if "DISPLAY" in ln or "graphical" in ln]


def test_a_preflight_refusal_does_not_become_a_restart_loop():
    """main() returns 2 for a config shape no restart can fix; without this
    line five restarts in ten minutes replace the message with systemd's."""
    assert "RestartPreventExitStatus=2" in UNIT.read_text()


def test_the_installer_makes_the_folders_and_does_not_start_it():
    sh = (Path(__file__).resolve().parents[1] /
          "scripts" / "setup_foldersync_service.sh").read_text()
    assert "Desktop/Jarvis/Outbox" in sh and "Desktop/Jarvis/Inbox" in sh
    assert "systemctl --user enable" in sh
    assert "systemctl --user start" not in sh.split("echo")[0]


def test_a_folder_shape_that_would_loop_refuses_to_start(home, monkeypatch):
    """Exit 2, not a running syncer: the one config mistake that would copy
    every arrival straight back."""
    same = fs.Paths(outbox=home / "Inbox", inbox=home / "Inbox",
                    sent=home / "Sent", status=home / "status.txt")
    conf = remote.read_config(Cfg())
    sconf = fs.SyncConfig(enabled=True, paths=same)
    monkeypatch.setattr(fs, "build", lambda cfg: (
        fs.Syncer(conf, sconf, FakeTransport()), fs.preflight(conf, same)))
    monkeypatch.setattr("jarvis.assistant_config.AssistantConfig.load",
                        staticmethod(lambda *a, **k: Cfg()))
    assert fs.main([]) == 2


def test_the_lane_ships_off_so_nothing_moves_until_he_says_so(home, monkeypatch):
    conf = remote.read_config(Cfg())
    sconf = fs.SyncConfig(enabled=False, paths=paths_for(home))
    t = FakeTransport(outbox={"x.txt": (1, "Sep  5 13:45")})
    monkeypatch.setattr(fs, "build",
                        lambda cfg: (fs.Syncer(conf, sconf, t), []))
    monkeypatch.setattr("jarvis.assistant_config.AssistantConfig.load",
                        staticmethod(lambda *a, **k: Cfg()))
    # 3, not 0: the unit is Restart=always, so a 0 here would have systemd
    # restarting this every 15 seconds for as long as the switch is off.
    assert fs.main([]) == 3
    assert t.calls == []
    assert "RestartPreventExitStatus=2 3" in UNIT.read_text()


# ------------------------------------------ what self-review caught (09-05)
def test_a_file_that_can_never_go_does_not_poll_hpcomputer_forever(home):
    """THE SPIN.  A name Windows refuses sits in the Outbox indefinitely.
    The fast local scan must not turn that into an ssh handshake every two
    seconds against a machine that may be asleep."""
    t = FakeTransport()
    drop(home, "CON.txt", b"x")
    s = syncer(home, t)
    ticks = [0]

    def fake_sleep(_):
        ticks[0] += 1
        if ticks[0] > 40:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        s.loop(sleep=fake_sleep)
    # 40 two-second ticks is 80 seconds: the 30 s remote cadence allows a
    # handful of passes, and the unchanged refused file adds none.
    assert len([c for c in t.calls if c[0] == "listing"]) <= 4


def test_a_newly_dropped_file_is_not_made_to_wait_for_the_remote_cadence(home):
    t = FakeTransport()
    s = syncer(home, t)
    ticks = [0]

    def fake_sleep(_):
        ticks[0] += 1
        if ticks[0] == 2:
            drop(home, "urgent.pdf", b"a" * 10)
        if ticks[0] > 5:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        s.loop(sleep=fake_sleep)
    assert (home / "Sent" / "urgent.pdf").exists()      # inside ~10 seconds


def test_an_overnight_outage_does_not_fill_the_record(home):
    """One row for the transition, not one per pass -- or a night with
    HPCOMPUTER off would push every real transfer out of the record."""
    t = FakeTransport()
    t.fail = "unreachable"
    drop(home, "waiting.pdf", b"a")
    s = syncer(home, t)
    for _ in range(200):
        s.run_pass()
    rows = s.history_path.read_text().splitlines() \
        if s.history_path.exists() else []
    assert len(rows) == 1
    assert json.loads(rows[0])["outcome"] == "link-down"


def test_a_symlink_out_of_the_folders_is_refused_not_followed(home):
    """A drop folder is a place anyone can put a thing.  A symlink to
    ~/.ssh/id_ed25519 must be refused by containment, not sent."""
    t = FakeTransport()
    secret = home.parent.parent / "pretend-key"
    secret.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----")
    link = home / "Outbox" / "innocent.txt"
    link.symlink_to(secret)
    s = syncer(home, t)
    events = s.push_once()
    assert [e.outcome for e in events] == ["outside"]
    assert not [c for c in t.calls if c[0] == "send"]
    assert link.exists()                                # left exactly as-is


def test_the_fail_counter_is_cleared_when_a_file_finally_goes(home):
    """The key is taken BEFORE the move; after it, stat_key() is "" and the
    parked-attempt row would have been left behind forever."""
    t = FakeTransport()
    t.short_write = 1
    drop(home, "flaky.pdf", b"aa")
    s = syncer(home, t)
    s.push_once()
    assert any(k.startswith("push:flaky.pdf") for k in s.ledger.fails)
    t.short_write = 0
    s.push_once()
    assert not [k for k in s.ledger.fails if k.startswith("push:flaky.pdf")]
    assert (home / "Sent" / "flaky.pdf").exists()


def test_the_status_never_claims_a_link_it_has_not_checked(home):
    """A fresh process has not spoken to HPCOMPUTER; saying "OK" there is
    the confident-wrong-number failure this project has had twice."""
    s = syncer(home, FakeTransport())
    assert "not checked yet" in s.status_text()
    assert "link      OK" not in s.status_text()
    s.run_pass()
    assert "link      OK, last answered" in s.status_text()


def test_a_down_status_says_since_when(home):
    t = FakeTransport()
    t.fail = "unreachable"
    drop(home, "waiting.pdf", b"a")
    s = syncer(home, t)
    s.run_pass()
    first = s._down_since
    for _ in range(3):
        s.run_pass()
    assert s._down_since == first          # the outage clock does not reset
    assert "not answering since" in (home / "status.txt").read_text()


# ==========================================================================
# The two defects an adversarial pass measured on 2026-09-05, pinned here.
#
# Neither is about a name or a size; both are about a WINDOW.  The pull
# landed a file onto a name it had checked at the top of the pass, seconds
# or minutes before the write, so a file HE put in the Inbox during the
# transfer was overwritten and the record called it "received".  And one
# unsendable local file was read as proof that HPCOMPUTER was gone, which
# stopped the inbound half of a link that was demonstrably answering.
# ==========================================================================

# ------------------------------------------------ 1. the pull never clobbers
def test_a_file_of_his_dropped_in_the_inbox_mid_fetch_is_never_destroyed(home):
    """MEASURED against the code before this fix: a 513-byte notes.txt of
    his, written into ~/Desktop/Jarvis/Inbox while a same-named file was in
    flight from Windows, was GONE after the pass -- no note, no duplicate --
    and the row read "received"."""
    his = b"h" * 513
    t = FakeTransport(outbox={"notes.txt": (300, "Sep  5 13:45")})
    real_fetch = t.fetch

    def he_drops_one_mid_transfer(key, name, dest):
        r = real_fetch(key, name, dest)
        (home / "Inbox" / "notes.txt").write_bytes(his)
        return r

    t.fetch = he_drops_one_mid_transfer
    s = syncer(home, t)
    events = s.pull_once()

    assert (home / "Inbox" / "notes.txt").read_bytes() == his   # HIS, intact
    assert (home / "Inbox" / "notes (2).txt").stat().st_size == 300
    assert [(e.outcome, e.detail) for e in events] == \
           [("received", "notes (2).txt")]


def test_the_record_and_the_status_name_the_file_that_actually_landed(home):
    """The old row said "received notes.txt" for a file that is not there
    under that name.  What he reads must be the name on disk."""
    t = FakeTransport(outbox={"notes.txt": (300, "Sep  5 13:45")})
    (home / "Inbox" / "notes.txt").write_bytes(b"h" * 513)
    s = syncer(home, t)
    s.run_pass()
    text = (home / "status.txt").read_text()
    assert "notes (2).txt" in text
    rows = [json.loads(r) for r in s.history_path.read_text().splitlines()]
    assert [(r["outcome"], r["detail"]) for r in rows] == \
           [("received", "notes (2).txt")]


def test_the_landing_refuses_a_name_taken_at_the_instant_it_is_claimed(home):
    """The window pinned as tight as it goes: his file appears BETWEEN the
    free name being chosen and the link being made.  A check-then-write
    loses here however narrow the check is; os.link cannot."""
    t = FakeTransport(outbox={"notes.txt": (300, "Sep  5 13:45")})
    real_link = os.link
    raced = []

    def racing_link(src, dst, **kw):
        if not raced:
            raced.append(str(dst))
            Path(dst).write_bytes(b"his, by a microsecond")
        return real_link(src, dst, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "link", racing_link)
        s = syncer(home, t)
        events = s.pull_once()

    assert raced == [str(home / "Inbox" / "notes.txt")]
    assert (home / "Inbox" / "notes.txt").read_bytes() == b"his, by a microsecond"
    assert (home / "Inbox" / "notes (2).txt").stat().st_size == 300
    assert [e.outcome for e in events] == ["received"]


def test_the_landing_still_never_overwrites_on_a_filesystem_without_links(home):
    """A FAT or exFAT Inbox has no hard links.  The fallback claims the name
    with O_EXCL, which is atomic too -- it must not degrade to a check."""
    t = FakeTransport(outbox={"notes.txt": (300, "Sep  5 13:45")})
    his = b"h" * 513
    (home / "Inbox" / "notes.txt").write_bytes(his)

    def no_hardlinks(src, dst, **kw):
        raise OSError(errno.EPERM, "hard links not supported here")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "link", no_hardlinks)
        s = syncer(home, t)
        events = s.pull_once()

    assert (home / "Inbox" / "notes.txt").read_bytes() == his
    assert (home / "Inbox" / "notes (2).txt").stat().st_size == 300
    assert [e.outcome for e in events] == ["received"]


def test_the_landing_takes_the_atomic_link_and_never_a_check(home):
    """Which path runs is the difference between a window of ZERO and one
    of 0.06 ms.  ~/Desktop is ext4 (measured with stat -f), so os.link is
    what runs on his machine -- and nothing may quietly degrade to the
    O_EXCL fallback without this noticing."""
    t = FakeTransport(outbox={"notes.txt": (3, "Sep  5 13:45")})
    used = []
    real_link, real_open = os.link, os.open

    def watch_link(src, dst, **kw):
        used.append("link")
        return real_link(src, dst, **kw)

    def watch_open(path, flags, *a, **k):
        if flags & os.O_EXCL:
            used.append("o_excl")
        return real_open(path, flags, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "link", watch_link)
        mp.setattr(os, "open", watch_open)
        syncer(home, t).pull_once()
    assert used == ["link"]
    assert (home / "Inbox" / "notes.txt").exists()


def test_the_landing_leaves_no_part_file_and_gives_up_past_fifty(home):
    t = FakeTransport(outbox={"notes.txt": (3, "Sep  5 13:45")})
    (home / "Inbox" / "notes.txt").write_bytes(b"x")
    for n in range(2, fs.MAX_COPIES + 2):
        (home / "Inbox" / f"notes ({n}).txt").write_bytes(b"x")
    s = syncer(home, t)
    events = s.pull_once()
    assert [e.outcome for e in events] == ["too-many-copies"]
    assert not [p for p in (home / "Inbox").iterdir()
                if p.name.startswith(fs.PART_PREFIX)]


def test_land_beside_is_the_one_landing_and_it_never_replaces(tmp_path):
    """Directly, without a transport: the helper both directions land
    through returns the name it actually took and leaves what was there."""
    folder = tmp_path / "f"
    folder.mkdir()
    (folder / "a.txt").write_bytes(b"first")
    src = tmp_path / "src"
    src.write_bytes(b"second")
    assert fs.land_beside(src, folder, "a.txt") == "a (2).txt"
    assert (folder / "a.txt").read_bytes() == b"first"
    assert (folder / "a (2).txt").read_bytes() == b"second"
    assert not src.exists()                       # it was a MOVE


def test_his_file_in_sent_survives_a_name_taken_at_the_instant_of_the_move(home):
    """The push side had the same shape, narrower: it re-read Sent and then
    called os.replace.  Same helper, same guarantee."""
    t = FakeTransport()
    drop(home, "report.pdf", b"the one being sent")
    (home / "Sent").mkdir()
    real_link = os.link
    raced = []

    def racing_link(src, dst, **kw):
        if not raced and Path(dst).parent.name == "Sent":
            raced.append(str(dst))
            Path(dst).write_bytes(b"an older keepsake")
        return real_link(src, dst, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "link", racing_link)
        s = syncer(home, t)
        events = s.push_once()

    assert [e.outcome for e in events] == ["sent"]
    assert (home / "Sent" / "report.pdf").read_bytes() == b"an older keepsake"
    assert (home / "Sent" / "report (2).pdf").read_bytes() == b"the one being sent"
    assert not (home / "Outbox" / "report.pdf").exists()


# ------------------------------------- 2. a file problem is not a link problem
def _poison(home, name="poison.txt"):
    """A far side that is HEALTHY -- its listings answer -- but that refuses
    one particular send with the reason classify_error falls back to when
    none of its seven regexes match."""
    t = FakeTransport()
    t.send_fail = {name: "failed"}
    drop(home, name, b"a" * 40)
    return t


def test_one_unsendable_file_does_not_stop_the_inbound_half(home):
    """MEASURED before this fix: 0 fetches over 12 passes, so a file waiting
    in the Windows Outbox never arrived at all."""
    t = _poison(home)
    t.put("outbox", "from_windows.txt", b"z" * 90)
    s = syncer(home, t)
    events = s.run_pass()
    assert (home / "Inbox" / "from_windows.txt").stat().st_size == 90
    assert "received" in [e.outcome for e in events]


def test_a_send_failure_never_claims_a_link_that_just_answered_is_down(home):
    """The same pass listed HPCOMPUTER twice, successfully.  Saying it
    "wouldn't answer" is a confident wrong number about a machine that did."""
    t = _poison(home)
    s = syncer(home, t)
    for _ in range(3):
        s.run_pass()
    text = (home / "status.txt").read_text()
    assert "link      DOWN" not in text
    assert "link      OK, last answered" in text
    assert s.interval_s == s.conf.remote_interval_s      # never backed off


def test_a_second_good_file_still_goes_when_the_first_cannot(home):
    """MEASURED before this fix: a perfectly good file went unsent for 6
    passes because the poison one broke out of the loop ahead of it."""
    t = _poison(home, "a_poison.txt")
    drop(home, "b_good.pdf", b"g" * 20)
    s = syncer(home, t)
    events = s.push_once()
    assert ("b_good.pdf", "sent") in [(e.name, e.outcome) for e in events]
    assert (home / "Sent" / "b_good.pdf").exists()


def test_an_unsendable_file_parks_instead_of_retrying_for_ever(home):
    """MEASURED before this fix: 30 send attempts over 30 passes with the
    ledger's fails table EMPTY -- MAX_ATTEMPTS was never applied on that
    path.  A file this box cannot send is a file, and files park."""
    t = _poison(home)
    s = syncer(home, t)
    for _ in range(fs.MAX_ATTEMPTS + 8):
        s.run_pass()
    sends = [c for c in t.calls if c[0] == "send"]
    assert len(sends) == fs.MAX_ATTEMPTS
    assert any(k.startswith("push:poison.txt") for k in s.ledger.fails)
    assert (home / "Outbox" / "poison.txt").exists()          # never deleted
    assert (home / "Outbox" / ("poison.txt" + fs.NOTE_SUFFIX)).exists()


def test_an_unsendable_file_does_not_flood_the_log_every_pass(home, caplog):
    """MEASURED before this fix: a WARNING plus an INFO "HPCOMPUTER is
    answering again" on EVERY pass, for ever, from a healthy link."""
    t = _poison(home)
    t.put("outbox", "from_windows.txt", b"z" * 5)
    s = syncer(home, t)
    with caplog.at_level("INFO", logger="jarvis.foldersync"):
        for _ in range(12):
            s.run_pass()
    msgs = [r.getMessage() for r in caplog.records]
    assert not [m for m in msgs
                if "answering again" in m or "backing off" in m]


def test_a_link_that_really_is_down_mid_pass_still_stops_the_queue(home):
    """The separation must not go the other way: when the box goes away
    between the listing and the send, that IS the link, the rest of the
    queue waits, and the inbound half is not attempted."""
    t = FakeTransport()
    drop(home, "one.txt", b"a")
    drop(home, "two.txt", b"b")
    s = syncer(home, t)

    def send_then_the_box_sleeps(local, name, key="inbox"):
        t.calls.append(("send", name))
        t.fail = "asleep"                  # every later call, listing included
        return "failed"

    t.send = send_then_the_box_sleeps
    events = s.run_pass()
    assert [e.outcome for e in events] == ["link-down"]
    assert len([c for c in t.calls if c[0] == "send"]) == 1   # it stopped
    assert not [c for c in t.calls if c[0] == "fetch"]
    assert s.interval_s > s.conf.remote_interval_s


def test_one_bad_fetch_does_not_declare_the_link_down_either(home):
    """The pull half had the same shape in a narrower form: a timeout on one
    oversized file spoke for the whole machine."""
    t = FakeTransport(outbox={"huge.bin": (100, "Sep  5 13:45"),
                              "small.txt": (4, "Sep  5 13:45")})
    t.fetch_fail = {"huge.bin": "timeout"}
    s = syncer(home, t)
    events = s.pull_once()
    outcomes = {(e.name, e.outcome) for e in events}
    assert ("small.txt", "received") in outcomes
    assert ("huge.bin", "timeout") in outcomes
    assert "link-down" not in [e.outcome for e in events]
    assert s.ledger.fails                                  # it counted
    assert "link      DOWN" not in s.status_text()


def test_a_verify_listing_that_fails_is_the_link_and_costs_the_file_nothing(home):
    """The other half of the rule, found by the same sweep: the listing
    that VERIFIES a push is a listing, and a listing that cannot answer is
    the one thing that really is evidence about the link.  It used to be
    swallowed -- no back-off, and the inbound half then dialled the same
    dead box again in the same pass.  His file must not be charged an
    attempt for an outage."""
    t = FakeTransport()
    drop(home, "report.pdf", b"a" * 50)
    s = syncer(home, t)
    real_listing = t.listing
    seen = []

    def listing_then_the_box_goes(key):
        seen.append(key)
        if len(seen) > 1:                  # the top-of-pass one answered
            return [], "asleep"
        return real_listing(key)

    t.listing = listing_then_the_box_goes
    events = s.run_pass()
    assert [e.outcome for e in events] == ["unverified", "link-down"]
    assert not [c for c in t.calls if c[0] == "fetch"]     # no second dial
    assert s.interval_s > s.conf.remote_interval_s         # it backed off
    assert not s.ledger.fails                              # never his fault
    assert (home / "Outbox" / "report.pdf").exists()


# ------------------------------------------- a folder on the far side is real
def test_the_sftp_listing_keeps_folders_instead_of_dropping_them(home):
    """parse_sftp_entries used to skip every line starting with "d", so a
    FOLDER on the Windows side was invisible to dedupe_name -- which is
    exactly how an unsendable name arises."""
    out = ('sftp> ls -ln "Desktop/Jarvis/Inbox"\n'
           "-rw-rw-r--    ? hunterp  hunterp   1 Sep  3 12:21 "
           "jarvis-inbox/a.txt\n"
           "drwxrwxr-x    ? hunterp  hunterp 4096 Sep  3 12:21 "
           "jarvis-inbox/reports\n")
    rows = fs.parse_sftp_entries(out)
    assert [(r.name, r.is_dir) for r in rows] == [("a.txt", False),
                                                  ("reports", True)]


def test_a_file_whose_name_is_a_folder_over_there_lands_beside_it(home):
    """The reachable poison case, closed at the source: the folder is in the
    listing now, so dedupe_name never picks that name in the first place."""
    t = FakeTransport()
    t.folders["inbox"].add("reports")
    drop(home, "reports", b"a file, not a folder")
    s = syncer(home, t)
    events = s.push_once()
    assert [(e.outcome, e.detail) for e in events] == [("sent", "reports (2)")]
    assert "reports" in t.folders["inbox"]           # his folder untouched


def test_a_folder_in_the_windows_outbox_is_not_fetched(home):
    t = FakeTransport(outbox={"ok.txt": (2, "Sep  5 13:45")})
    t.folders["outbox"].add("a whole folder")
    s = syncer(home, t)
    events = s.pull_once()
    assert [e.name for e in events] == ["ok.txt"]
    assert not [c for c in t.calls if c == ("fetch", "a whole folder")]
    assert sorted(p.name for p in (home / "Inbox").iterdir()) == ["ok.txt"]


def test_a_folder_over_there_never_passes_verification_as_a_file(home):
    """If a folder somehow appears at the name we just sent, the size check
    must FAIL rather than read the folder's own size as the file's."""
    t = FakeTransport()
    drop(home, "reports", b"a" * 4096)
    s = syncer(home, t)

    real_listing = t.listing
    calls = []

    def listing_that_grows_a_folder(key):
        calls.append(key)
        if len(calls) > 1:
            t.folders["inbox"].add("reports")
            t.dirs["inbox"].pop("reports", None)
        return real_listing(key)

    t.listing = listing_that_grows_a_folder
    events = s.push_once()
    assert [e.outcome for e in events] == ["verify-failed"]
    assert (home / "Outbox" / "reports").exists()          # his file stays


# ----------------------------------------- the window, measured not claimed
def test_measure_the_window_that_is_left_after_the_race_is_closed(tmp_path,
                                                                  capsys):
    """NUMBERS, because "we closed it" is a claim and this project has twice
    been damaged by confident numbers with no source.

    Two things are measured.  First a HAMMER: a thread looping on the same
    name while the lane lands file after file on it -- if the landing were
    a check followed by a write, this is where bytes go missing.  Then the
    one gap the no-hard-links fallback leaves, which is the ONLY window
    still open anywhere in either direction, and how many milliseconds
    wide it is.

    ~/Desktop is ext4 (measured 2026-09-05, `stat -f -c %T`), so the
    fallback does not run on his machine at all; it is there for a FAT or
    exFAT folder and is measured so the number is not a guess.
    """
    folder = tmp_path / "hammer"
    folder.mkdir()
    rounds = 2000
    stop = threading.Event()
    ours_wrong = 0

    def competitor():
        while not stop.is_set():
            try:
                (folder / "notes.txt").write_bytes(b"H" * 513)
            except OSError:
                pass

    th = threading.Thread(target=competitor, daemon=True)
    th.start()
    t0 = time.perf_counter()
    try:
        for i in range(rounds):
            src = tmp_path / f"src{i}"
            src.write_bytes(b"A" * 300)
            landed = folder / fs.land_beside(src, folder, "notes.txt")
            if landed.read_bytes() != b"A" * 300:
                ours_wrong += 1
            landed.unlink()
    finally:
        stop.set()
        th.join(timeout=5)
    per_landing_ms = (time.perf_counter() - t0) * 1000 / rounds

    assert ours_wrong == 0, \
        f"{ours_wrong} of {rounds} landings took somebody else's bytes"
    assert (folder / "notes.txt").exists(), \
        "the competitor's file was destroyed by a landing"

    # The fallback's only window: between the atomic O_EXCL claim and the
    # replace onto our own 0-byte claim.
    gaps = []
    for i in range(2000):
        src = tmp_path / f"g{i}"
        src.write_bytes(b"B" * 300)
        dest = tmp_path / f"d{i}.txt"
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        t1 = time.perf_counter_ns()             # window OPENS
        os.close(fd)
        os.replace(src, dest)                   # window CLOSES
        gaps.append(time.perf_counter_ns() - t1)
    gaps.sort()
    with capsys.disabled():
        print(f"\n  MEASURED, the window after the fix:"
              f"\n    hard-link path (his ext4 Desktop): {rounds} landings "
              f"against a\n      writer looping on the same name -- "
              f"{ours_wrong} lost, {per_landing_ms:.3f} ms each."
              f"\n      os.link is one syscall, so the window is ZERO, not "
              f"narrowed."
              f"\n    O_EXCL fallback (FAT/exFAT only): median "
              f"{gaps[len(gaps)//2]/1e6:.4f} ms, "
              f"p99 {gaps[int(len(gaps)*0.99)]/1e6:.4f} ms, "
              f"max {gaps[-1]/1e6:.4f} ms"
              f"\n      -- and a 0-byte file OF OURS holds the name for it, "
              f"so what\n      that window can cost is his WRITE, never his "
              f"file.")


# ==========================================================================
# ROUND 3.  The same shape as the pull race, found on the OTHER side.
#
# FINDING J: a push listed the remote Inbox ONCE at the top of a pass, chose
# a free name, and then scp'd straight at it.  scp TRUNCATES -- measured
# against the real box, a 22222-byte file replaced by 1111 bytes, exit 0, no
# message -- so a file of his that appeared at that name between the listing
# and the copy was destroyed, the pass said "sent", and his LOCAL original
# was then moved into Sent.  Both copies ours, his gone.  And the window is
# not milliseconds: the sends are sequential after ONE listing, so the tenth
# file in a queue is written minutes after its name was checked.
#
# The fix is not a narrower window.  MEASURED on HPCOMPUTER 2026-09-05: the
# sftp client's DEFAULT rename (posix-rename@openssh.com) silently replaces,
# 10 of 10; `rename -l` (SSH2_FXP_RENAME) REFUSES, 10 of 10, both files
# intact; and two sessions racing for one name, 12 rounds, produced exactly
# one winner each time with zero losses.  So: scp to a temp name of OURS,
# then claim the real name with a rename that can say no.
# ==========================================================================
def _plant_on_send(t, name, size):
    """He saves a file of his into the Windows Inbox AFTER the listing at
    the top of the pass and BEFORE our bytes land -- the exact window."""
    real = t.send

    def send(local, temp, key="inbox"):
        t.dirs["inbox"].setdefault(name, (size, t.stamp))
        return real(local, temp, key)

    t.send = send


def test_a_file_of_his_that_appears_on_hpcomputer_mid_push_is_never_overwritten(
        home):
    t = FakeTransport()
    drop(home, "report.pdf", b"o" * 30)
    s = syncer(home, t)
    _plant_on_send(t, "report.pdf", 99999)

    events = s.push_once()

    assert t.dirs["inbox"]["report.pdf"] == (99999, t.stamp)   # HIS, intact
    assert [(e.outcome, e.detail) for e in events] == [("sent", "report (2).pdf")]
    assert t.dirs["inbox"]["report (2).pdf"][0] == 30          # ours, beside
    assert not (home / "Outbox" / "report.pdf").exists()       # moved on
    assert (home / "Sent" / "report.pdf").read_bytes() == b"o" * 30


def test_the_push_writes_at_a_name_of_ours_and_never_at_his(home):
    """Every byte scp puts on that machine goes at a name this module made
    up; the name he will see is taken by a rename that can refuse."""
    t = FakeTransport()
    drop(home, "report.pdf", b"o" * 30)
    s = syncer(home, t)
    s.push_once()
    written = [name for kind, name in t.calls if kind == "send"]
    assert len(written) == 1
    assert written[0].startswith(remote.REMOTE_TEMP_PREFIX)
    assert written[0].endswith(".tmp")
    assert t.claims == [(written[0], "report.pdf")]


def test_the_tenth_file_in_a_queue_is_claimed_when_it_is_sent(home):
    """Not when the pass started.  One listing, ten sequential copies: the
    name of the last one was checked minutes before its bytes moved."""
    t = FakeTransport()
    for i in range(10):
        drop(home, f"f{i}.txt", b"x" * (i + 1))
    s = syncer(home, t)
    sends = []
    real = t.send

    def send(local, temp, key="inbox"):
        sends.append(temp)
        if len(sends) == 10:                       # he saves his over it now
            t.dirs["inbox"]["f9.txt"] = (99999, t.stamp)
        return real(local, temp, key)

    t.send = send
    events = s.push_once()

    assert t.dirs["inbox"]["f9.txt"] == (99999, t.stamp)       # HIS, intact
    landed = {e.name: e.detail for e in events if e.outcome == "sent"}
    assert landed["f9.txt"] == "f9 (2).txt"
    assert len(landed) == 10                       # the other nine unaffected


def test_a_claim_that_cannot_be_taken_leaves_nothing_of_ours_behind(home):
    """Fifty names taken and the fifty-first too: his file stays in the
    Outbox, he gets a note, and our temp is taken off his machine."""
    t = FakeTransport()
    drop(home, "notes.txt", b"a" * 10)
    s = syncer(home, t)

    def every_name_is_taken(temp, final, key="inbox"):
        t.claims.append((temp, final))
        return "taken"

    t.claim = every_name_is_taken
    events = s.push_once()

    assert [e.outcome for e in events] == ["name-taken"]
    assert (home / "Outbox" / "notes.txt").exists()            # his file stays
    assert t.discarded and all(n.startswith(remote.REMOTE_TEMP_PREFIX)
                               for n in t.discarded)
    assert not [n for n in t.dirs["inbox"]
                if n.startswith(remote.REMOTE_TEMP_PREFIX)]
    assert (home / "Outbox" / ("notes.txt" + fs.NOTE_SUFFIX)).exists()


def test_the_number_of_claim_attempts_is_bounded(home):
    t = FakeTransport()
    drop(home, "notes.txt", b"a" * 10)
    s = syncer(home, t)
    t.claim = lambda temp, final, key="inbox": (t.claims.append((temp, final))
                                                or "taken")
    s.push_once()
    assert 0 < len(t.claims) <= fs.MAX_CLAIM_TRIES


def test_a_leftover_temp_of_ours_is_taken_off_his_machine_next_pass(home):
    """A process killed between the copy and the claim leaves a temp in his
    Inbox.  It is ours, it is named ours, and it is the only thing this
    module may ever delete over there."""
    t = FakeTransport()
    stale = remote.remote_temp_name(0)
    t.dirs["inbox"][stale] = (10, t.stamp)
    t.dirs["inbox"]["his report.pdf"] = (5, t.stamp)
    drop(home, "new.txt", b"a")
    s = syncer(home, t)
    s.push_once()
    assert stale in t.discarded
    assert stale not in t.dirs["inbox"]
    assert "his report.pdf" in t.dirs["inbox"]                 # untouched


def test_the_verification_is_against_the_name_we_actually_took(home):
    """His original moves only when the far side holds OUR bytes at OUR
    size under the name the claim actually won -- not the name we asked
    for first."""
    t = FakeTransport()
    t.put("inbox", "report.pdf", b"h" * 4)                     # his, already
    drop(home, "report.pdf", b"o" * 30)
    s = syncer(home, t)
    real_listing = t.listing
    calls = []

    def listing(key):
        calls.append(key)
        if len(calls) > 1:                # the verify: our landing is short
            t.dirs["inbox"]["report (2).pdf"] = (7, t.stamp)
        return real_listing(key)

    t.listing = listing
    events = s.push_once()
    assert [e.outcome for e in events] == ["verify-failed"]
    assert (home / "Outbox" / "report.pdf").exists()           # not moved
    assert t.dirs["inbox"]["report.pdf"] == (4, t.stamp)       # his, intact


def test_a_claim_that_fails_because_the_box_went_away_is_the_link(home):
    """A rename that fails is not always contention.  When the machine has
    gone, that is the link -- the queue stops and his file is charged
    nothing."""
    t = FakeTransport()
    drop(home, "one.txt", b"a")
    drop(home, "two.txt", b"b")
    s = syncer(home, t)

    def claim(temp, final, key="inbox"):
        t.calls.append(("claim", final))
        t.fail = "asleep"
        return "asleep"

    t.claim = claim
    events = s.run_pass()
    assert [e.outcome for e in events] == ["link-down"]
    assert len([c for c in t.calls if c[0] == "send"]) == 1
    assert not [c for c in t.calls if c[0] == "fetch"]
    assert not s.ledger.fails                       # an outage is never his


# ==========================================================================
# FINDING K: a missing folder is a CONFIGURATION problem, not a dead link.
#
# MEASURED over 6 passes with the far side healthy and answering: a missing
# remote Inbox made the top-of-pass listing return "not-there", which went
# straight to the link handler -- 0 fetches, the file waiting in the Windows
# Outbox never arrived, the interval backed off 30s -> 300s, and status.txt
# said "link DOWN ... not answering since 15:36".  A control with the same
# transport and an empty local Outbox completed the pull, which proves the
# box was answering the whole time.
# ==========================================================================
def test_a_missing_remote_folder_does_not_back_off(home):
    t = FakeTransport()
    t.listing_fail = {"inbox": "not-there"}
    drop(home, "waiting.pdf", b"a")
    s = syncer(home, t)
    for _ in range(6):
        s.run_pass()
    assert s.interval_s == s.conf.remote_interval_s


def test_a_missing_inbox_does_not_stop_the_pull(home):
    """The control test, in code: the box is answering and the OUTBOX is
    fine.  The file waiting there must still arrive."""
    t = FakeTransport(outbox={"from-him.txt": (3, "Sep  5 13:45")})
    t.listing_fail = {"inbox": "not-there"}
    drop(home, "waiting.pdf", b"a")
    s = syncer(home, t)
    events = s.run_pass()
    assert ("from-him.txt", "received") in {(e.name, e.outcome) for e in events}
    assert (home / "Inbox" / "from-him.txt").exists()
    assert (home / "Outbox" / "waiting.pdf").exists()          # still waiting


def test_the_status_says_which_folder_it_could_not_find(home):
    t = FakeTransport()
    t.listing_fail = {"inbox": "not-there"}
    drop(home, "waiting.pdf", b"a")
    s = syncer(home, t)
    s.run_pass()
    text = (home / "status.txt").read_text()
    assert "jarvis-inbox" in text                   # the path it looked for
    assert "remote.inbox" in text                   # the setting to fix
    assert "not answering" not in text.lower()      # it IS answering
    assert "link      DOWN" not in text


def test_a_folder_that_comes_back_clears_the_complaint(home):
    t = FakeTransport()
    t.listing_fail = {"inbox": "not-there"}
    drop(home, "waiting.pdf", b"a")
    s = syncer(home, t)
    s.run_pass()
    assert "NOT THERE" in (home / "status.txt").read_text()
    t.listing_fail = {}
    s.run_pass()
    text = (home / "status.txt").read_text()
    assert "NOT THERE" not in text
    assert not (home / "Outbox" / "waiting.pdf").exists()      # and it went


def test_a_missing_folder_does_not_fill_the_record_either(home):
    t = FakeTransport()
    t.listing_fail = {"inbox": "not-there"}
    drop(home, "waiting.pdf", b"a")
    s = syncer(home, t)
    for _ in range(40):
        s.run_pass()
    rows = [json.loads(x) for x in s.history_path.read_text().splitlines()]
    assert len(rows) == 1


def test_a_missing_folder_says_so_in_the_log_once(home, caplog):
    t = FakeTransport()
    t.listing_fail = {"inbox": "not-there"}
    drop(home, "waiting.pdf", b"a")
    s = syncer(home, t)
    with caplog.at_level("WARNING", logger="jarvis.foldersync"):
        for _ in range(20):
            s.run_pass()
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "jarvis-inbox" in warnings[0].getMessage()


def test_a_setting_with_no_socket_behind_it_is_not_a_dead_link_either(home):
    """"no-key" never opened a connection, so it cannot be evidence that
    HPCOMPUTER is not answering, and it must not back off either."""
    t = FakeTransport()
    t.fail = "no-key"
    drop(home, "waiting.pdf", b"a")
    s = syncer(home, t)
    for _ in range(4):
        s.run_pass()
    text = (home / "status.txt").read_text()
    assert s.interval_s == s.conf.remote_interval_s
    assert "not answering" not in text.lower()
    assert "key" in text.lower()


# -------------------------------------------- caught before a socket opens
def test_preflight_catches_a_pull_folder_that_is_not_configured(home):
    rconf = remote.read_config(Cfg(**{"remote.local_roots": (str(home.parent),)}))
    sconf = fs.SyncConfig(enabled=True, paths=paths_for(home),
                          pull_from="outbxo")
    problems = fs.preflight(rconf, sconf.paths, sconf)
    assert any("pull_from" in p and "outbxo" in p for p in problems)


def test_preflight_catches_an_empty_remote_inbox(home):
    rconf = remote.read_config(Cfg(**{
        "remote.local_roots": (str(home.parent),)}))
    rconf = dataclasses.replace(rconf, inbox="")
    sconf = fs.SyncConfig(enabled=True, paths=paths_for(home))
    problems = fs.preflight(rconf, sconf.paths, sconf)
    assert any("remote.inbox" in p for p in problems)


def test_preflight_catches_a_remote_folder_it_could_never_quote(home):
    rconf = remote.read_config(Cfg(**{
        "remote.inbox": 'C:/Users/h2pey/"Inbox"',
        "remote.local_roots": (str(home.parent),)}))
    sconf = fs.SyncConfig(enabled=True, paths=paths_for(home))
    problems = fs.preflight(rconf, sconf.paths, sconf)
    assert any("remote.inbox" in p for p in problems)


def test_preflight_catches_a_remote_folder_too_deep_for_windows(home):
    rconf = remote.read_config(Cfg(**{
        "remote.inbox": "C:/" + "x" * 300,
        "remote.local_roots": (str(home.parent),)}))
    sconf = fs.SyncConfig(enabled=True, paths=paths_for(home))
    problems = fs.preflight(rconf, sconf.paths, sconf)
    assert any("remote.inbox" in p for p in problems)


def test_the_shipped_settings_pass_preflight(home):
    """The default pull_from really is a key remote.pull_dirs has."""
    rconf = remote.read_config(Cfg(**{"remote.local_roots": (str(home.parent),)}))
    sconf = fs.read_config(Cfg())
    assert fs.remote_folder_problems(rconf, sconf) == []


def test_check_asks_the_box_which_folder_is_missing(home, monkeypatch, capsys):
    """--check is his to run, and it is the one place that may ASK: an
    offline check cannot tell a mistyped path from a right one."""
    t = FakeTransport()
    t.listing_fail = {"inbox": "not-there"}
    s = syncer(home, t)
    monkeypatch.setattr(fs, "build", lambda cfg: (s, []))
    monkeypatch.setattr("jarvis.assistant_config.AssistantConfig.load",
                        staticmethod(lambda *a, **k: Cfg()))
    assert fs.main(["--check"]) == 1
    said = capsys.readouterr()
    assert "jarvis-inbox" in said.out + said.err
    assert "remote.inbox" in said.out + said.err


# ==================================================== the sweep, as a test
def test_no_remote_write_goes_at_a_name_of_his_without_a_claim(home):
    """The shape itself, pinned.  Every name scp is given on that machine
    is one this module made up; the only operation that ever touches a name
    he could have chosen is the claim, and the claim can refuse."""
    src = Path(fs.__file__).read_text()
    assert "def claim" in src or "self.transport.claim" in src
    t = FakeTransport(outbox={"in.txt": (2, "Sep  5 13:45")})
    for i in range(3):
        drop(home, f"f{i}.txt", b"x")
    s = syncer(home, t)
    s.run_pass()
    for kind, name in t.calls:
        if kind == "send":
            assert name.startswith(remote.REMOTE_TEMP_PREFIX)
        if kind == "discard":
            assert name.startswith(remote.REMOTE_TEMP_PREFIX)


def test_the_rename_is_the_one_that_refuses_not_the_one_that_replaces():
    """MEASURED, and the single most breakable line in this change: the
    sftp client's bare `rename` sends posix-rename@openssh.com, which this
    server offers and which SILENTLY REPLACED the target 10 times out of
    10.  `rename -l` forces SSH2_FXP_RENAME, which refused 10 out of 10.
    Writing `rename` here reproduces Finding J exactly."""
    seen = {}

    def fake_sftp(conf, batch, **kw):
        seen["batch"] = batch
        return remote.SshResult(True)

    conf = remote.read_config(Cfg(**{
        "remote.enabled": True, "remote.host": "h", "remote.user": "u"}))
    import jarvis.tools.remote as rmod
    old = rmod.run_sftp
    rmod.run_sftp = fake_sftp
    try:
        remote.sftp_rename(conf, "a/tmp.tmp", "a/final.pdf")
    finally:
        rmod.run_sftp = old
    assert seen["batch"].split()[:2] == ["rename", "-l"]


def test_measure_the_push_race_after_the_claim(home, capsys):
    """NUMBERS for the far side, the way round 2 produced them for this one.

    HONEST about what this measures and what it does not.  It measures the
    SHAPE: a file of his is planted at the exact target name during every
    single send, and if the push were still "list, pick a name, scp at it"
    every round would destroy one.  It cannot measure the kernel underneath
    -- that was measured against HPCOMPUTER itself on 2026-09-05 (two
    concurrent sftp sessions renaming onto one name, 12 rounds, 6-6, zero
    lost, zero both-moved), and the fake reproduces the two behaviours that
    measurement established: scp truncates, `rename -l` refuses.
    """
    rounds = 300
    t = FakeTransport()
    s = syncer(home, t)
    _plant_on_send(t, "notes.txt", 99999)
    his_lost = 0
    ours_wrong = 0
    for i in range(rounds):
        drop(home, "notes.txt", b"o" * 30)
        events = s.push_once()
        if t.dirs["inbox"].get("notes.txt") != (99999, t.stamp):
            his_lost += 1
        landed = [e.detail for e in events if e.outcome == "sent"]
        if landed and t.dirs["inbox"].get(landed[0], (0,))[0] != 30:
            ours_wrong += 1
        for name in list(t.dirs["inbox"]):
            if name != "notes.txt":
                t.dirs["inbox"].pop(name)
        for p in (home / "Sent").iterdir():
            p.unlink()

    assert his_lost == 0, f"{his_lost} of {rounds} pushes destroyed his file"
    assert ours_wrong == 0
    assert not [n for n in t.dirs["inbox"]
                if n.startswith(remote.REMOTE_TEMP_PREFIX)]
    with capsys.disabled():
        print(f"\n  MEASURED, the push after the fix:"
              f"\n    {rounds} pushes with a file of his appearing at the "
              f"target name\n      during EVERY send -- {his_lost} of his "
              f"destroyed, {ours_wrong} of ours wrong."
              f"\n    Every one landed beside his as \"notes (2).txt\" and "
              f"left no temp\n      behind.  The kernel underneath was "
              f"measured on HPCOMPUTER itself:\n      `rename -l` refused "
              f"10/10, and 12 racing rounds gave 6-6, 0 lost.")


def test_a_real_outage_takes_the_folder_complaint_down_with_it(home):
    """"That folder is not there" rested on the box having ANSWERED.  Once
    it stops answering, that sentence is no longer supported and must not
    sit in the status file beside "link DOWN"."""
    t = FakeTransport()
    t.listing_fail = {"inbox": "not-there"}
    drop(home, "waiting.pdf", b"a")
    s = syncer(home, t)
    s.run_pass()
    assert "NOT THERE" in (home / "status.txt").read_text()
    t.fail = "asleep"
    s.run_pass()
    text = (home / "status.txt").read_text()
    assert "NOT THERE" not in text
    assert "link      DOWN" in text
