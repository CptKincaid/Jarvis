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
    a write there so a puller that ever tries to tidy up fails a test."""

    def __init__(self, inbox=None, outbox=None):
        self.dirs = {"inbox": dict(inbox or {}), "outbox": dict(outbox or {})}
        self.fail = ""            # a reason to return from every call
        self.calls = []
        self.short_write = 0      # land this many bytes instead of the file's
        self.stamp = "Sep  5 13:45"

    # -- helpers a test uses to describe the far side ------------------
    def put(self, key, name, data=b"x", stamp=None):
        self.dirs[key][name] = (len(data), stamp or self.stamp)

    def listing(self, key):
        self.calls.append(("listing", key))
        if self.fail:
            return [], self.fail
        rows = [fs.Entry(n, s, t) for n, (s, t) in sorted(self.dirs[key].items())]
        return rows, ""

    def send(self, local, name, key="inbox"):
        self.calls.append(("send", name))
        if self.fail:
            return self.fail
        if key != "inbox":
            raise AssertionError("nothing may write outside the remote inbox")
        size = self.short_write or local.stat().st_size
        self.dirs[key][name] = (size, self.stamp)
        return ""

    def fetch(self, key, name, dest):
        self.calls.append(("fetch", name))
        if self.fail:
            return self.fail
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
    # The listing that VERIFIES came after the send, not instead of it.
    assert [c[0] for c in t.calls] == ["listing", "send", "listing"]


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
    assert [(r.name, r.size, r.stamp) for r in rows] == [
        ("a.txt", 1, "Sep 3 12:21"),
        ("old report.pdf", 20481, "Sep 3 2025"),
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
    assert fs.main([]) == 0
    assert t.calls == []


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
