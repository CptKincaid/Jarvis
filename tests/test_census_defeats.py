"""The three ways an adversary walked past round 4's census, PLANTED.

A census that reports a number is not the point.  A census that FAILS when
a new read-then-write appears is.  These three tests plant the adversary's
own three defeats into a MUTATED COPY of this lane's source -- in memory,
via ``census(sources=...)``, so nothing is written to the working tree --
and assert the census now sees each one.

All three were measured GREEN on the round-4 tip, with the row count
unchanged at 63.  Defeat 1 destroyed a 100000-byte file of his at
ok.txt.log (100000 bytes to 0) while doing it.

Nothing here opens a socket, touches ~/Desktop, or speaks to HPCOMPUTER.
"""
from pathlib import Path

from tests.write_census import MODULES, census

REPO = Path(__file__).resolve().parent.parent
FOLDERSYNC = "jarvis/foldersync.py"


def _mutated(old: str, new: str, append: str = "") -> dict:
    """foldersync.py with one edit, as TEXT.  The file on disk is untouched."""
    src = (REPO / FOLDERSYNC).read_text()
    if old:
        assert old in src, "the anchor moved; re-read the source"
        src = src.replace(old, new, 1)
    return {FOLDERSYNC: src + append}


def _rows(sources):
    return census(paths=MODULES, sources=sources)


def test_defeat_1_a_second_write_in_a_function_that_already_has_one():
    """The serious half of defeat 1: Syncer.record already has a write_text
    row, and round 4 kept ONE row per primitive per function, so adding a
    second one -- an ask-then-write on a name he chooses -- changed nothing
    the census could see.  MEASURED destroying his 100000-byte ok.txt.log.
    """
    src = _mutated(
        '''        except OSError:
            log.debug("foldersync: cannot write the record", exc_info=True)''',
        '''            sidecar = self.paths.outbox / (event.name + ".log")
            if sidecar.exists():                       # ASK
                sidecar.write_text("", encoding="utf-8")    # then WRITE
        except OSError:
            log.debug("foldersync: cannot write the record", exc_info=True)''')

    planted = [r for r in _rows(src)
               if r[1] == "Syncer.record" and r[2] == "write_text"]
    assert len(planted) == 2, (
        "the second write_text in Syncer.record is invisible; defeat 1 is "
        f"open again.  Rows found: {planted}")
    assert planted[1][3] == 2, planted

    # And the pin must actually reject it, not merely list it.
    from tests.test_write_census import KNOWN
    assert (FOLDERSYNC, "Syncer.record", "write_text", 2) not in KNOWN


def test_defeat_2_a_method_the_census_had_no_word_for():
    """rmtree, touch, and a run_ssh `del` on his Windows Inbox.  Round 4
    censused this as ZERO rows.  run_ssh is the one primitive in this lane
    that runs an arbitrary command on his machine."""
    src = _mutated(
        "    def note(self, target: Path, reason: str, extra: str = \"\") -> None:",
        '''    def _tidy_inbox(self) -> None:
        import shutil
        shutil.rmtree(self.paths.inbox / "old", ignore_errors=True)
        (self.paths.inbox / "marker").touch()
        remote.run_ssh(self.rconf, "del C:/Users/h2pey/Desktop/Jarvis/Inbox/*")

    def note(self, target: Path, reason: str, extra: str = "") -> None:''')

    planted = sorted(r[2] for r in _rows(src) if r[1] == "Syncer._tidy_inbox")
    assert planted == ["rmtree", "run_ssh", "touch"], (
        "a method that deletes his Windows Inbox censused as "
        f"{planted or 'NOTHING'}; defeat 2 is open again")


def test_defeat_3_a_write_that_is_not_inside_a_function():
    """Module level.  It runs at import and round 4's walk started at
    FunctionDef, so it did not exist."""
    src = _mutated("", "", append='''

_BOOT = Path("/tmp/jarvis-foldersync-boot")
if os.environ.get("JARVIS_FOLDERSYNC_BOOT"):
    _BOOT.write_text("started\\n", encoding="utf-8")
''')

    planted = [r for r in _rows(src)
               if r[1] == "<module>" and r[2] == "write_text"]
    assert planted, ("a module-level write censused as nothing; defeat 3 is "
                     "open again")


def test_the_three_defeats_each_move_the_count():
    """The number on the printout is the thing a reader trusts, and all
    three defeats left it at 63.  Every one of them must move it now."""
    base = len(census())
    for name, src in (
        ("defeat 1", _mutated(
            '''        except OSError:
            log.debug("foldersync: cannot write the record", exc_info=True)''',
            '''            sidecar = self.paths.outbox / (event.name + ".log")
            if sidecar.exists():
                sidecar.write_text("", encoding="utf-8")
        except OSError:
            log.debug("foldersync: cannot write the record", exc_info=True)''')),
        ("defeat 2", _mutated(
            "    def note(self, target: Path, reason: str, extra: str = \"\") -> None:",
            '''    def _tidy_inbox(self) -> None:
        import shutil
        shutil.rmtree(self.paths.inbox / "old", ignore_errors=True)
        (self.paths.inbox / "marker").touch()
        remote.run_ssh(self.rconf, "del x")

    def note(self, target: Path, reason: str, extra: str = "") -> None:''')),
        ("defeat 3", _mutated("", "", append='''

_B = Path("/tmp/x")
_B.write_text("boot", encoding="utf-8")
''')),
    ):
        assert len(_rows(src)) > base, f"{name} did not move the count"
