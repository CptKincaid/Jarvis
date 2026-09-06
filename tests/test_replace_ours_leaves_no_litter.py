"""A replace that cannot finish takes its own temp away.

Named by the round-10 adversary as its first remaining item, not a block.
Round 10 made ``_replace_ours`` claim its temp at a UNIQUE name with
O_CREAT|O_EXCL -- correct, and it closed a measured destruction. The cost
that came with uniqueness: where the old fixed name left exactly one temp
behind on a failure, the unique name leaves a fresh one PER ATTEMPT.
MEASURED by that adversary in tmp_path: status.txt made a directory by his
hand -> 10 empty temps after 10 passes beside his folders; disk full
(ENOSPC on the write) -> 10 in ~/Desktop/Jarvis and 40 in the state dir;
modelled at the 30 s cadence, ~2,880 a day. Never a byte of his -- but
litter he can see, beside folders he uses, and the docstring only admitted
the process-dies case.

THE RULE: the temp is ours by the kernel's word (we created it O_EXCL), so
on any failure after the claim we may and must remove it. The one case it
cannot cover is the process dying between claim and rename, which the
docstring already states.
"""
import errno
import os

import pytest

from jarvis import foldersync as fs


def _temps(target):
    return sorted(p.name for p in target.parent.iterdir()
                  if p.name.startswith(target.name + ".") and p.suffix == ".tmp")


def test_a_target_that_is_a_directory_leaves_no_temp(tmp_path):
    """His hand: status.txt made a folder. Ten passes, zero litter."""
    target = tmp_path / "status.txt"
    target.mkdir()
    for _ in range(10):
        with pytest.raises(OSError):
            fs._replace_ours(target, "ours\n")
    assert _temps(target) == [], (
        "temps of ours left beside his folder after failed replaces: %s"
        % _temps(target))
    assert target.is_dir(), "his directory must be untouched"


def test_a_write_that_fails_leaves_no_temp(tmp_path, monkeypatch):
    """Disk full at the write. The claim succeeded; the temp must go."""
    target = tmp_path / "ledger.json"
    real_fdopen = os.fdopen

    class _Full:
        def __init__(self, fh):
            self._fh = fh

        def write(self, _text):
            raise OSError(errno.ENOSPC, "No space left on device")

        def flush(self):
            pass

        def fileno(self):
            return self._fh.fileno()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self._fh.close()
            return False

    monkeypatch.setattr(os, "fdopen",
                        lambda fd, *a, **k: _Full(real_fdopen(fd, *a, **k)))
    for _ in range(10):
        with pytest.raises(OSError):
            fs._replace_ours(target, "ours\n")
    assert _temps(target) == [], "temps of ours left after ENOSPC"
    assert not target.exists(), "nothing landed at the target"


def test_a_replace_that_fails_leaves_no_temp(tmp_path, monkeypatch):
    """The rename itself refused. Same rule."""
    target = tmp_path / "status.txt"
    target.write_text("before\n")

    def refuse(src, dst):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError):
        fs._replace_ours(target, "after\n")
    assert _temps(target) == []
    assert target.read_text() == "before\n", "the target is untouched"


def test_a_successful_replace_is_unchanged(tmp_path):
    target = tmp_path / "status.txt"
    target.write_text("before\n")
    fs._replace_ours(target, "after\n", fsync=True)
    assert target.read_text() == "after\n"
    assert _temps(target) == []
