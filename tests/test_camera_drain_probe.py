"""The drain probe (scripts/camera_drain_probe.py) -- the second script
allowed to open the camera without the app, under the same rules as the
first (tests/test_camera_mode_probe.py).

Hunter: *"i dont want you to look at anything the camera sees without my
explicit permission."* This probe answers "what does the queue drain cost
and what does it buy" in frame rates and drop counts. What is pinned here,
without a device:

* its CODE names no call that would turn a buffer into a picture -- not
  ``read``, not ``retrieve``, not ``imshow``/``imwrite``/``imencode``/
  ``imdecode`` -- checked two ways: the script's own self-check, and a
  token-level grep from outside that does not trust the self-check;
* driven against a fake cv2, the only device methods it ever calls are
  ``isOpened``/``set``/``get``/``grab``/``release`` -- the fake's ``read``
  and ``retrieve`` raise if reached;
* IT REFUSES TO TAKE THE CAMERA OFF A RUNNING JARVIS. If anything else
  holds the node it stops and names the pid, and never opens the device;
* it measures the REAL policy: the arm it times is
  ``jarvis.camera.DrainingCapture.drain``, so a change to the drain changes
  what the probe reports, and the drop counts it prints are the same ones
  the app's own numbers line reports;
* every line it prints is a name, a rate, a count or a millisecond.
"""
from __future__ import annotations

import importlib.util
import io
import os
import re
import sys
import time
import tokenize

import pytest

from jarvis import camera as cam

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_HERE, "scripts", "camera_drain_probe.py")
_SPEC = importlib.util.spec_from_file_location("camera_drain_probe", _PATH)
probe = importlib.util.module_from_spec(_SPEC)
sys.modules["camera_drain_probe"] = probe
_SPEC.loader.exec_module(probe)


def _code_only(path: str) -> str:
    """The file with every comment and string literal removed, so the grep
    below looks at CODE -- the docstring discusses the forbidden calls."""
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(io.BytesIO(fh.read()).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


# ------------------------------------------------------------- the rule
def test_the_probe_names_no_call_that_could_show_or_save_a_frame():
    source = _code_only(_PATH)
    for pattern in (r"\bread\s*\(", r"\bretrieve\b", r"\bimshow\b",
                    r"\bimwrite\b", r"\bimencode\b", r"\bimdecode\b",
                    r"\bnamedWindow\b", r"\bwaitKey\b", r"\btofile\b",
                    r"\.save\s*\("):
        assert not re.search(pattern, source), pattern
    assert probe.self_check() == []


def test_the_self_check_catches_a_planted_retrieve(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("cap.grab()\nok, f = cap.retrieve()\n")
    hits = probe.self_check(str(bad))
    assert hits and hits[0].startswith("retrieve line 2")
    fine = tmp_path / "fine.py"
    fine.write_text("cap.grab()\nx = 'retrieve()'   # retrieve\n")
    assert probe.self_check(str(fine)) == []


# ----------------------------------------------------------- a fake cv2
class _Cap:
    """cv2.VideoCapture's surface. ``read``/``retrieve`` are TRAPS."""

    def __init__(self, opened=True, grab_ok=True):
        self._opened = opened
        self._grab_ok = grab_ok
        self.calls = []
        self.props = {}

    def isOpened(self):
        self.calls.append("isOpened")
        return self._opened

    def set(self, prop, value):
        self.calls.append("set")
        self.props[prop] = float(value)
        return True

    def get(self, prop):
        self.calls.append("get")
        return self.props.get(prop, 0.0)

    def grab(self):
        self.calls.append("grab")
        return self._grab_ok

    def release(self):
        self.calls.append("release")

    def read(self):
        raise AssertionError("the probe called read()")

    def retrieve(self, *_a):
        raise AssertionError("the probe called retrieve()")


class _Cv2:
    CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT = 3, 4
    CAP_PROP_FPS, CAP_PROP_FOURCC = 5, 6
    CAP_PROP_CONVERT_RGB, CAP_PROP_BUFFERSIZE = 16, 38
    __version__ = "0.fake"

    def __init__(self, opened=True, grab_ok=True):
        self.caps = []
        self._opened, self._grab_ok = opened, grab_ok

    @staticmethod
    def VideoWriter_fourcc(*chars):
        return sum(ord(c) << (8 * i) for i, c in enumerate(chars))

    def VideoCapture(self, _target):
        cap = _Cap(self._opened, self._grab_ok)
        self.caps.append(cap)
        return cap


def _no_holders(monkeypatch):
    monkeypatch.setattr(probe, "holders", lambda *_a, **_k: [])


# ------------------------------------------------- it never takes the camera
def test_a_node_somebody_else_holds_stops_it_before_the_device_is_opened(
        monkeypatch, capsys):
    """The house rule: if the camera is busy, say so and stop. A number is
    not worth blinding the running assistant."""
    cv2 = _Cv2()
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(probe, "holders", lambda *_a, **_k: [1747134])
    code = probe.main(["--seconds", "0.01"])
    assert code == 3
    out = capsys.readouterr().out
    assert "STOPPED" in out and "1747134" in out
    assert cv2.caps == []                    # nothing was ever opened


def test_holders_reads_proc_and_shrugs_at_what_it_cannot_see(monkeypatch,
                                                             tmp_path):
    """Best effort by design: a process belonging to another user is not
    visible in /proc to us, so the answer is "none I can see", never a
    guarantee -- and an unreadable entry must not raise."""
    (tmp_path / "42" / "fd").mkdir(parents=True)
    os.symlink("/dev/video0", tmp_path / "42" / "fd" / "8")
    (tmp_path / "77" / "fd").mkdir(parents=True)
    os.symlink("/dev/null", tmp_path / "77" / "fd" / "3")
    real_listdir, real_readlink = os.listdir, os.readlink

    def listdir(path):
        if path == "/proc":
            return ["42", "77", "99", "not-a-pid"]
        if path.startswith("/proc/"):
            pid = path.split("/")[2]
            if pid == "99":
                raise PermissionError("not ours")
            return real_listdir(str(tmp_path / pid / "fd"))
        return real_listdir(path)

    def readlink(path):
        if path.startswith("/proc/"):
            _, _, pid, _fd, num = path.split("/")
            return real_readlink(str(tmp_path / pid / "fd" / num))
        return real_readlink(path)

    monkeypatch.setattr(os, "listdir", listdir)
    monkeypatch.setattr(os, "readlink", readlink)
    assert probe.holders("/dev/video0") == [42]


# -------------------------------------------------- it only ever grabs
def test_a_full_run_against_a_fake_only_grabs_and_prints_numbers(monkeypatch,
                                                                 capsys):
    cv2 = _Cv2()
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    _no_holders(monkeypatch)
    code = probe.main(["--seconds", "0.02"])
    assert code == 0
    assert len(cv2.caps) == 1                # ONE handle, one consumer
    assert set(cv2.caps[0].calls) <= {"isOpened", "set", "get", "grab",
                                      "release"}
    assert cv2.caps[0].calls[-1] == "release"
    out = capsys.readouterr().out
    assert "drained" in out and "undrained" in out and "drop" in out
    for row in out.splitlines():
        assert len(row) < 200, row
        assert "array(" not in row and "[[" not in row


class _TimedCap(_Cap):
    """A device shaped like the real one: a four-deep queue filled every
    ``interval`` seconds, and a grab that WAITS when the queue is empty.

    Real time, real sleeps, so what the probe measures against it is the
    real policy's real timing. The "frame" is an integer ordinal and is
    never retrieved -- ``retrieve`` is still a trap.
    """

    def __init__(self, interval=0.01, depth=4):
        _Cap.__init__(self)
        self.interval = float(interval)
        self.depth = int(depth)
        self.next_at = time.perf_counter() + self.interval
        self.queue = []
        self.produced = 0

    def _fill(self):
        while time.perf_counter() >= self.next_at:
            self.produced += 1
            self.queue.append(self.produced)
            if len(self.queue) > self.depth:
                self.queue.pop(0)
            self.next_at += self.interval

    def grab(self):
        self.calls.append("grab")
        self._fill()
        if not self.queue:
            time.sleep(max(0.0, self.next_at - time.perf_counter()))
            self._fill()
        if not self.queue:
            return False
        self.queue.pop(0)
        return True


def test_the_arm_it_times_is_the_apps_own_drain():
    """Not a re-implementation. The probe drives
    ``jarvis.camera.DrainingCapture``, so a change to the policy changes
    what the probe reports -- which is the only reason its numbers are
    evidence about the app at all. A consumer keeping pace with the device
    drains NOTHING, which is the claim, and it is the class under test that
    decides that, not the probe."""
    cap = _TimedCap(interval=0.01)
    arm = probe.run_arm(cam.DrainingCapture, cap, 0.0, 0.30,
                        cam.DRAIN_MAX_DROPS, nominal=100.0)
    assert arm["ok"] is True
    assert arm["cycles"] >= 10
    # Only the OPEN's own read can discard anything here: the drain starts
    # believing the queue is full, which is where it measures the device's
    # interval, and the budget stops that read as soon as one grab waits.
    # Every read after it finds an empty queue and takes one buffer.
    assert arm["dropped"] <= 3
    assert arm["bounded"] <= 1
    # ...and it measured the device's own 10 ms interval to do it.
    assert arm["interval_ms"] == pytest.approx(10.0, rel=0.5)
    assert arm["fps"] == pytest.approx(100.0, rel=0.25)
    assert "retrieve" not in set(cap.calls)


def test_a_consumer_behind_the_device_drops_what_it_is_behind_by():
    """And the other half. A consumer at a fifth of the device's rate has
    four stale buffers waiting every cycle; the probe reports them as a
    drop RATE, which is the number the app's own log line carries."""
    cap = _TimedCap(interval=0.01)
    arm = probe.run_arm(cam.DrainingCapture, cap, 20.0, 0.50,
                        cam.DRAIN_MAX_DROPS, nominal=100.0)
    assert arm["ok"] is True
    assert arm["fps"] == pytest.approx(20.0, rel=0.35)
    assert arm["dropped"] >= arm["cycles"]     # at least one a cycle
    # delivered + dropped is the device's own rate, not the consumer's
    assert arm["frames_ps"] > arm["fps"] * 2.0
    assert "retrieve" not in set(cap.calls)


def test_the_undrained_arm_really_drops_nothing():
    cap = _TimedCap(interval=0.01)
    arm = probe.run_arm(cam.DrainingCapture, cap, 20.0, 0.20, 0)
    assert arm["ok"] is True
    assert arm["dropped"] == 0
    assert cap.calls.count("grab") == arm["cycles"]


def test_a_device_that_will_not_grab_is_reported_not_retried(monkeypatch,
                                                             capsys):
    cv2 = _Cv2(grab_ok=False)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    _no_holders(monkeypatch)
    code = probe.main(["--seconds", "0.02"])
    assert code == 1
    out = capsys.readouterr().out
    assert "STOPPED" in out and "Not retrying" in out
    assert cv2.caps[0].calls[-1] == "release"


def test_a_device_that_will_not_open_stops_the_whole_run(monkeypatch, capsys):
    cv2 = _Cv2(opened=False)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    _no_holders(monkeypatch)
    code = probe.main(["--seconds", "0.01"])
    assert code == 3
    assert "STOPPED" in capsys.readouterr().out
    assert len(cv2.caps) == 1                # one attempt, no loop


def test_it_stops_on_a_missing_node(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cv2", _Cv2())
    _no_holders(monkeypatch)
    code = probe.main(["--device", "/dev/video-does-not-exist"])
    assert code == 3
    assert "does not exist" in capsys.readouterr().out


@pytest.mark.parametrize("fps", probe.CONSUMER_FPS)
def test_every_consumer_rate_it_claims_to_test_is_a_rate(fps):
    assert fps >= 0.0
