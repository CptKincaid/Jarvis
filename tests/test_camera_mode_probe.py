"""The mode probe (scripts/camera_mode_probe.py) -- the ONE script that may
open the camera without the app, and the rules it runs under.

Hunter: *"i dont want you to look at anything the camera sees without my
explicit permission."* The probe exists to answer a question the running app
could not (what mode does the driver actually grant, and how fast does the
device deliver), and it answers in numbers. What is pinned here, without a
device:

* its CODE names no call that would turn a buffer into a picture -- not
  ``read``, not ``retrieve``, not ``imshow``/``imwrite``/``imencode``/
  ``imdecode`` -- checked two ways: the script's own self-check, and a
  token-level grep from outside that does not trust the self-check;
* the self-check really does catch a ``read()`` when one is planted;
* driven against a fake cv2, the only device methods it ever calls are
  ``isOpened``/``set``/``get``/``grab``/``getBackendName``/``release`` --
  the fake's ``read`` and ``retrieve`` raise if reached;
* a device that will not open stops it with a stated reason, once;
* every line it prints is a name, a size, a rate or a millisecond.
"""
from __future__ import annotations

import importlib.util
import io
import os
import re
import sys
import tokenize

import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_HERE, "scripts", "camera_mode_probe.py")
_SPEC = importlib.util.spec_from_file_location("camera_mode_probe", _PATH)
probe = importlib.util.module_from_spec(_SPEC)
sys.modules["camera_mode_probe"] = probe
_SPEC.loader.exec_module(probe)


def _code_only(path: str) -> str:
    """The file with every comment and string literal removed, so the grep
    below looks at CODE -- the docstring discusses ``read`` at length."""
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(io.BytesIO(fh.read()).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


# ------------------------------------------------------------ the rule
def test_the_probe_names_no_call_that_could_show_or_save_a_frame():
    source = _code_only(_PATH)
    for pattern in (r"\bread\s*\(", r"\bretrieve\b", r"\bimshow\b",
                    r"\bimwrite\b", r"\bimencode\b", r"\bimdecode\b",
                    r"\bnamedWindow\b", r"\bwaitKey\b", r"\btofile\b",
                    r"\.save\s*\(", r"\bnumpy\b", r"\bPIL\b"):
        assert not re.search(pattern, source), pattern
    assert probe.self_check() == []


def test_the_self_check_catches_a_planted_read(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import cv2\ncap = cv2.VideoCapture(0)\n"
                   "ok, frame = cap.read()\n")
    hits = probe.self_check(str(bad))
    assert hits and hits[0].startswith("read line 3")
    fine = tmp_path / "fine.py"
    fine.write_text("cap.grab()\nx = 'read()'   # read\n")
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

    def getBackendName(self):
        self.calls.append("getBackendName")
        return "FAKE"

    def release(self):
        self.calls.append("release")

    def read(self):
        raise AssertionError("the probe called read()")

    def retrieve(self, *_a):
        raise AssertionError("the probe called retrieve()")


class _Cv2:
    CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT = 3, 4
    CAP_PROP_FPS, CAP_PROP_FOURCC = 5, 6
    CAP_PROP_EXPOSURE, CAP_PROP_AUTO_EXPOSURE = 15, 21
    CAP_PROP_BUFFERSIZE = 38
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


def test_the_probe_only_grabs_and_never_retrieves():
    cv2 = _Cv2()
    result = probe.probe_one(cv2, 0, 1280, 720, "MJPG", "fourcc-first",
                             warmup=3, grabs=7, buffersize=True)
    cap = cv2.caps[0]
    assert set(cap.calls) <= {"isOpened", "set", "get", "grab",
                              "getBackendName", "release"}
    assert cap.calls.count("grab") == 10
    assert cap.calls[-1] == "release"
    assert result["opened"] is True
    assert result["timing"]["ok"] is True
    assert result["timing"]["grabs"] == 7
    assert result["granted"]["fourcc"] == "MJPG"
    assert result["granted"]["width"] == 1280.0
    assert cap.props[cv2.CAP_PROP_BUFFERSIZE] == 1.0
    assert cap.props[cv2.CAP_PROP_FPS] == probe.REQUEST_FPS


def test_both_set_orders_are_really_different_orders():
    for order, first in (("fourcc-first", probe.MODES[0][2]),
                         ("size-first", None)):
        cv2 = _Cv2()
        cap = cv2.VideoCapture(0)
        probe.request(cv2, cap, 640, 480, "YUYV", order)
        sets = [p for p, _ in
                [(c, None) for c in cap.calls if c == "set"]]
        assert len(sets) == 4                     # fourcc, w, h, fps
    cv2 = _Cv2()
    cap = cv2.VideoCapture(0)
    seen = []
    cap.set = lambda prop, value: seen.append(prop) or True
    probe.request(cv2, cap, 640, 480, "YUYV", "size-first")
    assert seen[:2] == [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]
    assert seen[2] == cv2.CAP_PROP_FOURCC
    seen.clear()
    probe.request(cv2, cap, 640, 480, "YUYV", "fourcc-first")
    assert seen[0] == cv2.CAP_PROP_FOURCC


def test_a_device_that_will_not_grab_is_reported_not_retried():
    cap = _Cap(grab_ok=False)
    timing = probe.time_grabs(cap, warmup=10, grabs=60)
    assert timing["ok"] is False
    assert cap.calls.count("grab") == 5           # five failures, then stop


def test_a_device_that_will_not_open_stops_the_whole_run(monkeypatch, capsys):
    cv2 = _Cv2(opened=False)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    code = probe.main(["--device", "0", "--grabs", "2", "--warmup", "1"])
    assert code == 3
    out = capsys.readouterr().out
    assert "STOPPED" in out and "Not retrying" in out
    assert len(cv2.caps) == 1                     # one attempt, no loop
    assert "grab" not in cv2.caps[0].calls


def test_a_full_run_against_a_fake_prints_numbers_only(monkeypatch, capsys):
    cv2 = _Cv2()
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    code = probe.main(["--grabs", "3", "--warmup", "1"])
    assert code == 0
    out = capsys.readouterr().out
    assert len(cv2.caps) == len(probe.MODES) * len(probe.ORDERS)
    for cap in cv2.caps:
        assert set(cap.calls) <= {"isOpened", "set", "get", "grab",
                                  "getBackendName", "release"}
    assert "granted" in out and "fps" in out
    # Every printed token is a word, a size, a rate or a millisecond --
    # there is no base64, no array repr, nothing longer than a sentence.
    for line in out.splitlines():
        assert len(line) < 160, line
        assert "array(" not in line and "[[" not in line


def test_the_probe_stops_on_a_missing_node(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "cv2", _Cv2())
    code = probe.main(["--device", "/dev/video-does-not-exist"])
    assert code == 3
    assert "does not exist" in capsys.readouterr().out


@pytest.mark.parametrize("value,name", [(0, ""), ("x", ""),
                                        (1196444237, "MJPG")])
def test_fourcc_names(value, name):
    assert probe.fourcc_name(value) == name
