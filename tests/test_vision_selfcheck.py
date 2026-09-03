"""The bring-up self-check (scripts/vision_selfcheck.py).

The script itself needs a camera; these tests do not, so what is pinned here
is the part that can be wrong without a camera being present:

* the v4l2-ctl parser, including the case that matters most on this box --
  v4l2-utils is NOT installed, and a missing tool must cost that one section
  and nothing else;
* that ``--models-only`` reaches a verdict without touching a device;
* that everything the script would print as JSON survives
  ``assert_numbers_only``, which is the mechanical form of "no image data
  leaves this script".
"""
from __future__ import annotations

import importlib.util
import os
import subprocess

import pytest

from jarvis.visionrig import assert_numbers_only

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "vision_selfcheck", os.path.join(_HERE, "scripts", "vision_selfcheck.py"))
selfcheck = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(selfcheck)


SAMPLE = """ioctl: VIDIOC_ENUM_FMT
\tType: Video Capture

\t[0]: 'YUYV' (YUYV 4:2:2)
\t\tSize: Discrete 640x480
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\t\tInterval: Discrete 0.067s (15.000 fps)
\t\tSize: Discrete 1280x720
\t\t\tInterval: Discrete 0.100s (10.000 fps)
\t[1]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Discrete 1280x720
\t\t\tInterval: Discrete 0.033s (30.000 fps)
"""


class Ran:
    def __init__(self, out="", err="", code=0):
        self.stdout, self.stderr, self.returncode = out, err, code


def test_the_format_list_becomes_sizes_and_frame_rates(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: Ran(out=SAMPLE))
    got = selfcheck.v4l2_formats("/dev/video0")
    assert got["available"] is True
    sizes = [(m["fourcc"], m["size"], m["fps"]) for m in got["modes"]]
    assert ("YUYV", "640x480", [30.0, 15.0]) in sizes
    assert ("MJPG", "1280x720", [30.0]) in sizes
    assert ("YUYV", "1280x720", [10.0]) in sizes
    assert_numbers_only(got)


def test_a_missing_v4l2_ctl_costs_one_section_and_nothing_else(monkeypatch):
    """v4l2-utils is not installed on this box and there is no sudo to add
    it. That must not take the whole check down."""
    def boom(*a, **k):
        raise FileNotFoundError("v4l2-ctl")

    monkeypatch.setattr(subprocess, "run", boom)
    got = selfcheck.v4l2_formats("/dev/video0")
    assert got["available"] is False and got["modes"] == []
    assert "v4l2-ctl" in got["reason"]


def test_a_failing_v4l2_ctl_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: Ran(err="No such device", code=1))
    got = selfcheck.v4l2_formats("/dev/video0")
    assert got["available"] is False and "No such device" in got["reason"]


def test_a_timeout_is_reported_not_raised(monkeypatch):
    def slow(*a, **k):
        raise subprocess.TimeoutExpired("v4l2-ctl", 5.0)

    monkeypatch.setattr(subprocess, "run", slow)
    assert selfcheck.v4l2_formats("/dev/video0")["available"] is False


def test_models_only_reaches_a_verdict_without_opening_a_device(monkeypatch):
    """conftest points JARVIS_FACE_MODEL_DIR at a throwaway directory, so the
    honest answer here is "not ready" -- and the point is that it is reached
    without a camera and without an exception."""
    opened = []
    monkeypatch.setattr(selfcheck.cam, "device_nodes",
                        lambda: opened.append("enumerated") or [])
    monkeypatch.setattr(selfcheck.cam, "open_capture",
                        lambda *a, **k: pytest.fail("opened a device"))
    code = selfcheck.main(["--models-only"])
    assert code == 1
    assert opened == []


def test_the_json_payload_is_numbers_only(monkeypatch, capsys):
    monkeypatch.setattr(selfcheck.cam, "device_nodes", lambda: [])
    monkeypatch.setattr(selfcheck.cam, "open_capture",
                        lambda *a, **k: pytest.fail("opened a device"))
    code = selfcheck.main(["--models-only", "--json"])
    assert code == 1
    payload = capsys.readouterr().out
    import json
    assert_numbers_only(json.loads(payload))


def test_synthetic_mode_opens_no_device_and_still_reports(monkeypatch, capsys):
    """The mode that works on a box with no camera: real weights (when they
    are there), generated frames, and cost numbers. conftest hides the real
    weights from the suite, so what is asserted here is the shape of the
    answer and, above all, that nothing was opened."""
    monkeypatch.setattr(selfcheck.cam, "device_nodes", lambda: [])
    monkeypatch.setattr(selfcheck.cam, "open_capture",
                        lambda *a, **k: pytest.fail("opened a device"))
    code = selfcheck.main(["--synthetic", "--frames", "3"])
    out = capsys.readouterr().out
    assert "No device is opened" in out
    assert code == 1                     # no weights in the test model dir
    assert "face_detection_yunet" in out
