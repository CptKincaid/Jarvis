"""Detecting whether a real microphone exists.

`MachineProfile.detect()` gates the hotword, `record_fixed`, and enrolment on
`has_mic`. It decided that by listing sounddevice inputs and ignoring anything
called default/pipewire/pulse -- correct as far as it went, because with NO mic
attached PipeWire still advertises a `default` input at 44100 Hz that carries
silence, and counting it would make voice look wired up when it is not.

The converse case broke it. While PipeWire holds a USB mic exclusively,
PortAudio cannot probe `hw:N` and enumerates ONLY those virtual devices --
capture through them works perfectly, but every name is filtered out, so
`has_mic` came back False with a live microphone plugged in. Observed on the
Blue Snowball after an unplug/replug: `arecord -l` and PipeWire both saw the
device, sounddevice showed only pipewire/default, and a restarted Jarvis would
have refused to start the wake word.

ALSA's own PCM table settles both directions, so it is the tiebreak.
"""
from __future__ import annotations

import sounddevice as sd

from jarvis import config as cfg

VIRTUAL_ONLY = [
    {"name": "pipewire", "max_input_channels": 64},
    {"name": "default", "max_input_channels": 64},
]

PCM_WITH_MIC = (
    "00-03: HDMI 0 : HDMI 0 : playback 1\n"
    "00-07: HDMI 1 : HDMI 1 : playback 1\n"
    "01-00: USB Audio : USB Audio : capture 1\n"
)
PCM_NO_MIC = (
    "00-03: HDMI 0 : HDMI 0 : playback 1\n"
    "00-07: HDMI 1 : HDMI 1 : playback 1\n"
)


def test_capture_line_is_detected(tmp_path):
    p = tmp_path / "pcm"
    p.write_text(PCM_WITH_MIC)
    assert cfg._alsa_has_capture(p) is True


def test_playback_only_is_not_a_microphone(tmp_path):
    p = tmp_path / "pcm"
    p.write_text(PCM_NO_MIC)
    assert cfg._alsa_has_capture(p) is False


def test_unreadable_proc_file_is_not_a_microphone(tmp_path):
    assert cfg._alsa_has_capture(tmp_path / "does-not-exist") is False


def test_live_mic_behind_pipewire_is_found(monkeypatch):
    """The regression: real hardware, but PortAudio only shows virtual names."""
    monkeypatch.setattr(sd, "query_devices", lambda: VIRTUAL_ONLY)
    monkeypatch.setattr(cfg, "_alsa_has_capture", lambda *a: True)

    m = cfg.MachineProfile.detect()

    assert m.mic_names == [], "virtual devices must still not be named as mics"
    assert m.has_mic is True, "a real capture PCM exists; voice must not be disabled"


def test_no_hardware_still_reports_no_mic(monkeypatch):
    """The case the original filter was written for must keep working."""
    monkeypatch.setattr(sd, "query_devices", lambda: VIRTUAL_ONLY)
    monkeypatch.setattr(cfg, "_alsa_has_capture", lambda *a: False)

    m = cfg.MachineProfile.detect()

    assert m.mic_names == []
    assert m.has_mic is False


def test_named_hardware_wins_without_consulting_alsa(monkeypatch):
    """A probeable device is proof enough; don't need /proc to agree."""
    monkeypatch.setattr(sd, "query_devices", lambda: VIRTUAL_ONLY + [
        {"name": "Snowball: USB Audio (hw:1,0)", "max_input_channels": 2},
    ])
    monkeypatch.setattr(cfg, "_alsa_has_capture",
                        lambda *a: pytest_fail_if_called())

    m = cfg.MachineProfile.detect()

    assert m.mic_names == ["Snowball: USB Audio (hw:1,0)"]
    assert m.has_mic is True


def pytest_fail_if_called():
    raise AssertionError("_alsa_has_capture should be short-circuited")
