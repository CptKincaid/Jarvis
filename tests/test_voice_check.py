"""Tests for jarvis.voice_check — parsers are pure; probes run offline."""
import json

from jarvis import voice_check
from jarvis.voice_check import Check, parse_devices, parse_sinks

# Verbatim from this machine (DGX Spark, no HDMI audio attached).
PACTL_DUMMY = "496\tauto_null\tPipeWire\tfloat32le 2ch 48000Hz\tSUSPENDED\n"
PACTL_HDMI = ("52\talsa_output.platform-NVDA2014_00.hdmi-stereo\tPipeWire\t"
              "s32le 2ch 48000Hz\tRUNNING\n" + PACTL_DUMMY)
DEVICES_NO_MIC = [
    {"name": "NVIDIA: HDMI 0 (hw:0,3)", "max_input_channels": 0},
    {"name": "pipewire", "max_input_channels": 64},
    {"name": "default", "max_input_channels": 64, "default_samplerate": 48000},
]
DEVICES_USB = DEVICES_NO_MIC + [
    {"name": "USB PnP Sound Device: Audio (hw:1,0)", "max_input_channels": 1,
     "default_samplerate": 44100}]


def test_parse_sinks_dummy_only():
    st = parse_sinks(PACTL_DUMMY)
    assert st == {"sinks": ["auto_null"], "dummy": True,
                  "suspended": ["auto_null"]}


def test_parse_sinks_real_sink_present():
    st = parse_sinks(PACTL_HDMI)
    assert st["dummy"] is False
    assert st["sinks"][0].startswith("alsa_output")


def test_parse_sinks_empty():
    assert parse_sinks("") == {"sinks": [], "dummy": True, "suspended": []}


def test_parse_devices_filters_virtual_inputs():
    assert parse_devices(DEVICES_NO_MIC) == []
    devs = parse_devices(DEVICES_USB)
    assert len(devs) == 1 and devs[0]["index"] == 3 and devs[0]["rate"] == 44100
    assert parse_devices(None) == []


def test_output_sink_state_uses_pactl():
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)

        class R:
            returncode = 0
            stdout = PACTL_DUMMY
        return R()
    st = voice_check.output_sink_state(run=run)
    assert calls == [["pactl", "list", "short", "sinks"]]
    assert st["dummy"] is True and st["probed"] is True


def test_output_sink_state_without_pactl():
    def run(cmd, **kw):
        raise FileNotFoundError("pactl")
    st = voice_check.output_sink_state(run=run)
    assert st["probed"] is False


def test_model_assets_keys():
    assets = voice_check.model_assets()
    for key in ("whisper_weights", "xtts_weights", "voice_ref", "oww_models",
                "hey_jarvis_model", "wakeword_verifier", "speaker_model",
                "voiceprint", "settings"):
        assert key in assets
    assert isinstance(assets["oww_models"], list)


def test_run_offline_checks_shape():
    checks = voice_check.run_offline_checks(load_models=False)
    names = [c.name for c in checks]
    assert "audio output sink" in names and "microphone" in names
    assert all(isinstance(c, Check) for c in checks)
    sink = next(c for c in checks if c.name == "audio output sink")
    assert (sink.fix != "") == (not sink.ok)


def test_cli_json(capsys):
    rc = voice_check.main(["--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, list) and data and "name" in data[0]
    assert rc in (0, 1)
