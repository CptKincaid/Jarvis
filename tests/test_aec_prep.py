"""The echo-cancellation groundwork under scripts/audio/, kept honest.

WHY. Live on 2026-09-01 the first AEC design (module in pipewire.conf.d,
installer restarting pipewire + wireplumber) was found to be hazardous on
this box -- a pipewire restart drops the Bluetooth soundbar -- and its
monitor.mode reference measured -180 dBFS on the bluez sink.  The replacement
routes the reference INTO a virtual sink and lives in filter-chain.conf.d, so
only the filter-chain unit is ever restarted.  These tests pin those two
lessons to the files so a "simplification" cannot quietly bring the old
design back, and check the measurement tool's maths on signals whose answer
is known.  Nothing here installs anything, records anything or plays
anything: spa-json-dump parses a file, bash -n parses a script.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
AUDIO = REPO / "scripts" / "audio"
CONF = AUDIO / "99-jarvis-echo-cancel.conf"
INSTALL = AUDIO / "aec-install.sh"
UNINSTALL = AUDIO / "aec-uninstall.sh"
MEASURE = AUDIO / "aec_measure.py"
DOC = REPO / "docs" / "echo-cancellation.md"

SNOWBALL = "alsa_input.usb-BLUE_MICROPHONE_Blue_Snowball_201506-00.analog-stereo"


def _measure():
    spec = importlib.util.spec_from_file_location("aec_measure", MEASURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _conf_json() -> dict:
    """The conf as PipeWire itself would read it.  spa-json-dump is the
    reference SPA-JSON parser (pipewire-bin); no ad-hoc parser here."""
    out = subprocess.run(["spa-json-dump", str(CONF)], capture_output=True,
                         text=True, timeout=10, check=True).stdout
    return json.loads(out)


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _commands(text: str) -> list[str]:
    """Shell lines that DO something: comments gone, and the echo lines that
    tell the operator what to do next are advice, not actions."""
    out = []
    for line in _strip_comments(text).splitlines():
        s = line.strip()
        if s and not s.startswith("echo "):
            out.append(s)
    return out


# ------------------------------------------------------------------ the conf

def test_conf_braces_balance():
    # spa-json-dump auto-closes an unterminated object, so a truncated conf
    # would still "parse"; this is the check it does not do.
    body = _strip_comments(CONF.read_text())
    assert body.count("{") == body.count("}")
    assert body.count("[") == body.count("]")


@pytest.mark.skipif(not shutil.which("spa-json-dump"), reason="pipewire-bin not installed")
def test_conf_parses_as_pipewire_config_with_the_sink_topology():
    conf = _conf_json()
    mods = conf["context.modules"]
    assert len(mods) == 1
    assert mods[0]["name"] == "libpipewire-module-echo-cancel"
    args = mods[0]["args"]
    assert args["library.name"] == "aec/libspa-aec-webrtc"
    # The reference is ROUTED IN through a sink; monitor.mode was measured
    # silent on the bluez A2DP sink and must not come back.
    assert "monitor.mode" not in args
    assert args["sink.props"]["node.name"] == "jarvis_aec_sink"
    assert args["source.props"]["node.name"] == "jarvis_aec_source"
    assert args["playback.props"]["node.name"] == "jarvis_aec_playback"


@pytest.mark.skipif(not shutil.which("spa-json-dump"), reason="pipewire-bin not installed")
def test_conf_pins_the_snowball_and_keeps_the_sink_off_default():
    args = _conf_json()["context.modules"][0]["args"]
    cap = args["capture.props"]
    # The default source becomes jarvis_aec_source; a capture that followed
    # the default would feed the canceller its own output.
    assert cap["target.object"] == SNOWBALL
    assert cap["node.dont-reconnect"] is True
    # A default AEC sink would route the desktop -- and the sink's own
    # playback -- back into the canceller; the soundbar sits far above 100.
    assert args["sink.props"]["priority.session"] == 100
    aec = args["aec.args"]
    # Both reshape the spectrum the ECAPA thresholds were measured on.
    assert aec["webrtc.noise_suppression"] is False
    assert aec["webrtc.gain_control"] is False


# --------------------------------------------------------------- the scripts

@pytest.mark.parametrize("script", [INSTALL, UNINSTALL])
def test_scripts_parse_and_are_executable(script):
    subprocess.run(["bash", "-n", str(script)], check=True, timeout=10)
    assert script.stat().st_mode & 0o111, f"{script.name} is not executable"


@pytest.mark.parametrize("script", [INSTALL, UNINSTALL])
def test_scripts_only_ever_restart_filter_chain(script):
    """The hazard the redesign exists for: a pipewire restart drops the
    Bluetooth soundbar.  Every systemctl verb in both scripts must name
    filter-chain and nothing else."""
    calls = [line for line in _commands(script.read_text()) if "systemctl" in line]
    assert calls, "expected a filter-chain reload"
    for call in calls:
        assert "filter-chain" in call, call
        for unit in ("pipewire", "wireplumber"):
            assert unit not in call.replace("filter-chain", ""), call


def test_install_writes_filter_chain_conf_d_not_pipewire_conf_d():
    body = _strip_comments(INSTALL.read_text())
    assert "filter-chain.conf.d" in body
    # pipewire.conf.d may only appear as the LEGACY path the installer
    # refuses to coexist with, never as a destination.
    for line in body.splitlines():
        if "pipewire.conf.d" in line:
            assert "LEGACY" in line, line


def test_scripts_read_the_mic_name_from_the_conf():
    """One copy of the Snowball's node name, in the conf; both scripts sed it
    out so the pin and the restore cannot drift apart."""
    for script in (INSTALL, UNINSTALL):
        body = _strip_comments(script.read_text())
        assert "target\\.object" in body, script.name
        assert SNOWBALL not in body, f"{script.name} duplicates the mic name"


def test_scripts_never_play_audio():
    for path in (INSTALL, UNINSTALL, MEASURE):
        body = "\n".join(_commands(path.read_text()))
        for player in ("paplay", "pw-play", "aplay", "speaker-test"):
            assert player not in body, f"{path.name} runs {player}"


# ------------------------------------------------------------- aec_measure.py

def test_measure_help_runs_without_touching_audio():
    r = subprocess.run([sys.executable, str(MEASURE), "--help"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "nothing here ever plays audio" in r.stdout


def test_attenuation_of_a_known_ratio():
    m = _measure()
    rate = 48000
    t = np.arange(rate * 4) / rate
    raw = 0.2 * np.sin(2 * np.pi * 440 * t)
    aec = raw / 4.0                       # 4x quieter = 12.04 dB
    res = m.attenuation(raw, aec, rate)
    assert res["atten_db"] == pytest.approx(12.04, abs=0.05)
    assert res["raw_db"] == pytest.approx(20 * np.log10(0.2 / np.sqrt(2)), abs=0.05)
    # 4 s minus the 0.8 s skip leaves three whole seconds.
    assert len(res["per_sec_db"]) == 3
    assert all(abs(x - 12.0) < 0.1 for x in res["per_sec_db"])
    assert all(isinstance(x, float) for x in res["per_sec_db"])


def test_attenuation_trims_to_the_shorter_file():
    m = _measure()
    rate = 16000
    raw = np.random.default_rng(1).normal(0, 0.1, rate * 5)
    res = m.attenuation(raw, raw[: rate * 3], rate)
    assert res["seconds"] == pytest.approx(3 - 0.8)
    assert res["atten_db"] == pytest.approx(0.0, abs=1e-9)


def test_lag_recovers_a_known_delay():
    m = _measure()
    rate = 16000
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 0.1, rate * 3)
    delay = int(0.137 * rate)             # 137 ms: the mic hears it later
    mic = np.concatenate([np.zeros(delay), ref[:-delay]]) + rng.normal(0, 0.01, len(ref))
    res = m.lag(mic, ref, rate)
    assert res["lag_ms"] == pytest.approx(137.0, abs=1.0)
    assert res["ncc"] > 0.5
    assert isinstance(res["lag_ms"], float)


def test_lag_of_unrelated_signals_has_no_peak():
    """The 2026-09-01 case: a silent reference.  The number printed must not
    look like a delay -- ncc is how the tool says 'these do not match'."""
    m = _measure()
    rate = 16000
    rng = np.random.default_rng(3)
    res = m.lag(rng.normal(0, 0.1, rate * 3), np.zeros(rate * 3), rate)
    assert res["ncc"] < 0.01


def test_record_uses_parecord_and_the_conf_mic():
    m = _measure()
    assert m.pinned_mic() == SNOWBALL
    argv = m.record_argv("jarvis_aec_source", Path("/x/aec.wav"))
    assert argv[0] == "parecord"
    assert "--device=jarvis_aec_source" in argv
    assert argv[-1] == "/x/aec.wav"


def test_sources_parses_the_short_roster():
    m = _measure()

    class R:
        stdout = ("55\talsa_input.usb-X.analog-stereo\tPipeWire\ts16le 2ch 48000Hz\tRUNNING\n"
                  "90\tjarvis_aec_source\tPipeWire\ts16le 1ch 48000Hz\tIDLE\n")

    assert m.sources(run=lambda *a, **k: R()) == {"alsa_input.usb-X.analog-stereo",
                                                  "jarvis_aec_source"}

    def boom(*a, **k):
        raise OSError("no pactl")

    assert m.sources(run=boom) == set()


# --------------------------------------------------------------------- doc

def test_doc_states_the_measured_facts_and_the_limits():
    text = DOC.read_text()
    # The two facts that decided the design, and the one that bounds it.
    assert "-180 dBFS" in text
    assert "filter-chain" in text and "pipewire.conf.d" in text
    assert "HPCOMPUTER" in text
    assert "UNMEASURED" in text
    # The interaction that must be settled before enabling.
    assert "jarvis_aec_playback" in text and "mixer" in text.lower()
    for name in ("aec-install.sh", "aec-uninstall.sh", "aec_measure.py", "playback_device"):
        assert name in text
    # The default-source switch only reaches Jarvis while voice_settings.json
    # says mic "Default"; a pinned "[N] name" mic would silently bypass the
    # canceller, so the doc has to say so next to the switch.
    assert '`"Default"`' in text
