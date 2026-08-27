"""Voice-path diagnostics for Jarvis: what works on this machine right now.

    ~/vss_env/bin/python -m jarvis.voice_check            # offline checks
    ~/vss_env/bin/python -m jarvis.voice_check --load     # + load models
    ~/vss_env/bin/python -m jarvis.voice_check --mic      # first-mic session
    ~/vss_env/bin/python -m jarvis.voice_check --json

Nothing here plays audio or touches the live app's /tmp/vss_voice unless
JARVIS_LOG_DIR is left unset (the CLI redirects it to a temp dir before
importing the rest of jarvis). The ``--mic`` session is the checklist in
docs/voice-first-mic-checklist.md executed end to end: record, level,
transcribe, wake-word score, speaker verify.

The pure parsers (``parse_sinks``, ``parse_devices``) are unit-tested; the
probes wrap them around the real commands.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

DUMMY_SINK_NAMES = ("auto_null", "dummy")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    data: dict = field(default_factory=dict)


# ----------------------------------------------------------------- parsers
def parse_sinks(pactl_short: str) -> dict:
    """Parse ``pactl list short sinks`` output.

    Returns {"sinks": [names], "dummy": bool, "suspended": [names]} where
    ``dummy`` is True when the only sinks are PipeWire/Pulse null sinks —
    i.e. speech would play into silence."""
    sinks, suspended = [], []
    for line in (pactl_short or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[1].strip()
        sinks.append(name)
        if len(parts) >= 5 and "SUSPENDED" in parts[4]:
            suspended.append(name)
    real = [s for s in sinks
            if not any(d in s.lower() for d in DUMMY_SINK_NAMES)]
    return {"sinks": sinks, "dummy": not real, "suspended": suspended}


def parse_devices(devices) -> list[dict]:
    """Real capture devices from ``sounddevice.query_devices()`` — the
    same rule ``MachineProfile.detect`` uses (pipewire/default/pulse are
    virtual and always advertise inputs)."""
    out = []
    for i, d in enumerate(devices or []):
        try:
            name = str(d.get("name", ""))
            ch = int(d.get("max_input_channels", 0))
        except Exception:
            continue
        if ch > 0 and name not in ("default", "pipewire", "pulse"):
            out.append({"index": i, "name": name, "channels": ch,
                        "rate": d.get("default_samplerate")})
    return out


# ------------------------------------------------------------------ probes
def output_sink_state(run=subprocess.run) -> dict:
    try:
        out = run(["pactl", "list", "short", "sinks"], capture_output=True,
                  text=True, timeout=3)
        text = out.stdout if getattr(out, "returncode", 1) == 0 else ""
    except Exception:
        text = ""
    state = parse_sinks(text)
    state["probed"] = bool(text)
    return state


def input_devices() -> list[dict]:
    try:
        import sounddevice as sd
        return parse_devices(sd.query_devices())
    except Exception:
        return []


def model_assets() -> dict:
    """Presence of every on-disk asset the voice paths need."""
    from jarvis.config import PATHS
    home = Path.home()
    xtts_dirs = list((home / ".local/share/tts").glob("tts_models--multilingual--multi-dataset--xtts_v2")) \
        if (home / ".local/share/tts").exists() else []
    whisper_cache = home / ".cache" / "whisper"
    try:
        from jarvis.config import CONFIG
        model = CONFIG.model
    except Exception:
        model = "small"
    try:
        import openwakeword
        oww_dir = Path(openwakeword.__file__).parent / "resources" / "models"
        oww_models = sorted(p.name for p in oww_dir.glob("*.onnx"))
    except Exception:
        oww_models = []
    return {
        "whisper_weights": (whisper_cache / f"{model}.pt").exists(),
        "xtts_weights": bool(xtts_dirs),
        "voice_ref": PATHS.VOICE_REF.exists(),
        "oww_models": oww_models,
        "hey_jarvis_model": "hey_jarvis_v0.1.onnx" in oww_models,
        "wakeword_verifier": PATHS.HEY_JARVIS_VERIFIER.exists(),
        "speaker_model": (PATHS.AIWS / "speaker_model" / "embedding_model.ckpt").exists(),
        "voiceprint": PATHS.VOICEPRINT.exists(),
        "settings": PATHS.SETTINGS_FILE.exists(),
    }


def run_offline_checks(load_models: bool = False) -> list[Check]:
    checks: list[Check] = []

    sink = output_sink_state()
    checks.append(Check(
        "audio output sink", not sink["dummy"],
        f"sinks={sink['sinks']}" if sink["probed"] else "pactl unavailable",
        fix=("Only a dummy sink exists: connect HDMI audio (a display with "
             "speakers) or a USB speaker, then check `wpctl status`.")
        if sink["dummy"] else "", data=sink))

    from jarvis.config import MACHINE
    devs = input_devices()
    checks.append(Check(
        "microphone", bool(devs),
        ", ".join(d["name"] for d in devs) if devs else
        "no capture devices (pipewire/default are virtual)",
        fix="" if devs else "Plug in a USB mic, then run --mic.",
        data={"devices": devs, "machine_has_mic": MACHINE.has_mic}))

    assets = model_assets()
    for key in ("whisper_weights", "xtts_weights", "voice_ref",
                "hey_jarvis_model", "wakeword_verifier", "speaker_model"):
        checks.append(Check(f"asset: {key}", bool(assets[key]),
                            str(assets[key])))
    checks.append(Check("asset: voiceprint (enrolled)", assets["voiceprint"],
                        "enrolled" if assets["voiceprint"] else
                        "not enrolled — speaker verify fails open",
                        fix="" if assets["voiceprint"] else
                        "Settings > Voice ID > Enroll (15 s of speech)."))

    for mod, label in (("edge_tts", "edge-tts"), ("whisper", "openai-whisper"),
                       ("faster_whisper", "faster-whisper"),
                       ("openwakeword", "openwakeword"),
                       ("speechbrain", "speechbrain"),
                       ("sounddevice", "sounddevice")):
        try:
            __import__(mod)
            checks.append(Check(f"import {label}", True))
        except Exception as e:
            checks.append(Check(f"import {label}", False, str(e)[:120]))

    try:
        import torch
        checks.append(Check("torch CUDA", torch.cuda.is_available(),
                            torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else "cpu only"))
    except Exception as e:
        checks.append(Check("torch CUDA", False, str(e)[:120]))

    if load_models:
        checks.extend(_load_model_checks())
    return checks


def _load_model_checks() -> list[Check]:
    out: list[Check] = []
    t0 = time.monotonic()
    try:
        from jarvis.transcriber import Transcriber
        tr = Transcriber()
        backend = tr.load()
        out.append(Check("whisper load", True,
                         f"{backend} in {time.monotonic() - t0:.1f}s"))
    except Exception as e:
        out.append(Check("whisper load", False, str(e)[:160]))
    t0 = time.monotonic()
    try:
        from openwakeword.model import Model
        m = Model()
        out.append(Check("openwakeword load", True,
                         f"{list(m.models)} in {time.monotonic() - t0:.1f}s"))
    except Exception as e:
        out.append(Check("openwakeword load", False, str(e)[:160]))
    t0 = time.monotonic()
    try:
        from jarvis.speaker import SpeakerVerifier
        v = SpeakerVerifier()
        ok = v.load_model()
        out.append(Check("ecapa load", ok,
                         f"{v._device} in {time.monotonic() - t0:.1f}s"))
    except Exception as e:
        out.append(Check("ecapa load", False, str(e)[:160]))
    t0 = time.monotonic()
    try:
        from jarvis.tts import TTS
        from jarvis.config import CONFIG
        tts = TTS(engine=CONFIG.tts_engine, cache=False)
        ok = tts.load()
        out.append(Check(f"tts load ({CONFIG.tts_engine})", ok,
                         f"engine now {tts.engine} in {time.monotonic() - t0:.1f}s"))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        t0 = time.monotonic()
        try:
            if tts.engine == "xtts" and tts._xtts is not None:
                tts._synth_xtts("All systems operational, sir.", path)
            else:
                tts._synth_edge("All systems operational, sir.", path)
            size = os.path.getsize(path)
            out.append(Check("tts synth", size > 0,
                             f"{size} bytes in {time.monotonic() - t0:.1f}s"))
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except Exception as e:
        out.append(Check("tts synth", False, str(e)[:160]))
    return out


# ------------------------------------------------------------ mic session
def run_mic_session(seconds: float = 4.0) -> list[Check]:
    """First-mic-session checklist, live: record, level, transcribe,
    wake-word score, speaker verify. Requires a capture device."""
    import numpy as np
    from jarvis.recorder import MicArbiter, Recorder
    from jarvis.config import MACHINE

    out: list[Check] = []
    devs = input_devices()
    if not devs or not MACHINE.has_mic:
        out.append(Check("microphone", False, "no capture device"))
        return out
    rec = Recorder(MicArbiter())
    out.append(Check("recorder devices", True, str(rec.mic_devices)))

    print(f"\n>>> Say 'Hey Jarvis, what time is it' — recording {seconds:.0f}s ...",
          flush=True)
    audio = rec.record_fixed(seconds)
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    out.append(Check("record_fixed", len(audio) > 0,
                     f"{len(audio) / 16000:.1f}s @16k, rms={rms:.4f}",
                     fix="" if rms > 0.003 else
                     "Level is near silence: check the input volume / mute."))
    if not len(audio):
        return out

    try:
        from jarvis.transcriber import Transcriber
        tr = Transcriber()
        backend = tr.load()
        res = tr.transcribe(audio)
        out.append(Check("transcribe", bool(res.text),
                         f"[{backend}] {res.text!r} conf={res.confidence:.2f} "
                         f"accepted={res.accepted}"))
    except Exception as e:
        out.append(Check("transcribe", False, str(e)[:160]))

    try:
        from openwakeword.model import Model
        m = Model()
        a16 = (audio * 32767).astype(np.int16)
        best = 0.0
        for i in range(0, len(a16) - 1279, 1280):
            p = m.predict(a16[i:i + 1280])
            best = max(best, p.get("hey_jarvis", 0.0), p.get("hey_mycroft", 0.0) * 0.7)
        out.append(Check("hotword score", best >= 0.3,
                         f"max rule score {best:.3f} (threshold 0.3)"))
    except Exception as e:
        out.append(Check("hotword score", False, str(e)[:160]))

    try:
        from jarvis.speaker import SpeakerVerifier
        v = SpeakerVerifier()
        v.load_model()
        v.load()
        if v.enrolled:
            ok, score = v.verify(audio)
            out.append(Check("speaker verify", ok, f"score={score:.3f}"))
        else:
            out.append(Check("speaker verify", True,
                             "no voiceprint enrolled (fail-open)",
                             fix="Enroll in Settings > Voice ID."))
    except Exception as e:
        out.append(Check("speaker verify", False, str(e)[:160]))

    t0 = time.monotonic()
    new_thr = rec.calibrate_noise()
    out.append(Check("calibrate_noise", True,
                     f"threshold={new_thr} in {time.monotonic() - t0:.1f}s"))
    return out


# ------------------------------------------------------------------- cli
def _print(checks: list[Check]) -> None:
    width = max(len(c.name) for c in checks) if checks else 20
    for c in checks:
        mark = "OK " if c.ok else "!! "
        print(f"{mark} {c.name.ljust(width)}  {c.detail}")
        if c.fix and not c.ok:
            print(f"      -> {c.fix}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--load", action="store_true", help="also load models")
    ap.add_argument("--mic", action="store_true",
                    help="run the live first-mic session")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not os.environ.get("JARVIS_LOG_DIR"):
        os.environ["JARVIS_LOG_DIR"] = tempfile.mkdtemp(prefix="jarvis-voicecheck-")
    checks = run_offline_checks(load_models=args.load)
    if args.mic:
        checks.extend(run_mic_session(args.seconds))
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=1, default=str))
    else:
        _print(checks)
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
