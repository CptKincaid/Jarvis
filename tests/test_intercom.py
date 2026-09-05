"""Phone intercom (jarvis/intercom.py + the command socket's audio_b64 leg).

Real clips are encoded in memory with soundfile -- the same library the
decode leg uses -- so the wav / ogg / opus / stereo / sample-rate handling is
exercised for real. The socket half runs over a real UNIX socket under tmp
with a fake app; nothing here loads Whisper, torch or a microphone.
"""
from __future__ import annotations

import base64
import io
import json
import socket
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import jarvis.ask as ask_mod
import jarvis.cmdsock as cs
import jarvis.intercom as ic
from jarvis.commander import CommandResult
from jarvis.events import JarvisReply, UserUtterance, bus
from tests.test_app_wiring import build, paths, seams  # noqa: F401 - fixtures


# ------------------------------------------------------------- helpers
def clip_bytes(seconds=1.5, rate=44100, channels=2, fmt="WAV", subtype="PCM_16"):
    """A real encoded clip: a 220 Hz tone, `channels` wide, at `rate`."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    buf = io.BytesIO()
    sf.write(buf, data, rate, format=fmt, subtype=subtype)
    return buf.getvalue()


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


class FakeResult:
    def __init__(self, text="what time is it", accepted=True):
        self.text = text
        self.accepted = accepted
        self.confidence = -0.3


class FakeApp:
    """Enough of JarvisApp for the socket: a decode seam and dispatch_text."""

    def __init__(self, **cfg):
        self.calls: list[tuple] = []
        self.decoded: list = []
        self.result = FakeResult()
        self.rejected = False
        self.verify_seen: list[bool] = []
        self.assistant = SimpleNamespace(get=lambda k, d=None: cfg.get(k, d))

    def decode_clip(self, audio, verify=True):
        self.decoded.append(audio)
        self.verify_seen.append(verify)
        if self.rejected:
            return audio, {"best_score": 0.11}, True, None
        return audio, {}, False, self.result

    def dispatch_text(self, text, source="typed", quiet=False, turn_id=""):
        self.calls.append((text, source, quiet))
        res = CommandResult(handled=True, reply="It is ten, sir.", speak=True,
                            status="Clock")
        bus.publish(JarvisReply(text=res.reply, speak=res.speak, turn_id=turn_id))
        return res


@pytest.fixture
def server(tmp_path):
    app = FakeApp()
    srv = cs.CommandSocket(tmp_path / "command.sock", app, timeout_s=5.0,
                           idle_grace_s=0.4)
    assert srv.start()
    yield SimpleNamespace(app=app, sock=srv.sock_path, srv=srv)
    srv.stop()


def _send_raw(sock_path, obj) -> list:
    """One request, raw, so a malformed one can be sent too."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect(str(sock_path))
    s.sendall((json.dumps(obj) + "\n").encode())
    data = b""
    while b'"end"' not in data:
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    s.close()
    return [json.loads(ln) for ln in data.decode().splitlines() if ln.strip()]


# ------------------------------------------------------------- decoding
def test_a_stereo_44k_wav_becomes_mono_float32_at_16k():
    audio = ic.to_mono_16k(clip_bytes(seconds=1.5, rate=44100, channels=2))
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert abs(len(audio) / ic.SAMPLE_RATE - 1.5) < 0.05
    assert 0.05 < float(np.abs(audio).max()) < 1.0


@pytest.mark.parametrize("rate,fmt,subtype", [
    (16000, "WAV", "PCM_16"),          # already at the pipeline rate
    (48000, "OGG", "OPUS"),            # what termux -e opus writes
    (44100, "OGG", "VORBIS"),
    (48000, "FLAC", "PCM_16"),
])
def test_every_format_libsndfile_reads_lands_at_16k(rate, fmt, subtype):
    audio = ic.to_mono_16k(clip_bytes(seconds=1.0, rate=rate, channels=1,
                                      fmt=fmt, subtype=subtype))
    # Opus adds its own encoder padding, so the duration is close, not exact.
    assert 0.7 < len(audio) / ic.SAMPLE_RATE < 1.3
    assert audio.dtype == np.float32


def test_aac_is_refused_with_the_format_the_phone_needs():
    """termux-microphone-record defaults to AAC and libsndfile cannot read
    it; the error has to say so or the phone leg is unfixable from the far
    end."""
    with pytest.raises(ic.IntercomError) as exc:
        ic.to_mono_16k(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 4096)
    assert exc.value.kind == "format"
    assert "wav" in exc.value.text and "AAC" in exc.value.text


def test_a_clip_longer_than_the_cap_is_truncated_not_refused():
    audio = ic.to_mono_16k(clip_bytes(seconds=2.0, rate=16000, channels=1))
    assert len(audio) == 32000
    long_clip = clip_bytes(seconds=ic.MAX_SECONDS + 5, rate=8000, channels=1)
    assert len(ic.to_mono_16k(long_clip)) == int(ic.MAX_SECONDS * ic.SAMPLE_RATE)


def test_silence_shorter_than_a_tenth_of_a_second_is_empty():
    with pytest.raises(ic.IntercomError) as exc:
        ic.to_mono_16k(clip_bytes(seconds=0.02, rate=16000, channels=1))
    assert exc.value.kind == "empty"


def test_base64_is_size_checked_before_it_is_decoded():
    """A 200 MB blob must be refused on the encoded text, not after it has
    been expanded into memory."""
    payload = "A" * (ic.MAX_B64_CHARS + 4096)
    with pytest.raises(ic.IntercomError) as exc:
        ic.decode_b64(payload)
    assert exc.value.kind == "size"
    # ...and again on the decoded length, for a cap set below the default
    raw = clip_bytes(seconds=1.0, rate=16000, channels=1)
    with pytest.raises(ic.IntercomError) as exc:
        ic.decode_b64(b64(raw), max_bytes=len(raw) - 1)
    assert exc.value.kind == "size"
    assert ic.decode_b64(b64(raw), max_bytes=len(raw)) == raw


@pytest.mark.parametrize("payload,kind", [("", "empty"), ("   ", "empty"),
                                          ("!!!!not base64!!!!", "b64"),
                                          (12345, "b64")])
def test_a_bad_base64_field_is_a_clean_error(payload, kind):
    with pytest.raises(ic.IntercomError) as exc:
        ic.decode_b64(payload)
    assert exc.value.kind == kind


def test_clip_from_b64_is_the_whole_leg():
    audio = ic.clip_from_b64(b64(clip_bytes(seconds=1.0, rate=48000, channels=2)))
    assert audio.dtype == np.float32 and 15000 < len(audio) < 17000


# -------------------------------------------------------- transcription
def test_transcribe_uses_the_apps_decode_seam_without_the_speaker_gate():
    app = FakeApp()
    audio = ic.clip_from_b64(b64(clip_bytes(seconds=1.0, rate=16000, channels=1)))
    assert ic.transcribe(app, audio) == "what time is it"
    assert app.verify_seen == [False]
    ic.transcribe(app, audio, verify=True)
    assert app.verify_seen == [False, True]


def test_a_clip_the_speaker_gate_drops_says_so():
    app = FakeApp()
    app.rejected = True
    with pytest.raises(ic.IntercomError) as exc:
        ic.transcribe(app, np.zeros(16000, dtype=np.float32), verify=True)
    assert exc.value.kind == "speaker" and exc.value.text == ic.NOT_HIM_LINE


def test_an_empty_transcript_is_an_error_but_a_shaky_one_is_dispatched():
    app = FakeApp()
    app.result = FakeResult(text="   ")
    with pytest.raises(ic.IntercomError) as exc:
        ic.transcribe(app, np.zeros(16000, dtype=np.float32))
    assert exc.value.kind == "empty"
    # Low confidence is NOT fatal: the sender cannot be asked to repeat, so
    # the best transcript travels and the doubt goes to the log.
    app.result = FakeResult(text="what time is it", accepted=False)
    assert ic.transcribe(app, np.zeros(16000, dtype=np.float32)) == "what time is it"


def test_a_looping_transcript_is_never_dispatched_from_the_phone():
    """A repetition loop or a prompt echo is not a low-confidence
    transcript of something he said, it is Whisper running away; the
    microphone path never dispatches one and the socket path must not
    either (review of 69afb9f)."""
    app = FakeApp()
    app.result = FakeResult(text="Quennevex, Quennevex, Quennevex, Quennevex,",
                            accepted=False)
    app.result.looping = True
    with pytest.raises(ic.IntercomError) as exc:
        ic.transcribe(app, np.zeros(16000, dtype=np.float32))
    assert exc.value.kind == "looping"
    assert exc.value.text == ic.NOTHING_HEARD_LINE


def test_an_app_without_the_decode_seam_fails_cleanly():
    with pytest.raises(ic.IntercomError):
        ic.transcribe(SimpleNamespace(), np.zeros(16000, dtype=np.float32))


# ------------------------------------------------------------ the socket
def test_a_clip_is_heard_then_dispatched_as_an_intercom_turn(server):
    seen = []
    bus.subscribe(UserUtterance, seen.append)
    try:
        msgs = list(cs.ask(server.sock, audio=clip_bytes(seconds=1.0, rate=16000,
                                                         channels=1)))
    finally:
        bus.unsubscribe(UserUtterance, seen.append)
    assert msgs[0] == {"kind": "heard", "text": "what time is it"}
    assert [m["text"] for m in msgs if m["kind"] == "reply"] == ["It is ten, sir."]
    assert msgs[-1]["reason"] == "done"
    # source AND the default silence: nothing is said in the room
    assert server.app.calls == [("what time is it", "intercom", True)]
    assert [(e.text, e.source) for e in seen] == [("what time is it", "intercom")]
    # and the app was handed the pipeline's own array shape
    assert server.app.decoded[0].dtype == np.float32
    assert server.app.decoded[0].ndim == 1


def test_speak_true_asks_for_the_answer_aloud(server):
    list(cs.ask(server.sock, audio=clip_bytes(rate=16000, channels=1), speak=True))
    assert server.app.calls[-1] == ("what time is it", "intercom", False)


def test_an_undecodable_clip_answers_with_the_reason_and_never_dispatches(server):
    msgs = _send_raw(server.sock, {"audio_b64": b64(b"ftypM4A " + b"\x00" * 2048)})
    assert msgs[0]["kind"] == "error" and "AAC" in msgs[0]["text"]
    assert msgs[-1] == {"kind": "end", "reason": "error"}
    assert server.app.calls == []


def test_the_speaker_gate_is_off_unless_the_config_asks_for_it(tmp_path):
    for verify in (False, True):
        app = FakeApp(**{"intercom.verify_speaker": verify})
        srv = cs.CommandSocket(tmp_path / f"v{verify}.sock", app, timeout_s=5.0)
        assert srv.start()
        try:
            list(cs.ask(srv.sock_path, audio=clip_bytes(rate=16000, channels=1)))
        finally:
            srv.stop()
        assert app.verify_seen == [verify]


def test_the_intercom_can_be_switched_off(tmp_path):
    app = FakeApp(**{"intercom.enabled": False})
    srv = cs.CommandSocket(tmp_path / "off.sock", app, timeout_s=5.0)
    assert srv.start()
    try:
        msgs = list(cs.ask(srv.sock_path, audio=clip_bytes(rate=16000, channels=1)))
    finally:
        srv.stop()
    assert msgs[0] == {"kind": "error", "text": cs.INTERCOM_OFF_LINE}
    assert app.calls == []


def test_the_configured_size_cap_is_enforced(tmp_path):
    app = FakeApp(**{"intercom.max_mb": 0.001})       # ~1 KB
    srv = cs.CommandSocket(tmp_path / "cap.sock", app, timeout_s=5.0)
    assert srv.start()
    try:
        msgs = list(cs.ask(srv.sock_path, audio=clip_bytes(seconds=2.0, rate=44100,
                                                           channels=2)))
    finally:
        srv.stop()
    assert msgs[0]["kind"] == "error" and "too large" in msgs[0]["text"]
    assert app.calls == []


def test_the_framing_carries_a_whole_clip_and_refuses_a_runaway_one(server):
    """64 KB was the old per-request cap; one base64 clip is ~13 MB, and a
    request with no newline at all must not be reported as bad JSON."""
    assert cs.MAX_LINE > ic.MAX_AUDIO_BYTES
    raw = clip_bytes(seconds=3.0, rate=44100, channels=2)
    assert len(b64(raw)) > 65536                     # would have been truncated
    msgs = list(cs.ask(server.sock, audio=raw))
    assert msgs[0]["kind"] == "heard"

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(20)
    s.connect(str(server.sock))
    try:
        s.sendall(b"x" * (cs.MAX_LINE + 4096))       # no newline, ever
        data = b""
        while b'"end"' not in data:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    finally:
        s.close()
    lines = [json.loads(ln) for ln in data.decode().splitlines() if ln.strip()]
    assert lines[0]["kind"] == "error" and "too large" in lines[0]["text"]


def test_a_text_request_is_untouched_by_any_of_this(server):
    msgs = list(cs.ask(server.sock, "what time is it"))
    assert "heard" not in {m["kind"] for m in msgs}
    assert server.app.calls == [("what time is it", "cli", False)]
    assert server.app.decoded == []


# --------------------------------------------------------------- the CLI
def test_the_cli_sends_a_clip_from_a_file(server, tmp_path, capsys):
    path = tmp_path / "clip.wav"
    path.write_bytes(clip_bytes(rate=16000, channels=1))
    rc = ask_mod.main(["--sock", str(server.sock), "--send-audio", str(path)])
    out, err = capsys.readouterr()
    assert rc == 0 and out.strip() == "It is ten, sir."
    assert "[heard] what time is it" in err
    assert server.app.calls[-1] == ("what time is it", "intercom", True)


def test_the_cli_sends_a_clip_from_stdin_without_reading_it_as_text(server,
                                                                    monkeypatch,
                                                                    capsys):
    raw = clip_bytes(rate=16000, channels=1)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(
        buffer=io.BytesIO(raw), isatty=lambda: False,
        read=lambda: (_ for _ in ()).throw(AssertionError("read as text"))))
    assert ask_mod.main(["--sock", str(server.sock), "--send-audio", "-"]) == 0
    assert server.app.calls[-1][1] == "intercom"
    assert capsys.readouterr().out.strip() == "It is ten, sir."


def test_the_cli_speak_flag_reaches_the_server(server, tmp_path):
    path = tmp_path / "clip.ogg"
    path.write_bytes(clip_bytes(rate=48000, channels=1, fmt="OGG", subtype="OPUS"))
    assert ask_mod.main(["--sock", str(server.sock), "--send-audio", str(path),
                         "--speak"]) == 0
    assert server.app.calls[-1][2] is False          # quiet=False -> spoken


def test_the_cli_reports_a_missing_or_empty_clip_as_four(server, tmp_path, capsys):
    rc = ask_mod.main(["--sock", str(server.sock), "--send-audio",
                       str(tmp_path / "nope.wav")])
    assert rc == 4 and "error:" in capsys.readouterr().err
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    assert ask_mod.main(["--sock", str(server.sock), "--send-audio", str(empty)]) == 4


def test_the_cli_refuses_an_oversized_clip_before_the_socket(tmp_path, capsys):
    big = tmp_path / "big.wav"
    big.write_bytes(b"\x00" * (ask_mod.MAX_CLIP_BYTES + 1))
    assert ask_mod.main(["--sock", str(tmp_path / "nothing.sock"),
                         "--send-audio", str(big)]) == 4
    assert "limit" in capsys.readouterr().err


def test_listen_says_so_when_the_box_has_no_recorder(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ask_mod.shutil, "which", lambda name: None)
    rc = ask_mod.main(["--sock", str(tmp_path / "nothing.sock"), "--listen", "2"])
    assert rc == 4 and "no recorder" in capsys.readouterr().err


def test_listen_prefers_termux_and_never_the_aac_default(monkeypatch, tmp_path):
    """termux-microphone-record writes AAC unless told otherwise, and
    libsndfile cannot read AAC -- so -e opus must be in the argv."""
    monkeypatch.setattr(ask_mod.shutil, "which",
                        lambda name: "/x/" + name if name == "termux-microphone-record"
                        else None)
    argv, needs_wait, path = ask_mod._recorder_argv(8, tmp_path)
    assert argv[0] == "termux-microphone-record" and needs_wait
    assert "-e" in argv and argv[argv.index("-e") + 1] == "opus"
    assert str(path) == argv[-1] and path.suffix == ".ogg"


def test_listen_uses_the_timeout_as_the_duration_for_parecord(monkeypatch, tmp_path):
    """parecord has no duration flag: it is stopped by the timeout, and a
    TimeoutExpired there is the normal end of a recording, not a failure."""
    monkeypatch.setattr(ask_mod.shutil, "which",
                        lambda name: "/x/parecord" if name == "parecord" else None)
    seen = {}

    def fake_run(argv, **kw):
        seen["timeout"] = kw.get("timeout")
        (tmp_path / "jarvis-intercom.wav").write_bytes(
            clip_bytes(seconds=1.0, rate=16000, channels=1))
        raise ask_mod.subprocess.TimeoutExpired(argv, kw.get("timeout"))

    monkeypatch.setattr(ask_mod.subprocess, "run", fake_run)
    assert ask_mod.record_clip(4, out_dir=tmp_path)
    assert seen["timeout"] == 4.0


def test_listen_records_and_sends_what_the_recorder_wrote(server, monkeypatch,
                                                          tmp_path, capsys):
    raw = clip_bytes(seconds=1.0, rate=16000, channels=1)

    def fake_which(name):
        return "/usr/bin/arecord" if name == "arecord" else None

    def fake_run(argv, **kw):
        Pth = tmp_path / "jarvis-intercom.wav"
        assert str(Pth) == argv[-1]
        Pth.write_bytes(raw)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ask_mod.shutil, "which", fake_which)
    monkeypatch.setattr(ask_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(ask_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    assert ask_mod.main(["--sock", str(server.sock), "--listen", "2"]) == 0
    assert server.app.calls[-1][1] == "intercom"
    assert capsys.readouterr().out.strip() == "It is ten, sir."


# ------------------------------------------------------------ the real app
def test_the_real_app_answers_a_clip_without_speaking_it(build, monkeypatch):  # noqa: F811
    """One round trip through the built JarvisApp: the intercom's transcript
    is dispatched like any turn, the answer comes back over the socket, and
    the room stays silent unless the clip asked for speech."""
    from jarvis.config import CONFIG, PATHS
    monkeypatch.setattr(CONFIG, "talkback", True)
    # The transcript gate fails SHUT once a voiceprint exists; with it armed
    # the intercom must still get through, because verify defaults off.
    monkeypatch.setattr(CONFIG, "speaker_verify", True)
    app = build()
    app.transcriber = SimpleNamespace(
        transcribe=lambda audio: FakeResult("jarvis what time is it"))
    app.speaker = SimpleNamespace(
        enrolled=True,
        filter_segments=lambda a: (_ for _ in ()).throw(
            AssertionError("the speaker gate ran on an intercom clip")))
    app.start_assistant(residency=False)
    raw = clip_bytes(seconds=1.0, rate=48000, channels=1, fmt="OGG", subtype="OPUS")

    msgs = list(cs.ask(PATHS.COMMAND_SOCK, audio=raw))
    assert msgs[0]["kind"] == "heard" and "what time is it" in msgs[0]["text"]
    assert [m["text"] for m in msgs if m["kind"] == "reply"]
    assert not app.tts.spoken                      # the soundbar stayed quiet
    assert not app._quiet_turn                     # the mute is per turn

    msgs = list(cs.ask(PATHS.COMMAND_SOCK, audio=raw, speak=True))
    assert [m["text"] for m in msgs if m["kind"] == "reply"]
    assert app.tts.spoken, "speak=true should have reached the TTS"
    app.stop_assistant()


def test_the_microphone_path_still_runs_the_speaker_gate(build, monkeypatch):  # noqa: F811
    """The intercom's verify=False must not have loosened the voice path."""
    from jarvis.config import CONFIG
    monkeypatch.setattr(CONFIG, "speaker_verify", True)
    app = build()
    ran = []
    app.speaker = SimpleNamespace(
        enrolled=True,
        filter_segments=lambda a: (ran.append(a), (None, {"best_score": 0.1}))[1])
    audio = np.zeros(16000, dtype=np.float32)
    _a, _stats, rejected, result = app.decode_clip(audio)          # the default
    assert rejected and result is None and len(ran) == 1
    _a, _stats, rejected, _r = app.decode_clip(audio, verify=False)
    assert not rejected and len(ran) == 1
