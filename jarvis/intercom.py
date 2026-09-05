"""Phone intercom: a recorded clip instead of a wake word.

The command socket (jarvis/cmdsock.py) already carries a whole turn back to
a shell client as JSON lines. This module is the other half of the trip: a
``{"audio_b64": ...}`` request is decoded here into exactly what the recorder
hands the pipeline -- mono float32 at 16 kHz -- so the resident Whisper and
the speaker gate can be reused verbatim, and the transcript is dispatched
with ``source="intercom"``.

Why it exists: from bed, or from the next room, the wake word is out of
range and the soundbar's answer would wake the house. Termux on the phone
records a clip and pipes it over the SSH session that already exists::

    termux-microphone-record -q -l 8 -e opus -f $HOME/j.ogg
    ssh spark 'jarvis --send-audio -' < $HOME/j.ogg

and the reply comes back as text on the phone. ``speak: true`` is opt-in,
so by default nothing is said aloud in the room.

Three things bite here and are handled in one place:

* **The format.** libsndfile reads wav / ogg / opus / flac and NOT the AAC
  that ``termux-microphone-record`` writes by default, so a clip in the
  default format fails with an unhelpful "Format not recognised". The error
  names the formats that work.
* **The rate.** Phones record at 44.1 or 48 kHz in stereo; Whisper (and the
  ECAPA speaker gate) want 16 kHz mono. Both conversions happen here.
* **The size.** A clip is ~13.4 MB of base64 for 10 MB of audio, orders of
  magnitude past the old 64 KB request framing, so the cap is explicit and
  is checked on the ENCODED text before anything is decoded into memory.

The speaker gate is available but off by default (``intercom.verify_speaker``):
the socket is mode 0600 and only reachable through the user's own SSH
session, which is the authentication, while a phone microphone and a lossy
codec move the ECAPA embedding far enough that the transcript gate -- which
fails SHUT once a voiceprint exists -- would reject his own voice. Turn it
on and the same ``filter_segments`` the microphone path uses runs on the
clip.
"""
from __future__ import annotations

import base64
import binascii
import io

from jarvis.logs import get_logger

log = get_logger("intercom")

SOURCE = "intercom"
SAMPLE_RATE = 16000               # what Transcriber.transcribe expects

# 10 MB of audio: ~5 minutes of 16-bit 16 kHz wav, far more than a push-to-talk
# clip, and small enough that the base64 of it still fits comfortably in RAM.
MAX_AUDIO_BYTES = 10 * 1024 * 1024
# Base64 is 4 bytes out per 3 in; the slack covers padding and newlines.
MAX_B64_CHARS = (MAX_AUDIO_BYTES + 2) // 3 * 4 + 1024
# A clip longer than this is truncated rather than refused: a phone left
# recording must not park the resident Whisper for ten minutes.
MAX_SECONDS = 120.0
MIN_SECONDS = 0.1

BAD_BASE64_LINE = "That clip wasn't valid base64, sir."
TOO_BIG_LINE = "That clip is too large, sir; keep it under {mb:.0f} megabytes."
BAD_FORMAT_LINE = ("I couldn't decode that clip, sir; send wav, ogg, opus or "
                   "flac -- AAC and m4a are not readable here.")
EMPTY_LINE = "That clip was empty, sir."
NO_DECODER_LINE = "I have no audio decoder installed, sir."
NOT_HIM_LINE = "That didn't sound like you, sir."
NOTHING_HEARD_LINE = "I couldn't make anything out of that clip, sir."
OFF_LINE = "The intercom is switched off, sir."


class IntercomError(Exception):
    """A clip that cannot become a turn. ``text`` is the line the client
    (and the phone) is shown; ``kind`` groups it for the log."""

    def __init__(self, text: str, kind: str = "audio"):
        super().__init__(text)
        self.text = text
        self.kind = kind


# ------------------------------------------------------------- decoding
def decode_b64(payload, max_bytes: int = MAX_AUDIO_BYTES) -> bytes:
    """The base64 field of a request -> the raw container bytes.

    The size is checked on the ENCODED text first: refusing a 200 MB blob
    only after decoding it would already have cost the memory."""
    if isinstance(payload, (bytes, bytearray)):
        payload = bytes(payload).decode("ascii", "replace")
    if not isinstance(payload, str):
        raise IntercomError(BAD_BASE64_LINE, "b64")
    payload = payload.strip()
    if not payload:
        raise IntercomError(EMPTY_LINE, "empty")
    limit_chars = (max_bytes + 2) // 3 * 4 + 1024
    if len(payload) > limit_chars:
        raise IntercomError(TOO_BIG_LINE.format(mb=max_bytes / 1048576), "size")
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise IntercomError(BAD_BASE64_LINE, "b64") from exc
    if not raw:
        raise IntercomError(EMPTY_LINE, "empty")
    if len(raw) > max_bytes:
        raise IntercomError(TOO_BIG_LINE.format(mb=max_bytes / 1048576), "size")
    return raw


def _resample(mono, in_rate: int):
    """`mono` at `in_rate` -> SAMPLE_RATE, float32.

    scipy's polyphase filter when it is installed (it is, in the venv);
    linear interpolation otherwise, which is poorer but keeps the intercom
    working on a box without scipy rather than refusing the clip."""
    import numpy as np

    in_rate = int(in_rate)
    if in_rate == SAMPLE_RATE or in_rate <= 0:
        return np.asarray(mono, dtype=np.float32)
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(in_rate, SAMPLE_RATE)
        out = resample_poly(mono, SAMPLE_RATE // g, in_rate // g)
    except Exception:                     # noqa: BLE001 - no scipy, or a odd rate
        log.debug("resample_poly unavailable; interpolating", exc_info=True)
        n_out = int(round(len(mono) * SAMPLE_RATE / float(in_rate)))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        src = np.arange(len(mono), dtype=np.float64)
        dst = np.linspace(0, len(mono) - 1, n_out, dtype=np.float64)
        out = np.interp(dst, src, mono)
    return np.asarray(out, dtype=np.float32)


def to_mono_16k(raw: bytes):
    """Container bytes -> the float32 @ 16 kHz mono array the pipeline uses.

    Decoding happens from memory (no temp file): soundfile reads any
    libsndfile container out of a BytesIO."""
    import numpy as np

    try:
        import soundfile as sf
    except Exception as exc:              # noqa: BLE001 - the venv has it
        log.warning("intercom: soundfile is not importable: %s", exc)
        raise IntercomError(NO_DECODER_LINE, "decoder") from exc
    try:
        data, in_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:              # noqa: BLE001 - LibsndfileError et al
        log.info("intercom: undecodable clip (%d bytes): %s", len(raw), exc)
        raise IntercomError(BAD_FORMAT_LINE, "format") from exc
    if data.size == 0:
        raise IntercomError(EMPTY_LINE, "empty")
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    audio = _resample(np.asarray(mono, dtype=np.float32), in_rate)
    limit = int(MAX_SECONDS * SAMPLE_RATE)
    if len(audio) > limit:
        log.info("intercom: clip of %.1fs truncated to %.0fs",
                 len(audio) / SAMPLE_RATE, MAX_SECONDS)
        audio = audio[:limit]
    if len(audio) < int(MIN_SECONDS * SAMPLE_RATE):
        raise IntercomError(EMPTY_LINE, "empty")
    return np.ascontiguousarray(audio, dtype=np.float32)


def clip_from_b64(payload, max_bytes: int = MAX_AUDIO_BYTES):
    """The one call a transport needs: base64 field -> pipeline audio."""
    return to_mono_16k(decode_b64(payload, max_bytes))


def settings(app) -> tuple:
    """``(enabled, verify_speaker, max_bytes)`` off the app's assistant
    config, with the defaults when there is no config (or it is
    unreadable).

    Every transport reads the SAME three switches here rather than
    keeping its own copy: the command socket (jarvis/cmdsock.py) and the
    phone client (jarvis/webapp.py) must not be able to disagree about
    whether the intercom is switched off or how large a clip may be."""
    cfg = getattr(app, "assistant", None)
    if cfg is None:
        return True, False, MAX_AUDIO_BYTES
    try:
        return (bool(cfg.get("intercom.enabled", True)),
                bool(cfg.get("intercom.verify_speaker", False)),
                max(1, int(float(cfg.get("intercom.max_mb", 10)) * 1048576)))
    except Exception:                     # noqa: BLE001 - config boundary
        log.debug("intercom config unreadable; using defaults", exc_info=True)
        return True, False, MAX_AUDIO_BYTES


def text_from_clip(app, raw: bytes) -> str:
    """Container bytes off a transport -> the transcript to dispatch.

    The whole trip in one call: the switches, the size cap, the decode to
    16 kHz mono, the speaker gate and Whisper. Raises IntercomError
    carrying the line to show the far end."""
    enabled, verify, max_bytes = settings(app)
    if not enabled:
        raise IntercomError(OFF_LINE, "disabled")
    raw = bytes(raw or b"")
    if not raw:
        raise IntercomError(EMPTY_LINE, "empty")
    if len(raw) > max_bytes:
        raise IntercomError(TOO_BIG_LINE.format(mb=max_bytes / 1048576), "size")
    return transcribe(app, to_mono_16k(raw), verify=verify)


def text_from_b64(app, payload) -> str:
    """text_from_clip for a transport that carries the clip as base64.

    The size is still checked on the ENCODED text (decode_b64), before the
    bytes exist in memory."""
    enabled, verify, max_bytes = settings(app)
    if not enabled:
        raise IntercomError(OFF_LINE, "disabled")
    return transcribe(app, clip_from_b64(payload, max_bytes=max_bytes),
                      verify=verify)


# ---------------------------------------------------------- transcription
def transcribe(app, audio, verify: bool = False) -> str:
    """Run one intercom clip through the app's own decode path.

    ``verify`` runs speaker.filter_segments exactly as the microphone path
    does (and fails shut the same way); the default trusts the socket.
    Returns the transcript, raises IntercomError when there is nothing to
    dispatch."""
    decode = getattr(app, "decode_clip", None)
    if not callable(decode):
        raise IntercomError(NOTHING_HEARD_LINE, "app")
    _audio, stats, rejected, result = decode(audio, verify=bool(verify))
    if rejected:
        score = 0.0
        if isinstance(stats, dict):
            try:
                score = float(stats.get("best_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
        log.info("intercom: clip rejected by the speaker gate (score %.2f)", score)
        raise IntercomError(NOT_HIM_LINE, "speaker")
    text = (getattr(result, "text", "") or "").strip()
    if not text:
        raise IntercomError(NOTHING_HEARD_LINE, "empty")
    # A repetition loop or a prompt echo is not a doubtful transcript of
    # something he said, it is Whisper running away; the microphone path
    # never dispatches one (app._process_audio, reason "looping") and the
    # phone must not either -- there is no intent gate on this source.
    if getattr(result, "looping", False):
        log.info("intercom: repetition loop or prompt echo, not dispatched")
        raise IntercomError(NOTHING_HEARD_LINE, "looping")
    # Low confidence is NOT fatal here: the microphone path can ask him to
    # say it again, but a phone clip is already sent and re-recording it is
    # the user's own choice, so the best transcript is dispatched and the
    # confidence goes to the log.
    if not getattr(result, "accepted", True):
        log.info("intercom: low-confidence transcript accepted anyway: %r", text)
    return text
