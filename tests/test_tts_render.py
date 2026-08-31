"""Rendering a reply for an ear that is NOT in this room (jarvis/tts.py).

The phone client answers in Jarvis's own voice, which means one thing above
all others: it must share the room's engine, reference clip and SPEECH
CACHE, so a line he has already said aloud costs a file read rather than a
trip to the GPU -- and it must share none of the room's playback, so the
Spark stays silent while the phone talks.

Nothing here renders anything. The engine is faked at the ``_synth_*``
seam, which is the same seam the fallback logic is tested at, so a test run
never puts a sentence on the GPU and never spawns a player.
"""
from __future__ import annotations

import io
import struct
import wave

import pytest

from jarvis.tts import TTS, Rendition, wav_header, wav_pcm


def wav_bytes(seconds=0.20, rate=24000, amp=6000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(rate * seconds)
        w.writeframes(b"".join(
            int(amp * ((i // 60) % 2 * 2 - 1)).to_bytes(2, "little",
                                                        signed=True)
            for i in range(n)))
    return buf.getvalue()


@pytest.fixture
def voice(tmp_path):
    """An f5 TTS whose sidecar is a function that writes a wav, and whose
    speech cache is a fresh directory. The worker thread it starts is a
    daemon and is never fed, because nothing here calls speak()."""
    tts = TTS(engine="f5", cache=True, cache_dir=tmp_path / "cache",
              pronunciation=False)
    rendered: list[str] = []

    def fake_f5(text, out_path):
        rendered.append(text)
        with open(out_path, "wb") as fh:
            fh.write(wav_bytes())

    tts._synth_f5 = fake_f5
    tts.load = lambda: True          # no sidecar, no model, no GPU
    tts.rendered = rendered
    return tts


def parse(raw: bytes) -> tuple[int, int, int, bytes]:
    """(rate, channels, sampwidth, pcm) of a wav the page would receive."""
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    channels, rate = struct.unpack("<HI", raw[22:28])
    bits = struct.unpack("<H", raw[34:36])[0]
    assert raw[36:40] == b"data"
    return rate, channels, bits // 8, raw[44:]


# ------------------------------------------------------------ 1. the header
def test_a_stream_header_says_it_does_not_know_the_length_yet():
    """0xFFFFFFFF is what ffmpeg writes down a pipe and what decoders read
    as "to the end of the stream"; a real length is written when there is
    one, so a cached clip can still be seeked and ranged."""
    head = wav_header(24000)
    assert len(head) == 44
    assert head[4:8] == b"\xff\xff\xff\xff"
    assert head[40:44] == b"\xff\xff\xff\xff"
    exact = wav_header(24000, data_bytes=1000)
    assert struct.unpack("<I", exact[40:44])[0] == 1000
    assert struct.unpack("<I", exact[4:8])[0] == 1036


def test_the_header_carries_the_rate_the_engine_actually_used():
    rate, channels, width, _ = parse(wav_header(16000, 2, 2, 0))
    assert (rate, channels, width) == (16000, 2, 2)


def test_a_pcm_chunk_is_read_without_a_decoder(tmp_path):
    path = tmp_path / "chunk.wav"
    path.write_bytes(wav_bytes(seconds=0.1, rate=24000))
    fmt, pcm = wav_pcm(str(path))
    assert fmt == (24000, 1, 2)
    assert len(pcm) == int(24000 * 0.1) * 2


def test_an_edge_chunk_is_mp3_wearing_a_wav_name_and_is_decoded(tmp_path):
    """Edge writes MP3 data into a .wav-suffixed file (see the tts module
    docstring), so the cache holds some. `wave` refuses it; libsndfile does
    not, and the phone gets PCM either way rather than a broken clip."""
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "edge.wav"
    tone = np.sin(2 * np.pi * 220 * np.arange(4800) / 24000).astype("float32")
    sf.write(str(path), tone * 0.2, 24000, format="MP3")
    assert path.read_bytes()[:4] != b"RIFF", "this test needs real mp3 bytes"
    fmt, pcm = wav_pcm(str(path))
    assert fmt[0] == 24000 and fmt[1] == 1 and fmt[2] == 2
    assert len(pcm) > 0


# ------------------------------------------------------------- 2. the plan
def test_the_chunks_are_the_ones_speak_would_have_cached(voice):
    """Same clean, same pronounce, same split -- because the cache key is
    the chunk text, and a different split is a guaranteed miss."""
    text = "Two items on Tuesday, sir. The first is at nine."
    assert voice.render_chunks(text) == voice._split_sentences(
        voice._pronounce(voice._clean_for_speech(text)))


def test_an_empty_reply_renders_nothing(voice):
    rend = voice.render("   ")
    assert not rend and len(rend) == 0
    assert list(rend.stream()) == []
    assert voice.rendered == []


def test_markup_is_cleaned_before_it_is_spoken(voice):
    """The page shows the reply; the voice gets _clean_for_speech's version
    of it, exactly as the room does."""
    chunks = voice.render_chunks("**Two** items, sir. See `notes.py`.")
    assert "**" not in " ".join(chunks) and "`" not in " ".join(chunks)


# ------------------------------------------------------------ 3. the cache
def test_a_fresh_reply_is_rendered_and_filed(voice):
    rend = voice.render("Two items on Tuesday, sir.")
    assert rend.cached is False
    raw = b"".join(rend.stream())
    rate, channels, width, pcm = parse(raw)
    assert (rate, channels, width) == (24000, 1, 2)
    assert len(pcm) > 0
    assert voice.rendered == rend.chunks


def test_the_same_reply_a_second_time_is_a_pure_cache_hit(voice):
    line = "Always, sir."
    first = parse(b"".join(voice.render(line).stream()))
    voice.rendered.clear()
    again = voice.render(line)
    assert again.cached is True, "the second ask must not reach the engine"
    # Same audio; only the header differs, because a hit knows its length
    # and a stream does not (see wav_header).
    assert parse(again.body()) == first
    assert voice.rendered == [], "nothing may be synthesized on a hit"


def test_a_line_the_ROOM_said_is_already_there_for_the_phone(voice):
    """The point of sharing the key: the room writes it, the phone reads
    it. This is the room's own store call, verbatim."""
    line = "Good night, sir. I'll be here."
    for chunk in voice.render_chunks(line):
        path = voice.cache.dir / "seed.wav"
        voice.cache.dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wav_bytes())
        voice._store("f5", chunk, str(path))
    rend = voice.render(line)
    assert rend.cached is True
    assert voice.rendered == []


def test_a_half_cached_reply_only_renders_the_half_that_is_missing(voice):
    text = "Two items on Tuesday, sir. The first is at nine."
    chunks = voice.render_chunks(text)
    assert len(chunks) == 2, "this test needs a reply that splits in two"
    voice.cache.dir.mkdir(parents=True, exist_ok=True)
    seed = voice.cache.dir / "seed.wav"
    seed.write_bytes(wav_bytes())
    voice._store("f5", chunks[0], str(seed))
    rend = voice.render(text)
    assert rend.cached is False
    list(rend.stream())
    assert voice.rendered == [chunks[1]]


def test_the_cached_and_the_rendered_clip_are_the_same_audio(voice):
    """A repeat must not merely be fast, it must be the same rendition --
    which for a sampling engine is the other half of why the cache exists."""
    line = "Two items on Tuesday, sir."
    fresh = parse(b"".join(voice.render(line).stream()))
    assert parse(voice.render(line).body()) == fresh


# ---------------------------------------------------------- 4. the streaming
def test_each_chunk_leaves_before_the_next_one_is_rendered(voice):
    """The whole point of streaming: the phone is playing sentence one
    while sentence two is still on the GPU."""
    text = "Two items on Tuesday, sir. The first is at nine."
    order: list[str] = []
    plain = voice._synth_f5

    def watched(chunk, out_path):
        order.append("render")
        plain(chunk, out_path)

    voice._synth_f5 = watched
    for block in voice.render(text).stream():
        order.append("send" if len(block) > 44 else "header")
    assert order == ["render", "header", "send", "render", "send"], order


def test_a_chunk_that_will_not_render_does_not_take_the_reply_with_it(voice):
    """One failed sentence loses that sentence, not the answer."""
    text = "Two items on Tuesday, sir. The first is at nine."
    chunks = voice.render_chunks(text)
    plain = voice._synth_f5

    def flaky(chunk, out_path):
        if chunk == chunks[0]:
            raise RuntimeError("the sidecar went away")
        plain(chunk, out_path)

    voice._synth_f5 = flaky
    raw = b"".join(voice.render(text).stream())
    assert raw[:4] == b"RIFF"
    assert len(parse(raw)[3]) > 0


def test_a_render_with_no_engine_at_all_raises_rather_than_sending_silence(
        voice):
    """503 on the wire beats a 200 carrying an empty clip: the phone can
    say what went wrong, and silence looks identical to a dead toggle."""
    voice.load = lambda: False
    with pytest.raises(RuntimeError):
        list(voice.render("Two items on Tuesday, sir.").stream())


def test_body_fills_the_length_in(voice):
    raw = voice.render("Always, sir.").body()
    assert struct.unpack("<I", raw[40:44])[0] == len(raw) - 44
    assert struct.unpack("<I", raw[4:8])[0] == len(raw) - 8


# ------------------------------------------------------------ 5. the room
def test_a_rendition_never_reaches_the_speaker(voice, monkeypatch):
    """The Spark's silence during a phone turn is a property of this code
    path, not of a flag: there is nothing here that can play."""
    played: list = []
    monkeypatch.setattr(voice, "_play", lambda p: played.append(p))
    monkeypatch.setattr(voice, "_play_stream", lambda s: played.append(s))
    monkeypatch.setattr(voice, "_start_amp_feeder",
                        lambda p: played.append(("amp", p)))
    list(voice.render("Two items on Tuesday, sir.").stream())
    assert played == []
    assert voice._q.empty(), "a render must not queue an utterance"
    assert voice.is_speaking is False


def test_the_engine_a_render_uses_is_the_one_the_cache_is_keyed_on(voice):
    assert voice.render_engine() == "f5"
    voice.engine = "edge"
    assert voice.render_engine() == "edge"
    assert Rendition(voice, "Always, sir.").engine == "edge"
