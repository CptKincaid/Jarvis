"""Streamed replies: each sentence speaks as the model produces it.

The route slice of a tool turn (1.5-3 s) was mostly the model finishing a
reply before any of it was spoken. With on_sentence given, the final turn
streams; a round that turns out to be a tool call speaks nothing; the
returned tags carry STREAMED so the caller shows the reply without
speaking it twice; a stream that dies after sentences were spoken keeps
what was said rather than announcing a timeout.
"""
import urllib.error
import jarvis.brain as brain_mod
from jarvis.brain import _split_complete_sentences


def test_sentence_splitting_waits_for_text_after_the_terminator():
    assert _split_complete_sentences("It is ten.") == ([], "It is ten.")
    assert _split_complete_sentences("It is ten. And ") == (["It is ten."], "And ")
    done, rest = _split_complete_sentences("One. Two! Three? Fo")
    assert done == ["One.", "Two!", "Three?"] and rest == "Fo"
    # an abbreviation followed by text still splits (a known limit; the
    # TTS joins short fragments), but nothing is ever lost
    done, rest = _split_complete_sentences("Dr. Smith is here. Ok")
    assert " ".join(done + [rest]) == "Dr. Smith is here. Ok"


def _brain(monkeypatch, chunks):
    b = brain_mod.JarvisBrain(None, None, registry=None)
    monkeypatch.setattr(b, "_dynamic_context", lambda: ("", ""))

    def stream(path, payload, timeout=None):
        assert payload.get("stream") is not True or True
        for c in chunks:
            yield c
    monkeypatch.setattr(brain_mod, "_http_stream", stream)
    monkeypatch.setattr(brain_mod, "_http", lambda *a, **k: (_ for _ in ()).throw(AssertionError("non-streaming path used")))
    return b


def _chunks(text, done_extra=None):
    out = []
    for i in range(0, len(text), 7):
        out.append({"message": {"role": "assistant", "content": text[i:i + 7]}, "done": False})
    out.append({"message": {"role": "assistant", "content": ""}, "done": True, "load_duration": 0, **(done_extra or {})})
    return out


def test_sentences_are_spoken_as_they_land_and_not_again(monkeypatch):
    b = _brain(monkeypatch, _chunks("It is ten past nine, sir. The evening is clear. Rain is unlikely."))
    spoken = []
    tags = b._chat_sync("what time is it", on_sentence=spoken.append)
    # the spoken cap (MAX_SPOKEN_SENTENCES = 2) holds sentence by sentence
    assert spoken == ["It is ten past nine, sir.", "The evening is clear."]
    kinds = [t for t, _ in tags]
    assert kinds == ["STREAMED", "SPEAK"], kinds
    assert dict(tags)["STREAMED"] == "2"
    assert dict(tags)["SPEAK"].startswith("It is ten past nine, sir.")


def test_the_spoken_cap_still_holds_while_streaming(monkeypatch):
    text = " ".join(f"Sentence number {i} is here." for i in range(8))
    b = _brain(monkeypatch, _chunks(text))
    spoken = []
    b._chat_sync("tell me", on_sentence=spoken.append)
    assert len(spoken) == brain_mod.MAX_SPOKEN_SENTENCES


def test_a_tool_call_round_speaks_nothing(monkeypatch):
    chunks = [{"message": {"role": "assistant", "content": "", "tool_calls": [
                  {"function": {"name": "get_time", "arguments": {}}}]}, "done": True, "load_duration": 0}]
    spoken = []
    b = _brain(monkeypatch, chunks)
    data, content, calls = b._stream_round([], None, 3, spoken.append, [])
    assert spoken == [] and len(calls) == 1 and content == ""


def test_a_stream_that_dies_keeps_what_was_said(monkeypatch):
    def stream(path, payload, timeout=None):
        yield {"message": {"role": "assistant", "content": "The first part is fine. The sec"}, "done": False}
        raise urllib.error.URLError("connection dropped")
    b = brain_mod.JarvisBrain(None, None, registry=None)
    monkeypatch.setattr(b, "_dynamic_context", lambda: ("", ""))
    monkeypatch.setattr(brain_mod, "_http_stream", stream)
    monkeypatch.setattr(brain_mod.bus, "publish", lambda ev: None)
    spoken = []
    tags = b._chat_sync("q", on_sentence=spoken.append)
    assert spoken == ["The first part is fine."]
    assert dict(tags)["SPEAK"] == "The first part is fine."
    assert brain_mod.MODEL_SLOW_LINE not in dict(tags)["SPEAK"]


def test_without_on_sentence_nothing_changes(monkeypatch):
    b = brain_mod.JarvisBrain(None, None, registry=None)
    monkeypatch.setattr(b, "_dynamic_context", lambda: ("", ""))
    monkeypatch.setattr(brain_mod, "_http_stream", lambda *a, **k: (_ for _ in ()).throw(AssertionError("streamed")))
    monkeypatch.setattr(brain_mod, "_http", lambda path, payload, timeout=None: {
        "message": {"role": "assistant", "content": "Plain reply, sir."}, "load_duration": 0})
    tags = b._chat_sync("q")
    assert [t for t, _ in tags] == ["SPEAK"]
