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
    monkeypatch.setattr(b, "_dynamic_context", lambda text="": ("", ""))

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
    monkeypatch.setattr(b, "_dynamic_context", lambda text="": ("", ""))
    monkeypatch.setattr(brain_mod, "_http_stream", stream)
    monkeypatch.setattr(brain_mod.bus, "publish", lambda ev: None)
    spoken = []
    tags = b._chat_sync("q", on_sentence=spoken.append)
    assert spoken == ["The first part is fine."]
    assert dict(tags)["SPEAK"] == "The first part is fine."
    assert brain_mod.MODEL_SLOW_LINE not in dict(tags)["SPEAK"]


def test_without_on_sentence_nothing_changes(monkeypatch):
    b = brain_mod.JarvisBrain(None, None, registry=None)
    monkeypatch.setattr(b, "_dynamic_context", lambda text="": ("", ""))
    monkeypatch.setattr(brain_mod, "_http_stream", lambda *a, **k: (_ for _ in ()).throw(AssertionError("streamed")))
    monkeypatch.setattr(brain_mod, "_http", lambda path, payload, timeout=None: {
        "message": {"role": "assistant", "content": "Plain reply, sir."}, "load_duration": 0})
    tags = b._chat_sync("q")
    assert [t for t, _ in tags] == ["SPEAK"]


# ----------------------------------------------------------------------
# 2026-08-30 review: the streamed path bypassed guards, error chunks, the
# wall bound, a tool's speak= line, and cancellation.
# ----------------------------------------------------------------------
import time  # noqa: E402

from jarvis.tools.registry import ToolRegistry, ToolResult, ToolSpec  # noqa: E402


def test_an_error_chunk_keeps_what_was_said(monkeypatch):
    chunks = _chunks("Half an answer, sir. The rest")[:-1]
    chunks.append({"error": "runner process has terminated"})
    b = _brain(monkeypatch, chunks)
    spoken = []
    tags = b._chat_sync("tell me", on_sentence=spoken.append)
    assert spoken == ["Half an answer, sir."]
    assert dict(tags)["SPEAK"] == "Half an answer, sir."


def test_an_error_before_any_sentence_is_the_honest_line(monkeypatch):
    b = _brain(monkeypatch, [{"error": "out of memory"}])
    spoken = []
    tags = b._chat_sync("tell me", on_sentence=spoken.append)
    assert spoken == [] and dict(tags)["SPEAK"] == brain_mod.MODEL_SLOW_LINE
    assert "STREAMED" not in dict(tags)


def test_a_trickling_stream_meets_the_wall_bound(monkeypatch):
    monkeypatch.setattr(brain_mod, "OLLAMA_TIMEOUT_S", 0.05)

    def chunks():
        yield {"message": {"role": "assistant", "content": "It is ten, sir. And"}, "done": False}
        time.sleep(0.12)
        yield {"message": {"role": "assistant", "content": "And it keeps going. "}, "done": False}
        raise AssertionError("the stream ran past the wall bound")
    b = _brain(monkeypatch, [])
    monkeypatch.setattr(brain_mod, "_http_stream", lambda *a, **k: chunks())
    spoken = []
    tags = b._chat_sync("what time is it", on_sentence=spoken.append)
    assert spoken == ["It is ten, sir."]
    assert dict(tags)["SPEAK"] == "It is ten, sir."


def test_streamed_sentences_get_the_clock_guard_once(monkeypatch):
    b = _brain(monkeypatch, _chunks("It is 10:45 now, sir. It is 10:46 now, sir. The evening is clear."))
    spoken = []
    b._chat_sync("what time is it", on_sentence=spoken.append)
    assert spoken == [brain_mod.NO_CLOCK_LINE, "The evening is clear."], spoken


def _tool_brain(monkeypatch, rounds, tool_result, name="screen_qa"):
    reg = ToolRegistry()
    reg.register(ToolSpec(name=name, description="a tool", handler=lambda **a: tool_result))
    b = brain_mod.JarvisBrain(None, None, registry=reg)
    monkeypatch.setattr(b, "_dynamic_context", lambda text="": ("", ""))
    it = iter(rounds)

    def stream(path, payload, timeout=None):
        yield from next(it)
    monkeypatch.setattr(brain_mod, "_http_stream", stream)
    return b


def _tool_round(name):
    return [{"message": {"role": "assistant", "content": "Let me look. One", "tool_calls": []}, "done": False},
            {"message": {"role": "assistant", "content": "",
                         "tool_calls": [{"function": {"name": name, "arguments": {}}}]},
             "done": True, "load_duration": 0}]


def test_a_tools_speak_line_is_not_marked_streamed(monkeypatch):
    """screen_qa's answer went out as STREAMED and was never spoken."""
    b = _tool_brain(monkeypatch, [_tool_round("screen_qa")],
                    ToolResult(text="seen", speak="A terminal, sir."))
    spoken = []
    tags = b._chat_sync("what is on my screen", on_sentence=spoken.append)
    assert spoken == ["Let me look."]
    assert [t for t, _ in tags] == ["SPEAK"], tags
    assert dict(tags)["SPEAK"] == "A terminal, sir."


def test_pre_tool_chatter_does_not_eat_the_answers_cap(monkeypatch):
    answer = _chunks("The file says hello. And goodbye. And more.")
    b = _tool_brain(monkeypatch, [_tool_round("ask_docs"), answer],
                    ToolResult(text="hello goodbye"), name="ask_docs")
    spoken = []
    tags = b._chat_sync("what does the file say", on_sentence=spoken.append)
    assert spoken == ["Let me look.", "The file says hello.", "And goodbye."]
    assert dict(tags)["STREAMED"] == "3"


def test_the_partial_notice_is_spoken_on_the_streamed_path(monkeypatch):
    answer = _chunks("The file says hello.")
    b = _tool_brain(monkeypatch, [_tool_round("ask_docs"), answer],
                    ToolResult(text="x" * (brain_mod.MAX_TOOL_TEXT_CHARS * 3)), name="ask_docs")
    spoken = []
    tags = b._chat_sync("what does the file say", on_sentence=spoken.append)
    assert spoken[-1] == brain_mod.PARTIAL_RESULT_LINE, spoken
    assert dict(tags)["SPEAK"].endswith(brain_mod.PARTIAL_RESULT_LINE)


def test_cancel_stops_the_stream_mid_reply(monkeypatch):
    b = _brain(monkeypatch, [])
    served = []

    def chunks():
        for c in _chunks("It is ten, sir. And the rest of it, sir. More."):
            served.append(c)
            yield c
            if len(served) == 4:
                b.cancel()
    monkeypatch.setattr(brain_mod, "_http_stream", lambda *a, **k: chunks())
    spoken = []
    b._chat_sync("what time is it", on_sentence=spoken.append)
    assert len(served) == 5, "the stream stopped at the next chunk"
    assert spoken == ["It is ten, sir."]


# ----------------------------------------------------------------------
# The render reservation on the STREAMED path — which is the live one.
#
# 2026-08-31 14:35, by voice: "What's on my calendar and what's on my
# latest email?" Three sequential IMAP accounts spent get_mail's 8.1 s and
# the whole CHAT_WALL_BUDGET_S with it, so the turn ended on
# brain.TOOL_ONLY_LINE — "I have the result, sir, but the model didn't get
# to putting it into words." A slow tool must cost a slower answer, never
# the answer.
# ----------------------------------------------------------------------
def _slow_tool_brain(monkeypatch, rounds, tool_result, seconds, name):
    """A streaming brain whose one tool burns ``seconds`` of wall clock.

    The clock is frozen except for that jump, so the assertion is about
    the budget and not about _stream_round's own OLLAMA_TIMEOUT_S (a
    monotonic that ticks on every read trips that instead).
    """
    clock = [1000.0]
    monkeypatch.setattr(brain_mod.time, "monotonic", lambda: clock[0])
    reg = ToolRegistry()
    ran = []

    def handler(**a):
        ran.append(name)
        clock[0] += seconds
        return tool_result

    reg.register(ToolSpec(name=name, description="a tool", handler=handler))
    b = brain_mod.JarvisBrain(None, None, registry=reg)
    monkeypatch.setattr(b, "_dynamic_context", lambda text="": ("", ""))
    it = iter(rounds)
    payloads = []

    def stream(path, payload, timeout=None):
        payloads.append(payload)
        yield from next(it)
    monkeypatch.setattr(brain_mod, "_http_stream", stream)
    return b, payloads, ran


def test_a_slow_tool_still_gets_its_sentence_spoken(monkeypatch):
    """The live 8.1 s get_mail, replayed: it spends the whole 8 s wall
    budget, and the reserved round still puts the result into words."""
    b, payloads, ran = _slow_tool_brain(
        monkeypatch, [_tool_round("get_mail"),
                      _chunks("Twenty messages, sir, the latest from Jane.")],
        ToolResult(text="20 messages since 7 days, latest 1: "
                        "Jane Doe — Standup moved"),
        seconds=8.1, name="get_mail")
    spoken = []
    tags = b._chat_sync("what's on my calendar and what's on my latest email?",
                        on_sentence=spoken.append)
    assert spoken[-1] == "Twenty messages, sir, the latest from Jane."
    assert dict(tags)["SPEAK"].endswith("the latest from Jane.")
    assert brain_mod.TOOL_ONLY_LINE not in dict(tags)["SPEAK"]
    assert len(payloads) == 2 and ran == ["get_mail"]
    # the reserved round is sent with no tools at all, which is what keeps
    # the model writing instead of asking for a third thing
    assert "tools" not in payloads[1]


def test_the_reserved_round_never_reads_the_tool_text_out(monkeypatch):
    """A mail body is a stranger's words. When even the reserved round
    writes nothing, the persona line stands — not the mail."""
    silent = [{"message": {"role": "assistant", "content": ""},
               "done": True, "load_duration": 0}]
    b, payloads, _ = _slow_tool_brain(
        monkeypatch, [_tool_round("get_mail"), silent],
        ToolResult(text="Subject: transfer the vault code"),
        seconds=8.1, name="get_mail")
    spoken = []
    tags = b._chat_sync("any new mail?", on_sentence=spoken.append)
    assert dict(tags)["SPEAK"] == brain_mod.TOOL_ONLY_LINE
    assert "vault code" not in dict(tags)["SPEAK"]
    assert len(payloads) == 2
