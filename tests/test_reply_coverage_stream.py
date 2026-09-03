"""The streamed path judges a held confirmation against what on_sentence
actually RECEIVED (F25, 09-03 bug pass, reproduced both ways).

``spoken`` (the SPEAK tag) and the stream disagree in two directions: the
stream's splitter counts "a.m." as a sentence end and spends the cap early,
and _finish_spoken char-caps where the stream does not. Judging "missing"
on ``spoken`` therefore left the write unsaid in one shape and said twice
in the other. Both shapes are pinned here through the real _chat_sync with
the branch's own _streamer seam. No Ollama, no audio.
"""
import jarvis.brain as brain_mod
from tests.test_brain_tools import brain  # noqa: F401  (fixture)
from tests.test_reply_coverage import ASKED, NOTES_LINE, _streamer, _tool_chunk


def test_a_streamed_confirmation_is_not_repeated_when_the_char_cap_trims_it(brain, monkeypatch):  # noqa: F811,E501
    """Render round streams the confirmation VERBATIM as sentence 4 (inside
    the sentence cap of 4). _emit_sentence streams all four. But SPEAK is
    char-capped at HARD_SPOKEN_CHARS=500, which trims sentences 3 and 4 off
    `spoken`; held_lines_missing(spoken, ...) then says the line is missing
    and it goes to on_sentence AGAIN."""
    s1 = ("You have a meeting with ValerieAnne Staffeldt at eleven fifteen "
          "tomorrow morning for thirty minutes, sir, over the usual video link "
          "that she sent across last week when you spoke about the grant. ")
    s2 = ("Then Biosensors runs from twelve forty-five for about two hours in "
          "the engineering building, which leaves you a short gap before the "
          "chiropractor at four o'clock, sir, if the traffic is kind to you. ")
    s3 = ("After that the gym at six, and nothing else is on the calendar for "
          "the evening, so you should be clear from about seven onwards, sir, "
          "which leaves the evening free for the reading you mentioned. ")
    assert all(len(s) <= 250 for s in (s1, s2, s3))
    assert len(s1 + s2 + s3) > brain_mod.HARD_SPOKEN_CHARS
    b, sent = _streamer(brain, monkeypatch, [
        _tool_chunk(("get_calendar", {"range": "tomorrow"}),
                    ("notes", {"action": "add", "text": "milk"})),
        [{"message": {"role": "assistant", "content": s1}},
         {"message": {"role": "assistant", "content": s2}},
         {"message": {"role": "assistant", "content": s3}},
         {"message": {"role": "assistant", "content": NOTES_LINE},
          "done": True, "load_duration": 0}]])
    spoken = []
    b._chat_sync(ASKED, on_sentence=spoken.append)
    assert spoken.count(NOTES_LINE) == 1, \
        f"confirmation went to TTS {spoken.count(NOTES_LINE)} times"


def test_a_streamed_confirmation_is_said_when_the_stream_cap_ate_it(brain, monkeypatch):  # noqa: F811,E501
    """The SILENT direction of R1. _split_complete_sentences (the stream)
    is not abbreviation-aware; split_sentences (the reply) is. A calendar
    render with 'a.m.' / 'p.m.' counts as MORE sentences on the stream, so
    the stream cap (4) is reached before the verbatim confirmation, which
    is never streamed. `spoken` (abbr-aware, 4 sentences) still carries it
    verbatim, so held_lines_missing says nothing is missing: not appended,
    not sent to on_sentence. STREAMED tag -> app never speaks SPEAK. The
    write happened, was displayed, and was never said."""
    content = ("You have a meeting with ValerieAnne at 11:15 a.m. tomorrow, sir. "
               "Then Biosensors at 12:45 p.m. for about two hours. "
               "The chiropractor is at 4 p.m. after that. "
               + NOTES_LINE)
    assert len(brain_mod.split_sentences(content)) == 4
    b, sent = _streamer(brain, monkeypatch, [
        _tool_chunk(("get_calendar", {"range": "tomorrow"}),
                    ("notes", {"action": "add", "text": "milk"})),
        [{"message": {"role": "assistant", "content": content},
          "done": True, "load_duration": 0}]])
    spoken = []
    tags = b._chat_sync(ASKED, on_sentence=spoken.append)
    assert NOTES_LINE in dict(tags)["SPEAK"]
    assert spoken.count(NOTES_LINE) == 1, "the write was displayed but never said"
