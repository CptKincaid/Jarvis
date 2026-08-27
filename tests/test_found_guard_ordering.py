"""Regression tests for a defect in jarvis.brain._finish_spoken().

DEFECT (jarvis/brain.py, _finish_spoken):

    text = spoken_from_ollama(strip_markdown(raw), guard_context, user_text, n)

``strip_markdown()`` ends with ``re.sub(r'\\s+', ' ', clean)``, which
collapses every newline into a space. It runs BEFORE
``clean_ollama_reply()`` (called inside ``spoken_from_ollama``), and two of
that function's regexes are anchored to line starts:

    _BULLET_RX = re.compile(r"^\\s*(?:[-*•]|\\d+[.)])\\s+", re.M)
    _TURN_RX   = re.compile(r"\\n\\s*(?:user|hunter)\\s*:.*", re.I | re.S)

By the time they run there are no newlines left, so both are dead in the
production path. They pass their own unit tests, which feed them raw text
directly, and they work if called in the other order:

    clean_ollama_reply("Three things, sir:\\n1. Commit\\n2. Test")
        -> 'Three things, sir:\\nCommit\\nTest'          (bullets stripped)
    clean_ollama_reply(strip_markdown(same))
        -> 'Three things, sir: 1. Commit 2. Test'       (bullets survive)

Consequences, all reproduced through _finish_spoken():

    numbered list -> 'Three things, sir: 1. Commit 2.'
        (the surviving "1." and "2." are then read as sentence ends by
         split_sentences, so the two-sentence cap truncates mid-list into
         nonsense before it reaches TTS)
    dash list     -> 'Options, sir: - one - two'
    markdown table-> 'Here you are, sir: | Day | Temp | |---|---| | Wed | 85 |'
    run-on turn   -> 'Yes, sir. User: and then?'   (Hunter's line spoken back)

The persona sweep measured 0% markdown and 0% lists over 71 replies, but
that is the model declining to emit them, not the guards catching them:
every one of these strings would reach text-to-speech as written.

FIXED 2026-08-26 (brain work item, the LOW guard-ordering finding):
spoken_from_ollama() now runs clean_ollama_reply() on the RAW text first
and strip_markdown() after it, so the line-anchored guards see line
starts; _finish_spoken() no longer pre-strips markdown. strip_markdown()
also handles tables now — the rule row goes and the cell pipes become
clause breaks, so "| Wed | 85 |" is spoken as "Wed, 85".

These tests were the strict-xfail record of the defect; they now pass.
"""
from jarvis.brain import _finish_spoken, clean_ollama_reply


def test_bullet_and_turn_regexes_work_when_given_raw_text():
    """Not a defect — this is the behaviour the guards were written for,
    pinned here so the ordering test below is unambiguous."""
    assert "1." not in clean_ollama_reply("Three things, sir:\n1. Commit\n2. Test")
    assert clean_ollama_reply("Yes, sir.\nUser: and then?") == "Yes, sir."


def test_numbered_list_markers_are_stripped_in_production_path():
    spoken = _finish_spoken("Three things, sir:\n1. Commit\n2. Test\n3. Deploy",
                            "", "", 2)
    assert "1." not in spoken and "2." not in spoken


def test_dash_list_markers_are_stripped_in_production_path():
    spoken = _finish_spoken("Options, sir:\n- one\n- two", "", "", 2)
    assert "- one" not in spoken


def test_run_on_user_turn_is_dropped_in_production_path():
    spoken = _finish_spoken("Yes, sir.\nUser: and then?", "", "", 2)
    assert "User:" not in spoken


def test_markdown_table_does_not_reach_tts():
    spoken = _finish_spoken(
        "Here you are, sir:\n\n| Day | Temp |\n|---|---|\n| Wed | 85 |",
        "", "", 2)
    assert "|" not in spoken
