"""The spelling hold: letters said one at a time keep the mic open.

WHAT HE REPORTED (2026-09-05, 12:25): "i did the email test where i
spelled out the name and he made it weirdish". He spelled an address out
letter by letter and the capture was chopped into FOUR separate turns
inside one intended sentence; the first three fragments were thrown away
and the fourth -- a headless tail -- was all that ever reached the
commander, which is why the uncertain-intent card fired.

WHY. The endpointer stops a capture CONFIG.endpoint_silence (0.8 s) after
the VAD last heard speech. A man spelling leaves a real pause between
letters, and the filler hold cannot help because he is saying letters, not
"um". Spelling is structurally incompatible with a 0.8 s endpoint.
test_reproduce_the_chop_measures_where_it_stops MEASURES that, so the fix
is aimed at a number.

THE RULE (jarvis/spelling.py, one sentence): a decode whose tail is a
SPELLING RUN -- two or more single-character tokens in a row, however
whisper punctuates them ("q-z-v", "q. z. v", "q z v") -- or a single
letter left dangling after a function word that cannot end an utterance
("send an email to q") is somebody mid-spelling, and the capture is held
open CONFIG.spell_hold_s longer through the SAME seam the filler hold
uses (Recorder.note_partial -> _hold_extra -> _counted_hold).

WHY IT CANNOT HOLD AN ORDINARY SENTENCE OPEN. A run needs two LETTERS
other than "a" and "i" -- the first version of this argument said "two
single characters", which was FALSE IN WRITING because digits counted as
run characters, and the verdict measured "gate 4 b" and four more like it
being held (section 8 below); a run of one is not a run, so "what is plan
B" is never held; and the dangling-letter case needs BOTH a bare letter
(never a digit, never "a"/"I") AND a preceding closed-class word that
cannot be the last word of a sentence, which "plan" is not.

Everything here runs on synthetic arrays and scripted fakes: no device is
opened, no audio is played, no real address appears -- every letter and
domain below is invented (tests/conftest.py's firewall covers the rest).
"""
from __future__ import annotations

import inspect
import logging
import time

import numpy as np
import pytest

import jarvis.recorder as recorder_mod
from jarvis import spelling
from jarvis.endpoint import CHUNK, SAMPLE_RATE, VoiceEndpointer
from jarvis.recorder import Recorder


# ----------------------------------------------------------- the harness
class ScriptedVAD:
    """probability per 512-sample chunk, from a list; last value repeats."""

    def __init__(self, probs):
        self.probs, self.i = list(probs), 0

    def __call__(self, chunk):
        p = self.probs[min(self.i, len(self.probs) - 1)]
        self.i += 1
        return p

    def reset(self):
        self.i = 0


def _chunks(n):
    return np.zeros(n * CHUNK, dtype=np.float32)


def _recorder(endpointer):
    """A recorder mid-capture, exactly as tests/test_filler_hold.py builds
    one: bare object, scripted stop, no device anywhere."""
    rec = object.__new__(Recorder)
    rec.recording = True
    rec.endpointer = endpointer
    rec._ep_cursor = 0
    rec._record_rate = SAMPLE_RATE
    rec._record_start_time = time.monotonic() - 10.0   # well past the grace
    rec._audio_frames = []
    rec._stop_endpoint, rec._stop_dead_air = "", None
    rec._voice_stopped = False
    rec._followup = False
    rec._silence_start = None
    rec._loud_chunks = 0
    rec.stops = []

    def stop(reason="manual", endpoint="", dead_air=None):
        rec.stops.append((reason, endpoint, dead_air))
        rec.recording = False
    rec.stop = stop
    return rec


def _push(rec, n_chunks):
    rec._audio_frames.append(_chunks(n_chunks).reshape(-1, 1))


def _hold_config(monkeypatch, **over):
    cfg = dict(endpoint_vad=True, endpoint_silence=0.8, silence_grace=0.0,
               silence_timeout=2.5,
               filler_hold=True, filler_hold_s=1.5, filler_max_holds=3,
               spell_hold=True, spell_hold_s=2.0, spell_max_holds=16)
    cfg.update(over)
    for k, v in cfg.items():
        monkeypatch.setattr(recorder_mod.CONFIG, k, v)


# The address he spells is INVENTED and so is its domain. Seven characters,
# one of them a digit, because whisper wrote a digit into the run he
# actually said.
LETTERS = ["q", "z", "v", "k", "b", "w", "7"]
PREFIX = "Send an email to"
LETTER_CHUNKS = 10        # 0.32 s of speech per letter
GAP_CHUNKS = 31           # 0.99 s of quiet between them -- his measured shape
PREVIEW_CHUNKS = 28       # the app's 0.9 s preview cadence


def _spelled(n: int) -> str:
    """What the live preview has decoded after n letters -- hyphen-joined,
    which is how whisper writes a spelled run."""
    return PREFIX if n <= 0 else PREFIX + " " + "-".join(LETTERS[:n])


def spell_a_capture(monkeypatch, **over):
    """Say LETTERS one at a time into a scripted VAD, reporting the live
    preview's newest decode the way jarvis.app._partial_loop does, and
    return (letters_heard_when_it_stopped, gap_it_stopped_on).

    This is the reproduction. Nothing here is a device: the VAD is a list
    of probabilities and the audio is zeros.
    """
    _hold_config(monkeypatch, **over)
    script = []
    for _ in LETTERS:
        script += [0.9] * LETTER_CHUNKS + [0.0] * GAP_CHUNKS
    script += [0.0] * 4000
    rec = _recorder(VoiceEndpointer(model=ScriptedVAD(script)))
    stride = LETTER_CHUNKS + GAP_CHUNKS
    pushed = 0
    since_preview = PREVIEW_CHUNKS
    total = len(LETTERS) * stride + 400          # plenty of trailing quiet
    while rec.recording and pushed < total:
        _push(rec, 1)
        pushed += 1
        since_preview += 1
        if since_preview >= PREVIEW_CHUNKS:
            since_preview = 0
            done = sum(1 for i in range(len(LETTERS))
                       if i * stride + LETTER_CHUNKS <= pushed)
            rec.note_partial(_spelled(done), pushed * CHUNK / SAMPLE_RATE,
                             rec.capture_id)
        rec._check_endpoint()
    heard = sum(1 for i in range(len(LETTERS))
                if i * stride + LETTER_CHUNKS <= pushed)
    gap = rec.stops[0][2] if rec.stops else None
    return heard, gap


def test_reproduce_the_chop_measures_where_it_stops(monkeypatch):
    """THE MEASUREMENT. With the hold off -- which is jarvis-v3 today --
    seven letters spelled with a 0.99 s pause between them are cut after
    the FIRST one, 0.8-0.9 s into the first gap. Everything after it
    became a separate capture; that is his four turns.

    With the hold on, every letter lands in one capture.
    """
    without, gap = spell_a_capture(monkeypatch, spell_hold=False)
    assert without == 1, f"expected the old 1-letter chop, measured {without}"
    assert 0.8 <= gap < 1.0

    with_hold, gap = spell_a_capture(monkeypatch)
    assert with_hold == len(LETTERS), (
        f"only {with_hold} of {len(LETTERS)} letters stayed in one capture")
    # ...and the capture still ends: spell_hold_s after the last letter.
    assert 2.8 <= gap < 3.1


# ------------------------------------------------- (1) the rule, on text
def test_a_run_of_two_or_more_single_characters_is_spelling():
    run = spelling.spelling_run
    assert run("q-z-v") == 3
    assert run("q z v") == 3
    assert run("Q. Z. V.") == 3
    assert run("q-z-v, k-b-w-7") == 7           # whisper's comma between groups
    assert run("send an email to q-z-v") == 3


def test_a_single_stray_letter_in_an_ordinary_sentence_is_not_spelling():
    """His pin: "what is plan B" must never be held. A run of one is not a
    run, and "plan" is not a word that leaves a letter dangling."""
    run = spelling.spelling_run
    assert run("what is plan B") == 0
    assert run("play track B") == 0
    assert run("I'll take option A") == 0
    assert run("vitamin C") == 0
    assert run("B") == 0                         # nothing in front of it at all


def test_a_letter_left_dangling_after_a_function_word_is_spelling():
    """"Send an email to q" is his first fragment: one letter, and the word
    in front of it cannot end an utterance."""
    run = spelling.spelling_run
    assert run("send an email to q") == 1
    assert run("send an email to Q.") == 1
    assert run("spell it with k") == 1


def test_a_dangling_digit_or_article_letter_is_never_spelling():
    """"set a timer for 5" and "give me an A" are ordinary turns: a digit
    never dangles, and neither do the two one-letter English words."""
    run = spelling.spelling_run
    assert run("set a timer for 5") == 0
    assert run("volume to 5") == 0
    assert run("that is a") == 0
    assert run("the answer is I") == 0


def test_the_run_survives_the_punctuation_whisper_hangs_on_letters():
    run = spelling.spelling_run
    assert run("q, z, v") == 3
    assert run("q. z. v") == 3
    assert run("q-z-v.") == 3
    assert run("") == 0
    assert run(None) == 0


def test_a_run_stops_at_the_first_multi_letter_word():
    """"at" ends the run: the domain is spoken, not spelled."""
    assert spelling.spelling_run("q-z-v at example dot com") == 0
    assert spelling.spelling_run("example.com") == 0
    assert spelling.spelling_run("t-shirt") == 0      # "shirt" is not a letter
    assert spelling.spelling_run("x-ray") == 0


def test_the_minimum_run_is_two_and_it_is_named():
    assert spelling.MIN_RUN == 2


# --------------------------------------------------- (2) the reassembly
def test_a_spelled_local_part_folds_into_a_word():
    assert spelling.fold("q-z-v") == "qzv"
    assert spelling.fold("q z v") == "qzv"
    assert spelling.fold("q-z-v, k-b-w") == "qzvkbw"


def test_a_spelled_local_part_with_a_spoken_domain_folds():
    assert spelling.fold("q-z-v at example dot com") == "qzv at example dot com"
    assert spelling.fold("q z v at example.com") == "qzv at example.com"


def test_a_mixed_address_folds_only_the_spelled_half():
    """Some spelled, some spoken: "dana dot q-z-v at example dot com"."""
    assert spelling.fold("dana dot q-z-v at example dot com") == \
        "dana dot qzv at example dot com"


def test_an_ordinary_sentence_is_returned_unchanged():
    for text in ("what is plan B", "set a timer for 5", "send an email to q",
                 "example.com", "t-shirt weather", ""):
        assert spelling.fold(text) == text


def test_the_fold_maps_every_character_back_to_the_raw_text():
    """The map is what lets outbox.address_span keep returning a span in
    the text AS GIVEN, so the commander can still cut an address out of a
    sentence it never folded."""
    raw = "send it to q-z-v at example dot com"
    folded, imap = spelling.fold_spans(raw)
    assert folded == "send it to qzv at example dot com"
    assert len(imap) == len(folded)
    i = folded.index("qzv")
    assert raw[imap[i]] == "q"
    assert raw[imap[i + 2]] == "v"
    assert imap[i + 2] + 1 == raw.index(" at ")


# ------------------------------------------- (3) the address, end to end
def test_a_spelled_address_parses_into_the_address_he_meant():
    from jarvis import outbox
    assert outbox.parse_address("q-z-v at example dot com") == "qzv@example.com"
    assert outbox.parse_address("send an email to q z v at example dot com") \
        == "qzv@example.com"
    assert outbox.parse_address("q-z-v, k-b-w at example dot com") \
        == "qzvkbw@example.com"


def test_the_comma_whisper_puts_before_at_does_not_break_the_parse():
    """His fourth fragment's exact shape: two spelled groups, a comma
    after the last one, then the domain. The comma before "at" is
    whisper's punctuation of the run, not a separator."""
    from jarvis import outbox
    assert outbox.parse_address("q-z-v, k-b-w-7, at example dot com") \
        == "qzvkbw7@example.com"


def test_a_punctuated_domain_is_read_when_the_dot_is_tight_his_ruling_b():
    """OVERTURNED BY HIM, 2026-09-05. This test used to pin the opposite
    -- "example.com" said as one word was refused on purpose, because
    ordinary prose has the same skeleton (the _SAID_TLDS note in
    jarvis/outbox.py). He ruled that it must count as a domain: it is the
    exact shape whisper wrote his own domain down in, and the refusal cost
    him the RE-ASK as well as the draft, so he got nothing at all.

    A SPACED punctuation is still refused, which is what keeps the prose
    corpus in tests/test_contacts.py byte-identical: a sentence boundary
    always has a space after the full stop."""
    from jarvis import outbox
    assert outbox.parse_address("q-z-v, k-b-w-7, at example.com.") \
        == "qzvkbw7@example.com"
    assert outbox.parse_address("dana at gmail. com") == ""


def test_a_mixed_spelled_and_spoken_address_parses():
    from jarvis import outbox
    assert outbox.parse_address("dana dot q-z-v at example dot com") \
        == "dana.qzv@example.com"


def test_the_span_of_a_spelled_address_is_in_the_raw_text(monkeypatch):
    from jarvis import outbox
    raw = "send it to q-z-v at example dot com please"
    span = outbox.address_span(raw)
    assert span is not None
    addr, start, end = span
    assert addr == "qzv@example.com"
    assert raw[start:end] == "q-z-v at example dot com"


def test_an_ordinary_sentence_still_parses_to_no_address():
    from jarvis import outbox
    assert outbox.parse_address("what is plan B") == ""
    assert outbox.parse_address("set a timer for 5") == ""


def test_a_spelled_address_is_masked_out_of_a_log_line():
    """His whole spelled local part must not reach a log line raw --
    the invariant jarvis/outbox.mask_addresses exists for."""
    from jarvis import outbox
    for said in ("send it to q-z-v at example dot com",
                 "send it to q z v at example dot com",
                 "send it to q-z-v, k-b-w at example dot com"):
        masked = outbox.mask_addresses(said)
        # the FIRST character survives (it is what makes the line readable)
        # and nothing after it does
        assert masked.startswith("send it to q…"), masked
        assert "example dot com" in masked
        for ch in ("z", "v", "k", "b", "w"):
            assert f" {ch} " not in masked and f"-{ch}" not in masked, masked


# ------------------------------------------------------- (4) the recorder
def _speaks_then_pauses(monkeypatch, **over):
    """0.96 s of speech (30 chunks) then silence, checked once so the VAD
    has heard the speech."""
    _hold_config(monkeypatch, **over)
    rec = _recorder(VoiceEndpointer(model=ScriptedVAD([0.9] * 30 + [0.0] * 4000)))
    _push(rec, 30)
    assert rec._check_endpoint() is False
    assert abs(rec.endpointer.last_speech_seconds - 0.96) < 1e-9
    return rec


def test_a_trailing_spelling_run_holds_the_stop(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("send it to q-z-v", 0.96, rec.capture_id)
    _push(rec, 26)                                   # 0.83 s: the old stop
    assert rec._check_endpoint() is False and rec.stops == []
    assert rec._spell_holds == 1 and rec._filler_holds == 0
    _push(rec, 60)                                   # 2.75 s: inside 0.8 + 2.0
    assert rec._check_endpoint() is False and rec.stops == []
    _push(rec, 6)                                    # 2.94 s: the hold is spent
    assert rec._check_endpoint() is True
    (reason, endpoint, dead_air), = rec.stops
    assert (reason, endpoint) == ("silence", "vad") and dead_air >= 2.8


def test_an_ordinary_sentence_with_a_stray_letter_is_not_held(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("what is plan B", 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec._spell_holds == 0
    assert rec.stops[0][2] < 1.0


def test_an_utterance_that_is_only_letters_holds(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("q z v", 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is False and rec._spell_holds == 1


def test_a_spelled_part_followed_by_a_spoken_domain_stops_normally(monkeypatch):
    """Once he has said "at example dot com" the tail is a word again, so
    the ordinary 0.8 s stop takes the turn -- no spelling latency is paid
    at the end of the sentence."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("q-z-v at example dot com", 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec._spell_holds == 0


def test_a_hold_is_counted_once_per_pause_not_per_tick(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("send it to q-z-v", 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is False and rec._spell_holds == 1
    for _ in range(6):
        _push(rec, 1)
        assert rec._check_endpoint() is False
    assert rec._spell_holds == 1


def test_the_hold_logs_the_letter_count_and_never_the_letters(monkeypatch, caplog):
    """A spelled local part IS half an address. The recorder stores and
    logs how MANY characters were in the run, never which ones."""
    caplog.set_level(logging.INFO, logger="jarvis.recorder")
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("send it to q-z-v", 0.96, rec.capture_id)
    _push(rec, 26)
    rec._check_endpoint()
    lines = [r.getMessage() for r in caplog.records if "spelling hold" in r.getMessage()]
    assert lines == ["spelling hold 1/16: 3 characters at 1.0s, waiting 2.0s"], lines
    assert not any("q-z-v" in r.getMessage() for r in caplog.records)
    assert "q" not in str(rec._latest_partial)


def test_at_most_spell_max_holds_pauses_per_capture(monkeypatch):
    _hold_config(monkeypatch, spell_max_holds=2)
    script = ([0.9] * 30 + [0.0] * 30) * 2 + [0.9] * 30 + [0.0] * 4000
    rec = _recorder(VoiceEndpointer(model=ScriptedVAD(script)))
    ep = rec.endpointer
    for pause in (1, 2):
        _push(rec, 30)
        assert rec._check_endpoint() is False
        rec.note_partial("send it to q-z-v", ep.last_speech_seconds, rec.capture_id)
        _push(rec, 30)                                # 0.96 s pause: held
        assert rec._check_endpoint() is False and rec.stops == []
        assert rec._spell_holds == pause
    _push(rec, 30)
    assert rec._check_endpoint() is False
    rec.note_partial("send it to q-z-v", ep.last_speech_seconds, rec.capture_id)
    _push(rec, 30)                                    # third pause: the cap
    assert rec._check_endpoint() is True
    assert rec._spell_holds == 2 and rec.stops[0][2] < 1.0


def test_the_hold_is_off_by_config(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch, spell_hold=False)
    rec.note_partial("send it to q-z-v", 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec._spell_holds == 0


def test_a_filler_and_a_spelling_run_are_counted_separately(monkeypatch):
    """One seam, two counters: the ledger's holds=N keeps meaning what it
    meant, and a spelling turn does not eat his three filler holds."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("send it to q-z-v", 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is False
    assert (rec._spell_holds, rec._filler_holds) == (1, 0)


def test_a_new_capture_forgets_the_last_ones_spelling_holds():
    rec = object.__new__(Recorder)
    rec._latest_partial = ("um", 3, 3.0, 1.0)
    rec._filler_holds, rec._filler_hold_key = 2, 3.0
    rec._spell_holds, rec._spell_hold_key = 4, 3.0
    rec._reset_filler_hold()
    assert rec._latest_partial is None
    assert rec._spell_holds == 0 and rec._spell_hold_key is None
    assert "self._reset_filler_hold()" in inspect.getsource(Recorder.start)


# --------------------------------------------- (5) the two hard stoppers
def test_the_sixty_second_cap_still_wins(monkeypatch):
    """A spelling hold lives in _check_endpoint; the cap lives in
    _check_silence and answers to nothing else. Sixteen holds of 2 s could
    otherwise add 32 s to a capture."""
    _hold_config(monkeypatch)
    rec = _recorder(VoiceEndpointer(model=ScriptedVAD([0.9] * 30 + [0.0] * 4000)))
    _push(rec, 30)
    rec._check_endpoint()
    rec.note_partial("send it to q-z-v", 0.96, rec.capture_id)
    rec._record_start_time = time.monotonic() - recorder_mod.MAX_RECORDING_SECONDS
    assert rec._check_silence() is True
    assert rec.stops[0][0] == "cap"


def test_the_energy_timer_can_still_end_a_held_pause(monkeypatch):
    """The 2.5 s energy timer beats a 2.8 s spelling hold, exactly as it
    beats the 2.3 s filler hold; the ledger's stop= then says energy."""
    _hold_config(monkeypatch, silence_timeout=2.5)
    rec = _recorder(VoiceEndpointer(model=ScriptedVAD([0.9] * 30 + [0.0] * 4000)))
    _push(rec, 30)
    rec._check_endpoint()
    rec.note_partial("send it to q-z-v", 0.96, rec.capture_id)
    for _ in range(26):                       # one frame per poll tick
        _push(rec, 1)
    assert rec._check_endpoint() is False and rec._spell_holds == 1
    # the energy detector has been in silence for 2.6 s -- past the 2.5 s
    # timer, and short of the 0.8 + 2.0 the spelling hold is waiting for
    rec._silence_start = time.monotonic() - 2.6
    assert rec._check_silence() is True
    assert rec.stops[0][:2] == ("silence", "energy")
    assert rec.stops[0][2] >= 2.6


# ------------------------------------------------------------ (6) config
def test_config_ships_the_spelling_hold_settings():
    from jarvis.config import Config
    cfg = Config()
    assert cfg.spell_hold is True
    assert cfg.spell_hold_s == 2.0
    assert cfg.spell_max_holds == 16


def test_the_spelling_settings_load_from_the_settings_file(tmp_path, monkeypatch):
    import json
    from jarvis import config
    path = tmp_path / "settings.json"
    monkeypatch.setattr(config.PATHS, "SETTINGS_FILE", path)
    path.write_text(json.dumps({"spell_hold_s": 3, "spell_max_holds": 4,
                                "spell_hold": False}))
    cfg = config.Config.load()
    assert cfg.spell_hold_s == 3.0 and isinstance(cfg.spell_hold_s, float)
    assert cfg.spell_max_holds == 4
    assert cfg.spell_hold is False


def test_the_stop_event_carries_the_spelling_hold_count():
    from jarvis.events import RecordingStopped
    assert RecordingStopped().spell_holds == 0


@pytest.mark.parametrize("text,folded", [
    ("q-z-v", "qzv"), ("q z v", "qzv"), ("Q.Z.V", "Q.Z.V"),
    ("q. z. v.", "qzv"), ("q-z-v, k-b-w", "qzvkbw"),
])
def test_every_shape_whisper_writes_a_run_in_reads_the_same(text, folded):
    """Whisper punctuates a spelled run four different ways in one
    transcript; all four are one man saying letters, and all four HOLD.
    Three of the four fold to the bare letters. The fourth -- the
    initialism shape, dots with no space -- is the one shape that could
    also be a dot he SAID ("d.a.n"), so since 09-06 its dots are KEPT and
    read back as "dot" (section 13 below): a read-back he can refuse,
    never a silent pick. "q. z. v." folds to "qzv" and not "qzv." now:
    the full stop after the last spelled character is whisper's."""
    assert spelling.spelling_run(text) >= 3
    assert spelling.fold(text) == folded


# ------------------------------ (7) the continuation path that already exists
#
# The brief asked whether a headless fragment should be JOINED to the one
# before it, and whether the commander already has a path for that. It
# does, and no parallel one was added: an email request with no readable
# recipient parks a SendAsk ("Who should I send it to, sir?", jarvis/
# commander.py `_pending_sendask`) and the NEXT utterance is read as the
# answer through `_recipient_answer` -> `outbox.parse_address`. Since that
# parser now folds a spelled run, the answer may be spelled.
#
# What was missing on 09-05 was not the path. It was that his first three
# fragments never reached the commander at all -- one `handle` line for
# four `Transcribed` lines -- so no question was ever asked for the fourth
# to answer. Keeping the utterance in ONE capture is this branch's fix;
# why a discarded fragment is discarded belongs to the lane that owns it.
def test_a_spelled_address_answers_the_who_should_i_send_it_to_question():
    from jarvis import commander as commander_mod
    from jarvis import outbox
    for said in ("q-z-v at example dot com",
                 "it's q-z-v at example dot com",
                 "send it to q z v at example dot com"):
        who = commander_mod._recipient_answer(said)
        assert outbox.parse_address(who) == "qzv@example.com", said


def test_a_spelled_local_part_no_longer_resolves_to_its_last_letter():
    """THE SILENT WRONG-MAILBOX CASE. Before the fold, the parser matched
    _LOCAL_LABEL against the LAST single letter of a spelled run, so
    "q z v at example dot com" drafted v@example.com -- a different,
    possibly real address that a read-back would have said back almost
    right."""
    from jarvis import outbox
    assert outbox.parse_address("q z v at example dot com") != "v@example.com"
    assert outbox.parse_address("q z v at example dot com") == "qzv@example.com"


# ============================ (8) THE VERDICT'S TWO REPAIRS, and his two
#                                  rulings, taken 2026-09-05 after the
#                                  measured verdict on this branch.
#
# The verdict measured the branch a large net win and blocked on ONE
# number: 20 of 67 realistic non-spelling phrases were also held, each
# costing +1792 ms ONCE on that turn (2592 ms of quiet instead of 800).
# Ordinary sentences are not delayed by a millisecond -- that was measured
# and was never at issue.
#
# HIS RULING (A): KEEP the dangling-letter rule. He accepts ~1.8 s
# occasionally on a phrase ending in one bare letter, because that rule is
# what buys him the FIRST character when he starts spelling, and being cut
# off mid-address is worse than a pause.
#
# HIS RULING (B): "example.com" said as ONE WORD is a domain. It is the
# exact shape whisper wrote down for him on 09-05, and today it is refused
# with no re-ask at all.
#
# THE BUG THAT IS NOT A TRADE: the module's stated safety argument was
# "no ordinary sentence ends on two single characters in a row" -- false,
# because DIGITS were counted as run characters. The five phrases below
# are all runs of two under the old rule.

# The five the verdict named. Each was a false hold and none of them is a
# man spelling: the letter LABELS the number in front of it.
DIGIT_FALSE_HOLDS = ["gate 4 b", "row 2 a", "channel 5 c", "unit 2 d",
                     "he got a c"]


@pytest.mark.parametrize("phrase", DIGIT_FALSE_HOLDS)
def test_a_number_and_one_label_letter_is_not_a_spelling_run(phrase):
    """The measured repair: a run's two-or-more single characters must
    include at least TWO letters that are not "a" or "i". These five are
    11 of the 20 false holds, removed at zero cost to the spelled cases."""
    assert spelling.spelling_run(phrase) == 0


@pytest.mark.parametrize("phrase", DIGIT_FALSE_HOLDS)
def test_a_number_and_one_label_letter_is_not_held_either(monkeypatch, phrase):
    """The same five at the seam that costs the milliseconds."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial(phrase, 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec._spell_holds == 0
    assert rec.stops[0][2] < 1.0


# The spelled cases, all eight, that the narrowing must not touch.
SPELLED_CASES = [
    ("q-z-v", 3),
    ("q z v", 3),
    ("Q. Z. V.", 3),
    ("q. z. v", 3),
    ("q, z, v", 3),
    ("q-z-v, k-b-w-7", 7),           # his fourth fragment's own shape
    ("send an email to q-z-v", 3),
    ("send an email to q", 1),       # the dangling case: his FIRST fragment
]


@pytest.mark.parametrize("said,count", SPELLED_CASES)
def test_the_narrowing_costs_none_of_the_eight_spelled_cases(said, count):
    assert spelling.spelling_run(said) == count


def test_a_digit_may_still_ride_inside_a_spelled_run():
    """k-b-w-7 is a spelled string with a digit in it, not a label: the
    two-letter floor is on the RUN, not on every character in it."""
    assert spelling.spelling_run("k-b-w-7") == 4
    assert spelling.spelling_run("send it to q 7 z") == 3
    assert spelling.spelling_run("7-k-b") == 3


@pytest.mark.parametrize("phrase", ["switch to b", "the answer is c"])
def test_ruling_a_the_dangling_letter_rule_is_kept_on_purpose(phrase):
    """HIS RULING (A), 09-05. These two phrases DO cost him ~1.8 s once,
    and he ruled to keep them held anyway, because the same rule is what
    catches the first character of an address he is starting to spell.
    This is a decision, not a defect: do not "fix" it."""
    assert spelling.spelling_run(phrase) == 1


# ---------------------- (9) HIS RULING (B): a run-together domain -------
def test_a_run_together_domain_is_a_domain_now():
    """"dana at example.com" -- whisper's own transcript of a said domain
    -- was refused outright, which cost him not just the draft but the
    RE-ASK: unresolved_address had nothing to hand back either."""
    from jarvis import outbox
    assert outbox.parse_address("dana at example.com") == "dana@example.com"


def test_a_run_together_domain_reads_inside_a_sentence():
    from jarvis import outbox
    raw = "send it to dana at example.com please"
    span = outbox.address_span(raw)
    assert span is not None
    addr, start, end = span
    assert addr == "dana@example.com"
    assert raw[start:end] == "dana at example.com"


def test_a_run_together_domain_survives_a_trailing_full_stop():
    from jarvis import outbox
    assert outbox.parse_address("dana at example.com.") == "dana@example.com"


def test_a_spelled_local_part_with_a_run_together_domain_parses():
    """His fourth fragment as whisper actually wrote it: two spelled
    groups, whisper's comma, and the domain as ONE WORD. Both rulings in
    one string -- and it used to parse to nothing at all."""
    from jarvis import outbox
    assert outbox.parse_address("q-z-v, k-b-w-7, at example.com.") \
        == "qzvkbw7@example.com"


def test_the_read_back_of_a_run_together_domain_says_it_the_long_way():
    """What he HEARS back is unchanged -- the read-back speaks an address
    the same way it always did -- and the parser reads its own words."""
    from jarvis import outbox
    addr = outbox.parse_address("dana at example.com")
    assert outbox.spoken_address(addr) == "dana at example dot com"
    assert outbox.parse_address(outbox.spoken_address(addr)) == addr


def test_a_run_together_domain_is_masked_out_of_a_log_line():
    from jarvis import outbox
    assert outbox.mask_addresses("send it to dana at example.com") == \
        "send it to d… at example.com"


@pytest.mark.parametrize("prose", [
    "meet me at 4.30",
    "the meeting is at 3.5 tomorrow",
    "I'm at home. See you at six.",
    "we stopped at noon. Then we left",
    "look at me. In the morning",
    "version 3 dot 12 at the latest",
])
def test_a_full_stop_in_prose_is_still_not_a_domain(prose):
    """The guard on ruling (B): a domain read from a punctuated dot is
    read only when the dot is TIGHT (no space on either side) and the top
    level is alphabetic. A sentence boundary has a space after the stop,
    and "4.30" has no top level, so prose is untouched."""
    from jarvis import outbox
    assert outbox.parse_address(prose) == ""
    assert outbox.address_span(prose) is None


# ==================== (10) THE HOLE THE NARROWING OPENED, and the clause
#                           that closes it. MEASURED HERE, not handed down.
#
# The verdict sized the digit repair as "zero cost to any of 8 spelled-
# address cases", and that is TRUE -- all eight are q-z-v / k-b-w-7
# shapes. It is also narrower than it sounds, and a probe on this branch
# measured what the eight do not cover: of 24 INVENTED first names, 17
# have "a" or "i" as their SECOND letter, so the run "d a" (dana), "m i"
# (mike), "k a" (kate) has only ONE letter that counts and the capture
# closes after the second character. The dangling rule buys him the first
# character and the narrowing then throws it away -- which is the exact
# outcome his ruling (A) exists to prevent.
#
# THE CLAUSE: a run that BEGINS on a real letter directly after one of the
# closed-class words that cannot end a sentence is the dangling case
# CONTINUING -- the same utterance one character later -- and is held.
# It is not a new rule and it re-admits none of the five phrases the
# verdict named: every one of those begins its run on a digit or on "a".
NAMES_WITH_AN_A_OR_I_SECOND = ["d a", "m i", "k a", "l i", "r a", "n i"]


@pytest.mark.parametrize("first_two", NAMES_WITH_AN_A_OR_I_SECOND)
def test_the_dangling_case_continues_into_the_second_character(first_two):
    assert spelling.spelling_run("send it to " + first_two) == 2


def test_the_clause_carries_a_digit_in_the_third_position_too():
    assert spelling.spelling_run("send it to d a 4") == 3


@pytest.mark.parametrize("phrase", DIGIT_FALSE_HOLDS)
def test_the_clause_re_admits_none_of_the_five_the_verdict_named(phrase):
    """Each of the five begins its run on a digit ("gate 4 b") or on "a"
    ("he got a c"), and neither can open a spelling."""
    assert spelling.spelling_run(phrase) == 0


@pytest.mark.parametrize("phrase", [
    "we are in row b 4",       # "row" is a noun, not a function word
    "put it on plan b 2",      # his own counter-example, one char longer
    "he got a c",              # the run opens on "a"
    "gate 4 b",                # ...or on a digit
])
def test_the_clause_needs_the_function_word_in_front_of_the_run(phrase):
    assert spelling.spelling_run(phrase) == 0


def test_what_the_clause_costs_and_it_is_pinned_not_hidden():
    """The price, said out loud: a grid or seat reference whose letter
    follows a function word IS held, and costs ~1.8 s once on that turn.
    That is the same trade as his ruling (A) and the same size."""
    assert spelling.spelling_run("it's in b 4") == 2
    assert spelling.spelling_run("he's at c 5") == 2


def test_what_ruling_b_costs_is_measured_and_pinned():
    """The price of the run-together domain, said out loud. A verb before
    "at" and a domain after it drafts a mailbox out of prose -- and did
    ALREADY on jarvis-v3 for the spoken-dot shape; the tight dot adds the
    same sentence spelt the way whisper spells it, plus ONE new sub-shape:
    a file extension read as a top level. Both only ever reach a read-back
    (tests/test_send_file.py section 27). Change these rows knowingly."""
    from jarvis import outbox
    assert outbox.parse_address("have a look at example dot com") == "look@example.com"  # v3 did this
    assert outbox.parse_address("have a look at example.com") == "look@example.com"      # ruling (B)
    assert outbox.parse_address("have a look at notes.txt") == "look@notes.txt"          # the new sub-shape
    assert outbox.parse_address("the file is notes.txt") == ""      # no "at": never an address
    assert outbox.parse_address("open report.pdf") == ""
    assert outbox.parse_address("have a look at 4.30") == ""        # a digit top level never


# ==================== (11) THE FULL STOP AFTER THE LAST CHARACTER (09-06)
#
# The 09-06 adversary, finding C: whisper writes a full stop after the
# LAST spelled character as often as a comma -- "q. z. v. k. b. w. 7. at
# example.com", and "q-z-v, k-b-w-7. at example.com" (his 09-05 shape with
# "." where whisper wrote ","). The comma was consumed; the stop was not,
# so the parse gave "" AND unresolved_address gave "": no draft, no
# "I heard ..." re-ask, only the generic one -- and "no, send it to
# <that>" dropped the draft. Seven tests were red through the commander,
# typed and voice; tests/test_send_file.py section 28 has them. Here: the
# fold and the parse.
FULL_STOP_SHAPES = [
    "q. z. v. k. b. w. 7. at example.com",
    "q-z-v, k-b-w-7. at example.com",
    "q-z-v. at example.com",
    "Q-Z-V, K-B-W-7, at Example.com.",
]


@pytest.mark.parametrize("said", FULL_STOP_SHAPES)
def test_the_full_stop_after_the_last_spelled_character_is_whispers(said):
    from jarvis import outbox
    assert outbox.parse_address(said).lower().startswith(
        spelling.fold(said).split(" at ")[0].lower() + "@"), (said, spelling.fold(said))
    assert outbox.parse_address(said) != ""


def test_the_full_stop_shapes_parse_to_the_characters_he_said():
    from jarvis import outbox
    assert outbox.parse_address("q. z. v. k. b. w. 7. at example.com") == "qzvkbw7@example.com"
    assert outbox.parse_address("q-z-v, k-b-w-7. at example.com") == "qzvkbw7@example.com"
    assert outbox.parse_address("q-z-v. at example.com") == "qzv@example.com"
    assert outbox.parse_address("Q-Z-V, K-B-W-7, at Example.com.") == "QZVKBW7@Example.com"


def test_a_stop_glued_to_a_word_is_that_words_not_the_runs():
    """"q-z-v.txt" is a file name, not a spelled run with a stop after it;
    and a joiner glued to a longer token ("j-r.smith") is not a run at all
    -- including by backtracking to a shorter one, so a TYPED address is
    never folded into a different mailbox (the adversary caught
    "j.r.smith@example.com" folding to "jr.smith@")."""
    assert spelling.fold("q-z-v.txt") == "q-z-v.txt"
    assert spelling.fold("j.r.smith@example.com") == "j.r.smith@example.com"
    assert spelling.fold("send it to j.r.smith@example.com") == "send it to j.r.smith@example.com"
    assert spelling.fold("j-r.smith at example.com") == "j-r.smith at example.com"


# ============ (12) THE DOMAIN THAT IS STILL BEING SAID, and the OWED answer
def test_a_bare_at_after_a_run_is_still_mid_address():
    """"q-z-v-k-b-w-7 at" -- drawing breath for the domain -- holds the
    run in front of the "at"; so do "at example" (no top level yet) and
    "at example dot" (it is coming). A finished domain ends the run."""
    run = spelling.spelling_run
    assert run("q-z-v-k-b-w-7 at") == 7
    assert run("send it to q-z-v at") == 3
    assert run("q-z-v at example") == 3
    assert run("q-z-v at example dot") == 3
    assert run("q-z-v at example dot com") == 0
    assert run("q-z-v at example.com") == 0
    assert run("look at me") == 0
    assert run("what are you looking at") == 0
    assert run("meet me at 5") == 0


def test_a_spelled_domains_first_letter_is_the_dangling_case_not_a_label():
    """"dana at g" is the first letter of a domain being spelled, held by
    the dangling rule ("at" cannot end a sentence). The first attempt at
    this round cut the text at the "at" and returned 0 for it."""
    run = spelling.spelling_run
    assert run("dana at g") == 1
    assert run("dana at g-m-a") == 3
    assert run("send it to q-z-v at g") == 1


@pytest.mark.parametrize("said,count", [
    ("q", 1), ("no, q", 1), ("it's q", 1), ("d a", 2), ("a l", 2), ("a", 1),
    ("7", 1), ("q-z-v-k-b-w-7", 7), ("q. z. v.", 3),
])
def test_when_an_address_is_owed_a_bare_answer_holds_from_its_first_character(said, count):
    """The 09-06 adversary's finding B: spelling_run("q") is 0 by the
    ordinary rule, so the answer to "What is it?" ALWAYS split [1, 6] and
    drafted a plausible address missing its first letter. With the
    recorder told an address is owed, the question just asked IS the
    function word: no letter floor, no a/i exclusion."""
    assert spelling.spelling_run(said) in (0, count)
    assert spelling.spelling_run(said, address_owed=True) == count


@pytest.mark.parametrize("said", ["yes", "no", "okay", "never mind", "set a timer for 5",
                                  "send it", "no, 7"])
def test_an_owed_address_does_not_hold_the_ordinary_answers(said):
    """A yes, a no, a sentence: not held even while an address is owed.
    A bare digit after a word is a number, not a spelled character."""
    assert spelling.spelling_run(said, address_owed=True) == 0


def test_the_recorder_reads_owed_once_per_capture_from_the_apps_probe():
    """Read at start() through address_owed_probe and never mid-capture,
    so a question that expires while he is spelling cannot drop the hold
    under him; a bare Recorder owes nothing; a probe that raises said no."""
    src = inspect.getsource(Recorder.start)
    assert "self._address_owed = self._probe_address_owed()" in src
    rec = object.__new__(Recorder)
    assert rec._probe_address_owed() is False
    rec.address_owed_probe = lambda: True
    assert rec._probe_address_owed() is True

    def boom():
        raise RuntimeError("no commander")
    rec.address_owed_probe = boom
    assert rec._probe_address_owed() is False


def test_an_owed_capture_holds_the_bare_first_letter(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch)
    rec._address_owed = True
    rec.note_partial("q", 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is False and rec._spell_holds == 1
    rec = _speaks_then_pauses(monkeypatch)
    rec._address_owed = False
    rec.note_partial("q", 0.96, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec._spell_holds == 0


def test_the_commander_owes_an_address_only_while_it_has_asked_for_one(tmp_path, monkeypatch):
    """True while "What is it?" (a recipient SendAsk) or a read-back he
    may correct is live and unexpired; False for the account and file
    questions, for "Which one, sir?", and for nothing at all."""
    import time as _t
    from jarvis import outbox
    from jarvis.commander import Commander, SendAsk, SENDASK_TTL_S
    c = object.__new__(Commander)
    assert c.address_owed() is False
    c._pending_sendask = SendAsk(kind="recipient", said_file="f", who="Dana", hint="",
                                 made_at=_t.monotonic())
    assert c.address_owed() is True
    c._pending_sendask = SendAsk(kind="account", said_file="f", who="Dana", hint="",
                                 made_at=_t.monotonic())
    assert c.address_owed() is False
    c._pending_sendask = SendAsk(kind="recipient", said_file="f", who="Dana", hint="",
                                 made_at=_t.monotonic() - SENDASK_TTL_S - 1)
    assert c.address_owed() is False, "an expired question owes nothing"
    c._pending_sendask = None
    draft = outbox.Draft(path=tmp_path / "f.txt", size=1, mtime=0.0,
                         to_addr="d@example.com", to_name="", account={},
                         subject="", made_at=_t.monotonic())
    c._pending_send = draft
    assert c.address_owed() is True
    draft.made_at = _t.monotonic() - outbox.DRAFT_TTL_S - 1
    assert c.address_owed() is False


def test_the_app_installs_the_probe_and_reads_it_through_the_commander():
    import jarvis.app as app_mod
    src = inspect.getsource(app_mod.JarvisApp)
    assert "self.recorder.address_owed_probe = " in src
    a = object.__new__(app_mod.JarvisApp)
    assert a._address_owed(object()) is False               # a slim commander
    assert a._address_owed(type("C", (), {"address_owed": lambda self: True})()) is True

    class Boom:
        def address_owed(self):
            raise RuntimeError("x")
    assert a._address_owed(Boom()) is False


# ====================== (13) THE TIGHT DOT IS KEPT AND READ BACK AS "DOT"
#
# Default taken for him, 09-06. "d.a.n at example.com" was folded to
# "dan" and "j.r.smith at example.com" read as the last label, "smith":
# two silently different mailboxes, drafted, read back almost right, and
# sent on a yes. Whisper wrote his spoken "dot" as "." for the domain on
# 09-05, so it will for a local part too. Now the dots are KEPT: the wire
# carries d.a.n@ / j.r.smith@ and the read-back says every dot, so
# whichever he meant, his ear hears exactly what will be sent. What it
# costs: whisper's initialism shape for a run he spelled WITHOUT dots
# ("Q.Z.V.K.B.W.7") is read back with dots he never said -- a no and a
# re-spell, never a wrong mailbox.
@pytest.mark.parametrize("said,addr", [
    ("d.a.n at example.com", "d.a.n@example.com"),
    ("j.r.smith at example.com", "j.r.smith@example.com"),
    ("d.a.n at example dot com", "d.a.n@example.com"),
    ("send it to d.a.n at example.com please", "d.a.n@example.com"),
    ("Q.Z.V.K.B.W.7 at example.com", "Q.Z.V.K.B.W.7@example.com"),
])
def test_a_tight_dotted_local_part_is_read_whole_dots_and_all(said, addr):
    from jarvis import outbox
    assert spelling.fold(said).split(" at ")[0].endswith(addr.split("@")[0])
    assert outbox.parse_address(said) == addr
    spoken = outbox.spoken_address(addr)
    assert spoken.count(" dot ") == addr.count("."), spoken   # every dot is spoken
    assert outbox.parse_address(spoken) == addr               # and reads back to itself


def test_a_tight_dot_is_never_flattened_into_a_different_mailbox():
    from jarvis import outbox
    assert outbox.parse_address("d.a.n at example.com") != "dan@example.com"
    assert outbox.parse_address("j.r.smith at example.com") != "smith@example.com"


def test_a_spaced_dot_is_still_whispers_punctuation_of_letters():
    """"q. z. v" (dot AND space) folds as before: those are letters."""
    assert spelling.fold("q. z. v at example.com") == "qzv at example.com"
    assert spelling.fold("d. a. n at example.com") == "dan at example.com"


def test_the_tight_dot_maps_back_to_the_raw_text():
    from jarvis import outbox
    raw = "send it to d.a.n at example.com now"
    span = outbox.address_span(raw)
    assert span is not None
    addr, start, end = span
    assert addr == "d.a.n@example.com" and raw[start:end] == "d.a.n at example.com"


# ==================== (14) THE DECODE IN FLIGHT: the stop waits for it
#
# The seam that makes the pass's report land in time (finding A, the
# recorder side). tests/test_spelling_survival.py measures it on the real
# cadence; these pin the rule at the tick.
def test_a_decode_running_on_a_snapshot_that_covers_the_pause_holds_the_stop(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_decoding(1.0, rec.capture_id)           # snapshot past the last speech (0.96)
    _push(rec, 26)                                    # 0.83 s: the stop is due
    assert rec._check_endpoint() is False and rec.stops == []
    assert rec._spell_holds == 0, "a wait is not a hold and is not counted as one"
    rec.note_speculative("send it to q", 1.0, rec.capture_id)   # it lands: a run
    _push(rec, 1)
    assert rec._check_endpoint() is False and rec._spell_holds == 1


def test_a_decode_that_lands_on_a_word_lets_the_stop_go_on_the_next_tick(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_decoding(1.0, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is False
    rec.note_partial("what time is it", 1.0, rec.capture_id)
    _push(rec, 1)
    assert rec._check_endpoint() is True and rec._spell_holds == 0
    assert rec.stops[0][2] < 1.0


def test_a_decode_snapshotted_before_the_burst_ended_cannot_say_and_is_not_waited_for(monkeypatch):
    """A snapshot taken mid-word holds none of the letter that followed:
    waiting for it would wait for nothing. DECODE_COVER_SLACK_S is the
    tolerance: a chunk or three of the last character's decay."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_decoding(0.96 - recorder_mod.DECODE_COVER_SLACK_S - 0.05, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec.stops[0][2] < 1.0
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_decoding(0.96 - recorder_mod.DECODE_COVER_SLACK_S + 0.02, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is False


def test_the_wait_is_bounded_by_the_cap_when_the_decode_never_returns(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_decoding(1.0, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is False
    _push(rec, int(recorder_mod.DECODE_WAIT_MAX_S / (CHUNK / SAMPLE_RATE)) + 2)
    assert rec._check_endpoint() is True
    assert rec.stops[0][2] >= 0.8 + recorder_mod.DECODE_WAIT_MAX_S
    assert recorder_mod.DECODE_WAIT_MAX_S < 2.0 <= 2.5, "below the spelling hold and the energy timer"


def test_the_wait_is_off_with_the_spelling_hold_off(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch, spell_hold=False)
    rec.note_decoding(1.0, rec.capture_id)
    _push(rec, 26)
    assert rec._check_endpoint() is True and rec.stops[0][2] < 1.0


def test_a_decode_announced_by_a_finished_capture_is_not_waited_for(monkeypatch):
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_decoding(1.0, rec.capture_id - 1)
    _push(rec, 26)
    assert rec._check_endpoint() is True


def test_a_new_capture_forgets_a_decode_in_flight():
    rec = object.__new__(Recorder)
    rec._decoding = (3.0, 1.0)
    rec._decode_wait_key = 3.0
    rec._reset_filler_hold()
    assert rec._decoding is None and rec._decode_wait_key is None


def test_the_speculative_report_only_ever_adds_a_hold_and_clears_the_wait(monkeypatch):
    """Additive: a pass whose text does not end on a run stores nothing
    (its clean full decode is the one most likely to have dropped a
    trailing letter, and the greedy stays the one thing that clears a
    hold) -- but it does clear the decode-in-flight marker, whatever it
    said, or a pass that came back empty would hold the mic for the cap."""
    rec = _speaks_then_pauses(monkeypatch)
    rec.note_partial("send it to q-z-v", 0.96, rec.capture_id)
    rec.note_decoding(1.0, rec.capture_id)
    rec.note_speculative("", 1.0, rec.capture_id)
    assert rec._decoding is None
    assert rec._latest_partial[1] == 3, "the greedy's run is still what the hold reads"
    rec.note_speculative("send it to q-z-v-k", 1.05, rec.capture_id)
    assert rec._latest_partial[1] == 4 and rec._latest_partial[0] == ""
    rec.note_speculative("send it to q-z-v-k, um", 1.1, rec.capture_id)
    assert rec._latest_partial[1] == 4, "a filler from the pass is never stored"


def test_the_partial_loop_announces_every_decode_before_the_model_runs(monkeypatch):
    """Recorder.note_decoding is called with the buffer end and the
    capture stamp BEFORE transcriber.partial; note_partial after."""
    import jarvis.events as events_mod
    monkeypatch.setattr(events_mod.bus, "publish", lambda ev: None)
    order = []

    class Rec:
        recording = True
        capture_id = 7

        def snapshot_audio(self):
            return np.zeros(int(SAMPLE_RATE * 2.0), dtype=np.float32)

        def note_decoding(self, end_s, capture_id=None):
            order.append(("announce", end_s, capture_id))

        def note_partial(self, text, end_s, capture_id=None):
            order.append(("partial", text, end_s, capture_id))
            Rec.recording = False

    class Tr:
        def partial(self, audio):
            order.append(("decode",))
            return "send it to q"

    import jarvis.app as app_mod
    pipe = object.__new__(app_mod.JarvisApp)
    pipe.recorder, pipe.transcriber = Rec(), Tr()
    pipe._PARTIAL_INTERVAL_S, pipe._PARTIAL_MIN_S, pipe._PARTIAL_MAX_S = 0.01, 0.5, 1.0
    pipe._partial_loop()
    assert order == [("announce", 2.0, 7), ("decode",), ("partial", "send it to q", 2.0, 7)], order


def test_the_speculative_pass_announces_itself_and_always_reports_back(monkeypatch):
    """Announce before _decode_clip; report after it through
    note_speculative whatever came back -- an empty decode included."""
    import threading
    from types import SimpleNamespace
    import jarvis.app as app_mod
    import jarvis.events as events_mod
    monkeypatch.setattr(events_mod.bus, "publish", lambda ev: None)
    monkeypatch.setattr(recorder_mod.CONFIG, "endpoint_vad", True)
    monkeypatch.setattr(recorder_mod.CONFIG, "noise_gate", False)
    monkeypatch.setattr(recorder_mod.CONFIG, "speaker_verify", False)
    order = []

    class Rec(Recorder):
        def __init__(self):
            self.recording = True
            self.endpointer = SimpleNamespace(silence_since_speech=0.3, last_speech_seconds=1.5)
            self._record_rate = SAMPLE_RATE
            self._resample_to_16k = lambda a: a
            self._audio_frames = [np.zeros((SAMPLE_RATE * 2, 1), dtype=np.float32)]

        def note_decoding(self, end_s, capture_id=None):
            order.append(("announce", end_s, capture_id))

        def note_speculative(self, text, end_s, capture_id=None):
            order.append(("report", text, end_s, capture_id))

    for text in ("send it to q", ""):
        order.clear()
        rec = Rec()
        a = object.__new__(app_mod.JarvisApp)
        a.assistant = SimpleNamespace(get=lambda k, d=None: d, user_name="Hunter")
        a._init_assistant_state()
        a.recorder = rec
        a.speaker = SimpleNamespace(enrolled=False)
        a._spec_lock = threading.Lock()
        a.transcriber = SimpleNamespace(
            loaded=True,
            transcribe=lambda audio: SimpleNamespace(text=text, confidence=-0.2, accepted=True))
        a._owner_has_phrase = lambda: False
        assert a._maybe_speculate() is True
        assert order == [("announce", 2.0, rec.capture_id), ("report", text, 2.0, rec.capture_id)], order
