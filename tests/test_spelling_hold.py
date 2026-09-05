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
    ("q-z-v", "qzv"), ("q z v", "qzv"), ("Q.Z.V", "QZV"),
    ("q. z. v.", "qzv."), ("q-z-v, k-b-w", "qzvkbw"),
])
def test_every_shape_whisper_writes_a_run_in_reads_the_same(text, folded):
    """Whisper punctuates a spelled run four different ways in one
    transcript; all four are one man saying letters."""
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
