"""The spelling hold on the recorder's REAL poll cadence, with the app's
preview thread modelled as jarvis.app._partial_loop actually runs it.

WHY THIS FILE EXISTS (the 09-06 adversary, finding A). tests/test_spelling
_hold.py measured the hold with a harness that reported the live preview
every 0.9 s and nothing else, and it showed all seven letters landing in
one capture. The running app is not that: its speculative full decode
(JarvisApp._maybe_speculate) starts 0.3 s into EVERY pause, holds the
model for its own latency, and pushes the greedy preview back a whole
interval -- and until 09-06 only the greedy preview reported to the
recorder. Modelled here on the recorder's own tick rates against a
virtual clock, with the speculative decode times from his jarvis.log
(n=238, p50 0.24 s, p90 0.46 s) and the greedy latency swept 0.2-0.6 s
because it is not logged, the first letter after "send an email to"
survived 4 of 9 preview phases and the whole seven-character address at
his 0.99 s pace 0 of 9. The fixer's own harness could not see it.

The model is the adversary's (survival_model.py), taken over whole: the
VAD is a list of probabilities, the audio is zeros, every utterance is a
string typed here, every clock is virtual. Nothing opens a device, and
the two seams the fix added -- Recorder.note_decoding before every decode
and Recorder.note_speculative after the speculative one -- are driven
exactly as the app drives them, with switches to run the BEFORE shape.

THE BAR (set in the round-3 brief): the whole address survives in at
least 8 of 9 phases at every greedy latency with the speculative pass ON.
"""
from __future__ import annotations

import numpy as np
import pytest

import jarvis.recorder as recorder_mod
from jarvis.config import CONFIG
from jarvis.endpoint import CHUNK, SAMPLE_RATE, VoiceEndpointer
from jarvis.recorder import Recorder

CHUNK_S = CHUNK / SAMPLE_RATE
POLL_S = Recorder._POLL_S                 # the AudioLevel / endpoint tick
SIL_POLL_S = Recorder._SILENCE_POLL_S     # the energy timer's tick
PREVIEW_S = 0.9                           # JarvisApp._PARTIAL_INTERVAL_S
SPEC_AFTER_S = 0.3                        # JarvisApp._SPECULATE_AFTER_S
SPEC_POLL_S = 0.08                        # JarvisApp._SPECULATE_POLL_S

LETTERS = list("qzvkbw7")                 # invented, as in test_spelling_hold
LETTER_S = 0.32                           # his measured character
GAP_S = 0.99                              # his measured pause between two
PHASES = [round(0.1 * i, 1) for i in range(9)]   # where the 0.9 s preview clock starts
GREEDY = (0.2, 0.4, 0.6)                  # greedy latency: not logged, swept
SPEC = {"p50": 0.24, "p90": 0.46}         # speculative latency: his log, n=238


class Clock:
    """A virtual time.monotonic for the recorder."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class ScriptedVAD:
    def __init__(self, probs):
        self.probs, self.i = list(probs), 0

    def __call__(self, chunk):
        p = self.probs[min(self.i, len(self.probs) - 1)]
        self.i += 1
        return p

    def reset(self):
        self.i = 0


def _recorder(clock, ep, owed=False):
    rec = object.__new__(Recorder)
    rec.recording = True
    rec.endpointer = ep
    rec._ep_cursor = 0
    rec._record_rate = SAMPLE_RATE
    rec._record_start_time = clock.monotonic() - 10.0
    rec._audio_frames = []
    rec._stop_endpoint, rec._stop_dead_air = "", None
    rec._voice_stopped = False
    rec._followup = False
    rec._silence_start = None
    rec._loud_chunks = 0
    rec._speaker_silence_start = None
    rec.stops = []
    rec._reset_filler_hold()
    rec._address_owed = owed

    def stop(reason="manual", endpoint="", dead_air=None):
        rec.stops.append((reason, endpoint, dead_air, clock.monotonic()))
        rec.recording = False
    rec.stop = stop
    return rec


def run_capture(segments, *, phase=0.0, greedy_s=0.3, spec_s=0.24, park=True,
                spec_reports=True, announce=True, owed=False, max_s=60.0):
    """One capture. ``segments`` is [(text once this segment has ended,
    speech seconds, gap seconds)]. ``park``: the speculative pass runs
    (and parks the preview, as the app's does); ``spec_reports``: it
    reports through note_speculative; ``announce``: both decodes announce
    themselves through note_decoding first. BEFORE the fix is
    park=True, spec_reports=False, announce=False."""
    clock = Clock()
    recorder_mod.time = clock
    script, loud, seg_end = [], [], []
    for _text, sp, gp in segments:
        n_sp, n_gp = int(round(sp / CHUNK_S)), int(round(gp / CHUNK_S))
        script += [0.9] * n_sp
        loud += [True] * n_sp
        seg_end.append(len(script))
        script += [0.0] * n_gp
        loud += [False] * n_gp
    script += [0.0] * 5000
    loud += [False] * 5000
    rec = _recorder(clock, VoiceEndpointer(model=ScriptedVAD(script)), owed=owed)
    t0 = clock.monotonic()
    next_chunk = next_poll = next_sil = t0
    due, next_idle = t0 + phase, t0
    pending = None                  # (kind, lands_at, text, end_s, capture, started)
    last_spec_key = None
    pushed = 0
    last_speech_wall = None
    spec_runs = 0

    def text_so_far():
        done = sum(1 for e in seg_end if e <= pushed)
        return segments[done - 1][0] if done else ""

    while rec.recording and clock.monotonic() - t0 < max_s:
        now = clock.monotonic()
        if now >= next_chunk:                       # the audio callback
            next_chunk += CHUNK_S
            rec._audio_frames.append(np.zeros((CHUNK, 1), dtype=np.float32))
            is_loud = loud[min(pushed, len(loud) - 1)]
            if not is_loud:
                rec._loud_chunks = 0
                if rec._silence_start is None:
                    rec._silence_start = now
            else:
                rec._loud_chunks += 1
                if rec._loud_chunks >= 2:
                    rec._silence_start = None
                last_speech_wall = now
            pushed += 1
        if pending is not None and now >= pending[1]:   # a decode lands
            kind, _land, text, end_s, capture, started = pending
            pending = None
            if kind == "greedy":
                rec.note_partial(text, end_s, capture)
                due = now + max(0.05, PREVIEW_S - (now - started))
            else:
                if spec_reports:
                    rec.note_speculative(text, end_s, capture)
                due = now + PREVIEW_S               # the preview pushed back a whole interval
        if pending is None and now >= next_idle:        # the partial thread's loop
            ep = rec.endpointer
            gap, key = ep.silence_since_speech, ep.last_speech_seconds
            if park and gap is not None and key is not None and gap >= SPEC_AFTER_S \
                    and key != last_spec_key:
                last_spec_key = key
                spec_runs += 1
                end_s = pushed * CHUNK_S
                if announce:
                    rec.note_decoding(end_s, rec.capture_id)
                pending = ("spec", now + spec_s, text_so_far(), end_s, rec.capture_id, now)
            elif now >= due:
                end_s = pushed * CHUNK_S
                if announce:
                    rec.note_decoding(end_s, rec.capture_id)
                pending = ("greedy", now + greedy_s, text_so_far(), end_s, rec.capture_id, now)
            else:
                next_idle = now + SPEC_POLL_S
        if now >= next_poll:                        # Recorder._poll_loop
            next_poll += POLL_S
            if rec._check_endpoint():
                break
        if now >= next_sil:
            next_sil += SIL_POLL_S
            if rec._check_silence():
                break
        clock.sleep(0.004)
    heard = sum(1 for e in seg_end if e <= pushed)
    if rec.stops:
        reason, endpoint, dead_air, t_stop = rec.stops[0]
        quiet = (t_stop - last_speech_wall) if last_speech_wall else None
    else:
        reason, endpoint, quiet = "none", "", None
    return dict(reason=reason, endpoint=endpoint, quiet_s=quiet, heard=heard,
                spell_holds=rec._spell_holds, spec_runs=spec_runs)


def spelled_segments(prefix, letters=LETTERS, gap=GAP_S, tail=""):
    segs = []
    for i in range(len(letters)):
        text = (prefix + " " if prefix else "") + "-".join(letters[:i + 1])
        segs.append((text, LETTER_S, gap))
    if tail:
        segs.append((segs[-1][0] + tail, 0.9, 3.5))
    return segs


def sweep(prefix, *, park=True, spec_reports=True, announce=True, owed=False):
    """{cell: (first_letter_kept, whole_address_kept)} out of 9 phases."""
    segs = spelled_segments(prefix)
    table = {}
    for g in GREEDY:
        for sname, s in SPEC.items():
            if not park and sname == "p90":
                continue                       # no pass, no pass latency
            first = whole = 0
            for ph in PHASES:
                r = run_capture(segs, phase=ph, greedy_s=g, spec_s=s, park=park,
                                spec_reports=spec_reports, announce=announce, owed=owed)
                whole += r["heard"] == len(LETTERS)
                first += r["heard"] >= 2
            table[(g, sname if park else "off")] = (first, whole)
    return table


@pytest.fixture(autouse=True)
def _hold_on(monkeypatch):
    monkeypatch.setattr(CONFIG, "spell_hold", True)
    monkeypatch.setattr(CONFIG, "spell_hold_s", 2.0)
    monkeypatch.setattr(CONFIG, "spell_max_holds", 16)
    monkeypatch.setattr(CONFIG, "filler_hold", True)
    monkeypatch.setattr(CONFIG, "endpoint_vad", True)
    monkeypatch.setattr(CONFIG, "endpoint_silence", 0.8)
    monkeypatch.setattr(CONFIG, "silence_grace", 0.0)
    monkeypatch.setattr(CONFIG, "silence_timeout", 2.5)
    yield
    recorder_mod.time = __import__("time")


# ---------------------------------------------------------------- BEFORE
def test_before_the_speculative_pass_hid_the_letters_the_hold_needed():
    """The adversary's finding A, reproduced: with only the greedy
    preview reporting and the pass parking it, the first letter survives
    4 of 9 phases (3 at a 0.6 s decode) and the whole address never."""
    t = sweep("Send an email to", park=True, spec_reports=False, announce=False)
    assert t[(0.2, "p50")][0] <= 4 and t[(0.4, "p50")][0] <= 4 and t[(0.6, "p50")][0] <= 3
    assert all(whole == 0 for _first, whole in t.values()), t


def test_before_a_bare_spelled_answer_was_never_held_by_the_rule():
    """Finding B: spelling_run("q") is 0 with no function word in front of
    it, so the answer to "What is it?" split [1, 6] in every phase."""
    t = sweep("", park=True, spec_reports=False, announce=False, owed=False)
    assert all(first == 0 for first, _whole in t.values()), t


def test_the_pass_reporting_alone_was_not_enough_at_his_p90():
    """The first attempt at this round: the pass reports, no wait. The
    p90 pass lands ~40 ms after the 0.8 s stop, so the letter it carries
    is stopped past."""
    t = sweep("Send an email to", park=True, spec_reports=True, announce=False)
    assert t[(0.2, "p50")][1] >= 8 and t[(0.4, "p50")][1] >= 8
    assert t[(0.4, "p90")][1] <= 5 and t[(0.6, "p90")][1] <= 5, t


# ----------------------------------------------------------------- AFTER
def test_the_whole_address_survives_at_every_latency_with_the_pass_on():
    """THE BAR: at least 8 of 9 at every greedy latency, at both of his
    speculative latencies, speculative pass ON."""
    t = sweep("Send an email to")
    for cell, (first, whole) in t.items():
        assert first == 9 and whole >= 8, (cell, first, whole, t)


def test_a_bare_spelled_answer_is_held_from_its_first_character_when_owed():
    """Finding B closed: the question just asked is the function word."""
    t = sweep("", owed=True)
    for cell, (first, whole) in t.items():
        assert first == 9 and whole >= 8, (cell, first, whole, t)


def test_the_address_still_survives_with_the_pass_switched_off_by_config():
    """listening.speculative_stt off: the greedy preview is the only
    decoder, on its 0.9 s clock, and the wait covers the one in flight."""
    t = sweep("Send an email to", park=False)
    for cell, (first, whole) in t.items():
        assert whole >= 8, (cell, first, whole, t)


# ------------------------------------------------ what it costs, measured
def test_an_ordinary_sentence_is_never_held_and_the_wait_is_bounded():
    """No spelling hold on an ordinary turn in any phase; and where the
    stop waited on a decode, it waited at most DECODE_WAIT_MAX_S past the
    endpoint (the cap) -- measured worst 1.46 s of quiet against 0.88
    before, in the phase where a 0.6 s greedy and then the pass both
    held the model. That wait is time the real path spent anyway: it
    joins a pass in flight and queues behind a greedy one on the model
    lock, so the words reach the commander at the same moment."""
    worst = 0.0
    for text, sp in (("what time is it", 0.9), ("set a timer for five minutes", 1.4)):
        for ph in PHASES:
            for g in GREEDY:
                for s in SPEC.values():
                    r = run_capture([(text, sp, 4.0)], phase=ph, greedy_s=g, spec_s=s)
                    assert r["spell_holds"] == 0, (text, ph, g, s, r)
                    worst = max(worst, r["quiet_s"])
    assert worst <= 0.8 + recorder_mod.DECODE_WAIT_MAX_S + 2 * POLL_S, worst


def test_the_dangling_turn_he_ruled_to_keep_is_now_held_nearly_every_time():
    """His ruling (A), applied consistently: "switch to b" used to be
    held only when the preview happened to land in time (27 of 81 phase
    x latency cells); with the pass reporting it is held in all 81. The
    price per turn is unchanged (~1.8 s once, capped by the 2.5 s energy
    timer); what changed is that he pays it every time, not by chance."""
    held = 0
    worst = 0.0
    cells = 0
    for text in ("switch to b", "the answer is c", "set it to f"):
        sp = max(0.6, 0.055 * len(text))
        for ph in PHASES:
            for g in GREEDY:
                cells += 1
                r = run_capture([(text, sp, 4.0)], phase=ph, greedy_s=g, spec_s=0.24)
                held += bool(r["spell_holds"])
                worst = max(worst, r["quiet_s"])
    assert held >= cells - 2, (held, cells)
    assert worst <= 2.9, worst


def test_the_capture_still_ends_after_the_domain_and_the_cap_still_wins():
    r = run_capture(spelled_segments("Send an email to", tail=", at example.com."),
                    greedy_s=0.2, spec_s=0.24)
    assert r["heard"] == len(LETTERS) + 1 and r["reason"] == "silence", r
    # +~1 s on the end of a spelled address: the greedy note taken during
    # the domain still ends on the run and the pass (additive) cannot
    # clear it; the next greedy does. Pinned as a cost, not hidden.
    assert 1.5 <= r["quiet_s"] <= 2.0, r
    segs = [(" ".join(["send it to"] + ["q"] * (i + 1)), LETTER_S, 1.7) for i in range(60)]
    r = run_capture(segs, greedy_s=0.2, spec_s=0.24, max_s=120.0)
    assert r["spell_holds"] == CONFIG.spell_max_holds, r
    assert r["reason"] == "silence" and r["heard"] < 60, r
