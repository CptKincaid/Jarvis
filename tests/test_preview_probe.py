"""The ghost-card instrument: it RECORDS, and it records shape, not speech.

2026-09-03: words repeated on the console transcript and stopped on their
own. Three diagnoses ran, none reached high confidence, and all three failed
for the same reason -- the live preview publishes to the screen through a
path with no log call, so the evidence never existed. The cause is still
unproven. These tests therefore pin the INSTRUMENT, not a fix: that a
repeat is detected and described, that the description is enough to
diagnose the next occurrence, that it holds none of what was said, and
above all that it changes nothing about what reaches the screen.

The load-bearing tests here are the ones that would fail if this ever
quietly turned into a filter (``test_observe_never_suppresses``,
``test_probe_failure_never_reaches_the_caller``) and the ones that would
fail if it started keeping his words (``test_record_holds_no_text``,
``test_fingerprint_is_salted_per_process``).
"""
import json

import pytest

from jarvis.previewprobe import (PATH_GREEDY, PATH_SPECULATIVE, PreviewProbe,
                                 PreviewRepeat, normalise)

# A fixed salt so fingerprints are stable WITHIN a test. Production uses a
# random one per process; see test_fingerprint_is_salted_per_process.
SALT = b"test-salt-not-the-production-one"

# Stand-in strings. Deliberately nothing he said, and nothing word-like that
# a reader could mistake for a transcript: the module must never need to
# know what the text is, so the tests must never depend on it either.
ALPHA = "alpha alpha"
BETA = "beta"


def _probe(**kw):
    """A probe with a manual clock and a captured emit."""
    now = {"t": 0.0}
    got = []
    kw.setdefault("salt", SALT)
    kw.setdefault("clock", lambda: now["t"])
    kw.setdefault("emit", got.append)
    return PreviewProbe(**kw), now, got


# --------------------------------------------------------------- detection

def test_below_threshold_reports_nothing():
    """Two identical previews are normal -- whisper re-decodes a growing
    buffer and often lands on the same words twice."""
    probe, _, got = _probe(repeat_threshold=3)
    assert probe.observe(ALPHA) is None
    assert probe.observe(ALPHA) is None
    assert got == []


def test_repeat_is_reported_at_the_threshold():
    probe, now, got = _probe(repeat_threshold=3)
    for i in range(3):
        now["t"] = i * 1.5
        rec = probe.observe(ALPHA, path=PATH_GREEDY, audio_s=0.9)
    assert isinstance(rec, PreviewRepeat)
    assert rec.count == 3
    assert rec.span_s == pytest.approx(3.0)     # first to last, not the window
    assert rec.path == PATH_GREEDY
    assert rec.words == 2
    assert rec.chars == len(ALPHA)
    assert rec.audio_s == pytest.approx(0.9)
    assert got == [rec]


def test_distinct_strings_do_not_pool():
    """A repeat is one string recurring, not "three previews happened"."""
    probe, _, got = _probe(repeat_threshold=3)
    for text in (ALPHA, BETA, "gamma"):
        assert probe.observe(text) is None
    assert got == []


def test_case_and_punctuation_variants_are_one_repeat():
    """Whisper alternates spellings across passes -- "Beta." / "beta" /
    " Beta" is one hallucination, not three, and a fingerprint that split
    on punctuation would have missed the 09-03 shape entirely."""
    probe, _, got = _probe(repeat_threshold=3)
    probe.observe(" Beta ")
    probe.observe("Beta.")
    rec = probe.observe("beta")
    assert rec is not None and rec.count == 3
    assert len({probe.fingerprint(t) for t in ("Beta.", " beta", "BETA")}) == 1


def test_emissions_outside_the_window_do_not_count():
    probe, now, got = _probe(repeat_threshold=3, window_s=10.0)
    now["t"] = 0.0
    probe.observe(ALPHA)
    now["t"] = 5.0
    probe.observe(ALPHA)
    now["t"] = 30.0                 # the first two have aged out
    assert probe.observe(ALPHA) is None
    assert got == []


def test_long_run_escalates_instead_of_one_line_per_emission():
    """A stuck preview must not write a log line every 0.9 s; it must
    still say the run is getting worse."""
    probe, now, got = _probe(repeat_threshold=3, window_s=1e6)
    for i in range(12):
        now["t"] = float(i)
        probe.observe(ALPHA)
    assert [r.count for r in got] == [3, 6, 12]


# ---------------------------------------------------------------- counters

def test_counters_expose_the_decode_to_emission_gap():
    """The number no 09-03 diagnosis could get: the preview decodes ~1/s
    for a whole capture, but only the passes that CHANGED the card were
    ever visible. decodes - emissions is that invisible remainder."""
    probe, _, _ = _probe()
    for _ in range(5):
        probe.decoded(PATH_GREEDY)
    probe.observe(ALPHA, path=PATH_GREEDY)
    probe.observe(BETA, path=PATH_GREEDY)
    probe.retracted(PATH_GREEDY)

    c = probe.counters()
    assert c["decodes"] == 5
    assert c["emissions"] == 2
    assert c["retractions"] == 1
    assert c["by_path"][PATH_GREEDY]["decodes"] == 5


def test_counters_separate_the_two_emitters():
    """Both paths publish the SAME PartialText event. Without the path tag
    a repeat on screen cannot be attributed after the fact -- which is
    exactly what cost the 09-03 diagnosis its confidence."""
    probe, _, _ = _probe()
    probe.observe(ALPHA, path=PATH_GREEDY)
    probe.observe(BETA, path=PATH_SPECULATIVE)
    c = probe.counters()
    assert c["by_path"][PATH_GREEDY]["emissions"] == 1
    assert c["by_path"][PATH_SPECULATIVE]["emissions"] == 1


def test_blank_previews_are_not_emissions():
    """PartialText("") is the RETRACTION, not something shown."""
    probe, _, got = _probe()
    assert probe.observe("") is None
    assert probe.observe("   ") is None
    assert probe.counters()["emissions"] == 0


# ----------------------------------------------------------------- privacy

def test_record_holds_no_text():
    """The whole point of shape. A record that carried the string would
    put unverified mic decodes -- text that never passed the speaker gate,
    and may not even be his -- into a file agents read."""
    probe, now, got = _probe(repeat_threshold=2)
    secret = "zzqqx wibble"
    now["t"] = 1.0
    probe.observe(secret)
    now["t"] = 2.0
    rec = probe.observe(secret)

    assert not hasattr(rec, "text")
    blob = json.dumps(rec.__dict__) + rec.line()
    for word in secret.split():
        assert word not in blob
    assert "zz" not in blob and "wib" not in blob


def test_jsonl_ledger_writes_shape_only(tmp_path):
    path = tmp_path / "previews.jsonl"
    probe = PreviewProbe(repeat_threshold=2, salt=SALT, jsonl_path=path)
    secret = "zzqqx wibble"
    probe.observe(secret, path=PATH_GREEDY, audio_s=1.25)
    probe.observe(secret, path=PATH_GREEDY, audio_s=1.25)

    raw = path.read_text()
    assert "zzqqx" not in raw and "wibble" not in raw
    rec = json.loads(raw.splitlines()[-1])
    assert rec["count"] == 2 and rec["path"] == PATH_GREEDY
    assert rec["words"] == 2 and rec["chars"] == len(secret)
    assert rec["audio_s"] == pytest.approx(1.25)
    assert "at" in rec                       # wall clock, for correlating


def test_fingerprint_is_salted_per_process():
    """An UNSALTED digest of a short word is reversible against a wordlist
    in seconds, so it would BE the word. Two probes must therefore disagree
    on the same text -- a digest that survived the process would be a
    persistent pseudonym for something he said."""
    a = PreviewProbe()
    b = PreviewProbe()
    assert a.fingerprint(BETA) != b.fingerprint(BETA)
    assert a.fingerprint(BETA) == a.fingerprint(BETA)    # stable within a run


def test_normalise_is_only_a_fingerprint_helper():
    assert normalise("  Beta,  BETA! ") == "beta beta"
    assert normalise("") == ""
    assert normalise(None) == ""


# --------------------------------------------- it must never become a gate

def test_observe_never_suppresses():
    """The contract. observe() reports; it does not decide. Its return
    value is advisory and every call is safe to ignore -- suppressing a
    preview on an unproven cause is the 2026-08-31 silent confidence gate
    that ate his commands, wearing a different hat."""
    probe, now, _ = _probe(repeat_threshold=2)
    published = []
    for i in range(6):
        now["t"] = float(i)
        probe.observe(ALPHA)        # result deliberately discarded
        published.append(ALPHA)     # the caller publishes regardless
    assert published == [ALPHA] * 6
    assert probe.counters()["emissions"] == 6


def test_probe_failure_never_reaches_the_caller():
    """A preview must never delay, disturb or fail the real transcription.
    A broken ledger or a broken emit is the instrument's problem alone."""
    def boom(_rec):
        raise RuntimeError("ledger on fire")

    probe = PreviewProbe(repeat_threshold=2, salt=SALT, emit=boom)
    probe.observe(ALPHA)
    assert probe.observe(ALPHA) is None      # swallowed, not raised
    probe.decoded()
    probe.retracted()
    probe.shown(ALPHA)
    assert probe.counters()["emissions"] == 3


def test_tracked_fingerprints_are_bounded():
    """A runaway preview loop must not grow the probe without limit."""
    from jarvis.previewprobe import MAX_TRACKED
    probe, now, _ = _probe(repeat_threshold=2, window_s=5.0)
    for i in range(MAX_TRACKED * 2):
        now["t"] = float(i)
        probe.observe(f"text-{i}")
        probe.observe(f"text-{i}")
    assert probe.counters()["tracked"] <= MAX_TRACKED + 2


def test_reset_clears_counts_and_history():
    probe, _, _ = _probe(repeat_threshold=2)
    probe.decoded()
    probe.observe(ALPHA)
    probe.reset()
    c = probe.counters()
    assert c["decodes"] == 0 and c["emissions"] == 0 and c["by_path"] == {}
    assert probe.observe(ALPHA) is None      # history gone, so not a repeat
