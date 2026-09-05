"""The shadow-mode scorecard: can he read, in ten seconds, what the owner
gate WOULD have done to him?

Nothing here reads his log, his people book or his voiceprint. Every record
is synthetic, built by this file, and the parser is pinned against them.

The two questions the tests exist to hold down:

  * THE LEDGER MUST CARRY DECISIONS AND SCORES, NEVER WORDS. A gate ledger
    that leaked the sentence, the passphrase or the override code would be
    strictly worse than the prose log it replaces, because it would be
    machine-readable.
  * THE SCORECARD MUST NOT CLAIM TO KNOW WHO WAS SPEAKING. The log records
    what the gate decided, not the truth. Every test that counts a refusal
    also checks the output says so out loud.
"""
import json
import os
import time

import pytest

from jarvis import gate as gt
from jarvis import gateledger as gl
from jarvis import passphrase as pp
from jarvis.config import PATHS
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER, Person, Registry

FAKE_PHRASE = "xxx-not-a-real-phrase-xxx"
SENTENCE = "put the kettle on and tell me about the lab report"

MATCHED = {"total": 3, "matched": 2, "scores": [0.38, 0.41, 0.12]}
NO_MATCH = {"total": 3, "matched": 0, "scores": [0.11, 0.09, 0.12]}
NOT_RUNNING = {}


def _registry(tmp_path, *, phrase=False, known=False):
    r = Registry(path=tmp_path / "people.json")
    r.add_person(Person(label="hunter", name="Hunter", role=ROLE_OWNER,
                        voice=True))
    if known:
        r.add_person(Person(label="heather", name="Heather", role=ROLE_KNOWN,
                            face="heather", consent="typed"))
    if phrase:
        r.set_secret("hunter", "phrase_hash",
                     pp.hash_secret(pp.normalise_spoken(FAKE_PHRASE)))
    r.save()
    return r


def _gate(tmp_path, mode="shadow", record=None, **kw):
    opts = {"owner.mode": mode}
    return gt.OwnerGate(registry=_registry(tmp_path, **kw), owner="hunter",
                        get_option=lambda k, d=None: opts.get(k, d),
                        record=record)


# --------------------------------------------------- the gate writes a row
def test_every_gated_verdict_leaves_one_row(tmp_path):
    """One row per judged turn -- the prose log line cannot be counted."""
    rows = []
    g = _gate(tmp_path, record=rows.append)
    g.judge("voice", SENTENCE, stats=MATCHED)
    assert len(rows) == 1
    assert rows[0]["how"] == gt.HOW_VOICE
    assert rows[0]["role"] == ROLE_OWNER
    assert rows[0]["admit"] is True
    assert rows[0]["mode"] == "shadow"
    assert rows[0]["source"] == "voice"
    assert isinstance(rows[0]["at"], float)


def test_an_ungated_source_leaves_no_row(tmp_path):
    """The socket and the phone are exempt BY CONSTRUCTION; counting them
    would drown the voice verdicts the scorecard exists to weigh."""
    rows = []
    g = _gate(tmp_path, record=rows.append)
    g.judge("cli", SENTENCE, stats=MATCHED)
    assert rows == []


def test_a_shadow_refusal_is_recorded_as_would_have_refused(tmp_path):
    """The whole point of shadow: the row says what enforce WOULD have done
    while the turn was answered anyway."""
    rows = []
    g = _gate(tmp_path, record=rows.append)
    d = g.judge("voice", SENTENCE, stats=NO_MATCH)
    assert d.admit is True                       # shadow refuses nobody
    assert rows[0]["would_refuse"] is True
    assert rows[0]["how"] == gt.HOW_NOBODY


def test_an_enforced_refusal_is_recorded_too(tmp_path):
    rows = []
    g = _gate(tmp_path, mode="enforce", record=rows.append)
    d = g.judge("voice", SENTENCE, stats=NO_MATCH)
    assert d.admit is False
    assert rows[0]["admit"] is False
    assert rows[0]["mode"] == "enforce"


def test_a_turn_with_nothing_measuring_is_recorded_as_no_opinion(tmp_path):
    """"Nobody is here" and "nothing was running" are different sentences
    and the scorecard must never add them together."""
    rows = []
    g = _gate(tmp_path, record=rows.append)
    g.judge("voice", SENTENCE, stats=NOT_RUNNING)
    assert rows[0]["how"] == gt.HOW_BLIND
    assert rows[0]["voice_running"] is False
    assert rows[0]["face_running"] is False


def test_the_speaker_score_rides_along_as_a_number(tmp_path):
    """A number is exactly what he is allowed to have, and it is the only
    thing that can tell a near miss from a stranger."""
    rows = []
    g = _gate(tmp_path, record=rows.append)
    g.judge("voice", SENTENCE, stats=NO_MATCH)
    assert rows[0]["score"] == pytest.approx(0.12)


# ------------------------------------------------- the ledger carries no words
def test_the_row_never_carries_the_sentence(tmp_path):
    rows = []
    g = _gate(tmp_path, record=rows.append)
    g.judge("voice", SENTENCE, stats=MATCHED)
    blob = json.dumps(rows[0])
    for word in ("kettle", "lab report", SENTENCE):
        assert word not in blob


def test_the_row_never_carries_the_passphrase(tmp_path):
    """The one turn where a leak would be unforgivable."""
    rows = []
    g = _gate(tmp_path, mode="shadow", record=rows.append, phrase=True)
    d = g.judge("voice", FAKE_PHRASE, stats=NO_MATCH)
    assert d.consumed is True
    blob = json.dumps(rows[0])
    assert FAKE_PHRASE not in blob
    assert "not-a-real-phrase" not in blob
    assert rows[0]["consumed"] is True
    assert rows[0]["how"] == gt.HOW_PHRASE


def test_a_gate_fault_records_the_type_and_not_the_message(tmp_path):
    """`why` is code-authored everywhere else; the fault path is the one
    place a stray exception string could carry something it should not."""
    rows = []
    g = _gate(tmp_path, record=rows.append)

    def boom(*a, **k):
        raise RuntimeError(SENTENCE)

    g._judge = boom
    d = g.judge("voice", SENTENCE, stats=MATCHED)
    assert d.admit is True                       # a broken gate never refuses
    assert rows[0]["how"] == gt.HOW_FAULT
    assert "kettle" not in json.dumps(rows[0])
    assert "RuntimeError" in rows[0]["why"]


def test_a_broken_recorder_cannot_stop_him_talking(tmp_path):
    """Every other failure in this file resolves to an admit. So does this."""
    def boom(_rec):
        raise OSError("disk full")

    g = _gate(tmp_path, record=boom)
    d = g.judge("voice", SENTENCE, stats=MATCHED)
    assert d.admit is True
    assert d.who == "hunter"


# ------------------------------------------------------------- the file seam
def test_append_and_read_round_trip(tmp_path):
    p = tmp_path / "gate.jsonl"
    gl.append(p, {"at": 1.0, "how": "voice"})
    gl.append(p, {"at": 2.0, "how": "nobody"})
    assert [r["how"] for r in gl.read(p)] == ["voice", "nobody"]


def test_a_corrupt_line_is_skipped_not_fatal(tmp_path):
    p = tmp_path / "gate.jsonl"
    gl.append(p, {"at": 1.0, "how": "voice"})
    with open(p, "a", encoding="utf-8") as fh:
        fh.write("{not json\n\n")
    gl.append(p, {"at": 2.0, "how": "nobody"})
    assert len(gl.read(p)) == 2


def test_reading_a_ledger_that_does_not_exist_is_empty(tmp_path):
    assert gl.read(tmp_path / "nope.jsonl") == []


def test_append_never_raises_when_the_path_is_impossible(tmp_path):
    gl.append(tmp_path / "no" / "such" / "dir" / "g.jsonl", {"at": 1.0})


# ---------------------------------------------------------- the arithmetic
def _row(at, **kw):
    rec = {"at": float(at), "mode": "shadow", "source": "voice",
           "admit": True, "would_refuse": False, "consumed": False,
           "how": gt.HOW_VOICE, "role": ROLE_OWNER, "who": "hunter",
           "why": "voice named hunter", "voice_running": True,
           "face_running": False, "score": 0.41, "rescued": False}
    rec.update(kw)
    return rec


def _refusal(at, **kw):
    return _row(at, would_refuse=True, how=gt.HOW_NOBODY, role="unknown",
                who="", why="no leg named anyone enrolled", score=0.11, **kw)


NOW = 1_757_000_000.0
DAY = 86400.0


def test_the_six_counts_he_asked_for(tmp_path):
    rows = ([_row(NOW - 10 * i) for i in range(20)]              # 20 admits
            + [_refusal(NOW - 100 - 10 * i) for i in range(5)]   # 5 refusals
            + [_row(NOW - 500, how=gt.HOW_BLIND, role="unknown", who="",
                    voice_running=False, face_running=False, score=None)]
            + [_row(NOW - 600, consumed=True, how=gt.HOW_PHRASE)]
            + [_row(NOW - 700, role=ROLE_KNOWN, who="heather",
                    how=gt.HOW_FACE)])
    s = gl.summarise(rows, now=NOW, window_s=DAY)
    assert s.judged == 28
    assert s.admitted_owner == 20 + 1          # the phrase turn admits him too
    assert s.admitted_other == 1
    assert s.refused == 5
    assert s.no_opinion == 1
    assert s.phrase_turns == 1


def test_refusals_split_by_nobody_versus_out_of_scope(tmp_path):
    rows = [_refusal(NOW - 1), _refusal(NOW - 2),
            _row(NOW - 3, would_refuse=True, how=gt.HOW_FACE,
                 role=ROLE_KNOWN, who="heather", why="out of scope for known")]
    s = gl.summarise(rows, now=NOW, window_s=DAY)
    assert s.refused == 3
    assert s.refused_nobody == 2
    assert s.refused_scope == 1


def test_a_window_admit_is_counted_apart_because_it_admits_the_room(tmp_path):
    """GRANT_S is five minutes of answering ANYBODY as him. That is the
    other failure mode and it must not hide inside "admitted as him"."""
    rows = [_row(NOW - 1, how=gt.HOW_GRANT), _row(NOW - 2, how=gt.HOW_CODE),
            _row(NOW - 3)]
    s = gl.summarise(rows, now=NOW, window_s=DAY)
    assert s.window_admits == 2
    assert s.admitted_owner == 3


def test_the_window_excludes_what_is_older_than_it(tmp_path):
    rows = [_row(NOW - 10), _row(NOW - 2 * DAY), _row(NOW - 30 * DAY)]
    assert gl.summarise(rows, now=NOW, window_s=DAY).judged == 1
    assert gl.summarise(rows, now=NOW, window_s=7 * DAY).judged == 2
    assert gl.summarise(rows, now=NOW, window_s=90 * DAY).judged == 3


def test_a_refusal_he_answered_with_the_phrase_is_the_nearest_evidence(tmp_path):
    """He cannot be asked "was that you?" after the fact. But a refusal
    followed by him opening the floor by hand IS him saying it was."""
    rows = [_refusal(NOW - 400),
            _row(NOW - 390, consumed=True, how=gt.HOW_PHRASE),
            _refusal(NOW - 100)]                     # nothing followed this one
    s = gl.summarise(rows, now=NOW, window_s=DAY)
    assert s.refused == 2
    assert s.refused_then_he_opened == 1


def test_a_phrase_long_after_a_refusal_is_not_evidence_about_it(tmp_path):
    rows = [_refusal(NOW - 5000),
            _row(NOW - 100, consumed=True, how=gt.HOW_PHRASE)]
    s = gl.summarise(rows, now=NOW, window_s=DAY)
    assert s.refused_then_he_opened == 0


def test_the_modes_seen_are_reported_because_a_window_can_span_a_change(tmp_path):
    rows = [_row(NOW - 10, mode="shadow"), _row(NOW - 20, mode="shadow"),
            _row(NOW - 30, mode="enforce")]
    s = gl.summarise(rows, now=NOW, window_s=DAY)
    assert s.modes == {"shadow": 2, "enforce": 1}


def test_the_trend_is_buckets_not_one_lump(tmp_path):
    rows = ([_row(NOW - 10) for _ in range(3)]
            + [_refusal(NOW - 20)]
            + [_row(NOW - 2 * DAY) for _ in range(2)])
    s = gl.summarise(rows, now=NOW, window_s=7 * DAY, buckets=7)
    assert len(s.trend) == 7
    assert sum(b.judged for b in s.trend) == 6
    assert sum(b.refused for b in s.trend) == 1
    last = s.trend[-1]
    assert last.judged == 4 and last.refused == 1


def test_an_empty_window_summarises_to_zero_and_not_a_crash(tmp_path):
    s = gl.summarise([], now=NOW, window_s=DAY)
    assert s.judged == 0
    assert s.refused_rate is None                 # never 0/0 dressed as 0%


# ------------------------------------------------------------- the printout
def _render(rows, **kw):
    return gl.render(gl.summarise(rows, now=NOW, window_s=DAY, **kw), now=NOW,
                     window_s=DAY, path="<synthetic>")


def test_the_summary_says_the_counts_in_his_terms(tmp_path):
    text = _render([_row(NOW - 10) for _ in range(9)] + [_refusal(NOW - 20)])
    head = text.split("Detail")[0]
    assert "10" in head                            # turns judged
    assert "refus" in head.lower()
    assert "no opinion" in head.lower()


def test_the_printout_states_what_it_cannot_know(tmp_path):
    """The assumption goes in the OUTPUT, not a docstring he will not read."""
    text = _render([_refusal(NOW - 20)])
    assert "cannot" in text.lower()
    assert "who was really speaking" in text.lower()


def test_the_printout_names_no_threshold_it_cannot_justify(tmp_path):
    """There is no measured number that says "enforce is safe now", and
    inventing one is how he gets locked out of his own house."""
    text = _render([_row(NOW - 10)])
    assert "enforce" in text.lower()
    for invented in ("99.9%", "99%", "95%", "safe to switch"):
        assert invented not in text


def test_an_empty_ledger_prints_the_reason_and_not_a_table_of_zeroes(tmp_path):
    text = gl.render(gl.summarise([], now=NOW, window_s=DAY), now=NOW,
                     window_s=DAY, path="<synthetic>")
    assert "no verdict" in text.lower()
    assert "people.json" in text or "enrol" in text.lower()


def test_the_printout_never_contains_a_path_to_a_secret(tmp_path):
    text = _render([_row(NOW - 10), _refusal(NOW - 20)])
    for secret in ("assistant.json", "voiceprint", ".ssh", "room-sensors"):
        assert secret not in text


# ------------------------------------------------------------ the script
def test_the_script_runs_read_only_over_a_synthetic_ledger(tmp_path, capsys):
    import scripts.gate_scorecard as sc

    p = tmp_path / "gate.jsonl"
    for r in [_row(time.time() - 10), _refusal(time.time() - 20)]:
        gl.append(p, r)
    before = p.stat().st_mtime_ns
    assert sc.main(["--path", str(p), "--days", "7"]) == 0
    out = capsys.readouterr().out
    assert "2" in out
    assert p.stat().st_mtime_ns == before        # read-only, on his live box


def test_the_script_takes_hours_as_well_as_days(tmp_path, capsys):
    import scripts.gate_scorecard as sc

    p = tmp_path / "gate.jsonl"
    gl.append(p, _row(time.time() - 3600 * 5))
    assert sc.main(["--path", str(p), "--hours", "1"]) == 0
    assert "0" in capsys.readouterr().out


def test_a_missing_ledger_is_not_an_error_he_has_to_debug(tmp_path, capsys):
    import scripts.gate_scorecard as sc

    assert sc.main(["--path", str(tmp_path / "nope.jsonl")]) == 0
    assert "no verdict" in capsys.readouterr().out.lower()


def test_the_json_form_is_the_same_numbers(tmp_path, capsys):
    import scripts.gate_scorecard as sc

    p = tmp_path / "gate.jsonl"
    gl.append(p, _row(time.time() - 10))
    assert sc.main(["--path", str(p), "--json"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["judged"] == 1
    assert got["cannot_know"]


# ----------------------------------------------------------- the wiring
def test_the_ledger_lives_beside_the_turn_ledger():
    """Same directory, same shape, same reason: turns.jsonl made latency
    countable and this makes admission countable."""
    assert gl.DEFAULT_PATH.name == "gate.jsonl"
    assert gl.DEFAULT_PATH.parent.name == PATHS.LOG_DIR.name


def test_the_app_hands_the_gate_a_recorder():
    """A source pin, and it earns its place: the gate's ledger seam is
    optional, so a wiring that quietly went missing would leave shadow mode
    exactly as unreadable as it was before this branch -- with every test
    above still green, because they all inject their own recorder."""
    import inspect

    import jarvis.app as app_mod

    src = inspect.getsource(app_mod)
    assert "record=gateledger.writer(" in src
    assert 'PATHS.LOG_DIR / "gate.jsonl"' in src


def test_the_writer_appends_where_it_was_pointed(tmp_path):
    p = tmp_path / "gate.jsonl"
    w = gl.writer(p)
    w({"at": 1.0, "how": "voice"})
    w({"at": 2.0, "how": "nobody"})
    assert len(gl.read(p)) == 2


# ------------------------------------------------------- the walkthrough
# HE RUNS THE ENROLMENT HIMSELF. It is his identity data and nothing here
# may write a row into his people book -- so the walkthrough has to be in
# the setup document, spelled exactly as the CLIs spell it, or the three
# features that wait on it do not exist for him. These tests NEVER RUN a
# CLI: scripts/jarvis_people.py loads the REAL registry and the REAL
# assistant config the moment its arguments parse. The subcommands and
# flags are read out of the source instead, the idiom
# tests/test_knightfall_docs.py already uses.
import re                                                    # noqa: E402
from pathlib import Path                                     # noqa: E402

import jarvis                                                # noqa: E402

ROOT = Path(jarvis.__file__).parent.parent
DOC = ROOT / "docs" / "assistant-setup.md"
SECTION = "## 87."


def _section() -> str:
    text = DOC.read_text()
    assert SECTION in text, "no owner-gate walkthrough in the setup document"
    body = text.split(SECTION, 1)[1]
    return body.split("\n## ", 1)[0]


def _commands(section: str):
    """(script name, flags) for every command line in the section's shell
    blocks, backslash continuations joined."""
    out = []
    for block in re.findall(r"```bash\n(.*?)```", section, re.S):
        for line in block.replace("\\\n", " ").splitlines():
            m = re.search(r"scripts/(\w+)\.py(.*)$", line)
            if m:
                out.append((m.group(1), re.findall(r"--[a-z][a-z-]*",
                                                   m.group(2))))
    return out


def _flags(script: str):
    src = (ROOT / "scripts" / ("%s.py" % script)).read_text()
    return set(re.findall(r'add_argument\(\s*"(--[a-z][a-z-]*)"', src))


def _subcommands(script: str):
    src = (ROOT / "scripts" / ("%s.py" % script)).read_text()
    named = set(re.findall(r'add_parser\(\s*"([a-z-]+)"', src))
    for group in re.findall(r'for name in \(([^)]*)\)', src):
        named |= set(re.findall(r'"([a-z-]+)"', group))
    return named


def test_the_walkthrough_exists_and_names_the_gap_it_closes():
    s = _section()
    assert "people.json" in s
    assert "shadow" in s and "enforce" in s
    assert "gate.jsonl" in s
    assert "gate_scorecard.py" in s


def test_every_flag_in_the_walkthrough_is_one_the_cli_really_has():
    """The failure this stops: a walkthrough he follows literally, which
    dies on `unrecognized arguments` at the one step nobody else may take
    for him."""
    for script, flags in _commands(_section()):
        have = _flags(script)
        for flag in flags:
            assert flag in have, "%s has no %s" % (script, flag)


def test_the_enrolment_line_uses_the_real_subcommand():
    s = _section()
    assert "jarvis_people.py add hunter" in s
    assert "jarvis_people.py list" in s
    assert "add" in _subcommands("jarvis_people")
    assert "list" in _subcommands("jarvis_people")


def test_the_role_it_tells_him_to_type_is_a_role_the_cli_accepts():
    from jarvis.identity import ROLE_OWNER

    assert "--role %s" % ROLE_OWNER in _section()


def test_it_promises_no_threshold_it_cannot_measure():
    """He asked what number should make him comfortable. The honest answer
    is that there is not one, and it has to be IN the document rather than
    left for him to work out after he has been locked out once."""
    s = _section()
    assert "There is not one" in s
    for invented in ("99.9%", "99%", "95%", "98%",
                     "safe to switch", "guaranteed"):
        assert invented not in s


def test_it_says_the_ledger_carries_no_words():
    s = _section().lower()
    assert "passphrase" in s and "no words" in s


def test_it_does_not_tell_him_to_print_a_secret_bearing_file():
    """assistant.json holds app passwords and tokens. The document may name
    it -- owner.mode lives in it -- but must never hand him a `cat`."""
    for block in re.findall(r"```bash\n(.*?)```", _section(), re.S):
        assert "cat " not in block
        assert "assistant.json" not in block


def test_the_empty_printout_points_at_a_heading_that_exists():
    """A pointer to a section he cannot find is worse than none: the empty
    case is exactly the moment he is looking for the next step."""
    quoted = re.findall(r'"([^"]{20,})"', gl.EMPTY_HELP)
    assert quoted, "the empty help names no section"
    heads = [ln.lstrip("# ").strip() for ln in DOC.read_text().splitlines()
             if ln.startswith("#")]
    for title in quoted:
        assert any(title in h for h in heads), "no heading %r" % title


# ------------------------------------ what it does when the file is BAD
# THE PRINCIPLE THIS BLOCK ENFORCES. A scorecard that silently reports 0
# when it could not read something is worse than no scorecard at all,
# because he acts on it. MEASURED before this block existed: a MISSING
# file, a file where EVERY line is unparseable, a file it has no PERMISSION
# to read, and a --path pointing at a DIRECTORY all printed output that was
# BYTE-IDENTICAL to a healthy, quiet, empty ledger -- and the text he then
# read said "the usual reason is that nobody is enrolled ... Enrol yourself
# first", sending him back to redo enrolment he had already done.
def _healthy_empty(tmp_path):
    """The one input that really IS a quiet week: the ledger exists, it is
    readable, and it holds nothing. Everything else must not look like it."""
    p = tmp_path / "healthy.jsonl"
    p.write_text("")
    return p


def _card(path, *, now=None, window_s=None):
    """The whole printout for one path, built the way the script builds it,
    so a test cannot pass on a code path he never sees."""
    now = NOW if now is None else now
    window_s = DAY if window_s is None else window_s
    rep = gl.read_report(path)
    s = gl.summarise(rep, now=now, window_s=window_s)
    return gl.render(s, now=now, window_s=window_s, path=str(path))


def _above_the_fold(text):
    """What he actually reads: everything before the standing assumptions."""
    return text.split("What this cannot know")[0].lower()


def test_a_missing_ledger_does_not_read_as_a_quiet_week(tmp_path):
    text = _card(tmp_path / "nope.jsonl")
    assert "does not exist" in _above_the_fold(text)
    assert text != _card(_healthy_empty(tmp_path))


def test_a_missing_ledger_is_not_dressed_up_as_you_are_not_enrolled(tmp_path):
    """The specific harm: a typo in --path or a rotated file sent him back
    to redo enrolment. "I could not read that" must come FIRST."""
    low = _above_the_fold(_card(tmp_path / "nope.jsonl"))
    assert "does not exist" in low
    if "enrol" in low:
        assert low.index("does not exist") < low.index("enrol")


def test_a_ledger_of_rubbish_is_not_reported_as_an_empty_week(tmp_path):
    p = tmp_path / "gate.jsonl"
    p.write_text("{not json\nalso not json\n[1, 2, 3]\n")
    low = _above_the_fold(_card(p))
    assert "could not" in low or "unreadable" in low
    assert "3" in low                       # all three lines, counted
    assert _card(p) != _card(_healthy_empty(tmp_path))


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read anything")
def test_a_ledger_he_may_not_read_says_permission(tmp_path):
    p = tmp_path / "gate.jsonl"
    gl.append(p, _row(NOW - 10))
    p.chmod(0o000)
    try:
        text = _card(p)
    finally:
        p.chmod(0o600)
    assert "permission" in _above_the_fold(text)
    assert text != _card(_healthy_empty(tmp_path))


def test_a_path_that_is_a_directory_says_so(tmp_path):
    d = tmp_path / "not-a-ledger"
    d.mkdir()
    text = _card(d)
    assert "director" in _above_the_fold(text)
    assert text != _card(_healthy_empty(tmp_path))


def test_no_bad_input_is_byte_identical_to_a_healthy_quiet_ledger(tmp_path):
    """The measured defect, pinned as one assertion."""
    rubbish = tmp_path / "rubbish.jsonl"
    rubbish.write_text("{not json\n")
    adir = tmp_path / "adir"
    adir.mkdir()
    base = _card(_healthy_empty(tmp_path))
    for bad in (tmp_path / "missing.jsonl", rubbish, adir):
        assert _card(bad) != base, "%s reads as a quiet week" % bad.name


def test_lines_it_could_not_read_are_counted_where_he_will_see_them(tmp_path):
    """Partial corruption was silent: a 5-line file with 2 unreadable lines
    reported 3 turns and never mentioned the 2."""
    p = tmp_path / "gate.jsonl"
    gl.append(p, _row(NOW - 10))
    with open(p, "a", encoding="utf-8") as fh:
        fh.write("{torn\nalso torn\n")
    gl.append(p, _row(NOW - 20))
    gl.append(p, _row(NOW - 30))
    rep = gl.read_report(p)
    assert rep.lines_seen == 5 and rep.lines_dropped == 2
    low = _above_the_fold(_card(p))
    assert "2 of 5" in low


def test_rows_outside_the_window_are_not_dressed_as_nobody_enrolled(tmp_path):
    """A populated ledger and an empty window are not the same thing, and
    the advice for one is wrong for the other."""
    p = tmp_path / "gate.jsonl"
    for i in range(5):
        gl.append(p, _row(NOW - 30 * DAY - i))
    low = _above_the_fold(_card(p))
    assert "5" in low
    assert "outside" in low or "older" in low


def test_a_healthy_empty_ledger_still_gets_the_enrolment_advice(tmp_path):
    """The fix must not throw away the help that IS right for a real empty
    file -- it must stop giving it for the three cases where it is wrong."""
    low = _above_the_fold(_card(_healthy_empty(tmp_path)))
    assert "enrol" in low
    for trouble in ("could not", "does not exist", "permission", "director"):
        assert trouble not in low


def test_the_json_form_carries_the_trouble_too(tmp_path, capsys):
    """--json is for a plot, and a plot of zeroes he cannot read is the
    same lie in a different shape."""
    import scripts.gate_scorecard as sc

    assert sc.main(["--path", str(tmp_path / "nope.jsonl"), "--json"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["trouble"]
    assert "does not exist" in got["trouble"].lower()


def test_the_script_says_it_out_loud_on_a_missing_file(tmp_path, capsys):
    import scripts.gate_scorecard as sc

    assert sc.main(["--path", str(tmp_path / "nope.jsonl")]) == 0
    assert "does not exist" in capsys.readouterr().out.lower()


# ------------------------------- one utterance is one turn, not two rows
# MEASURED on the real app path: a clip the speaker filter dropped and a
# window rescued is judged TWICE -- once inside _gate_rescue_inner
# (rejected=True) and again at _process_audio's `verdict = self._gate_judge`
# once the rescue has cleared `rejected`. Two ledger rows, one thing he
# said. In SHADOW -- his live mode -- the rescue fires for the phrase and
# code windows, which is exactly the path he exercises the first time he
# tests Knightfall. So the headline number AND the "somebody else got in"
# figure both inflate on the first thing he tries.
def test_one_rescued_utterance_is_one_turn_not_two(tmp_path):
    rows = [_row(NOW - 10, how=gt.HOW_GRANT, rescued=True, turn="t1"),
            _row(NOW - 9.5, how=gt.HOW_GRANT, rescued=False, turn="t1")]
    s = gl.summarise(rows, now=NOW, window_s=DAY)
    assert s.judged == 1
    assert s.window_admits == 1
    assert s.admitted_owner == 1


def test_the_verdict_count_is_still_available_and_is_named_apart(tmp_path):
    """Both numbers are true and they are different questions. The one
    labelled "turns" must be turns."""
    rows = [_row(NOW - 10, how=gt.HOW_GRANT, rescued=True, turn="t1"),
            _row(NOW - 9.5, how=gt.HOW_GRANT, rescued=False, turn="t1")]
    s = gl.summarise(rows, now=NOW, window_s=DAY)
    assert s.verdicts == 2 and s.judged == 1
    text = gl.render(s, now=NOW, window_s=DAY, path="<synthetic>")
    assert "verdict" in text.lower()


def test_rows_with_no_turn_id_are_each_their_own_turn(tmp_path):
    """The safe degradation: a ledger written by a gate that did not stamp
    turns must never be merged by guesswork. Two rows, two turns."""
    rows = [_row(NOW - 10), _row(NOW - 9.5)]
    for r in rows:
        r.pop("turn", None)
    assert gl.summarise(rows, now=NOW, window_s=DAY).judged == 2


def test_two_real_turns_that_share_nothing_are_never_merged(tmp_path):
    rows = [_row(NOW - 10, turn="t1"), _row(NOW - 9.5, turn="t2")]
    assert gl.summarise(rows, now=NOW, window_s=DAY).judged == 2


def test_the_gate_stamps_both_verdicts_of_one_utterance_with_one_turn(tmp_path):
    """The seam, driven through the REAL gate: the app hands the same turn
    id to the rescue's judge and to the judge that follows it."""
    rows = []
    g = _gate(tmp_path, record=rows.append)
    g.judge("voice", "", stats=NO_MATCH, rejected=True, turn="t7")
    g.judge("voice", SENTENCE, stats=NO_MATCH, turn="t7")
    assert [r["turn"] for r in rows] == ["t7", "t7"]
    at = time.time()
    for i, r in enumerate(rows):
        r["at"] = at - 10 + i
    assert gl.summarise(rows, now=at, window_s=DAY).judged == 1


def test_the_app_stamps_one_turn_id_on_both_gate_calls():
    """A source pin. Without it the double count comes straight back and
    every test above still passes, because they all pass their own id."""
    import inspect

    import jarvis.app as app_mod

    proc = inspect.getsource(app_mod.JarvisApp._process_audio)
    assert "turn=tid" in proc, "_process_audio hands no turn id down"
    assert proc.count("turn=tid") >= 2, "only one of the two calls is stamped"
    for name in ("_gate_rescue", "_gate_rescue_inner", "_gate_judge"):
        src = inspect.getsource(getattr(app_mod.JarvisApp, name))
        assert "turn" in src, "%s drops the turn id" % name


# --------------------------- a verdict this reader does not understand
def test_a_leg_this_reader_does_not_know_is_counted_and_named(tmp_path):
    """MEASURED before the fix: 3 judged turns with two unknown legs printed
    33.3% / 0% / 0% / 0% and the missing 2 appeared NOWHERE. A future gate
    that adds a leg must not make this scorecard quietly lose turns."""
    rows = [_row(NOW - 10),
            _row(NOW - 11, how="sunglasses", role="unknown", who=""),
            _row(NOW - 12, how="sunglasses", role="unknown", who="")]
    s = gl.summarise(rows, now=NOW, window_s=DAY)
    assert s.judged == 3
    assert s.unclassified == 2
    text = gl.render(s, now=NOW, window_s=DAY, path="<synthetic>")
    assert "sunglasses" in text


def test_every_judged_turn_lands_in_exactly_one_line(tmp_path):
    """The invariant that makes the printed percentages honest."""
    rows = [_row(NOW - 10),
            _row(NOW - 11, role=ROLE_KNOWN, who="heather", how=gt.HOW_FACE),
            _refusal(NOW - 12),
            _row(NOW - 13, how=gt.HOW_BLIND, role="unknown", who="",
                 voice_running=False, face_running=False, score=None),
            _row(NOW - 14, how="a-leg-from-the-future", role="unknown",
                 who="")]
    s = gl.summarise(rows, now=NOW, window_s=DAY)
    assert (s.admitted_owner + s.admitted_other + s.refused + s.no_opinion
            + s.unclassified) == s.judged == 5


# ------------------------------------------- it must not overstate itself
def test_the_line_he_opened_the_floor_says_it_is_a_subset(tmp_path):
    """It is indented level with four MUTUALLY EXCLUSIVE lines but is a
    subset of "ANSWERED you", so it reads as a fifth sibling that does not
    add up. It has to say on the line that it is counted above."""
    text = _render([_row(NOW - 10, consumed=True, how=gt.HOW_PHRASE)])
    line = [ln for ln in text.splitlines()
            if "opened the floor" in ln.lower()]
    assert line, "the floor line is gone"
    assert "of the above" in line[0].lower() or "included" in line[0].lower()


def test_the_script_does_not_claim_more_about_itself_than_is_true():
    """MEASURED: a real run mkdir's a log directory, appends 215 bytes to
    /tmp/jarvis-adhoc/jarvis.log and shells out to nvidia-smi and ldconfig
    -- all side effects of importing jarvis.config. Nothing of his leaks and
    /tmp/vss_voice is untouched, but "it opens exactly one file ... it
    writes nothing" is FALSE, and a file whose whole value is honesty about
    itself must not overstate itself."""
    import scripts.gate_scorecard as sc

    doc = " ".join(sc.__doc__.split())
    assert "opens exactly one file" not in doc
    assert "writes nothing" not in doc
    assert "one file" in doc.lower()             # it still says what it reads


# --------------------------------- the walkthrough's face-leg step, fixed
def test_it_does_not_send_him_to_read_a_number_that_is_never_printed():
    """MEASURED: the substring "dim" appears ZERO times in
    scripts/face_enrol.py -- --status prints the gallery, the generations,
    the labels and the cohesion, and never an embedding width. Telling him
    to "note the embedding width" from it is a step he cannot complete."""
    src = (ROOT / "scripts" / "face_enrol.py").read_text().lower()
    assert "dim" not in src, "face_enrol.py prints a width after all"
    # Pin the TRUTH rather than the absence of a phrase: the paragraph that
    # sends him to --status has to say, in that paragraph, that --status
    # does not print a width -- otherwise the next reader re-adds the step.
    para = _section().split("face_enrol.py --status")[1].split("###")[0]
    assert "does not print" in para.lower() or "never prints" in para.lower()
    assert "note two things" not in para.lower()


def test_the_width_the_walkthrough_hands_him_matches_this_tree():
    """MEASURED: the doc said --face-dim 512 while jarvis/facegallery.py on
    this tree has EMBED_DIM 128. Follow it verbatim and app.py reports "the
    gallery is 128-D and hunter was enrolled at 512-D; re-enrol to bring the
    face leg back" -- a silently dead face leg, from a document."""
    from jarvis.facegallery import EMBED_DIM

    widths = {int(w) for w in re.findall(r"--face-dim\s+(\d+)", _section())}
    assert widths <= {EMBED_DIM}, \
        "the walkthrough hands him %r, this tree is %d-D" % (widths, EMBED_DIM)


def test_the_walkthrough_names_where_the_width_really_comes_from():
    """If it asks him for a width at all, it has to say which file states
    it, so the next model swap makes the document wrong out loud."""
    s = _section()
    if "--face-dim" in s:
        assert "facegallery" in s


def test_the_walkthrough_does_not_repeat_the_scripts_false_self_claim():
    s = " ".join(_section().split())
    assert "writes nothing" not in s
    assert "opens exactly one file" not in s


def test_the_walkthrough_tells_him_what_an_unreadable_ledger_looks_like():
    """§8 of the walkthrough is the empty case, and the empty case is now
    four different sentences. It must not promise only one."""
    s = _section().lower()
    assert "could not" in s or "unreadable" in s


def test_every_help_text_points_at_a_heading_that_exists():
    """The wider form of the pin already on EMPTY_HELP: any section title
    quoted by ANY of the reader's help texts has to be findable."""
    heads = [ln.lstrip("# ").strip() for ln in DOC.read_text().splitlines()
             if ln.startswith("#")]
    for text in gl.HELP_TEXTS:
        for title in re.findall(r'"([^"]{20,})"', text):
            assert any(title in h for h in heads), "no heading %r" % title
