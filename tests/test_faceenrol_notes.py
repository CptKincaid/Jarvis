"""Enrolment with a NOTE on every take, custom poses, other people -- and the
proof that none of it can grant a recognised face anything.

WHAT HE ASKED FOR, 2026-09-02: "add the enrollment option into Jarvis so i can
enroll others or add more ways for me be recognized (looking at my phone,
looking away etc) with a note on what i am doing in the take or something."

WHAT IS BEING PINNED HERE, and why each one is a promise that intention alone
cannot keep:

* **His live enrolment must survive the format change.**
  ``~/.aiws_trainer/face_gallery/gen-00001.npz`` was written at 23:29 on
  2026-09-02: 13 embeddings, label "hunter", ``_format`` 1, and NO note keys
  (verified by listing the file's key names, never its vectors). A notes
  feature that made that file unreadable would destroy the only enrolment he
  has, which is the exact shape of the incident jarvis/facegallery.py exists
  to answer. So notes are extra ``note_``/``yaw_`` keys at the SAME format
  number, and the first test below rebuilds his file key for key and reads it.
* **A note is only worth having if it answers a question later.** The
  question is "which pose is weak", so the notes are grouped, scored by
  cohesion and printed worst-first, in the report and in --status.
* **Coverage is measured per SIDE.** Every sample in his enrolment and in his
  verification carried a POSITIVE yaw -- he has no left-turn coverage at all
  -- and the existing pose_spread check cannot see that, because it counts
  ``abs(yaw)``. The two sides are counted separately and the missing one is
  said out loud.
* **A second person's data is theirs.** Enrolling somebody else takes THEIR
  typed consent, at the keyboard, and neither ``--yes`` nor ``--json`` can
  give it for them. Deleting them destroys every generation that holds them
  and leaves everyone else's alone.
* **AND THE RULING THAT DOES NOT MOVE: identity may REMOVE capability or ADD
  a name; it must never GRANT capability the existing gates do not already
  grant.** A second enrolled face can only ever make Jarvis more careful.
  Pinned three ways below: the wake matrix, the body anchor, and a scan of
  every reader of ``Attention.identity`` in the tree.

No test here opens a device, loads a model, creates a window, or looks at,
saves or asserts on the content of any frame.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from jarvis import enrolentry as ee
from jarvis import faceenrol as fe
from jarvis import visionrig as vr
from jarvis.eye import Attention, FaceIdentifier, SessionIdentity, resolve_wake
from jarvis.facegallery import FaceGallery, Take, cosine
from jarvis.facemodels import LIFECAM_CINEMA

from tests.test_faceenrol import (ENROL, PLAN_YAWS,  # noqa: F401
                                  _REAL_CONFIG_LOAD, FakeSource,
                                  ScriptedDetector, ScriptedRecogniser,
                                  base_vec, face_enrol, same_face, wire)


# ------------------------------------------------------------------ fakes
def his_generation(root: Path, n: int = 13, label: str = "hunter",
                   seed: int = 21) -> list:
    """A generation with EXACTLY the key set his live file has.

    Verified 2026-09-02 by listing the key NAMES of
    ~/.aiws_trainer/face_gallery/gen-00001.npz (never its vectors):
    ``_created_ns``, ``_format``, ``_reason`` and ``emb_hunter_0000`` ..
    ``emb_hunter_0012``. No note key, no yaw key. This is what a notes-aware
    build has to keep reading."""
    root.mkdir(parents=True, exist_ok=True)
    vecs = same_face(base_vec(seed), n, seed=seed + 1)
    arrays = {"emb_%s_%04d" % (label, i): v for i, v in enumerate(vecs)}
    arrays["_format"] = np.array([1])
    arrays["_created_ns"] = np.array([1756870140000000000])
    arrays["_reason"] = np.array(["face_enrol"])
    with open(root / "gen-00001.npz", "wb") as fh:
        np.savez(fh, **arrays)
    return vecs


def a_session(tmp_path, label="hunter", yaws=None, vectors=None, n=13):
    gallery = FaceGallery(root=tmp_path / "gallery")
    det = ScriptedDetector(yaws or PLAN_YAWS)
    rec = ScriptedRecogniser(vectors if vectors is not None
                             else same_face(base_vec(), n))
    session = fe.EnrolmentSession(gallery, label, LIFECAM_CINEMA, det, rec,
                                  fe.SampleLimits(min_conf=0.6))
    return gallery, session


def run(session, notes=(), n=None):
    """Offer ``n`` frames, cycling the notes. Returns the session."""
    source = FakeSource()
    notes = list(notes) or [""]
    for i in range(n if n is not None else len(PLAN_YAWS)):
        _ok, frame = source.read()
        session.offer(frame, "plan", note=notes[i % len(notes)])
    return session


# ----------------------------------------------- his data, before anything
def test_his_note_less_generation_still_loads_and_still_matches(tmp_path):
    """THE ONE THAT MATTERS. 13 embeddings, label hunter, no note keys,
    saved at 23:29 on 2026-09-02 and verified working (120 of 120 frames
    matched, cosine p50 0.739 against a 0.363 bar). If this test fails, the
    feature has destroyed his enrolment."""
    root = tmp_path / "gallery"
    vecs = his_generation(root)
    g = FaceGallery(root=root)
    assert g.load() is True
    assert g.loaded_generation == 1
    assert g.labels() == ("hunter",)
    assert g.total() == 13
    label, score = g.match(vecs[7])
    assert label == "hunter" and score > 0.999
    # ...and it is HONEST about carrying no record, rather than inventing one.
    takes = g.takes("hunter")
    assert len(takes) == 13
    assert all(t.note == "" and t.yaw_deg is None for t in takes)
    assert all(t.recorded is False for t in takes)


def test_a_note_less_generation_can_be_appended_to_and_only_the_new_takes_carry_notes(
        tmp_path):
    """Re-enrolling on top of his gallery must not have to throw it away to
    start recording poses."""
    root = tmp_path / "gallery"
    his_generation(root)
    g = FaceGallery(root=root)
    g.load()
    g.add("hunter", base_vec(77), note="looking at my phone", yaw_deg=48.0)
    g.save(reason="append")

    back = FaceGallery(root=root)
    assert back.load() is True and back.total() == 14
    takes = back.takes("hunter")
    assert [t.note for t in takes[:13]] == [""] * 13
    assert takes[13].note == "looking at my phone"
    assert takes[13].yaw_deg == pytest.approx(48.0)


def test_his_generation_still_matches_after_a_notes_aware_save(tmp_path):
    """A save from a notes-aware build must leave his 13 vectors byte for
    byte what they were -- the notes ride beside them, not through them."""
    root = tmp_path / "gallery"
    vecs = his_generation(root)
    g = FaceGallery(root=root)
    g.load()
    g.save(reason="rewrite")
    back = FaceGallery(root=root)
    back.load()
    for want, got in zip(vecs, back.embeddings("hunter")):
        assert cosine(want, got) > 0.99999


# ---------------------------------------------------------------- the notes
def test_notes_and_yaw_round_trip_through_a_save(tmp_path):
    g = FaceGallery(root=tmp_path / "g")
    for i, vec in enumerate(same_face(base_vec(3), 9, seed=4)):
        g.add("hunter", vec, note="looking at my phone" if i < 4 else "",
              yaw_deg=48.0 + i)
    g.save(reason="notes")
    back = FaceGallery(root=tmp_path / "g")
    assert back.load() is True
    takes = back.takes("hunter")
    assert [t.note for t in takes] == ["looking at my phone"] * 4 + [""] * 5
    assert [round(t.yaw_deg, 1) for t in takes] == \
        [48.0 + i for i in range(9)]


def test_a_take_with_no_note_writes_no_note_key(tmp_path):
    """The file a note-less enrolment produces stays the shape his is, so
    "his generation still loads" is not a claim about one direction only."""
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(5), 3, seed=6):
        g.add("hunter", vec)
    g.save(reason="plain")
    keys = sorted(np.load(g.path_for(1)).files)
    # _model joined the header on 2026-09-03 and is written on every save;
    # its ABSENCE is what identifies his pre-swap SFace generations, so it
    # can never be written empty. FORMAT is deliberately NOT bumped.
    assert keys == ["_created_ns", "_format", "_model", "_reason",
                    "emb_hunter_0000", "emb_hunter_0001", "emb_hunter_0002"]


def test_a_note_is_cleaned_before_it_is_stored(tmp_path):
    """A note is free text he speaks or types and then PASTES in a report, so
    it is squeezed to one printable line and capped."""
    g = FaceGallery(root=tmp_path / "g")
    g.add("hunter", base_vec(9), note="  looking\tat\nmy   phone \x00 ")
    g.add("hunter", base_vec(10), note="x" * 400)
    takes = g.takes("hunter")
    assert takes[0].note == "looking at my phone"
    assert len(takes[1].note) == 120


def test_a_non_finite_yaw_is_no_record_rather_than_a_wrong_one(tmp_path):
    g = FaceGallery(root=tmp_path / "g")
    g.add("hunter", base_vec(9), note="x", yaw_deg=float("nan"))
    assert g.takes("hunter")[0].yaw_deg is None


def test_forgetting_a_label_takes_its_notes_with_it(tmp_path):
    g = FaceGallery(root=tmp_path / "g")
    g.add("hunter", base_vec(1), note="lens")
    g.add("heather", base_vec(2), note="lens")
    assert g.forget("heather") == 1
    assert g.labels() == ("hunter",)
    assert g.takes("heather") == []
    assert [t.note for t in g.takes("hunter")] == ["lens"]


def test_a_degenerate_stored_vector_takes_its_note_with_it(tmp_path):
    """``_read`` drops a vector that cannot be a face. Its note must go with
    it, or every note after it describes the wrong embedding."""
    root = tmp_path / "g"
    root.mkdir()
    vecs = same_face(base_vec(11), 3, seed=12)
    arrays = {"emb_hunter_0000": vecs[0],
              "emb_hunter_0001": np.zeros(128, dtype=np.float32),
              "emb_hunter_0002": vecs[2],
              "note_hunter_0000": np.array(["first"]),
              "note_hunter_0001": np.array(["the degenerate one"]),
              "note_hunter_0002": np.array(["third"]),
              "_format": np.array([1]), "_created_ns": np.array([1]),
              "_reason": np.array(["hand made"])}
    with open(root / "gen-00001.npz", "wb") as fh:
        np.savez(fh, **arrays)
    g = FaceGallery(root=root)
    assert g.load() is True
    assert g.total() == 2
    assert [t.note for t in g.takes("hunter")] == ["first", "third"]


# -------------------------------------------------- which pose is weak
def test_the_notes_say_which_pose_is_weakest(tmp_path):
    """THE PAYOFF. A note is only worth storing if it answers a question
    later, and the question is "which pose is letting me down"."""
    base = base_vec(31)
    good = same_face(base, 8, seed=3)             # him, at his desk
    weak = same_face(base_vec(4242), 3, seed=9)   # a pose that scores badly
    g = FaceGallery(root=tmp_path / "g")
    for vec in good:
        g.add("hunter", vec, note="looking at my screen", yaw_deg=50.0)
    for vec in weak:
        g.add("hunter", vec, note="looking at my phone", yaw_deg=-40.0)
    rows = fe.note_rows(g.embeddings("hunter"), g.takes("hunter"))
    assert rows[0].note == "looking at my phone", [r.note for r in rows]
    assert rows[0].cohesion_p50 < rows[1].cohesion_p50
    assert rows[0].count == 3 and rows[1].count == 8
    assert fe.weakest_note(rows) == "looking at my phone"


def test_takes_with_no_note_are_counted_and_named_rather_than_hidden(tmp_path):
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(13), 5, seed=14):
        g.add("hunter", vec)
    rows = fe.note_rows(g.embeddings("hunter"), g.takes("hunter"))
    assert len(rows) == 1
    assert rows[0].note == fe.NO_NOTE
    assert rows[0].count == 5


# ------------------------------------------------------- the missing side
def test_coverage_counts_the_two_sides_separately(tmp_path):
    """His measured enrolment was ALL positive yaw. ``pose_spread`` counts
    abs(yaw) and cannot see that; this can."""
    takes = [Take(note="lens", yaw_deg=6.0), Take(note="lens", yaw_deg=2.0),
             Take(note="screen", yaw_deg=55.0),
             Take(note="screen", yaw_deg=48.0),
             Take(note="", yaw_deg=None)]
    cov = fe.coverage(takes)
    assert cov["frontal"] == 2
    assert cov["positive"] == 2
    assert cov["negative"] == 0
    assert cov["unrecorded"] == 1
    assert cov["recorded"] == 4


def test_a_gallery_with_one_side_only_says_which_side_is_missing(tmp_path):
    takes = [Take(note="lens", yaw_deg=4.0)] * 3 + \
            [Take(note="screen", yaw_deg=50.0)] * 3
    lines = fe.coverage_lines(fe.coverage(takes))
    text = "\n".join(lines)
    assert "no takes at all turned the OTHER way" in text
    assert "negative" in text


def test_missing_stations_asks_for_what_is_missing_not_what_he_has():
    """Better than another fixed list: the guidance reads the gallery."""
    takes = [Take(note="lens", yaw_deg=4.0)] * 3 + \
            [Take(note="screen", yaw_deg=50.0)] * 3
    plan = fe.missing_stations(takes)
    keys = [s.key for s in plan]
    assert keys == ["across"], keys
    assert plan[0].samples == fe.COVERAGE_WANT
    assert plan[0].yaw_hi < 0.0


def test_missing_stations_falls_back_to_the_measured_five_when_nothing_is_recorded():
    """His generation records nothing, so there is nothing to reason from --
    and the five stations were chosen against his measured geometry."""
    assert fe.missing_stations([Take()] * 13) == fe.DEFAULT_PLAN
    assert fe.missing_stations([]) == fe.DEFAULT_PLAN


def test_missing_stations_asks_for_nothing_when_the_coverage_is_complete():
    takes = ([Take(note="lens", yaw_deg=4.0)] * 3
             + [Take(note="screen", yaw_deg=50.0)] * 3
             + [Take(note="across", yaw_deg=-40.0)] * 3)
    assert fe.missing_stations(takes) == ()


def test_the_default_plan_carries_a_note_on_every_station():
    """The five stations he already runs are the first five notes he gets,
    for free -- so a first enrolment is not a note-less one."""
    assert all(s.note for s in fe.DEFAULT_PLAN)
    assert "lens" in fe.DEFAULT_PLAN[0].note


# ------------------------------------------------------------ custom poses
def test_a_custom_pose_becomes_a_station_with_his_words_on_it():
    plan = fe.custom_stations(["looking at my phone", "with my glasses off"],
                              samples=4)
    assert [s.note for s in plan] == ["looking at my phone",
                                      "with my glasses off"]
    assert [s.samples for s in plan] == [4, 4]
    assert plan[0].key == "phone" and plan[1].key == "glasses"
    # A named take makes no claim about head angle, so it judges none.
    assert plan[0].wants(0.0) and plan[0].wants(55.0) and plan[0].wants(-55.0)


def test_two_custom_poses_that_slug_the_same_stay_distinguishable():
    plan = fe.custom_stations(["looking away", "looking at my phone"])
    assert len({s.key for s in plan}) == 2


def test_an_empty_custom_pose_is_refused_rather_than_stored_blank():
    with pytest.raises(ValueError):
        fe.custom_stations(["   "])


def test_the_plan_chooser_prefers_his_words_over_both_scripts():
    plan, why = fe.choose_plan([Take()] * 13, poses=["looking at my phone"])
    assert [s.note for s in plan] == ["looking at my phone"]
    assert "named" in why


def test_the_plan_chooser_asks_for_the_gap_when_one_is_recorded():
    takes = [Take(note="lens", yaw_deg=4.0)] * 3 + \
            [Take(note="screen", yaw_deg=50.0)] * 3
    plan, why = fe.choose_plan(takes, poses=())
    assert [s.key for s in plan] == ["across"]
    assert "missing" in why


def test_the_plan_chooser_runs_the_five_when_the_gallery_records_nothing():
    plan, why = fe.choose_plan([Take()] * 13, poses=())
    assert plan == fe.DEFAULT_PLAN
    assert "no pose record" in why


# ------------------------------------------- the notes reach the report
def test_the_report_names_the_weakest_take(tmp_path):
    _g, session = a_session(tmp_path)
    run(session, notes=["looking at the lens"] * 6 + ["looking at my phone"])
    rep = session.report()
    text = "\n".join(rep.lines())
    assert "by take" in text
    assert "looking at my phone" in text
    assert "coverage" in text


def test_the_report_is_still_numbers_and_strings_only(tmp_path):
    _g, session = a_session(tmp_path)
    run(session, notes=["looking at my phone"])
    payload = session.report().to_dict()
    vr.assert_numbers_only(payload)
    assert payload["notes"]
    assert any(row[0] == "looking at my phone" for row in payload["notes"])


def test_every_sample_line_carries_the_note_it_was_taken_under(tmp_path):
    _g, session = a_session(tmp_path)
    run(session, notes=["looking at my phone"])
    lines = [s.line() for s in session.report().samples]
    assert all("looking at my phone" in ln for ln in lines)


def test_the_note_reaches_the_gallery_and_not_only_the_report(tmp_path):
    gallery, session = a_session(tmp_path)
    run(session, notes=["looking at my phone"])
    takes = gallery.takes("hunter")
    assert takes and all(t.note == "looking at my phone" for t in takes)
    assert all(t.yaw_deg is not None for t in takes)


# --------------------------------------------------- the command, end to end
def test_a_custom_pose_runs_end_to_end_and_is_stored(monkeypatch, tmp_path,
                                                     capsys):
    gallery, _feed = wire(monkeypatch, tmp_path)
    code = face_enrol.main(ENROL + ["--pose", "looking at my phone",
                                    "--pose", "looking away",
                                    "--pose", "leaning back",
                                    "--pose", "turned the other way",
                                    "--pose", "at the lens",
                                    "--pose-samples", "3"])
    out = capsys.readouterr().out
    assert code == 0, out
    back = FaceGallery(root=gallery.root)
    back.load()
    notes = {t.note for t in back.takes("hunter")}
    assert "looking at my phone" in notes
    assert "looking away" in notes
    assert "by take" in out


def test_status_says_which_take_is_weakest_and_opens_nothing(monkeypatch,
                                                             tmp_path,
                                                             capsys):
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    capsys.readouterr()
    monkeypatch.setattr(face_enrol, "build_feed", lambda *a, **k: (_ for _ in
                                                                  ()).throw(
        AssertionError("--status must not open a device")))
    code = face_enrol.main(["--status"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "by take" in out
    assert "looking at the lens" in out
    assert "coverage" in out


def test_status_json_with_notes_is_still_numbers_and_strings_only(
        monkeypatch, tmp_path, capsys):
    wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    capsys.readouterr()
    assert face_enrol.main(["--status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    vr.assert_numbers_only(payload)
    assert payload["result"]["notes"]


# ------------------------------------------------------------ other people
CONSENT = ["--label", "heather"]
# ENROL carries --auto, and somebody ELSE's enrolment refuses --auto by
# design: a station nobody has to press a key for is a station nobody has to
# be PRESENT for. So a second person's run takes the stations one Enter at a
# time, which the patched input() answers exactly as it answers the consent
# prompt.
OTHER = [a for a in ENROL if a != "--auto"] + CONSENT


@pytest.fixture(autouse=True)
def _consent_happens_at_a_terminal(monkeypatch):
    """Consent needs a real terminal at BOTH ends -- somebody typed it, and
    they could read what they were agreeing to -- and under pytest neither
    stream is one. Every test in this module that exercises the ceremony
    would otherwise be testing the pipe refusal by accident.

    The refusal itself is pinned by
    ``test_consent_cannot_be_typed_by_a_pipe``, which opts back out."""
    monkeypatch.setattr(face_enrol, "_isatty", lambda _stream: True)


def test_enrolling_somebody_else_needs_their_typed_consent(monkeypatch,
                                                           tmp_path, capsys):
    """Her biometric data is hers to agree to, not his."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_a: "")
    code = face_enrol.main(OTHER)
    out = capsys.readouterr().out
    assert code == 1, out
    assert gallery.generations() == []
    assert "CONSENT" in out
    assert "heather" in out


def test_consent_says_what_is_stored_and_how_to_delete_it(monkeypatch,
                                                          tmp_path, capsys):
    wire(monkeypatch, tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_a: "no")
    face_enrol.main(OTHER)
    out = capsys.readouterr().out
    assert "128" in out                       # what is stored
    assert "--delete --label heather" in out  # how to undo it
    assert "never" in out.lower()             # and what it does not grant


def test_yes_cannot_give_another_persons_consent(monkeypatch, tmp_path,
                                                 capsys):
    """--yes is his flag. Consent is not his to give."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_a: "")
    code = face_enrol.main(OTHER + ["--yes"])
    assert code == 1
    assert gallery.generations() == []
    assert "CONSENT" in capsys.readouterr().out


def test_json_cannot_give_another_persons_consent(monkeypatch, tmp_path,
                                                  capsys):
    """--json silences stdout, so the consent text nobody can see is a
    consent nobody gave."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    code = face_enrol.main(OTHER + ["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert gallery.generations() == []
    assert "consent" in json.dumps(payload).lower()


def test_a_second_person_does_not_disturb_the_first(monkeypatch, tmp_path,
                                                    capsys):
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    first = FaceGallery(root=gallery.root)
    first.load()
    his = [v.copy() for v in first.embeddings("hunter")]
    capsys.readouterr()

    wire(monkeypatch, tmp_path, vectors=same_face(base_vec(808), 13, seed=17))
    monkeypatch.setattr("builtins.input", lambda *_a: "heather")
    code = face_enrol.main(OTHER)
    out = capsys.readouterr().out
    assert code == 0, out

    back = FaceGallery(root=gallery.root)
    back.load()
    assert back.labels() == ("heather", "hunter")
    assert back.count("hunter") == len(his)
    for want, got in zip(his, back.embeddings("hunter")):
        assert cosine(want, got) > 0.99999


def test_a_second_person_is_judged_on_her_own_pool(monkeypatch, tmp_path,
                                                   capsys):
    """Her 13 takes are a cohesive pool of HER. Judging her against his would
    fail every enrolment that is not him."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    capsys.readouterr()
    wire(monkeypatch, tmp_path, vectors=same_face(base_vec(808), 13, seed=17))
    monkeypatch.setattr("builtins.input", lambda *_a: "heather")
    assert face_enrol.main(OTHER) == 0
    out = capsys.readouterr().out
    assert "[FAIL]" not in out
    assert "label      heather" in out


def test_re_enrolling_him_does_not_drop_her(monkeypatch, tmp_path, capsys):
    """A plain re-run writes a NEW generation from the samples it captured.
    Without loading the others first, that generation would hold him alone
    and she would silently stop being recognised."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    wire(monkeypatch, tmp_path, vectors=same_face(base_vec(808), 13, seed=17))
    monkeypatch.setattr("builtins.input", lambda *_a: "heather")
    assert face_enrol.main(OTHER) == 0
    capsys.readouterr()

    wire(monkeypatch, tmp_path, vectors=same_face(base_vec(5), 13, seed=19))
    assert face_enrol.main(ENROL) == 0
    back = FaceGallery(root=gallery.root)
    back.load()
    assert back.labels() == ("heather", "hunter")
    assert back.count("heather") == 13


def test_a_bad_label_is_refused_before_the_camera_is_opened(monkeypatch,
                                                            tmp_path, capsys):
    gallery, _feed = wire(monkeypatch, tmp_path, feed_guard=True)
    code = face_enrol.main(ENROL + ["--label", "Heather Smith!"])
    out = capsys.readouterr().out
    assert code == 1
    assert gallery.generations() == []
    assert "label" in out


def test_enrolling_somebody_else_is_refused_while_sensing_says_no(
        monkeypatch, tmp_path, capsys):
    gallery, _feed = wire(monkeypatch, tmp_path, camera=False,
                          feed_guard=True)
    monkeypatch.setattr("builtins.input", lambda *_a: "heather")
    code = face_enrol.main(OTHER)
    out = capsys.readouterr().out
    assert code == 2
    assert "sensing says the camera may not run" in out
    assert gallery.generations() == []


# ------------------------------------------------------- deleting a person
def test_deleting_one_person_removes_only_that_person(tmp_path):
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 6, seed=2):
        g.add("hunter", vec, note="lens", yaw_deg=4.0)
    g.save(reason="one")
    for vec in same_face(base_vec(999), 6, seed=3):
        g.add("heather", vec, note="lens", yaw_deg=4.0)
    g.save(reason="two")

    out = g.purge_label("heather", reason="test")
    assert out["removed"] >= 1
    back = FaceGallery(root=tmp_path / "g")
    assert back.load() is True
    assert back.labels() == ("hunter",)
    assert back.count("hunter") == 6


def test_deleting_a_person_takes_them_out_of_the_OLDER_generations_too(
        tmp_path):
    """A delete that leaves her in generation 2 is not a delete: --rollback
    brings her back, and her embeddings are still on the disk."""
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 6, seed=2):
        g.add("hunter", vec)
    g.save(reason="one")
    for vec in same_face(base_vec(999), 6, seed=3):
        g.add("heather", vec)
    g.save(reason="two")
    g.save(reason="three")

    g.purge_label("heather", reason="test")
    for gen in FaceGallery(root=tmp_path / "g").generations():
        one = FaceGallery(root=tmp_path / "g")
        one.load(generation=gen)
        assert "heather" not in one.labels(), \
            "generation %d still holds her" % gen
    for path in (tmp_path / "g").iterdir():
        assert b"heather" not in path.read_bytes()


def test_deleting_the_only_person_leaves_nothing_behind(tmp_path):
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 6, seed=2):
        g.add("heather", vec)
    g.save(reason="one")
    out = g.purge_label("heather", reason="test")
    assert out["left"] == 0
    assert FaceGallery(root=tmp_path / "g").load() is False


def test_deleting_a_person_who_is_not_enrolled_destroys_nothing(tmp_path):
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 6, seed=2):
        g.add("hunter", vec)
    g.save(reason="one")
    out = g.purge_label("heather", reason="test")
    assert out["removed"] == 0
    assert FaceGallery(root=tmp_path / "g").generations() == [1]


def test_the_command_deletes_one_person_and_says_what_went(monkeypatch,
                                                           tmp_path, capsys):
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    wire(monkeypatch, tmp_path, vectors=same_face(base_vec(808), 13, seed=17))
    monkeypatch.setattr("builtins.input", lambda *_a: "heather")
    assert face_enrol.main(OTHER) == 0
    capsys.readouterr()

    code = face_enrol.main(["--delete", "--label", "heather", "--yes"])
    out = capsys.readouterr().out
    assert code == 0, out
    back = FaceGallery(root=gallery.root)
    back.load()
    assert back.labels() == ("hunter",)
    assert "heather" in out


def test_deleting_one_person_leaves_the_identity_flag_alone(monkeypatch,
                                                            tmp_path, capsys):
    """``camera.identity`` is HIS switch on the whole feature. Removing
    somebody else must not turn his own recognition off."""
    wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    wire(monkeypatch, tmp_path, vectors=same_face(base_vec(808), 13, seed=17))
    monkeypatch.setattr("builtins.input", lambda *_a: "heather")
    assert face_enrol.main(OTHER) == 0
    capsys.readouterr()
    assert face_enrol.main(["--delete", "--label", "heather", "--yes"]) == 0
    cfg = face_enrol.AssistantConfig.load()
    assert bool(cfg.get("camera.identity", False)) is True


def test_deleting_one_person_asks_for_their_name_first(monkeypatch, tmp_path,
                                                       capsys):
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    wire(monkeypatch, tmp_path, vectors=same_face(base_vec(808), 13, seed=17))
    monkeypatch.setattr("builtins.input", lambda *_a: "heather")
    assert face_enrol.main(OTHER) == 0
    capsys.readouterr()
    monkeypatch.setattr("builtins.input", lambda *_a: "no")
    code = face_enrol.main(["--delete", "--label", "heather"])
    out = capsys.readouterr().out
    assert code == 1
    back = FaceGallery(root=gallery.root)
    back.load()
    assert "heather" in back.labels()
    assert "Not deleted" in out


# ------------------------------------------------ NOTHING IS GRANTED, EVER
def test_a_recognised_second_person_wakes_nothing(tmp_path):
    """HIS RULING. A face may REMOVE capability or ADD a name. Enrolling
    Heather must never make Jarvis answer Heather."""
    att = Attention(faces=1, attending=True, dwell_s=1.0, identity="heather",
                    id_score=0.9, age_s=0.1)
    out = resolve_wake("suppress (score 0.18)", False, att, owner="hunter")
    assert out.ok is False
    assert out.verdict == "suppress (score 0.18)"


def test_enrolling_a_second_person_can_only_TAKE_a_promotion_AWAY(tmp_path):
    """The matrix, so the direction is a property and not an anecdote: an
    anonymous face promotes, HE promotes, and SHE does not -- which is
    strictly less than the anonymous case, never more."""
    def promoted(identity):
        att = Attention(faces=1, attending=True, dwell_s=1.0,
                        identity=identity, id_score=0.9, age_s=0.1)
        return resolve_wake("suppress (score 0.18)", False, att,
                            owner="hunter").ok
    assert promoted("") is True            # nobody enrolled: today's behaviour
    assert promoted("hunter") is True      # him: exactly the same outcome
    assert promoted("heather") is False    # her: strictly less


def test_a_recognised_second_person_never_anchors_the_body(tmp_path):
    """The body anchor carries an identity forward while he faces his
    monitor. Anchoring on her would let a re-identification model -- which
    mostly encodes CLOTHING -- speak for a person it cannot identify."""
    gallery = FaceGallery()
    her = base_vec(808)
    gallery.add("heather", her, note="lens")

    class Rec:
        def embed(self, frame, row):
            return her

    session = SessionIdentity(match_min=0.75)
    ident = FaceIdentifier(gallery, Rec(), min_conf=0.6, owner="hunter",
                           session=session)
    row = np.zeros(15, dtype=np.float32)
    row[14] = 0.9
    label, score = ident.identify(None, row, body_vec=base_vec(3))
    assert label == "heather" and score > 0.99
    assert session.identify(base_vec(3)) == ("", 0.0), \
        "her face anchored the body"


def test_a_second_label_cannot_become_the_owner_label(tmp_path):
    """``owner`` is read from his config, never from the gallery: the set of
    enrolled names may grow without the set of privileged names growing."""
    gallery = FaceGallery()
    gallery.add("heather", base_vec(808))

    class Rec:
        def embed(self, frame, row):
            return base_vec(808)

    ident = FaceIdentifier(gallery, Rec(), min_conf=0.6, owner="hunter")
    assert ident.status()["owner"] == "hunter"
    row = np.zeros(15, dtype=np.float32)
    row[14] = 0.9
    assert ident.identify(None, row)[0] == "heather"
    assert ident.status()["owner"] == "hunter"


def test_only_eye_reads_the_identity_at_all():
    """TRACE EVERY CONSUMER, mechanically. ``Attention.identity`` is the
    only thing a recognised face produces, and the only code allowed to read
    it is the wake fusion in jarvis/eye.py -- which uses it to WITHHOLD a
    promotion. A new reader somewhere else is how "it may never grant
    capability" would quietly stop being true, so the tree is scanned rather
    than reasoned about.

    If this fails, read the new line: it is only allowed to make Jarvis do
    LESS."""
    root = Path(__file__).resolve().parent.parent / "jarvis"
    rx = re.compile(r"\b(?:eye|att|attention|latest|reading|obs)\.identity\b")
    found = set()
    for path in sorted(root.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if rx.search(line):
                found.add(path.name)
    assert found == {"eye.py"}, sorted(found)


def test_the_wake_fusion_still_spells_the_rule_the_way_the_rule_reads():
    """The one line the whole ruling rests on, pinned as text: a recognised
    name that is not his must not be in the set that vouches."""
    src = (Path(__file__).resolve().parent.parent / "jarvis" / "eye.py"
           ).read_text()
    assert 'eye.identity in ("", owner)' in src


# --------------------------------------------------- the way in, from Jarvis
def test_the_voice_entry_point_opens_no_camera_and_hands_over_the_command(
        tmp_path):
    """IN-APP ENTRY, SAID PLAINLY: this does not capture inside the window.
    The guided run needs the lens and a keyboard, and three other branches
    are in jarvis/ui right now, so what Jarvis does is hand over the exact
    command.

    THE CLIPBOARD IS THE CONVENIENCE, NOT THE DELIVERY -- changed 2026-09-03
    after Jarvis claimed a clipboard write that had not survived to his
    paste. The command now always rides in ``display_only``, which the
    console shows and the TTS never reads. See
    tests/test_clipboard_claim.py."""
    gallery = FaceGallery(root=tmp_path / "g")
    clipped = []
    out = ee.enrol_answer(gallery, "hunter", owner="hunter",
                          clipboard=lambda text: clipped.append(text) or True)
    assert "face_enrol.py" in out["command"]
    assert clipped == [out["command"]]
    assert out["display_only"] == out["command"]
    assert out["reply"]
    assert "clipboard" in out["reply"].lower()


def test_the_entry_point_says_the_consent_step_before_he_starts(tmp_path):
    gallery = FaceGallery(root=tmp_path / "g")
    out = ee.enrol_answer(gallery, "heather", owner="hunter",
                          clipboard=lambda text: True)
    assert "--label heather" in out["command"]
    assert "consent" in out["reply"].lower()
    assert "heather" in out["reply"].lower()


def test_the_entry_point_carries_his_named_poses_into_the_command(tmp_path):
    gallery = FaceGallery(root=tmp_path / "g")
    out = ee.enrol_answer(gallery, "hunter", owner="hunter",
                          poses=["looking at my phone"],
                          clipboard=lambda text: True)
    import shlex
    assert shlex.split(out["command"])[-2:] == ["--pose",
                                                "looking at my phone"]


def test_the_command_the_entry_point_hands_over_actually_parses(tmp_path):
    """A command he cannot run is worse than no command."""
    gallery = FaceGallery(root=tmp_path / "g")
    out = ee.enrol_answer(gallery, "heather", owner="hunter",
                          poses=["looking at my phone"],
                          clipboard=lambda text: True)
    import shlex
    argv = shlex.split(out["command"])
    assert argv[1].endswith("face_enrol.py")
    args = face_enrol.build_parser().parse_args(argv[2:])
    assert args.label == "heather"
    assert args.pose == ["looking at my phone"]


def test_the_entry_point_falls_back_to_the_text_when_there_is_no_clipboard(
        tmp_path):
    """It used to paste the command INTO the spoken reply on failure, which
    made a failed clipboard the only way to see the command at all. Now the
    command is in ``display_only`` on BOTH branches -- shown every time,
    spoken on neither, because speaking a file path is a bad minute of
    text-to-speech."""
    gallery = FaceGallery(root=tmp_path / "g")
    out = ee.enrol_answer(gallery, "hunter", owner="hunter",
                          clipboard=lambda text: False)
    assert out["command"] in out["display_only"]
    assert out["command"] not in out["reply"]


def test_the_gallery_answer_reads_the_gallery_and_opens_no_camera(tmp_path):
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 6, seed=2):
        g.add("hunter", vec, note="looking at the lens", yaw_deg=4.0)
    for vec in same_face(base_vec(4242), 3, seed=9):
        g.add("hunter", vec, note="looking at my phone", yaw_deg=-40.0)
    for vec in same_face(base_vec(808), 5, seed=17):
        g.add("heather", vec, note="looking at the lens", yaw_deg=4.0)
    g.save(reason="test")

    out = ee.gallery_answer(FaceGallery(root=g.root), owner="hunter")
    assert "hunter" in out["reply"] or "you" in out["reply"].lower()
    assert "heather" in out["reply"].lower()
    assert "looking at my phone" in out["reply"]


def test_the_gallery_answer_says_nothing_is_enrolled_rather_than_guessing(
        tmp_path):
    out = ee.gallery_answer(FaceGallery(root=tmp_path / "nope"),
                            owner="hunter")
    assert "nothing" in out["reply"].lower()


def test_the_gallery_answer_says_when_the_takes_carry_no_pose_record(tmp_path):
    """His generation is exactly this case, and the honest answer is that
    the gallery cannot say what it covers."""
    root = tmp_path / "g"
    his_generation(root)
    out = ee.gallery_answer(FaceGallery(root=root), owner="hunter")
    assert "no pose record" in out["reply"].lower() or \
        "no note" in out["reply"].lower()


def test_the_forget_answer_never_deletes_by_voice(tmp_path):
    """A misheard word may not destroy biometric data. The voice command
    hands over the command; the typed confirmation still happens in a
    terminal."""
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(808), 6, seed=17):
        g.add("heather", vec)
    g.save(reason="test")
    out = ee.forget_answer(FaceGallery(root=g.root), "heather",
                           owner="hunter", clipboard=lambda text: True)
    assert "--delete --label heather" in out["command"]
    assert FaceGallery(root=g.root).load() is True, "it deleted by voice"


def test_the_owner_forgetting_his_own_face_still_names_the_label(tmp_path):
    """"Forget my face" handed over a bare --delete, which the script
    defines as EVERYBODY plus camera.identity off -- Heather's enrolment
    would have gone with a sentence about his own six takes (F32). The
    owner's label is elided from every other command; not from a delete."""
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(808), 6, seed=17):
        g.add("hunter", vec)
    for vec in same_face(base_vec(909), 6, seed=18):
        g.add("heather", vec)
    g.save(reason="test")
    out = ee.forget_answer(FaceGallery(root=g.root), "hunter",
                           owner="hunter", clipboard=lambda text: True)
    assert "--delete --label hunter" in out["command"]
    assert ee.command_line("hunter", owner="hunter", delete=True).endswith(
        "--delete --label hunter")
    assert "--label" not in ee.command_line("hunter", owner="hunter")


def test_the_entry_point_never_imports_a_camera(tmp_path):
    """The in-app entry point reads a gallery and formats sentences. If it
    ever grows an import that can open a lens, that is a capture UI arriving
    by accident in the file whose whole argument is that it is not one."""
    src = (Path(__file__).resolve().parent.parent / "jarvis"
           / "enrolentry.py").read_text()
    imports = [ln.strip() for ln in src.splitlines()
               if ln.startswith(("import ", "from "))]
    for line in imports:
        for banned in ("cv2", "jarvis.camera", "jarvis.eye", "jarvis.visionrig",
                       "jarvis.facedetect", "PIL", "numpy"):
            assert banned not in line, line


# ---------------------------------------------------- nothing hits the disk
def test_a_whole_run_puts_nothing_but_embeddings_on_the_disk(monkeypatch,
                                                             tmp_path,
                                                             capsys):
    """The rule, checked mechanically over what actually landed: every file
    in the gallery is an npz, every array in it is either a 128-float
    embedding or a one-element scalar, and no array is image-shaped."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    capsys.readouterr()
    files = sorted(p for p in gallery.root.iterdir())
    assert any(k.startswith("note_") for k in np.load(files[0]).files), \
        "the run stored no notes at all"
    assert files, "nothing was saved"
    for path in files:
        assert path.suffix == ".npz", path.name
        data = np.load(path)
        for key in data.files:
            arr = np.asarray(data[key])
            assert arr.ndim <= 1, "%s in %s is %dD" % (key, path.name,
                                                       arr.ndim)
            assert arr.size <= 128, "%s in %s holds %d elements" \
                % (key, path.name, arr.size)


def test_the_run_holds_no_frame_after_it_returns(tmp_path):
    gallery, session = a_session(tmp_path)
    run(session, notes=["looking at my phone"])
    for name, value in vars(session).items():
        arr = np.asarray(value) if isinstance(value, np.ndarray) else None
        assert arr is None or arr.ndim <= 1, name
    assert not any(getattr(s, "frame", None) for s in session.samples)


def test_no_note_can_smuggle_an_array_into_the_store(tmp_path):
    """A note is a string. Handing it an array must not put one on the
    disk under a name the numbers-only checker never looks at."""
    g = FaceGallery(root=tmp_path / "g")
    g.add("hunter", base_vec(1), note=np.zeros((4, 4)))
    assert isinstance(g.takes("hunter")[0].note, str)
    assert len(g.takes("hunter")[0].note) <= 120


# ------------------------------------------------- the voice command itself
class FakeCommander:
    """Enough of ``jarvis.commander.Commander`` for the three face handlers:
    they ask for one service and nothing else."""

    def __init__(self, name="hunter"):
        self._name = name

    def _svc(self, which):
        if which != "assistant":
            return None
        outer = self

        class Cfg:
            def get(self, key, default=None):
                return outer._name if key == "user.name" else default
        return Cfg()


def _face_cmd(name):
    from jarvis import commander as cm
    for cmd in cm.REGISTRY:
        if cmd.name == name:
            return cmd
    raise AssertionError("no command named %r" % name)


@pytest.mark.parametrize("text,name", [
    ("enrol my face", "face enrol"),
    ("enroll my face", "face enrol"),
    ("add heather's face", "face enrol"),
    ("register heather's face to the gallery", "face enrol"),
    ("forget heather's face", "face forget"),
    ("delete heather's face from the gallery", "face forget"),
    ("who do you recognise", "face gallery"),
    ("who do you recognize?", "face gallery"),
    ("whose faces do you know", "face gallery"),
    ("what's in the face gallery", "face gallery"),
    ("which pose is weakest", "face gallery"),
])
def test_the_spoken_forms_reach_the_right_command(text, name):
    from jarvis import commander as cm
    hits = [c.name for c in cm.REGISTRY
            if c.name.startswith("face") and c.matcher(text)]
    assert hits and hits[0] == name, hits


@pytest.mark.parametrize("text", [
    "add milk to the list", "forget it", "delete the last note",
    "add a reminder for six", "who is heather", "remove the timer",
])
def test_the_face_commands_shadow_nothing_they_should_not(text):
    from jarvis import commander as cm
    assert not [c.name for c in cm.REGISTRY
                if c.name.startswith("face") and c.matcher(text)]


def test_the_face_commands_answer_without_the_wake_word():
    """They are said at the desk with the wake word already eaten, like
    every other surface verb -- and without Tier 1 they reach a model that
    would answer "who do you recognise" by inventing an answer."""
    from jarvis import commander as cm
    names = {c.name for c in cm.ASSISTANT_TIER1}
    assert {"face enrol", "face forget", "face gallery"} <= names


def test_the_voice_command_opens_no_camera_and_hands_the_command_over(
        monkeypatch, tmp_path):
    from jarvis import commander as cm
    gallery = FaceGallery(root=tmp_path / "g")
    monkeypatch.setattr(cm, "_face_gallery", lambda _c: gallery)
    monkeypatch.setattr(ee, "to_clipboard", lambda text, run=None: True)
    cmd = _face_cmd("face enrol")
    out = cmd.handler(FakeCommander(), "enrol my face",
                      cmd.matcher("enrol my face"))
    assert out.handled and out.speak
    assert "terminal" in out.reply
    assert gallery.generations() == []


def test_the_voice_command_names_the_person_and_the_consent_step(monkeypatch,
                                                                 tmp_path):
    from jarvis import commander as cm
    monkeypatch.setattr(cm, "_face_gallery",
                        lambda _c: FaceGallery(root=tmp_path / "g"))
    monkeypatch.setattr(ee, "to_clipboard", lambda text, run=None: True)
    cmd = _face_cmd("face enrol")
    text = "add heather's face"
    out = cmd.handler(FakeCommander(), text, cmd.matcher(text))
    assert "consent" in out.reply.lower()
    assert "Heather" in out.reply


def test_the_voice_command_asks_whose_face_rather_than_guessing(monkeypatch,
                                                                tmp_path):
    from jarvis import commander as cm
    monkeypatch.setattr(cm, "_face_gallery",
                        lambda _c: FaceGallery(root=tmp_path / "g"))
    cmd = _face_cmd("face enrol")
    text = "add another face"
    m = cmd.matcher(text)
    assert m, "the phrase must still reach the command"
    out = cmd.handler(FakeCommander(), text, m)
    assert "whose" in out.reply.lower()


def test_the_voice_delete_destroys_nothing(monkeypatch, tmp_path):
    """A misheard word may not destroy biometric data."""
    from jarvis import commander as cm
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(808), 6, seed=17):
        g.add("heather", vec)
    g.save(reason="test")
    monkeypatch.setattr(cm, "_face_gallery",
                        lambda _c: FaceGallery(root=g.root))
    monkeypatch.setattr(ee, "to_clipboard", lambda text, run=None: True)
    cmd = _face_cmd("face forget")
    text = "forget heather's face"
    out = cmd.handler(FakeCommander(), text, cmd.matcher(text))
    assert out.handled and "6 takes" in out.reply
    assert "type the name" in out.reply.lower()
    back = FaceGallery(root=g.root)
    assert back.load() is True and "heather" in back.labels()
    assert back.count("heather") == 6


def test_the_owner_label_comes_from_his_config_and_never_the_gallery(
        monkeypatch, tmp_path):
    """WHERE THE RULING IS ANCHORED. If the owner were read from the
    gallery, enrolling a second person would enlarge the set of privileged
    names -- which is exactly the thing that may never happen."""
    from jarvis import commander as cm
    g = FaceGallery(root=tmp_path / "g")
    g.add("heather", base_vec(808))
    g.save(reason="test")
    monkeypatch.setattr(cm, "_face_gallery", lambda _c: FaceGallery(root=g.root))
    assert cm._face_owner(FakeCommander("Hunter")) == "hunter"
    assert cm._face_owner(FakeCommander("")) == "hunter"


def test_the_gallery_question_is_answered_without_a_lens(monkeypatch,
                                                          tmp_path):
    from jarvis import commander as cm
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 6, seed=2):
        g.add("hunter", vec, note="looking at the lens", yaw_deg=4.0)
    g.save(reason="test")
    monkeypatch.setattr(cm, "_face_gallery", lambda _c: FaceGallery(root=g.root))
    cmd = _face_cmd("face gallery")
    text = "who do you recognise"
    out = cmd.handler(FakeCommander(), text, cmd.matcher(text))
    assert out.handled and "1 face" in out.reply


# ------------------------------- the sensing owner, through the REAL object
def _real_policy(tmp_path, when=None, offline=False):
    """A real ``SensingPolicy`` on a throwaway state file, with its clock
    seam driven. Not a fake: what is being pinned is that enrolment obeys
    the object the running Jarvis obeys, in all three of its refusals."""
    import datetime as _dt

    from jarvis.sensing import SensingPolicy
    at = when or _dt.datetime(2026, 9, 3, 12, 0, 0)
    # _REAL_CONFIG_LOAD, not AssistantConfig.load: ``wire`` monkeypatches
    # that to a zero-argument lambda, and this needs a real config on a
    # throwaway path.
    pol = SensingPolicy(cfg=_REAL_CONFIG_LOAD(tmp_path / "sensing-cfg.json"),
                        path=tmp_path / "sensing.json",
                        now=lambda: at.timestamp())
    if not offline:
        pol.enable(source="test")
    return pol


@pytest.mark.parametrize("kind", ["failsafe", "offline", "curfew"])
def test_enrolment_is_refused_in_every_state_the_real_policy_denies(
        kind, monkeypatch, tmp_path, capsys):
    """OFFLINE, THE CURFEW AND THE FAIL-SAFE, through the real
    ``SensingPolicy`` rather than a stand-in -- because the thing worth
    pinning is that this script reads the same object the running Jarvis
    reads, and that all three of its refusals stop the run before a device
    is opened."""
    import datetime as _dt
    gallery, _feed = wire(monkeypatch, tmp_path, feed_guard=True)
    if kind == "failsafe":
        # A missing state file starts OFFLINE and FAILSAFE by design.
        pol = _real_policy(tmp_path, offline=True)
    elif kind == "offline":
        pol = _real_policy(tmp_path)
        pol.disable(source="test")
    else:
        pol = _real_policy(tmp_path,
                           when=_dt.datetime(2026, 9, 3, 23, 30, 0))
    assert pol.status()["camera"] is False, kind
    monkeypatch.setattr(face_enrol, "SensingPolicy", lambda cfg=None: pol)
    monkeypatch.setattr("builtins.input", lambda *_a: "heather")
    code = face_enrol.main(ENROL + ["--label", "heather"])
    out = capsys.readouterr().out
    assert code == 2, out
    assert "sensing says the camera may not run" in out
    assert gallery.generations() == [], "a denied run reached the disk"
    # ...and the consent prompt never even happened: a run sensing refused
    # must not take somebody's biometric consent for a capture that cannot
    # occur.
    assert "CONSENT" not in out


def test_the_real_policy_lets_a_normal_enrolment_through(monkeypatch,
                                                          tmp_path, capsys):
    """The other half: the three refusals above are the policy's, not a
    blanket refusal that would make this test suite meaningless."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    pol = _real_policy(tmp_path)
    assert pol.status()["camera"] is True
    monkeypatch.setattr(face_enrol, "SensingPolicy", lambda cfg=None: pol)
    assert face_enrol.main(ENROL) == 0, capsys.readouterr().out
    assert gallery.generations() == [1]


def test_the_gap_is_measured_against_what_the_run_KEEPS(monkeypatch,
                                                        tmp_path, capsys):
    """A plain re-run REPLACES this label's pool, so asking it for the
    missing station alone would write a three-sample gallery over a
    thirteen-sample one and call that coverage. Only --append carries the
    stored takes forward, and only there does "the station you are missing"
    mean anything."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0                  # 13, all five poses
    capsys.readouterr()

    wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0                  # plain: the five
    out = capsys.readouterr().out
    assert "5 stations" in out, out
    back = FaceGallery(root=gallery.root)
    back.load()
    assert back.count("hunter") == 13


def test_an_append_asks_only_for_the_station_that_is_missing(monkeypatch,
                                                             tmp_path,
                                                             capsys):
    g = FaceGallery(root=tmp_path / "gallery")
    for i, vec in enumerate(same_face(base_vec(1), 12, seed=2)):
        g.add("hunter", vec,
              note="looking at the lens" if i < 6 else "looking at my screen",
              yaw_deg=4.0 if i < 6 else 50.0)
    g.save(reason="one side only")

    wire(monkeypatch, tmp_path, yaws=[-40.0] * 13,
         vectors=same_face(base_vec(1), 13, seed=2))   # the same face
    code = face_enrol.main(ENROL + ["--append"])
    out = capsys.readouterr().out
    assert "1 station" in out, out
    assert "missing" in out
    assert "turned the other way" in out
    assert code == 0, out
    back = FaceGallery(root=tmp_path / "gallery")
    back.load()
    assert back.count("hunter") == 14
    assert fe.coverage(back.takes("hunter"))["negative"] == 2


def test_plan_missing_without_append_says_why_rather_than_running_the_five(
        monkeypatch, tmp_path, capsys):
    gallery, _feed = wire(monkeypatch, tmp_path)
    code = face_enrol.main(ENROL + ["--plan", "missing"])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "--append" in out
    assert gallery.generations() == []


def test_an_append_onto_a_note_less_generation_still_earns_the_spread_alone(
        monkeypatch, tmp_path, capsys):
    """THE GUARANTEE THAT SURVIVES THE RELAXATION. pose_spread now counts
    angles RECORDED with earlier takes, which is what makes a gap-filling
    append possible at all. His generation 1 records none -- so appending
    onto it inherits nothing, and a one-pose append still fails, exactly as
    it did before. Evidence is used where it exists and assumed nowhere."""
    root = tmp_path / "gallery"
    vecs = his_generation(root, n=13, seed=1)
    wire(monkeypatch, tmp_path, yaws=[-40.0] * 13,
         vectors=same_face(base_vec(1), 13, seed=2))
    assert [np.asarray(v).size for v in vecs] == [128] * 13
    code = face_enrol.main(ENROL + ["--append"])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "[FAIL] pose_spread" in out
    assert "13 stored take(s) carry no angle and count for nothing" in out
    assert FaceGallery(root=root).generations() == [1]


def test_a_recorded_earlier_take_is_what_makes_the_gap_append_possible(
        tmp_path):
    """The same check, both ways round, as arithmetic: 12 recorded angles
    across two poses plus a two-sample third pose PASSES, and the identical
    run against 12 UNRECORDED takes FAILS."""
    samples = [fe.Sample(index=i, station="across", faces=1, conf=0.9,
                         face_px=300.0, eye_px=140.0, yaw_deg=-40.0,
                         roll_deg=0.0, bearing_deg=0.0, sharpness=0.5,
                         accepted=True, reason="ok",
                         note="turned the other way")
               for i in range(2)]
    embs = same_face(base_vec(1), 2, seed=5)
    pool = same_face(base_vec(1), 12, seed=2) + embs
    recorded = ([Take("looking at the lens", 4.0)] * 6
                + [Take("looking at my screen", 50.0)] * 6
                + [Take("turned the other way", -40.0)] * 2)
    blank = [Take()] * 12 + [Take("turned the other way", -40.0)] * 2

    def spread(takes):
        checks = fe.judge_gallery(samples, embs, pool=pool, takes=takes)
        return {c.name: c.ok for c in checks}["pose_spread"]

    assert spread(recorded) is True
    assert spread(blank) is False


def test_naming_poses_on_top_of_a_gallery_hands_over_an_append(tmp_path):
    """"More ways for me to be recognised" is additive by intent. A plain
    two-pose run is six takes, under the eight-sample floor -- it would
    capture, fail and save nothing, which costs him the minute AND the
    enrolment he already had."""
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 13, seed=2):
        g.add("hunter", vec, note="looking at the lens", yaw_deg=4.0)
    g.save(reason="test")
    out = ee.enrol_answer(FaceGallery(root=g.root), "hunter", owner="hunter",
                          poses=["looking at my phone", "looking away"],
                          clipboard=lambda text: True)
    import shlex
    assert "--append" in shlex.split(out["command"])
    args = face_enrol.build_parser().parse_args(
        shlex.split(out["command"])[2:])
    assert args.append is True and len(args.pose) == 2


def test_naming_poses_on_an_empty_gallery_is_not_an_append(tmp_path):
    out = ee.enrol_answer(FaceGallery(root=tmp_path / "nope"), "hunter",
                          owner="hunter", poses=["looking at my phone"],
                          clipboard=lambda text: True)
    assert "--append" not in out["command"]


def test_the_command_warns_before_it_replaces_a_pool_with_named_takes(
        monkeypatch, tmp_path, capsys):
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    capsys.readouterr()
    wire(monkeypatch, tmp_path)
    face_enrol.main(ENROL + ["--pose", "looking at my phone"])
    out = capsys.readouterr().out
    assert "will REPLACE the 13 already stored" in out
    assert "--append is almost certainly what you want" in out
    assert gallery.generations() == [1], "the refused run wrote a generation"


def test_a_refused_consent_does_not_turn_the_identity_gate_on(monkeypatch,
                                                              tmp_path,
                                                              capsys):
    """``camera.identity`` says faces may be written down at all. Somebody
    declining to be enrolled must not have flipped it on by declining."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_a: "no")
    code = face_enrol.main(["--gap-s", "0", "--enable-identity",
                            "--label", "heather"])
    assert code == 1, capsys.readouterr().out
    cfg = face_enrol.AssistantConfig.load()
    assert bool(cfg.get("camera.identity", False)) is False
    assert gallery.generations() == []


def test_nobody_is_asked_to_consent_to_a_run_that_was_going_to_stop(
        monkeypatch, tmp_path, capsys):
    """The read half of the phase gate runs BEFORE the consent text, so she
    is not asked to agree to a capture the flag was going to refuse."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    asked = []
    monkeypatch.setattr("builtins.input",
                        lambda *a: asked.append(a) or "heather")
    code = face_enrol.main(["--gap-s", "0", "--label", "heather"])
    out = capsys.readouterr().out
    assert code == 1
    assert "camera.identity is false" in out
    assert "CONSENT" not in out
    assert asked == []
    assert gallery.generations() == []


def test_an_unreadable_gallery_is_not_reported_as_an_empty_one(tmp_path):
    """Different sentences, because they send him to different commands: an
    empty gallery means enrol, an unreadable one means --rollback."""
    root = tmp_path / "g"
    root.mkdir()
    (root / "gen-00001.npz").write_bytes(b"not an npz")
    out = ee.gallery_answer(FaceGallery(root=root), owner="hunter")
    assert "can't read" in out["reply"]
    assert "nothing is enrolled" not in out["reply"].lower()


def test_the_gallery_answer_says_HAS_for_somebody_who_is_not_him(tmp_path):
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(808), 4, seed=17):
        g.add("heather", vec, note="looking at the lens", yaw_deg=4.0)
    g.save(reason="test")
    reply = ee.gallery_answer(FaceGallery(root=g.root), owner="hunter")["reply"]
    assert "Heather has no takes turned the other way" in reply


@pytest.mark.parametrize("said,want", [
    ("my", "hunter"), ("me", "hunter"), ("", "hunter"),
    ("heather", "heather"), ("heather's", "heather"),
    ("Mary Jane", "mary-jane"),
    ("a new", ""), ("another", ""), ("someone", ""), ("the", ""),
    ("her", ""), ("this", ""),
])
def test_the_spoken_name_is_resolved_or_refused_never_guessed(said, want):
    """"Add the face" must not enrol somebody under the label "the" -- a
    perfectly valid gallery key and permanent junk in a store whose whole
    point is knowing who is in it."""
    assert ee.spoken_label(said, owner="hunter") == want


# ---------------------------------------------------------------------------
# THE SECOND PASS. Every test below is one a reviewer wrote down as MISSING
# after reproducing the failure it pins on a throwaway gallery. They are
# grouped rather than scattered because they share one sentence: a command
# that deletes biometric data must destroy exactly what it proved it should,
# say what it could not prove, and never claim more than it did.
# ---------------------------------------------------------------------------

def _unreadable(path: Path) -> None:
    """Make a real generation unreadable the way a FUTURE BUILD would.

    Not by truncating it -- by bumping ``_format``, which is precisely the
    state ``FaceGallery._read`` is DESIGNED to produce on a format bump. That
    matters: it means "unreadable" is not an exotic corruption, it is the
    ordinary condition of every existing generation on the day the format
    changes, and it is why destroying one on somebody's say-so is not a
    tidy-up but a way to lose the whole gallery."""
    data = dict(np.load(path))
    data["_format"] = np.array([int(data["_format"][0]) + 1])
    np.savez(path, **data)


def test_purging_an_absent_label_never_destroys_an_unreadable_generation(
        tmp_path):
    """THE ONE THAT COULD HAVE EMPTIED HIS GALLERY.

    ``purge_label`` used to shred every generation it could not PARSE, on the
    reasoning that nothing can prove an unreadable file does not hold her.
    Reproduced end to end on 2026-09-03: 13 hunter embeddings, one
    generation, ``_format`` bumped by one -- ``--delete --label heather``
    reported "1 unreadable generation(s) destroyed as well / nothing is left;
    the gallery is empty", exited 0, and left an empty directory. Heather was
    never enrolled."""
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 13, seed=2):
        g.add("hunter", vec)
    g.save(reason="one")
    _unreadable(g.root / "gen-00001.npz")

    out = FaceGallery(root=g.root).purge_label("heather", reason="test")
    assert out["removed"] == 0
    assert out["unreadable"] == [1]
    assert sorted(p.name for p in g.root.iterdir()) == ["gen-00001.npz"]


def test_purging_an_absent_label_spares_the_unreadable_one_beside_a_good_one(
        tmp_path):
    """The same thing with a readable generation next to it: neither goes,
    because neither was proven to hold her, and a no-op delete may not write
    a redundant generation either."""
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 13, seed=2):
        g.add("hunter", vec)
    g.save(reason="one")
    g.save(reason="two")
    _unreadable(g.root / "gen-00002.npz")

    out = FaceGallery(root=g.root).purge_label("heather", reason="test")
    assert out["removed"] == 0
    assert out["generation"] == 0, "a delete of nobody must write nothing"
    assert out["unreadable"] == [2]
    assert sorted(p.name for p in g.root.iterdir()) == ["gen-00001.npz",
                                                        "gen-00002.npz"]


def test_the_command_refuses_to_claim_a_delete_it_could_not_verify(
        monkeypatch, tmp_path, capsys):
    """And it has to SAY so. The exit code is non-zero and the word
    "verified" does not appear, because there is a file on the disk nothing
    can open and therefore nothing can vouch for."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    capsys.readouterr()
    _unreadable(gallery.root / "gen-00001.npz")

    code = face_enrol.main(["--delete", "--label", "heather", "--yes"])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "verified" not in out
    assert "could NOT be read" in out
    assert (gallery.root / "gen-00001.npz").exists()


def test_purging_the_last_person_takes_the_crashed_save_tmp_with_her(tmp_path):
    """CONSENT WITHDRAWAL OVER DATA THAT IS STILL THERE.

    ``gen-00002.npz.tmp`` holds a whole pool and does not match ``_GEN_RE``,
    so ``generations()`` -- and every read-back that walks it -- is blind to
    it. When somebody else survives the delete, ``save()`` -> ``_prune()``
    shreds the tmps on the way past, which is exactly why this never showed:
    when NOBODY survives there is no save, so her complete embedding set sat
    on the disk under a command that printed "verified" and exited 0."""
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 9, seed=2):
        g.add("heather", vec)
    g.save(reason="one")
    tmp = g.root / "gen-00002.npz.tmp"
    tmp.write_bytes((g.root / "gen-00001.npz").read_bytes())

    out = FaceGallery(root=g.root).purge_label("heather", reason="test")
    assert out["tmp_removed"] == 1
    assert list(g.root.iterdir()) == []


def test_purging_takes_a_tmp_that_is_the_only_place_she_survives(tmp_path):
    """She is in NO generation and only in a crashed save's leftovers. The
    early return used to fire before anything was touched."""
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 13, seed=2):
        g.add("hunter", vec)
    g.save(reason="one")
    hers = FaceGallery(root=tmp_path / "h")
    for vec in same_face(base_vec(808), 9, seed=17):
        hers.add("heather", vec)
    hers.save(reason="hers")
    (g.root / "gen-00009.npz.tmp").write_bytes(
        (hers.root / "gen-00001.npz").read_bytes())

    out = FaceGallery(root=g.root).purge_label("heather", reason="test")
    assert out["tmp_removed"] == 1
    assert sorted(p.name for p in g.root.iterdir()) == ["gen-00001.npz"]
    for path in g.root.iterdir():
        assert b"heather" not in path.read_bytes()


def test_the_command_verifies_the_leftovers_and_not_only_the_generations(
        monkeypatch, tmp_path, capsys):
    """The read-back has to be able to FAIL on a tmp, or "verified" means
    "no generation", which is not what the sentence says."""
    gallery, _feed = wire(monkeypatch, tmp_path)
    assert face_enrol.main(ENROL) == 0
    wire(monkeypatch, tmp_path, vectors=same_face(base_vec(808), 13, seed=17))
    monkeypatch.setattr("builtins.input", lambda *_a: "heather")
    assert face_enrol.main(OTHER) == 0
    capsys.readouterr()
    gens = FaceGallery(root=gallery.root).generations()
    (gallery.root / "gen-09999.npz.tmp").write_bytes(
        (gallery.root / ("gen-%05d.npz" % gens[-1])).read_bytes())

    assert face_enrol.main(["--delete", "--label", "heather", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "verified" in out
    assert FaceGallery(root=gallery.root).leftovers() == [], out


def test_a_failed_save_leaves_the_leftovers_alone_too(tmp_path, monkeypatch):
    """"Nothing was destroyed" has to mean nothing, tmps included."""
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 13, seed=2):
        g.add("hunter", vec)
    for vec in same_face(base_vec(808), 9, seed=17):
        g.add("heather", vec)
    g.save(reason="one")
    (g.root / "gen-00002.npz.tmp").write_bytes(b"leftovers")

    live = FaceGallery(root=g.root)
    monkeypatch.setattr(FaceGallery, "save",
                        lambda *a, **k: (_ for _ in ()).throw(
                            ValueError("disk full")))
    out = live.purge_label("heather", reason="test")
    assert out["reason"] == "disk full"
    assert out["removed"] == 0 and out["tmp_removed"] == 0
    assert sorted(p.name for p in g.root.iterdir()) == \
        ["gen-00001.npz", "gen-00002.npz.tmp"]


def test_a_malformed_note_key_costs_the_note_and_not_the_embeddings(tmp_path):
    """A COSMETIC FIELD MAY NOT COST THIRTEEN FACES.

    Reproduced before the fix: a zero-length ``note_hunter_0003`` (IndexError
    on ``[0]``) made the whole generation unreadable, ``load()`` fell back to
    the one before, and -- compounding with the bug above -- a purge of
    somebody who was never enrolled then shredded it."""
    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 13, seed=2):
        g.add("hunter", vec)
    g.save(reason="one")
    data = dict(np.load(g.root / "gen-00001.npz"))
    data["note_hunter_0003"] = np.array([])
    data["yaw_hunter_0004"] = np.array([])
    np.savez(g.root / "gen-00002.npz", **data)

    back = FaceGallery(root=g.root)
    assert back.load() is True
    assert back.loaded_generation == 2, "a bad note cost the whole generation"
    assert back.count("hunter") == 13
    assert back.takes("hunter")[3].note == ""
    assert back.takes("hunter")[4].yaw_deg is None


def test_consent_cannot_be_typed_by_a_pipe(tmp_path, monkeypatch, capsys):
    """THE MECHANISM --json WAS NAMED AFTER, LEFT OPEN.

    Blocking --json closed the flag and not the pipe: ``echo heather |
    face_enrol.py --label heather --auto --yes`` satisfied the prompt with
    nobody at the keyboard, and the run then wrote "consent typed at the
    keyboard" into the generation's provenance -- a false attestation on
    disk, which is worse than no record at all."""
    wire(monkeypatch, tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_a: "heather")
    # OPTING BACK OUT of the module fixture: this is the one test that wants
    # the streams pytest actually gives it, which are pipes.
    monkeypatch.setattr(face_enrol, "_isatty", lambda _stream: False)
    code = face_enrol.main(OTHER)
    out = capsys.readouterr().out
    assert code == 1, out
    assert "pipe" in out
    assert not FaceGallery(root=tmp_path / "face_gallery").generations()


def test_auto_cannot_capture_somebody_else(tmp_path, monkeypatch, capsys):
    """A station nobody has to press a key for is a station nobody has to be
    PRESENT for. --auto stays available for his own face."""
    wire(monkeypatch, tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_a: "heather")
    code = face_enrol.main(OTHER + ["--auto"])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "--auto" in out


def test_a_username_the_gallery_cannot_store_stops_before_the_camera(tmp_path):
    """Not after a minute in front of it. ``owner_label`` keeps any character
    ``isalnum()`` likes -- non-ASCII included -- while the gallery stores
    under ``^[a-z0-9][a-z0-9_-]{0,30}$``, so a ``user.name`` of "José"
    produced a label ``add()`` refuses on every single sample."""
    class Cfg(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    class Args:
        label = ""

    for name in ("José", "x" * 40, "-bob"):
        label, why = face_enrol.target_label(Cfg({"user.name": name}), Args())
        assert label == "" and "user.name" in why, name
    label, why = face_enrol.target_label(Cfg({"user.name": "Hunter"}), Args())
    assert (label, why) == ("hunter", "")


def test_status_json_reports_each_pool_and_not_the_last_one_twice(
        monkeypatch, tmp_path, capsys):
    """--json's cohesion figures used to be written flat onto the payload
    inside the per-label loop, so the last label overwrote the first: a
    healthy 0.998 printed over a gallery whose weakest pool was 0.466. The
    printed TEXT was right, which is why nobody saw it."""
    g, _feed = wire(monkeypatch, tmp_path)
    for vec in same_face(base_vec(1), 13, k=0.15, seed=2):
        g.add("hunter", vec)
    for vec in same_face(base_vec(808), 9, k=0.9, seed=17):
        g.add("heather", vec)
    g.save(reason="two people")

    assert face_enrol.main(["--status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)["result"]
    by = payload["by_label"]
    assert set(by) == {"hunter", "heather"}
    assert by["hunter"]["cohesion_min"] != by["heather"]["cohesion_min"]
    for label, one in by.items():
        assert one["pairs"] > 0, label
    assert "cohesion_min" not in payload, \
        "a gallery-wide figure would be one pool's, reported as everyone's"


@pytest.mark.parametrize("text", [
    # THE CLASS THE FIRST GRAMMAR SWALLOWED. _FACE_WHO allowed 31 arbitrary
    # characters INCLUDING SPACES between the verb and "face", so every
    # ordinary "<verb> <filler> face" utterance became a face command --
    # spoke a forty-word consent paragraph and overwrote his clipboard,
    # without the background-chat gate ever getting a say.
    "add a reminder to wash my face",
    "add cream for my face",
    "delete that photo of my face",
    "remove the hair from my face",
    "forget what i said about her face",
    "add a note about her face",
    "add sunscreen to my face",
    "remove hair from face",
    "add a face mask to the shopping list",
    "remember that i like coffee",
    "remember to wash my face",
    "remember my dentist appointment",
    "learn about my calendar",
    "put cream on my face",
    "what do you remember",
    "do you recognize me now",
    "forget me",
    "remember me",
])
def test_an_ordinary_sentence_with_the_word_face_in_it_is_not_a_face_command(
        text):
    from jarvis import commander as cm
    assert not [c.name for c in cm.REGISTRY
                if c.name.startswith("face") and c.matcher(text)]
    assert not [c.name for c in cm.ASSISTANT_TIER1
                if c.name.startswith("face") and c.matcher(text)]


@pytest.mark.parametrize("text,name", [
    ("enrol me", "face enrol"), ("enroll me", "face enrol"),
    ("register me", "face enrol"),
    ("remember my face", "face enrol"),
    ("learn heather's face", "face enrol"),
    ("memorise my face", "face enrol"),
    ("enrol my face again", "face enrol"),
    ("enrol my face looking at my phone", "face enrol"),
    ("add my face wearing glasses", "face enrol"),
    ("add mary jane's face", "face enrol"),
    ("unenrol heather's face", "face forget"),
    ("what faces do you know", "face gallery"),
    ("how many faces do you know", "face gallery"),
    ("who's in the face gallery", "face gallery"),
    ("am i enrolled", "face gallery"),
])
def test_the_natural_phrasings_reach_the_command_too(text, name):
    """42 of 50 natural phrasings used to miss and fall through to a model
    that would invent an answer -- the exact failure Tier 1 exists to stop."""
    from jarvis import commander as cm
    hits = [c.name for c in cm.REGISTRY
            if c.name.startswith("face") and c.matcher(text)]
    assert hits and hits[0] == name, hits


def test_a_spoken_pose_reaches_the_handed_over_command(tmp_path):
    """REQUIREMENT (c), FROM INSIDE JARVIS. ``--pose`` and ``--append`` used
    to be unreachable from voice -- ``poses`` was never parsed and never
    passed -- so every spoken enrolment handed over a POOL-REPLACING run over
    his thirteen stored takes."""
    from jarvis import commander as cm
    m = cm._FACE_ENROL_RX.match("enrol my face looking at my phone")
    assert m and ee.spoken_pose(m.group("pose")) == "looking at my phone"

    g = FaceGallery(root=tmp_path / "g")
    for vec in same_face(base_vec(1), 13, seed=2):
        g.add("hunter", vec)
    g.save(reason="one")
    out = ee.enrol_answer(FaceGallery(root=g.root), "hunter", owner="hunter",
                          poses=("looking at my phone",),
                          clipboard=lambda _t: True)
    assert "--pose 'looking at my phone'" in out["command"]
    assert "--append" in out["command"], \
        "a pose run over 13 stored takes must not replace the pool"


def test_a_spoken_pose_cannot_become_shell(tmp_path):
    """The pose comes off a speech recogniser and goes onto his CLIPBOARD,
    which is a place he pastes things into a terminal. ``command_line``
    shlex-quotes every part, so the whole clause stays ONE argument."""
    import shlex
    cmd = ee.command_line("hunter", owner="hunter",
                          poses=("'; rm -rf ~; echo '",))
    parts = shlex.split(cmd)
    assert "rm" not in parts and "-rf" not in parts
    assert parts[parts.index("--pose") + 1] == "'; rm -rf ~; echo '"
    assert parts[-1] == "'; rm -rf ~; echo '", "the pose is the last word"
