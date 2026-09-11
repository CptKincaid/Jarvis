"""WHAT THE GLIMPSE STEP COSTS, over a synthetic day, in numbers.

THE CHANGE UNDER TEST. ``presencevote.DepartureSequence.room`` now accepts a
GLIMPSE -- a kitchen run that lit for too few polls to clear the 2 s enter
hold -- as the kitchen step of his "office then kitchen then phone
disconnect" departure. It had to: on 2026-09-06 he walked out at ~10:51 and
the kitchen published NOTHING, because roomfabric needs a room occupied for
enter_hold_s 2.0 s across a 2.0 s poll, so a pass-through needs about 4 s in
the beam. The sequence never left SEQ_DESK and "away" arrived about fourteen
minutes late off the corroboration window instead.

THE RISK IT BUYS, STATED PLAINLY. The LD2410 reads through plasterboard --
roomfabric's own docstring and the kitchen mount note both say so -- so a
spurious kitchen glimpse is a real event, and the false path is:

    office RoomChanged -> a spurious kitchen glimpse inside 120 s
                       -> his phone stops answering inside 900 s

NOBODY HAS MEASURED HIS SPURIOUS-GLIMPSE RATE. So this file does not invent
one. It SWEEPS the rate and reports what each one costs, which is the honest
shape: the reader picks the row that matches his flat, and the day he has a
measured rate the sweep is replaced by one number.

THE COST CEILING IS THE REASON THIS IS CHEAP. ``DepartureSequence.speaks`` is
False and ``seq.left`` is read by nothing outside the class (``grep``:
``departure_note`` has one caller, ``left`` none). So a false departure today
costs EXACTLY ONE LOG LINE. The moment anything consumes ``left`` -- an
earlier grounded "away", a re-armed door watch -- the cost becomes real, and
the glimpse step should then additionally require a non-glimpse
corroboration. ``test_the_cost_ceiling_is_still_one_log_line`` fails on the
day that changes.

Everything here is SYNTHETIC and DETERMINISTIC: a seeded PRNG, invented
timestamps, no clock, no file, no device.
"""
from __future__ import annotations

import random

from jarvis import presencevote as pv

DAY_S = 24 * 3600.0

# HIS FLAT, AS THE SIMULATION MODELS IT. All three are CHOSEN, not measured,
# and each is stated so the reader can disagree with it by editing one line.
REAL_OUTINGS = 4          # walks out of the building in a day
DESK_VISITS = 40          # times the office becomes the active room
PHONE_NAPS = 12           # times his iPhone drops off Wi-Fi power-save


def _day(rng, spurious_glimpses: int, glimpse_step: bool) -> dict:
    """One synthetic day. Returns counts, never a verdict.

    ``glimpse_step`` False is the BASELINE -- the sequence exactly as it
    shipped, where a glimpse is dropped on the floor -- so every row below
    is a difference against the code it replaces rather than an absolute
    nobody can calibrate.
    """
    seq = pv.DepartureSequence()
    # (at, what, is_glimpse, real). ``real`` is the ground truth this
    # simulation owns and the product never sees, so attribution is exact:
    # a fire is TRUE exactly when the phone event that completed it belongs
    # to a real walk-out.
    events = []
    for _ in range(DESK_VISITS):
        events.append((rng.uniform(0.0, DAY_S), "office", False, False))
    for _ in range(spurious_glimpses):
        events.append((rng.uniform(0.0, DAY_S), "kitchen", True, False))
    for _ in range(PHONE_NAPS):
        events.append((rng.uniform(0.0, DAY_S), "phone", False, False))
    # A REAL walk-out: the office, then the kitchen ~12 s later as a glimpse
    # (that is the 09-06 measurement -- the kitchen never cleared the hold),
    # then the phone going out of range ~90 s after that.
    for _ in range(REAL_OUTINGS):
        t = rng.uniform(0.0, DAY_S - 300.0)
        events.append((t, "office", False, True))
        events.append((t + 12.0, "kitchen", True, True))
        events.append((t + 102.0, "phone", False, True))
    events.sort(key=lambda e: e[0])

    # ATTRIBUTION IS BY THE KITCHEN STEP, NOT BY THE PHONE EVENT. The
    # sequence's claim is "he walked out of the building", and that claim
    # is TRUE exactly when the kitchen step that armed it belongs to a real
    # walk-out -- whichever of his phone's drops happens to complete it. A
    # real outing completed by a coincidental nap is still a real outing,
    # and scoring it as a miss would flatter the baseline.
    caught = false = 0
    armed_real = False
    for at, what, glimpse, real in events:
        if what == "phone":
            if seq.phone_gone(at):
                if armed_real:
                    caught += 1
                else:
                    false += 1
                armed_real = False
            continue
        if glimpse and not glimpse_step:
            continue                      # the shipped behaviour: dropped
        seq.room(room=what, at=at, glimpse=glimpse)
        if seq.stage == pv.SEQ_DOOR:
            armed_real = real
        elif seq.stage != pv.SEQ_LEFT:
            armed_real = False
    return {"fired": caught + false, "caught": caught, "false": false}


def sweep(seed: int = 20260906, days: int = 30) -> list:
    """The instrument. One row per spurious-glimpse rate:
    ``(per_day, real_departures_caught, false_departures)`` summed over
    ``days`` synthetic days, for the glimpse step and for the baseline."""
    rows = []
    for per_day in (0, 1, 4, 12, 48):
        on = {"caught": 0, "false": 0}
        off = {"caught": 0, "false": 0}
        rng = random.Random(seed + per_day)
        for _ in range(days):
            for step, acc in ((True, on), (False, off)):
                out = _day(random.Random(rng.random()), per_day, step)
                acc["caught"] += out["caught"]
                acc["false"] += out["false"]
        rows.append((per_day, on, off))
    return rows


def test_the_shipped_sequence_catches_none_of_his_walk_outs():
    """THE DEFECT, in one number. His kitchen pass-through never clears the
    2 s enter hold, so without the glimpse step the sequence catches ZERO
    real departures a day -- which is exactly what 2026-09-06 recorded."""
    rows = sweep(days=30)
    for _per_day, _on, off in rows:
        assert off["caught"] == 0
        assert off["false"] == 0, "and it cannot be wrong, because it is mute"


def test_the_glimpse_step_catches_most_of_his_walk_outs():
    """MOST, not all, and the shortfall is honest rather than a bug: a
    random office change landing between the kitchen glimpse and the phone
    dropping resets the sequence to the desk, which is the sequence
    correctly deciding he came back and then left again unobserved. Over 30
    synthetic days at 4 outings a day the step catches the large majority
    of 120, against a baseline of ZERO."""
    rows = sweep(days=30)
    for per_day, on, off in rows:
        assert off["caught"] == 0, per_day
        assert on["caught"] >= 0.85 * REAL_OUTINGS * 30, (per_day, on)


def test_the_false_departure_cost_is_reported_per_rate(capsys):
    """THE TABLE HE ASKED FOR. Printed as well as asserted, because the
    number that matters is his spurious-glimpse rate and nobody has it: the
    row he can measure is the row he should read."""
    rows = sweep(days=30)
    print("\nspurious kitchen glimpses/day | real caught | FALSE departures"
          " | per day")
    for per_day, on, _off in rows:
        print("%29d | %11d | %16d | %.2f"
              % (per_day, on["caught"], on["false"], on["false"] / 30.0))
    out = capsys.readouterr().out
    assert "FALSE departures" in out
    # A flat with NO spurious glimpses cannot produce a false departure at
    # all: the arithmetic has to bottom out at zero or the model is wrong.
    assert rows[0][1]["false"] == 0


def test_a_glimpse_alone_never_completes_a_departure():
    """THE THREE PLACES A GLIMPSE MAY NOT GO, pinned rather than trusted.

    It may not ARM the sequence at the desk (that would make a departure
    EASIER to reach, the wrong direction); it may not CLEAR one (a
    through-wall flicker is not him coming back); and it may not fire the
    departure without his phone actually going.
    """
    seq = pv.DepartureSequence()
    seq.room(room="office", at=0.0, glimpse=True)
    assert seq.stage != pv.SEQ_DESK, "a flicker is not him sitting down"

    seq.room(room="office", at=10.0)
    seq.room(room="kitchen", at=20.0, glimpse=True)
    assert seq.stage == pv.SEQ_DOOR
    seq.room(room="kitchen", at=30.0, glimpse=True)
    assert seq.stage == pv.SEQ_DOOR, "a second flicker may not break it"
    assert seq.phone_gone(10000.0) is False, "past the 900 s window"


def test_the_note_says_which_evidence_completed_the_kitchen_step():
    """"the kitchen saw him" and "the kitchen saw something for one poll"
    are different claims and the line he reads has to say which."""
    seq = pv.DepartureSequence()
    seq.room(room="office", at=0.0)
    seq.room(room="kitchen", at=12.0, glimpse=True)
    assert seq.phone_gone(102.0) is True
    note = pv.departure_note(seq)
    assert "glimpse" in note

    solid = pv.DepartureSequence()
    solid.room(room="office", at=0.0)
    solid.room(room="kitchen", at=12.0)
    assert solid.phone_gone(102.0) is True
    assert "glimpse" not in pv.departure_note(solid)


def test_the_cost_ceiling_is_still_one_log_line():
    """THE ASSUMPTION THE WHOLE EXPERIMENT RESTS ON. A false departure is
    cheap only because nothing acts on one. If this fails, something has
    started consuming ``left`` and the glimpse step now needs a non-glimpse
    corroboration before it may complete the kitchen step.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "jarvis"
    # ``seq`` is the name every caller binds it to (app._departure_seq_*),
    # and ``departure_note`` is the only other door out of the class.
    # Deliberately NOT a bare ``.left``: ``DoorWatch.left()`` and
    # ``visionrig``'s frame counter share the word and mean nothing like it.
    rx = re.compile(r"\bseq\.left\b|\bdeparture_note\s*\(")
    users = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "presencevote.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in rx.finditer(text):
            line = text[:m.start()].count("\n") + 1
            users.append("%s:%d" % (path.name, line))
    assert len(users) <= 1, (
        "something new reads the departure sequence's verdict (%s); a false "
        "departure now costs more than a log line, so the glimpse step must "
        "require a non-glimpse corroboration" % ", ".join(users))
