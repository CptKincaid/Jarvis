"""ROUND 5: the number the bar actually used, and THE WIND-UP.

TWO THINGS, and the second is the one four rounds have waited for.

(1) THE REFUSAL DIAGNOSTIC REPORTED THE WRONG NUMBER. ``speed_us`` on a
refusal is ``_peak_us``, the fastest step of the WHOLE carry -- and the bar
is not compared against that. ``_was_flung`` asks whether the hand was at
``throw_speed_us`` INSIDE ``fling_grant_s`` of the moment the throw would
fire, so a carry with one quick correction early and a slow exit is refused
on speed while the diagnostic prints a number ABOVE the bar. This is the
instrument nominated for him to settle whether his own throw is inside the
measured surface, so a wrong number here is worse than none. The event now
carries ``fling_us`` -- the fastest CREDITED step inside the window the bar
was compared against -- and that is what the refusal quotes.

(2) THE WIND-UP. Asked directly on 2026-09-05 -- "when you throw, does your
hand pull back a little first, before it swings?" -- HE ANSWERED YES. It was
absent from every throw model here (reach, close, hold, swing), which is
exactly why it had to be asked rather than measured, and it is the one
signal a carry cannot fake at any speed: carrying something away never
reverses before it leaves.

IT IS NOT THE REVERSAL ROUND 3 DISPROVED. That was an APPROACH-VERSUS-EXIT
reversal and it scored nothing, because his hand comes in from his left in
both the throw and the carry. THIS is a small BACKWARD RETRACTION inside the
throw itself, after the dwell and before the fling, measured along the axis
the throw eventually took.

AND THE TRAP, which is the whole methodological risk: A GRID THAT INVENTS A
WIND-UP AND THEN DETECTS IT HAS PROVED NOTHING. So the measurement lives in
castgrid/ under four rules -- amplitude and duration SWEPT including values
too small to detect, every non-gesture family re-examined for motions that
also retract, recall reported as a function of amplitude, and his own
amplitude stated as UNKNOWN. This file pins the mechanism; castgrid/
measures what it is worth.

NO CAMERA, NO CAPTURE DEVICE, NO FRAME. Every hand here is a hand-built
observation or a synthetic 21-point row projected in-process.
"""
from __future__ import annotations

import math

import pytest

from jarvis import gesture as g
from jarvis.gesture import CastGesture, CastThresholds, HandObservation


class Clock:
    def __init__(self, t=100.0):
        self.t = float(t)

    def __call__(self):
        return self.t


def obs(cx, cy, *, closed=0.30, reach=3.0, unit=100.0):
    return HandObservation(palm_diag=unit, closed=closed, reach=reach,
                           cx=float(cx), cy=float(cy), conf=1.0, ok=True,
                           unit=unit, curled=0.5)


def carry(m, rows, *, fps=7.5, clock=None, frame0=0):
    out = []
    for i, row in enumerate(rows):
        e = m.update(() if row is None else (row,), 60.0, frame0 + i,
                     looking=True)
        if e is not None:
            out.append(e)
        if clock is not None:
            clock.t += 1.0 / fps
    return out


def grab_then(m, clock, tail, *, unit=100.0, fps=7.5):
    t = m.t
    rows = [obs(500, 300, closed=0.95, unit=unit)] * t.open_lookback_frames
    rows += [obs(500, 300, closed=0.30, unit=unit)] * (t.dwell_frames + 1)
    return carry(m, rows + list(tail), fps=fps, clock=clock)


def machine(clock, **kw):
    return CastGesture(CastThresholds(**kw), (1280, 720), now=clock,
                       preview_fps=7.5)


def ended(evs):
    ends = [e for e in evs if e.kind in ("throw", "drop")]
    assert ends, [(e.kind, e.why) for e in evs]
    return ends[-1]


# ================================= 1. the refusal quotes the right number
class TestTheRefusalQuotesTheNumberTheBarUsed:
    """THE INSTRUMENT HE IS BEING HANDED. ``screens_status()["refused"]``
    is what settles whether his real throw is inside the measured surface,
    and it was reporting the fastest step of the whole carry against a bar
    that was never compared with it."""

    def slow_exit_after_a_jerk(self, m, clock):
        """One quick correction -- shifting a grip on a mug -- and then a
        long slow carry out. The fling window has expired by the time the
        hand leaves, so this is refused on SPEED, and the peak of the whole
        carry is well above the bar."""
        tail = [obs(500 + 42, 300, closed=0.30)]          # the correction
        tail += [obs(542 + 7 * k, 300, closed=0.30) for k in range(1, 12)]
        tail += [obs(626, 300, closed=0.95)]              # opened, far enough
        return grab_then(m, clock, tail)

    def test_the_carry_peak_and_the_window_peak_are_different_numbers(self):
        clock = Clock()
        m = machine(clock)
        last = ended(self.slow_exit_after_a_jerk(m, clock))
        bar = CastThresholds().throw_speed_us
        print("\n  %s refused=%r" % (last.kind, last.refused))
        print("  carry peak speed_us = %.3f   (bar %.2f)"
              % (last.speed_us, bar))
        print("  inside the fling window fling_us = %.3f" % last.fling_us)
        assert last.refused == "speed", last.refused
        assert last.speed_us > bar, "the old number is above the bar"
        assert last.fling_us < bar, "the number the bar used is below it"

    def test_the_two_numbers_agree_with_the_decision_every_time(self):
        """THE LAW, and it is the only one worth having here: a refusal on
        speed means ``fling_us`` is BELOW the bar, and a throw means it is
        at or above. ``speed_us`` carries no such promise and never did."""
        clock = Clock()
        bar = CastThresholds().throw_speed_us
        rows = []
        for step in (7, 14, 22, 30, 42, 60, 80):
            clk = Clock()
            m = machine(clk)
            tail = [obs(500 + step * k, 300, closed=0.30) for k in range(1, 8)]
            tail += [obs(500 + step * 8, 300, closed=0.95)]
            last = ended(grab_then(m, clk, tail))
            rows.append((step, last.kind, last.refused, last.speed_us,
                         last.fling_us))
        print("\n  step px   kind   refused    speed_us  fling_us  (bar %.2f)"
              % bar)
        for step, kind, refused, sp, fl in rows:
            print("  %7d   %-6s %-9s %8.3f  %8.3f"
                  % (step, kind, refused or "-", sp, fl))
        for _step, kind, refused, _sp, fl in rows:
            if kind == "throw":
                assert fl >= bar
            elif refused == "speed":
                assert fl < bar
        assert clock is not None

    def test_the_number_reaches_the_status_and_the_log(self, caplog):
        import logging

        from tests.test_gesturecast import build
        out = build()
        ev = g.CastEvent(kind="drop", at=1.0, frame=9, refused="speed",
                         dist_u=0.94, speed_us=2.61, fling_us=1.88,
                         toward="left", why="carried out of frame, not flung")
        with caplog.at_level(logging.INFO):
            out.courier.on_event(ev)
        st = out.courier.screens_status()
        text = "\n".join(r.getMessage() for r in caplog.records)
        print("\n  last_refusal=%s" % st["last_refusal"])
        print("  log: %s" % text.strip())
        assert st["last_refusal"]["fling_us"] == 1.88
        assert "1.88" in text, "the refusal must quote the number the bar used"
        assert "2.61" in text, "...and the carry peak is still reported"

    def test_the_event_row_is_still_numbers_only(self):
        from jarvis.visionrig import assert_numbers_only
        row = g.CastEvent(kind="drop", at=1.0, frame=3, refused="speed",
                          speed_us=2.61, fling_us=1.88,
                          windup_u=0.31).numbers_only()
        print("\n  %s" % row)
        assert_numbers_only(row)
        assert row["fling_us"] == 1.88 and row["windup_u"] == 0.31


# ============================================== 2. the wind-up is MEASURED
class TestTheWindUpIsMeasured:
    """A small BACKWARD retraction immediately before the swing, measured
    along the axis the throw eventually took, in hand-units. It is a
    measurement first and a gate second, and the gate ships inert until the
    numbers say otherwise."""

    def swing(self, m, clock, *, back_px=0.0, back_frames=2, unit=100.0):
        """Hold at the anchor, retract ``back_px`` opposite the throw over
        ``back_frames``, then swing his-left and open."""
        tail = []
        for k in range(1, back_frames + 1):
            tail.append(obs(500 - back_px * k / back_frames, 300,
                            closed=0.30, unit=unit))
        start = 500 - back_px
        tail += [obs(start + 60 * k, 300, closed=0.30, unit=unit)
                 for k in range(1, 5)]
        tail += [obs(start + 300, 300, closed=0.95, unit=unit)]
        return grab_then(m, clock, tail, unit=unit)

    def test_a_throw_with_no_windup_measures_zero(self):
        clock = Clock()
        m = machine(clock)
        last = ended(self.swing(m, clock, back_px=0.0))
        print("\n  no retraction: kind=%s windup_u=%.4f"
              % (last.kind, last.windup_u))
        assert last.kind == "throw"
        assert last.windup_u == pytest.approx(0.0, abs=1e-6)

    def test_a_retraction_before_the_swing_is_measured_in_hand_units(self):
        """25 px against a 100 px hand-unit is 0.25 u, and that is what it
        must read -- not the distance travelled, not the whole excursion."""
        clock = Clock()
        m = machine(clock)
        last = ended(self.swing(m, clock, back_px=25.0))
        print("\n  25 px back on a 100 px unit: windup_u=%.4f"
              % last.windup_u)
        assert last.kind == "throw"
        assert last.windup_u == pytest.approx(0.25, abs=0.02)

    def test_the_measurement_scales_with_the_hand_unit(self):
        """It is a RATIO, so the same millimetres nearer the lens read the
        same. A wind-up quoted in pixels would move with his reach."""
        seen = []
        for unit in (80.0, 100.0, 140.0):
            clock = Clock()
            m = machine(clock)
            last = ended(self.swing(m, clock, back_px=0.25 * unit,
                                    unit=unit))
            seen.append((unit, last.windup_u))
        print("\n  unit px -> windup_u: %s"
              % ["%.0f -> %.4f" % r for r in seen])
        for _u, w in seen:
            assert w == pytest.approx(0.25, abs=0.03)

    def test_a_pull_back_AFTER_the_furthest_point_is_not_a_wind_up(self):
        """The retraction has to come BEFORE the swing. A hand that goes
        out and comes back is ``exit-return``, which is one of the
        attacker's own non-gesture families, and it must not be able to
        buy itself a wind-up by returning."""
        clock = Clock()
        m = machine(clock)
        tail = [obs(500 + 60 * k, 300, closed=0.30) for k in range(1, 5)]
        tail += [obs(700, 300, closed=0.30), obs(640, 300, closed=0.30)]
        tail += [obs(620, 300, closed=0.95)]
        last = ended(carry(m, [obs(500, 300, closed=0.95)] * 12
                           + [obs(500, 300, closed=0.30)] * 4 + tail,
                           clock=clock))
        print("\n  out and back: windup_u=%.4f (kind=%s)"
              % (last.windup_u, last.kind))
        assert last.windup_u == pytest.approx(0.0, abs=1e-6)

    def test_the_measurement_is_on_every_carry_end_not_only_throws(self):
        clock = Clock()
        m = machine(clock)
        tail = [obs(500 - 20, 300, closed=0.30), obs(500 - 25, 300,
                                                     closed=0.30)]
        tail += [obs(500 + 8 * k, 300, closed=0.30) for k in range(1, 9)]
        tail += [obs(580, 300, closed=0.95)]
        last = ended(grab_then(m, clock, tail))
        print("\n  a refused carry still reports windup_u=%.4f (refused=%r)"
              % (last.windup_u, last.refused))
        assert last.kind == "drop"
        assert last.windup_u > 0.2


class TestTheWindUpGate:
    """A SECOND INDEPENDENT SIGNAL, ANDED with the others -- never ORed,
    for the same reason the look gate is anded: a throw is an outward
    irreversible action and every signal has to agree."""

    def swing(self, m, clock, *, back_px=0.0):
        tail = [obs(500 - back_px / 2, 300, closed=0.30),
                obs(500 - back_px, 300, closed=0.30)]
        start = 500 - back_px
        tail += [obs(start + 60 * k, 300, closed=0.30) for k in range(1, 5)]
        tail += [obs(start + 300, 300, closed=0.95)]
        return grab_then(m, clock, tail)

    def test_the_gate_is_inert_at_the_shipped_default(self):
        """IT SHIPS OFF. The measurement is the deliverable; the gate is
        only worth arming if the numbers say it separates, and
        castgrid/test_r5_windup.py is where that is settled."""
        t = CastThresholds()
        print("\n  shipped windup_min_u = %.3f" % t.windup_min_u)
        assert t.windup_min_u == 0.0
        clock = Clock()
        m = machine(clock)
        last = ended(self.swing(m, clock, back_px=0.0))
        assert last.kind == "throw", "a bar of 0 may refuse nothing"

    def test_a_bar_above_his_retraction_refuses_and_says_which_bar(self):
        clock = Clock()
        m = machine(clock, windup_min_u=0.30)
        last = ended(self.swing(m, clock, back_px=10.0))     # 0.10 u
        print("\n  bar 0.30, retraction 0.10 u: kind=%s refused=%r why=%r"
              % (last.kind, last.refused, last.why))
        assert last.kind == "drop"
        assert last.refused == "windup"
        assert "wind" in last.why

    def test_a_bar_below_his_retraction_lets_it_through(self):
        clock = Clock()
        m = machine(clock, windup_min_u=0.30)
        last = ended(self.swing(m, clock, back_px=45.0))     # 0.45 u
        print("\n  bar 0.30, retraction 0.45 u: kind=%s windup_u=%.3f"
              % (last.kind, last.windup_u))
        assert last.kind == "throw" and last.refused == ""

    def test_windup_is_in_the_closed_set_of_refusals(self):
        assert "windup" in g.REFUSALS
        print("\n  REFUSALS = %s" % (g.REFUSALS,))

    def test_the_gate_cannot_turn_a_non_throw_into_a_throw(self):
        """ANDED, not ORed. A carry that fails the speed bar is still a
        drop however hard it pulled back first."""
        clock = Clock()
        m = machine(clock, windup_min_u=0.10)
        tail = [obs(500 - 30, 300, closed=0.30)]
        tail += [obs(470 + 9 * k, 300, closed=0.30) for k in range(1, 14)]
        tail += [obs(600, 300, closed=0.95)]
        last = ended(grab_then(m, clock, tail))
        print("\n  windup %.3f but slow: kind=%s refused=%r"
              % (last.windup_u, last.kind, last.refused))
        assert last.windup_u > 0.10
        assert last.kind == "drop" and last.refused == "speed"

    def test_the_status_and_the_bars_carry_it(self):
        clock = Clock()
        m = machine(clock, windup_min_u=0.20)
        st = m.status()
        print("\n  status windup_u=%r" % st.get("windup_u"))
        assert "windup_u" in st
        from tests.test_gesturecast import build
        bars = build().courier.screens_status()["bars"]
        print("  bars windup_min_u=%r" % bars.get("windup_min_u"))
        assert "windup_min_u" in bars


class TestTheWindUpIsHonestAboutWhatItIsNot:
    """The rule the previous adversary set and it is right: a grid that
    invents a wind-up and then detects it has proved nothing. These pin the
    two claims the module is allowed to make."""

    def test_the_module_says_his_own_amplitude_is_unknown(self):
        doc = g.__doc__ or ""
        for token in ("wind-up", "UNKNOWN"):
            assert token in doc, token
        print("\n  the docstring states his amplitude is unknown")

    def test_the_measurement_survives_a_lost_frame(self):
        """A dropped detection inside the retraction must not zero it --
        the carry survives lost_grace_frames and so must the path."""
        clock = Clock()
        m = machine(clock)
        tail = [obs(485, 300, closed=0.30), None,
                obs(470, 300, closed=0.30)]
        tail += [obs(470 + 60 * k, 300, closed=0.30) for k in range(1, 5)]
        tail += [obs(770, 300, closed=0.95)]
        last = ended(grab_then(m, clock, tail))
        print("\n  one frame lost inside the retraction: windup_u=%.4f"
              % last.windup_u)
        assert last.windup_u == pytest.approx(0.30, abs=0.03)

    def test_an_ambiguous_frame_is_not_wind_up_evidence(self):
        """The same discipline the fling has: a frame the association
        could not be sure about is followed but not credited. A wind-up
        made of his OTHER hand appearing is not a wind-up."""
        clock = Clock()
        m = machine(clock)
        rows = [obs(500, 300, closed=0.95)] * 12
        rows += [obs(500, 300, closed=0.30)] * 4
        out = carry(m, rows, clock=clock)
        assert any(e.kind == "grab" for e in out)
        # two hands at the same scale: ambiguous, and the nearer one is at
        # a big backward offset
        two = (obs(470, 300, closed=0.30), obs(500, 300, closed=0.30))
        m.update(two, 60.0, 99, looking=True)
        clock.t += 1.0 / 7.5
        print("\n  ambiguous frame: windup so far %.4f"
              % m.status()["windup_u"])
        assert m.status()["windup_u"] == pytest.approx(0.0, abs=1e-6)


def test_the_hand_unit_is_still_the_only_scale():
    """No new millimetre constant crept in: the wind-up is in hand-units
    like every other distance bar, because the unit itself is guessed
    anthropometry and a mm figure here would be a number this rig cannot
    support."""
    t = CastThresholds()
    assert isinstance(t.windup_min_u, float)
    assert math.isfinite(t.windup_min_u)
    src = (g.__file__)
    with open(src, "r", encoding="utf-8") as fh:
        code = "\n".join(ln for ln in fh.read().splitlines()
                         if not ln.lstrip().startswith("#"))
    assert "windup_mm" not in code
    print("\n  the wind-up is in hand-units, like every other bar")


# ================================== the numbers he reads are THIS round's
class TestTheShippedNumbersAreThisRounds:
    """A documented value that the shipped file contradicts is not a
    documented value -- round 4's own finding, applied to round 5's rate.
    Both places that carry the measured false-fire rate now carry the one
    measured on the HONEST grid, not the preserved-only number that made
    the wind-up look like a solution."""

    def config_block(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "jarvis" / "assistant_config.py").read_text()
        start = src.index('"gesture": {')
        return src[max(0, start - 5000):start + 400]

    def test_the_config_carries_the_round_five_rate_and_the_bar(self):
        block = self.config_block()
        for token in ("0.4986", "0.2362", "0.005", "wind-up", "0.000",
                      "--windup"):
            assert token in block, token
        print("\n  the config carries 0.4986 (262 families), 0.2362 (the")
        print("  preserved 172), the 0.005 bar, and the instrument he runs")

    def test_the_module_docstring_carries_the_same_numbers(self):
        doc = g.__doc__ or ""
        for token in ("0.4986", "0.2362", "0.005", "wind-up", "UNKNOWN",
                      "gesture_selfcheck.py"):
            assert token in doc, token
        print("  jarvis/gesture.py's docstring carries them too")

    def test_the_gesture_still_ships_off(self):
        from jarvis.assistant_config import DEFAULTS
        assert DEFAULTS["gesture"]["enabled"] is False
        assert DEFAULTS["gesture"]["windup_min_u"] == 0.0
        print("  gesture.enabled False, windup_min_u 0.0 (inert)")

    def test_the_config_and_the_module_still_agree_field_by_field(self):
        """The drift test round 4 added, re-run over the new field."""
        import dataclasses

        from jarvis.assistant_config import DEFAULTS
        from jarvis.handstage import thresholds_from_options

        def get_option(key, default=None):
            node = DEFAULTS
            for part in key.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return default
            return node

        base = CastThresholds()
        live = thresholds_from_options(get_option)
        drift = [(f.name, getattr(base, f.name), getattr(live, f.name))
                 for f in dataclasses.fields(base)
                 if getattr(base, f.name) != getattr(live, f.name)]
        print("  fields where assistant.json disagrees: %s" % (drift or "none"))
        assert drift == []

    def test_the_instrument_he_runs_exists_and_takes_the_flag(self):
        """The brief's rule 4: build the numbers-only instrument and print
        the exact command. It has to be real."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "scripts" / "gesture_selfcheck.py").read_text()
        assert '"--windup"' in src
        assert "_report_windup" in src
        assert "NO IMAGE DATA LEAVES THIS SCRIPT" in src
        print("  scripts/gesture_selfcheck.py --seconds 30 --windup")
