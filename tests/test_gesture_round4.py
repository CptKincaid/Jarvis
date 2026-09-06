"""ROUND 4: the recall cliff, and the silence that came with it.

TWO DEFECTS, AND THE SECOND IS HALF THE FIRST.

(1) THE CLIFF. Round 3 measured recall at ONE point -- a 300-310 mm swing
ending at z = 440 -- and that point is the single most favourable spot on
a surface round 3 itself created. MEASURED over the whole surface (swings
270-330 mm, end depths 420-470 mm, both directions, the 5.5-8.0 fps band,
0/3/5 px landmark noise, 16 sub-frame phases, both finger orientations;
castgrid/test_r4_recall.py): round 3 was BELOW round 2 in 41 of the 84
cells, worst 0.7569 -> 0.1910.

THE CAUSE IS NOT THE DISTANCE RESCALE, which is what it looked like.
Ablated one change at a time: putting every distance bar back to its
round-2 value changed the surface by nothing at all, and so did switching
off ``closed_ratio_max`` and ``assoc_step_u``. Every failing cell dropped
for one reason and the machine says which -- "carried out of frame, NOT
FLUNG". It is the speed bar, and the reason is that the bar
was placed INSIDE the spread of his own gesture. MEASURED: his slow
deliberate throw peaks at 2.512-3.180 u/s across this surface, and round
3's bar was 2.65 -- which refuses every 270 and 280 mm swing and most
290s. 2.45 sits just below the whole spread.

(2) THE SILENCE. A throw refused by that bar produced a drop, a tone, and
nothing he could read. "It just did nothing" was the whole story he had.
A refusal now names the bar that refused it, carries the two numbers that
decided it, counts, and reaches ``screens_status()``.

NO CAMERA, NO CAPTURE DEVICE, NO FRAME. Every hand here is either a
synthetic 21-point row projected in-process or a hand-built observation.
"""
from __future__ import annotations

import dataclasses

import pytest

from jarvis import gesture as g
from jarvis.gesture import CastGesture, CastThresholds, HandObservation


class Clock:
    def __init__(self, t=100.0):
        self.t = float(t)

    def __call__(self):
        return self.t


def obs(cx, cy, *, closed=0.30, reach=3.0, unit=100.0):
    """A hand-built row. ``palm_diag`` is the unit round 2 measured in and
    ``unit`` the pose-corrected one; here they are the same so the sums are
    readable."""
    return HandObservation(palm_diag=unit, closed=closed, reach=reach,
                           cx=float(cx), cy=float(cy), conf=1.0, ok=True,
                           unit=unit, curled=0.5)


def carry(m, rows, *, fps=7.5, clock=None, frame0=0):
    """Feed rows one frame apart. Returns every event."""
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
    """Close a fist at the anchor, dwell it, then run ``tail``."""
    t = m.t
    rows = [obs(500, 300, closed=0.95, unit=unit)] * t.open_lookback_frames
    rows += [obs(500, 300, closed=0.30, unit=unit)] * (t.dwell_frames + 1)
    return carry(m, rows + list(tail), fps=fps, clock=clock)


# ======================================================= the refusal speaks
class TestARefusalIsNeverSilent:
    """He cannot debug a gesture that fails mutely. Every carry that ends
    without a throw now names the bar that refused it, in a CLOSED SET, and
    carries the two numbers that decided it."""

    def test_the_event_names_which_bar_refused_the_throw(self):
        clock = Clock()
        m = CastGesture(CastThresholds(), (1280, 720), now=clock,
                        preview_fps=7.5)
        # a slow carry a long way: far enough, never fast enough
        tail = [obs(500 + 12 * k, 300, closed=0.30) for k in range(1, 14)]
        tail += [obs(660, 300, closed=0.95)]        # ...and he opens it
        evs = grab_then(m, clock, tail)
        ends = [e for e in evs if e.kind in ("throw", "drop")]
        assert ends, evs
        last = ends[-1]
        print("\n  %s: why=%r refused=%r dist_u=%.3f speed_us=%.3f"
              % (last.kind, last.why, last.refused, last.dist_u,
                 last.speed_us))
        assert last.kind == "drop"
        assert last.refused == "speed", last.refused
        assert last.speed_us > 0.0, "the number that decided it must be there"

    def test_a_carry_that_never_travelled_says_distance_not_speed(self):
        clock = Clock()
        m = CastGesture(CastThresholds(), (1280, 720), now=clock,
                        preview_fps=7.5)
        tail = [obs(505, 300, closed=0.30), obs(510, 300, closed=0.30),
                obs(512, 300, closed=0.95)]
        evs = grab_then(m, clock, tail)
        last = [e for e in evs if e.kind in ("throw", "drop")][-1]
        print("\n  %s: why=%r refused=%r dist_u=%.3f"
              % (last.kind, last.why, last.refused, last.dist_u))
        assert last.kind == "drop" and last.refused == "distance"

    def test_every_refusal_code_is_in_a_closed_set(self):
        """The same discipline the cast verbs are held to: a code a log
        line and a status dict carry has to be enumerable."""
        assert "" in g.REFUSALS
        for code in ("speed", "distance", "look", "direction", "sector",
                     "cancel", "carry"):
            assert code in g.REFUSALS, code
        print("\n  REFUSALS = %s" % (g.REFUSALS,))

    def test_the_numbers_that_decided_it_are_on_the_event(self):
        e = g.CastEvent(kind="drop", at=1.0, frame=3, refused="speed",
                        speed_us=2.31)
        row = e.numbers_only()
        print("\n  %s" % row)
        assert row["refused"] == "speed"
        assert row["speed_us"] == 2.31
        from jarvis.visionrig import assert_numbers_only
        assert_numbers_only(row)

    def test_a_throw_that_lands_is_not_marked_refused(self):
        clock = Clock()
        m = CastGesture(CastThresholds(), (1280, 720), now=clock,
                        preview_fps=7.5)
        tail = [obs(500 + 60 * k, 300, closed=0.30) for k in range(1, 5)]
        tail += [obs(740, 300, closed=0.95)]
        evs = grab_then(m, clock, tail)
        thr = [e for e in evs if e.kind == "throw"]
        assert thr, [(e.kind, e.why) for e in evs]
        print("\n  throw: refused=%r speed_us=%.3f"
              % (thr[-1].refused, thr[-1].speed_us))
        assert thr[-1].refused == ""
        assert thr[-1].speed_us >= CastThresholds().throw_speed_us


# ============================================== the courier makes it reachable
class TestTheCourierCountsAndSaysSo:
    """A counter, a log line and the stage status -- the three places the
    brief names, because "it just did nothing" must never be the whole
    story."""

    def build(self):
        from tests.test_gesturecast import build
        return build()

    def test_the_stage_status_carries_the_refusals_and_the_bars(self):
        out = self.build()
        st = out.courier.screens_status()
        print("\n  short_throws=%r refused=%r" % (st.get("short_throws"),
                                                  st.get("refused")))
        print("  bars=%r" % (st.get("bars"),))
        assert "short_throws" in st and "refused" in st
        assert "last_refusal" in st
        bars = st["bars"]
        for key in ("throw_speed_us", "throw_release_u", "throw_exit_u",
                    "throw_lost_u", "fling_window_s", "fling_grant_s"):
            assert key in bars, key

    def test_a_refused_throw_is_counted_and_logged(self, caplog):
        import logging
        out = self.build()
        courier = out.courier
        ev = g.CastEvent(kind="drop", at=1.0, frame=9, refused="speed",
                         dist_u=0.94, speed_us=2.41, toward="left",
                         why="carried out of frame, not flung")
        with caplog.at_level(logging.INFO):
            courier.on_event(ev)
        st = courier.screens_status()
        print("\n  short_throws=%d refused=%s" % (st["short_throws"],
                                                  st["refused"]))
        print("  last_refusal=%s" % st["last_refusal"])
        text = "\n".join(r.getMessage() for r in caplog.records)
        print("  log:\n    %s" % text.replace("\n", "\n    "))
        assert st["short_throws"] == 1
        assert st["refused"].get("speed") == 1
        assert st["last_refusal"]["speed_us"] == 2.41
        assert "2.41" in text and "refused" in text

    def test_a_put_down_is_not_counted_as_a_short_throw(self):
        """A carry he ended deliberately is not a gesture that failed."""
        out = self.build()
        courier = out.courier
        courier.on_event(g.CastEvent(kind="drop", at=1.0, frame=9,
                                     refused="cancel", toward="down"))
        courier.on_event(g.CastEvent(kind="drop", at=2.0, frame=19,
                                     refused="carry", why="timeout"))
        st = courier.screens_status()
        print("\n  short_throws=%d refused=%s" % (st["short_throws"],
                                                  st["refused"]))
        assert st["short_throws"] == 0
        assert st["refused"] == {"cancel": 1, "carry": 1}


# ================================================== the bar itself
class TestTheSpeedBarIsWhereTheMeasurementPutIt:
    """The constant, and the law it has to keep. The surface itself is
    measured in castgrid/test_r4_recall.py, which is an instrument and
    costs minutes; this pins the value and the arithmetic behind it."""

    def test_the_bar_is_the_round_four_value(self):
        assert CastThresholds().throw_speed_us == pytest.approx(2.45)

    def test_the_fling_window_was_not_traded_away_to_get_it(self):
        """ROUND 3'S MEASUREMENT FIX IS KEPT. The window is still measured
        to the moment the throw would fire, and it is still the number in
        the field -- 0.55 s, floored at the loss grace. Recovering round 2's
        recall cost nothing here."""
        t = CastThresholds()
        assert t.fling_window_s == pytest.approx(0.55)
        m = CastGesture(t, (1280, 720), preview_fps=7.5)
        print("\n  fling grant at 7.5 fps: %.3f s" % m.fling_grant_s())
        assert m.fling_grant_s() == pytest.approx(0.55)

    def test_the_shipped_config_does_not_contradict_the_module(self):
        """THE DRIFT THIS ROUND FOUND. ``assistant_config.DEFAULTS`` still
        carried round 2's distance bars, and ``thresholds_from_options``
        applies them ON TOP of the module's, so on his actual box five of
        round 3's rescaled bars were silently put back:
        anchor_drift_u 0.52 -> 0.6, throw_release_u 0.86 -> 1.0,
        throw_exit_u 0.22 -> 0.25, throw_lost_u 0.43 -> 0.5,
        exit_step_u 0.30 -> 0.35. A documented value that the shipped file
        overrides is not a documented value."""
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
        print("\n  fields where assistant.json disagrees with the module: %s"
              % (drift or "none"))
        assert drift == [], drift


# ============================================== the off switch stays off
class TestTheOffSwitchIsHonest:
    """BLOCK A IS NOT CLOSED AND IS NOT TOUCHED. The gesture's false-fire
    rate is forty times its bar and cannot be brought down from the hand
    track alone. It ships OFF, and the next person to reach for that switch
    must read the number before they throw it."""

    def test_the_gesture_ships_off(self):
        from jarvis.assistant_config import DEFAULTS
        assert DEFAULTS["gesture"]["enabled"] is False

    def test_the_config_comment_carries_the_measured_rate_and_the_bar(self):
        """Not prose about being careful: the NUMBER, the BAR, and the one
        thing that is still owed."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "jarvis" / "assistant_config.py").read_text()
        start = src.index('"gesture": {')
        block = src[max(0, start - 4000):start + 200]
        for token in ("0.2362", "0.005", "wind-up"):
            assert token in block, token
        print("\n  the config comment carries the rate, the bar and the "
              "one question only he can answer")

    def test_the_module_docstring_carries_the_same_number(self):
        doc = g.__doc__ or ""
        for token in ("0.2362", "0.005", "wind-up"):
            assert token in doc, token
        print("  jarvis/gesture.py's docstring carries them too")
