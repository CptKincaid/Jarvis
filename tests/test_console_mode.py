"""The console's one state machine — standby, ambient, power-up.

Everything under test is a pure function of its inputs or a driver whose
every seam is injected, so no Tk widget is created anywhere in this file
and no clock is read that the test did not set.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from jarvis.ui import ambient
from jarvis.ui import console_mode as cm


# ------------------------------------------------------------ next_mode
def test_a_live_turn_pins_the_console_awake_however_idle_the_desk_is():
    assert cm.next_mode(idle_s=9999, busy=True) == cm.ACTIVE


def test_a_ringing_alarm_is_never_shown_behind_a_room_clock():
    assert cm.next_mode(idle_s=9999, alarm=True) == cm.ACTIVE


def test_an_unknown_idle_time_never_guesses_that_nobody_is_there():
    assert cm.next_mode(idle_s=None) == cm.ACTIVE
    assert cm.next_mode(idle_s="not a number") == cm.ACTIVE


def test_the_thresholds_walk_active_to_ambient_to_standby():
    kw = dict(ambient_after_s=45.0, standby_after_s=720.0)
    assert cm.next_mode(idle_s=10, **kw) == cm.ACTIVE
    assert cm.next_mode(idle_s=45, **kw) == cm.AMBIENT
    assert cm.next_mode(idle_s=719, **kw) == cm.AMBIENT
    assert cm.next_mode(idle_s=720, **kw) == cm.STANDBY


def test_either_half_can_be_switched_off_without_disabling_the_other():
    assert cm.next_mode(idle_s=9999, standby=False) == cm.AMBIENT
    assert cm.next_mode(idle_s=9999, ambient=False) == cm.STANDBY
    assert cm.next_mode(idle_s=9999, ambient=False, standby=False) == cm.ACTIVE


# -------------------------------------------------------------- dimming
def test_the_working_day_is_fully_lit_and_the_small_hours_are_not():
    assert cm.night_curve(13) == 1.0
    assert cm.night_curve(3) == 0.0
    assert 0.0 < cm.night_curve(22) < 1.0        # on the evening shoulder


def test_the_night_curve_has_no_step_anywhere_on_its_shoulders():
    for hour in (19.9, 20.0, 20.1, 1.9, 2.0, 6.9, 7.0, 8.9, 9.0):
        a, b = cm.night_curve(hour - 0.05), cm.night_curve(hour + 0.05)
        assert abs(a - b) < 0.2, hour


def test_active_is_never_dimmed_and_standby_never_below_the_floor():
    assert cm.dim_factor(cm.ACTIVE, 3.0) == 1.0
    for hour in range(24):
        f = cm.dim_factor(cm.STANDBY, float(hour), floor=0.35)
        assert 0.35 <= f <= 1.0


def test_the_ambient_slab_stays_readable_at_any_hour():
    assert cm.dim_factor(cm.AMBIENT, 3.0) == cm.dim_factor(cm.AMBIENT, 13.0)


def test_dimming_blends_toward_the_ground_and_never_touches_a_photo():
    assert cm.dim("#ffffff", 1.0) == "#ffffff"
    assert cm.dim("#ffffff", 0.0, ground="#000000") == "#000000"
    assert cm.dim("#ffffff", 0.5, ground="#000000") == "#7f7f7f"
    assert cm.dim("not a colour", 0.5) == "not a colour"


# ---------------------------------------------------------------- drift
def test_the_burn_in_walk_stays_inside_its_radius():
    for t in range(0, 4000, 7):
        dx, dy = cm.drift_offset(float(t), px_per_min=3.0, radius=40)
        assert abs(dx) <= 40 and abs(dy) <= 40


def test_the_walk_is_a_pure_function_of_elapsed_time_not_an_accumulator():
    """A late tick must land where it should; an incrementing offset would
    have drifted by however many ticks were missed."""
    assert cm.drift_offset(600.0) == cm.drift_offset(600.0)
    assert cm.drift_offset(0.0) != cm.drift_offset(300.0)


def test_a_zero_radius_or_speed_simply_does_not_drift():
    assert cm.drift_offset(500.0, radius=0) == (0, 0)
    assert cm.drift_offset(500.0, px_per_min=0) == (0, 0)


def test_the_walk_is_not_a_circle_retracing_one_ring_of_pixels():
    ys = {cm.drift_offset(float(t))[1] for t in range(0, 3000, 60)}
    assert len(ys) > 8


# -------------------------------------------------------------- powerup
def test_the_sweep_runs_once_a_day_off_the_same_date_latch():
    now = datetime(2026, 8, 30, 8, 0)
    assert cm.powerup_due("", now=now) is True
    assert cm.powerup_due("2026-08-30", now=now) is False
    assert cm.powerup_due("2026-08-29", now=now) is True


def test_a_short_break_is_not_an_overnight_gap():
    now = datetime(2026, 8, 30, 8, 0)
    assert cm.powerup_due("", now=now, idle_s=600, gap_h=6) is False
    assert cm.powerup_due("", now=now, idle_s=8 * 3600, gap_h=6) is True


def test_without_a_desk_probe_the_first_wake_of_the_day_is_the_signal():
    now = datetime(2026, 8, 30, 8, 0)
    assert cm.powerup_due("", now=now, idle_s=None) is True


def test_the_reveal_schedule_stages_every_panel_then_settles():
    sched = cm.reveal_schedule(3, step_ms=100, hold_ms=500)
    assert [c for _, c in sched] == [1, 2, 3, None]
    assert [d for d, _ in sched] == [0, 100, 200, 800]


def test_no_panels_still_ends_with_the_settle_stage():
    assert cm.reveal_schedule(0) == [(cm.POWERUP_HOLD_MS, None)]


# --------------------------------------------------------- the idle seam
def test_the_services_seam_wins_over_everything_else():
    services = SimpleNamespace(desk_idle_s=lambda: 12.0)
    assert cm.resolve_idle_fn(services)() == 12.0


def test_the_probe_is_cached_off_the_tk_thread_and_read_as_an_attribute():
    """The desk-presence probe shells out to gdbus; a subprocess spawn on
    the frame loop is a dropped avatar frame. DeskWatch is what keeps the
    mode tick to an attribute read."""
    calls = []

    def probe():
        calls.append(1)
        return 42.0
    watch = cm.DeskWatch(probe, interval=0.05)
    assert watch.read() is None            # nothing polled yet
    assert watch.poll() == 42.0
    assert watch.read() == 42.0 and calls == [1]   # read does NOT probe


def test_a_failing_probe_caches_unknown_and_never_away():
    def boom():
        raise RuntimeError("gdbus is gone")
    watch = cm.DeskWatch(boom)
    assert watch.poll() is None and watch.read() is None
    assert cm.next_mode(idle_s=watch.read()) == cm.ACTIVE


def test_the_desk_watch_starts_and_joins_cleanly():
    watch = cm.DeskWatch(lambda: 1.0, interval=0.05)
    watch.start()
    assert watch.running
    watch.stop()
    assert not watch.running
    watch.stop()                           # idempotent: quit may arrive twice


def test_a_desk_watch_with_no_probe_never_starts_a_thread():
    watch = cm.DeskWatch(None)
    watch.start()
    assert not watch.running and watch.read() is None


def test_without_the_seam_a_provider_is_still_resolved():
    """Either a desk module or the XScreenSaver fallback — the feature must
    not be dark on the real box because two changes landed in the other
    order."""
    fn = cm.resolve_idle_fn(SimpleNamespace())
    assert callable(fn)


# ------------------------------------------------------------ the driver
class FakeAfter:
    """Collects scheduled callbacks instead of running a Tk loop."""

    def __init__(self):
        self.jobs = []

    def __call__(self, ms, fn):
        self.jobs.append((ms, fn))

    def run_all(self):
        jobs, self.jobs = list(self.jobs), []
        for _ms, fn in jobs:
            fn()


def driver(idle=0.0, **kw):
    after = FakeAfter()
    seen = {"mode": [], "dim": [], "drift": [], "stage": []}
    opts = kw.pop("opts", {})
    modes = cm.ConsoleModes(
        after=after,
        idle_fn=(idle if callable(idle) else (lambda: idle)),
        on_mode=seen["mode"].append,
        on_dim=seen["dim"].append,
        on_drift=lambda dx, dy: seen["drift"].append((dx, dy)),
        on_stage=seen["stage"].append,
        option=lambda k, d=None: opts.get(k, d),
        clock=kw.pop("clock", lambda: 0.0),
        now=kw.pop("now", lambda: datetime(2026, 8, 30, 13, 0)),
        **kw)
    return modes, after, seen


def test_the_driver_walks_into_standby_and_reports_it_once():
    modes, _after, seen = driver(idle=9999)
    assert modes.tick() == cm.STANDBY
    assert seen["mode"] == [cm.STANDBY]
    modes.tick()
    assert seen["mode"] == [cm.STANDBY]           # no repeat for no change


def test_activity_wakes_the_console_instantly_and_bumps_the_generation():
    modes, _after, seen = driver(idle=9999)
    modes.tick()
    gen = modes.generation
    modes.note_activity()
    assert modes.mode == cm.ACTIVE and modes.generation == gen + 1
    assert seen["mode"][-1] == cm.ACTIVE


def test_a_probe_that_raises_leaves_the_console_awake():
    def boom():
        raise RuntimeError("gdbus is gone")
    modes, _after, _seen = driver(idle=boom)
    assert modes.tick() == cm.ACTIVE


def test_a_busy_turn_beats_the_idle_clock():
    after = FakeAfter()
    modes = cm.ConsoleModes(after=after, idle_fn=lambda: 9999,
                            busy_fn=lambda: True)
    assert modes.tick() == cm.ACTIVE


def test_drift_is_only_applied_in_standby_and_zeroed_on_the_way_out():
    ticks = iter([0.0, 0.0, 600.0, 600.0, 600.0, 600.0])
    modes, _after, seen = driver(idle=9999, clock=lambda: next(ticks))
    modes.tick()                       # -> standby, drift from t=0
    modes.tick()                       # still standby, 600 s later
    assert seen["drift"][-1] != (0, 0)
    modes.note_activity()
    assert seen["drift"][-1] == (0, 0)


def test_stop_restores_the_console_and_cannot_leave_it_dimmed():
    modes, _after, seen = driver(idle=9999)
    modes.tick()
    modes.stop()
    assert modes.mode == cm.ACTIVE
    assert seen["dim"][-1] == 1.0
    assert seen["drift"][-1] == (0, 0)


def test_the_sweep_reveals_every_panel_then_settles():
    modes, after, seen = driver()
    modes.power_up(3)
    after.run_all()
    assert seen["stage"] == [1, 2, 3, None]
    assert modes.sweeping is False


def test_the_first_thing_he_says_cancels_the_rest_of_the_sweep():
    modes, after, seen = driver()
    modes.power_up(4)
    modes.note_activity()               # the reply wins
    after.run_all()
    # The cancel itself delivers the settle (None); every stage the sweep
    # had still queued is swallowed by the generation bump.
    assert seen["stage"] == [None]
    assert modes.sweeping is False


def test_something_to_show_wakes_the_console_without_killing_the_sweep():
    """A reply, a briefing card or an alarm modal must not be drawn behind
    the ambient slab — but the sweep is MEANT to play under the morning
    briefing, so only what he did cancels it."""
    modes, after, seen = driver(idle=9999)
    modes.tick()
    assert modes.mode == cm.STANDBY
    gen = modes.generation
    modes.note_output()
    assert modes.mode == cm.ACTIVE and modes.generation == gen
    modes.power_up(2)
    modes.note_output()                 # the briefing lands mid-sweep
    after.run_all()
    assert seen["stage"] == [1, 2, None]


def test_the_sweep_can_be_switched_off():
    modes, after, seen = driver(opts={"console.powerup": False})
    modes.power_up(3)
    after.run_all()
    assert seen["stage"] == []


def test_config_thresholds_are_read_through_the_option_seam():
    modes, _after, _seen = driver(idle=120,
                                  opts={"console.standby_after_min": 1})
    assert modes.tick() == cm.STANDBY


# ------------------------------------------------------ the room slab
def room(**kw):
    base = {"playing": "Kind of Blue — Miles Davis", "next": "Biosensors 10:00",
            "due": "Lab 3 report tonight", "temp": "72°", "arc": "evening",
            "presence": "", "quiet": "", "gpu": 0.12}
    base.update(kw)
    return base


def test_the_ambient_band_shows_the_room_and_hides_what_is_not_known():
    rows = dict(ambient.room_rows(room()))
    # Upper case on purpose since 2026-08-31 (his request): the labels
    # always were, and a panel read across a room wants both halves the
    # same weight.
    assert rows["NOW"].startswith("KIND OF BLUE")
    assert "WHERE" not in rows              # presence is unconfigured


def test_a_configured_presence_earns_its_row():
    assert "WHERE" in dict(ambient.room_rows(room(presence="away")))


def test_quiet_hours_are_the_palette_and_never_a_row():
    rows = dict(ambient.room_rows(room(quiet="quiet hours until 7:00 am")))
    assert not any("quiet" in str(v).lower() for v in rows.values())
    assert ambient.slab_tone(room(quiet="quiet hours until 7")) == "quiet"


def test_gpu_load_is_a_fraction_for_a_bar_not_a_number_for_a_row():
    assert "GPU" not in dict(ambient.room_rows(room()))
    assert ambient.gpu_fraction(room(gpu=0.5)) == 0.5
    assert ambient.gpu_fraction(room(gpu=None)) is None
    assert ambient.gpu_fraction(room(gpu="nonsense")) is None
    assert ambient.gpu_fraction(room(gpu=4.0)) == 1.0


def test_standby_drops_what_nobody_reads_at_three_in_the_morning():
    rows = dict(ambient.standby_rows(room()))
    assert set(rows) == {"NEXT", "DUE", "OUTSIDE"}


def test_the_room_clock_has_no_seconds_hand_and_no_leading_zero():
    assert ambient.clock_text(datetime(2026, 8, 30, 14, 5)) == "2:05"
    assert ambient.clock_text(datetime(2026, 8, 30, 0, 5)) == "12:05"
    assert ambient.meridiem(datetime(2026, 8, 30, 0, 5)) == "AM"


def test_the_date_line_leads_with_the_day_name():
    assert ambient.date_text(datetime(2026, 8, 30)).startswith("SUNDAY")


def test_rows_truncate_rather_than_draw_off_the_bottom_edge():
    rows = [("A", "1"), ("B", "2"), ("C", "3")]
    assert ambient.visible_rows(rows, height=100, row_h=26, top=40) == rows[:2]
    assert ambient.visible_rows(rows, height=10, row_h=26, top=40) == []


def test_an_empty_room_renders_nothing_rather_than_placeholder_dashes():
    assert ambient.room_rows({}) == []
    assert ambient.standby_rows(None) == []
    assert ambient.slab_tone(None) == "normal"


# ---------------------------------------------- regressions (review a/b)
def test_the_desk_idle_fallback_finds_the_module_that_exports_the_probe():
    """n=15. The fallback used to read `desk_idle_s` off `jarvis.desk`, a
    module that has only ever exported `idle_seconds`; the gdbus probe of
    that name lives in `jarvis.deskpresence`. Both documented preferences
    were therefore dead and every standby decision silently fell through to
    XScreenSaver."""
    from jarvis import deskpresence

    fn = cm.resolve_idle_fn(SimpleNamespace())
    assert fn is deskpresence.desk_idle_s
    assert fn is not cm.xss_idle_s


def test_the_second_module_answers_when_deskpresence_is_missing(monkeypatch):
    """n=15. The chain must survive either merge order, so jarvis.desk's
    own probe is the next candidate — under its real name."""
    from jarvis import desk

    real = cm.importlib.import_module

    def only_desk(name):
        if name == "jarvis.deskpresence":
            raise ImportError("not merged yet")
        return real(name)
    monkeypatch.setattr(cm.importlib, "import_module", only_desk)
    assert cm.resolve_idle_fn(SimpleNamespace()) is desk.idle_seconds


def test_with_no_desk_module_at_all_the_room_clock_still_ticks(monkeypatch):
    def nothing(name):
        raise ImportError(name)
    monkeypatch.setattr(cm.importlib, "import_module", nothing)
    assert cm.resolve_idle_fn(SimpleNamespace()) is cm.xss_idle_s


def test_a_barge_in_mid_sweep_unlatches_the_partial_reveal():
    """n=24. The reveal count is a latch: BoardWindow._revealed and the room
    slab's _reveal keep drawing only the first N panels until someone sends
    None. Cancelling the sweep by bumping the generation swallowed the
    schedule's terminal None along with everything else, so a barge-in
    halfway through the power-up left the Board short of panels until the
    NEXT DAY's once-a-day sweep."""
    modes, after, seen = driver()
    modes.power_up(4)
    jobs, after.jobs = after.jobs, []
    jobs[0][1]()                        # the first panel is revealed
    jobs[1][1]()                        # and the second
    assert seen["stage"] == [1, 2]
    modes.note_activity()               # he speaks over the sweep
    for _ms, fn in jobs[2:]:
        fn()
    after.run_all()
    assert seen["stage"][-1] is None    # the latch is released
    assert seen["stage"] == [1, 2, None]
    assert modes.sweeping is False


def test_ordinary_utterances_do_not_redraw_the_board_every_time():
    """n=24. note_activity runs on EVERY utterance; emitting the settle
    unconditionally would force a Board + room-slab redraw each turn."""
    modes, after, seen = driver()
    modes.power_up(2)
    after.run_all()
    assert seen["stage"] == [1, 2, None]
    modes.note_activity()
    modes.note_activity()
    assert seen["stage"] == [1, 2, None]


def test_a_raising_stage_callback_never_escapes_a_cancel():
    def boom(_count):
        raise RuntimeError("the board is gone")
    after = FakeAfter()
    modes = cm.ConsoleModes(after=after, on_stage=boom,
                            option=lambda k, d=None: d)
    modes.power_up(3)
    modes.note_activity()               # must not raise through the cancel
    assert modes.sweeping is False
