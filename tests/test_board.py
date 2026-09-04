"""The Board's state layer (jarvis/board.py) — pure, provider-injected, no Tk.

Every provider here is a fake: a health snapshot is a SimpleNamespace, the
Canvas half is a list of strings, the turn ledger a list of dicts. That is
the whole point of the module — the live sources spawn nvidia-smi and make
REST calls, and a unit test that touched either would flake.

The layout half (jarvis/ui/board.py) is tested for its PURE helpers only;
no Tk widget is constructed anywhere in this file.
"""
from __future__ import annotations

from types import SimpleNamespace

from jarvis import board
from jarvis.ui import board as ui_board


def snap(**kw):
    base = dict(mem_total_gb=120.0, mem_avail_gb=80.0, load1=1.2,
                gpu={"util_pct": 12.0, "temp_c": 43.0, "power_w": 20.0},
                disks=[], top=[], trainers=[])
    base.update(kw)
    return SimpleNamespace(**base)


def turn(wait, outcome="audio"):
    return {"outcome": outcome, "wait": wait}


# ----------------------------------------------------------- sparkline
def test_sparkline_normalises_against_its_own_peak():
    assert board.sparkline([1, 2, 4], width=3) == (0.25, 0.5, 1.0)


def test_a_short_series_pads_on_the_left_so_it_draws_from_the_right():
    spark = board.sparkline([2.0], width=4)
    assert spark == (0.0, 0.0, 0.0, 1.0)


def test_a_long_series_buckets_down_to_the_width():
    spark = board.sparkline(list(range(1, 101)), width=10)
    assert len(spark) == 10 and spark[-1] == 1.0 and spark[0] < spark[-1]


def test_an_all_zero_series_is_flat_not_a_division_by_zero():
    assert board.sparkline([0, 0, 0], width=3) == (0.0, 0.0, 0.0)


def test_sparkline_of_nothing_is_empty():
    assert board.sparkline([]) == () and board.sparkline(None) == ()


def test_aborted_turns_are_not_waits_anybody_sat_through():
    recs = [turn(1.0), turn(9.0, "abort"), turn(2.0, "superseded"), turn(3.0)]
    assert board.wait_series(recs) == [1.0, 3.0]


def test_a_record_without_a_wait_has_not_happened_yet():
    assert board.wait_series([{"outcome": "audio"}, turn(1.5)]) == [1.5]


# ------------------------------------------------------------ formatting
def test_durations_read_at_the_scale_they_live_at():
    assert board.fmt_secs(0.84) == "840ms"
    assert board.fmt_secs(2.44) == "2.4s"
    assert board.fmt_secs(65) == "1:05"
    assert board.fmt_secs(3720) == "1h02"
    assert board.fmt_secs(None) == "--"


def test_due_times_are_relative_because_the_board_is_glanced_at():
    now = 1_000_000.0
    assert board.fmt_when(now + 30, now) == "now"
    assert board.fmt_when(now + 12 * 60, now) == "in 12m"
    assert board.fmt_when(now + 3 * 3600, now) == "in 3h"
    assert board.fmt_when(now + 48 * 3600, now) == "in 2d"
    assert board.fmt_when(now - 600, now) == "past"


# --------------------------------------------------------------- vitals
def test_vitals_reads_a_snapshot_into_rows_and_a_sentence():
    p = board.vitals_panel(snap())
    assert dict(p.rows)["MEMORY"] == "80G free of 120G"
    assert p.tone == "ok" and "80G of 120G free" in p.line


def test_low_memory_goes_amber_and_critical_goes_red():
    assert board.vitals_panel(snap(mem_avail_gb=12.0)).tone == "warn"
    assert board.vitals_panel(snap(mem_avail_gb=4.0)).tone == "error"


def test_a_trainer_is_named_and_tints_the_panel():
    p = board.vitals_panel(snap(trainers=[SimpleNamespace(name="python",
                                                          rss_gb=38.0)]))
    assert dict(p.rows)["TRAINER"] == "python 38G"
    assert p.tone == "warn" and "1 trainer running" in p.line


def test_an_unreadable_snapshot_is_an_honest_dark_panel():
    p = board.vitals_panel(None)
    assert p.tone == "off" and p.rows == [] and "can't read" in p.line


# ---------------------------------------------------------------- turns
def test_the_turn_panel_reports_the_median_not_the_mean():
    p = board.turns_panel([turn(1.0), turn(1.0), turn(1.0), turn(30.0)])
    assert dict(p.rows)["MEDIAN"] == "1.0s"
    assert dict(p.rows)["WORST"] == "30.0s"
    assert p.spark and len(p.spark) == board.SPARK_WIDTH


def test_a_slow_turn_tints_the_ledger_amber():
    assert board.turns_panel([turn(9.0)]).tone == "warn"
    assert board.turns_panel([turn(0.9)]).tone == "ok"


def test_an_empty_ledger_says_so_rather_than_drawing_a_flat_line():
    p = board.turns_panel([])
    assert p.tone == "off" and p.spark == ()


# ------------------------------------------------------------- sessions
def session(slug, title=""):
    return SimpleNamespace(slug=slug, title=title, first_user="", mtime=0.0)


def test_a_running_task_outranks_its_own_session_row():
    p = board.sessions_panel([session("jarvis", "old work")],
                             {"jarvis": "running"})
    assert p.rows == [("JARVIS", "RUNNING")]
    assert p.tone == "ok" and "1 task running" in p.line


def test_a_waiting_task_is_the_loudest_thing_on_the_panel():
    p = board.sessions_panel([], {"vss": "waiting"})
    assert p.tone == "warn" and "permission" in p.line


def test_recent_sessions_fill_in_under_the_live_ones():
    p = board.sessions_panel([session("vss", "detector work")], {})
    assert p.rows == [("VSS", "detector work")] and p.tone == "idle"


def test_no_sessions_at_all_is_a_dark_panel():
    assert board.sessions_panel([], {}).tone == "off"


# ------------------------------------------------------------ deadlines
def item(label, due):
    return SimpleNamespace(kind="reminder", label=label, effective_due=due)


def test_deadlines_sort_soonest_first_across_both_sources():
    now = 1_000_000.0
    rows = board.deadline_rows([item("late", now + 7200),
                                item("soon", now + 600)],
                               ["BIOSEN - Lab 3 report, tonight"], now)
    assert rows[0] == ("SOON", "in 10m")
    assert rows[1] == ("LATE", "in 2h")
    assert rows[2][0] == "BIOSEN"


def test_nothing_due_is_idle_not_broken():
    p = board.deadlines_panel([], [], 0.0)
    assert p.tone == "idle" and "Nothing due" in p.line


# ---------------------------------------------------------------- focus
def test_a_running_block_speaks_the_sessions_own_words():
    focus = SimpleNamespace(active=True, phase="block", blocks_done=1,
                            label="thesis",
                            time_left=lambda: "Nine minutes left, sir.")
    p = board.focus_panel(focus)
    assert dict(p.rows)["PHASE"] == "BLOCK" and p.tone == "ok"
    assert p.line == "Nine minutes left, sir."


def test_a_focus_session_whose_clock_raises_still_renders():
    def boom():
        raise RuntimeError("timekeeper down")
    focus = SimpleNamespace(active=True, phase="break", blocks_done=2,
                            label="", time_left=boom)
    p = board.focus_panel(focus)
    assert p.tone == "ok" and "break is running" in p.line


def test_no_focus_session_is_idle_and_no_focus_module_is_off():
    assert board.focus_panel(SimpleNamespace(active=False)).tone == "idle"
    assert board.focus_panel(None).tone == "off"


# ---------------------------------------------------------------- quiet
def test_an_unconfigured_presence_never_prints_a_confident_home():
    """presence.is_home() answers True when home is None, so the row is
    gated on `configured` — see jarvis/presence.py:159."""
    p = board.quiet_panel("", "unknown", 0, configured=False)
    assert "PRESENCE" not in dict(p.rows)


def test_a_configured_presence_shows_the_state_word():
    p = board.quiet_panel("", "away", 0, configured=True)
    assert dict(p.rows)["PRESENCE"] == "AWAY"


def test_a_quiet_reason_is_amber_and_counts_what_is_held():
    p = board.quiet_panel("quiet hours until 7:00 am", "home", 2, True)
    assert p.tone == "warn" and dict(p.rows)["HELD"] == "2"
    assert "2 lines waiting" in p.line


# ------------------------------------------------------------- assembly
def test_board_state_composes_every_panel_in_order():
    st = board.board_state(
        health=lambda: snap(),
        turns=lambda: [turn(1.2)],
        sessions=lambda: [session("jarvis", "work")],
        tasks=lambda: {},
        schedule=lambda: [],
        canvas=lambda: [],
        focus=lambda: SimpleNamespace(active=False),
        quiet=lambda: SimpleNamespace(reason=lambda: "", held=[]),
        presence=lambda: SimpleNamespace(state="home", configured=True),
        now=1_000_000.0)
    assert st.keys == board.PANEL_ORDER
    assert st.at == 1_000_000.0


def test_one_exploding_provider_cannot_sink_the_board():
    def boom():
        raise RuntimeError("nvidia-smi wedged")
    st = board.board_state(health=boom, turns=lambda: [turn(1.0)])
    assert st.get("vitals").tone == "off"
    assert st.get("turns").tone == "ok"          # the rest still rendered


def test_no_providers_at_all_still_yields_every_panel():
    st = board.board_state()
    assert len(st.panels) == len(board.PANEL_ORDER)
    assert board.board_text(st)                  # renders without raising


def test_a_quiet_policy_that_raises_leaves_the_panel_readable():
    st = board.board_state(
        quiet=lambda: SimpleNamespace(reason=_boom, held=[]))
    assert st.get("quiet").rows[0] == ("HOLDING", "NO")


def _boom():
    raise RuntimeError("quiet read failed")


# ------------------------------------------------------- the spoken read
def test_a_spoken_panel_name_resolves_to_the_panel_it_means():
    assert board.resolve_panel("the sessions") == "sessions"
    assert board.resolve_panel("claude sessions") == "sessions"
    assert board.resolve_panel("engines") == "vitals"
    assert board.resolve_panel("on the deadlines") == "deadlines"
    assert board.resolve_panel("the weather") == ""


def test_focus_on_a_panel_gives_a_sentence_not_a_highlight():
    st = board.board_state(turns=lambda: [turn(1.0), turn(2.0)])
    line = board.panel_line(st, "the turn ledger")
    assert line.endswith("sir.") and "median" in line


def test_a_panel_nobody_named_reads_back_empty():
    st = board.board_state()
    assert board.panel_line(st, "the fridge") == ""


def test_board_text_labels_a_dark_panel_as_unavailable():
    st = board.board_state()
    assert "VITALS (unavailable)" in board.board_text(st)


# ------------------------------------------------------------- the feed
def test_the_feed_publishes_what_the_provider_gave_it():
    sent = []
    st = board.board_state()
    feed = board.BoardFeed(lambda: st, publish=sent.append)
    assert feed.tick() is st
    assert len(sent) == 1 and sent[0].state is st


def test_a_wedged_provider_does_not_kill_the_poll_thread():
    def boom():
        raise RuntimeError("nvidia-smi hung")
    sent = []
    feed = board.BoardFeed(boom, publish=sent.append)
    assert feed.tick() is None and sent == []


def test_a_provider_that_answers_nothing_publishes_nothing():
    sent = []
    assert board.BoardFeed(lambda: None, publish=sent.append).tick() is None
    assert sent == []


def test_the_feed_starts_and_joins_cleanly():
    feed = board.BoardFeed(board.board_state, interval=0.05,
                           publish=lambda _e: None)
    feed.start()
    assert feed.running
    feed.stop()
    assert not feed.running
    feed.stop()                       # idempotent: quit may arrive twice


# ------------------------------------------------- layout (pure, no Tk)
def test_the_board_docks_down_the_right_flank_clear_of_the_console():
    geo = ui_board.dock_geometry(3840, 2160, width=520, margin=24,
                                 console_w=520)
    assert geo == "520x2112+3296+24"


def test_the_dock_never_walks_off_a_small_screen():
    geo = ui_board.dock_geometry(800, 600, width=520, margin=24,
                                 console_w=520)
    w, _, rest = geo.partition("x")
    assert int(w) <= 800 and "+" in rest


def test_panels_split_the_height_and_a_sparkline_panel_gets_more():
    boxes = ui_board.panel_boxes(1000, ["vitals", "turns", "sessions"],
                                 pad=10, tall={"turns"})
    assert [k for k, _, _ in boxes] == ["vitals", "turns", "sessions"]
    heights = {k: y1 - y0 for k, y0, y1 in boxes}
    assert heights["turns"] > heights["vitals"]
    assert boxes[0][1] == 10 and boxes[-1][2] <= 1000 - 10


def test_a_height_too_small_for_the_panels_still_returns_ordered_boxes():
    boxes = ui_board.panel_boxes(40, ["a", "b", "c"], pad=10)
    assert len(boxes) == 3
    assert all(y1 >= y0 for _, y0, y1 in boxes)
    assert [b[1] for b in boxes] == sorted(b[1] for b in boxes)


def test_spark_points_walk_left_to_right_with_the_peak_at_the_top():
    pts = ui_board.spark_points((0.0, 1.0), x0=0, y0=0, w=10, h=100)
    assert pts[0] == 0 and pts[1] == 100      # zero sits on the baseline
    assert pts[2] == 10 and pts[3] == 0       # the peak reaches the top


# ------------------------------------- the WM close seam (review a/b n=23)
def _window():
    """A MainWindow shell with no Tk behind it. Only the two attributes the
    Board hooks touch are populated; nothing here constructs a widget."""
    from jarvis.ui import main_window as mw

    win = mw.MainWindow.__new__(mw.MainWindow)
    win.board = None
    return win


def test_closing_the_board_with_the_window_manager_stops_its_feed():
    """n=23. BoardWindow was built without its on_close hook, so Alt+F4
    withdrew the window while the app's 5 s BoardFeed kept spawning
    nvidia-smi, walking tmux and reading turns.jsonl until quit — and
    "bring up the board" answered "Already up, sir." for a window nobody
    could see."""
    from jarvis.ui import main_window as mw

    stopped = []
    win = _window()
    win.services = mw.Services(board_closed=lambda: stopped.append("hide"))
    win._board_closed()
    assert stopped == ["hide"]


def test_the_wm_close_hook_falls_back_to_the_board_service_shape():
    """The app-side seam may land after this one; the commander-shaped
    services object carries the same teardown as services.board.hide."""
    stopped = []
    win = _window()
    win.services = SimpleNamespace(
        board=SimpleNamespace(hide=lambda: stopped.append("hide")))
    win._board_closed()
    assert stopped == ["hide"]


def test_an_unwired_wm_close_is_survivable_not_fatal():
    win = _window()
    win.services = SimpleNamespace()
    win._board_closed()                 # logs a warning, does not raise


def test_a_failing_teardown_never_escapes_the_wm_handler():
    def boom():
        raise RuntimeError("the feed is already gone")
    win = _window()
    win.services = SimpleNamespace(board_closed=boom)
    win._board_closed()


def test_the_board_is_built_with_the_close_hook_attached(monkeypatch):
    """The regression was purely at the construction site: the parameter
    existed, hide() invoked it, and no caller ever passed it."""
    from jarvis.ui import main_window as mw

    built = {}

    class FakeBoard:
        def __init__(self, master, console_w=0, on_close=None):
            built["on_close"] = on_close

    monkeypatch.setattr(mw, "BoardWindow", FakeBoard)
    monkeypatch.setattr(mw, "board_enabled", lambda _opt: True)
    win = _window()
    win.services = mw.Services()
    win.root = SimpleNamespace(winfo_width=lambda: 1200)
    board_win = win._ensure_board()
    assert board_win is not None
    assert built["on_close"] == win._board_closed


def test_hiding_the_board_withdraws_before_it_notifies():
    """Order matters: hide() clears _visible BEFORE calling on_close, so the
    app's BoardCommand(action="hide") re-enters hide() and returns instead
    of recursing."""
    seen = []
    shell = SimpleNamespace(
        _visible=True,
        top=SimpleNamespace(withdraw=lambda: seen.append("withdraw")),
        on_close=lambda: seen.append("notify"))
    ui_board.BoardWindow.hide(shell)
    assert seen == ["withdraw", "notify"]
    assert shell._visible is False
    ui_board.BoardWindow.hide(shell)     # already down: no second notify
    assert seen == ["withdraw", "notify"]


def test_a_raising_close_hook_does_not_leave_the_board_half_closed():
    def boom():
        raise RuntimeError("services are torn down")
    shell = SimpleNamespace(_visible=True,
                            top=SimpleNamespace(withdraw=lambda: None),
                            on_close=boom)
    ui_board.BoardWindow.hide(shell)     # Tk's WM handler must not see it
    assert shell._visible is False


# ----------------------------------------------------------- the cast slab
def _thrown(**kw):
    base = dict(status="landed", spoken="the thesis draft", target="the board",
                kind="screen", by="gesture", at=990.0)
    base.update(kw)
    return base


def test_the_cast_slab_appears_only_after_a_throw_and_ages_out():
    assert "cast" not in board.board_state(now=1000.0).keys
    st = board.board_state(cast=lambda: _thrown(), now=1000.0)
    assert st.keys == board.PANEL_ORDER + ("cast",)
    p = st.get("cast")
    assert ("STATUS", "LANDED") in p.rows and ("WHAT", "the thesis draft") in p.rows
    assert ("TO", "THE BOARD") in p.rows and p.tone == "ok"
    assert "the thesis draft" in p.line
    late = 990.0 + board.CAST_TTL_S + 1.0
    assert "cast" not in board.board_state(cast=lambda: _thrown(), now=late).keys


def test_a_held_cast_is_a_warning_that_names_the_deaf_target():
    p = board.cast_panel(_thrown(status="held", target="HPCOMPUTER"), 1000.0)
    assert p.tone == "warn" and ("STATUS", "HELD") in p.rows
    assert "HPCOMPUTER" in p.line and "holding" in p.line


def test_a_bad_cast_record_cannot_sink_the_board():
    assert board.cast_panel(None, 1000.0) is None
    assert board.cast_panel({"at": "soon"}, 1000.0) is None
    assert board.cast_panel({}, 1000.0) is None
    st = board.board_state(cast=lambda: (_ for _ in ()).throw(RuntimeError("x")),
                           now=1000.0)
    assert st.keys == board.PANEL_ORDER


def test_the_throw_can_be_asked_about_by_name():
    assert board.resolve_panel("the last throw") == "cast"
    assert board.resolve_panel("what i threw") == "cast"
