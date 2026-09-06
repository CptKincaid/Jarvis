"""Casting a SCREEN VIEW between his two machines (jarvis/castview.py) and
the relay the Windows startup script polls.

NOTHING IN THIS FILE STARTS A RUSTDESK SESSION, in either direction. That
would throw a window onto a screen he is using. Every launcher is an
injected recorder, every poll is a function call, and no socket is opened.

WHAT IS PINNED, and why each needs a test rather than a promise:

* THE CLOSED VERB SET IS ENFORCED WHERE IT LEAVES. The Windows side holds
  its own fixed command lines and receives a VERB from {show-spark, stop,
  none}. A bug on the Jarvis side must not be able to emit a command string
  even if it tries, so the enumeration is on the SERVER too and a poisoned
  verb comes back as "none".
* A CAST CANNOT FIRE WHILE ONE IS UP. The harm here is a window appearing
  on a screen he is using, and it is not undone by a tone; a double fling
  must not open two viewers.
* THE STOP IS AS EASY AS THE START, because of that same asymmetry.
* NO WINDOWS HELPER, NO CAST -- said out loud, with the reason. It HOLDS;
  it never silently does nothing and it never pretends it landed.
* THE REVERSE DIRECTION NEEDS NO WINDOWS CHANGE AT ALL, and holds honestly
  when this box has no viewer to open.
* THE MODULE HOLDS NO TRANSPORT OF ITS OWN. Pinned at source level, the
  same way tests/test_cast.py pins the camera out of jarvis/cast.py.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from jarvis import castview as cv
from jarvis import screens as sc
from jarvis.cast import CastSubject
from jarvis.visionrig import assert_numbers_only


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


def subject(kind: str = "screen", spoken: str = "the Spark's screen"):
    return CastSubject(kind, spoken, at=1000.0)


def deck(clock=None, *, launch=True, helper_at=None, answers=True,
         viewer_up=True, served=True):
    """Both sinks and the shared live-cast state, with recorders behind
    every outward edge.

    NEITHER SINK MAY CLAIM A LANDING IT CANNOT SEE (09-05), so this deck
    now supplies both confirmations. ``answers`` plays the Windows script's
    half of the round trip -- wake on the parked long poll, take the verb,
    act, poll again quoting the sequence it acted on, which is the receipt.
    ``viewer_up`` is the Spark viewer's aliveness a moment after launch,
    and ``served`` (ROUND 4) is whether HPCOMPUTER is actually pulling the
    Spark's screen once its helper has acknowledged the verb. Set any of
    them False for a cast that goes out and is never confirmed.
    """
    clock = clock or Clock()
    rec = {"launched": [], "stopped": [], "verbs": [], "naps": [],
           "later": []}
    state = cv.ViewState(now=clock)
    relay = cv.CastRelay(now=clock)
    if helper_at is not None:
        relay.note(mon=1, layout="0,1920,1920,1920", at=helper_at)
    helper_seq = [-1]

    def helper(_timeout_s):
        if not answers:
            return False
        got = relay.poll(helper_seq[0], timeout_s=0.0)
        if got["seq"] != helper_seq[0]:
            helper_seq[0] = got["seq"]
            rec["verbs"].append(got["verb"])
            relay.poll(helper_seq[0], timeout_s=0.0)   # ...and it says so
        return True

    # ROUND 3: a live process is not a connection, so the sink now needs a
    # ``connected`` probe too, and a ``later`` for the second look. Both
    # are injected recorders here; nothing opens a viewer or a socket.
    # ROUND 5: THE RECORDER MODELS THE REAL SEAM'S OWN DEATH. ``stop`` used
    # to record the call and leave ``alive`` answering True for ever, which
    # no real viewer does -- ``JarvisApp._rustdesk_close`` drops the Popen
    # so ``_rustdesk_alive()`` is False the moment after. The stop verb now
    # CHECKS with that probe, so a recorder that never dies is a recorder
    # that models a viewer which would not close.
    up = {"viewer": bool(viewer_up)}

    def stop_viewer():
        rec["stopped"].append(1)
        up["viewer"] = False

    spark = cv.SparkViewSink(
        launch=(lambda host: (rec["launched"].append(host),
                              up.__setitem__("viewer", bool(viewer_up)))
                ) if launch else None,
        stop=stop_viewer,
        alive=lambda: up["viewer"], connected=lambda: up["viewer"],
        settle=rec["naps"].append,
        later=lambda d, fn: rec["later"].append((d, fn)),
        state=state, now=clock)
    # ROUND 4: HpViewSink is held to the same rule. A receipt says the
    # Windows script ACTED; it never said a desktop arrived, and a viewer
    # on an accept-or-password prompt satisfies it exactly. ``served`` is
    # the connection probe for that direction and it is an injected
    # recorder here; nothing opens a socket.
    hp = cv.HpViewSink(relay=relay, state=state, now=clock, wait=helper,
                       connected=lambda: served,
                       settle=rec["naps"].append,
                       later=lambda d, fn: rec["later"].append((d, fn)))
    return spark, hp, relay, state, rec, clock


# ------------------------------------------------------- the closed set
class TestTheClosedVerbSet:
    """His ruling, and it is not negotiable: the Windows side receives a
    VERB FROM A CLOSED SET, never a command string. That line is what keeps
    this from being remote execution into his live session."""

    def test_the_set_is_three_words_and_none_of_them_is_a_command(self):
        assert cv.VERBS == ("none", "show-spark", "stop")
        for verb in cv.VERBS:
            assert " " not in verb
            assert "-" in verb or verb in ("none", "stop")
            assert "/" not in verb and "\\" not in verb

    def test_a_relay_refuses_anything_outside_the_set(self):
        relay = cv.CastRelay(now=Clock())
        for bad in ("rustdesk --connect 1.2.3.4", "show-spark; rm -rf /",
                    "SHOW-SPARK", "", None, 7, "shutdown"):
            with pytest.raises(ValueError):
                relay.set_verb(bad)
        assert relay.verb == "none"

    def test_a_poisoned_verb_comes_back_as_none_at_the_route(self):
        """The enumeration is on the SERVER side too. If something inside
        Jarvis ever manages to park a command string on the relay, what
        goes out on the wire is still a verb from the set."""
        relay = cv.CastRelay(now=Clock())
        relay._verb = "rustdesk --connect 192.168.50.109"    # a bug, simulated
        out = relay.poll(seq=-1, timeout_s=0.0)
        assert out["verb"] == "none"
        assert out["verb"] in cv.VERBS

    def test_the_reply_is_numbers_and_a_verb_and_nothing_else(self):
        relay = cv.CastRelay(now=Clock())
        out = relay.poll(seq=-1, timeout_s=0.0)
        assert sorted(out) == ["seq", "verb"]
        assert_numbers_only(out)

    def test_the_verb_is_idempotent_across_repeat_polls(self):
        """Without a sequence number a held "show-spark" would relaunch the
        viewer on every poll, seven times a second when he is unlucky."""
        clock = Clock()
        relay = cv.CastRelay(now=clock)
        relay.set_verb("show-spark")
        first = relay.poll(seq=-1, timeout_s=0.0)
        assert first["verb"] == "show-spark"
        again = relay.poll(seq=first["seq"], timeout_s=0.0)
        assert again["seq"] == first["seq"]
        assert again["verb"] == "none"

    def test_a_stop_is_a_new_sequence(self):
        relay = cv.CastRelay(now=Clock())
        relay.set_verb("show-spark")
        one = relay.poll(seq=-1, timeout_s=0.0)
        relay.set_verb("stop")
        two = relay.poll(seq=one["seq"], timeout_s=0.0)
        assert two["verb"] == "stop"
        assert two["seq"] != one["seq"]


# ---------------------------------------------------- the Windows helper
class TestTheHelper:
    """What the poll tells Jarvis about what he is doing is one small
    integer and a layout string. No window title, no process name, no path
    and no cursor coordinate ever leaves that machine."""

    def test_the_helper_reports_a_monitor_index_and_a_layout_only(self):
        clock = Clock()
        relay = cv.CastRelay(now=clock)
        relay.note(mon=2, layout="0,1920,1920,1920,3840,2560")
        assert relay.mon == 2
        assert relay.layout == "0,1920,1920,1920,3840,2560"
        assert_numbers_only(relay.numbers_only())

    def test_a_silent_helper_is_not_alive(self):
        clock = Clock(1000.0)
        relay = cv.CastRelay(now=clock, alive_s=60.0)
        assert not relay.alive()
        relay.note(mon=0, layout="0,1920")
        assert relay.alive()
        clock.t += 61.0
        assert not relay.alive()

    def test_rubbish_from_the_helper_is_dropped_not_stored(self):
        relay = cv.CastRelay(now=Clock())
        relay.note(mon="; rm -rf /", layout={"x": 1})
        assert relay.mon == -1
        assert relay.layout == ""

    def test_a_layout_is_bounded_in_length(self):
        relay = cv.CastRelay(now=Clock())
        relay.note(mon=0, layout="0,1920," * 500)
        assert len(relay.layout) <= cv.MAX_LAYOUT_CHARS


# ------------------------------------------- Spark -> HPCOMPUTER, the verb
class TestSparkToHpcomputer:
    def test_with_the_helper_polling_it_lands_and_sets_the_verb(self):
        spark, hp, relay, state, rec, clock = deck(helper_at=1000.0)
        res = hp.deliver(subject())
        assert res.landed
        assert relay.verb == "show-spark"
        assert "HPCOMPUTER" in res.spoken

    def test_with_no_helper_running_it_HOLDS_and_says_why(self):
        """It degrades honestly: it never silently does nothing, and it
        never says it landed. The startup script is his to install."""
        spark, hp, relay, state, rec, clock = deck(helper_at=None)
        res = hp.deliver(subject())
        assert not res.landed
        assert res.held
        assert res.spoken
        assert relay.verb == "none"                 # nothing was queued

    def test_a_helper_that_has_gone_quiet_holds_too(self):
        spark, hp, relay, state, rec, clock = deck(helper_at=1000.0)
        clock.t += 600.0
        res = hp.deliver(subject())
        assert res.held
        assert relay.verb == "none"

    def test_it_never_falls_back_to_the_board(self):
        """A row on the board is not a smaller version of a desktop on a
        monitor; it is a different thing. Holding says so."""
        spark, hp, relay, state, rec, clock = deck(helper_at=None)
        assert hp.deliver(subject()).fallback == ""

    def test_it_carries_no_bytes(self):
        """A screen VIEW is a live connection between two machines, not a
        file. Nothing is captured, materialised or sent."""
        assert cv.HpViewSink.wants_bytes is False
        assert cv.SparkViewSink.wants_bytes is False


# ------------------------------------------ HPCOMPUTER -> Spark, the viewer
class TestHpcomputerToSpark:
    """The easy direction, and it needs NO Windows change at all: the Spark
    runs its own viewer outbound, which the Windows firewall does not
    touch."""

    def test_it_launches_the_spark_s_own_viewer_through_the_seam(self):
        spark, hp_sink, relay, state, rec, clock = deck()
        res = spark.deliver(subject("screen", "HPCOMPUTER's screen"))
        assert res.landed
        assert rec["launched"] == [cv.HPCOMPUTER_HOST]

    def test_it_needs_no_windows_helper(self):
        spark, hp_sink, relay, state, rec, clock = deck(helper_at=None)
        assert spark.available()[0]
        assert spark.deliver(subject()).landed

    def test_with_no_viewer_wired_it_HOLDS_and_says_so(self):
        """``launch`` defaults to None -- the same discipline
        HpcomputerSink(transport=None) already holds, and for the same
        stated reason: a transport that cannot be tested against the real
        host is a guess dressed as a feature."""
        spark, hp_sink, relay, state, rec, clock = deck(launch=False)
        ok, why = spark.available()
        assert not ok and why
        res = spark.deliver(subject())
        assert res.held and res.spoken and not res.landed

    def test_a_launcher_that_raises_holds_rather_than_claiming_it_landed(self):
        clock = Clock()
        state = cv.ViewState(now=clock)

        def boom(host):
            raise OSError("no rustdesk on PATH")

        sink = cv.SparkViewSink(launch=boom, alive=lambda: True,
                                connected=lambda: True,
                                settle=lambda s: None,
                                later=lambda d, fn: None,
                                state=state, now=clock)
        res = sink.deliver(subject())
        assert res.held and not res.landed
        assert state.live == ""            # and the deck is not left busy


# ----------------------------------------------------- one cast at a time
class TestOneAtATime:
    """The harm is a window appearing, not bytes leaving, so a stop must be
    as easy as a start and a double fling must not open two viewers."""

    def test_a_second_cast_while_one_is_up_is_refused(self):
        spark, hp_sink, relay, state, rec, clock = deck(helper_at=1000.0)
        assert spark.deliver(subject()).landed
        clock.t += 30.0
        res = hp_sink.deliver(subject())
        assert res.held
        assert cv.BUSY_LINE in res.spoken
        assert relay.verb == "none"

    def test_the_same_direction_twice_is_refused_too(self):
        spark, hp_sink, relay, state, rec, clock = deck()
        assert spark.deliver(subject()).landed
        clock.t += 30.0
        assert spark.deliver(subject()).held
        assert rec["launched"] == [cv.HPCOMPUTER_HOST]

    def test_a_double_fling_inside_the_suppression_cannot_open_two(self):
        """8 frames of gesture cooldown is about 1.1 s. A cast is
        suppressed for longer than that on purpose."""
        spark, hp_sink, relay, state, rec, clock = deck()
        assert spark.deliver(subject()).landed
        spark.stop_cast()
        clock.t += 0.4
        assert spark.deliver(subject()).held
        clock.t += cv.CAST_SUPPRESS_S
        assert spark.deliver(subject()).landed

    def test_stopping_releases_the_deck_and_stops_the_viewer(self):
        spark, hp_sink, relay, state, rec, clock = deck(helper_at=1000.0)
        spark.deliver(subject())
        assert state.live == spark.name
        assert spark.stop_cast()
        assert rec["stopped"] == [1]
        assert state.live == ""

    def test_stopping_the_windows_direction_sends_the_stop_verb(self):
        spark, hp_sink, relay, state, rec, clock = deck(helper_at=1000.0)
        hp_sink.deliver(subject())
        assert relay.verb == "show-spark"
        assert hp_sink.stop_cast()
        assert relay.verb == "stop"
        assert state.live == ""

    def test_stopping_nothing_says_so_rather_than_pretending(self):
        spark, hp_sink, relay, state, rec, clock = deck()
        assert not spark.stop_cast()
        assert rec["stopped"] == []

    def test_the_state_is_numbers_and_short_names_only(self):
        spark, hp_sink, relay, state, rec, clock = deck()
        spark.deliver(subject())
        assert_numbers_only(state.numbers_only())


# ------------------------------------------------------- the source rule
class TestTheSourceRule:
    """The sink for a destination machine, and there are only two."""

    def test_each_machine_has_exactly_one_view_sink(self):
        spark, hp_sink, relay, state, rec, clock = deck()
        reg = cv.registry(spark, hp_sink)
        assert sorted(reg) == sorted(sc.MACHINES)
        assert reg[sc.SPARK] is spark
        assert reg[sc.HPCOMPUTER] is hp_sink

    def test_the_subject_of_a_screen_cast_names_the_SOURCE_screen(self):
        """He grabs at a screen and throws it: what travels is the screen,
        not whatever document happened to be under his hand."""
        assert "Spark" in cv.view_subject(sc.SPARK, at=1.0).spoken
        assert "HPCOMPUTER" in cv.view_subject(sc.HPCOMPUTER, at=1.0).spoken
        assert cv.view_subject(sc.SPARK, at=1.0).kind == "screen"
        assert cv.view_subject("nonsense", at=1.0) is None


# ------------------------------------------------------------- the source
def test_the_module_holds_no_transport_of_its_own():
    """Every outward edge here is an INJECTED callable. The real launcher
    is built in jarvis/app.py; the suite passes a recorder. Pinned in the
    source rather than in a comment, because a later edit that quietly adds
    a subprocess call would make the whole "nothing in this session starts
    a RustDesk session" claim untrue without any test noticing."""
    tree = ast.parse(Path(cv.__file__).read_text())
    banned = {"subprocess", "socket", "urllib", "requests", "http",
              "shutil", "os", "cv2", "tkinter", "webbrowser", "asyncio"}
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            seen.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            seen.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.Attribute):
            seen.add(node.attr)
    assert not (banned & seen), (
        "jarvis/castview.py must not reach for %s" % sorted(banned & seen))
    for word in ("Popen", "Toplevel", "xdg-open", "VideoCapture",
                 "ImageGrab", "rustdesk"):
        assert word not in seen, f"jarvis/castview.py must not reach for {word}"


# ------------------------------------------------- the Windows startup file
class TestTheStartupFile:
    """scripts/windows/cast-poll.ps1 -- HIS file, on HIS machine. Nothing in
    this repo installs it and nothing here runs it; these are source-level
    pins on the properties his ruling depends on.

    Read the four of them together and they are the whole argument that
    this is not remote execution into his live session: the response is
    compared against string literals, the executable and the target are
    literals in the file, nothing from the response is ever interpolated
    into a path or an argument, and a stop only ever kills the process the
    script itself started.
    """

    @staticmethod
    def source() -> str:
        root = Path(cv.__file__).resolve().parents[1]
        return (root / "scripts" / "windows" / "cast-poll.ps1").read_text()

    def test_it_compares_the_verb_against_literals_from_the_closed_set(self):
        src = self.source()
        assert "-eq 'show-spark'" in src
        assert "-eq 'stop'" in src
        # A switch or an acting else branch would turn an unknown verb into
        # behaviour; there must be neither.
        assert "switch" not in src.lower()
        for verb in cv.VERBS:
            if verb != cv.VERB_NONE:
                assert "'%s'" % verb in src

    def test_the_executable_and_the_target_are_literals(self):
        src = self.source()
        assert "$RustDesk = " in src and "ProgramFiles" in src
        assert "$Target   = '192.168.50.109'" in src

    def test_nothing_from_the_response_is_ever_run(self):
        """The response fields are ``$r.seq`` and ``$r.verb`` and they only
        ever reach an integer cast and a string comparison."""
        src = self.source()
        for danger in ("Invoke-Expression", "iex ", "& $verb", "& $r",
                       "-ArgumentList $", "cmd /c", "Start-Process $"):
            assert danger not in src, danger
        assert "-ArgumentList '--connect', $Target" in src

    def test_a_stop_only_kills_what_this_script_started(self):
        """The first thing the design got wrong: ``Get-Process rustdesk |
        Stop-Process`` would have closed a session he opened himself."""
        src = self.source()
        # ROUND 3 moved the kill into Stop-Cast, which takes the pid this
        # script recorded and nothing else. The property is the same one:
        # only a pid this script started is ever stopped.
        assert "Stop-Process -Id $ProcId" in src
        assert "Stop-Cast -ProcId $castPid" in src
        # ROUND 4 needed Get-Process to tell "already exited" (which is a
        # SUCCESS and must not be reported) from "would not die" (which
        # must). Banning the cmdlet by name was a proxy for the real law;
        # the law itself is that EVERY process this script touches is
        # named by the pid it recorded, and never by a name -- because
        # ``Get-Process rustdesk | Stop-Process`` would close a session he
        # opened himself.
        import re as _re
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith("#"))
        for call in _re.findall(r"(?:Get|Stop)-Process[^\n]*", code):
            assert "-Id $ProcId" in call, call
        for by_name in ("Get-Process rustdesk", "-Name ", "-ProcessName",
                        "-InputObject"):
            assert by_name not in code, by_name

    def test_it_backs_off_rather_than_hammering_a_spark_that_is_down(self):
        src = self.source()
        assert "catch { Start-Sleep -Seconds 5 }" in src

    def test_the_key_travels_as_a_header_not_in_the_url(self):
        """A query string ends up in logs; a bearer header does not."""
        src = self.source()
        assert 'Authorization = "Bearer $Token"' in src
        assert "?t=" not in src

    def test_it_sends_a_monitor_INDEX_and_a_layout_and_nothing_else(self):
        src = self.source()
        assert "mon = (Get-MonIndex)" in src
        assert "layout = (Get-Layout)" in src
        for leak in ("MainWindowTitle", "GetForegroundWindow", "Get-Clipboard",
                     "Cursor]::Position.X", "$env:USERNAME"):
            assert leak not in src, leak


# ==================================== never say it landed until it has
class TestReceiptNotHope:
    """cast.py's oldest promise, made true in both directions.

    ``CastResult.landed`` and ``.held`` are separate booleans and neither
    is ever inferred from the other. Both view sinks broke the spirit of
    that: HpViewSink said "The Spark's screen is on HPCOMPUTER, sir" the
    instant the verb was PARKED for a helper that may last have polled
    59 seconds ago, and SparkViewSink said the mirror of it whenever
    ``launch()`` merely failed to raise -- and the real launcher is a
    Popen, so a viewer that starts and dies a moment later reported
    success. Neither had any confirmation behind it at all.
    """

    # -- Spark -> HPCOMPUTER: the helper must acknowledge RECEIPT --------
    def test_a_parked_verb_is_not_a_landing(self):
        """The helper is inside its alive window but never comes back for
        the verb. Parked is not received, so this HOLDS."""
        spark, hp, relay, state, rec, clock = deck(helper_at=1000.0,
                                                   answers=False)
        res = hp.deliver(subject())
        assert not res.landed
        assert res.held and res.spoken
        assert state.live == ""            # and the deck is given back

    def test_it_lands_when_the_helper_polls_back_for_the_verb(self):
        """The receipt IS the helper's next poll: it long-polls carrying
        the sequence it last acted on, so a poll quoting our sequence is
        it saying 'I have this one'."""
        spark, hp, relay, state, rec, clock = deck(helper_at=1000.0)
        seen = []

        def helper(timeout_s):
            # the Windows script wakes, takes the verb, acts, polls again
            got = relay.poll(relay.seq - 1, timeout_s=0.0)
            seen.append(got["verb"])
            relay.poll(got["seq"], timeout_s=0.0)
            return True

        res = hp.deliver(subject(), wait=helper)
        assert seen == [cv.VERB_SHOW]
        assert res.landed and not res.held
        assert "HPCOMPUTER" in res.spoken
        assert state.live == "hp-view"

    def test_an_unreceipted_verb_is_taken_back_off_the_relay(self):
        """Nothing is left queued for a machine that did not answer: a
        helper that wakes up a minute later must not open a window for an
        intention Jarvis has already said it did not carry out."""
        spark, hp, relay, state, rec, clock = deck(helper_at=1000.0,
                                                   answers=False)
        hp.deliver(subject())
        assert relay.verb == cv.VERB_STOP

    def test_the_receipt_must_be_for_THIS_verb_not_an_older_one(self):
        """An ack for an earlier sequence is not an ack for this one."""
        spark, hp, relay, state, rec, clock = deck(helper_at=1000.0)
        relay.poll(-1, timeout_s=0.0)              # the helper is up to date
        stale = relay.seq
        res = hp.deliver(subject(), wait=lambda _s: relay.poll(stale,
                                                               timeout_s=0.0))
        assert res.held and not res.landed

    def test_the_relay_records_what_the_helper_has_acknowledged(self):
        relay = cv.CastRelay(now=Clock())
        assert relay.acked == -1
        seq = relay.set_verb(cv.VERB_SHOW)
        assert not relay.acked_through(seq)
        relay.poll(seq, timeout_s=0.0)
        assert relay.acked_through(seq) and relay.acked == seq
        assert_numbers_only(relay.numbers_only())

    # -- HPCOMPUTER -> Spark: the viewer must still be alive -------------
    def test_a_viewer_that_dies_a_moment_later_is_not_a_landing(self):
        clock = Clock()
        state = cv.ViewState(now=clock)
        naps = []
        sink = cv.SparkViewSink(launch=lambda host: None, stop=lambda: None,
                                alive=lambda: False, connected=lambda: False,
                                settle=naps.append,
                                later=lambda d, fn: None,
                                state=state, now=clock)
        res = sink.deliver(subject())
        assert not res.landed
        assert res.held and res.spoken
        assert state.live == ""
        assert naps and naps[0] > 0.0          # it waited before it looked

    def test_a_viewer_still_up_a_moment_later_is_a_landing(self):
        clock = Clock()
        state = cv.ViewState(now=clock)
        sink = cv.SparkViewSink(launch=lambda host: None, alive=lambda: True,
                                connected=lambda: True,
                                settle=lambda s: None,
                                later=lambda d, fn: None,
                                state=state, now=clock)
        res = sink.deliver(subject())
        assert res.landed and not res.held
        assert state.live == "spark-view"

    def test_a_viewer_with_no_way_to_confirm_it_is_never_claimed(self):
        """No aliveness probe wired is the same class as no launcher
        wired: a viewer that cannot be confirmed is a guess dressed as a
        feature, so the sink says it is unavailable rather than claiming
        a landing it cannot see."""
        clock = Clock()
        sink = cv.SparkViewSink(launch=lambda host: None, state=cv.ViewState(
            now=clock), now=clock)
        ok, why = sink.available()
        assert not ok and why
        res = sink.deliver(subject())
        assert res.held and not res.landed

    def test_a_dead_viewer_is_cleaned_up_not_left_behind(self):
        clock = Clock()
        state = cv.ViewState(now=clock)
        stopped = []
        sink = cv.SparkViewSink(launch=lambda host: None,
                                stop=lambda: stopped.append(1),
                                alive=lambda: False, connected=lambda: False,
                                settle=lambda s: None,
                                later=lambda d, fn: None,
                                state=state, now=clock)
        sink.deliver(subject())
        assert stopped == [1]

    def test_neither_sink_ever_returns_landed_and_held_together(self):
        """The constructor refuses it; this pins that every path here goes
        through the constructor rather than around it."""
        clock = Clock()
        state = cv.ViewState(now=clock)
        relay = cv.CastRelay(now=clock)
        relay.note(mon=1, layout="0,1920", at=clock.t)
        outs = [cv.SparkViewSink(launch=lambda h: None, alive=lambda: False,
                                 connected=lambda: False,
                                 settle=lambda s: None,
                                 later=lambda d, fn: None, state=state,
                                 now=clock).deliver(subject()),
                cv.HpViewSink(relay=relay, state=state,
                              now=clock).deliver(subject())]
        for res in outs:
            assert res.landed is not res.held
