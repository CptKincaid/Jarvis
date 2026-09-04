"""What a throw carries, and where it can honestly land (jarvis/cast.py).

Five promises are pinned here, and each one is a promise the design cannot
keep by intention alone:

* **A held cast never reports success.** ``landed`` and ``held`` are separate
  booleans on purpose and the suite asserts, for every sink and every path
  through ``cast()``, that they are never both true. The worst outcome this
  feature has is Jarvis saying a thing arrived somewhere it did not.
* **The board sink cannot open a window.** The 2026-08-26 desktop freeze came
  from window churn on :1, and the ruling that a cast fires immediately with
  no read-back depends entirely on it being cheap when it is wrong. A source
  test forbids the vocabulary (Tk, toplevel, launch_app, windowactivate) so a
  later edit that makes it expensive fails here rather than on his desktop.
* **An irreversible sink is unreachable by gesture.** With no ``propose``
  callable wired, ``cast()`` REFUSES; it never falls back to doing the thing.
  A fling proposes, a sentence confirms.
* **No URL is ever built from an mDNS name.** ``avahi-resolve -n
  spark-509f.local`` answers 172.17.0.1 -- the docker0 bridge -- so a handoff
  URL built from the .local name is unreachable from the machine it is meant
  for. The test asserts ".local" never appears in a handoff URL.
* **Nothing here touches the network.** Every probe is injected. The default
  TCP probe is never called: a test asserts constructing the sink does not
  probe, and every other test hands in a fake. The suite must pass with
  HPCOMPUTER up, down, or absent.

No camera, no display, no subprocess, no socket.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis import cast as cast_mod
from jarvis.cast import (
    BoardSink,
    CastOutbox,
    CastResult,
    CastSubject,
    HandoffSink,
    HpcomputerSink,
    cast,
    direction_word,
    is_taught,
    materialise,
    parse_side_teaching,
    pick_sink,
    resolve_sink,
    resolve_subject,
    subject_line,
    teach_sink,
    window_spoken_name,
)


# ------------------------------------------------ promise five, enforced
@pytest.fixture(autouse=True)
def _no_network_no_xdotool(monkeypatch):
    """No socket and no xdotool anywhere in this suite. A reach is RECORDED
    and fails the test at teardown, not merely raised: both the subject
    ladder and the sink swallow a raising boundary by design, so a bare
    raise would be silently turned into "empty" or "held"."""
    reached = []

    def boom(*a, **kw):
        reached.append((a, kw))
        raise AssertionError("the suite must not touch the network or spawn xdotool")

    monkeypatch.setattr(cast_mod.socket, "socket", boom)
    monkeypatch.setattr(cast_mod.socket, "getaddrinfo", boom)
    monkeypatch.setattr(cast_mod, "focused_window_title", boom)
    yield
    assert reached == [], "a test reached the network or the default screen rung"


# --------------------------------------------------------------- fakes
class Clock:
    """A hand-wound monotonic clock; nothing here waits on the real one."""

    def __init__(self, t: float = 1000.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += float(dt)
        return self.t


class Probe:
    """A stand-in for the live TCP reachability probe. Counts its calls so
    the cache can be proved, and never opens a socket."""

    def __init__(self, result=(False, "nothing on it is listening", "")):
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.result


def _doc(name="lab_report.pdf", when=None, now=1000.0):
    """A commander stand-in carrying one recently-explained document."""
    return SimpleNamespace(
        _last_document=(Path(f"/home/h/{name}"),
                        now if when is None else when))


def _spotify(playing=True, line="Kashmir by Led Zeppelin, sir — on HPCOMPUTER."):
    calls = {"now": 0}

    def now_playing():
        calls["now"] += 1
        return SimpleNamespace(speak=line, text=line)

    return SimpleNamespace(music_playing=lambda: playing,
                           now_playing=now_playing, calls=calls)


def _sub(kind="document", spoken="the lab report", **kw):
    return CastSubject(kind=kind, spoken=spoken, at=1000.0, **kw)


# ------------------------------------------------- the spoken window name
class TestWindowSpokenName:
    """A raw X11 title is unspeakable; the chip and the voice need the noun."""

    def test_drops_the_application_after_the_dash(self):
        assert window_spoken_name("thesis.tex - TeXstudio") == "thesis"

    def test_handles_the_em_dash_separator(self):
        assert window_spoken_name("Inbox — Thunderbird") == "Inbox"

    def test_drops_a_trailing_path_parenthetical_and_extension(self):
        raw = "Good evening Ali and Heather.txt (~/) - Text Editor"
        assert window_spoken_name(raw) == "Good evening Ali and Heather"

    def test_caps_the_word_count(self):
        raw = "one two three four five six seven eight"
        assert window_spoken_name(raw, max_words=6) == \
            "one two three four five six"

    def test_empty_and_separator_only_titles_yield_nothing(self):
        assert window_spoken_name("") == ""
        assert window_spoken_name(None) == ""
        assert window_spoken_name(" - ") == ""

    def test_never_returns_a_trailing_separator(self):
        assert not window_spoken_name("Firefox").endswith("-")


# ------------------------------------------------------- the subject ladder
class TestResolveSubject:
    """HELD -> DOCUMENT -> TRACK -> SCREEN -> EMPTY, first hit wins.

    Resolution runs on the REACH, not the grab: at 7.5 fps the reach buys
    300-400 ms, and spending it here is what makes the fist closing feel
    instant. So the ladder must be cheap and must never raise.
    """

    def test_held_wins_and_consults_nothing_else(self):
        held = _sub(kind="track", spoken="Kashmir by Led Zeppelin")
        called = []

        def boom():
            called.append(1)
            raise AssertionError("a second grab must not re-resolve")

        got = resolve_subject(held=held, now=1000.0,
                              providers={"document": boom, "track": boom,
                                         "screen": boom})
        assert got is held
        assert called == []

    def test_an_unholdable_held_subject_does_not_win(self):
        empty = CastSubject(kind="empty", spoken="", at=1.0)
        got = resolve_subject(held=empty, now=1000.0,
                              providers={"screen": lambda: _sub("screen", "Firefox")})
        assert got.kind == "screen"

    def test_a_fresh_document_beats_the_track_and_the_screen(self):
        got = resolve_subject(_doc(now=1000.0), _spotify(), now=1000.0,
                              providers={"screen": lambda: _sub("screen", "X")})
        assert got.kind == "document"
        assert got.path == Path("/home/h/lab_report.pdf")

    def test_the_document_name_is_spoken_not_filed(self):
        got = resolve_subject(_doc("intro-to_biosensors.pdf", now=1000.0),
                              now=1000.0)
        assert got.spoken == "intro to biosensors"

    def test_a_stale_document_falls_through_to_the_track(self):
        stale = _doc(when=1000.0 - 901.0, now=1000.0)
        got = resolve_subject(stale, _spotify(), now=1000.0)
        assert got.kind == "track"

    def test_the_document_window_is_the_house_window(self):
        """Reuses commander.LAST_DOCUMENT_S rather than inventing a second
        staleness rule; 899 s in, 901 s out."""
        assert cast_mod.DOCUMENT_MAX_AGE_S == 900.0
        inside = resolve_subject(_doc(when=1000.0 - 899.0, now=1000.0),
                                 now=1000.0)
        assert inside.kind == "document"

    def test_the_track_only_counts_while_it_is_playing(self):
        paused = _spotify(playing=False)
        got = resolve_subject(None, paused, now=1000.0,
                              providers={"screen": lambda: _sub("screen", "X")})
        assert got.kind == "screen"
        assert paused.calls["now"] == 0, "a paused deck costs no API call"

    def test_the_track_name_drops_the_address(self):
        got = resolve_subject(None, _spotify(), now=1000.0)
        assert got.kind == "track"
        assert got.spoken == "Kashmir by Led Zeppelin"

    def test_the_screen_is_the_floor(self):
        got = resolve_subject(None, None, now=1000.0,
                              providers={"screen": lambda: _sub("screen", "thesis")})
        assert got.kind == "screen"
        assert got.holdable

    def test_empty_only_when_nothing_at_all_resolves(self):
        got = resolve_subject(None, None, now=1000.0,
                              providers={"screen": lambda: None})
        assert got.kind == "empty"
        assert not got.holdable

    def test_a_provider_that_raises_does_not_break_the_ladder(self):
        def boom():
            raise RuntimeError("xdotool went away")

        got = resolve_subject(None, None, now=1000.0,
                              providers={"document": boom, "track": boom,
                                         "screen": lambda: _sub("screen", "X")})
        assert got.kind == "screen"

    def test_a_commander_of_the_wrong_shape_is_survivable(self):
        got = resolve_subject(object(), object(), now=1000.0,
                              providers={"screen": lambda: None})
        assert got.kind == "empty"


class TestSubjectLine:
    def test_names_what_it_picked_up(self):
        assert subject_line(_sub(spoken="the lab report")) == \
            "Holding the lab report, sir."

    def test_empty_has_its_own_line(self):
        line = subject_line(CastSubject(kind="empty", spoken="", at=0.0))
        assert line == cast_mod.NOTHING_LINE
        assert "sir" in line


# ----------------------------------------------------- the lazy screen PNG
class TestMaterialise:
    """The screen subject carries the NAME on the grab path and the BYTES
    only when a sink actually needs them. A 3840x2160 ImageGrab on every
    reach would put a visible hole in the metaphor."""

    def test_a_screen_subject_starts_without_bytes(self):
        got = resolve_subject(None, None, now=1000.0,
                              providers={"screen": lambda: _sub("screen", "X")})
        assert got.path is None

    def test_materialise_fills_the_path(self, tmp_path):
        png = tmp_path / "latest.png"
        png.write_bytes(b"\x89PNG")
        sub = _sub("screen", "thesis")
        got = materialise(sub, capture=lambda: {"screenshot": str(png)})
        assert got.path == png
        assert got.mime == "image/png"
        assert got.spoken == "thesis", "the name he heard must not change"

    def test_materialise_is_a_no_op_for_a_document(self, tmp_path):
        sub = _sub("document", "the lab report", path=tmp_path / "a.pdf")
        assert materialise(sub, capture=lambda: {"screenshot": "/nope"}) is sub

    def test_a_failed_capture_leaves_the_subject_without_bytes(self):
        sub = _sub("screen", "thesis")
        got = materialise(sub, capture=lambda: None)
        assert got.path is None

    def test_a_capture_that_raises_is_survivable(self):
        def boom():
            raise OSError("no X")

        got = materialise(_sub("screen", "thesis"), capture=boom)
        assert got.path is None


# ------------------------------------------------------------- the board
class TestBoardSink:
    """Jarvis's own console surface. Reversible, so it fires immediately --
    a throw that asks permission every time stops feeling like a throw."""

    def test_it_is_reversible_and_asks_nothing(self):
        sink = BoardSink(publish=lambda card: None)
        assert sink.reversible
        assert not sink.needs_readback()
        assert not sink.needs_identity

    def test_it_lands_and_hands_the_publisher_the_subject(self):
        seen = []
        sink = BoardSink(publish=seen.append)
        res = sink.deliver(_sub(spoken="the lab report"))
        assert res.landed and not res.held
        assert len(seen) == 1
        assert seen[0].spoken == "the lab report"

    def test_with_no_publisher_it_says_so_and_does_not_land(self):
        sink = BoardSink()
        ok, why = sink.available()
        assert not ok and why
        res = sink.deliver(_sub())
        assert not res.landed
        assert res.held

    def test_a_publisher_that_raises_never_reports_success(self):
        def boom(card):
            raise RuntimeError("the board went away")

        res = BoardSink(publish=boom).deliver(_sub())
        assert not res.landed
        assert res.held

    def test_it_does_not_want_bytes(self):
        """The board shows a name; it is not a file drop."""
        assert not BoardSink(publish=lambda c: None).wants_bytes


def test_the_module_cannot_open_a_window_or_a_lens():
    """The fire-immediately ruling rests on a false board cast being cheap:
    a panel change on Jarvis's own surface, no toplevel, no focus steal. It
    also rests on this module never reaching the camera. Both are pinned in
    the source rather than in a comment, because a later edit that breaks
    either one silently makes the ruling wrong."""
    src = Path(cast_mod.__file__).read_text()
    for banned in ("tkinter", "Toplevel", "launch_app", "windowactivate",
                   "xdg-open", "VideoCapture", "imwrite", "cv2",
                   "ImageGrab", "wm_attributes"):
        assert banned not in src, f"jarvis/cast.py must not reach for {banned}"


# --------------------------------------------------------- HPCOMPUTER
class TestHpcomputerSink:
    """Measured 2026-09-03: hpcomputer.local resolves to 192.168.50.114, its
    ARP entry goes REACHABLE under probe, ping is 100% loss, and 22/445/
    3389/5900/8008/2343 at 00:45 then 22/445/3389 again at 02:40 and 07:21
    all timed out. It is powered on and dropping IP. So this sink HOLDS,
    and holding is a visible, honest, recoverable state."""

    def test_an_unreachable_target_holds_and_never_lands(self):
        sink = HpcomputerSink(probe=Probe(), now=Clock())
        res = sink.deliver(_sub())
        assert not res.landed
        assert res.held
        assert res.fallback == "board"

    def test_the_refusal_says_what_is_wrong_in_his_words(self):
        sink = HpcomputerSink(probe=Probe(), now=Clock())
        line = sink.deliver(_sub()).spoken
        assert "HPCOMPUTER" in line
        assert "listening" in line
        assert "sir" in line

    def test_a_repeat_refusal_inside_a_minute_is_shortened(self):
        clock = Clock()
        sink = HpcomputerSink(probe=Probe(), now=clock)
        first = sink.deliver(_sub()).spoken
        clock.advance(5.0)
        second = sink.deliver(_sub()).spoken
        assert second != first
        assert len(second) < len(first)
        clock.advance(cast_mod.REFUSAL_REPEAT_S + 1.0)
        assert sink.deliver(_sub()).spoken == first

    def test_an_open_port_still_does_not_claim_a_landing(self):
        """If Windows ever opens a port, there is still no transport here.
        A different reason, still held -- never a fabricated success."""
        sink = HpcomputerSink(probe=Probe((True, "", "445")), now=Clock())
        res = sink.deliver(_sub())
        assert not res.landed
        assert res.held
        assert "listening" not in res.spoken

    def test_the_probe_is_cached_between_casts(self):
        probe = Probe()
        clock = Clock()
        sink = HpcomputerSink(probe=probe, now=clock, cache_s=20.0)
        sink.deliver(_sub())
        clock.advance(5.0)
        sink.deliver(_sub())
        assert probe.calls == 1
        clock.advance(20.0)
        sink.deliver(_sub())
        assert probe.calls == 2

    def test_a_probe_that_raises_holds_rather_than_lands(self):
        def boom():
            raise OSError("no route")

        res = HpcomputerSink(probe=boom, now=Clock()).deliver(_sub())
        assert not res.landed and res.held

    def test_constructing_it_probes_nothing(self):
        probe = Probe()
        HpcomputerSink(probe=probe, now=Clock())
        assert probe.calls == 0

    # -- the one payload that genuinely lands there today ---------------
    def test_a_track_rides_spotify_connect_and_lands(self):
        """The one thing that reaches that machine today. Its Spotify client
        makes an OUTBOUND connection, so the firewall that blocks everything
        inbound is irrelevant -- and Jarvis already does this on the spoken
        'play it on hpcomputer'."""
        moved = []
        sink = HpcomputerSink(probe=Probe(), now=Clock(),
                              transfer=lambda dev: moved.append(dev))
        res = sink.deliver(_sub("track", "Kashmir by Led Zeppelin",
                                uri="spotify:track:1"))
        assert res.landed and not res.held
        assert moved == ["HPCOMPUTER"]

    def test_the_track_path_does_not_consult_the_lan_probe(self):
        probe = Probe()
        sink = HpcomputerSink(probe=probe, now=Clock(),
                              transfer=lambda dev: None)
        sink.deliver(_sub("track", "Kashmir"))
        assert probe.calls == 0

    def test_a_track_transfer_needs_no_read_back(self):
        sink = HpcomputerSink(probe=Probe(), now=Clock(),
                              transfer=lambda dev: None)
        assert not sink.needs_readback(_sub("track", "Kashmir"))
        assert sink.needs_readback(_sub("document", "the lab report"))

    def test_a_failed_transfer_holds_and_never_lands(self):
        def boom(dev):
            raise RuntimeError("Spotify said no")

        sink = HpcomputerSink(probe=Probe(), now=Clock(), transfer=boom)
        res = sink.deliver(_sub("track", "Kashmir"))
        assert not res.landed
        assert res.held

    def test_the_track_route_can_be_switched_off(self):
        sink = HpcomputerSink(probe=Probe(), now=Clock(),
                              transfer=lambda dev: None, allow_track=False)
        res = sink.deliver(_sub("track", "Kashmir"))
        assert not res.landed and res.held

    def test_with_no_transfer_wired_a_track_holds(self):
        res = HpcomputerSink(probe=Probe(), now=Clock()).deliver(
            _sub("track", "Kashmir"))
        assert not res.landed and res.held


# ----------------------------------------------------------- the handoff
class TestHandoffSink:
    """The Spark serves, HPCOMPUTER fetches: reversing the arrow is what
    makes a LAN target usable at all through a Windows firewall. It needs
    him to open a URL, so it is a handoff, not a cast."""

    def test_it_needs_a_read_back_because_it_leaves_the_box(self):
        sink = HandoffSink(address=lambda: "192.168.50.109",
                           token=lambda: "t", serving=lambda: True,
                           publish_url=lambda url, sub: None)
        assert not sink.reversible
        assert sink.needs_readback()
        assert sink.wants_bytes

    def test_the_url_is_numeric_never_the_mdns_name(self):
        """avahi-resolve -n spark-509f.local answers 172.17.0.1, the docker0
        bridge. Any URL built from the .local name is unreachable from the
        machine it is for."""
        seen = []
        sink = HandoffSink(address=lambda: "192.168.50.109", port=lambda: 8765,
                           token=lambda: "tok", serving=lambda: True,
                           publish_url=lambda url, sub: seen.append(url))
        res = sink.deliver(_sub("document", "the lab report",
                                path=Path("/home/h/lab.pdf")))
        assert res.landed
        assert seen and ".local" not in seen[0]
        assert "192.168.50.109:8765" in seen[0]
        assert "tok" in seen[0]

    def test_a_name_for_an_address_is_refused_not_resolved(self):
        """An address() that hands back the mDNS name must never become a
        URL: it is refused as no address at all, and nothing is served."""
        seen = []
        sink = HandoffSink(address=lambda: "spark-509f.local", token=lambda: "t",
                           serving=lambda: True,
                           publish_url=lambda url, sub: seen.append(url))
        assert not sink.available()[0]
        res = sink.deliver(_sub("document", "d", path=Path("/home/h/lab.pdf")))
        assert res.held and not res.landed
        assert seen == []
        assert sink.url_for("x") == ""
        assert sink.served() == {}

    def test_a_dark_server_is_unavailable_not_a_silent_failure(self):
        sink = HandoffSink(address=lambda: "192.168.50.109",
                           token=lambda: "t", serving=lambda: False,
                           publish_url=lambda url, sub: None)
        ok, why = sink.available()
        assert not ok and why
        res = sink.deliver(_sub())
        assert not res.landed and res.held

    def test_a_public_address_is_refused(self):
        """webapp.lan_address() returns '' when the default route is public.
        Serving a file there is not a handoff, it is an exposure."""
        sink = HandoffSink(address=lambda: "", token=lambda: "t",
                           serving=lambda: True,
                           publish_url=lambda url, sub: None)
        ok, _ = sink.available()
        assert not ok

    def test_no_token_is_no_handoff(self):
        sink = HandoffSink(address=lambda: "192.168.50.109", token=lambda: "",
                           serving=lambda: True,
                           publish_url=lambda url, sub: None)
        assert not sink.available()[0]

    def test_a_subject_with_no_bytes_cannot_be_handed_off(self):
        sink = HandoffSink(address=lambda: "192.168.50.109", token=lambda: "t",
                           serving=lambda: True,
                           publish_url=lambda url, sub: None)
        res = sink.deliver(_sub("screen", "thesis"))
        assert not res.landed and res.held


# ------------------------------------------------------- choosing a sink
class TestPickSink:
    """The sector map ships EMPTY, so every fling goes to the board. Only he
    knows where the machines sit relative to the lens; a direction map for a
    set of size one is wrong-target risk for no benefit."""

    def test_the_default_map_sends_everything_to_the_board(self):
        registry = {"board": BoardSink(publish=lambda c: None)}
        for d in ("left", "right", "up", "down", ""):
            assert pick_sink(d, registry=registry).name == "board"

    def test_a_configured_direction_routes(self):
        registry = {"board": BoardSink(publish=lambda c: None),
                    "hpcomputer": HpcomputerSink(probe=Probe(), now=Clock())}
        got = pick_sink("right", get_option=lambda k, d=None:
                        {"right": "hpcomputer"} if k == cast_mod.OPTION_SINKS
                        else d, registry=registry)
        assert got.name == "hpcomputer"

    def test_an_unknown_sink_name_falls_back_to_the_board(self):
        registry = {"board": BoardSink(publish=lambda c: None)}
        got = pick_sink("left", get_option=lambda k, d=None:
                        {"left": "the moon"} if k == cast_mod.OPTION_SINKS
                        else d, registry=registry)
        assert got.name == "board"

    def test_a_config_of_the_wrong_shape_falls_back_to_the_board(self):
        registry = {"board": BoardSink(publish=lambda c: None)}
        got = pick_sink("left", get_option=lambda k, d=None: "not a map",
                        registry=registry)
        assert got.name == "board"

    @pytest.mark.parametrize("deg,word", [
        (0.0, "right"), (44.0, "right"), (46.0, "up"), (90.0, "up"),
        (134.0, "up"), (136.0, "left"), (180.0, "left"), (225.0, "down"),
        (270.0, "down"), (315.0, "right"), (-90.0, "down"), (360.0, "right"),
    ])
    def test_direction_is_a_sign_not_an_angle(self, deg, word):
        """Two or three samples at 7.5 fps cannot honestly carry more
        resolution than a quadrant."""
        assert direction_word(deg) == word

    def test_resolve_sink_takes_degrees(self):
        registry = {"board": BoardSink(publish=lambda c: None)}
        assert resolve_sink(90.0, registry=registry).name == "board"


# ------------------------------------------------------------ casting
class _Recorder:
    """A sink that records what it was asked to do."""

    name = "recorder"
    needs_identity = False

    def __init__(self, *, reversible=True, wants_bytes=False, ok=True):
        self.reversible = reversible
        self.wants_bytes = wants_bytes
        self.ok = ok
        self.delivered = []

    def available(self):
        return (True, "") if self.ok else (False, "shut")

    def needs_readback(self, subject=None):
        return not self.reversible

    def deliver(self, subject):
        self.delivered.append(subject)
        return CastResult(landed=True, held=False, spoken="On the board, sir.",
                          sink=self.name)


class TestCast:
    def test_a_reversible_sink_acts_at_once(self):
        sink = _Recorder()
        said = []
        assert cast(sink, _sub(), speak=said.append) == "landed"
        assert len(sink.delivered) == 1
        assert said == ["On the board, sir."]

    def test_an_irreversible_sink_only_proposes(self):
        sink = _Recorder(reversible=False)
        offers = []
        status = cast(sink, _sub(), speak=lambda s: None,
                      propose=lambda run, line: offers.append((run, line)))
        assert status == "proposed"
        assert sink.delivered == [], "nothing may happen before the spoken yes"
        assert len(offers) == 1
        assert "sir" in offers[0][1]
        offers[0][0]()
        assert len(sink.delivered) == 1

    def test_an_irreversible_sink_with_no_propose_is_refused(self):
        """A fling is a much weaker signal of intent than a sentence. With
        no confirmation route wired the answer is no, not 'do it anyway'."""
        sink = _Recorder(reversible=False)
        said = []
        assert cast(sink, _sub(), speak=said.append) == "refused"
        assert sink.delivered == []
        assert said and "sir" in said[0]

    def test_an_empty_subject_casts_nothing_and_says_nothing(self):
        sink = _Recorder()
        said = []
        empty = CastSubject(kind="empty", spoken="", at=0.0)
        assert cast(sink, empty, speak=said.append) == "empty"
        assert sink.delivered == []
        assert said == []

    def test_a_sink_that_wants_bytes_gets_them(self, tmp_path):
        png = tmp_path / "latest.png"
        png.write_bytes(b"\x89PNG")
        sink = _Recorder(wants_bytes=True)
        cast(sink, _sub("screen", "thesis"), speak=lambda s: None,
             capture=lambda: {"screenshot": str(png)})
        assert sink.delivered[0].path == png

    def test_a_sink_that_does_not_want_bytes_never_captures(self):
        calls = []
        cast(_Recorder(), _sub("screen", "thesis"), speak=lambda s: None,
             capture=lambda: calls.append(1))
        assert calls == []

    def test_a_held_cast_reports_held_not_landed(self):
        sink = HpcomputerSink(probe=Probe(), now=Clock())
        said = []
        assert cast(sink, _sub(), speak=said.append) == "held"
        assert said and "HPCOMPUTER" in said[0]

    def test_a_sink_that_raises_never_reports_a_landing(self):
        class Broken(_Recorder):
            def deliver(self, subject):
                raise RuntimeError("gone")

        said = []
        assert cast(Broken(), _sub(), speak=said.append) == "held"
        assert said

    def test_landed_and_held_are_never_both_true(self):
        """The single most important invariant in the module."""
        clock = Clock()
        sinks = [BoardSink(publish=lambda c: None), BoardSink(),
                 HpcomputerSink(probe=Probe(), now=clock),
                 HpcomputerSink(probe=Probe((True, "", "445")), now=Clock()),
                 HpcomputerSink(probe=Probe(), now=Clock(),
                                transfer=lambda d: None),
                 HandoffSink(address=lambda: "192.168.50.109",
                             token=lambda: "t", serving=lambda: True,
                             publish_url=lambda u, s: None),
                 HandoffSink(address=lambda: "", token=lambda: "",
                             serving=lambda: False)]
        subjects = [_sub("document", "d", path=Path("/tmp/a.pdf")),
                    _sub("track", "t"), _sub("screen", "s")]
        for sink in sinks:
            for sub in subjects:
                res = sink.deliver(sub)
                assert not (res.landed and res.held), (sink.name, sub.kind)
                assert res.spoken or not res.held, (sink.name, sub.kind)


# ------------------------------------------------------- the held register
class TestCastOutbox:
    """What happens to a held cast: it is kept, it is visible, it expires,
    and when the target comes back it OFFERS rather than delivering -- the
    same ruling he made about the morning briefing."""

    def test_a_held_cast_is_pending(self):
        box = CastOutbox(now=Clock())
        box.hold(_sub(), "hpcomputer", "nothing listening")
        assert len(box.pending()) == 1
        assert box.pending()[0].sink == "hpcomputer"

    def test_it_does_not_expire_early(self):
        clock = Clock()
        box = CastOutbox(now=clock, ttl_s=1800.0)
        box.hold(_sub(), "hpcomputer", "shut")
        clock.advance(1799.0)
        assert box.poll() == []
        assert len(box.pending()) == 1

    def test_it_expires_and_says_so_once(self):
        clock = Clock()
        box = CastOutbox(now=clock, ttl_s=100.0)
        box.hold(_sub(), "hpcomputer", "shut")
        clock.advance(101.0)
        events = box.poll()
        assert [e.kind for e in events] == ["expired"]
        assert box.pending() == ()
        assert box.poll() == []

    def test_a_target_coming_back_offers_and_never_delivers(self):
        clock = Clock()
        box = CastOutbox(now=clock)
        box.hold(_sub(spoken="the lab report"), "hpcomputer", "shut")
        events = box.poll(available=lambda name: (True, ""))
        assert [e.kind for e in events] == ["ready"]
        assert "the lab report" in events[0].spoken
        assert box.pending(), "a ready cast is still held until he says yes"

    def test_the_offer_is_made_once_not_every_poll(self):
        clock = Clock()
        box = CastOutbox(now=clock)
        box.hold(_sub(), "hpcomputer", "shut")
        assert box.poll(available=lambda n: (True, "")) != []
        assert box.poll(available=lambda n: (True, "")) == []

    def test_an_availability_check_that_raises_keeps_the_cast(self):
        def boom(name):
            raise OSError("no route")

        box = CastOutbox(now=Clock())
        box.hold(_sub(), "hpcomputer", "shut")
        assert box.poll(available=boom) == []
        assert len(box.pending()) == 1

    def test_capacity_evicts_the_oldest_and_says_so(self):
        """Nothing may vanish unannounced: an eviction is an event he can be
        told about, not a silent forget."""
        clock = Clock()
        box = CastOutbox(now=clock, capacity=2)
        box.hold(_sub(spoken="one"), "hpcomputer", "shut")
        clock.advance(1.0)
        box.hold(_sub(spoken="two"), "hpcomputer", "shut")
        clock.advance(1.0)
        box.hold(_sub(spoken="three"), "hpcomputer", "shut")
        assert len(box.pending()) == 2
        assert [h.subject.spoken for h in box.pending()] == ["two", "three"]
        events = box.poll()
        assert [e.kind for e in events] == ["evicted"]
        assert events[0].held.subject.spoken == "one"

    def test_an_empty_subject_is_never_held(self):
        box = CastOutbox(now=Clock())
        assert box.hold(CastSubject(kind="empty", spoken="", at=0.0),
                        "hpcomputer", "shut") is None
        assert box.pending() == ()

    def test_drop_and_clear(self):
        box = CastOutbox(now=Clock())
        held = box.hold(_sub(), "hpcomputer", "shut")
        assert box.drop(held.ident)
        assert not box.drop(held.ident)
        box.hold(_sub(), "hpcomputer", "shut")
        box.clear()
        assert box.pending() == ()


# --------------------------------------------------------- the live probe
class TestDefaultProbe:
    """The refusal line must come from a LIVE reading, never a constant: the
    day Windows opens a port, Jarvis must stop saying nothing is listening.
    The probe itself is never exercised here -- the suite must pass with
    HPCOMPUTER up, down or absent."""

    def test_the_sink_defaults_to_the_live_probe(self):
        sink = HpcomputerSink(now=Clock())
        assert callable(sink._probe)
        assert sink._probe is not None

    def test_the_probe_budget_is_short_enough_for_a_gesture(self):
        """A cast is a physical metaphor; a two-second stall inside it reads
        as the feature being broken."""
        assert cast_mod.PROBE_TIMEOUT_S <= 0.5
        assert cast_mod.PROBE_CACHE_S >= 5.0

    def test_no_socket_is_opened_by_importing_or_constructing(self, monkeypatch):
        def boom(*a, **kw):
            raise AssertionError("the suite must not touch the network")

        monkeypatch.setattr(cast_mod.socket, "socket", boom)
        monkeypatch.setattr(cast_mod.socket, "getaddrinfo", boom)
        HpcomputerSink(now=Clock())
        HandoffSink(address=lambda: "192.168.50.109", token=lambda: "t",
                    serving=lambda: True, publish_url=lambda u, s: None)
        resolve_subject(None, None, now=1000.0,
                        providers={"screen": lambda: None})

    def test_the_probe_reports_a_reason_it_did_not_connect(self, monkeypatch):
        """No network: an empty port list can answer without a socket."""
        ok, reason, detail = cast_mod.probe_tcp("192.0.2.1", (), timeout=0.01)
        assert not ok
        assert reason
        assert detail == ""


def test_time_is_injectable_everywhere():
    """Nothing in this module may wait on the wall clock; every deadline is
    driven by an injected now()."""
    clock = Clock()
    box = CastOutbox(now=clock)
    sink = HpcomputerSink(probe=Probe(), now=clock)
    t0 = time.monotonic()
    box.hold(_sub(), "hpcomputer", "shut")
    clock.advance(10_000.0)
    sink.deliver(_sub())
    assert box.poll() != []
    assert time.monotonic() - t0 < 1.0


# ------------------------------------------------ the house numbers agree
class TestHouseNumbers:
    """Two constants are mirrored here rather than imported, because
    commander is 8000 lines and webapp opens a server; the mirror must not
    drift."""

    def test_the_document_window_is_commanders(self):
        from jarvis import commander
        assert cast_mod.DOCUMENT_MAX_AGE_S == commander.LAST_DOCUMENT_S

    def test_the_handoff_port_is_the_phone_pages(self):
        from jarvis import webapp
        assert cast_mod.HANDOFF_PORT == webapp.DEFAULT_PORT

    def test_private_address_agrees_with_webapp(self):
        from jarvis import webapp
        for host in ("192.168.50.109", "10.1.2.3", "100.70.145.63", "8.8.8.8",
                     "", "0.0.0.0", "spark-509f.local", "::1", "2001:db8::1"):
            assert cast_mod._private_address(host) == webapp.is_private_host(host), host

    def test_the_hpcomputer_device_name_is_spotifys(self):
        from jarvis.tools import spotify
        assert cast_mod.HPCOMPUTER_DEVICE == spotify.DEFAULT_DEVICE

    def test_the_host_is_a_number_never_a_name(self):
        assert ".local" not in cast_mod.HPCOMPUTER_HOST
        assert cast_mod._numeric_host(cast_mod.HPCOMPUTER_HOST) == "192.168.50.114"

    def test_the_four_earcons_are_distinct_and_real(self):
        from jarvis import earcons
        names = [cast_mod.earcon_for(k) for k in ("grab", "landed", "held", "drop")]
        assert len(set(names)) == 4, "a shared tone is dropped by the 4 s cooldown"
        for n in names:
            assert n in earcons.NAMES, n
        for silent in ("empty", "refused", "proposed", "vetoed", ""):
            assert cast_mod.earcon_for(silent) == ""


class TestCastResultInvariant:
    def test_both_true_is_refused_at_construction(self):
        with pytest.raises(ValueError):
            CastResult(landed=True, held=True, spoken="x")

    def test_a_silent_hold_is_refused(self):
        with pytest.raises(ValueError):
            CastResult(landed=False, held=True, spoken="")


# ------------------------------------------------------------ identity
class TestIdentity:
    """Consulted, never required: '' is no opinion, another name is a veto,
    and a positive match is demanded only by a sink that says so."""

    def test_no_opinion_lets_a_reversible_cast_through(self):
        sink = _Recorder()
        assert cast(sink, _sub(), speak=lambda s: None, identity="") == "landed"
        assert cast(sink, _sub(), speak=lambda s: None, identity=None) == "landed"

    def test_a_different_name_vetoes_silently(self):
        sink = _Recorder()
        said = []
        assert cast(sink, _sub(), speak=said.append, identity="ali") == "vetoed"
        assert sink.delivered == []
        assert said == []

    def test_an_irreversible_sink_that_needs_identity_refuses_without_it(self):
        sink = _Recorder(reversible=False)
        sink.needs_identity = True
        said = []
        status = cast(sink, _sub(), speak=said.append,
                      propose=lambda run, line: None, identity="")
        assert status == "refused"
        assert said == [cast_mod.IDENTITY_LINE]
        assert sink.delivered == []

    def test_a_positive_match_lets_it_propose(self):
        sink = _Recorder(reversible=False)
        sink.needs_identity = True
        offers = []
        status = cast(sink, _sub(), speak=lambda s: None,
                      propose=lambda run, line: offers.append(line),
                      identity="Hunter")
        assert status == "proposed"
        assert offers and "sir" in offers[0]

    def test_an_unreachable_target_holds_before_identity_is_asked(self):
        """Nothing leaves a box that is deaf, whoever threw it."""
        sink = HpcomputerSink(probe=Probe(), now=Clock())
        said = []
        assert cast(sink, _sub(), speak=said.append, identity="") == "held"
        assert "HPCOMPUTER" in said[0]


# ------------------------------------------------------------ fallback
class TestFallback:
    def test_a_held_cast_lands_on_the_board_silently(self):
        board_seen = []
        board = BoardSink(publish=board_seen.append)
        sink = HpcomputerSink(probe=Probe(), now=Clock())
        said = []
        assert cast(sink, _sub(spoken="the lab report"), speak=said.append,
                    fallback=board) == "held"
        assert [s.spoken for s in board_seen] == ["the lab report"]
        assert len(said) == 1 and "HPCOMPUTER" in said[0]

    def test_a_fallback_that_fails_does_not_change_the_answer(self):
        def boom(card):
            raise RuntimeError("no board")

        sink = HpcomputerSink(probe=Probe(), now=Clock())
        assert cast(sink, _sub(), speak=lambda s: None,
                    fallback=BoardSink(publish=boom)) == "held"

    def test_a_landed_cast_never_touches_the_fallback(self):
        board_seen = []
        cast(_Recorder(), _sub(), speak=lambda s: None,
             fallback=BoardSink(publish=board_seen.append))
        assert board_seen == []


# ----------------------------------------------------- the HPCOMPUTER seam
class TestHpcomputerSeam:
    """The SSH transport Hunter is enabling on his side plugs in here.
    Until it does, available() is False whatever the probe says."""

    def test_available_is_false_without_a_transport_even_with_a_port_open(self):
        sink = HpcomputerSink(probe=Probe((True, "", "22")), now=Clock())
        ok, why = sink.available()
        assert not ok and "22" in why

    def test_a_transport_and_a_live_port_make_it_available(self):
        sink = HpcomputerSink(probe=Probe((True, "", "22")), now=Clock(),
                              transport=lambda sub: "ok")
        assert sink.available() == (True, "")

    def test_a_transport_with_a_dead_port_still_holds(self):
        sink = HpcomputerSink(probe=Probe(), now=Clock(),
                              transport=lambda sub: "ok")
        res = sink.deliver(_sub("document", "d", path=Path("/tmp/a.pdf")))
        assert res.held and not res.landed

    def test_a_transport_lands_a_file(self):
        pushed = []
        sink = HpcomputerSink(probe=Probe((True, "", "22")), now=Clock(),
                              transport=lambda sub: pushed.append(sub.path))
        res = sink.deliver(_sub("document", "the lab report",
                                path=Path("/tmp/a.pdf")))
        assert res.landed and not res.held
        assert pushed == [Path("/tmp/a.pdf")]
        assert "HPCOMPUTER" in res.spoken

    def test_a_transport_that_raises_holds(self):
        def boom(sub):
            raise OSError("ssh: connection refused")

        sink = HpcomputerSink(probe=Probe((True, "", "22")), now=Clock(),
                              transport=boom)
        res = sink.deliver(_sub("document", "d", path=Path("/tmp/a.pdf")))
        assert res.held and not res.landed

    def test_a_transport_with_no_bytes_holds(self):
        sink = HpcomputerSink(probe=Probe((True, "", "22")), now=Clock(),
                              transport=lambda sub: "ok")
        res = sink.deliver(_sub("screen", "thesis"))
        assert res.held and not res.landed

    def test_through_cast_a_live_seam_is_proposed_not_acted_on(self):
        """With the SSH seam plugged in, a fling still only proposes."""
        pushed = []
        sink = HpcomputerSink(probe=Probe((True, "", "22")), now=Clock(),
                              transport=lambda sub: pushed.append(1),
                              needs_identity=False)
        offers = []
        status = cast(sink, _sub("document", "d", path=Path("/tmp/a.pdf")),
                      speak=lambda s: None,
                      propose=lambda run, line: offers.append((run, line)))
        assert status == "proposed"
        assert pushed == []
        assert "HPCOMPUTER" in offers[0][1]
        res = offers[0][0]()
        assert res.landed and pushed == [1]

    def test_the_default_probe_refuses_a_name_without_a_socket(self, monkeypatch):
        def boom(*a, **kw):
            raise AssertionError("no socket for a name")

        monkeypatch.setattr(cast_mod.socket, "socket", boom)
        ok, reason, detail = cast_mod.probe_tcp("hpcomputer.local", (22,), timeout=0.01)
        assert not ok and reason and detail == ""


# ---------------------------------------------------------- the handoff page
class TestHandoffRegister:
    def _sink(self, seen, now):
        return HandoffSink(address=lambda: "192.168.50.109", token=lambda: "tok",
                           serving=lambda: True, now=now,
                           publish_url=lambda url, sub: seen.append(url))

    def test_the_served_file_is_behind_the_token(self, tmp_path):
        pdf = tmp_path / "lab.pdf"
        pdf.write_bytes(b"%PDF")
        seen, clock = [], Clock()
        sink = self._sink(seen, clock)
        assert sink.deliver(_sub("document", "d", path=pdf)).landed
        ident = seen[0].split("/cast/")[1].split("?")[0]
        assert sink.path_for(ident, "tok") == pdf
        assert sink.path_for(ident, "wrong") is None
        assert sink.path_for("nope", "tok") is None

    def test_a_served_url_expires(self, tmp_path):
        pdf = tmp_path / "lab.pdf"
        pdf.write_bytes(b"%PDF")
        seen, clock = [], Clock()
        sink = self._sink(seen, clock)
        sink.deliver(_sub("document", "d", path=pdf))
        ident = seen[0].split("/cast/")[1].split("?")[0]
        clock.advance(cast_mod.HANDOFF_TTL_S + 1.0)
        assert sink.path_for(ident, "tok") is None
        assert sink.served() == {}

    def test_a_vanished_file_is_not_served(self, tmp_path):
        seen = []
        sink = self._sink(seen, Clock())
        sink.deliver(_sub("document", "d", path=tmp_path / "gone.pdf"))
        ident = seen[0].split("/cast/")[1].split("?")[0]
        assert sink.path_for(ident, "tok") is None

    def test_a_failed_publish_takes_the_page_down(self, tmp_path):
        pdf = tmp_path / "lab.pdf"
        pdf.write_bytes(b"%PDF")

        def boom(url, sub):
            raise RuntimeError("no board")

        sink = HandoffSink(address=lambda: "192.168.50.109", token=lambda: "tok",
                           serving=lambda: True, publish_url=boom)
        res = sink.deliver(_sub("document", "d", path=pdf))
        assert res.held and not res.landed
        assert sink.served() == {}


# ------------------------------------------------------- teaching a side
class TestTeaching:
    """Which side HPCOMPUTER sits on is his open question; until he answers
    it every throw goes to the board and Jarvis says so."""

    @pytest.mark.parametrize("text,want", [
        ("HPCOMPUTER is on my right", ("right", "hpcomputer")),
        ("the pc is to the left", ("left", "hpcomputer")),
        ("hp computer is on the left of me", ("left", "hpcomputer")),
        ("right is HPCOMPUTER", ("right", "hpcomputer")),
        ("the left is the board", ("left", "board")),
        ("throw right to the pc", ("right", "hpcomputer")),
        ("throwing left goes to the board", ("left", "board")),
        ("right is the pc", ("right", "hpcomputer")),
        ("HPCOMPUTER is behind the monitor", None),
        ("what's the weather", None),
        ("", None),
        # The shape of a teaching with no target in it: NOT a teaching.
        ("right is fine", None),
        ("the left is better", None),
        ("what is left", None),
        ("the light is on the left", None),
        ("turn right", None),
    ])
    def test_parse(self, text, want):
        assert parse_side_teaching(text) == want

    def test_teaching_writes_the_map_and_says_so(self):
        saved = {}
        registry = {"board": BoardSink(publish=lambda c: None),
                    "hpcomputer": HpcomputerSink(probe=Probe(), now=Clock())}
        line = teach_sink("right", "the pc", registry=registry,
                          set_option=lambda k, v: saved.update({k: v}) or True,
                          get_option=lambda k, d=None: saved.get(k, d))
        assert saved == {cast_mod.OPTION_SINKS: {"right": "hpcomputer"}}
        assert line == "Right is HPCOMPUTER from now on, sir."
        assert pick_sink("right", get_option=lambda k, d=None: saved.get(k, d),
                         registry=registry).name == "hpcomputer"
        assert is_taught(lambda k, d=None: saved.get(k, d))

    def test_moving_a_machine_drops_its_old_side(self):
        saved = {cast_mod.OPTION_SINKS: {"left": "hpcomputer"}}
        teach_sink("right", "hpcomputer",
                   set_option=lambda k, v: saved.update({k: v}) or True,
                   get_option=lambda k, d=None: saved.get(k, d))
        assert saved[cast_mod.OPTION_SINKS] == {"right": "hpcomputer"}

    def test_an_unknown_target_is_refused(self):
        line = teach_sink("right", "the moon", registry={"board": None},
                          set_option=lambda k, v: True)
        assert "moon" in line and "sir" in line

    def test_a_failed_save_is_reported(self):
        assert teach_sink("left", "hpcomputer",
                          set_option=lambda k, v: False) == cast_mod.TAUGHT_FAILED_LINE

    def test_untaught_by_default(self):
        assert not is_taught(None)
        assert not is_taught(lambda k, d=None: d)
        assert "HPCOMPUTER" in cast_mod.UNTAUGHT_LINE

    def test_vertical_entries_are_ignored(self):
        """Down is the cancel and up is not a target (gesture.py); a map
        entry for either must not route."""
        registry = {"board": BoardSink(publish=lambda c: None),
                    "hpcomputer": HpcomputerSink(probe=Probe(), now=Clock())}
        opt = lambda k, d=None: {"up": "hpcomputer", "down": "hpcomputer"}  # noqa: E731
        assert pick_sink("up", get_option=opt, registry=registry).name == "board"
        assert pick_sink("down", get_option=opt, registry=registry).name == "board"


# --------------------------------------------------------- the names
class TestNames:
    def test_track_name_shapes(self):
        assert cast_mod.track_spoken_name("Kashmir by Led Zeppelin, sir — on HPCOMPUTER.") == \
            "Kashmir by Led Zeppelin"
        assert cast_mod.track_spoken_name("Kashmir by Led Zeppelin, sir, paused on Spark.") == \
            "Kashmir by Led Zeppelin"
        assert cast_mod.track_spoken_name("") == ""

    def test_a_titleless_window_is_still_the_screen(self):
        got = cast_mod.screen_subject("", now=1.0)
        assert got is not None and got.kind == "screen" and got.spoken == "the screen"
        assert cast_mod.screen_subject(None) is None

    def test_a_services_bag_is_unwrapped_for_the_track(self):
        spot = _spotify()
        bag = SimpleNamespace(spotify=spot)
        got = resolve_subject(None, bag, now=1000.0)
        assert got.kind == "track"


class TestOutboxMore:
    def test_a_target_that_goes_away_again_is_offered_again_when_it_returns(self):
        box = CastOutbox(now=Clock())
        box.hold(_sub(), "hpcomputer", "shut")
        assert [e.kind for e in box.poll(available=lambda n: (True, ""))] == ["ready"]
        assert box.poll(available=lambda n: (False, "shut")) == []
        assert [e.kind for e in box.poll(available=lambda n: (True, ""))] == ["ready"]

    def test_take_spends_the_cast(self):
        box = CastOutbox(now=Clock())
        held = box.hold(_sub(), "hpcomputer", "shut")
        assert box.take(held.ident) is held
        assert box.pending() == ()
        assert box.take(held.ident) is None
