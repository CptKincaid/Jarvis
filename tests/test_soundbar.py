"""The sink sentinel (jarvis/soundbar.py).

The fixture's HDMI row is a real ``pactl list short sinks`` line from this
box (taken 2026-08-31, with the soundbar's battery flat -- which is why the
bluez row beside it carries PipeWire's standard name shape rather than a
live capture of the soundbar itself).

Nothing here runs pactl: every test drives ``FakeRun``. A test that talked
to the live PipeWire could move the user's default sink.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis import soundbar as sb

FIXTURE = Path(__file__).parent / "fixtures" / "pactl_sinks.txt"
BOTH = FIXTURE.read_text()
HDMI = "alsa_output.platform-NVDA2014_00.hdmi-stereo"
BAR = "bluez_output.FC_58_FA_31_9C_2B.1"
HDMI_ONLY = BOTH.splitlines()[0] + "\n"
NONE_AT_ALL = "0\tauto_null\tPipeWire\ts16le 2ch 48000Hz\tSUSPENDED\n"


class FakeRun:
    """The one subprocess seam: answers the two listings, records writes."""

    def __init__(self, sinks=BOTH, default=HDMI, rc=0):
        self.sinks = sinks
        self.default = default
        self.rc = rc
        self.calls: list[list] = []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if argv[:4] == ["pactl", "list", "short", "sinks"]:
            out = self.sinks
        elif argv[:2] == ["pactl", "get-default-sink"]:
            out = self.default + "\n"
        elif argv[:2] == ["pactl", "set-default-sink"]:
            self.default = argv[2]
            out = ""
        else:
            out = ""
        return SimpleNamespace(returncode=self.rc, stdout=out, stderr="")

    @property
    def writes(self):
        return [c for c in self.calls if "set-default-sink" in c]


class Speaker:
    def __init__(self):
        self.lines: list[tuple] = []

    def __call__(self, text, proactive=False, kind="message"):
        self.lines.append((text, proactive, kind))

    @property
    def texts(self):
        return [t for t, _p, _k in self.lines]


def sentinel(tmp_path, run=None, say=None, pinned="", speaking=False, **cfg):
    settings = {"audio.sink_watch": True, "audio.preferred_sink": "",
                "audio.restore_sink": False}
    settings.update(cfg)
    conf = SimpleNamespace(get=lambda k, d=None: settings.get(k, d))
    return sb.SoundbarSentinel(
        cfg=conf, say=say, run=run or FakeRun(),
        state_path=tmp_path / "soundbar.json",
        playback_device=lambda: pinned,
        speaking=lambda: speaking)


# ---------------------------------------------------------------- parsing
def test_parse_sinks_reads_the_live_row():
    sinks = sb.parse_sinks(BOTH)
    assert [s["name"] for s in sinks] == [HDMI, BAR]
    assert sinks[0]["index"] == 54
    assert sinks[0]["state"] == "RUNNING"
    assert sinks[1]["driver"] == "PipeWire"


def test_parse_sinks_ignores_junk_lines():
    assert sb.parse_sinks("") == []
    assert sb.parse_sinks("not a sink line\n") == []


def test_classify_and_describe_name_the_two_devices():
    assert sb.classify(BAR) == sb.BLUETOOTH
    assert sb.classify(HDMI) == sb.HDMI
    assert sb.classify("auto_null") == sb.DUMMY
    assert sb.describe(BAR) == "the soundbar"
    assert sb.describe(HDMI) == "the monitor"


def test_match_sink_takes_a_fragment_or_the_whole_name():
    names = [HDMI, BAR]
    assert sb.match_sink(names, BAR) == BAR
    assert sb.match_sink(names, "bluez") == BAR
    assert sb.match_sink(names, "FC_58_FA") == BAR
    assert sb.match_sink(names, "hdmi") == HDMI
    assert sb.match_sink(names, "") == ""
    assert sb.match_sink(names, "headphones") == ""


# ------------------------------------------------------------- the reading
def test_healthy_box_says_nothing_and_learns_the_soundbar(tmp_path):
    say = Speaker()
    s = sentinel(tmp_path, run=FakeRun(default=BAR), say=say)
    reading = s.tick()
    assert reading["status"] == "ok"
    assert reading["preferred"] == BAR          # learned: the only bluez sink
    assert say.texts == []                      # a sink appearing is not news


def test_the_night_it_was_written_one_line_about_the_monitor(tmp_path):
    """The soundbar's battery dies: PipeWire moves the default to HDMI."""
    say = Speaker()
    run = FakeRun(default=BAR)
    s = sentinel(tmp_path, run=run, say=say)
    s.tick()                                    # healthy: learns the soundbar
    run.sinks, run.default = HDMI_ONLY, HDMI    # the battery goes flat
    reading = s.tick()
    assert reading["status"] == "missing"
    assert say.texts == ["I'm coming out of the monitor, sir; "
                         "the soundbar has dropped."]
    assert say.lines[0][1] is True              # proactive: quiet hours hold it


def test_the_drop_is_spoken_once_not_every_tick(tmp_path):
    say = Speaker()
    run = FakeRun(default=BAR)
    s = sentinel(tmp_path, run=run, say=say)
    s.tick()
    run.sinks, run.default = HDMI_ONLY, HDMI
    for _ in range(4):
        s.tick()
    assert len(say.texts) == 1


def test_a_restart_into_a_dead_soundbar_does_not_say_it_again(tmp_path):
    """The latch is the state file: faults.py's rule about telling him
    twice applies across a restart, which is when it actually bites."""
    say = Speaker()
    run = FakeRun(default=BAR)
    first = sentinel(tmp_path, run=run, say=say)
    first.tick()
    run.sinks, run.default = HDMI_ONLY, HDMI
    first.tick()
    assert len(say.texts) == 1
    again = sentinel(tmp_path, run=FakeRun(sinks=HDMI_ONLY, default=HDMI), say=say)
    again.tick()
    assert len(say.texts) == 1


def test_a_configured_preference_is_missed_from_the_first_tick(tmp_path):
    """No learning needed: a name in assistant.json is a preference even
    before the sentinel has ever seen that sink."""
    say = Speaker()
    s = sentinel(tmp_path, run=FakeRun(sinks=HDMI_ONLY, default=HDMI), say=say,
                 **{"audio.preferred_sink": "bluez_output"})
    reading = s.tick()
    assert reading["status"] == "missing"
    assert "the soundbar has dropped" in say.texts[0]


def test_the_soundbar_coming_back_points_at_it(tmp_path):
    say = Speaker()
    run = FakeRun(sinks=HDMI_ONLY, default=HDMI)
    s = sentinel(tmp_path, run=run, say=say,
                 **{"audio.preferred_sink": "bluez_output"})
    s.tick()
    run.sinks = BOTH                            # he charges it; default stays HDMI
    reading = s.tick()
    assert reading["status"] == "elsewhere"
    assert say.texts[-1] == ("The soundbar is back, sir; "
                             "I'm still coming out of the monitor.")
    assert run.writes == []                     # restore is off by default


def test_restore_is_off_by_default_and_never_writes(tmp_path):
    run = FakeRun(sinks=BOTH, default=HDMI)
    s = sentinel(tmp_path, run=run, **{"audio.preferred_sink": "bluez"})
    for _ in range(3):
        s.tick()
    assert run.writes == []
    assert run.default == HDMI


def test_restore_moves_the_default_back_when_switched_on(tmp_path):
    say = Speaker()
    run = FakeRun(sinks=HDMI_ONLY, default=HDMI)
    s = sentinel(tmp_path, run=run, say=say,
                 **{"audio.preferred_sink": "bluez", "audio.restore_sink": True})
    s.tick()                                    # missing
    run.sinks = BOTH
    reading = s.tick()
    assert reading["moved"] is True
    assert run.writes == [["pactl", "set-default-sink", BAR]]
    assert run.default == BAR
    assert say.texts[-1] == "The soundbar is back, sir; I've moved my voice across."


def test_restore_waits_while_jarvis_is_mid_burst(tmp_path):
    """Moving the default under a playing paplay splits the sentence
    across two speakers; it waits for the next tick instead."""
    run = FakeRun(sinks=HDMI_ONLY, default=HDMI)
    s = sentinel(tmp_path, run=run, speaking=True,
                 **{"audio.preferred_sink": "bluez", "audio.restore_sink": True})
    s.tick()
    run.sinks = BOTH
    reading = s.tick()
    assert reading["moved"] is False
    assert run.writes == []


def test_restore_records_what_it_moved_away_from(tmp_path):
    import json
    run = FakeRun(sinks=HDMI_ONLY, default=HDMI)
    s = sentinel(tmp_path, run=run,
                 **{"audio.preferred_sink": "bluez", "audio.restore_sink": True})
    s.tick()
    run.sinks = BOTH
    s.tick()
    state = json.loads((tmp_path / "soundbar.json").read_text())
    assert state["restored_from"] == HDMI       # reversible by hand


def test_a_pinned_playback_device_stops_the_restore(tmp_path):
    """With config.playback_device set, the default sink is not where his
    voice goes -- moving it would change his desktop and buy nothing."""
    run = FakeRun(sinks=BOTH, default=HDMI)
    s = sentinel(tmp_path, run=run, pinned=BAR,
                 **{"audio.preferred_sink": "bluez", "audio.restore_sink": True})
    reading = s.tick()
    assert reading["target"] == BAR             # paplay --device wins
    assert reading["status"] == "ok"
    assert run.writes == []


def test_a_pinned_device_that_is_gone_reads_as_missing(tmp_path):
    say = Speaker()
    s = sentinel(tmp_path, run=FakeRun(sinks=HDMI_ONLY, default=HDMI), say=say,
                 pinned=BAR)
    reading = s.tick()
    assert reading["status"] == "missing"
    assert say.texts and "nowhere to speak from" in say.texts[0]


def test_a_box_with_only_a_null_sink_is_dead(tmp_path):
    say = Speaker()
    s = sentinel(tmp_path, run=FakeRun(sinks=NONE_AT_ALL, default="auto_null"),
                 say=say)
    reading = s.tick()
    assert reading["status"] == "dead"
    assert say.texts == [sb.DEAD_LINE]


def test_no_pactl_says_nothing_at_all(tmp_path):
    """A probe that cannot answer must never read as "there are no sinks"."""
    say = Speaker()

    def broken(argv, **kw):
        raise FileNotFoundError("pactl")

    s = sentinel(tmp_path, run=broken, say=say)
    reading = s.tick()
    assert reading["status"] == "blind"
    assert say.texts == []


def test_a_failing_pactl_return_code_is_blind_too(tmp_path):
    say = Speaker()
    s = sentinel(tmp_path, run=FakeRun(rc=1), say=say)
    assert s.tick()["status"] == "blind"
    assert say.texts == []


def test_switch_by_hand_between_two_present_sinks_is_not_our_business(tmp_path):
    say = Speaker()
    run = FakeRun(sinks=BOTH, default=BAR)
    s = sentinel(tmp_path, run=run, say=say, **{"audio.preferred_sink": "bluez"})
    s.tick()
    run.default = HDMI                          # he moved it himself
    reading = s.tick()
    assert reading["status"] == "elsewhere"
    assert say.texts == []


def test_the_watch_can_be_switched_off(tmp_path):
    run = FakeRun()
    s = sentinel(tmp_path, run=run, **{"audio.sink_watch": False})
    assert s.tick()["status"] == "off"
    assert run.calls == []
    s.start()
    assert s.running is False


# ------------------------------------------------------------ the answer
def test_status_line_answers_from_the_last_tick(tmp_path):
    run = FakeRun(default=BAR)
    s = sentinel(tmp_path, run=run)
    s.tick()
    before = len(run.calls)
    assert s.status_line() == ("I'm coming out of the soundbar, sir, "
                               "which is where I should be.")
    assert len(run.calls) == before             # no subprocess on the reply path


def test_status_line_says_where_the_voice_actually_is(tmp_path):
    run = FakeRun(default=BAR)
    s = sentinel(tmp_path, run=run)
    s.tick()
    run.sinks, run.default = HDMI_ONLY, HDMI
    s.tick()
    assert s.status_line() == ("I'm coming out of the monitor, sir; "
                               "the soundbar has dropped.")


def test_status_line_is_honest_without_pactl(tmp_path):
    def broken(argv, **kw):
        raise FileNotFoundError("pactl")

    s = sentinel(tmp_path, run=broken)
    assert s.status_line() == sb.BLIND_LINE


def test_start_and_stop_are_joinable(tmp_path):
    s = sentinel(tmp_path)
    s.start()
    assert s.running is True
    s.stop()
    assert s.running is False


def test_a_corrupt_state_file_is_survived(tmp_path):
    (tmp_path / "soundbar.json").write_text("{not json")
    s = sentinel(tmp_path, run=FakeRun(default=BAR))
    assert s.tick()["status"] == "ok"


@pytest.mark.parametrize("line", sb.__dict__["PERSONA_LINES"])
def test_persona_lines_address_him_as_sir(line):
    assert "sir" in line


# ------------------------------------------------------------ the command
@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from jarvis.commander import Commander, IntentClassifier
    from jarvis.config import CONFIG

    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "il.json")
    monkeypatch.setattr(CONFIG, "talkback", False)
    s = sentinel(tmp_path, run=FakeRun(default=BAR))
    s.tick()
    return Commander(SimpleNamespace(tts=MagicMock(), soundbar=s))


@pytest.mark.parametrize("said", [
    "where's your voice coming out", "where is my audio going",
    "which speaker are you on", "what output are you using",
    "where's the sound coming from"])
def test_the_question_reaches_the_sentinel(cmdr, said):
    from jarvis.commander import _AUDIO_OUT_RX, _h_audio_out

    assert _AUDIO_OUT_RX.match(said), said
    res = _h_audio_out(cmdr, said, None)
    assert res.handled and res.speak
    assert res.reply == ("I'm coming out of the soundbar, sir, "
                         "which is where I should be.")


def test_without_a_sentinel_the_question_falls_through(tmp_path, monkeypatch):
    """No sentinel means nobody probed: the honest answer is the model's
    shrug, not a confident sentence about a speaker."""
    from unittest.mock import MagicMock

    from jarvis.commander import Commander, IntentClassifier, _h_audio_out

    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "il2.json")
    c = Commander(SimpleNamespace(tts=MagicMock()))
    assert _h_audio_out(c, "where's your voice coming out", None) is None


def test_the_question_never_forks_a_process_mid_turn(cmdr):
    """aside.py's rule: nothing on the spoken path may block on a
    subprocess. The answer comes from the watcher's last tick."""
    from jarvis.commander import _h_audio_out

    run = cmdr.services.soundbar._run
    before = len(run.calls)
    _h_audio_out(cmdr, "where's your voice coming out", None)
    assert len(run.calls) == before
