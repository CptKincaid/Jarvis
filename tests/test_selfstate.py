"""A proper self: "how are you" and "run diagnostics" off ONE state sheet.

jarvis/selfstate.py renders the dict JarvisApp.self_state() gathers, in
three registers -- the plain sheet (card, cmdsock "status", ask.py
--status, and the formal register), the film register that is spoken, and
one honest clause for the courtesy. Never two gatherers that can drift.

Pure functions over a dict: no Tk, no app, no subprocess.
"""
from types import SimpleNamespace

import pytest

from jarvis import selfstate
from jarvis.commander import (COURTESY_REPLIES, _h_diagnostics, _h_greeting,
                              greeting_kind)


def state(**over):
    base = {
        "uptime_s": 3 * 3600 + 12 * 60, "stt_model": "small",
        "brain_model": "gemma4", "tts_engine": "f5", "endpointing": True,
        "turns_today": 9, "median_wait": 1.32,
        "mem_free_gb": 84.0, "mem_total_gb": 119.0,
        "gpu_temp_c": 54.0, "gpu_mhz": 2424.0, "gpu_util_pct": 2.0,
        "lent": False, "enrolled": True, "num_samples": 14,
        "held": 0, "quiet_reason": "",
        "sink": "alsa_output.pci-0000_01_00.1.hdmi-stereo", "sink_dummy": False,
        "claude_panes": 4, "claude_working": 2,
    }
    base.update(over)
    return base


# ------------------------------------------------------------- the plain sheet
def test_the_plain_sheet_reads_every_probe():
    line = selfstate.diagnostics_line(state())
    for fragment in ("Up 3 hours and 12 minutes", "whisper small on the GPU",
                     "gemma4 answering", "f5 speaking",
                     "voice-activity endpointing live",
                     "9 turns today, median wait 1.3 seconds",
                     "Memory 84 of 119 gigabytes free",
                     "GPU at 54 degrees, 2424 megahertz, 2 percent busy",
                     "My model is resident",
                     "Your voice is going out through the HDMI output",
                     "4 Claude terminals of your own, 2 mid-turn",
                     "Your voiceprint holds 14 samples"):
        assert fragment in line, fragment


def test_the_toy_line_is_gone_from_both_renderings_and_from_the_courtesy():
    """brain.VOICE_RULES bans the phrase "all systems"; app.py hardcoded it
    anyway, and COURTESY_REPLIES opened with it. All three, one change."""
    for text in (selfstate.diagnostics_line(state()),
                 selfstate.stark_line(state()),
                 selfstate.wellbeing_line(state()),
                 " ".join(COURTESY_REPLIES["wellbeing"])):
        assert "all systems" not in text.lower(), text


def test_a_sink_name_is_never_read_out():
    """pactl sink names are paths in all but spelling, and VOICE_RULES bans
    reciting one: say the KIND of output or say nothing."""
    line = selfstate.diagnostics_line(state())
    assert "alsa_output" not in line and "pci-0000" not in line
    assert selfstate.sink_words("bluez_output.AC_BC_32.1") == "a Bluetooth speaker"
    assert selfstate.sink_words("alsa_output.usb-Focusrite.analog-stereo")
    assert selfstate.sink_words("weird_vendor_thing") == ""
    quiet = selfstate.diagnostics_line(state(sink="weird_vendor_thing"))
    assert "going out through" not in quiet


def test_a_dead_sink_is_said_out_loud():
    for render in (selfstate.diagnostics_line, selfstate.stark_line):
        assert "nowhere to go" in render(state(sink_dummy=True))


def test_missing_fields_are_omitted_not_guessed():
    bare = selfstate.diagnostics_line(state(
        gpu_temp_c=None, gpu_mhz=None, gpu_util_pct=None, turns_today=0,
        median_wait=None, enrolled=False, num_samples=0, claude_panes=0,
        claude_working=0, sink="", endpointing=False))
    assert "GPU at" not in bare and "turns today" not in bare
    assert "voiceprint" not in bare and "endpointing" not in bare
    assert "Memory 84 of 119 gigabytes free." in bare
    # a partial nvidia-smi row renders only the fields it answered
    partial = selfstate.gpu_words(state(gpu_mhz=None, gpu_util_pct=None))
    assert partial == "54 degrees"


def test_unreadable_counters_are_admitted_rather_than_dropped():
    """The UNREADABLE_LINE discipline of jarvis/tools/health.py: name what
    could not be read. A silently shorter sheet reads as good news."""
    blind = state(mem_free_gb=None, mem_total_gb=None, gpu_temp_c=None,
                  gpu_mhz=None, gpu_util_pct=None)
    assert selfstate.COUNTERS_LINE in selfstate.diagnostics_line(blind)
    assert selfstate.COUNTERS_LINE in selfstate.stark_line(blind)
    assert selfstate.diagnostics_line({}) == selfstate.COUNTERS_LINE
    assert selfstate.stark_line(None) == selfstate.COUNTERS_LINE


def test_no_per_device_vram_figure_is_ever_implied():
    """The GB10 pool is unified and nvidia-smi reads memory.used/total as
    N/A there: MemAvailable is the only honest memory number."""
    for text in (selfstate.diagnostics_line(state()),
                 selfstate.stark_line(state())):
        assert "VRAM" not in text and "video memory" not in text
        assert "GPU memory" not in text


# ------------------------------------------------------------ film register
def test_the_film_register_speaks_the_clock_and_never_the_draw():
    """611 MHz against 2400 is a 4x hit he has been bitten by; idle power
    reads ~15 W in BOTH states, so the draw alone is not a wedge tell."""
    line = selfstate.stark_line(state())
    assert line.startswith("Power to the local model at full, sir.")
    assert "2424 megahertz" in line and "2 percent busy" in line
    assert "watt" not in line.lower() and "power draw" not in line.lower()
    assert "4 terminals of yours on the board, 2 mid-turn" in line
    assert "9 turns today" in line


def test_the_wedge_is_named_when_the_clock_is_down():
    wedged = state(gpu_mhz=611.0, gpu_util_pct=0.0)
    assert selfstate.is_throttled(wedged)
    assert "dragging its feet at 54 degrees, 611 megahertz" in \
        selfstate.stark_line(wedged)
    assert not selfstate.is_throttled(state())
    assert not selfstate.is_throttled(state(gpu_mhz=None))


def test_a_lent_model_is_said_first_in_both_registers():
    lent = state(lent=True)
    assert selfstate.stark_line(lent).startswith("The local model is lent out")
    assert "My model is lent to your trainer." in selfstate.diagnostics_line(lent)


def test_the_film_register_invents_no_figure():
    """Every number in it is a reading; nothing is routed through the model,
    where "never invent a figure" is a prompt and not a guarantee."""
    import re
    blind = state(gpu_temp_c=None, gpu_mhz=None, gpu_util_pct=None,
                  turns_today=0, claude_panes=0, claude_working=0,
                  mem_free_gb=None, mem_total_gb=None)
    numbers = re.findall(r"\d+", selfstate.stark_line(blind))
    assert numbers == ["3", "12"]          # the uptime, which IS a reading


# --------------------------------------------------------------- courtesy
def test_wellbeing_is_answered_from_state_not_from_a_list_of_three():
    assert selfstate.wellbeing_line(state()) == \
        "Very well, sir; 9 turns today and nothing amiss."
    assert selfstate.wellbeing_line(state(turns_today=0)) == \
        selfstate.WELLBEING_IDLE
    assert selfstate.wellbeing_line(state(lent=True)) == selfstate.WELLBEING_LENT
    assert selfstate.wellbeing_line(state(gpu_mhz=611.0)) == \
        selfstate.WELLBEING_THROTTLED
    assert selfstate.wellbeing_line(state(held=3)) == selfstate.WELLBEING_HELD
    assert selfstate.wellbeing_line(state(quiet_reason="you're out")) == \
        selfstate.WELLBEING_QUIET
    assert selfstate.wellbeing_line(state(enrolled=False)) == \
        selfstate.WELLBEING_DEAF


def test_the_formal_register_pays_no_cache_miss_for_a_courtesy():
    """A state-derived sentence is an F5 cache miss on the fastest exchange
    in the system; the formal answer is a prewarmed fixed string."""
    assert selfstate.wellbeing_line(state(), register="formal") == \
        selfstate.WELLBEING_PLAIN
    assert selfstate.wellbeing_line(None) == selfstate.WELLBEING_PLAIN


def test_every_numberless_variant_is_prewarmed():
    for line in (selfstate.WELLBEING_PLAIN, selfstate.WELLBEING_LENT,
                 selfstate.WELLBEING_HELD, selfstate.WELLBEING_DEAF,
                 selfstate.WELLBEING_THROTTLED, selfstate.WELLBEING_QUIET,
                 selfstate.WELLBEING_IDLE, selfstate.AVAILABILITY_BUSY):
        assert line in selfstate.PREWARM_LINES
        assert not any(ch.isdigit() for ch in line), line
        assert "{" not in line                        # never a template
        assert len(line.split()) <= 12, line


def test_availability_answers_from_what_he_is_doing_or_defers():
    assert selfstate.availability_line(state(claude_working=0)) is None
    assert selfstate.availability_line(None) is None
    assert selfstate.availability_line(state(lent=True)) == \
        selfstate.AVAILABILITY_BUSY
    assert selfstate.availability_line(state(held=2)) == \
        selfstate.AVAILABILITY_BUSY
    assert selfstate.availability_line(state(claude_working=1)) == \
        selfstate.AVAILABILITY_BUSY


# ------------------------------------------------------------- the handlers
@pytest.fixture
def svc():
    """A commander stand-in with the diagnostics and self_state services."""
    def make(**services):
        return SimpleNamespace(_svc=lambda name: services.get(name))
    return make


def test_how_do_you_feel_joined_the_wellbeing_family():
    for said in ("how are you", "how do you feel", "how are you feeling",
                 "jarvis, how do you feel?", "how are you holding up"):
        assert greeting_kind(said) == "wellbeing", said
    assert greeting_kind("how do you feel about the merge") is None


def test_the_greeting_handler_prefers_state_and_asks_only_for_the_cheap_half(svc):
    """full=False: the sink probe and the tmux pane count say nothing about
    his wellbeing and would put two subprocesses on the fastest exchange."""
    calls = []

    def self_state(full=True):
        calls.append(full)
        return state(turns_today=0)

    c = svc(self_state=self_state)
    res = _h_greeting(c, "how are you", "wellbeing")
    assert res.handled and res.speak and res.reply == selfstate.WELLBEING_IDLE
    assert calls == [False]


def test_the_greeting_handler_falls_back_when_there_is_no_sheet(svc):
    res = _h_greeting(svc(), "how are you", "wellbeing")
    assert res.reply in COURTESY_REPLIES["wellbeing"]
    res = _h_greeting(svc(), "are you busy", "availability")
    assert res.reply in COURTESY_REPLIES["availability"]
    # a plain hello never pays for a probe at all
    def boom(**_):
        raise AssertionError("the greeting asked for the state sheet")
    assert _h_greeting(svc(self_state=boom), "hello", "greeting").handled


def test_diagnostics_speaks_the_film_register_and_cards_the_plain_sheet(svc):
    from jarvis.events import JarvisReply, bus
    seen = []
    sub = bus.subscribe(JarvisReply, lambda ev: seen.append((ev.text, ev.speak)))
    try:
        c = svc(diagnostics=lambda: selfstate.diagnostics_line(state()),
                self_state=lambda full=True: state())
        res = _h_diagnostics(c, "run diagnostics", None)
    finally:
        bus.unsubscribe(JarvisReply, sub)
    assert res.handled and res.speak
    assert res.reply.startswith("Power to the local model at full, sir.")
    assert seen and seen[0][1] is False
    assert "Your voiceprint holds 14 samples" in seen[0][0]


def test_one_sheet_feeds_both_renderings(svc):
    """Never two gatherers that can drift: with a state dict in hand the
    plain `diagnostics` service is not asked a second time."""
    def boom():
        raise AssertionError("the sheet was gathered twice")
    res = _h_diagnostics(svc(diagnostics=boom, self_state=lambda **k: state()),
                         "run diagnostics", None)
    assert res.reply == selfstate.stark_line(state())


def test_a_failed_diagnostics_probe_still_answers(svc):
    def boom():
        raise OSError("no counters")
    res = _h_diagnostics(svc(diagnostics=boom), "diagnostics", None)
    assert res.handled and "didn't complete" in res.reply
    # an app that predates the sheet still gets its own plain line spoken
    old = _h_diagnostics(svc(diagnostics=lambda: "Up 2 minutes, sir."),
                         "diagnostics", None)
    assert old.reply == "Up 2 minutes, sir."
    assert _h_diagnostics(svc(), "diagnostics", None) is None


def test_the_sticky_modes_stay_on_the_sheet_in_both_registers():
    """"jarvis status" from a shell is the only way to notice an open
    lecture or quiz, so a mode must not fall out when the readout moved
    into the renderers."""
    modes = "Dictation mode on."
    assert modes in selfstate.diagnostics_line(state(modes=modes))
    assert modes in selfstate.stark_line(state(modes=modes))
    assert modes not in selfstate.diagnostics_line(state(modes=""))


def test_the_sheet_survives_an_app_that_is_only_half_built():
    """self_state() is read by "jarvis status" from a shell and by the
    tests off a bare object; a collaborator that isn't wired yet must cost
    one field, never the whole sheet."""
    from jarvis import app as app_mod
    a = object.__new__(app_mod.JarvisApp)
    a.commander = None
    sheet = a.self_state(full=True)
    # no brain, no quiet, no speaker, no claude, no tts, no recorder
    assert sheet["lent"] is False and sheet["held"] == 0
    assert sheet["enrolled"] is False and sheet["claude_panes"] == 0
    assert sheet["modes"] == "" and sheet["uptime_s"] >= 0
    # ... and it still renders in every register
    assert selfstate.diagnostics_line(sheet)
    assert selfstate.stark_line(sheet)
    assert selfstate.wellbeing_line(sheet)
