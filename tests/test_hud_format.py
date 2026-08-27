"""Pure formatting helpers behind the clean HUD (no display needed)."""
from jarvis.ui.main_window import fmt_asr, fmt_mem_gb, resolve_state
from jarvis.ui.reactor import fmt_llm
from jarvis.ui.views import CommandBar, fmt_temps, split_temps


def test_fmt_temps():
    assert fmt_temps("cpu 40° 1%") == "40°C · 1%"
    assert fmt_temps("gpu 38° 5%") == "38°C · 5%"
    assert fmt_temps("gpu 38°") == "38°C"
    assert fmt_temps("") == "--"
    assert fmt_temps("garbage") == "--"
    assert fmt_temps(None) == "--"


def test_split_temps():
    segs = split_temps("cpu 40° 1% · gpu 38° 5%")
    assert segs == {"cpu": "cpu 40° 1%", "gpu": "gpu 38° 5%"}
    assert split_temps("") == {}


def test_fmt_mem_gb():
    total_kb = 128 * 1048576
    avail_kb = total_kb - int(26.8 * 1048576)
    assert fmt_mem_gb(total_kb, avail_kb) == "26.8 GB"
    assert fmt_mem_gb(0, 0) == "--"
    assert fmt_mem_gb("x", 1) == "--"


def test_fmt_llm_strips_only_latest():
    assert fmt_llm("llama3.2:latest") == "LLAMA3.2"
    assert fmt_llm("qwen2.5:32b") == "QWEN2.5:32B"
    assert fmt_llm("LLM llama3.2:latest") == "LLAMA3.2"
    assert fmt_llm("gemma4:26b") == "GEMMA4:26B"        # the assistant model
    assert fmt_llm("") == "--"
    assert fmt_llm(None) == "--"


def test_fmt_asr_drops_backend_suffix():
    assert fmt_asr("small · GPU fp16") == "WHISPER SMALL"
    assert fmt_asr("small") == "WHISPER SMALL"


def test_state_precedence_matches_reactor():
    assert resolve_state(True, True, True, True) == "speaking"
    assert resolve_state(False, True, True, True) == "listening"
    assert resolve_state(False, False, True, True) == "thinking"
    assert resolve_state(False, False, False, True) == "error"
    assert resolve_state(False, False, False, False) == "idle"
    # Claude-task states sit between thinking and error (test_ui_assistant)
    assert resolve_state(False, False, False, True, working=True) == "working"


def test_placeholders_are_truthful_and_typographic():
    assert CommandBar.PLACEHOLDER == "Type a command"
    assert CommandBar.PLACEHOLDER_HOT == "Type a command — or say “Jarvis”"


def test_engine_card_rows_are_hear_speak_think_device():
    from jarvis.ui.reactor import CARD_ROWS
    assert [lab for lab, _key in CARD_ROWS] == ["HEAR", "SPEAK", "THINK",
                                                "DEVICE"]
    assert [key for _lab, key in CARD_ROWS] == ["asr", "tts", "llm", "dev"]


def test_you_card_width_shrink_wraps_between_35_and_70_percent():
    from jarvis.ui.views import you_card_width
    usable, pad = 1000, 24
    assert you_card_width(0, usable, pad) == 350          # floor 35%
    assert you_card_width(200, usable, pad) == 350        # still floor
    assert you_card_width(400, usable, pad) == 400 + 2 * pad + 4
    assert you_card_width(5000, usable, pad) == 700       # cap 70%
