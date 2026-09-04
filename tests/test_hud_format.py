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


def test_fmt_mem_gb_says_used_of_total_so_nobody_has_to_ask():
    """2026-09-02, verbatim: "is that free or used?". It was USED, and
    nothing on the strip said so -- at the compact elision level the
    segment loses its MEM label and showed a bare '59.9 GB' on a 122 GB
    box. The value now carries the answer at EVERY level."""
    total_kb = 128 * 1048576
    avail_kb = total_kb - int(26.8 * 1048576)
    assert fmt_mem_gb(total_kb, avail_kb) == "26.8/128 GB"
    # the left number is the used half, and it moves
    assert fmt_mem_gb(total_kb, total_kb) == "0.0/128 GB"
    assert fmt_mem_gb(total_kb, 0) == "128.0/128 GB"
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


def test_engine_card_rows_are_hear_speak_think_device_fault():
    # FAULT joined the card in the fault-lane change (jarvis/faults.py): a
    # warn Status holds its pill for 4 s and an error for 6 s, so a fault
    # raised while the room was empty left no trace on the board at all.
    from jarvis.ui.reactor import CARD_ROWS
    assert [lab for lab, _key in CARD_ROWS] == ["HEAR", "SPEAK", "THINK",
                                                "DEVICE", "FAULT"]
    assert [key for _lab, key in CARD_ROWS] == ["asr", "tts", "llm", "dev",
                                                "fault"]


def test_every_card_row_has_a_value_and_a_source():
    """A row is three couplings, not one tuple: CARD_ROWS, the values dict
    in Reactor._apply_telemetry, and MainWindow._telemetry. Miss either of
    the last two and the row renders a permanent "--" that looks like a
    healthy reading -- which is why the literal list above is not enough."""
    import inspect
    from jarvis.ui.reactor import CARD_ROWS, Reactor
    from jarvis.ui.main_window import MainWindow
    apply_src = inspect.getsource(Reactor._apply_telemetry)
    telem_src = inspect.getsource(MainWindow._telemetry)
    for _lab, key in CARD_ROWS:
        assert f'"{key}"' in apply_src, f"{key} missing from _apply_telemetry"
        assert f'"{key}"' in telem_src, f"{key} missing from _telemetry"


def test_you_card_width_shrink_wraps_between_35_and_70_percent():
    from jarvis.ui.views import you_card_width
    usable, pad = 1000, 24
    assert you_card_width(0, usable, pad) == 350          # floor 35%
    assert you_card_width(200, usable, pad) == 350        # still floor
    assert you_card_width(400, usable, pad) == 400 + 2 * pad + 4
    assert you_card_width(5000, usable, pad) == 700       # cap 70%
