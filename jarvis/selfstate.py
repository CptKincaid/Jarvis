"""One self-state sheet, three registers.

``JarvisApp.self_state()`` gathers the facts (uptime, the three models, the
turn ledger, memory and the GPU counters, the local model's residency, the
sink his voice actually leaves by, his own Claude panes, the voiceprint,
anything quiet hours is holding, and the sticky modes he left open).
Everything here RENDERS that one dict, so the readout and the courtesy can never drift apart:

* ``diagnostics_line``  the plain sheet -- the card, the cmdsock "status"
  answer, ``ask.py --status`` and the formal register;
* ``stark_line``        the same probes in film register, spoken;
* ``wellbeing_line``    one honest clause for "how are you?";
* ``availability_line`` "are you busy?" answered from what he is doing, or
  None to fall back to the templated courtesy.

Nothing here calls a model. The old "All systems nominal, sir." is gone
from both renderings: jarvis/brain.py's VOICE_RULES forbid the phrase
"all systems", and app.py hardcoded it anyway.

Pure functions over a dict: no Tk, no subprocess, no clock.
"""
from __future__ import annotations

# The GPU wedge of 2026-08-28 sat at 611 MHz against a healthy 2400; power
# draw is NOT the tell (idle reads ~15 W in both states), the clock is.
THROTTLE_MHZ = 1200
# Said when neither /proc/meminfo nor nvidia-smi would answer -- the
# UNREADABLE_LINE discipline of jarvis/tools/health.py: name what could not
# be read rather than quietly dropping the sentence.
COUNTERS_LINE = "I'm afraid I can't read the machine's counters, sir."
OPENING_LINE = "Everything's where I left it, sir."

# Fixed, numberless lines worth prewarming into the speech cache: a
# state-derived sentence is an F5 cache miss (~0.5-1 s) on the fastest
# exchange in the system, so the variants that carry no figure are baked.
WELLBEING_PLAIN = "Very well, sir."
WELLBEING_LENT = "Running on reflexes, sir; my model is with your trainer."
WELLBEING_HELD = "Well enough, sir; I'm holding a few things until you're free."
WELLBEING_DEAF = "Well, sir, though you've never enrolled my ears."
WELLBEING_THROTTLED = "A little sluggish, sir; the GPU is throttled."
WELLBEING_QUIET = "Quite well, sir, and keeping my voice down."
WELLBEING_IDLE = "Very well, sir; nothing amiss."
AVAILABILITY_BUSY = "A little occupied, sir, but never for you."
PREWARM_LINES = (WELLBEING_PLAIN, WELLBEING_LENT, WELLBEING_HELD,
                 WELLBEING_DEAF, WELLBEING_THROTTLED, WELLBEING_QUIET,
                 WELLBEING_IDLE, AVAILABILITY_BUSY, COUNTERS_LINE)

# pactl sink names are paths in all but spelling, and VOICE_RULES bans
# reciting one. Say the KIND of output or say nothing.
_SINK_WORDS = (
    ("bluez", "a Bluetooth speaker"),
    ("bluetooth", "a Bluetooth speaker"),
    ("hdmi", "the HDMI output"),
    ("usb", "the USB audio device"),
    ("analog", "the analogue output"),
    ("iec958", "the digital output"),
    ("headphone", "your headphones"),
)


def _plural(n, word) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def uptime_words(seconds) -> str:
    """"3 hours and 12 minutes" / "7 minutes"."""
    try:
        total = max(0, int(seconds or 0))
    except (TypeError, ValueError):
        return ""
    hours, mins = total // 3600, (total % 3600) // 60
    if hours:
        return f"{_plural(hours, 'hour')} and {_plural(mins, 'minute')}"
    return _plural(mins, "minute")


def sink_words(name) -> str:
    """The kind of output a sink is, or "" when it is not worth saying."""
    low = str(name or "").lower()
    if not low:
        return ""
    for needle, words in _SINK_WORDS:
        if needle in low:
            return words
    return ""


def gpu_words(state) -> str:
    """"54 degrees, 2424 megahertz, 2 percent busy" from whatever fields
    the parser actually returned ("" when it returned none of them)."""
    bits = []
    if state.get("gpu_temp_c") is not None:
        bits.append(f"{state['gpu_temp_c']:.0f} degrees")
    if state.get("gpu_mhz") is not None:
        bits.append(f"{state['gpu_mhz']:.0f} megahertz")
    if state.get("gpu_util_pct") is not None:
        bits.append(f"{state['gpu_util_pct']:.0f} percent busy")
    return ", ".join(bits)


def is_throttled(state) -> bool:
    mhz = state.get("gpu_mhz")
    return mhz is not None and mhz < THROTTLE_MHZ


def _panes_words(state) -> str:
    panes = int(state.get("claude_panes") or 0)
    if panes <= 0:
        return ""
    working = int(state.get("claude_working") or 0)
    words = f"{_plural(panes, 'Claude terminal')} of your own"
    return f"{words}, {working} mid-turn" if working else words


# --------------------------------------------------------------- plain
def diagnostics_line(state) -> str:
    """The plain sheet: the card, "status" on the socket, and the answer
    the formal register gets."""
    if not state:
        return COUNTERS_LINE
    parts = []
    head = OPENING_LINE
    up = uptime_words(state.get("uptime_s"))
    if up:
        head += f" Up {up};"
    else:
        head += " Running;"
    head += (f" whisper {state.get('stt_model', '?')} on the GPU, "
             f"{state.get('brain_model', '?')} answering, "
             f"{state.get('tts_engine', '?')} speaking")
    if state.get("endpointing"):
        head += ", voice-activity endpointing live"
    parts.append(head + ".")

    turns = int(state.get("turns_today") or 0)
    if turns:
        wait = state.get("median_wait")
        med = f", median wait {wait:.1f} seconds" if wait is not None else ""
        parts.append(f"{_plural(turns, 'turn')} today{med}.")

    free, total = state.get("mem_free_gb"), state.get("mem_total_gb")
    gpu = gpu_words(state)
    if free is not None and total is not None:
        line = f"Memory {free:.0f} of {total:.0f} gigabytes free"
        parts.append(f"{line}, GPU at {gpu}." if gpu else line + ".")
    elif gpu:
        parts.append(f"GPU at {gpu}.")
    else:
        parts.append(COUNTERS_LINE)

    parts.append("My model is lent to your trainer." if state.get("lent")
                 else "My model is resident.")
    sink = sink_words(state.get("sink"))
    if state.get("sink_dummy"):
        parts.append("Your voice has nowhere to go: the only sink is a dummy.")
    elif sink:
        parts.append(f"Your voice is going out through {sink}.")
    panes = _panes_words(state)
    if panes:
        parts.append(f"{panes}.")
    if state.get("enrolled"):
        parts.append(f"Your voiceprint holds "
                     f"{_plural(int(state.get('num_samples') or 0), 'sample')}.")
    # The sticky modes (lecture notes, dictation, an open quiz) are already
    # a finished sentence from app.open_modes_line().
    if state.get("modes"):
        parts.append(str(state["modes"]))
    return " ".join(parts)


# ---------------------------------------------------------------- film
def stark_line(state) -> str:
    """The same sheet in the register of the films -- hand-written, never
    handed to the model: "never invent a figure" is a prompt, not a
    guarantee, and every figure here is a real reading."""
    if not state:
        return COUNTERS_LINE
    parts = []
    if state.get("lent"):
        parts.append("The local model is lent out to your trainer, sir; "
                     "I'm on reflexes until you take it back.")
    else:
        parts.append("Power to the local model at full, sir.")
    gpu = gpu_words(state)
    if gpu and is_throttled(state):
        parts.append(f"The GPU is dragging its feet at {gpu}.")
    elif gpu:
        parts.append(f"The GPU is running at {gpu}.")
    panes = int(state.get("claude_panes") or 0)
    if panes > 0:
        working = int(state.get("claude_working") or 0)
        board = f"{_plural(panes, 'terminal')} of yours on the board"
        parts.append(f"{board}, {working} mid-turn." if working else board + ".")
    turns = int(state.get("turns_today") or 0)
    up = uptime_words(state.get("uptime_s"))
    if turns and up:
        parts.append(f"{_plural(turns, 'turn')} today, and I've been up {up}.")
    elif up:
        parts.append(f"Up {up}, and nothing asked of me yet.")
    if state.get("sink_dummy"):
        parts.append("You should know my voice has nowhere to go: "
                     "the only sink is a dummy.")
    if state.get("modes"):
        parts.append(str(state["modes"]))
    if state.get("mem_free_gb") is None and not gpu:
        parts.append(COUNTERS_LINE)
    return " ".join(parts)


# ------------------------------------------------------------ courtesy
def wellbeing_line(state, register="normal") -> str:
    """"How are you?" / "how do you feel?" -- one clause, from real state.

    The formal register answers with the templated line and pays no cache
    miss; the others take at most one figure.
    """
    if register == "formal" or not state:
        return WELLBEING_PLAIN
    if state.get("lent"):
        return WELLBEING_LENT
    if is_throttled(state):
        return WELLBEING_THROTTLED
    if int(state.get("held") or 0) > 0:
        return WELLBEING_HELD
    if state.get("quiet_reason"):
        return WELLBEING_QUIET
    if state.get("enrolled") is False:
        return WELLBEING_DEAF
    turns = int(state.get("turns_today") or 0)
    if turns >= 2:
        return f"Very well, sir; {_plural(turns, 'turn')} today and nothing amiss."
    return WELLBEING_IDLE


def availability_line(state):
    """"Are you busy?" from what he is actually doing, or None to let the
    templated courtesy answer (which is prewarmed and instant)."""
    if not state:
        return None
    if state.get("lent") or int(state.get("held") or 0) > 0 or \
            int(state.get("claude_working") or 0) > 0:
        return AVAILABILITY_BUSY
    return None
