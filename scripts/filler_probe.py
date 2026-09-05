#!/usr/bin/env python3
"""Does "um" actually buy him time? -- the filler hold measured on a real
microphone, in numbers only.

    ~/vss_env/bin/python scripts/filler_probe.py
    ~/vss_env/bin/python scripts/filler_probe.py --takes 6 --show
    ~/vss_env/bin/python scripts/filler_probe.py --no-speculate

THIS SCRIPT IS HIS TO RUN. It is the one place the filler hold meets a
microphone, and it opens the mic ONLY when he runs it from a shell and
presses Enter for a take; the test suite drives its logic with fakes and
never constructs the real recorder. It never saves audio: JARVIS_DEBUG_AUDIO
is cleared before the first capture, nothing here writes a file, and the
transcript's WORDS are printed only under --show -- by default every line
is a word from the filler list, a capture second, a gap, a count or a
length. Its log lines go to /tmp/jarvis-adhoc (jarvis/logs.py routes any
process that is not the app there), never to the live jarvis.log.

WHY. jarvis/recorder.py stops a capture 0.8 s after the VAD last heard
speech (CONFIG.endpoint_silence). The filler hold waits filler_hold_s
longer when the live preview's newest decode ended on "um"/"uh" as the
last thing heard -- but the preview re-decodes every 0.9 s, the endpoint is
0.8 s, and the app's speculative pass parks the preview for a full decode
from 0.3 s into every pause. So whether the um is DECODED before the stop
is due is a race, and only a microphone can say how often it is won. Two
things are unmeasured until this runs:

  * the catch rate: of the takes where he said an um, how many had the
    preview end on it (holds fired) versus the tail never being decoded;
  * CONFIG.filler_prompt_hint: whether adding "Um, uh, hmm, er." to the
    PREVIEW's prompt makes whisper write the um down more often. It ships
    OFF; the takes here alternate it off/on so the two can be compared.

WHAT IT BUILDS. The app's own objects, the way jarvis/app.py builds them
(Recorder + MicArbiter, Transcriber with jarvis.vocab.build_prompt,
VoiceEndpointer warmed and installed) -- no UI, no hotword, no speaker
gate, no turn ledger. Its take loop is the greedy preview from
JarvisApp._partial_loop (same cadence, same minimum span, the same
note_partial seam), and it mirrors _maybe_speculate: one full decode per
pause 0.3 s in, holding the model as live does, so the timing measured is
the live timing. --no-speculate switches that mirror off to measure how
much the parking costs. Not mirrored: the speaker filter inside the app's
speculative pass and the voice-ID silence stop (neither moves the VAD's
timing). CONFIG is changed in memory only (sound off, the hint per take)
and never written to disk.

RUN IT WITH JARVIS STOPPED, or at least without saying the wake word: a
second process on the same microphone is fine for PipeWire, but a live
Jarvis would answer whatever you say to this. It loads its own whisper.

WHAT TO PASTE BACK: the four summary lines at the end.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from jarvis.config import CONFIG
from jarvis.endpoint import trailing_filler
from jarvis.events import RecordingStopped, bus
from jarvis.recorder import FILLER_SLACK_S, SAMPLE_RATE

# The preview's own numbers (JarvisApp._PARTIAL_*, _SPECULATE_AFTER_S),
# repeated here so importing this file does not import the whole app.
CADENCE_S = 0.9            # re-decode cadence while speaking
MIN_S = 0.7                # below this whisper mostly invents words
MAX_SPAN_S = 20.0          # decode only the newest span
SPECULATE_AFTER_S = 0.3    # VAD silence before the speculative full decode
POLL_S = 0.05
MAX_TAKE_S = 20.0          # a take that runs this long is stopped by hand
STOP_WAIT_S = 5.0          # how long to wait for the poll thread's stop event

INSTRUCTION = ("Say a command with an um in the middle, e.g. "
               "'set a timer for, um, ten minutes'. Pause on the um the way "
               "you would when thinking.")


@dataclass
class Pipeline:
    recorder: object
    transcriber: object
    endpointer: object = None
    backend: str = ""


@dataclass
class Take:
    """One capture, as numbers. ``transcript`` is kept for --show only and
    never reaches the summary."""
    n: int
    hint: bool
    fillers: list = field(default_factory=list)       # (word, capture second)
    partial_ends: list = field(default_factory=list)  # where each decode's span ended
    last_speech_s: Optional[float] = None
    gap_s: Optional[float] = None                     # the silence that ended it
    endpoint: str = "?"                               # vad | energy | cap | manual
    holds: int = 0
    cut: Optional[bool] = None                        # his answer; None = no answer
    transcript_len: int = 0
    transcript: str = ""
    spec_passes: int = 0

    @property
    def tail_decoded(self) -> Optional[bool]:
        """Did the newest decode reach the last speech (within the
        recorder's slack)? False is the limit the probe exists to count: the
        um was said after the last snapshot and nothing ever decoded it."""
        if not self.partial_ends or self.last_speech_s is None:
            return None
        return self.partial_ends[-1] >= self.last_speech_s - FILLER_SLACK_S


def forbid_audio_dumps() -> bool:
    """Clear JARVIS_DEBUG_AUDIO so the recorder cannot keep a capture on
    disk. Returns True when it was set."""
    return os.environ.pop("JARVIS_DEBUG_AUDIO", None) is not None


def plan(takes: int) -> list:
    """Hint OFF / ON alternating, so neither order effect nor his warming
    up lands on one side only."""
    return [bool(i % 2) for i in range(max(0, int(takes)))]


def build_pipeline() -> Pipeline:
    """The app's objects, built the way jarvis/app.py builds them
    (JarvisApp.__init__ for the recorder and transcriber, _load_models for
    the whisper load, _install_endpointer for the VAD). Opens NOTHING: the
    mic opens in run_take, when he presses Enter."""
    from jarvis import vocab as vocab_mod
    from jarvis.endpoint import MODEL_FILE, VoiceEndpointer
    from jarvis.recorder import MicArbiter, Recorder
    from jarvis.transcriber import Transcriber

    if not CONFIG.endpoint_vad:
        raise SystemExit("endpoint_vad is off in settings; the filler hold needs the VAD")
    CONFIG.sound = False                    # no chime from a probe (memory only)
    recorder = Recorder(MicArbiter())
    if not recorder.mic_available:
        raise SystemExit("no microphone detected")
    transcriber = Transcriber(prompt_provider=vocab_mod.build_prompt)
    backend = transcriber.load()
    endpointer = VoiceEndpointer()
    if not endpointer.warm():
        raise SystemExit("silero VAD did not load from %s" % MODEL_FILE)
    recorder.endpointer = endpointer
    return Pipeline(recorder=recorder, transcriber=transcriber,
                    endpointer=endpointer, backend=backend)


def _speculate_due(recorder, done_key) -> bool:
    """_maybe_speculate's trigger: SPECULATE_AFTER_S of VAD silence since a
    last-speech position that has not been decoded yet."""
    ep = getattr(recorder, "endpointer", None)
    if ep is None:
        return False
    try:
        gap, key = ep.silence_since_speech, ep.last_speech_seconds
    except Exception:
        return False
    return gap is not None and key is not None and gap >= SPECULATE_AFTER_S \
        and key != done_key


def run_take(pipe: Pipeline, n: int, hint: bool, ask: Callable, say: Callable,
             show: bool = False, speculate: bool = True,
             cadence_s: float = CADENCE_S, min_s: float = MIN_S,
             max_take_s: float = MAX_TAKE_S,
             clock: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] = time.sleep) -> Optional[Take]:
    """One capture: open the mic on Enter, run the preview loop until the
    recorder stops itself (or max_take_s), decode the final clip, ask him
    whether he was cut off, and print the numbers. None when the mic did
    not open."""
    recorder, transcriber = pipe.recorder, pipe.transcriber
    stopped: list = []
    handler = bus.subscribe(RecordingStopped, stopped.append)
    hint_before = CONFIG.filler_prompt_hint
    CONFIG.filler_prompt_hint = bool(hint)
    partials: list = []                    # (end_s, filler, text) -- text for --show only
    spec_passes, spec_key = 0, None
    try:
        say("--- take %d: hint %s ---" % (n, "ON" if hint else "OFF"))
        say(INSTRUCTION)
        ask("Press Enter to open the mic, then speak: ")
        recorder.start()
        if not recorder.recording:
            say("take %d: the mic did not open (no device, or the recorder refused); skipped" % n)
            return None
        t0 = clock()
        due = 0.0
        while recorder.recording:
            now = clock()
            if now - t0 >= max_take_s:
                say("take %d: %.0f s reached; stopping the capture" % (n, max_take_s))
                recorder.stop(reason="manual")
                break
            if speculate and _speculate_due(recorder, spec_key):
                spec_key = recorder.endpointer.last_speech_seconds
                clip = recorder.snapshot_final()
                if clip is not None:
                    try:
                        transcriber.transcribe(clip)      # holds the model, as live
                    except Exception:
                        pass
                    spec_passes += 1
                due = clock() + cadence_s                 # the preview rests, as live
                continue
            if now < due:
                sleep(POLL_S)
                continue
            started = clock()
            audio = recorder.snapshot_audio()
            if audio is not None and len(audio) >= int(SAMPLE_RATE * min_s):
                end_s = len(audio) / SAMPLE_RATE           # the buffer's END
                audio = audio[-int(SAMPLE_RATE * MAX_SPAN_S):]
                try:
                    text = (transcriber.partial(audio) or "").strip()
                except Exception:
                    text = ""
                recorder.note_partial(text, end_s)
                partials.append((end_s, trailing_filler(text), text))
            due = clock() + max(0.05, cadence_s - (clock() - started))
        deadline = clock() + STOP_WAIT_S
        while not stopped and clock() < deadline:          # the poll thread publishes it
            sleep(POLL_S)
    finally:
        CONFIG.filler_prompt_hint = hint_before
        bus.unsubscribe(RecordingStopped, handler)

    ev = stopped[-1] if stopped else None
    text = ""
    clip = getattr(recorder, "last_audio", None)
    if clip is not None:
        try:
            text = (transcriber.transcribe(clip).text or "").strip()
        except Exception as exc:                          # noqa: BLE001 - reported, not fatal
            say("take %d: final decode failed: %s" % (n, exc))
    ep = getattr(recorder, "endpointer", None)
    answer = (ask("Were you cut off before your last word? [y/n] ") or "").strip().lower()
    take = Take(
        n=n, hint=bool(hint),
        fillers=[(f, round(e, 2)) for e, f, _ in partials if f],
        partial_ends=[round(e, 2) for e, _, _ in partials],
        last_speech_s=getattr(ep, "last_speech_seconds", None),
        gap_s=getattr(ev, "dead_air_s", None) if ev is not None else None,
        endpoint=(ev.endpoint or ev.reason) if ev is not None else "?",
        holds=int(getattr(ev, "filler_holds", 0) or 0) if ev is not None else 0,
        cut=answer.startswith("y") if answer else None,
        transcript_len=len(text), transcript=text, spec_passes=spec_passes)
    for line in describe(take, show=show):
        say(line)
    return take


def _s(value) -> str:
    return "-" if value is None else "%.1fs" % value


def describe(take: Take, show: bool = False) -> list:
    """The per-take lines: numbers, and the words only under --show."""
    fillers = ", ".join("%s@%.1fs" % (w, s) for w, s in take.fillers) or "none"
    tail = {True: "yes", False: "NO", None: "-"}[take.tail_decoded]
    cut = {True: "y", False: "n", None: "-"}[take.cut]
    lines = ["take %d (hint %s): fillers in preview: %s | last speech %s | "
             "stop %s after %s gap | holds %d | tail decoded %s | cut off: %s | "
             "transcript %d chars | speculative passes %d | decodes %d"
             % (take.n, "ON" if take.hint else "OFF", fillers, _s(take.last_speech_s),
                take.endpoint, _s(take.gap_s), take.holds, tail, cut,
                take.transcript_len, take.spec_passes, len(take.partial_ends))]
    if show:
        lines.append("  transcript: %s" % take.transcript)
    return lines


def summarize(takes: list, speculate: bool = True) -> list:
    """Four lines he can paste back."""
    off = [t for t in takes if not t.hint]
    on = [t for t in takes if t.hint]

    def row(label, group):
        n = len(group)
        seen = sum(1 for t in group if t.fillers)
        holds = sum(t.holds for t in group)
        cut = sum(1 for t in group if t.cut)
        undecoded = sum(1 for t in group if t.tail_decoded is False)
        return ("%s: um seen in preview %d/%d takes, holds %d, cut off %d/%d, "
                "tail undecoded %d/%d" % (label, seen, n, holds, cut, n, undecoded, n))

    stops = Counter(t.endpoint for t in takes)
    vad, energy = stops.pop("vad", 0), stops.pop("energy", 0)
    other = sum(stops.values())
    gaps = [t.gap_s for t in takes if t.gap_s is not None]
    mean_gap = "%.1f s" % (sum(gaps) / len(gaps)) if gaps else "-"
    lens = [t.transcript_len for t in takes]
    mean_len = "%.0f chars" % (sum(lens) / len(lens)) if lens else "-"
    return [
        "filler-probe: %d takes (%d hint off, %d hint on), speculative pass mirrored: %s"
        % (len(takes), len(off), len(on), "yes" if speculate else "no"),
        row("hint off", off),
        row("hint on ", on),
        "stops: vad %d, energy %d, other %d; mean gap %s; mean transcript %s"
        % (vad, energy, other, mean_gap, mean_len),
    ]


def main(argv=None, build: Optional[Callable[[], Pipeline]] = None,
         ask: Callable = input, say: Callable = print,
         clock: Callable[[], float] = time.monotonic,
         sleep: Callable[[float], None] = time.sleep) -> int:
    ap = argparse.ArgumentParser(
        description="The filler hold, measured on the mic in numbers only.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--takes", type=int, default=4,
                    help="captures to run, hint off/on alternating (default 4)")
    ap.add_argument("--show", action="store_true",
                    help="print each take's transcript words (default: length only)")
    ap.add_argument("--no-speculate", action="store_true",
                    help="do not mirror the app's speculative pass (measures its cost)")
    ap.add_argument("--max-take-s", type=float, default=MAX_TAKE_S,
                    help="stop a capture by hand after this long (default %.0f)" % MAX_TAKE_S)
    args = ap.parse_args(argv)

    if forbid_audio_dumps():
        say("JARVIS_DEBUG_AUDIO was set; cleared for this run (the probe never saves audio)")
    pipe = (build or build_pipeline)()
    say("hold: filler_hold=%s filler_hold_s=%s filler_max_holds=%s | endpoint_silence=%s "
        "endpoint_vad=%s | whisper %s"
        % (CONFIG.filler_hold, CONFIG.filler_hold_s, CONFIG.filler_max_holds,
           CONFIG.endpoint_silence, CONFIG.endpoint_vad,
           getattr(pipe, "backend", "") or "?"))
    speculate = not args.no_speculate
    takes: list = []
    try:
        for i, hint in enumerate(plan(args.takes), 1):
            take = run_take(pipe, n=i, hint=hint, ask=ask, say=say, show=args.show,
                            speculate=speculate, max_take_s=args.max_take_s,
                            clock=clock, sleep=sleep)
            if take is not None:
                takes.append(take)
    except KeyboardInterrupt:
        say("interrupted; summarising the takes that finished")
        try:
            pipe.recorder.stop(reason="manual")
        except Exception:                                  # noqa: BLE001
            pass
    say("")
    for line in summarize(takes, speculate=speculate):
        say(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
