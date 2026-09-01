#!/usr/bin/env python3
"""What the transcript says vs what Jarvis actually says, over many turns.

HIS OWN REQUEST, 2026-08-31, after an evening of working through 155
features by voice:

    "Set up a way to catch what the transcript says vs what he actually
     says and compare the two and do a bunch of runs on that."

He asked for it because several of that evening's defects WERE that gap.
Two are in the log verbatim:

  * "Four on your shopping list, sir: milk, milk, eggs, and and bread."
    (20:56:05)
  * "Sir, your 30-second 30 seconds timer timer is up." (20:42:40)

and his note reports a third, the diagnostics turn, that the log CANNOT
confirm or deny -- which is the point. The only record of a spoken line was
``speaking (f5): %.60s``: sixty characters of the text BEFORE the
pronunciation pass rewrote it, and no record at all of what the card said.
There was nothing to compare.

WHAT THIS DOES
    A reply travels: authored text -> the transcript card, and the SAME
    text -> _clean_for_speech -> the pronunciation pass -> sentence chunks
    -> the engine. This runs both halves for a corpus of lines and diffs
    them, using jarvis.tts's own comparators (``speech_divergence``,
    ``stutters``) so the harness and the live path can never disagree.

    No GPU, no speaker, no microphone, no model: the text pipeline is pure.

CORPORA
    --corpus builtin    his real lines, transcribed from the 2026-08-31
                        session log and pasted in here because /tmp is
                        wiped at boot and jarvis.log goes with it.
    --corpus <path>     one line per reply.
    --log <jarvis.log>  harvest every reply Jarvis spoke in a real session.
                        NOTE: the log truncates spoken lines, so a finding
                        of "the tail is not spoken" from --log may be the
                        LOG's truncation and not the speech path's. Lines
                        at exactly the truncation width are skipped.

DRIVING REAL TURNS
    --live drives the RUNNING Jarvis over its command socket, one
    ``jarvis -q`` per utterance -- the text path only, so it never touches
    the wake word, the microphone or the Whisper gate, and -q means the
    room stays silent. His reply comes back on stdout; that is the
    transcript side, and the spoken side is computed here. Utterances that
    would CHANGE something (set a timer, add to a list, remember, delete)
    are refused unless --allow-mutating is passed: this points at his real
    assistant, with his real lists in it.

USAGE
    ~/vss_env/bin/python scripts/speech_diff.py
    ~/vss_env/bin/python scripts/speech_diff.py --log /tmp/vss_voice/jarvis.log
    ~/vss_env/bin/python scripts/speech_diff.py --live --runs 20
    ~/vss_env/bin/python scripts/speech_diff.py --engine edge --verbose

Exit code 1 when any BLOCKING finding is present (something the room never
hears, a stutter, silence); 0 otherwise. ADVISORY findings never fail it.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jarvis import tts as tts_mod                                # noqa: E402

# --------------------------------------------------------------- findings
# BLOCKING is the half he would notice from the sofa: words on the card the
# room never hears, a phrase said twice, a reply that renders as nothing.
# ADVISORY is the half that is usually the pipeline doing its job.
BLOCKING = ("silent", "dropped", "truncated", "stutter")
ADVISORY = ("added", "split", "unterminated")


@dataclass
class Finding:
    kind: str
    detail: str
    shown: str
    spoken: str = ""

    @property
    def blocking(self) -> bool:
        return self.kind in BLOCKING


@dataclass
class LineResult:
    shown: str
    spoken: str
    chunks: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


# A chunk that does not end on sentence punctuation is a sentence that was
# cut apart mid-clause, and each piece is then rendered as its own
# utterance. Measured on the live F5 sidecar (2026-08-31): the pitch resets
# +45 Hz at the cut and the sentence's declination is lost -- which is
# exactly "Jarvis gets monotone with long sentences or lists". See
# TTS._ENGINE_CHUNKING.
_ENDS_A_SENTENCE = re.compile(r'[.!?;:]["\')\]]*$')
_TERMINATOR = re.compile(r'[.!?](?=\s|$)')

# Anything that would write to his real store, his real calendar, his real
# mail or his real music. --live is pointed at the assistant he actually
# uses, so the default is: ask, never change.
#
# Matched on the LEADING verb, the way he actually gives an order ("Set a
# timer for 30 seconds", "Add milk to the shopping list", "Remember that
# it is Mara and I's anniversary"). A bare keyword search is too blunt --
# it refuses "What do you remember?", which changes nothing.
_MUTATING = re.compile(
    r"^\s*(?:jarvis[,\s]+)?(?:please\s+|could you\s+|can you\s+|go ahead"
    r"\s+and\s+|just\s+|ok\s+)*"
    r"(set|add|remind|remember|forget|delete|remove|cancel|clear|scratch"
    r"|cross|strike|take|put|play|pause|skip|queue|cue|start|stop|brighten"
    r"|dim|mute|open|close|send|reply|archive|write|create|note|move|rename"
    r"|turn|volume|lend|restart|shut)\b",
    re.I)


def check_line(tts, shown: str) -> LineResult:
    """One reply, both halves, diffed."""
    spoken = tts.spoken_form(shown)
    res = LineResult(shown=shown, spoken=spoken)
    if not spoken.strip():
        res.findings.append(Finding("silent", "nothing is spoken at all",
                                    shown, spoken))
        return res

    for note in tts_mod.speech_divergence(shown, spoken):
        if "not spoken" not in note:
            res.findings.append(Finding("added", note, shown, spoken))
        elif len(shown) > tts.MAX_SPEAK_LENGTH and note.startswith("the tail"):
            # The cleaner cuts at MAX_SPEAK_LENGTH and says nothing about
            # it; the card still shows the whole reply.
            res.findings.append(Finding(
                "truncated",
                f"cut at MAX_SPEAK_LENGTH={tts.MAX_SPEAK_LENGTH}; {note}",
                shown, spoken))
        else:
            res.findings.append(Finding("dropped", note, shown, spoken))

    for phrase in tts_mod.stutters(spoken):
        res.findings.append(Finding("stutter", f'said twice: "{phrase}"',
                                    shown, spoken))

    # A reply that ends on a full stop on the card but not in the speech has
    # had its sentence terminator eaten, and the engine then gives it no
    # terminal fall -- the other half of "monotone". Live example:
    # pronounce.speak_times turns "presentation at 4:10 pm." into
    # "presentation at four ten pee em", because its am/pm pattern ends
    # `m\.?` and swallows the sentence's own full stop.
    if _ENDS_A_SENTENCE.search(shown.strip()) and \
            not _ENDS_A_SENTENCE.search(spoken.strip()):
        res.findings.append(Finding(
            "unterminated",
            f"the card ends {shown.strip()[-12:]!r} but the speech ends "
            f"{spoken.strip()[-12:]!r}: no sentence terminator for the engine",
            shown, spoken))
    else:
        # ...and terminators lost in the MIDDLE, which is worse: they merge
        # two sentences into one run-on that _split_sentences can then only
        # break at a comma. His live schedule reply on 2026-08-31 lost the
        # stops after "4:10 pm." and "3:00 pm." and became one 400-character
        # sentence.
        lost = len(_TERMINATOR.findall(shown)) - len(_TERMINATOR.findall(spoken))
        if lost > 0:
            res.findings.append(Finding(
                "unterminated",
                f"{lost} sentence terminator(s) on the card are gone from "
                f"the speech: the sentences run together",
                shown, spoken))

    res.chunks = tts.render_chunks(shown)
    for i, chunk in enumerate(res.chunks[:-1]):
        if not _ENDS_A_SENTENCE.search(chunk.strip()):
            res.findings.append(Finding(
                "split",
                f"chunk {i + 1}/{len(res.chunks)} is cut mid-sentence after "
                f"{len(chunk)} chars: ...{chunk.strip()[-40:]!r}",
                shown, spoken))
    return res


# ---------------------------------------------------------------- corpora
def load_log(path: Path) -> tuple[list[str], int]:
    """Every line Jarvis spoke in a real session, from its log.

    Returns (lines, skipped). The log truncates the spoken text, so any line
    sitting exactly on a truncation width is dropped rather than reported as
    a speech-path truncation it never was."""
    rx = re.compile(r"jarvis\.tts INFO speaking \([a-z0-9]+(?:, cached)?\): "
                    r"(.*)$")
    seen, out, skipped = set(), [], 0
    for raw in path.read_text(errors="replace").splitlines():
        m = rx.search(raw)
        if not m:
            continue
        line = m.group(1).strip()
        if len(line) in (60, 120):        # the log's own %.60s / %.120s
            skipped += 1
            continue
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return out, skipped


def load_file(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


# ------------------------------------------------------------------ live
def ask_live(text: str, timeout: float) -> str | None:
    """Ask the running Jarvis in TEXT ONLY and return his reply.

    ``jarvis -q`` goes over /tmp/vss_voice/command.sock, so no wake word, no
    microphone, no Whisper -- and the soundbar stays silent. Exit 2 means
    Jarvis is not running; 3 means the turn produced no reply."""
    proc = subprocess.run(
        [str(REPO / "scripts" / "jarvis"), "-q", "-t", str(int(timeout)), text],
        capture_output=True, text=True, timeout=timeout + 15)
    if proc.returncode == 2:
        raise SystemExit("Jarvis is not running; --live has nothing to drive")
    reply = proc.stdout.strip()
    return reply or None


# ----------------------------------------------------------------- report
def report(results: list[LineResult], verbose: bool) -> int:
    counts: Counter = Counter()
    for res in results:
        for f in res.findings:
            counts[f.kind] += 1

    shown_any = False
    for kind in BLOCKING + ADVISORY:
        hits = [(r, f) for r in results for f in r.findings if f.kind == kind]
        if not hits:
            continue
        shown_any = True
        tag = "BLOCKING" if kind in BLOCKING else "advisory"
        print(f"\n=== {kind}  ({len(hits)})  [{tag}]")
        for res, f in hits[:200]:
            print(f"  shown : {res.shown}")
            print(f"  spoken: {res.spoken}")
            print(f"  ->     {f.detail}")
            if verbose and res.chunks:
                for i, c in enumerate(res.chunks, 1):
                    print(f"         chunk {i}: {c!r}")
            print()

    clean = sum(1 for r in results if not r.findings)
    print(f"\n{len(results)} lines · {clean} clean · "
          + (", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
             or "no findings"))
    if not shown_any:
        print("the room hears what the transcript says, on every line.")
    blocking = sum(v for k, v in counts.items() if k in BLOCKING)
    return 1 if blocking else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="builtin",
                    help="'builtin' or a path with one reply per line")
    ap.add_argument("--log", type=Path,
                    help="harvest the spoken lines out of a jarvis.log")
    ap.add_argument("--engine", default=None,
                    help="edge|xtts|f5|fish (default: the configured one)")
    ap.add_argument("--live", action="store_true",
                    help="drive the RUNNING Jarvis, one text turn per line")
    ap.add_argument("--runs", type=int, default=0,
                    help="how many corpus entries to use (0 = all)")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--allow-mutating", action="store_true",
                    help="--live: also send turns that CHANGE his data")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    engine = args.engine or configured_engine()
    # cache=False: this is a text harness and must never touch, warm or
    # prune the cache the live Jarvis is playing out of.
    tts = tts_mod.TTS(engine=engine, cache=False)

    if args.live:
        asked = [u for u in (ASKED[:args.runs] if args.runs else ASKED)
                 if args.allow_mutating or not _MUTATING.search(u)]
        print(f"engine {engine} · driving {len(asked)} live turns "
              f"(text only, nothing spoken aloud)\n")
        lines = []
        for i, utterance in enumerate(asked, 1):
            print(f"  [{i}/{len(asked)}] {utterance}", flush=True)
            reply = ask_live(utterance, args.timeout)
            if reply:
                lines.append(reply)
    elif args.log:
        lines, skipped = load_log(args.log)
        print(f"engine {engine} · {len(lines)} distinct spoken lines from "
              f"{args.log} ({skipped} skipped as log-truncated)")
    elif args.corpus != "builtin":
        lines = load_file(Path(args.corpus))
        print(f"engine {engine} · {len(lines)} lines from {args.corpus}")
    else:
        lines = SPOKEN
        print(f"engine {engine} · {len(lines)} lines from his 2026-08-31 "
              f"session")

    if args.runs and not args.live:
        lines = lines[:args.runs]

    results = [check_line(tts, ln) for ln in lines]

    if args.json:
        print(json.dumps([{"shown": r.shown, "spoken": r.spoken,
                           "chunks": r.chunks,
                           "findings": [{"kind": f.kind, "detail": f.detail}
                                        for f in r.findings]}
                          for r in results], indent=2))
        return 1 if any(f.blocking for r in results for f in r.findings) else 0
    return report(results, args.verbose)


def configured_engine() -> str:
    """The engine the live Jarvis is set to, so the harness chunks and pads
    the way the room does. voice_settings.json is READ, never written."""
    try:
        settings = json.loads(
            (Path.home() / ".aiws_trainer" / "voice_settings.json").read_text())
        engine = settings.get("tts_engine", "")
    except Exception:
        engine = ""
    return engine if engine in tts_mod._ENGINES else "f5"


# ----------------------------------------------------------------- corpus
#
# Real text from the 2026-08-31 session, pasted in because /tmp is wiped at
# boot and jarvis.log with it. SPOKEN is what Jarvis said; ASKED is what
# Hunter said, for --live.
SPOKEN = [
    # --- the three that started this ---------------------------------
    "Four on your shopping list, sir: milk, milk, eggs, and and bread.",
    "Sir, your 30-second 30 seconds timer timer is up.",
    "One to-do, sir: a buy milk.",
    # --- lists and schedules: the long-sentence cases -----------------
    "You have four items on Monday, sir. There is BIOSENSORS at 9:10 am, "
    "MAGNETIC RESONANCE ENGR at 12:40 pm, an ELECTRICAL DESIGN LAB II "
    "presentation at 4:10 pm, and a BMEN 427 lab due Saturday.",
    "You have BIOSENSORS at 9:10 am, MAGNETIC RESONANCE ENGR at 12:40 pm, "
    "an ELECTRICAL DESIGN LAB II presentation at 4:10 pm, and a BMEN 427 "
    "lab due at midnight, sir.",
    "On Monday, sir, you have BIOSENSORS at 9:10 am, MAGNETIC RESONANCE "
    "ENGR at 12:40 pm, and an ELECTRICAL DESIGN LAB II presentation at "
    "4:10 pm.",
    "It is 87 degrees and partly cloudy in College Station, sir, with a "
    "high of 94 and a low of 74.",
    "Two lists, sir: shopping and packing.",
    "One on your packing list, sir: sunscreen.",
    "Three added to your shopping list, sir.",
    "Off the shopping list, sir; three left.",
    "Struck off, sir; the list is clear.",
    "The list is clear, sir.",
    "You have a shopping list and a packing list, sir.",
    "Clear all three off your shopping list, sir?",
    # --- times, dates, numbers: the pronunciation pass ----------------
    "18 percent of 74 is 13.32, sir.",
    "It's 8:30 in the evening, sir.",
    "It's 9:18 at night, sir.",
    "It's 2:31 in the morning on Tuesday the 1st of September, sir.",
    "It's Monday the 31st of August, sir.",
    "Added hello, Tuesday at 4:30 PM, to your calendar, sir.",
    "hello has been added to tomorrow at 4:30 pm, sir.",
    "Your final presentation for ECEN 404 is in 79 days.",
    "I would advise against a 8:49 pm alarm, sir; you're MAGNETIC "
    "RESONANCE ENGR until 8:50 pm.",
    "Setting it anyway, sir. Alarm in 6 minutes, sir.",
    "30 seconds, sir; I'll let you know.",
    "10 minutes, sir; I'll let you know.",
    "30 seconds, sir; I'll let you know. Added to your list.",
    "Cancelled 3, sir.",
    "Timer scrapped, sir.",
    "No timers running, sir.",
    "No alarms set, sir.",
    "No reminders set, sir.",
    "Nothing to cancel, sir.",
    # --- memory, people ----------------------------------------------
    "Noted. I'll remember that.",
    "Noted, sir: your mom is Heather.",
    "Noted, sir: your dad is Ali.",
    "I remember that your anniversary with Mara is on September 20th, sir.",
    "Understood, sir; I've cleared that from my memory.",
    "One note, sir: Bettany voice is the XTTS-1.",
    "You asked about the last thing we worked on, sir.",
    # --- short lines: the F5 pad path ---------------------------------
    "Noted, sir.",
    "Not at all, sir.",
    "Right here, sir.",
    "Welcome back, sir.",
    "Board down, sir.",
    "The board, sir.",
    "Repeat on, sir.",
    "Volume 100, sir.",
    "Say that again, sir?",
    "Was that for me?",
    "Good evening, sir.",
    "Good night, sir; I'll be here in the morning.",
    "Nothing's running, sir.",
    "Looking that up, sir.",
    "Looking into it now, sir.",
    "One moment, sir — I'm checking.",
    "Checking right now, sir. One moment.",
    "Dropping the needle, sir.",
    # --- the refusals and the apologies -------------------------------
    "I'm afraid I don't have that information, sir.",
    "I'm afraid I cannot brighten your screen, sir; I lack the hardware "
    "control for it.",
    "I'm afraid I don't know when those teams are playing, sir.",
    "I'm afraid I'm not quite sure what you'd like me to say, sir.",
    "My apologies, sir; I'll keep quiet then.",
    "My apologies, sir; I must have been imagining things.",
    "I only answer to Hunter, sir.",
    "My vision model isn't answering, sir.",
    "I couldn't get Claude started in the terminal, sir.",
    "I can't see Spark on Spotify, sir; I can see HPCOMPUTER.",
    "I couldn't make out the time, sir; try 'in ten minutes' or 'at half "
    "past four'.",
    # --- music, systems, identifiers ----------------------------------
    "A Cut This Deep by Ike Dweck, sir — on HPCOMPUTER.",
    "All my liked songs, sir — on HPCOMPUTER.",
    "Save Your Tears by The Weeknd is up next, sir.",
    "Tests passed in wf 1794aa61 1e7 1, sir.",
    "The tests errored in wf a20f39d5 c33 2, sir.",
    "Tests passed in Jarvis, sir.",
    "Knightfall is up on demon-bot, sir, along with the other eight "
    "services.",
    "The Spark is breathing quite comfortably, sir.",
    "I'm coming out of the soundbar, sir, which is where I should be.",
    "Quite well, sir, and keeping my voice down.",
    "I'm merely observing the silence, sir; it's quite a peaceful hour "
    "for it.",
    "I manage your calendar, weather, mail, reminders, and alarms, sir.",
    "You are in College Station, Texas, sir.",
    "King Charles the Third is the current King of England, sir.",
    "The test project is set up, sir; ready when you are.",
    "I am beginning the creation of your markdown file, sir.",
    "I have no documents indexed yet, sir; put them in the documents "
    "folder and I will read them.",
]

# What HE said, verbatim from the Transcribed: lines of that session. Used
# by --live; the mutating ones are filtered out unless --allow-mutating.
ASKED = [
    "Are you there?",
    "What time is it?",
    "What's the date today?",
    "What's the time in London?",
    "How are you?",
    "What can you do?",
    "Where am I?",
    "What's the weather?",
    "What's the weather outside?",
    "What's the weather tomorrow?",
    "What's 18% of 74?",
    "What's on today?",
    "What's on this week?",
    "What's on my schedule?",
    "What do I have going on tomorrow?",
    "What am I doing today?",
    "How many days until my biosensor's midterm?",
    "When is my next quiz?",
    "How long is my biosensors lab tomorrow?",
    "What's on my list?",
    "What are my lists?",
    "What list do I have?",
    "What's on the shopping list?",
    "What is my notes?",
    "Any timers running?",
    "Any reminders?",
    "Any mail",
    "What's my last email about?",
    "What do you remember?",
    "What was the last thing we worked on?",
    "What did I previously ask you?",
    "What speaker are you coming out of?",
    "How's the spark doing?",
    "System Health",
    "Is Nightfall up?",
    "Why was that slow?",
    "Why are you being quiet?",
    "Tell me a joke.",
    "Good morning.",
    "Good evening.",
    "Thank you.",
    "status report",
]


if __name__ == "__main__":
    raise SystemExit(main())
