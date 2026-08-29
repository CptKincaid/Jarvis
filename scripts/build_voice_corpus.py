"""Generate a Piper/XTTS training corpus in Jarvis's own voice, via Fish.

WHY. Fine-tuning a local voice needs 10-30+ minutes of clean, single-speaker
audio with transcripts. Scraped film dialogue carries score, room tone and the
character's in-universe processing (tried 2026-08-28, rejected by ear). The
Fish API renders exactly the voice already chosen, with no background at all,
and the transcript is known by construction because we wrote it.

So: pay a few cents to mint a clean corpus, then fine-tune a LOCAL model on it
and stop depending on the network.

COST. Billing is per UTF-8 byte of input text at $15/1M. The whole default
corpus is a few hundred sentences -- well under a dollar. --count previews a
handful first; nothing large runs without you asking for it.

OUTPUT (what Piper training expects):
    <out>/wavs/0001.wav ...        22.05 kHz mono
    <out>/metadata.csv             "0001|transcript" per line, LJSpeech style
"""
import argparse
import csv
import pathlib
import random
import re
import subprocess
import sys
import time

# Assistant-domain sentences: what Jarvis actually says, so the fine-tuned
# voice is strongest exactly where it gets used. Spread deliberately across
# times, numbers, proper nouns, questions, apologies and dry asides -- a
# corpus of only status lines makes a voice that can only read status lines.
TEMPLATES = [
    "Good {tod}, sir.",
    "Good {tod}, sir. Everything is in order.",
    "It is {h} {ampm}, sir.",
    "Your first appointment is at {h} {ampm}, in the {place}.",
    "You have {n} items on {weekday}, sir.",
    "You have {n} items {when}, sir.",
    "The first is {course} at {h} {ampm}, in the {place}.",
    "{weekday} begins with {course}, then {course2} after lunch.",
    "{when} begins with {course}, then {course2} after lunch.",
    "It is {temp} degrees and {sky} in {city}.",
    "Rain is likely by {h} {ampm}, sir. You may want the umbrella.",
    "There are {n} new messages, {n2} of them from the {org} account.",
    "I have moved the {thing} to {weekday} afternoon, as you asked.",
    "The {thing} is still scheduled for {h} {ampm}.",
    "Your reminder for the {thing} is set, sir.",
    "The timer has finished, sir.",
    "The alarm is set for {h} {ampm} tomorrow.",
    "I'm afraid I couldn't reach the {thing} just now, sir.",
    "That didn't work, sir. I'll try again shortly.",
    "I have no record of that, sir.",
    "Would you like me to move it, sir?",
    "Shall I read them out, sir?",
    "Always, sir.",
    "Of course, sir.",
    "Right away, sir.",
    "It's done, sir.",
    "As you wish, sir.",
    "The {thing} finished overnight, with no errors.",
    "Validation loss came down to zero point {dec} over {n} epochs.",
    "The training run is at epoch {n}, sir, and still improving.",
    "The {thing} completed in {n} minutes, which is faster than last time.",
    "You have {n} meetings before noon. I'd offer sympathy, sir, "
    "but you scheduled them yourself.",
    "It is {temp} degrees outside. I would not recommend the walk.",
    "The house is quiet, sir. Everything is where you left it.",
    "Nothing further requires your attention tonight, sir.",
    "Should you need anything at all, you have only to ask.",
    "I'll keep an eye on it and wake you if anything changes.",
]
FILL = {
    "tod": ["morning", "afternoon", "evening"],
    "h": ["one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve", "nine ten", "half past eight",
          "quarter past six", "twenty past three"],
    # NOT "ay em"/"pee em": that spelling was an XTTS workaround, and Fish
    # voices it as "I'm" (heard 2026-08-28). Fish normalises ordinary a.m./
    # p.m. correctly, so the corpus uses ordinary English — which is also
    # what a fine-tuned model should learn to read.
    "ampm": ["a.m.", "p.m.", "o'clock", "in the morning", "in the evening"],
    # split: {weekday} follows "on", {when} stands alone. Together they
    # used to produce "on today", which a fine-tune would learn as correct.
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"],
    "when": ["today", "tomorrow", "this afternoon", "this evening"],
    "n": ["two", "three", "four", "five", "six", "seven", "eight", "ten",
          "twelve", "forty"],
    "n2": ["one", "two", "three"],
    "place": ["Wisenbaker building", "Emerging Technologies Building",
              "Zachry building", "east wing", "main hall", "study"],
    "course": ["Biosensors", "Thermodynamics", "Magnetic Resonance "
               "Engineering", "Electrical Design Lab", "the seminar",
               "the quarterly review"],
    "course2": ["Calculus", "the vendor call", "the site walk",
                "the budget review", "the hardware inspection"],
    "temp": ["forty two", "fifty five", "sixty eight", "seventy two",
             "eighty one", "ninety eight"],
    "sky": ["clear", "overcast", "raining lightly", "windy", "partly cloudy"],
    "city": ["College Station", "Houston", "Austin", "town"],
    "org": ["Kincaid", "work", "personal", "university"],
    "thing": ["vendor call", "dentist", "advisor meeting", "calendar",
              "backup", "training run", "briefing", "deployment"],
    "dec": ["one nine", "two four", "zero eight", "three one"],
}


def sentences(count: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    seen, out = set(), []
    tries = 0
    while len(out) < count and tries < count * 200:
        tries += 1
        t = rng.choice(TEMPLATES)
        s = t
        for key, choices in FILL.items():
            token = "{" + key + "}"
            while token in s:
                s = s.replace(token, rng.choice(choices), 1)
        s = _tidy(s)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _tidy(s: str) -> str:
    """Clean up the seams where fillers meet templates.

    "a.m." already ends in a period, so "{h} {ampm}." produced "two a.m..".
    Training on malformed text teaches malformed reading, so it is fixed
    here rather than hoped away.
    """
    s = re.sub(r"\.{2,}(?=$|\s)", ".", s)      # "a.m.." -> "a.m."
    # only collapse a SPACED ". ," — "a.m.," is correct English and the
    # abbreviation's period must survive.
    s = re.sub(r"\.\s+,", ".,", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5,
                    help="how many clips (default 5 — a preview)")
    ap.add_argument("--out", default=str(pathlib.Path.home() /
                                         ".aiws_trainer/voice_corpus"))
    ap.add_argument("--model", default=None,
                    help="Fish reference_id (default: the configured voice)")
    ap.add_argument("--rate", type=int, default=22050, help="output sample rate")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the sentences and the cost, generate nothing")
    a = ap.parse_args()

    lines = sentences(a.count)
    total_bytes = sum(len(s.encode("utf-8")) for s in lines)
    cost = total_bytes * 15 / 1_000_000
    est_audio_s = total_bytes / 15.0        # ~15 bytes/sec of speech, measured
    print(f"{len(lines)} sentences, {total_bytes} bytes -> ${cost:.4f}, "
          f"~{est_audio_s/60:.1f} min of audio")
    if a.dry_run:
        for i, s in enumerate(lines, 1):
            print(f"  {i:04d}  {s}")
        return 0

    sys.path.insert(0, "/home/hunterp/Jarvis")
    from jarvis.tts import _fish_stream_model, _fish_creds
    _key, configured = _fish_creds()
    model = a.model or configured
    if not model:
        print("no voice model configured", file=sys.stderr)
        return 1

    out = pathlib.Path(a.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, text in enumerate(lines, 1):
        stem = f"{i:04d}"
        raw = out / "wavs" / f"{stem}.raw.wav"
        final = out / "wavs" / f"{stem}.wav"
        for attempt in (1, 2, 3):
            try:
                _fish_stream_model(text, str(raw), 30.0, model)
                break
            except Exception as exc:
                if attempt == 3:
                    print(f"  {stem}: FAILED {exc}", file=sys.stderr)
                    raw = None
                    break
                time.sleep(1.5 * attempt)
        if raw is None:
            continue
        # Piper wants mono at a fixed rate; normalise here so training does
        # not have to care what the API handed back.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                        "-ac", "1", "-ar", str(a.rate), str(final)], check=True)
        raw.unlink()
        rows.append((stem, text))
        print(f"  {stem}  {text[:64]}", flush=True)

    with open(out / "metadata.csv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for stem, text in rows:
            w.writerow([stem, text])
    print(f"\n{len(rows)} clips -> {out}")
    print(f"metadata: {out/'metadata.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
