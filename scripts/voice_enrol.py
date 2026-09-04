#!/usr/bin/env python3
"""Enrol a voice into the MULTI-SPEAKER gallery -- yours, or a guest's.

``scripts/enroll_voice.py`` writes ``voiceprint.npz``: one pool, one centroid,
one person, and no labels anywhere in the file. This writes
``jarvis/voicegallery.py``'s store instead, which holds several people under
names, so Jarvis can say WHICH of them is speaking rather than only whether it
is him.

THE OWNER DOES NOT RE-ENROL. ``--migrate`` carries his existing fourteen takes
into the gallery under his label with no microphone involved at all -- same
encoder, same 192 dimensions, so the vectors move rather than being recomputed.
Run that first; run this the long way only for somebody new.

ENROLLING SOMEBODY ELSE TAKES THEIR CONSENT, AT THE KEYBOARD, IN PERSON. The
rule is ``scripts/face_enrol.consent`` -- imported, not copied, so there is one
rule rather than two that can drift -- and the wording below is this store's,
because printing the face script's "128 numbers describing your FACE" over a
voice enrolment would be a false statement of what is being kept. Neither
``--json`` nor ``--auto`` nor a pipe can give consent: stdin and stdout must
both be a terminal, so somebody typed it and could read what they agreed to.

WHAT IT REFUSES, AND IT PRINTS THE NUMBER EVERY TIME:
  * a take quieter than MIN_RMS -- that is a muted mic, not a voice;
  * a finished pool whose median pairwise cosine is above 0.90 -- that is one
    take recorded eight times, not eight takes;
  * a finished pool whose centroid sits within the margin of somebody already
    enrolled -- because two people that close cannot be told apart, and
    storing them anyway would make every later verdict UNKNOWN for both of
    them without saying why.

Usage:
    ~/vss_env/bin/python scripts/voice_enrol.py --migrate            # the owner, no mic
    ~/vss_env/bin/python scripts/voice_enrol.py --label mara         # a guest, 8 takes
    ~/vss_env/bin/python scripts/voice_enrol.py --status
    ~/vss_env/bin/python scripts/voice_enrol.py --delete --label mara

THIS IS RECOGNITION, NOT A LOCK. A recording of a voice defeats it, and
jarvis/identity.py says so in its opening paragraph. It is for presence and
courtesy; it is not security and must never be described as any.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis import voicegallery as vg               # noqa: E402
from jarvis.config import CONFIG, PATHS             # noqa: E402
from jarvis.identity import owner_label             # noqa: E402

JARVIS_PID = Path("/tmp/vss_voice/jarvis.pid")

TAKES = 8               # measured: a centroid is at cos 0.975 of its converged
SECONDS = 8.0           # position by 8 takes, and 0.987 by 10. Past 8 it is flat.
MIN_RMS = 0.004         # below this the take is effectively silence

# Varied prompts beat one long monotone take: identify() compares against the
# centroid of a person's takes, so a spread of natural phrasing and distance
# generalises better than eight readings of one sentence. Copied in spirit
# from scripts/enroll_voice.py and reworded for somebody who is not the owner.
PROMPTS = [
    ("Speak normally, from where you are sitting.",
     "Hello Jarvis, my name is {name} and this is my normal speaking voice."),
    ("Same spot, same voice.",
     "What time is it, and what is the weather doing this afternoon?"),
    ("A little quieter, as if it were late.",
     "Could you turn the music down a bit please."),
    ("Lean back, or sit a little further away.",
     "Jarvis, can you tell me what is on the calendar for tomorrow."),
    ("Normal again, but a bit faster.",
     "I'm just passing through, no need to do anything."),
    ("Relaxed and conversational.",
     "That's interesting -- go on, tell me a little more about that."),
    ("Normal distance, normal voice.",
     "Thank you, that is all I needed for now."),
    ("Last one. Say anything you like, about eight seconds' worth.",
     "(anything at all -- your own words are better than mine)"),
]

CONSENT_LINES = (
    "CONSENT -- this is %(who)s's data, not yours.",
    "",
    "Enrolling %(who)s stores a measurement of %(who)s's VOICE: 192 numbers",
    "per take, in %(root)s, at 0600 in a 0700",
    "directory. NO RECORDING IS KEPT. The audio is turned into those numbers",
    "and thrown away; there is no file anywhere that can be played back. The",
    "numbers cannot be turned back into speech, and nothing leaves this",
    "machine -- no cloud, no API, no upload, ever.",
    "",
    "What it is FOR: Jarvis can tell %(who)s apart from the other people he",
    "knows, greet %(who)s by name, and keep something private when somebody",
    "else is in the room.",
    "",
    "WHAT IT IS NOT: it is not a password and not a lock. A recording of",
    "%(who)s's voice would defeat it, and it is not meant to withstand",
    "somebody determined. Being recognised here does NOT let %(who)s change",
    "settings, read the owner's mail or calendar, or send anything off this",
    "machine.",
    "",
    "Deleting it, at any time, and it takes about a second:",
    "    %(python)s %(script)s --delete --label %(who)s",
    "which destroys every generation that holds %(who)s -- including the old",
    "ones -- and leaves everybody else's alone.",
    "",
    "%(who)s must type their own name below. Nobody may type it for them,",
    "and --yes cannot do it either: that is the owner's flag, and this is",
    "not the owner's consent to give.",
)


def face_enrol():
    """``scripts/face_enrol.py`` as a module, loaded by PATH.

    THE CONSENT RULE IS IMPORTED, NOT COPIED -- two copies of a consent rule
    is two rules that can drift, which is the argument facegallery makes for
    not exporting its label pattern. ``scripts/`` has no ``__init__.py``, and
    tests/test_faceenrol.py already loads it this way, so this uses the same
    mechanism rather than relying on namespace-package resolution that
    changes with how the script was invoked.
    """
    import importlib.util
    path = Path(__file__).resolve().parent / "face_enrol.py"
    spec = importlib.util.spec_from_file_location("_face_enrol_for_voice", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------- the pure decisions
def take_ok(rms, min_rms=MIN_RMS):
    """``(ok, why not)`` for one take's loudness. Pure, so it is tested."""
    try:
        value = float(rms)
    except (TypeError, ValueError):
        return False, "the take's loudness could not be measured"
    if not np.isfinite(value):
        return False, "the take's loudness is not a number"
    if value < float(min_rms):
        return False, ("too quiet (rms %.4f, floor %.4f) -- muted mic or the "
                       "wrong device?" % (value, float(min_rms)))
    return True, ""


def separation_margins(mine, theirs):
    """The margins ``identify`` would actually see between two pools:
    ``(median margin for mine, median margin for theirs)``.

    THE SAME QUANTITY THE MARGIN BAR IS APPLIED TO, and not a proxy for it.
    ``identify`` names somebody when ``cos(probe, own centroid) - cos(probe,
    other centroid) >= vg.MARGIN``. So each take here is scored leave-one-out
    against its own pool's centroid and against the other pool's, exactly as a
    fresh utterance would be, and the median over the takes is what more than
    half of that person's verdicts will look like.

    The first version of ``pool_ok`` compared the two CENTROIDS to each other
    and refused above cosine 0.80. Measured 2026-09-04 on synthetic pools at
    his within-person spread: at a centroid cosine of 0.71 that check passed
    and 91-97% of verdicts for BOTH people then failed the 0.20 margin. It
    fired only past 0.80, where every verdict had already been failing since
    about 0.60 -- the wrong quantity, never firing in the regime it existed
    for. Leave-one-out medians track fresh-probe medians within 0.01 on the
    same data, so this is the number, not an estimate of it.
    """
    def _loo(pool, other_c):
        out = []
        for i, e in enumerate(pool):
            rest = [x for j, x in enumerate(pool) if j != i]
            out.append(vg.cosine(e, vg.centroid(rest)) - vg.cosine(e, other_c))
        return float(np.median(out))
    return _loo(mine, vg.centroid(theirs)), _loo(theirs, vg.centroid(mine))


def pool_ok(gallery, label, vectors):
    """``(ok, why not)`` for a FINISHED pool, before it is saved.

    Two refusals, and each one prints its number, because "that did not work"
    is not something anybody can act on at eleven at night.

    1. COHESION. A median pairwise cosine above 0.90 is one take recorded
       several times. His own fourteen-take pool measures 0.485.
    2. SEPARATION. If the MARGIN this pool's takes would clear against
       somebody already enrolled -- or theirs against this pool -- has a
       median under ``vg.MARGIN``, ``identify`` would fail to name that
       person more often than not: every such verdict comes back UNKNOWN on
       the margin, which looks exactly like the feature being broken. Better
       to say so now, with the number, than to store it and let them both
       quietly stop working. See ``separation_margins`` for why this is the
       margin itself and not the centroids' cosine.
    """
    if len(vectors) < 2:
        return False, "a pool needs at least two takes"
    med = vg.median_pairwise(vectors)
    if med is not None and med > vg.COLLAPSED_MEDIAN_COSINE:
        return False, ("these %d takes have a median pairwise cosine of %.3f, "
                       "above %.2f -- that is one take recorded several times, "
                       "not several takes. A real pool measures around 0.485."
                       % (len(vectors), med, vg.COLLAPSED_MEDIAN_COSINE))
    for other in sorted(gallery.labels()):
        if other == label:
            continue
        theirs = gallery.embeddings(other)
        if len(theirs) < 2:
            continue
        m_mine, m_theirs = separation_margins(vectors, theirs)
        if min(m_mine, m_theirs) < vg.MARGIN:
            return False, (
                "%s's takes clear %s's by a median margin of %.3f, and %s's "
                "clear %s's by %.3f; the bar is %.2f. Jarvis could not tell "
                "the two of you apart: more than half the verdicts for %s "
                "would come back \"I can't tell which of you\". Re-record in "
                "a different spot, or use a different microphone."
                % (label, other, m_mine, other, label, m_theirs, vg.MARGIN,
                   label if m_mine < m_theirs else other))
    return True, ""


def separation_report(gallery):
    """Every enrolled pair's centroid cosine and its margin, as text.

    Numbers, not a verdict: with fewer than two REAL people ever recorded on
    this box there is nothing here that could validate a bar. See
    scripts/voice_model_compare.py, which refuses to suggest one.
    """
    cents = gallery.centroids()
    labels = sorted(cents)
    if len(labels) < 2:
        return ["only %d label enrolled: there is no between-people number "
                "on this machine at all." % len(labels)]
    out = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            ea, eb = gallery.embeddings(a), gallery.embeddings(b)
            if len(ea) >= 2 and len(eb) >= 2:
                ma, mb = separation_margins(ea, eb)
                margin = "median margin %s %.3f, %s %.3f (bar %.2f)" % (
                    a, ma, b, mb, vg.MARGIN)
            else:
                margin = "margin needs two takes each"
            out.append("  %-12s vs %-12s  centroid cosine %.3f  %s" %
                       (a, b, vg.cosine(cents[a], cents[b]), margin))
    return out


# ------------------------------------------------------------------ reporting
def show_status(gallery) -> None:
    print("voice gallery   : %s" % gallery.root)
    print("generations     : %s" % (gallery.generations() or "-"))
    print("model           : %s" % gallery.model)
    print("threshold       : %s (voice_settings.json)" % CONFIG.speaker_threshold)
    print("margin          : %.2f%s" % (vg.MARGIN,
                                        "  (PROVISIONAL -- see below)"
                                        if vg.MARGIN_IS_PROVISIONAL else ""))
    if not gallery.labels():
        print("enrolled        : nobody")
        print("\nThe owner's existing voiceprint can be carried in with:")
        print("    %s %s --migrate" % (sys.executable, __file__))
        return
    print("enrolled        :")
    for label in gallery.labels():
        n = gallery.count(label)
        floor = gallery.genuine_floor(label)
        med = vg.median_pairwise(gallery.embeddings(label))
        print("  %-12s %2d take(s)%s  cohesion %s  own floor %s"
              % (label, n,
                 "  PROVISIONAL, will not be named" if gallery.provisional(label) else "",
                 "-" if med is None else "%.3f" % med,
                 "-" if floor is None else "%.3f" % floor))
        if gallery.consent(label):
            print("               consent: %s" % gallery.consent(label))
    print("\nseparation between enrolled people:")
    for line in separation_report(gallery):
        print(line)
    if vg.MARGIN_IS_PROVISIONAL:
        print("\nThe %.2f margin is DERIVED, not validated: it comes from "
              "splitting one\nperson's own takes into two pretend people, so "
              "it bounds what one voice's\nown variation can produce. It is "
              "not evidence that two real people are\nseparable by it. "
              "scripts/voice_model_compare.py measures the real thing\nonce a "
              "second person is enrolled." % vg.MARGIN)


# -------------------------------------------------------------------- the mic
def record_take(rec, seconds, heading, line):
    print("\n  %s" % heading)
    print('  Say: "%s"' % line)
    for n in (3, 2, 1):
        print("    %d..." % n, end="", flush=True)
        time.sleep(1)
    print(" recording %.0fs -- go." % seconds)
    audio = rec.record_fixed(seconds)
    print("    done.")
    return audio


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="",
                    help="who is being enrolled (lowercase, as in the registry)")
    ap.add_argument("--name", default="",
                    help="their display name, for the prompts")
    ap.add_argument("--takes", type=int, default=TAKES)
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--migrate", action="store_true",
                    help="carry the owner's voiceprint.npz in; no microphone")
    ap.add_argument("--delete", action="store_true",
                    help="destroy one person's embeddings, everywhere")
    ap.add_argument("--yes", action="store_true",
                    help="do not prompt to redo a take (the OWNER's flag; it "
                         "cannot give anybody else's consent)")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable status; cannot enrol anybody")
    args = ap.parse_args(argv)

    gallery = vg.default_gallery()
    gallery.load()
    owner = owner_label(CONFIG)

    if args.status:
        show_status(gallery)
        return 0

    if args.migrate:
        label = args.label or owner
        out = gallery.migrate_voiceprint(label)
        if not out["ok"]:
            print("migration refused: %s" % out["why"], file=sys.stderr)
            return 1
        print("migrated %d take(s) from %s into generation %d as %r."
              % (out["migrated"], PATHS.VOICEPRINT.name, out["generation"], label))
        print("%s is untouched and stays the rollback." % PATHS.VOICEPRINT)
        if out["dropped"]:
            print("%d vector(s) were dropped as unusable." % out["dropped"])
        print()
        show_status(gallery)
        return 0

    if not vg.label_ok(args.label):
        print("--label is required and must be lowercase letters, digits, - "
              "or _ (the registry and both galleries share the pattern).",
              file=sys.stderr)
        return 2

    if args.delete:
        out = gallery.purge_label(args.label, reason="consent withdrawn")
        print("removed from %d generation(s); %d file(s) overwritten and "
              "unlinked." % (len(out["generations_with"]), out["removed"]))
        print("still enrolled: %s" % (", ".join(out["labels_on_disk"]) or "nobody"))
        if not out["complete"]:
            print("\nTHE DELETE IS NOT COMPLETE. %s" % (out["reason"] or ""),
                  file=sys.stderr)
            if out["still_holding"]:
                print("generation(s) %s still hold %r"
                      % (out["still_holding"], args.label), file=sys.stderr)
            return 1
        print("complete. (Overwritten and unlinked -- a filesystem-level "
              "erase, not a device-level one.)")
        return 0

    if args.json:
        print("--json cannot enrol anybody: consent has to be read and typed "
              "at a terminal.", file=sys.stderr)
        return 2

    # -------------------------------------------------------------- consent
    ok, how = face_enrol().consent(args.label, owner, gallery.root, print,
                                   args, lines=CONSENT_LINES)
    if not ok:
        print("\n%s" % how, file=sys.stderr)
        return 3

    # ------------------------------------------------------------ the takes
    if JARVIS_PID.exists():
        print("\nNOTE: Jarvis appears to be running. It holds the microphone "
              "and its\n      wake word will fire on this speech. Stop it "
              "first for a clean run.\n")

    from jarvis.recorder import MicArbiter, Recorder
    from jarvis.speaker import SpeakerVerifier

    verifier = SpeakerVerifier(gpu=0, threshold=CONFIG.speaker_threshold)
    print("Loading the ECAPA-TDNN speaker model...")
    if not verifier.load_model():
        print("ERROR: the speaker model failed to load; cannot enrol.",
              file=sys.stderr)
        return 1
    rec = Recorder(MicArbiter(), speaker_verifier=None)
    if not rec.mic_available:
        print("ERROR: no microphone detected.", file=sys.stderr)
        return 1

    name = args.name or args.label.capitalize()
    print("\nEnrolling %s: %d takes of %.0f seconds."
          % (name, args.takes, args.seconds))
    print("Speak the way you normally would -- this is what Jarvis matches "
          "against.")

    staged = []
    take = 0
    while len(staged) < args.takes:
        heading, line = PROMPTS[take % len(PROMPTS)]
        take += 1
        audio = record_take(
            rec, args.seconds,
            "[%d/%d] %s" % (len(staged) + 1, args.takes, heading),
            line.format(name=name))
        if audio is None or len(audio) == 0:
            print("    no audio captured -- retrying.")
            continue
        rms = float(np.sqrt(np.mean(np.square(np.asarray(audio,
                                                         dtype=np.float64)))))
        good, why = take_ok(rms)
        if not good:
            print("    %s retrying." % why)
            continue
        emb = verifier._extract_embedding(audio)
        if emb is None:
            print("    not enough speech in that take -- retrying.")
            continue
        from jarvis.speaker import SAMPLE_RATE, trim_silence
        speech_s = len(trim_silence(audio)) / SAMPLE_RATE
        staged.append((emb, speech_s, rms, heading))
        print("    kept. rms %.4f, %.2fs of speech. %d of %d."
              % (rms, speech_s, len(staged), args.takes))

    # --------------------------------------------------- the pool's own bars
    vectors = [e for e, _s, _r, _h in staged]
    good, why = pool_ok(gallery, args.label, vectors)
    if not good:
        print("\nREFUSED, and nothing was written:\n  %s" % why, file=sys.stderr)
        return 4

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    for emb, speech_s, rms, heading in staged:
        gallery.add(args.label, emb, len_s=speech_s, rms=rms, note=heading,
                    at=stamp, src="enrol")
    gallery.set_consent(args.label, how)
    try:
        gen = gallery.save("enrol %s (%d takes, consent %s)"
                           % (args.label, len(staged), how))
    except ValueError as exc:
        print("\nREFUSED, and nothing was written:\n  %s" % exc, file=sys.stderr)
        return 4
    print("\n--- enrolment complete: generation %d ---" % gen)
    show_status(gallery)
    return 0


if __name__ == "__main__":
    sys.exit(main())
