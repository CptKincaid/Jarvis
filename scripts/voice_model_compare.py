#!/usr/bin/env python3
"""Measure the voice gallery from its STORED NUMBERS, and refuse to guess.

``scripts/face_model_compare.py`` is the same instrument for the face gallery
and this is deliberately its twin: it reads embeddings off the disk, computes
cohesion and separation, prints a table, and never opens the device. There is
no microphone in this file, no recording is read, and no audio exists in the
store to read -- an enrolment take is 192 floats and the sound it came from was
thrown away.

WHAT IT WILL NOT DO, AND THIS IS THE POINT OF THE FILE.

It will not recommend a threshold while fewer than two people are enrolled.
With one person there is NO different-person evidence anywhere on this machine:
the "non-match" figures quoted in jarvis/speaker.py (-0.03..-0.10) are silence,
a television and room tone, not a second human. A bar chosen from one person's
numbers is a bar chosen from nothing, and that is exactly the mistake that put
0.40 into the settings -- a number that sat INSIDE his own genuine band of
0.377-0.397 and measured a 40% false-reject rate before anybody checked.

It will not quote the model card's published error rate as this system's
accuracy either. That figure was measured on a benchmark corpus through a
different microphone in a different room, and repeating it here would be the
same mistake wearing a citation.

What it DOES give you, all of it measured on this box:

    per label   how many takes, the median pairwise cosine, the
                sample-to-centroid range and the LEAVE-ONE-OUT floor -- the
                bottom of that person's own band, which is the only honest
                "how well does this store know them" number available.
    per pair    both directions: every take of A scored against B's centroid,
                and the centroid-to-centroid cosine, with the margin.
    split-half  one person's takes cut into two pretend people, so you can see
                what the margin bar is actually bounding.
    stability   how far a k-take centroid is from the converged one, which is
                where MIN_TAKES_TO_NAME came from.
    noise probe random vectors against the collapsed-pool line, so the guard's
                number is visible rather than asserted.

Usage:
    ~/vss_env/bin/python scripts/voice_model_compare.py
    ~/vss_env/bin/python scripts/voice_model_compare.py --generation 3
    ~/vss_env/bin/python scripts/voice_model_compare.py --trials 5000
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis import voicegallery as vg          # noqa: E402
from jarvis.config import CONFIG               # noqa: E402

RULE = "-" * 72


def _loo(vectors):
    """Every take against the centroid of the others."""
    rows = [np.asarray(v, dtype=np.float64).ravel() for v in vectors]
    if len(rows) < 2:
        return []
    return [vg.cosine(rows[i], vg.centroid([e for j, e in enumerate(rows)
                                            if j != i]))
            for i in range(len(rows))]


def per_label(gallery, out):
    out(RULE)
    out("PER LABEL -- how well this store knows each person")
    out(RULE)
    out("%-14s %5s %9s %17s %11s %s"
        % ("label", "takes", "cohesion", "vs own centroid", "LOO floor",
           "state"))
    for label in gallery.labels():
        pool = gallery.embeddings(label)
        cent = gallery.centroid(label)
        svc = [vg.cosine(e, cent) for e in pool]
        loo = _loo(pool)
        med = vg.median_pairwise(pool)
        out("%-14s %5d %9s %17s %11s %s"
            % (label, len(pool),
               "-" if med is None else "%.3f" % med,
               "%.3f - %.3f" % (min(svc), max(svc)) if svc else "-",
               "-" if not loo else "%.3f" % min(loo),
               "PROVISIONAL" if gallery.provisional(label)
               else "nameable"))
    out("")
    out("cohesion is the median pairwise cosine WITHIN the label. A real pool")
    out("measures around 0.485 (his, 14 takes, 2026-09-04); above %.2f the"
        % vg.COLLAPSED_MEDIAN_COSINE)
    out("store refuses it as one take saved several times.")
    out("The LOO floor is the bottom of that person's own band. It is the bar")
    out("a passive sample would have to beat -- not the accept bar, which is")
    out("only where recognition starts.")


def per_pair(gallery, out):
    labels = list(gallery.labels())
    out("")
    out(RULE)
    out("BETWEEN PEOPLE -- both directions")
    out(RULE)
    if len(labels) < 2:
        out("Only %d label enrolled." % len(labels))
        out("")
        out("THERE IS NO BETWEEN-PEOPLE NUMBER ON THIS MACHINE. Not a weak")
        out("one, not an approximate one: none. Everything this instrument")
        out("could say about telling two voices apart would be invented.")
        return False
    cents = gallery.centroids()
    out("%-14s %-14s %10s %12s %8s"
        % ("speaker", "against", "centroids", "their takes", "margin"))
    for a, b in itertools.permutations(labels, 2):
        theirs = [vg.cosine(e, cents[b]) for e in gallery.embeddings(a)]
        own = [vg.cosine(e, cents[a]) for e in gallery.embeddings(a)]
        margin = min(o - t for o, t in zip(own, theirs))
        out("%-14s %-14s %10.3f %12s %8.3f"
            % (a, b, vg.cosine(cents[a], cents[b]),
               "%.3f-%.3f" % (min(theirs), max(theirs)), margin))
    out("")
    out("'their takes' is every take of the SPEAKER scored against the other")
    out("person's centroid. 'margin' is the WORST per-take gap -- the closest")
    out("any single utterance came to being attributed to the wrong person.")
    return True


def split_half(gallery, out, trials):
    out("")
    out(RULE)
    out("SPLIT-HALF -- what the margin bar is actually bounding")
    out(RULE)
    rng = np.random.default_rng(20260904)
    for label in gallery.labels():
        pool = [np.asarray(v, dtype=np.float64)
                for v in gallery.embeddings(label)]
        if len(pool) < 5:
            out("%-14s too few takes to split" % label)
            continue
        margins = []
        for _ in range(trials):
            idx = rng.permutation(len(pool))
            probe, rest = pool[idx[0]], idx[1:]
            half = len(rest) // 2
            ca = vg.centroid([pool[i] for i in rest[:half]])
            cb = vg.centroid([pool[i] for i in rest[half:]])
            margins.append(abs(vg.cosine(probe, ca) - vg.cosine(probe, cb)))
        m = np.array(margins)
        out("%-14s p50 %.3f  p90 %.3f  p95 %.3f  p99 %.3f  max %.3f  "
            ">= %.2f: %.2f%%"
            % (label, np.percentile(m, 50), np.percentile(m, 90),
               np.percentile(m, 95), np.percentile(m, 99), m.max(),
               vg.MARGIN, 100.0 * float((m >= vg.MARGIN).mean())))
    out("")
    out("ONE person's takes, cut into two pretend people. This is what the")
    out("%.2f margin was derived from: the gap that one voice's OWN variation"
        % vg.MARGIN)
    out("can produce by itself. Clearing it means a verdict is saying")
    out("something that noise could not have said. It does NOT mean two real")
    out("people are separable by %.2f, and nothing here may be read that way."
        % vg.MARGIN)


def stability(gallery, out):
    out("")
    out(RULE)
    out("CENTROID STABILITY -- where MIN_TAKES_TO_NAME = %d came from"
        % vg.MIN_TAKES_TO_NAME)
    out(RULE)
    for label in gallery.labels():
        pool = [np.asarray(v, dtype=np.float64)
                for v in gallery.embeddings(label)]
        if len(pool) < 4:
            continue
        full = vg.centroid(pool)
        row = []
        for k in range(2, len(pool) + 1):
            r = np.random.default_rng(1000 + k)
            vals = [vg.cosine(vg.centroid([pool[i] for i in
                                           r.choice(len(pool), k, replace=False)]),
                              full) for _ in range(400)]
            row.append("k=%d %.3f" % (k, float(np.mean(vals))))
        out("%-14s %s" % (label, "  ".join(row)))
    out("")
    out("cos(centroid of k takes, the converged centroid). Below %d takes a"
        % vg.MIN_TAKES_TO_NAME)
    out("centroid is still moving, so a label with fewer takes scores and")
    out("logs but never names anybody.")


def noise_probe(out):
    out("")
    out(RULE)
    out("NOISE PROBE -- what the collapsed-pool guard does and does not catch")
    out(RULE)
    for tag, gen in (("zero-mean random", lambda r: r.normal(size=(8, 192))),
                     ("offset random", lambda r: r.normal(size=(8, 192)) + 3.0),
                     ("one vector, 8 times",
                      lambda r: np.repeat(r.normal(size=(1, 192)), 8, axis=0))):
        meds = []
        for s in range(12):
            P = gen(np.random.default_rng(s))
            meds.append(vg.median_pairwise(P))
        out("%-22s median pairwise cosine %.3f .. %.3f  %s"
            % (tag, min(meds), max(meds),
               "REFUSED" if max(meds) > vg.COLLAPSED_MEDIAN_COSINE
               else "would pass"))
    out("")
    out("The guard catches COLLAPSE -- the same vector saved several times,")
    out("which is the shape that destroyed the voiceprint on 2026-09-02. It")
    out("does NOT catch rubbish: a pool of zero-mean random vectors sails")
    out("through it. Rubbish is stopped upstream by the trimming, the VAD and")
    out("the recorder's auto-stop, and none of those may be relaxed on the")
    out("strength of this guard.")


def verdict(gallery, out, have_pairs):
    out("")
    out(RULE)
    out("WHAT THIS INSTRUMENT WILL AND WILL NOT SAY")
    out(RULE)
    out("accept bar : %.2f, from voice_settings.json (default %.2f)"
        % (CONFIG.speaker_threshold, vg.ACCEPT_DEFAULT))
    out("margin     : %.2f%s"
        % (vg.MARGIN, "  PROVISIONAL" if vg.MARGIN_IS_PROVISIONAL else ""))
    out("")
    if not have_pairs:
        out("NO THRESHOLD IS RECOMMENDED, AND NONE CAN BE.")
        out("")
        out("Fewer than two people are enrolled, so this machine holds no")
        out("different-person evidence at all. Every number above describes")
        out("one voice's variation against itself. A bar picked from that is")
        out("a bar picked from nothing -- which is how 0.40 came to sit")
        out("inside his own 0.377-0.397 genuine band and reject 40% of what")
        out("he said.")
        out("")
        out("Enrol a second person and run this again:")
        out("    ~/vss_env/bin/python scripts/voice_enrol.py --label <name>")
        return
    out("A second person IS enrolled, so the pair table above is the first")
    out("real between-people measurement on this box. Read it before")
    out("changing anything: if the worst per-take margin is comfortably")
    out("above %.2f the bar is doing its job, and if it is not, the answer" % vg.MARGIN)
    out("is more takes or a different microphone position -- not a lower bar.")
    out("")
    out("Even now this recommends nothing automatically. A threshold that")
    out("moved because a script suggested it is a threshold nobody measured.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generation", type=int, default=None)
    ap.add_argument("--trials", type=int, default=2800)
    args = ap.parse_args(argv)

    gallery = vg.default_gallery()
    lines = []

    def out(text=""):
        lines.append(text)
        print(text)

    if not gallery.load(args.generation):
        out("Nothing is enrolled in the voice gallery at %s." % gallery.root)
        if gallery.foreign_generations:
            out("%d generation(s) there were written by another encoder and "
                "are left alone." % len(gallery.foreign_generations))
        out("")
        out("Carry the owner's existing voiceprint in with:")
        out("    ~/vss_env/bin/python scripts/voice_enrol.py --migrate")
        return 1

    out("voice gallery %s, generation %d, model %s"
        % (gallery.root, gallery.loaded_generation, gallery.model))
    out("%d take(s) over %d label(s): %s"
        % (gallery.total(), len(gallery.labels()),
           ", ".join(gallery.labels())))
    out("")
    per_label(gallery, out)
    have_pairs = per_pair(gallery, out)
    split_half(gallery, out, max(100, int(args.trials)))
    stability(gallery, out)
    noise_probe(out)
    verdict(gallery, out, have_pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
