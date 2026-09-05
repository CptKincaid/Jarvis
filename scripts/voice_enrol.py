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

THE ORDER IS NOT OPTIONAL. ``--migrate`` takes ONLY the owner's label (it
carries HIS voiceprint; ``--label`` anybody else is refused before the
gallery is opened), and a guest cannot be enrolled until the owner has a
pool somewhere -- a guest enrolled first on a fresh box measurably left
Jarvis listening for her alone (see ``owner_ready``).
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

# THE NUMBER NAMING REQUIRES, AND NOT A NUMBER OF ITS OWN. A label under
# ``vg.MIN_TAKES_TO_NAME`` takes is PROVISIONAL: it scores and logs, it can
# never be named, and since 2026-09-04 it cannot match as anybody either. A
# guest enrolled with six takes would therefore be a person Jarvis can hear and
# never answer. So the default is the naming floor itself (8 -- measured: a
# centroid is at cos 0.975 of its converged position by 8 takes, 0.987 by 10,
# and flat past 8), ``takes_ok`` refuses fewer, and the two numbers cannot
# drift apart because there is only one.
TAKES = vg.MIN_TAKES_TO_NAME
SECONDS = 8.0
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
def takes_ok(n):
    """``(ok, why not)`` for the number of takes asked for. Pure."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return False, "--takes must be a number"
    if n < vg.MIN_TAKES_TO_NAME:
        return False, ("%d take(s) is under %d, the number a label needs "
                       "before Jarvis will name it. Fewer would enrol somebody "
                       "he can hear and never answer -- a provisional label "
                       "scores, logs, and matches nobody."
                       % (n, vg.MIN_TAKES_TO_NAME))
    return True, ""


def owner_ready(gallery, owner, label, voiceprint_exists):
    """``(ok, why not)``: may ``label`` be enrolled as a SECOND person yet?

    THE OWNER MUST HAVE A POOL BEFORE ANYBODY ELSE IS ENROLLED -- in the
    gallery, or at least in ``voiceprint.npz`` -- and the script REFUSES
    rather than migrating him on the guest's behalf. Three reasons.

    1. ``identify`` ranks GALLERY labels only. With a guest in the gallery
       and the owner still only in ``voiceprint.npz``, it cannot rank him
       against her: the margin never applies, and a voice of hers that also
       clears his voiceprint bar is answered by the voiceprint alone. The
       runtime is not locked out by that layout (a match on the voiceprint
       is still his -- speaker.filter_segments carries WHICH pool matched),
       but the gallery cannot do its one job in it, which is telling the two
       of them apart.
    2. ``--migrate`` is a write of HIS data under his label with his consent
       ("owner"), and it can be refused (a format-1 voiceprint, degenerate
       vectors). That belongs to its own run with its own message, not to
       the middle of somebody else's consent flow.
    3. A GUEST FIRST ON A FRESH BOX LOCKS HIM OUT. The first version of
       this rule allowed it, claiming a box that never had voice ID could
       not be locked out of it. Measured 2026-09-04 on synthetic vectors
       (tests/test_voice_owner_lockout.py): with her ten takes in the
       gallery and him nowhere, the wake gate woke him 0 of 50 (its
       fail-open None became a number under the bar, against HER
       centroid) and the transcript gate admitted him 0 of 50. The
       runtime now reads that gallery as no instrument for him, but the
       script must not build it in the first place.

    ENROLLING THE OWNER HIMSELF IS ALLOWED, BUT NOT OVER AN UN-MIGRATED
    VOICEPRINT. That short-circuit used to be unconditional ("enrolling the
    owner is always allowed") and it was half of the 2026-09-05 blocker:
    whoever was at the microphone got his name for typing it. The refusal
    here is not the identity check -- that needs the takes, and ``pool_ok``
    does it -- it is the one part that can be decided BEFORE eight takes and
    somebody's consent, and it is decided the same way the runtime decides
    it. A fresh pool under his name would not match ``voiceprint.npz``
    (measured 0.896-0.937 against the 0.98 line), so ``speaker._owner_pools``
    would not read it as him: the script must not spend eight takes building
    a pool the runtime will disown. ``--migrate`` is the supported way his
    label comes to exist, and it needs no microphone.
    """
    if label == owner:
        if voiceprint_exists and owner not in gallery.labels():
            return False, (
                "the owner (%s) has a voiceprint that has never been carried "
                "into the voice gallery, and fresh takes under his name would "
                "NOT be read as his (they do not match voiceprint.npz). Carry "
                "it in first -- no microphone needed:\n    %s %s --migrate\n"
                "If somebody else is at the microphone, enrol them under "
                "their own name: --label <their-name>."
                % (owner, sys.executable, __file__))
        return True, ""
    if owner in gallery.labels():
        return True, ""
    if voiceprint_exists:
        return False, (
            "the owner (%s) is not in the voice gallery yet, and the gallery "
            "cannot tell %s from %s until he is. Run this first, no "
            "microphone needed:\n    %s %s --migrate"
            % (owner, label, owner, sys.executable, __file__))
    return False, (
        "the owner (%s) has no voice enrolled anywhere -- no voiceprint.npz "
        "and not in the voice gallery%s. Enrolling %s first would leave "
        "Jarvis listening for %s and nobody else: the wake word would "
        "suppress him and the transcript gate would refuse him. Enrol him "
        "first:\n    %s scripts/enroll_voice.py\n    %s %s --migrate"
        % (owner,
           " (which holds %s)" % ", ".join(gallery.labels())
           if gallery.labels() else "",
           label, label, sys.executable, sys.executable, __file__))


def migrate_label_ok(label, owner):
    """``(ok, why not)``: ``--migrate`` carries the OWNER's voiceprint and
    takes no other label. Pure, and asked BEFORE the gallery is opened.

    Measured 2026-09-04 without this check: ``--migrate --label mara``
    returned 0, filed his fourteen vectors under her label with consent
    recorded "owner", and at runtime the cosine fold (a copy of his pool
    sits at 1.000, over the 0.98 alias line) made "mara" HIS pool -- so
    "read my mail" in his own voice was answered as Mara with KNOWN scope,
    30 of 30. The voiceprint has no label in it; the only label it may
    ever be filed under is his.
    """
    label = str(label or "")
    if not label or label == owner:
        return True, ""
    return False, (
        "--migrate carries the owner's (%s) voiceprint and takes no other "
        "label: %r would file HIS takes under somebody else's name, and "
        "Jarvis would then answer him as %s. Drop --label to migrate him, "
        "or enrol %s at the microphone:\n    %s %s --label %s"
        % (owner, label, label, label, sys.executable, __file__, label))


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


def voiceprint_vectors(path):
    """The owner's stored embeddings, as a list, or [] when there is nothing
    this script may anchor to.

    NUMBERS ONLY, AND NOTHING IS WRITTEN. Format 1 returns [] for
    ``migrate_voiceprint``'s reason: those vectors were pooled before silence
    trimming and score low against trimmed probes, so anchoring to them would
    refuse HIM. [] means "no anchor", and every check below stands down.
    """
    try:
        path = Path(path)
        if not path.exists():
            return []
        data = np.load(path)
        names = list(data.files)
        fmt = int(data["_format"][0]) if "_format" in names else 1
        if fmt != 2:
            return []
        return [np.asarray(data[k], dtype=np.float32).ravel()
                for k in sorted(n for n in names if n.startswith("emb_"))]
    except Exception:  # noqa: BLE001 - an unreadable voiceprint is no anchor
        return []


def pool_ok(gallery, label, vectors, owner="", owner_vectors=None):
    """``(ok, why not)`` for a FINISHED pool, before it is saved.

    Four refusals, and each one prints its number, because "that did not work"
    is not something anybody can act on at eleven at night.

    1. COHESION. A median pairwise cosine above 0.90 is one take recorded
       several times. His own fourteen-take pool measures 0.485.
    2. THE SAME NAME IS THE SAME PERSON. Topping up a label compares the new
       takes against THE ONES ALREADY UNDER IT -- the comparison this loop
       used to skip outright (``if other == label: continue``), which is how
       a stranger's takes could be recorded under anybody's name and never
       measured against the person whose name it was. Same quantity as
       refusal 3, opposite sense: if the two pools ARE separable by the margin
       bar they are two people, and one name cannot hold both.
    3. SEPARATION. If the MARGIN this pool's takes would clear against
       somebody already enrolled -- or theirs against this pool -- has a
       median under ``vg.MARGIN``, ``identify`` would fail to name that
       person more often than not: every such verdict comes back UNKNOWN on
       the margin, which looks exactly like the feature being broken. Better
       to say so now, with the number, than to store it and let them both
       quietly stop working. See ``separation_margins`` for why this is the
       margin itself and not the centroids' cosine.
    4. THE OWNER'S ANCHOR. Under HIS label, the pool that would be stored is
       measured against ``voiceprint.npz`` itself on
       ``vg.OWNER_POOL_COSINE`` -- THE SAME NUMBER speaker._owner_pools
       applies at runtime, so this script can never write a pool the runtime
       would refuse to read as his. It is what makes refusal 2 hold at the
       separations where the margin bar cannot: measured 2026-09-05, her ten
       takes added to his migrated fourteen leave the label's centroid at
       0.809-0.972 against the 0.98 line and are refused at every separation,
       including the ones where she is too confusable for a margin to notice.
    """
    if len(vectors) < 2:
        return False, "a pool needs at least two takes"
    med = vg.median_pairwise(vectors)
    if med is not None and med > vg.COLLAPSED_MEDIAN_COSINE:
        return False, ("these %d takes have a median pairwise cosine of %.3f, "
                       "above %.2f -- that is one take recorded several times, "
                       "not several takes. A real pool measures around 0.485."
                       % (len(vectors), med, vg.COLLAPSED_MEDIAN_COSINE))

    # 2. the same name is the same person
    already = gallery.embeddings(label)
    if len(already) >= 2:
        m_new, m_old = separation_margins(vectors, already)
        if min(m_new, m_old) >= vg.MARGIN:
            return False, (
                "%r already holds %d take(s), and these %d are a DIFFERENT "
                "voice: they clear that pool by a median margin of %.3f and "
                "it clears them by %.3f, where anything at or above %.2f is "
                "two people Jarvis can tell apart. One name cannot hold two "
                "people -- Jarvis would answer whoever it heard as %r. Use "
                "--label for the person actually at the microphone."
                % (label, len(already), len(vectors), m_new, m_old,
                   vg.MARGIN, label))

    # 4. the owner's anchor -- the same number the runtime applies
    if owner and label == owner and owner_vectors is not None \
            and len(owner_vectors) >= 1:
        mine = vg.centroid(list(owner_vectors))
        would_be = vg.centroid(list(already) + list(vectors))
        if mine is not None and would_be is not None:
            sim = vg.cosine(would_be, mine)
            if sim < vg.OWNER_POOL_COSINE:
                if not already:
                    return False, (
                        "%r would be a NEW pool under the owner's name that "
                        "does not match voiceprint.npz (cosine %.3f, needs "
                        "%.2f), so Jarvis would not read it as him anyway. "
                        "Carry his existing voiceprint in first -- no "
                        "microphone needed:\n    %s %s --migrate\n"
                        "If somebody else is at the microphone, enrol them "
                        "under their own name with --label."
                        % (label, sim, vg.OWNER_POOL_COSINE,
                           sys.executable, __file__))
                return False, (
                    "these %d takes would pull %r away from voiceprint.npz "
                    "(the pool would sit at cosine %.3f, and %.2f is where "
                    "Jarvis stops reading it as the owner). Either they are "
                    "not his voice, or they were recorded somewhere his "
                    "voiceprint would not recognise. Nothing was written.\n"
                    "If his pool has ALREADY come apart from voiceprint.npz, "
                    "more takes cannot close it and this refusal is the "
                    "dead end that used to follow: re-anchor the two instead, "
                    "no microphone needed:\n    %s %s --reanchor"
                    % (len(vectors), label, sim, vg.OWNER_POOL_COSINE,
                       sys.executable, __file__))

    for other in sorted(gallery.labels()):
        if other == label:
            continue                # refusal 2 above is this label's own test
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
    owner = owner_label(CONFIG)
    if owner and owner not in gallery.labels():
        print("  ** the owner (%s) is NOT in the gallery: it cannot tell him "
              "from anybody here. Run --migrate. **" % owner)
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
    ap.add_argument("--reanchor", action="store_true",
                    help="his pool has drifted from voiceprint.npz -- refill "
                         "it from the voiceprint; no microphone")
    ap.add_argument("--delete", action="store_true",
                    help="destroy one person's embeddings, everywhere")
    ap.add_argument("--yes", action="store_true",
                    help="do not prompt to redo a take (the OWNER's flag; it "
                         "cannot give anybody else's consent)")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable status; cannot enrol anybody")
    args = ap.parse_args(argv)

    owner = owner_label(CONFIG)
    if args.migrate and args.reanchor:
        print("REFUSED: --migrate creates his pool and --reanchor repairs "
              "one; they are different runs. Ask --status which you need.",
              file=sys.stderr)
        return 2
    if args.migrate or args.reanchor:
        # Refused BEFORE the gallery is opened: nothing read, nothing
        # written, under a label that is not his. A re-anchor lives under
        # the SAME rule as a migration and for the same reason -- the
        # voiceprint has no label in it, and the only one it may ever be
        # filed under is his.
        ok, why = migrate_label_ok(args.label, owner)
        if not ok:
            print("REFUSED: %s" % why, file=sys.stderr)
            return 2

    gallery = vg.default_gallery()
    gallery.load()

    if args.status:
        show_status(gallery)
        return 0

    if args.migrate:
        label = owner
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

    if args.reanchor:
        # THE WAY BACK, and it takes no microphone. See
        # voicegallery.reanchor_voiceprint for the four refusals and for the
        # honest note on what they are and are not: whoever can write
        # voiceprint.npz is already the owner as far as the runtime is
        # concerned, so these catch a MISTAKE, not a takeover.
        out = gallery.reanchor_voiceprint(owner)
        if not out["ok"]:
            print("re-anchor refused: %s" % out["why"], file=sys.stderr)
            if out["cosine"] is not None:
                print("  %s measures %.4f against %r's stored takes "
                      "(the line is %.2f)."
                      % (PATHS.VOICEPRINT.name, out["cosine"], owner,
                         vg.OWNER_POOL_COSINE), file=sys.stderr)
            return 1
        print("re-anchored %r to %s: %d take(s) replaced %d, generation %d."
              % (owner, PATHS.VOICEPRINT.name, out["migrated"],
                 out["replaced"], out["generation"]))
        print("it was at cosine %.4f of the voiceprint and the line is %.2f; "
              "it is 1.0000 now." % (out["cosine"], vg.OWNER_POOL_COSINE))
        print("%s is untouched and stays the rollback. Generation %d is still "
              "on disk if you want it back." % (PATHS.VOICEPRINT,
                                                out["generation"] - 1))
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

    ok, why = takes_ok(args.takes)
    if not ok:
        print("REFUSED: %s" % why, file=sys.stderr)
        return 2
    ok, why = owner_ready(gallery, owner, args.label, PATHS.VOICEPRINT.exists())
    if not ok:
        print("REFUSED: %s" % why, file=sys.stderr)
        return 5

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
    good, why = pool_ok(gallery, args.label, vectors, owner=owner,
                        owner_vectors=voiceprint_vectors(PATHS.VOICEPRINT))
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
