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
# THE BARS LIVE IN THE PACKAGE, not here, since the in-app run
# (jarvis/voicerun.py) needed the same numbers. They MOVED unchanged and
# are re-exported under their old names so this script's own tests, its
# printing and its refusals are exactly what they were -- and so there is
# only ever ONE loudness floor, ONE cohesion bar and ONE separation bar
# in the tree. See jarvis/voiceenrol.py's header for why a second copy is
# the failure being avoided.
from jarvis.voiceenrol import (MIN_RMS, PROMPTS,   # noqa: E402,F401
                               SECONDS, TAKES, pool_ok,
                               separation_margins, take_ok,
                               voiceprint_vectors)
from jarvis.config import CONFIG, PATHS             # noqa: E402
from jarvis import consent as cs                    # noqa: E402
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
# THE VOICE CONSENT WORDS MOVED to jarvis/consent.py as
# ``consent.WHAT_VOICE``, unchanged, because a second caller now
# exists (the in-app run) and two copies of a consent paragraph
# is how one of them quietly stops describing what is stored.
CONSENT_LINES = cs.VOICE_LINES

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


def his_labels(gallery, vectors=None):
    """Every gallery label that IS the owner's pool, by the two routes the
    runtime uses -- ``speaker._owner_pools``' question, asked here so the
    script and the runtime cannot come to different answers.

    1. PROVENANCE: the takes were carried out of ``voiceprint.npz``
       (``--migrate`` or ``--reanchor`` stamped every one of them). His by
       construction, whatever the pool measures today.
    2. MEASUREMENT: the pool's centroid is within ``OWNER_POOL_COSINE`` of
       the voiceprint's. This is the route that catches a pool nobody
       stamped, and it is the only route on a box with no provenance.

    MORE THAN ONE IS THE ROUND-3 BLOCKER. Two labels holding one voice made
    the gallery rank him against himself: the margin is a floor on his own
    within-person noise, so it could never be cleared, ``near_miss`` fired
    every turn and the gate answered "I can't tell which of you" to him,
    alone -- 0 of 100 measured 2026-09-05. ``identify(same=...)`` now folds
    them so it is no longer a lockout, but one voice still belongs under one
    label and --status says so.
    """
    labels = list(gallery.labels())
    try:
        carried = set(gallery.voiceprint_labels())
    except Exception:  # noqa: BLE001 - an unreadable store attests nothing
        carried = set()
    mine = None
    vecs = voiceprint_vectors(PATHS.VOICEPRINT) if vectors is None else vectors
    if vecs:
        mine = vg.centroid(list(vecs))
    out = []
    for label in labels:
        if label in carried:
            out.append(label)
            continue
        if mine is None:
            continue
        theirs = vg.centroid(gallery.embeddings(label))
        if theirs is not None and vg.cosine(theirs, mine) >= vg.OWNER_POOL_COSINE:
            out.append(label)
    return out


def delete_ok(gallery, label, owner, vectors=None):
    """``(ok, why not)``: may ``--delete`` take this label away?

    THE HOLE THIS CLOSES, measured by the round-3 review on 2026-09-05 and
    identical on d41c17d, so it is older than this lane's round 2.
    ``--delete --label <his own label>`` had NO guard at all. On a box with a
    guest enrolled it walked straight back to the layout ``owner_ready``
    refuses to BUILD -- him in ``voiceprint.npz`` only, her in the gallery --
    where ``identify`` cannot rank him against her at all and answers his own
    voice with the one label it has: measured 100 of 100 -> 89 of 100 at
    apart 0.7 and 99 of 100 -> 19 of 100 at apart 1.0.

    THE REFUSAL IS NARROW ON PURPOSE, AND THE FIRST DRAFT OF IT WAS WRONG.
    It refused his last pool whenever anybody else was enrolled -- which
    would have refused the recovery ``migrate_voiceprint``'s own message
    tells him to run (delete the old slug, then ``--migrate`` under the new
    name). A guard that blocks the way out of another guard is not a guard.

    So the line is RECOVERABILITY WITHOUT A MICROPHONE, which is this lane's
    one hard rule. With a readable format-2 ``voiceprint.npz`` on disk, this
    delete is one ``--migrate`` away from being undone and it is ALLOWED --
    loudly, see ``delete_warning``. With no voiceprint to carry back in, the
    only way to a pool of his is eight takes at the microphone, and that is
    the bill this lane promised he would never be sent: REFUSED.

    ``--yes`` overrides even that, because it is his voice and his machine
    and a withdrawal of his own consent must not be something a tool can
    veto. The refusal makes the cost visible; it does not lock a door.
    """
    label = str(label or "")
    if label not in gallery.labels():
        return True, ""                    # nothing to lose; purge_label says so
    mine = his_labels(gallery, vectors)
    if label not in mine or len(mine) > 1:
        return True, ""
    others = [x for x in gallery.labels() if x != label]
    if not others:
        # The gallery becomes EMPTY, which is no instrument at all: the
        # verifier fails open exactly as it did before this feature existed.
        # There is nothing here to protect him from.
        return True, ""
    if vectors is None:
        vectors = voiceprint_vectors(PATHS.VOICEPRINT)
    if vectors:
        return True, ""                    # one --migrate brings it back
    return False, (
        "%s is the ONLY pool in the voice gallery that is the owner's, %s "
        "stay(s) enrolled, and there is no readable %s to carry back in -- "
        "so removing it leaves Jarvis listening for them and not for him, "
        "and the only way back is eight takes at the microphone. That is the "
        "one bill this feature promised never to send him.\n"
        "Remove the others first, or if you mean it anyway: "
        "--delete --label %s --yes"
        % (label, ", ".join(others), PATHS.VOICEPRINT.name, label))


def delete_warning(gallery, label, vectors=None):
    """``""`` or the cost of a delete that IS recoverable, said out loud.

    Deleting his only gallery pool while somebody else is enrolled leaves the
    layout the enrolment script refuses to build. It is undoable in one
    command with no microphone, which is why ``delete_ok`` allows it -- but
    "allowed" is not "free", and nothing used to say so at all.
    """
    label = str(label or "")
    if label not in gallery.labels():
        return ""
    mine = his_labels(gallery, vectors)
    if label not in mine or len(mine) > 1:
        return ""
    others = [x for x in gallery.labels() if x != label]
    if not others:
        return ""                          # back to the pre-feature box
    return (
        "NOTE: %s is the owner's own pool and %s stay(s) enrolled. Without "
        "his\npool the gallery cannot rank him against them and answers his "
        "own voice with\nthe one label it has -- measured, his own turns fall "
        "from 99 of 100 to 19 of\n100 at the widest separation the script "
        "will enrol. Put it back when you are\ndone, no microphone needed:"
        "\n    %s %s --migrate"
        % (label, ", ".join(others), sys.executable, __file__))


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


def _owner_anchor_line(gallery, label) -> None:
    """Whether his pool still MEASURES as voiceprint.npz, printed beside it.

    THE WARNING NAMED THIS INSTRUMENT AND THIS INSTRUMENT COULD NOT SEE IT.
    ``speaker._disowned`` logs "it is NOT being read as the owner ... Check
    with: scripts/voice_enrol.py --status", and --status printed take counts,
    cohesion and floors -- every number except the one that decides whether
    Jarvis answers him. Passive learning used to walk this cosine under the
    line on its own (see speaker._would_leave_his_own_pool), which is exactly
    the fault a person would come here to diagnose.

    The number is the runtime's own: vg.OWNER_POOL_COSINE over the two
    centroids, the same comparison speaker._owner_alias_cosine makes.
    """
    vectors = voiceprint_vectors(PATHS.VOICEPRINT)
    if not vectors:
        print("               anchor : no %s to measure against"
              % PATHS.VOICEPRINT.name)
        return
    mine = vg.centroid(list(vectors))
    theirs = vg.centroid(gallery.embeddings(label))
    if mine is None or theirs is None:
        return
    sim = vg.cosine(theirs, mine)
    if sim >= vg.OWNER_POOL_COSINE:
        print("               anchor : %.4f of %s (needs %.2f) -- read as his"
              % (sim, PATHS.VOICEPRINT.name, vg.OWNER_POOL_COSINE))
        return
    print("               anchor : %.4f of %s, under %.2f"
          % (sim, PATHS.VOICEPRINT.name, vg.OWNER_POOL_COSINE))
    if gallery.carried_from_voiceprint(label):
        # STALE, NOT STOLEN, AND THE TWO USED TO PRINT THE SAME SENTENCE.
        # This pool was COPIED OUT OF voiceprint.npz, so it is his by
        # construction and speaker._disowned no longer drops it -- he is not
        # refused. What it does cost is real and is the whole reason this
        # line still shouts: passive learning stalls while the pool is off
        # its anchor, and every margin is measured against a centroid that no
        # longer describes what the microphone hears. The usual cause is
        # scripts/enroll_voice.py re-recording the voiceprint afterwards.
        print("  ** his pool and his voiceprint have come APART. He is still "
              "read as\n     himself -- this pool came out of %s, so it is "
              "his whatever it\n     measures -- but passive learning is "
              "stalled and every margin is\n     measured against a stale "
              "centroid. Put them back, no microphone\n     needed:"
              "\n         %s %s --reanchor **"
              % (PATHS.VOICEPRINT.name, sys.executable, __file__))
        return
    print("  ** this pool is NOT being read as %s. Jarvis will refuse his own "
          "turns.\n     Takes recorded under his name by somebody ELSE look "
          "exactly like\n     this, which is why nothing here assumes it is "
          "him. If it IS his own\n     pool, carry his voiceprint in over "
          "it:\n         %s %s --reanchor **"
          % (label, sys.executable, __file__))


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
    mine = his_labels(gallery)
    if owner and owner not in gallery.labels() and not mine:
        print("  ** the owner (%s) is NOT in the gallery: it cannot tell him "
              "from anybody here. Run --migrate. **" % owner)
    if len(mine) > 1:
        # THE ROUND-3 BLOCKER, NAMED WHERE HE WOULD LOOK FOR IT. Two labels
        # holding one voice used to refuse him every turn (0 of 100, measured
        # 2026-09-05) and NOTHING said so: the log was silent, --status
        # printed two healthy-looking pools, and the recovery was a command
        # nobody had written down. identify(same=...) folds them now, so this
        # is untidiness rather than a lockout -- but an untidiness that
        # stalls passive learning and confuses every margin, so it is still
        # printed first and still has one command.
        print("  ** MORE THAN ONE LABEL HOLDS THE OWNER'S VOICE: %s."
              % ", ".join(mine))
        print("     One voice belongs under one label. Jarvis folds them "
              "rather than\n     ranking him against himself, so he is not "
              "locked out -- but keep one:")
        print("         %s %s --delete --label %s"
              % (sys.executable, __file__, mine[0]))
        print("     (the one to keep is the one --migrate would write now: "
              "%s) **" % (owner or "the owner's label"))
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
        if label in mine or (owner and label == owner):
            _owner_anchor_line(gallery, label)
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
        ok, why = delete_ok(gallery, args.label, owner)
        if not ok and not args.yes:
            print("REFUSED: %s" % why, file=sys.stderr)
            return 6
        note = delete_warning(gallery, args.label)
        if note:
            print("%s\n" % note)
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
