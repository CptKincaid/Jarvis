"""The voice enrolment BARS: one set of numbers, three callers.

WHY THIS FILE EXISTS, and it is the same argument ``jarvis/consent.py`` made
when it took the consent rule out of ``scripts/face_enrol.py``. Until now the
loudness floor, the cohesion bar, the separation bar and the owner's anchor
lived inside ``scripts/voice_enrol.py`` -- a script, with no ``__init__.py``
above it, loadable only by path. So the in-app voice enrolment
(``jarvis/voicerun.py``) had exactly two choices: import a script by file
path from inside the running assistant, or write a SECOND set of quality bars
beside the first. The second is how two numbers that must agree come to
disagree, and it is the thing ``jarvis/enrolrun.py``'s header refuses in the
same words: "no second copy of the station loop, no second set of quality
bars and no second way to save."

So the functions MOVED here, unchanged. ``scripts/voice_enrol.py`` imports
them back and keeps its own command line, its own printing and its own
terminal consent ceremony; the tests that already pin these refusals
(tests/test_voice_enrol_script.py) call them through the script exactly as
they did, because the script still exposes the same names.

NOTHING HERE OPENS A MICROPHONE, and nothing here writes a file. It is
arithmetic over stored floats and a sentence for each refusal -- which is why
the in-app run can be judged without a device and the numbers it prints are
the numbers the terminal prints.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from jarvis import voicegallery as vg

# THE REMEDY LINES NAME THE SCRIPT, NEVER THIS MODULE. Two of ``pool_ok``'s
# refusals end with the exact command that fixes them, and in the script those
# were ``sys.executable`` and its own ``__file__``. Moved here unchanged they
# would print the path of a module with no ``main()``, which is a remedy that
# cannot be run -- so the script's path is derived once, here, and the
# sentences are otherwise byte-identical.
SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" /
             "voice_enrol.py")

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
                           sys.executable, SCRIPT))
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
                       sys.executable, SCRIPT))

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


