"""ROUND 6: THE FAMILY COUNT IS READ OFF THE GRID, NOT TYPED IN.

He is told to read the comment on ``gesture.enabled`` before he touches
the switch, so it has to be true. Round 5's comment said 54 retraction
families and a 226-family honest grid; counted directly from
``castgrid.r5windup.retraction_grid()`` it is 90 and 262 -- two sides, five
amplitudes, nine shapes, beside the preserved 172. Every RATE in that
comment was right (262 x 16 phases is the 4192 sequences it quotes); only
the count was wrong, and it was wrong in five places at once because each
was typed by hand. This file derives the numbers from the grid and checks
every place that quotes them, so the comment cannot drift from the code
again. If the grid changes, this fails and names the places -- and the
RATES quoted beside them will need re-measuring too, which no test can do
for you.

NO CAMERA, NO CAPTURE DEVICE, NO FRAME. ``retraction_grid()`` builds
closures over synthetic trajectories; nothing here runs a sequence.
"""
from __future__ import annotations

import inspect
import pathlib
import re
from collections import Counter

from castgrid import r5windup as R
from tests import atkgrid

REPO = pathlib.Path(__file__).resolve().parents[1]


def counts():
    """(retraction families, preserved families, combined, sequences) --
    every one read off the code, none typed here."""
    n_ret = len(R.retraction_grid())
    n_base = len(atkgrid.build())
    phases = inspect.signature(R.sweep_fires).parameters["phases"].default
    return n_ret, n_base, n_ret + n_base, (n_ret + n_base) * int(phases)


# Every place that quotes a count, and the exact one-line phrase it uses.
# Keep the number and the word "families" on one line in the source, or
# this cannot see it.
def quoted(n_ret, n_base, n_all, n_seq):
    return {
        "jarvis/assistant_config.py": [
            "P(fire | NOT a cast gesture), %d families, %d sequences"
            % (n_all, n_seq),
            "preserved %d-family grid alone, %d sequences"
            % (n_base, n_base * (n_seq // n_all)),
        ],
        "jarvis/gesture.py": [
            "added %d families that DO retract" % n_ret,
            "On the honest %d-family grid, %d sequences" % (n_all, n_seq),
            "PLUS %d families invented" % n_ret,
        ],
        "castgrid/r5windup.py": [
            "``retraction_grid()`` below is %d families" % n_ret,
        ],
        "castgrid/test_r5_windup.py": [
            "Add %d families that DO retract" % n_ret,
        ],
        "tests/test_gesture_round5.py": [
            "0.4986 (%d families)" % n_all,
        ],
    }


class TestTheFamilyCountIsDerived:
    def test_the_grid_counts_and_does_not_overlap_the_preserved_one(self):
        n_ret, n_base, n_all, n_seq = counts()
        fam = R.retraction_grid()
        shapes = Counter(name.split("/")[0] for name in fam)
        by_shape = ", ".join("%s %d" % kv for kv in sorted(shapes.items()))
        print("\n  retraction families %d  (%s)" % (n_ret, by_shape))
        print("  preserved %d, combined %d, x%d phases = %d sequences"
              % (n_base, n_all, n_seq // n_all, n_seq))
        assert not set(fam) & set(atkgrid.build()), "a family counted twice"
        assert n_ret > 0 and n_base > 0

    def test_every_place_that_quotes_the_count_quotes_this_one(self):
        n_ret, n_base, n_all, n_seq = counts()
        missing = []
        for rel, phrases in quoted(n_ret, n_base, n_all, n_seq).items():
            text = (REPO / rel).read_text(encoding="utf-8")
            for phrase in phrases:
                if phrase not in text:
                    missing.append("%s lacks %r" % (rel, phrase))
        print("\n  %d phrases checked across %d files"
              % (sum(len(p) for p in quoted(n_ret, n_base, n_all, n_seq)
                     .values()),
                 len(quoted(n_ret, n_base, n_all, n_seq))))
        assert not missing, "the comment has drifted from the grid:\n  " + \
            "\n  ".join(missing)

    def test_the_round_five_miscount_is_gone(self):
        """The numbers the comment used to carry. If the grid ever really
        does have 54 or 226 families this will need editing, and that is
        the point: nobody types these in without looking."""
        n_ret, n_base, n_all, n_seq = counts()
        stale = re.compile(r"\b(?:54|226)(?:-| )famil|\b40 new famil")
        hits = []
        for rel in quoted(n_ret, n_base, n_all, n_seq):
            text = (REPO / rel).read_text(encoding="utf-8")
            for m in stale.finditer(text):
                if not {54, 226}.isdisjoint({n_ret, n_all}):
                    continue                    # then it would not be stale
                line = text.count("\n", 0, m.start()) + 1
                hits.append("%s:%d %r" % (rel, line, m.group(0)))
        assert not hits, "stale counts:\n  " + "\n  ".join(hits)
