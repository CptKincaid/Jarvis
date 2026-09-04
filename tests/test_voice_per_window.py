"""The round-2 review's fixtures, as tests. Identity is decided PER WINDOW,
an abstention keeps its flag through every path, and the gate asks WHOSE
POOL matched rather than how many labels exist.

WHAT THE REVIEW MEASURED ON 84f5cb6 (re-measured here on the same box before
the fix, same fixtures, synthetic vectors):

  1. ESCALATION   named guest (10 takes), owner not migrated, her 1.22 s
                  command inside a 4 s clip: admitted as HUNTER with owner
                  scope 50/50. Her window cleared the bar on HER centroid,
                  identify() abstained, speaker._who dropped the flag, and
                  "one label -> owner" minted him.
  2. LOCKOUT      same path, owner migrated: HIS 1.22 s commands refused
                  50/50 ("two labels -> nobody").
  3. LOCKOUT      two guests enrolled without --migrate: every turn of his
                  refused 50/50, at 10 takes each and at 6 (provisional
                  labels counted though they can match nobody).
  4. LAUNDERING   6 s capture, him 0-3 s and her 3-6 s: her half kept and
                  the whole clip run as HUNTER 61/100.
  5. PROVISIONAL  6-take "probably mara", un-migrated, at a separation where
                  her voice clears his voiceprint bar: minted as him
                  74/150 (apart 1.0) and 150/150 (apart 2.0, 3.0).
  6. NEAR MISS    found by round 3 in its OWN first draft, on the reviewer's
                  grid: a failed margin with the owner on top and the bar
                  cleared on his pool was admitted as him, and at apart 3.0
                  with him migrated that minted her 5/150 (10 takes) and
                  22/150 (6 takes) where the count rule had minted 0.

THE FIX, IN ONE SENTENCE: every window records which person's centroid it
cleared the bar on (``matched_label``), the clip is one person's and windows
on anybody else's pool are dropped, the verdict comes from the kept windows
with its ``abstained`` flag intact, a name that disagrees with the pool is
withheld on both paths, and the gate makes a nameless match the owner only
when it was measured on HIS pool, the margin did not fail, and the gallery's
best guess above the bar (``top``) is not somebody else.

Every number below is a measurement of the CODE at a synthetic separation
(tests/synthvoice.py); none of it is evidence about two real people. No
microphone, no recording, no real voiceprint.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

from jarvis import gate as gt
from jarvis import speaker as sp
from jarvis import voicegallery as vg
from tests.synthvoice import Voices, centroid, cos
from tests.test_voice_multispeaker_wiring import (_FakeEncoder, _clip, _enrol,
                                                 _gate)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    """A verifier told who the owner is, the way app.py builds it."""
    monkeypatch.setattr(sp, "VOICEPRINT_FILE", tmp_path / "voiceprint.npz")
    enc = _FakeEncoder()
    v = sp.SpeakerVerifier()
    # Set rather than passed, so the same file measures the PRE-fix code
    # (which had no such argument) failing on its numbers, not on a TypeError.
    v.owner_label = "hunter"
    v._model_loaded = True
    monkeypatch.setattr(v, "_ensure_model", lambda: True)
    monkeypatch.setattr(v, "_extract_embedding", enc)
    v.gallery = vg.VoiceGallery(root=tmp_path / "voice_gallery")
    return v, enc, _gate(tmp_path)


def _layout(v, world, migrated, guests):
    him = world.takes("hunter", 14)
    v._embeddings = list(him)
    v._recompute_centroid()
    if migrated:
        for e in him:
            v.gallery.add("hunter", e)
    for label, n in guests:
        _enrol(v.gallery, world, label, n)
    return him


def _short(seconds, speech_s, marker):
    """A clip whose TRIMMED speech is ``speech_s``: a loud head and a
    near-silent tail, so the real trim_silence finds exactly that much."""
    n = int(sp.SAMPLE_RATE * seconds)
    rng = np.random.default_rng(int(abs(marker) * 1000) + 5)
    a = (rng.normal(size=n) * 0.001).astype(np.float32)
    k = int(sp.SAMPLE_RATE * speech_s)
    a[:k] = (rng.normal(size=k) * 0.2).astype(np.float32)
    a[0] = np.float32(marker)
    return a


def _judge(v, gate, clip, text="unlock the door"):
    out, stats = v.filter_segments(clip)
    return out, stats, gate.judge("voice", text, stats=stats,
                                  rejected=out is None)


# ------------------------------------------------ 1 + 2. the short window
@pytest.mark.parametrize("migrated", [False, True],
                         ids=["voiceprint+mara", "voiceprint+hunter+mara"])
def test_a_short_window_is_whoever_it_matched_and_keeps_its_abstention(
        rig, migrated):
    """1.22 s of speech in a 4 s clip, named guest, both layouts. Hers is
    never him (was 50/50 un-migrated); his is never refused (was 50/50
    migrated). The abstention arrives at the gate AS an abstention."""
    v, enc, gate = rig
    world = Voices(seed=101, apart=0.3)
    _layout(v, world, migrated, [("mara", 10)])
    hers_as_owner = hers_admitted = his_refused = 0
    flags = set()
    for i in range(50):
        c = _short(4.0, 1.22, 20.0 + i * 0.01)
        enc.teach(c, world.take("mara"))
        _o, st, d = _judge(v, gate, c)
        assert abs(len(sp.trim_silence(c[:3 * sp.SAMPLE_RATE]))
                   / sp.SAMPLE_RATE - 1.22) < 0.1, "fixture: speech length"
        hers_as_owner += (d.admit and d.who == "hunter")
        hers_admitted += d.admit
        if st["matched"]:
            flags.add((st["abstained"], st["matched_label"]))
        c2 = _short(4.0, 1.22, 30.0 + i * 0.01)
        enc.teach(c2, world.take("hunter"))
        _o, st2, d2 = _judge(v, gate, c2, "what's the time")
        his_refused += (not d2.admit)
        if st2["matched"]:
            flags.add((st2["abstained"], st2["matched_label"]))
    assert hers_as_owner == 0, "%d/50 of her short commands were him" % hers_as_owner
    assert hers_admitted == 0, "%d/50 of her short commands admitted" % hers_admitted
    assert his_refused == 0, "%d/50 of his short commands refused" % his_refused
    assert flags == {(True, "mara"), (True, "")}, flags


# ------------------------------------------------ the whole-clip path
@pytest.mark.parametrize("migrated", [False, True],
                         ids=["voiceprint+mara", "voiceprint+hunter+mara"])
def test_the_whole_clip_path_is_whoever_it_matched(rig, migrated):
    """A capture under 3 s never sees a window: it goes through
    _verify_named. 2.5 s of hers (named guest) is named her and never him;
    2.5 s of his is never refused. Same fact, same pool, both paths."""
    v, enc, gate = rig
    world = Voices(seed=112, apart=0.3)
    _layout(v, world, migrated, [("mara", 10)])
    hers_as_owner = hers_named = his_refused = 0
    for i in range(50):
        c = _clip(2.5, 120.0 + i * 0.01)
        enc.teach(c, world.take("mara"))
        _o, st, d = _judge(v, gate, c, "what's the time")
        hers_as_owner += (d.admit and d.who == "hunter")
        hers_named += (d.who == "mara")
        assert st["matched_label"] == "mara"
        c2 = _clip(2.5, 130.0 + i * 0.01)
        enc.teach(c2, world.take("hunter"))
        _o, st2, d2 = _judge(v, gate, c2, "read my mail")
        his_refused += (not (d2.admit and d2.who == "hunter"))
        assert st2["matched_label"] == ""
    assert hers_as_owner == 0, hers_as_owner
    assert hers_named == 50, hers_named
    assert his_refused == 0, his_refused


def test_the_whole_clip_path_withholds_a_name_that_disagrees_with_the_pool(
        rig):
    """Un-migrated and confusable (apart 2.0): identify() holds only her
    label and names it on HIS voice whenever it clears her bar (pre-fix, the
    windowed path ran 150/150 of his 4 s clips under her name here). The
    reconciliation is one helper for both paths now: a name whose pool is
    not the pool the bar was cleared on is withheld, ``top`` carries the
    guess, and the gate answers nobody -- never her."""
    v, enc, gate = rig
    world = Voices(seed=113, apart=2.0)
    _layout(v, world, False, [("mara", 10)])
    withheld = 0
    for i in range(50):
        c = _clip(2.5, 140.0 + i * 0.01)
        enc.teach(c, world.take("hunter"))
        _o, st, d = _judge(v, gate, c, "read my mail")
        if st["matched_label"] == "":
            # his pool out-scored hers: the gallery's "mara" is withheld
            assert st["who"] == "" and d.who != "mara"
            withheld += (st["top"] == "mara")
        else:
            # her centroid out-scored his: the pool and the name agree
            assert st["matched_label"] == "mara"
        assert d.who != "hunter" or st["top"] in ("", "hunter")
    assert withheld >= 25, withheld


# ------------------------------------------------ 3. two guests, no migrate
@pytest.mark.parametrize("takes", [10, 6], ids=["named", "provisional"])
def test_two_guests_enrolled_without_migrate_cannot_lock_him_out(rig, takes):
    v, enc, gate = rig
    world = Voices(seed=102, apart=0.3)
    _layout(v, world, False, [("heather", takes), ("mara", takes)])
    refused = 0
    for i in range(50):
        c = _clip(4.0, 40.0 + i * 0.01)
        enc.teach(c, world.take("hunter"))
        _o, st, d = _judge(v, gate, c, "read my mail")
        refused += (not d.admit or d.who != "hunter")
        assert st["matched_label"] == ""
    assert refused == 0, "%d/50 of his turns refused" % refused


# ------------------------------------------------ 4. the mixed capture
@pytest.mark.parametrize("migrated", [False, True],
                         ids=["voiceprint+mara", "voiceprint+hunter+mara"])
def test_a_mixed_capture_is_one_persons_and_the_other_half_is_dropped(
        rig, migrated):
    """Him 0-3 s, her 3-6 s. Windows at 0 (his), 1.5 (a blend), 3.0 (hers)
    and the 1.5 s tail at 4.5 (hers). Her audio never reaches Whisper under
    his name and his never under hers -- 100/100, was 61/100 laundered.
    Measured outcomes: migrated 61 hunter / 39 mara / 0 nobody; un-migrated
    14 / 39 / 47 (the blend window names her in the gallery that holds only
    her, the pool says him, the name is withheld: nobody, never laundered).

    THE RESOLUTION LIMIT, SAID: a 3 s window that straddles the change of
    speaker (1.5-4.5 s) holds 1.5 s of each. It clears the bar on whichever
    centroid it sits nearer and is kept with that person's clip, so up to
    one hop of the other person's audio can ride along. That is the window
    geometry, not the identity rule, and it was the same before this
    branch; this test measures the clean windows at 0 and 4.5 s."""
    v, enc, gate = rig
    world = Voices(seed=103, apart=0.3)
    _layout(v, world, migrated, [("mara", 10)])
    w, h = int(3.0 * sp.SAMPLE_RATE), int(1.5 * sp.SAMPLE_RATE)
    laundered = 0
    outcome = {"hunter": 0, "mara": 0, "nobody": 0}
    for i in range(100):
        base = _clip(6.0, 50.0 + i * 0.01)
        n = len(base)
        his, hers = world.take("hunter"), world.take("mara")
        # The straddling window at 1.5 s hears both of them: a unit blend.
        blend = his / np.linalg.norm(his) + hers / np.linalg.norm(hers)
        blend = (blend / np.linalg.norm(blend) * 300.0).astype(np.float32)
        for pos, vec in ((0, his), (h, blend), (2 * h, hers), (3 * h, hers)):
            base[pos] = np.float32(60.0 + i * 0.01 + pos / n)
            enc.teach(base[pos:pos + w], vec)
        out, st, d = _judge(v, gate, base, "what's the time")
        if not d.admit:
            # A blend window that out-scored both clean ones can be an
            # honest near miss; nobody is answered and nothing is laundered.
            outcome["nobody"] += 1
            assert d.who == "" and out is not None
            continue
        assert d.who in ("hunter", "mara")
        outcome[d.who] += 1
        # Which windows survived: his 0-3 s region and her 4.5-6 s region
        # are told apart by their marker samples being in the output.
        his_in = float(out[0]) == float(base[0])
        hers_in = np.float32(base[3 * h]) in out
        if d.who == "hunter" and hers_in:
            laundered += 1
        if d.who == "mara" and his_in:
            laundered += 1
        assert st["matched_label"] == ("" if d.who == "hunter" else "mara")
    assert laundered == 0, ("%d/100 clips carried the other person's audio "
                            "under this one's name (%s)" % (laundered, outcome))
    assert outcome["hunter"] > 0 and outcome["mara"] > 0, outcome


# ------------------------------------------------ 5. the provisional hit
@pytest.mark.parametrize("apart", [1.0, 2.0], ids=["apart-1.0", "apart-2.0"])
def test_a_provisional_hit_is_not_minted_as_the_owner(rig, apart):
    """6-take mara, un-migrated, at separations where her voice clears his
    voiceprint bar. identify() logs "probably mara"; the gate used to mint
    him anyway (63/150 and 150/150). ``top`` carries the guess now."""
    v, enc, gate = rig
    world = Voices(seed=104, apart=apart)
    him = _layout(v, world, False, [("mara", 6)])
    gap = cos(centroid(him), v.gallery.centroid("mara"))
    assert gap >= v.threshold, "fixture: her centroid must be inside his bar"
    as_owner = probably = 0
    for i in range(150):
        c = _clip(4.0, 70.0 + i * 0.01)
        enc.teach(c, world.take("mara"))
        _o, st, d = _judge(v, gate, c)
        as_owner += (d.admit and d.who == "hunter")
        probably += (st["provisional"] == "mara" and st["top"] == "mara")
    assert as_owner == 0, "%d/150 of a 'probably mara' were him" % as_owner
    assert probably >= 100, probably


# ------------------------------------------------ the reviewer's whole grid
@pytest.mark.parametrize("takes", [6, 10], ids=["provisional", "named"])
@pytest.mark.parametrize("apart", [0.3, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("migrated", [False, True],
                         ids=["voiceprint+mara", "voiceprint+hunter+mara"])
def test_at_every_separation_she_is_never_him(rig, takes, apart, migrated):
    """All sixteen cells the round-2 review ran: 150 of her clips each,
    hers as the owner 0/150 in every one. Round 3's first draft passed the
    fourteen the review had measured and failed the two it had not (apart
    3.0, migrated: 5/150 and 22/150), which is why the whole grid is here.

    HIS side is asserted where scripts/voice_enrol.pool_ok would let the
    guest in (measured: median margins 0.72/0.72 at apart 0.3 and
    0.39/0.42 at 1.0 against the 0.20 bar; 2.0 and 3.0 are refused at
    0.15/0.19 and 0.07/0.11): at 0.3 his voice is his 150/150 in both
    layouts; at 1.0 migrated the measured cost is 1/150, one failed margin,
    the same one as before this branch. Un-migrated at 1.0 is asserted
    separately below. At 2.0 and 3.0 his numbers are reported by the probe
    and not asserted: those are separations the script refuses to build.

    HIS WORDS UNDER HER NAME: never, in every layout the script can build.
    The one cell where it happens is un-migrated at apart 3.0, 10 takes:
    3/150 of his clips score higher on HER centroid than on his own
    voiceprint, so the pool is hers, the gallery (which holds only her)
    agrees, and the numbers cannot tell that clip from hers. The script
    refuses that layout twice over (no --migrate, and a separation pool_ok
    rejects); it is measured here, not hidden."""
    v, enc, gate = rig
    world = Voices(seed=104, apart=apart)
    _layout(v, world, migrated, [("mara", takes)])
    hers_as_owner = his_lost = his_as_mara = 0
    for i in range(150):
        c = _clip(4.0, 70.0 + i * 0.01)
        enc.teach(c, world.take("mara"))
        _o, st, d = _judge(v, gate, c)
        hers_as_owner += (d.admit and d.who == "hunter")
        c2 = _clip(4.0, 80.0 + i * 0.01)
        enc.teach(c2, world.take("hunter"))
        _o, st2, d2 = _judge(v, gate, c2, "read my mail")
        his_lost += (not (d2.admit and d2.who == "hunter"))
        his_as_mara += (d2.who == "mara")
    assert hers_as_owner == 0, "%d/150 of hers were him" % hers_as_owner
    if migrated or apart < 3.0:
        assert his_as_mara == 0, "%d/150 of his ran under her name" % his_as_mara
    else:
        assert his_as_mara <= 5, "%d/150 of his ran under her name" % his_as_mara
    if apart <= 0.3:
        assert his_lost == 0, "%d/150 of his were not him" % his_lost
    elif apart <= 1.0 and migrated:
        assert his_lost <= 2, "%d/150 of his were not him" % his_lost


def test_the_un_migrated_layout_costs_him_at_a_confusable_separation(rig):
    """SAID OUT LOUD. With a 10-take guest in the gallery and the owner only
    in voiceprint.npz, identify() can rank nobody but her, so his voice
    clearing HER bar names her outright. Measured at apart 1.0: 43/150 of
    his turns, and pre-fix every one of the 43 RAN UNDER HER NAME (KNOWN
    scope, his words attributed to her); now the name is withheld because
    the bar was cleared on HIS pool, and those turns are refused as nobody
    -- still a cost, never her. This is why scripts/voice_enrol.py refuses
    to build the layout and speaker.load_gallery warns when it finds one."""
    v, enc, gate = rig
    world = Voices(seed=104, apart=1.0)
    _layout(v, world, False, [("mara", 10)])
    lost = as_mara = 0
    for i in range(150):
        c2 = _clip(4.0, 80.0 + i * 0.01)
        enc.teach(c2, world.take("hunter"))
        _o, st2, d2 = _judge(v, gate, c2, "read my mail")
        lost += (not (d2.admit and d2.who == "hunter"))
        as_mara += (d2.who == "mara")
    assert as_mara == 0, "%d/150 of his turns ran under her name" % as_mara
    assert lost < 75, lost


# ------------------------------------------------ the owner's two pools
def test_his_migrated_label_and_his_voiceprint_are_one_pool(rig):
    """Measured: a migrated copy sits at cos 1.000 to the voiceprint, the
    same pool after PASSIVE_CAP passive samples stays above the alias
    line, and a different synthetic person never reaches it."""
    v, enc, _g = rig
    world = Voices(seed=105, apart=0.3)
    him = _layout(v, world, True, [("mara", 10)])
    cents = v._all_centroids()
    assert cos(cents[""], cents["hunter"]) > 0.9999
    assert v._owner_pools(cents) == {"hunter"}
    # A verifier NOBODY told the owner's label still folds the copy in.
    v.owner_label = ""
    assert v._owner_pools(cents) == {"hunter"}
    # ...and after two passive samples at his own floor it still does.
    v._embeddings = list(him) + world.takes("hunter", sp.PASSIVE_CAP)
    v._recompute_centroid()
    drifted = cos(v._centroid, cents["hunter"])
    assert drifted >= sp.MIGRATED_ALIAS_COSINE, drifted
    assert v._owner_pools(v._all_centroids()) == {"hunter"}
    # A different person at every separation the suite uses never reaches
    # the alias line (measured across five seeds: 0.00-0.17, 0.41-0.53,
    # 0.70-0.77 and 0.81-0.85 for apart 0.3, 1.0, 2.0 and 3.0).
    for apart in (0.3, 1.0, 2.0, 3.0):
        for seed in (106, 206, 306):
            w2 = Voices(seed=seed, apart=apart)
            her = centroid(w2.takes("mara", 10))
            mine = centroid(w2.takes("hunter", 14))
            assert cos(her, mine) < sp.MIGRATED_ALIAS_COSINE, \
                (apart, seed, cos(her, mine))


def test_his_clip_never_splits_between_his_two_pools(rig):
    """Three windows of his, migrated: every one is kept and the clip is on
    his pool, whichever centroid the float rounding favoured."""
    v, enc, gate = rig
    world = Voices(seed=107, apart=0.3)
    _layout(v, world, True, [("mara", 10)])
    w, h = int(3.0 * sp.SAMPLE_RATE), int(1.5 * sp.SAMPLE_RATE)
    for i in range(30):
        base = _clip(6.0, 90.0 + i * 0.01)
        for k, pos in enumerate((0, h, 2 * h, 3 * h)):
            base[pos] = np.float32(95.0 + i * 0.01 + k * 0.001)
            enc.teach(base[pos:pos + w], world.take("hunter"))
        out, st, d = _judge(v, gate, base, "read my mail")
        assert st["matched"] == st["total"] == 4, st
        assert len(out) == len(base)
        assert d.admit and d.who == "hunter"


# ------------------------------------------------ the keys, every path
def test_every_path_carries_the_new_keys(rig):
    v, enc, _g = rig
    keys = {"who", "who_scores", "labels", "matched_label", "top",
            "provisional", "near_miss", "abstained", "who_fault"}
    clip = _clip(4.0, 0.81)
    assert keys <= set(v.filter_segments(clip)[1])          # nothing enrolled
    world = Voices(seed=108, apart=0.3)
    _layout(v, world, True, [("mara", 10)])
    enc.teach(clip, world.take("hunter"))
    _o, matched = v.filter_segments(clip)                    # a match
    assert keys <= set(matched) and matched["matched_label"] == ""
    assert matched["who"] == "hunter" and matched["top"] == "hunter"
    short = _clip(1.0, 0.82)
    enc.teach(short, world.take("hunter"))
    _o, abst = v.filter_segments(short)                      # whole-clip abstain
    assert abst["abstained"] is True and abst["matched_label"] == ""
    stranger = _clip(4.0, 0.83)
    enc.teach(stranger, world.take("nobody"))
    out, rej = v.filter_segments(stranger)                   # a rejection
    if out is None:
        assert keys <= set(rej) and rej["who"] == "" and rej["who_scores"]
    v._ensure_model = lambda: False
    assert keys <= set(v.filter_segments(clip)[1])           # fail shut


def test_the_wake_gate_is_untouched_by_pools(rig):
    """score() is still the maximum over every matchable centroid: a label
    can only raise it (fail-open), and the pool name never leaves here."""
    v, enc, _g = rig
    world = Voices(seed=109, apart=0.3)
    _layout(v, world, False, [])
    clip = _clip(2.0, 0.84)
    enc.teach(clip, world.take("mara"))
    before = v.score(clip)
    _enrol(v.gallery, world, "mara", 10)
    assert v.score(clip) > before
    assert isinstance(v.score(clip), float)


# ------------------------------------------------ the gate's near-miss line
def test_the_near_miss_line_is_prewarmed_and_names_nobody():
    assert gt.NEAR_MISS_LINE in gt.PREWARM_LINES
    assert "hunter" not in gt.NEAR_MISS_LINE.lower()


# ------------------------------------------------ the enrol script
def _script():
    path = ROOT / "scripts" / "voice_enrol.py"
    spec = importlib.util.spec_from_file_location("_voice_enrol_pw", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_guests_enrol_with_the_number_naming_requires():
    """MIN_TAKES_TO_NAME is 8; a guest enrolled with six would be a person
    Jarvis can hear and never answer. One number, not two."""
    ve = _script()
    assert ve.TAKES == vg.MIN_TAKES_TO_NAME == 8
    assert ve.takes_ok(8) == (True, "")
    ok, why = ve.takes_ok(6)
    assert ok is False and "8" in why and "6" in why
    assert ve.takes_ok("x")[0] is False


def test_a_second_person_is_refused_until_the_owner_is_in_the_gallery():
    """REFUSE, not auto-migrate: --migrate is a write of HIS data under his
    consent and can itself be refused; it gets its own run. The message
    names the command."""
    ve = _script()
    g = vg.VoiceGallery()
    world = Voices(seed=110, apart=0.3)
    # nobody in the gallery, a voiceprint on disk: a guest must wait
    ok, why = ve.owner_ready(g, "hunter", "mara", True)
    assert ok is False and "--migrate" in why and "hunter" in why
    # the owner himself is always allowed
    assert ve.owner_ready(g, "hunter", "hunter", True) == (True, "")
    # no voiceprint and nobody enrolled: a box that never had voice ID
    assert ve.owner_ready(g, "hunter", "mara", False) == (True, "")
    # a guest already there without him and no voiceprint: enrol him first
    _enrol(g, world, "heather", 10)
    ok, why = ve.owner_ready(g, "hunter", "mara", False)
    assert ok is False and "--label hunter" in why
    # migrated: fine
    _enrol(g, world, "hunter", 14)
    assert ve.owner_ready(g, "hunter", "mara", True) == (True, "")


def test_main_refuses_before_touching_the_microphone(tmp_path, monkeypatch,
                                                     capsys):
    """Both refusals happen before consent and before the recorder: a
    guest with a short --takes, and a guest before the owner."""
    ve = _script()
    g = vg.VoiceGallery(root=tmp_path / "vg")
    monkeypatch.setattr(ve.vg, "default_gallery", lambda: g)
    monkeypatch.setattr(ve, "owner_label", lambda cfg: "hunter")
    monkeypatch.setattr(ve.PATHS, "VOICEPRINT", tmp_path / "voiceprint.npz")
    assert ve.main(["--label", "mara", "--takes", "6"]) == 2
    assert "8" in capsys.readouterr().err
    (tmp_path / "voiceprint.npz").write_bytes(b"")
    assert ve.main(["--label", "mara"]) == 5
    assert "--migrate" in capsys.readouterr().err


def test_the_app_tells_the_verifier_who_the_owner_is():
    src = inspect.getsource(__import__("jarvis.app", fromlist=["JarvisApp"]))
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "SpeakerVerifier"]
    assert calls, "app.py no longer builds a SpeakerVerifier?"
    assert any(kw.arg == "owner_label" for c in calls for kw in c.keywords)


def test_load_gallery_warns_when_the_owner_is_missing_beside_a_guest(
        tmp_path, monkeypatch, caplog):
    world = Voices(seed=111, apart=0.3)
    g = vg.VoiceGallery(root=tmp_path / "vg")
    _enrol(g, world, "mara", 10)
    g.save("test")
    monkeypatch.setattr(vg, "default_gallery", lambda model=None: vg.VoiceGallery(
        root=tmp_path / "vg"))
    v = sp.SpeakerVerifier(owner_label="hunter")
    with caplog.at_level("WARNING", logger="speaker"):
        v.load_gallery()
    assert v.gallery is not None
    assert any("--migrate" in r.getMessage() for r in caplog.records)


# ------------------------------------------------ passive learning stays his
def test_passive_learning_never_takes_a_clip_attributed_to_somebody_else(
        monkeypatch, tmp_path):
    """A KNOWN person's admitted turn reaches _maybe_learn_voice too. Her
    clip -- on her pool, or named or guessed as her -- must never join HIS
    voiceprint; a clip on his pool, named him or nameless, still does."""
    import threading
    from types import SimpleNamespace

    from jarvis.config import CONFIG
    from tests.test_assistant_features import _app

    monkeypatch.setattr(CONFIG, "speaker_verify", True)
    monkeypatch.setattr(CONFIG, "speaker_threshold", 0.3)
    a = _app(monkeypatch, state=tmp_path / "b.json")
    learned = []
    done = threading.Event()
    a.speaker = SimpleNamespace(
        owner_label="hunter",
        add_sample=lambda audio: learned.append(len(audio)) or done.set())
    audio = np.zeros(16000 * 2, dtype=np.float32)
    hers = [{"scores": [0.7], "matched_label": "mara"},
            {"scores": [0.7], "matched_label": "", "who": "mara"},
            {"scores": [0.7], "matched_label": "", "top": "mara",
             "provisional": "mara"}]
    for st in hers:
        a._last_learn_ts = -1e9
        a._maybe_learn_voice(audio, st)
    assert not done.wait(0.2) and learned == []
    for st in ({"scores": [0.7], "matched_label": "", "who": "hunter"},
               {"scores": [0.7]}):
        done.clear()
        a._last_learn_ts = -1e9
        a._maybe_learn_voice(audio, st)
        assert done.wait(2)
    assert learned == [32000, 32000]
