"""HIS NAME IS NOT A CREDENTIAL.

THE HOLE THE ROUND-3 REVIEW AND THE VERDICT BOTH REPRODUCED AT 100/100.
``speaker._owner_pools`` folded a gallery label into the owner's pool for no
reason but the label STRING::

    if label == self.owner_label:
        out.add(label)          # no cosine, no provenance, no consent

so takes recorded under his name WERE him, and every guard in
``scripts/voice_enrol.py`` permitted building it: ``owner_ready``
short-circuits on ``label == owner`` ("Enrolling the OWNER himself is always
allowed") and ``pool_ok``'s separation loop skips the one comparison that
would have noticed (``if other == label: continue``). Two routes out, both
measured 100/100 before this file's fix:

  ROUTE 1, THE NAME   her ten takes under "hunter", 3.00 s of her speech:
                      identify() NAMES her "hunter", the pool she matched is
                      "hunter" so _reconcile agrees, and gate._voice_leg maps
                      the name to the owner. (who='hunter', admit=True)
  ROUTE 2, THE FOLD   the same layout, 1.22 s so identify() abstains:
                      _owner_pools has already made her centroid HIS pool,
                      matched_label="" and the gate's documented fail-open
                      mints her. Needs no name at all, and works on a
                      PROVISIONAL guest too.
  ROUTE 3, POISONING  her ten takes appended to his migrated fourteen: the
                      blended "hunter" centroid answers her at 0.519 on a
                      0.30 bar, and the label is still folded.

THE FIX IS A DESIGN CHANGE, NOT A CONDITION. The owner's identity is anchored
in ``voiceprint.npz`` -- the one store the enrol script refuses to overwrite
-- and NOT in a name any operator can type. So ``_owner_pools`` now asks ONE
question of every gallery label, his included: does this centroid MEASURE as
the voiceprint's pool (>= voicegallery.OWNER_POOL_COSINE)? A label bearing his
name that does not is DISOWNED: not folded, not matchable (the same rule a
provisional label already lives under -- "a label that cannot be named cannot
match as anybody"), and its name is withheld in ``_who`` so nothing downstream
can read it as his. The label string is trusted for exactly one thing, and
only when there is nothing to measure against: bootstrapping a box that has no
voiceprint at all, where the gallery is the only anchor identity has.

WHY THE BAR IS THE ONE THAT WAS ALREADY THERE (0.98, the migration alias
line). Measured here on synthetic vectors, 8 seeds, centroid cosine against
the voiceprint, at apart 0.3 / 1.0 / 2.0 / 3.0 (tests/synthvoice.py; `apart`
has no real-world referent, so these are measurements of the CODE):

    migrated copy + 2 passive takes      0.9953 - 0.9962   folded
    migrated 14 + 8 fresh takes of his   0.9856 - 0.9880   folded
    POISONED: his 14 + her 10            0.8091 - 0.9717   disowned
    IMPOSTOR: her 10 alone under his name  -0.002 - 0.8422 disowned
    his own fresh takes, never migrated  0.8963 - 0.9372   disowned

The last row is the one cost, and it is stated rather than hidden: a pool of
his OWN takes filed under his own name without ``--migrate`` first is not read
as his. It is not a lockout -- the voiceprint still carries him, measured
below -- and ``pool_ok`` now refuses to create that layout at all, pointing at
``--migrate``. The script and the runtime apply the SAME number, so the script
can never write a pool the runtime will not trust.

Synthetic vectors only. No microphone, no recording, no real voiceprint, no
real gallery, nothing rendered anywhere.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from jarvis import speaker as sp
from jarvis import voicegallery as vg
from tests.synthvoice import Voices
from tests.test_voice_multispeaker_wiring import _FakeEncoder, _clip, _gate

ROOT = Path(__file__).resolve().parent.parent


def _file_under(gallery, world, label, speaker, n):
    """``n`` takes of ``speaker``'s voice, stored under ``label``. The two
    are separate arguments ON PURPOSE: that mismatch IS the attack."""
    for e in world.takes(speaker, n):
        gallery.add(label, e)


def _voice_enrol():
    spec = importlib.util.spec_from_file_location(
        "_voice_enrol_under_test", ROOT / "scripts" / "voice_enrol.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    """A verifier told who the owner is, the way app.py builds it."""
    monkeypatch.setattr(sp, "VOICEPRINT_FILE", tmp_path / "voiceprint.npz")
    enc = _FakeEncoder()
    v = sp.SpeakerVerifier()
    v.owner_label = "hunter"
    v._model_loaded = True
    monkeypatch.setattr(v, "_ensure_model", lambda: True)
    monkeypatch.setattr(v, "_extract_embedding", enc)
    v.gallery = vg.VoiceGallery(root=tmp_path / "voice_gallery")
    return v, enc, _gate(tmp_path)


def _short(seconds, speech_s, marker):
    """A clip whose TRIMMED speech is ``speech_s`` exactly."""
    n = int(sp.SAMPLE_RATE * seconds)
    rng = np.random.default_rng(int(abs(marker) * 1000) + 5)
    a = (rng.normal(size=n) * 0.001).astype(np.float32)
    k = int(sp.SAMPLE_RATE * speech_s)
    a[:k] = (rng.normal(size=k) * 0.2).astype(np.float32)
    a[0] = np.float32(marker)
    return a


def _judge(v, gate, clip, text="unlock the door"):
    out, stats = v.filter_segments(clip)
    return stats, gate.judge("voice", text, stats=stats, rejected=out is None)


def _voiceprint(v, him):
    v._embeddings = list(him)
    v._recompute_centroid()


# --------------------------------------------------- the blocker, three routes
def _hers_as_owner(v, enc, gate, world, n=100, total_s=3.0, speech_s=3.0,
                   base=0.4):
    out = 0
    for i in range(n):
        clip = _short(total_s, speech_s, base + i * 1e-4)
        enc.teach(clip, world.take("mara"))
        _stats, d = _judge(v, gate, clip)
        out += int(d.admit and d.who == "hunter")
    return out


@pytest.mark.parametrize("apart", [0.3, 1.0, 2.0])
@pytest.mark.parametrize("speech_s,total_s", [(3.00, 3.0), (1.22, 4.0)],
                         ids=["named-3.00s", "abstains-1.22s"])
@pytest.mark.parametrize("poisoned", [False, True],
                         ids=["her-takes-alone", "appended-to-his"])
def test_a_guest_recorded_under_his_name_gains_nothing_by_it(
        rig, apart, speech_s, total_s, poisoned):
    """ROUTES 1, 2 AND 3, AND THE INVARIANT STATED THE WAY IT IS TRUE.

    Her takes are filed under the gallery label "hunter" -- exactly what
    ``scripts/voice_enrol.py --label hunter`` at the microphone used to build.
    The claim being tested is NOT "she is never admitted": at a synthetic
    separation where her voice clears his 0.30 bar she is admitted on a box
    with NO GALLERY AT ALL, and no label rule can change that. The claim is
    that HIS NAME BUYS HER NOTHING -- she may do no better wearing it than
    she does against the bare voiceprint.

    Measured 2026-09-05, hers-as-owner per 100, BEFORE this fix / baseline
    with an empty gallery / AFTER:

        apart 0.3    100  /    0  /    0
        apart 1.0    100  /   73  /   71
        apart 2.0    100  /  100  /  100

    So the label was worth +100, +27 and 0 points of somebody else's identity,
    and is now worth none. The 73 and 100 are the single-speaker verifier's
    own false-accept rate at separations `apart` invents; the margin bar in
    ``pool_ok`` is what keeps a real pool that confusable off the box (at
    apart 2.0 it refuses the enrolment outright, measured margin 0.119
    against the 0.20 bar). Enrolling her under HER OWN name instead measures
    0/100 at both -- the feature working: she matches her own pool and is
    named.
    """
    v, enc, gate = rig
    world = Voices(seed=202, apart=apart)
    him = world.takes("hunter", 14)

    # THE BASELINE, measured in this same test with the same seeds so the two
    # numbers are comparable: the voiceprint alone, no gallery, no labels.
    _voiceprint(v, him)
    baseline = _hers_as_owner(v, enc, gate, world, total_s=total_s,
                              speech_s=speech_s)

    v2, enc2, gate2 = rig[0], rig[1], rig[2]
    v2.gallery = vg.VoiceGallery(root=None)
    if poisoned:
        for e in him:
            v2.gallery.add("hunter", e)
    _file_under(v2.gallery, world, "hunter", "mara", 10)  # HER ten, HIS name
    wearing_his_name = _hers_as_owner(v2, enc2, gate2, world, total_s=total_s,
                                      speech_s=speech_s)

    assert wearing_his_name <= baseline, (
        "his name bought her %d/100 over a baseline of %d/100 "
        "(apart=%.1f, %.2fs, poisoned=%s)"
        % (wearing_his_name, baseline, apart, speech_s, poisoned))
    if apart == 0.3:
        assert baseline == 0 and wearing_his_name == 0, (baseline,
                                                         wearing_his_name)


def test_even_a_provisional_guest_under_his_name_is_never_him(rig):
    """ROUTE 2 needs no name, so six takes are enough: a PROVISIONAL label
    can be named nobody, but the fold made it his pool regardless."""
    v, enc, gate = rig
    world = Voices(seed=203, apart=0.3)
    _voiceprint(v, world.takes("hunter", 14))
    _file_under(v.gallery, world, "hunter", "mara", 6)
    hers_as_owner = 0
    for i in range(100):
        clip = _short(4.0, 1.22, 0.6 + i * 1e-4)
        enc.teach(clip, world.take("mara"))
        _stats, d = _judge(v, gate, clip)
        hers_as_owner += int(d.admit and d.who == "hunter")
    assert hers_as_owner == 0, "%d/100" % hers_as_owner


@pytest.mark.parametrize("layout", ["nothing", "mara-under-her-own-name",
                                   "mara-under-his-name"])
def test_a_capture_too_short_to_judge_is_the_fail_open_and_not_this_guard(
        rig, layout):
    """WHAT THIS GUARD DOES NOT CLOSE, said out loud so nobody re-files it.

    A capture SHORTER THAN ONE WINDOW with under ABSTAIN_SECONDS of speech
    never reaches the gallery at all: verify() abstains before it extracts an
    embedding and the pipeline FAILS OPEN, because every "Yes." he says
    arrives that short and refusing it is the concrete lockout
    (tests/test_owner_gate.py:74). Measured here at 50/50 in all three
    layouts INCLUDING AN EMPTY GALLERY -- so it is the pipeline's documented
    fail-open, it predates every label, and no label rule can close it. The
    same 1.22 s inside a 4 s capture takes the WINDOWED path, where identity
    IS decided, and there the guard holds at 0/50.
    """
    v, enc, gate = rig
    world = Voices(seed=214, apart=0.3)
    _voiceprint(v, world.takes("hunter", 14))
    if layout == "mara-under-her-own-name":
        _file_under(v.gallery, world, "mara", "mara", 10)
    elif layout == "mara-under-his-name":
        _file_under(v.gallery, world, "hunter", "mara", 10)
    short_clip = windowed = 0
    for i in range(50):
        c = _short(2.0, 1.22, 4.4 + i * 1e-4)
        enc.teach(c, world.take("mara"))
        short_clip += int(_judge(v, gate, c)[1].admit)
        c = _short(4.0, 1.22, 5.4 + i * 1e-4)
        enc.teach(c, world.take("mara"))
        d = _judge(v, gate, c)[1]
        windowed += int(d.admit and d.who == "hunter")
    assert short_clip == 50, "the fail-open moved: %d/50" % short_clip
    assert windowed == 0, "the windowed path leaked: %d/50" % windowed


def test_his_name_is_withheld_from_a_pool_that_is_not_his(rig):
    """The gallery may NAME her "hunter" -- it ranks labels and knows nothing
    of the voiceprint. Nothing downstream may repeat it."""
    v, enc, gate = rig
    world = Voices(seed=204, apart=0.3)
    _voiceprint(v, world.takes("hunter", 14))
    _file_under(v.gallery, world, "hunter", "mara", 10)
    named = 0
    for i in range(60):
        clip = _clip(3.0, 0.8 + i * 1e-4)
        enc.teach(clip, world.take("mara"))
        stats, _d = _judge(v, gate, clip)
        named += int(stats.get("who") == "hunter"
                     or stats.get("top") == "hunter"
                     or stats.get("matched_label") == "hunter")
    assert named == 0, "his name reached the stats dict %d/60" % named


# ------------------------------------------------------- he is never locked out
@pytest.mark.parametrize("speech_s,total_s", [(3.00, 3.0), (1.22, 4.0)],
                         ids=["named-3.00s", "abstains-1.22s"])
@pytest.mark.parametrize("topped_up", [False, True],
                         ids=["migrated", "migrated+8-fresh-of-his"])
def test_a_properly_migrated_owner_still_answers(rig, speech_s, total_s,
                                                 topped_up):
    """THE CONTROL THAT MAKES THE FIX SAFE. A migrated pool -- and a migrated
    pool topped up with eight more of HIS OWN takes -- still measures as his,
    so he is answered exactly as before."""
    v, enc, gate = rig
    world = Voices(seed=205, apart=0.3)
    him = world.takes("hunter", 14)
    _voiceprint(v, him)
    for e in him:
        v.gallery.add("hunter", e)
    if topped_up:
        for e in world.takes("hunter", 8):
            v.gallery.add("hunter", e)
    _file_under(v.gallery, world, "mara", "mara", 10)
    his = 0
    for i in range(100):
        clip = _short(total_s, speech_s, 1.2 + i * 1e-4)
        enc.teach(clip, world.take("hunter"))
        _stats, d = _judge(v, gate, clip)
        his += int(d.admit and d.who == "hunter")
    assert his == 100, "his own turns answered %d/100" % his


def test_with_no_voiceprint_the_label_is_the_only_anchor_there_is(rig):
    """BOOTSTRAP. Nothing to measure against, so the gallery label IS the
    owner -- unchanged, and the reason the fix cannot silence a fresh box."""
    v, enc, gate = rig
    world = Voices(seed=206, apart=0.3)
    for e in world.takes("hunter", 10):
        v.gallery.add("hunter", e)
    his = 0
    for i in range(60):
        clip = _clip(3.0, 2.2 + i * 1e-4)
        enc.teach(clip, world.take("hunter"))
        _stats, d = _judge(v, gate, clip)
        his += int(d.admit and d.who == "hunter")
    assert his == 60, "%d/60" % his


def test_an_unmigrated_owner_is_still_answered_on_his_voiceprint(rig):
    """THE STATED COST, measured: his own fresh takes under his own name
    without --migrate are DISOWNED (0.896-0.937, under the 0.98 line), and he
    is answered anyway -- the voiceprint is what carries him."""
    v, enc, gate = rig
    world = Voices(seed=207, apart=0.3)
    _voiceprint(v, world.takes("hunter", 14))
    for e in world.takes("hunter", 10):
        v.gallery.add("hunter", e)
    his = 0
    for i in range(60):
        clip = _clip(3.0, 2.6 + i * 1e-4)
        enc.teach(clip, world.take("hunter"))
        _stats, d = _judge(v, gate, clip)
        his += int(d.admit and d.who == "hunter")
    assert his == 60, "%d/60" % his


def test_disowning_never_lowers_the_wake_gate_below_his_own_voiceprint(rig):
    """``score()`` is a MAXIMUM and the wake gate reads it as a boolean, so
    dropping a centroid can only lower it. It may never drop below the score
    his own voiceprint would have given on its own."""
    v, enc, _gate_ = rig
    world = Voices(seed=208, apart=0.3)
    him = world.takes("hunter", 14)
    _voiceprint(v, him)
    alone = []
    for i in range(30):
        clip = _clip(3.0, 3.1 + i * 1e-4)
        enc.teach(clip, world.take("hunter"))
        alone.append(v.score(clip))
    _file_under(v.gallery, world, "hunter", "mara", 10)  # hers, his name
    after = []
    for i in range(30):
        clip = _clip(3.0, 3.1 + i * 1e-4)
        after.append(v.score(clip))
    assert all(b is not None and a is not None and b >= a - 1e-6
               for a, b in zip(alone, after)), (min(alone), min(after))


# ------------------------------------------------------ the enrolment guards
def test_pool_ok_refuses_a_different_person_under_an_existing_label(tmp_path):
    """``pool_ok`` skipped the one comparison that mattered. Topping up a
    label with somebody else's voice is refused, for a guest label too."""
    ve = _voice_enrol()
    world = Voices(seed=209, apart=0.3)
    g = vg.VoiceGallery(root=tmp_path / "g")
    for e in world.takes("mara", 10):
        g.add("mara", e)
    ok, why = ve.pool_ok(g, "mara", world.takes("heather", 10))
    assert ok is False and "mara" in why, (ok, why)
    ok, why = ve.pool_ok(g, "mara", world.takes("mara", 10))
    assert ok is True, why


def test_pool_ok_refuses_a_guest_recorded_under_the_owners_name(tmp_path):
    """THE SUPPORTED PATH THAT BUILT THE BLOCKER, both layouts: his label
    empty (un-migrated voiceprint) and his label already migrated."""
    ve = _voice_enrol()
    world = Voices(seed=210, apart=0.3)
    him = world.takes("hunter", 14)
    for migrated in (False, True):
        g = vg.VoiceGallery(root=tmp_path / ("g%d" % migrated))
        if migrated:
            for e in him:
                g.add("hunter", e)
        ok, why = ve.pool_ok(g, "hunter", world.takes("mara", 10),
                             owner="hunter", owner_vectors=him)
        assert ok is False, "permitted with migrated=%s" % migrated
        assert "hunter" in why


def test_pool_ok_allows_him_to_top_up_his_own_migrated_pool(tmp_path):
    """The legitimate use of the same command still works."""
    ve = _voice_enrol()
    world = Voices(seed=211, apart=0.3)
    him = world.takes("hunter", 14)
    g = vg.VoiceGallery(root=tmp_path / "g")
    for e in him:
        g.add("hunter", e)
    ok, why = ve.pool_ok(g, "hunter", world.takes("hunter", 8),
                         owner="hunter", owner_vectors=him)
    assert ok is True, why


def test_pool_ok_sends_an_unmigrated_owner_to_migrate_first(tmp_path):
    """His OWN takes, his own name, voiceprint present but never migrated:
    refused with the command that fixes it, because the runtime would not
    read that pool as his (0.896-0.937 against the 0.98 line)."""
    ve = _voice_enrol()
    world = Voices(seed=212, apart=0.3)
    g = vg.VoiceGallery(root=tmp_path / "g")
    ok, why = ve.pool_ok(g, "hunter", world.takes("hunter", 10),
                         owner="hunter", owner_vectors=world.takes("hunter", 14))
    assert ok is False and "--migrate" in why, (ok, why)


def test_main_passes_the_voiceprint_to_pool_ok(tmp_path):
    """The guard is worthless if main() never hands it the anchor."""
    ve = _voice_enrol()
    import inspect
    src = inspect.getsource(ve.main)
    assert "owner_vectors" in src and "owner=owner" in src, src[-2000:]


# ------------------------------------------------------------- C6, the retry
def test_a_failed_migration_rolls_back_on_any_error(tmp_path, monkeypatch):
    """``migrate_voiceprint`` caught only ValueError, so an OSError from
    save() escaped with fourteen takes still staged in memory and the retry
    stored TWENTY-EIGHT -- his pool doubled, every later margin quietly
    wrong. voiceprint.npz is untouched either way, so this was never his
    identity; it was his margins."""
    world = Voices(seed=213, apart=0.3)
    him = world.takes("hunter", 14)
    src = tmp_path / "voiceprint.npz"
    np.savez(src, _format=np.array([2]),
             **{"emb_%02d" % i: e for i, e in enumerate(him)})
    g = vg.VoiceGallery(root=tmp_path / "g")
    real_save = g.save
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real_save(*a, **k)

    monkeypatch.setattr(g, "save", boom)
    out = g.migrate_voiceprint("hunter", path=src)
    assert out["ok"] is False and out["why"], out
    assert g.count("hunter") == 0, "takes left staged in memory"
    out2 = g.migrate_voiceprint("hunter", path=src)
    assert out2["ok"] is True, out2
    assert g.count("hunter") == 14, "the retry stored %d" % g.count("hunter")
    assert np.load(src)["emb_00"].shape == him[0].shape


def test_owner_ready_refuses_fresh_takes_over_an_unmigrated_voiceprint(tmp_path):
    """The half of the blocker that can be decided BEFORE eight takes and
    somebody's consent. ``owner_ready`` used to return (True, "") for his
    label unconditionally -- "Enrolling the OWNER himself is always allowed"
    -- which is what let anybody at the microphone claim his name."""
    ve = _voice_enrol()
    world = Voices(seed=215, apart=0.3)
    g = vg.VoiceGallery(root=tmp_path / "g")
    ok, why = ve.owner_ready(g, "hunter", "hunter", voiceprint_exists=True)
    assert ok is False and "--migrate" in why, (ok, why)
    # ...but a MIGRATED owner may still top his own pool up,
    for e in world.takes("hunter", 14):
        g.add("hunter", e)
    assert ve.owner_ready(g, "hunter", "hunter", voiceprint_exists=True)[0]
    # ...and a box with no voiceprint at all still bootstraps.
    g2 = vg.VoiceGallery(root=tmp_path / "g2")
    assert ve.owner_ready(g2, "hunter", "hunter", voiceprint_exists=False)[0]
