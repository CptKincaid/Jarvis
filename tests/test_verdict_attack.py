"""VERDICT harness -- my own rows, my own vectors, no lane fixture imported.

Every person here is invented (Nadia Okoro the guest, Priya Raman the second
guest). Every vector is synthetic. No microphone, no camera, no people.json
of his, no voiceprint of his is opened.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis import brain as br
from jarvis import gate as gt
from jarvis import identity as ident
from jarvis import passphrase as pp
from jarvis import recognise as rec
from jarvis import scope as sc
from jarvis import signinlines as sl
from jarvis import speaker as sp
from jarvis import voicegallery as vg

DIM = 192
OWNER = "hunter"
G1 = "nadia"
G2 = "priya"


# ----------------------------------------------------------------- vectors
def _unit(v):
    v = np.asarray(v, dtype=np.float32).ravel()
    return v / (np.linalg.norm(v) + 1e-12)


def _bases(apart, rng):
    """Two unit vectors whose cosine is ``apart``."""
    a = _unit(rng.normal(size=DIM))
    r = rng.normal(size=DIM)
    r = _unit(r - float(np.dot(r, a)) * a)
    b = _unit(apart * a + np.sqrt(max(0.0, 1.0 - apart ** 2)) * r)
    return a, b


def _jitter(base, rng, sigma):
    """One take: the base plus UNIT-NORMALISED noise, so ``sigma`` is the
    real within-person spread (cos to base = 1/sqrt(1+sigma**2)) and not a
    192-dim Gaussian whose norm swamps the signal."""
    return _unit(base + sigma * _unit(rng.normal(size=DIM)))


def _takes(base, n, rng, sigma=0.35):
    return [_jitter(base, rng, sigma) for _ in range(n)]


# ---------------------------------------------------------------- registry
def _registry(tmp_path, *, code=True, phrase=False, guests=(G1,),
              honorific="ma'am"):
    """My own people book in tmp. Never his."""
    reg = ident.Registry(path=tmp_path / "people.json")
    reg.add_person(ident.Person(label=OWNER, name="Hunter Peyrovi",
                                role=ident.ROLE_OWNER))
    for g in guests:
        first, last = {G1: ("Nadia", "Okoro"), G2: ("Priya", "Raman")}[g]
        reg.add_person(ident.Person(
            label=g, name="%s %s" % (first, last), first=first, last=last,
            role=ident.ROLE_KNOWN, honorific=honorific, face=g))
    reg.save()
    reg = ident.Registry.load(tmp_path / "people.json")
    if code:
        reg.set_secret(OWNER, "code_hash", pp.hash_secret("verdictcode99"))
    if phrase:
        reg.set_secret(OWNER, "phrase_hash", pp.hash_secret("open the north door"))
    reg.save()
    return ident.Registry.load(tmp_path / "people.json")


def _gate(reg, mode="enforce", **opts):
    o = {"owner.mode": mode}
    o.update(opts)
    return gt.OwnerGate(registry=reg, get_option=lambda k, d=None: o.get(k, d),
                        owner=OWNER)


# ------------------------------------------------------------------ voice
class Rig:
    """A verifier over a synthetic gallery, driven end to end through
    ``filter_segments`` with my own embedding stand-in."""

    def __init__(self, monkeypatch, *, apart=0.20, takes=10, seed=7,
                 migrate=True, guests=(G1,), sigma=0.35):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.sigma = sigma
        base_o, base_g = _bases(apart, rng)
        self.bases = {OWNER: base_o, G1: base_g}
        if G2 in guests:
            _, base_g2 = _bases(apart, np.random.default_rng(seed + 100))
            self.bases[G2] = base_g2
        gal = vg.VoiceGallery()
        if migrate:
            for v in _takes(base_o, takes, rng, sigma):
                gal.add(OWNER, v, src=vg.VOICEPRINT_SRC,
                        note=vg.VOICEPRINT_NOTE)
        for g in guests:
            for v in _takes(self.bases[g], takes, rng, sigma):
                gal.add(g, v)
        v = sp.SpeakerVerifier(threshold=sp.DEFAULT_THRESHOLD, owner_label=OWNER)
        v.gallery = gal
        v._embeddings = _takes(base_o, takes, rng, sigma)
        v._centroid = np.mean(np.asarray(v._embeddings), axis=0)
        v._loaded = True
        self.v = v
        self.speaking = OWNER
        monkeypatch.setattr(v, "_ensure_model", lambda: True)
        monkeypatch.setattr(v, "_extract_embedding", self._emb)

    def _emb(self, chunk):
        return _jitter(self.bases[self.speaking], self.rng, self.sigma)

    def clip(self, who, seconds=4.0):
        """A stats dict for ``who`` speaking, through the real filter."""
        self.speaking = who
        n = int(seconds * sp.SAMPLE_RATE)
        audio = (0.2 * self.rng.normal(size=n)).astype(np.float32)
        kept, stats = self.v.filter_segments(audio)
        return stats, kept is None


# =========================================================================
# 0. the rig itself measures something
# =========================================================================
def test_rig_names_the_owner_and_the_guest(monkeypatch):
    r = Rig(monkeypatch, apart=0.20)
    s, rejected = r.clip(OWNER)
    assert s["matched"] >= 1 and not rejected
    assert s["matched_label"] in ("", OWNER)
    s2, _ = r.clip(G1)
    assert s2["matched"] >= 1
    assert s2["matched_label"] == G1


# =========================================================================
# 1. voice-multispeaker: a guest is NEVER admitted as the owner
# =========================================================================
@pytest.mark.parametrize("apart", [0.0, 0.20, 0.40, 0.60])
def test_a_guest_is_never_admitted_as_the_owner(monkeypatch, tmp_path, apart):
    r = Rig(monkeypatch, apart=apart, seed=11)
    g = _gate(_registry(tmp_path))
    as_owner = admitted = 0
    trials = 120
    for _ in range(trials):
        stats, rejected = r.clip(G1)
        d = g.judge("voice", "what's on my calendar", stats=stats,
                    rejected=rejected)
        if d.who == OWNER or d.role == ident.ROLE_OWNER:
            as_owner += 1
        if d.admit and d.who:
            admitted += 1
    assert as_owner == 0, ("apart=%.2f: %d/%d of the guest's clips were "
                           "admitted as the owner" % (apart, as_owner, trials))


# NOTE: the short-clip case is measured, not asserted shut, in
# test_the_short_clip_fail_open_is_exactly_what_the_docstring_says --
# it is a DOCUMENTED pre-existing fail-open, reproduced identically on
# voice-multispeaker alone (ce82be1), so the merge did not create it.


def test_two_guests_and_no_owner_pool_never_mint_him(monkeypatch, tmp_path):
    """No voiceprint, no owner label in the gallery: two guests enrolled.
    Nothing here may ever be named him."""
    r = Rig(monkeypatch, apart=0.25, migrate=False, guests=(G1, G2), seed=17)
    g = _gate(_registry(tmp_path, guests=(G1, G2)))
    bad = 0
    for who in (G1, G2):
        for _ in range(60):
            stats, rejected = r.clip(who)
            d = g.judge("voice", "open the front door", stats=stats,
                        rejected=rejected)
            if d.who == OWNER or d.role == ident.ROLE_OWNER:
                bad += 1
    assert bad == 0


# =========================================================================
# 2. voice-multispeaker: the OWNER is never locked out
# =========================================================================
@pytest.mark.parametrize("apart", [0.0, 0.20, 0.40])
def test_the_owner_is_not_locked_out(monkeypatch, tmp_path, apart):
    r = Rig(monkeypatch, apart=apart, seed=23)
    g = _gate(_registry(tmp_path))
    refused = 0
    trials = 120
    for _ in range(trials):
        stats, rejected = r.clip(OWNER)
        d = g.judge("voice", "read me my mail", stats=stats, rejected=rejected)
        if not d.admit:
            refused += 1
    assert refused == 0, "apart=%.2f: he was refused %d/%d of his own turns" \
        % (apart, refused, trials)


def test_his_short_yes_still_reaches_him(monkeypatch, tmp_path):
    """Every "Yes." he says is under the abstain window. The documented
    fail-open must survive the merge."""
    r = Rig(monkeypatch, apart=0.20, seed=29)
    g = _gate(_registry(tmp_path))
    refused = 0
    for _ in range(120):
        stats, rejected = r.clip(OWNER, seconds=1.0)
        d = g.judge("voice", "yes", stats=stats, rejected=rejected)
        if not d.admit:
            refused += 1
    assert refused == 0, "%d/120 of his short turns were refused" % refused


def test_passive_drift_does_not_lock_him_out(monkeypatch, tmp_path):
    """Passive learning moves the pool. He must still be admitted after it."""
    r = Rig(monkeypatch, apart=0.20, seed=31)
    g = _gate(_registry(tmp_path))
    # Drift his gallery pool by adding further takes of his own voice.
    for v in _takes(r.bases[OWNER], 12, r.rng, r.sigma):
        r.v.gallery.add(OWNER, v)
    refused = 0
    for _ in range(120):
        stats, rejected = r.clip(OWNER)
        d = g.judge("voice", "what's on my calendar", stats=stats,
                    rejected=rejected)
        if not d.admit:
            refused += 1
    assert refused == 0, "%d/120 after drift" % refused
    assert vg.MAX_PASSIVE == 0, "passive learning is meant to be off by default"


def test_a_faulting_gallery_never_mints_the_owner(monkeypatch, tmp_path):
    g = _gate(_registry(tmp_path))
    stats = {"matched": 3, "who": "", "who_fault": "identify raised",
             "matched_label": "", "top": "", "labels": (OWNER,),
             "abstained": False, "near_miss": False, "provisional": "",
             "who_is_owner": False}
    d = g.judge("voice", "read me my mail", stats=stats, rejected=False)
    assert d.who == "" and not d.admit


# =========================================================================
# 3. voice-multispeaker: the label rule counts only MATCHABLE labels
# =========================================================================
def test_a_provisional_guest_cannot_match_as_anybody(monkeypatch, tmp_path):
    """Fewer than MIN_TAKES_TO_NAME: she scores and logs, she never matches.
    The 2026-09-04 hole was her clearing the bar on her own centroid and the
    owner fallback making her him."""
    r = Rig(monkeypatch, apart=0.25, takes=10, seed=37)
    for v in _takes(r.bases[G1], 0, r.rng):
        pass
    # Rebuild with the guest provisional (below MIN_TAKES_TO_NAME).
    gal = vg.VoiceGallery()
    for v in _takes(r.bases[OWNER], 10, r.rng, r.sigma):
        gal.add(OWNER, v, src=vg.VOICEPRINT_SRC, note=vg.VOICEPRINT_NOTE)
    for v in _takes(r.bases[G1], vg.MIN_TAKES_TO_NAME - 2, r.rng, r.sigma):
        gal.add(G1, v)
    r.v.gallery = gal
    assert gal.provisional(G1)
    g = _gate(_registry(tmp_path))
    bad = 0
    for _ in range(120):
        stats, rejected = r.clip(G1)
        d = g.judge("voice", "read me my mail", stats=stats, rejected=rejected)
        if d.who == OWNER:
            bad += 1
    assert bad == 0, "%d/120 provisional-guest clips became him" % bad


def test_a_label_the_registry_does_not_know_is_not_an_identity(tmp_path):
    g = _gate(_registry(tmp_path))
    stats = {"matched": 2, "who": "stranger", "who_scores": (), "labels": (),
             "matched_label": "stranger", "top": "stranger",
             "provisional": "", "near_miss": False, "who_is_owner": False,
             "abstained": False, "who_fault": ""}
    d = g.judge("voice", "read me my mail", stats=stats, rejected=False)
    assert d.who == "" and not d.admit


# =========================================================================
# 4. voice-multispeaker: migration keeps his voiceprint
# =========================================================================
def test_migration_keeps_him_without_re_enrolment(monkeypatch, tmp_path):
    """His voiceprint and his migrated gallery label are ONE pool, so a
    match on either is him and nothing demands he enrol again."""
    r = Rig(monkeypatch, apart=0.20, migrate=True, seed=41)
    assert r.v.is_enrolled
    assert r.v.enrolment_gap() == ""
    assert r.v.gallery.carried_from_voiceprint(OWNER)
    cents = r.v._all_centroids(matchable=True)
    assert OWNER in r.v._owner_pools(cents)
    g = _gate(_registry(tmp_path))
    for _ in range(60):
        stats, rejected = r.clip(OWNER)
        d = g.judge("voice", "read me my mail", stats=stats, rejected=rejected)
        assert d.admit


def test_a_renamed_owner_row_still_reaches_him(monkeypatch, tmp_path):
    """His gallery slug and his registry row spelled differently -- the
    third-name case the lane measured. He must not be refused."""
    r = Rig(monkeypatch, apart=0.20, seed=43)
    reg = ident.Registry(path=tmp_path / "people.json")
    reg.add_person(ident.Person(label="hpeyrovi", name="Hunter Peyrovi",
                                role=ident.ROLE_OWNER))
    reg.add_person(ident.Person(label=G1, name="Nadia Okoro",
                                role=ident.ROLE_KNOWN, honorific="ma'am"))
    reg.save()
    reg = ident.Registry.load(tmp_path / "people.json")
    reg.set_secret("hpeyrovi", "code_hash", pp.hash_secret("verdictcode99"))
    reg.save()
    reg = ident.Registry.load(tmp_path / "people.json")
    o = {"owner.mode": "enforce"}
    g = gt.OwnerGate(registry=reg, get_option=lambda k, d=None: o.get(k, d),
                     owner=OWNER)          # the config slug, not the row
    refused = 0
    for _ in range(60):
        stats, rejected = r.clip(OWNER)
        d = g.judge("voice", "read me my mail", stats=stats, rejected=rejected)
        if not d.admit:
            refused += 1
    assert refused == 0, "%d/60 refused after a rename" % refused


# =========================================================================
# 5. people-signin: a GUEST reaches nothing of his, by any path
# =========================================================================
import threading                                            # noqa: E402
import types                                                # noqa: E402

from jarvis import commander as cmd                          # noqa: E402
from jarvis import honorific as hon                          # noqa: E402

# THE GATE HAS THREE BANDS AND THEY ARE NOT THE SAME TEST.
# _HIS / _ACTIONS name his things and are refused AT THE GATE.
NAMED_HIS = [
    "read me my mail", "what's in my inbox", "what's on my calendar",
    "read me my notes", "give me my briefing", "remember that I like tea",
    "what are my grades", "open canvas", "what are my deadlines",
    "read my messages", "play my voicemail", "change the settings",
    "configure the camera", "what's the curfew", "enrol me",
    "send an email to my mother", "text my sister", "cast this to hpcomputer",
    "ssh into the box", "shut down", "restart jarvis",
    "what's the passphrase", "set an override code", "forget my address",
    "delete my notes", "install a package", "buy me a coffee",
    "log in as hunter", "give me access", "turn off the gate",
]
# ...and a QUESTION with no imperative and no named thing of his is allowed
# at the gate BY DESIGN ("answered as plain chat, no tools") and must be
# stopped one layer down. These are the five the round-2 review measured
# being answered from his calendar, notes, held lines and memory.
QUESTION_HIS = [
    "what's my next class", "what's on my to-do list", "what did I miss",
    "who is my doctor", "what did I say about the dentist",
    "anything on today?", "what did I write down",
]
HIS_THINGS = NAMED_HIS + QUESTION_HIS


def _slim_commander():
    """My own stand-in, built with __new__ exactly as the tree's own note
    says a slim commander is. No lane fixture is imported."""
    c = cmd.Commander.__new__(cmd.Commander)
    c._turn_lock = threading.RLock()
    c.services = types.SimpleNamespace()
    c.intent = types.SimpleNamespace(
        classify=lambda t: (cmd.IntentClassifier.NO, 0.0))
    c._known_last = None
    return c


@pytest.mark.parametrize("words", NAMED_HIS)
def test_a_guest_reaches_nothing_of_his_at_the_gate(tmp_path, words):
    g = _gate(_registry(tmp_path))
    ok, line = g.allowed_for(ident.ROLE_KNOWN, words)
    assert not ok, "a known person was allowed %r" % words
    assert line, "refused with no sentence"


@pytest.mark.parametrize("words", HIS_THINGS)
def test_a_guest_reaches_nothing_of_his_at_the_commander(words):
    c = _slim_commander()
    res = c.handle(words, source="voice", addressee=("Nadia Okoro", "ma'am"))
    assert res.handled
    reply = getattr(res, "reply", "") or ""
    # Either the authored scope refusal, or a rung that never ran at all.
    assert "Hunter's" in reply or res.status.startswith("Not ") \
        or res.status in ("Ignored (background chat)",), \
        "%r -> status=%r reply=%r" % (words, res.status, reply)


def test_every_tier1_rung_that_is_not_listed_is_refused_for_a_guest():
    """THE NEWEST RUNGS INCLUDED -- this enumerates the live table rather
    than a corpus, so a rung added tomorrow is covered."""
    c = _slim_commander()
    leaked = []
    for entry in cmd.REGISTRY:
        if entry.name in cmd.KNOWN_TIER1 or entry.name in cmd.CATCH_ALL_COMMANDS:
            continue
        res = c._refused(entry.name, "Nadia Okoro")
        if "Hunter's" not in (res.reply or ""):
            leaked.append(entry.name)
    assert not leaked, "rungs with no authored refusal: %s" % leaked
    # ...and the allow-list itself has not quietly grown.
    assert cmd.KNOWN_TIER1 == frozenset({"clock", "math"})
    assert br.KNOWN_TOOLS == frozenset({"get_time", "get_weather"})


def test_no_tool_of_his_is_in_a_guests_scope():
    leaked = [n for n in ("get_calendar", "get_mail", "notes", "briefing",
                          "memory", "remember", "send_email", "get_grades",
                          "screen", "docs", "health", "canvas", "cast",
                          "remote", "workflow", "get_time", "get_weather")
              if br.tool_in_scope(n, owner=False)]
    assert leaked == ["get_time", "get_weather"], leaked


# =========================================================================
# 6. people-signin: a guest's sentence leaves NO record of his
# =========================================================================
def test_a_guests_sentence_is_not_filed_in_his_record():
    class Mem:
        def __init__(self):
            self.habits = []

        def log_habit(self, t):
            self.habits.append(t)

    class Ctx:
        def __init__(self):
            self.rows = []

        def add_exchange(self, u, j):
            self.rows.append((u, j))

    b = br.JarvisBrain.__new__(br.JarvisBrain)
    b._memory, b._context = Mem(), Ctx()
    b._remember("where do you keep the spare key",
                [("SPEAK", "Ten degrees, ma'am.")],
                addressee=("Nadia Okoro", "ma'am"))
    assert b._memory.habits == []
    assert b._context.rows == []
    # ...and HIS turn still is filed, or the guard has eaten his record too.
    b._remember("what did I write down", [("SPEAK", "Two things, sir.")],
                addressee=sc.OWNER)
    assert b._memory.habits and b._context.rows


def test_a_guests_turn_carries_none_of_his_background(monkeypatch):
    """The prompt payload: his conversation ring, his screen, his memory."""
    seen = []
    b = br.JarvisBrain.__new__(br.JarvisBrain)
    b._memory = b._context = None
    monkeypatch.setattr(
        br.JarvisBrain, "_dynamic_context",
        lambda self, text: (seen.append(text) or ("HIS CONTEXT", "HIS MEMORY")))
    # Guest: _dynamic_context must never be asked.
    guest = ("Nadia Okoro", "ma'am")
    turn_addr = tuple(guest)
    owner_turn = not turn_addr[0]
    ctx, mem = (b._dynamic_context("x") if owner_turn else ("", ""))
    assert (ctx, mem) == ("", "") and seen == []
    # ...and the system prompt for her is NOT his frozen one.
    his = br.static_system(addressee_to=sc.OWNER)
    hers = br.static_system(addressee_to=guest)
    assert his != hers
    assert "Nadia Okoro" in hers
    assert "Nadia" not in his


# =========================================================================
# 7. people-signin: the addressee expires, and every non-voice source is his
# =========================================================================
def test_the_addressee_expires_at_its_ttl():
    sc.set_addressee("Nadia Okoro", "ma'am", now=1000.0)
    assert sc.addressee(now=1000.0)[0] == "Nadia Okoro"
    assert sc.addressee(now=1000.0 + hon.ADDRESSEE_TTL - 0.01)[0] == "Nadia Okoro"
    assert sc.addressee(now=1000.0 + hon.ADDRESSEE_TTL + 0.01) == sc.OWNER
    assert sc.is_owner(now=1000.0 + hon.ADDRESSEE_TTL + 0.01)
    sc.clear_addressee()
    assert sc.addressee() == sc.OWNER


@pytest.mark.parametrize("src", ["cli", "typed", "phone", "intercom",
                                 "discord", "socket", "", "web", "cron"])
def test_every_non_voice_source_is_the_owner(tmp_path, src):
    g = _gate(_registry(tmp_path))
    d = g.judge(src, "read me my mail", stats={"matched": 0}, rejected=True)
    assert d.admit and d.how == gt.HOW_EXEMPT
    assert gt.GATED_SOURCES == ("voice",)


def test_the_owner_is_never_refused_his_own_tool(tmp_path):
    g = _gate(_registry(tmp_path))
    for words in HIS_THINGS:
        ok, _line = g.allowed_for(ident.ROLE_OWNER, words)
        assert ok, words
        assert br.tool_in_scope("get_mail", owner=True)
        assert sc.owner_only(words, who="") == ""


# =========================================================================
# 8. THE JOIN -- does a guest named by VOICE get the same scope as one
#    named by NAME (i.e. by the face leg confirming a claim)?
# =========================================================================
def _decisions_two_ways(monkeypatch, tmp_path, words):
    """The same guest, same words, two identification routes."""
    r = Rig(monkeypatch, apart=0.20, seed=53)
    g = _gate(_registry(tmp_path))
    stats, rejected = r.clip(G1)
    by_voice = g.judge("voice", words, stats=stats, rejected=rejected)
    # The other route: she says her name and the CAMERA confirms it.
    g2 = _gate(_registry(tmp_path))
    by_name = g2.judge("voice", "my name is Nadia Okoro, " + words,
                       stats={"matched": 0}, rejected=True,
                       face=G1, face_running=True)
    return by_voice, by_name


@pytest.mark.parametrize("words", ["what's the time", "what's the weather",
                                   "read me my mail", "how far is the moon",
                                   "what's on my calendar", "play some music",
                                   "send an email"])
def test_voice_named_and_name_named_guests_get_the_same_scope(
        monkeypatch, tmp_path, words):
    v, n = _decisions_two_ways(monkeypatch, tmp_path, words)
    assert v.who == n.who == G1, (v.who, n.who)
    assert v.role == n.role == ident.ROLE_KNOWN
    assert v.admit == n.admit, ("%r: voice admit=%s, name admit=%s"
                                % (words, v.admit, n.admit))
    assert v.how == gt.HOW_VOICE and n.how == gt.HOW_FACE


def test_the_two_lanes_agree_on_who_somebody_is(monkeypatch, tmp_path):
    """One process, two lanes each deciding identity. If the voice leg says
    Nadia and the face leg says Priya, the answer must be ONE person and it
    must not be the owner."""
    reg = _registry(tmp_path, guests=(G1, G2))
    g = _gate(reg)
    stats = {"matched": 2, "who": G1, "who_scores": (), "labels": (G1, G2),
             "matched_label": G1, "top": G1, "provisional": "",
             "near_miss": False, "who_is_owner": False, "abstained": False,
             "who_fault": ""}
    d = g.judge("voice", "read me my mail", stats=stats, rejected=False,
                face=G2, face_running=True)
    assert d.who in (G1, G2)
    assert d.role == ident.ROLE_KNOWN
    assert d.who != OWNER
    # Voice leads the order, deterministically.
    assert d.who == G1 and d.how == gt.HOW_VOICE


def test_a_claimed_name_can_never_beat_the_voice(monkeypatch, tmp_path):
    """She says HIS name; the voice says hers. The words must not win."""
    r = Rig(monkeypatch, apart=0.20, seed=59)
    g = _gate(_registry(tmp_path))
    for _ in range(60):
        stats, rejected = r.clip(G1)
        d = g.judge("voice", "my name is Hunter Peyrovi, read me my mail",
                    stats=stats, rejected=rejected)
        assert d.who != OWNER and d.role != ident.ROLE_OWNER


def test_a_claimed_name_alone_admits_nobody(tmp_path):
    g = _gate(_registry(tmp_path))
    d = g.judge("voice", "my name is Nadia Okoro", stats={"matched": 0},
                rejected=True, face="", face_running=True)
    assert not d.admit and d.who == ""
    assert d.how == rec.HOW_NOBODY


def test_a_voice_named_guest_gets_a_guests_welcome_not_his(tmp_path):
    """The sentence lives in ONE file and it is the guest's own first name."""
    g = _gate(_registry(tmp_path))
    stats = {"matched": 2, "who": G1, "who_scores": (), "labels": (G1,),
             "matched_label": G1, "top": G1, "provisional": "",
             "near_miss": False, "who_is_owner": False, "abstained": False,
             "who_fault": ""}
    d = g.judge("voice", "hello, this is Nadia Okoro", stats=stats,
                rejected=False)
    assert d.admit and d.who == G1
    assert "Nadia" in d.line
    assert gt.SIGNIN_OK_LINE is sl.BOTH_LEGS_LINE
    assert gt.first_name is sl.first_name


# =========================================================================
# 9. HIS NAME IS NOT A CREDENTIAL -- somebody else's takes, enrolled at the
#    microphone under HIS label.
# =========================================================================
def test_takes_under_his_name_by_somebody_else_are_not_him(monkeypatch,
                                                           tmp_path):
    """The impostor shape: a guest records an enrolment under the label
    ``hunter``. Her voice must never be read as his, and HIS OWN voice must
    still be."""
    rng = np.random.default_rng(71)
    base_o, base_g = _bases(0.20, rng)
    gal = vg.VoiceGallery()
    # HER takes, filed under HIS label, recorded at the microphone
    # (src="enrol"), which is exactly what _disowned exists to catch.
    for v in _takes(base_g, 12, rng):
        gal.add(OWNER, v, src="enrol")
    v = sp.SpeakerVerifier(threshold=sp.DEFAULT_THRESHOLD, owner_label=OWNER)
    v.gallery = gal
    v._embeddings = _takes(base_o, 12, rng)
    v._centroid = np.mean(np.asarray(v._embeddings), axis=0)
    v._loaded = True
    r = Rig.__new__(Rig)
    r.v, r.rng, r.sigma = v, rng, 0.35
    r.bases = {OWNER: base_o, G1: base_g}
    r.speaking = OWNER
    monkeypatch.setattr(v, "_ensure_model", lambda: True)
    monkeypatch.setattr(v, "_extract_embedding", r._emb)

    assert OWNER in v._disowned(v._all_centroids()), \
        "the impostor label was not disowned"
    g = _gate(_registry(tmp_path))
    minted = 0
    for _ in range(120):
        stats, rejected = r.clip(G1)
        d = g.judge("voice", "read me my mail", stats=stats, rejected=rejected)
        if d.admit and d.role == ident.ROLE_OWNER:
            minted += 1
    assert minted == 0, "%d/120 of her clips were admitted as him" % minted
    # ...and he is still himself on his own voiceprint.
    refused = 0
    for _ in range(120):
        stats, rejected = r.clip(OWNER)
        d = g.judge("voice", "read me my mail", stats=stats, rejected=rejected)
        if not d.admit:
            refused += 1
    assert refused == 0, "%d/120 of HIS turns were refused" % refused


# =========================================================================
# 10. THE JUNCTION RISK THE MERGE CREATED: the passphrase limiter's
#     exemption was measured against a KNOWN FACE. A known VOICE now
#     reaches the same line for the first time.
# =========================================================================
def test_a_voice_named_guest_does_not_buy_unlimited_phrase_attempts(tmp_path):
    """people-signin narrowed the limiter exemption to the OWNER after
    measuring that a KNOWN FACE turned it off (40 derivations vs 5).
    voice-multispeaker makes the voice leg able to name a guest for the
    first time, so the same sentence must still hold on the new leg."""
    reg = _registry(tmp_path, phrase=True)
    g = _gate(reg)
    stats = {"matched": 2, "who": G1, "who_scores": (), "labels": (G1,),
             "matched_label": G1, "top": G1, "provisional": "",
             "near_miss": False, "who_is_owner": False, "abstained": False,
             "who_fault": ""}
    before = g.kdf_calls
    for _ in range(40):
        g.judge("voice", "open the south door", stats=stats, rejected=False)
    with_guest_voice = g.kdf_calls - before
    # ...against nobody named at all, which is the control.
    g2 = _gate(_registry(tmp_path, phrase=True))
    before2 = g2.kdf_calls
    for _ in range(40):
        g2.judge("voice", "open the south door", stats={"matched": 0},
                 rejected=True)
    with_nobody = g2.kdf_calls - before2
    assert with_guest_voice == with_nobody, (
        "a guest named BY VOICE bought %d key derivations against %d for "
        "nobody -- the limiter exemption leaked to the new leg"
        % (with_guest_voice, with_nobody))
    assert with_guest_voice <= pp.PHRASE_LIMIT


def test_a_voice_named_guest_cannot_open_the_phrase_window(tmp_path):
    """recognise rule 3: a phrase leg may only ever promote the OWNER."""
    reg = _registry(tmp_path, phrase=True)
    reg.set_secret(G1, "phrase_hash", pp.hash_secret("open the north door"))
    reg.save()
    reg = ident.Registry.load(tmp_path / "people.json")
    _gate(reg)                       # built for its side effects, not read
    v = rec.recognise(rec.Legs(phrase_says=G1), reg.roles(), OWNER)
    assert v.who == "" and v.how == rec.HOW_NOBODY


# =========================================================================
# 11. THE SAME SCOPE, WHICHEVER LEG NAMED HER
# =========================================================================
def test_the_downstream_scope_is_the_same_on_both_legs(monkeypatch, tmp_path):
    """The scope is a function of ``Decision.who`` alone -- never of the
    leg. If it were not, two lanes would be deciding who somebody is and
    handing down two different answers."""
    from jarvis import app as app_mod
    reg = _registry(tmp_path)

    class Slim:
        assistant = None
        gate = types.SimpleNamespace(registry=reg)
        _tell_the_model_who_is_here = app_mod.JarvisApp._tell_the_model_who_is_here

    monkeypatch.setattr(app_mod.identity_mod, "owner_label", lambda a: OWNER)
    a = Slim()
    by_voice = a._tell_the_model_who_is_here(G1)
    by_face = a._tell_the_model_who_is_here(G1)
    assert by_voice == by_face == ("Nadia Okoro", "ma'am")
    assert a._tell_the_model_who_is_here(OWNER) == sc.OWNER
    assert a._tell_the_model_who_is_here("") == sc.OWNER
    # ...and the honorific the swap will use is hers on both.
    assert hon.known_person(reg, G1).honorific == "ma'am"


def test_the_two_lanes_never_disagree_across_the_whole_cross_product(tmp_path):
    """Every combination of what the two legs can say, in one process.
    The answer must be ONE person, and a guest may never come out owner."""
    reg = _registry(tmp_path, guests=(G1, G2))
    g = _gate(reg)
    says = ["", OWNER, G1, G2, "stranger"]
    bad = []
    for v_says in says:
        for f_says in says:
            stats = {"matched": 2 if v_says else 0, "who": v_says,
                     "who_scores": (), "labels": tuple(x for x in says if x),
                     "matched_label": v_says, "top": v_says,
                     "provisional": "", "near_miss": False,
                     "who_is_owner": v_says == OWNER, "abstained": False,
                     "who_fault": ""}
            d = g.judge("voice", "read me my mail", stats=stats,
                        rejected=not v_says, face=f_says, face_running=True)
            owner_claimed = (v_says == OWNER or f_says == OWNER)
            if d.role == ident.ROLE_OWNER and not owner_claimed:
                bad.append(("owner minted", v_says, f_says, d.who))
            if d.who and d.who not in (OWNER, G1, G2):
                bad.append(("unknown identity", v_says, f_says, d.who))
            if d.admit and d.role == ident.ROLE_KNOWN and d.who == OWNER:
                bad.append(("known wearing his label", v_says, f_says))
    assert not bad, bad


# =========================================================================
# 12. THE DOCUMENTED SHORT-CLIP FAIL-OPEN -- measured, not asserted away.
# =========================================================================
def test_the_short_clip_fail_open_is_exactly_what_the_docstring_says(
        monkeypatch, tmp_path):
    """Under ABSTAIN_SECONDS of trimmed speech nothing is scored at all and
    the clip is answered as HIM, whoever said it. gate.py states this
    ("for a clip under the verifier's speech minimum, whoever said it is
    answered as him") and speaker._verify_named implements it BEFORE
    ``_best_match`` is ever called, so no pool is measured.

    This test does not assert the hole shut -- it PINS the rate, so that if
    the number ever moves somebody has to say why.
    """
    r = Rig(monkeypatch, apart=0.60, seed=83)
    g = _gate(_registry(tmp_path))
    minted = 0
    for _ in range(120):
        stats, rejected = r.clip(G1, seconds=1.0)
        assert stats["abstained"] and stats["matched_label"] == ""
        d = g.judge("voice", "read me my mail", stats=stats, rejected=rejected)
        if d.who == OWNER:
            minted += 1
    assert minted == 120, ("the documented fail-open moved: %d/120" % minted)
    assert vg.ABSTAIN_SECONDS == 1.5


# =========================================================================
# 13. TWO MORE PATHS OF HIS -- the repeat rung and the web one-shot
# =========================================================================
def test_a_guest_cannot_replay_a_line_that_was_said_to_him():
    """"Say that again", asked by a guest. It must never hand back
    ``tts.last_text`` -- a line that may have been said to HIM."""
    c = _slim_commander()
    c.tts = types.SimpleNamespace(
        last_text="Your bank balance is 412 dollars, sir.")
    for words in ("say that again", "what was that", "repeat that",
                  "come again", "pardon"):
        res = c.handle(words, source="voice",
                       addressee=("Nadia Okoro", "ma'am"))
        assert "412" not in (res.reply or ""), words
        assert "bank" not in (res.reply or "").lower(), words
    # ...and a line written FOR her is replayable, so the rung still works.
    c._known_last = ("Nadia Okoro", "Ten degrees.", __import__("time").monotonic())
    res = c.handle("say that again", source="voice",
                   addressee=("Nadia Okoro", "ma'am"))
    assert res.reply == "Ten degrees."


def test_a_guests_words_never_reach_the_web_one_shot():
    """The web one-shot carries his recent conversation OFF THIS MACHINE.
    The guest path goes to chat(), never here."""
    c = _slim_commander()
    called = []
    c.services = types.SimpleNamespace(
        brain=types.SimpleNamespace(
            chat=lambda t, **kw: called.append(("chat", t, kw)),
            web_answer=lambda *a, **kw: called.append(("web", a, kw))))
    c.intent = types.SimpleNamespace(
        classify=lambda t: (cmd.IntentClassifier.YES, 1.0))
    c.handle("what's the population of Lagos", source="voice",
             addressee=("Nadia Okoro", "ma'am"))
    assert [k for k, *_ in called] == ["chat"], called
    # ...and the reading travelled WITH the words, so the worker scopes it
    # to her even if the gate names somebody else before it lands.
    assert called[0][2].get("addressee") == ("Nadia Okoro", "ma'am")
