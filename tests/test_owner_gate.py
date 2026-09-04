"""Admission: who is answered, who is refused, and every way this could
have stopped him talking to his own assistant.

No mic, no lens, no model. The gate is fed the dicts the real pipeline
produces -- including the one that made a short "Yes." look like a score of
zero -- and its own failure is a test case, not an accident.
"""
import logging

import pytest

from jarvis import gate as gt
from jarvis import passphrase as pp
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER, Person, Registry

FAKE_PHRASE = "xxx-not-a-real-phrase-xxx"
FAKE_CODE = "xxx000"


def _registry(tmp_path, *, phrase=False, code=True, known=False):
    # code=True BY DEFAULT since the enforce guard landed: gate._mode_unsafe
    # downgrades enforce to shadow when no owner carries an override code,
    # because a wrong verdict would then have no way back in. A fixture
    # without one is not an enforcing gate, so every test that means to
    # enforce needs it. tests/test_signin.py pins the guard itself.
    r = Registry(path=tmp_path / "people.json")
    r.add_person(Person(label="hunter", name="Hunter", role=ROLE_OWNER,
                        voice=True, honorific="sir"))
    if known:
        r.add_person(Person(label="heather", name="Heather", role=ROLE_KNOWN,
                            first="Heather", last="Vance",
                            honorific="ma'am", face="heather",
                            consent="typed"))
    if phrase:
        r.set_secret("hunter", "phrase_hash",
                     pp.hash_secret(pp.normalise_spoken(FAKE_PHRASE)))
    if code:
        r.set_secret("hunter", "code_hash", pp.hash_secret(FAKE_CODE))
    r.save()
    return r


def _gate(tmp_path, mode="enforce", **kw):
    reg = kw.pop("registry", None)
    if reg is None:
        reg = _registry(tmp_path, **kw)
    opts = {"owner.mode": mode}
    return gt.OwnerGate(registry=reg, owner="hunter",
                        get_option=lambda k, d=None: opts.get(k, d))


# The dict a real filter_segments returns when it matched him.
MATCHED = {"total": 3, "matched": 2, "scores": [0.38, 0.41, 0.12]}
# The dict a real filter_segments returns for a SHORT clip -- verify()
# abstains under ABSTAIN_SECONDS and reports (True, 0.0). A gate reading
# max(scores) instead of matched refuses every follow-up he ever gives.
ABSTAINED = {"total": 1, "matched": 1, "scores": [0.0]}
# What _decode_clip hands over when the verifier is not running at all.
NOT_RUNNING = {}


# --------------------------------------------------- the voice margin (L3/L4)
def test_the_gate_owns_no_voice_threshold(tmp_path):
    """His own scores sit at 0.377-0.397 against a bar already lowered to
    0.30 from measured data. A gate that demands more than the wake word
    already demands locks him out of his own house, so it demands nothing:
    it reuses the verdict filter_segments already reached."""
    import inspect
    src = inspect.getsource(gt)
    bars = []
    for tok in src.replace("(", " ").replace(")", " ").replace(",", " ").split():
        try:
            val = float(tok)
        except ValueError:
            continue
        if 0.0 < val < 1.0:
            bars.append(tok)
    assert bars == [], "gate.py grew a threshold of its own: %r" % (bars,)


def test_a_short_follow_up_is_him_not_a_stranger(tmp_path):
    """THE concrete lockout. scores == [0.0] is an ABSTENTION, not a
    rejection: it is what verify() returns for a clip under 1.5 s of
    speech, and every "Yes." he ever says arrives that way."""
    d = _gate(tmp_path).judge("voice", "Yes.", stats=ABSTAINED)
    assert d.admit is True and d.who == "hunter"


def test_the_gate_reads_matched_and_never_scores_or_best_score(tmp_path):
    """`best_score` is never set by filter_segments -- app.py:4378 and
    intercom.py:241 have always read 0.0 from it. Reading it here would
    refuse him permanently.

    Asked of the SYNTAX TREE, not the text: which keys does this module
    actually look up? The prose is allowed to name the trap it avoids."""
    import ast
    import inspect
    read = set()
    for node in ast.walk(ast.parse(inspect.getsource(gt))):
        if isinstance(node, ast.Subscript) and \
                isinstance(node.slice, ast.Constant):
            read.add(node.slice.value)
        elif isinstance(node, ast.Compare):
            for side in [node.left] + list(node.comparators):
                if isinstance(side, ast.Constant):
                    read.add(side.value)
        elif isinstance(node, ast.Call) and \
                getattr(node.func, "attr", "") == "get":
            for arg in node.args[:1]:
                if isinstance(arg, ast.Constant):
                    read.add(arg.value)
    assert "best_score" not in read
    assert "scores" not in read
    assert "matched" in read


# ------------------------------------------------------- the exempt paths
@pytest.mark.parametrize("source", ["cli", "intercom", "phone", "typed",
                                    "discord", "timer", "anything-new"])
def test_every_source_but_the_microphone_is_exempt_by_construction(
        tmp_path, source):
    """GATED_SOURCES is an ALLOW-LIST, so a source nobody thought of is
    exempt by default rather than gated by accident. The command socket is
    mode 0600 (cmdsock.py:160): the filesystem is already the credential
    there, and gating it would cost him his way in when the mic is dead."""
    assert gt.GATED_SOURCES == ("voice",)
    d = _gate(tmp_path).judge(source, "who is in the house")
    assert d.admit is True and d.how == gt.HOW_EXEMPT


def test_the_socket_still_answers_with_the_registry_deleted(tmp_path):
    g = _gate(tmp_path)
    (tmp_path / "people.json").unlink()
    g.reload()
    assert g.judge("cli", "restart yourself").admit is True


# -------------------------------------------------------- failing open
def test_a_missing_registry_turns_the_gate_off_and_says_so(tmp_path):
    reg = Registry.load(tmp_path / "nothing.json")
    g = gt.OwnerGate(registry=reg, owner="hunter",
                     get_option=lambda k, d=None: "enforce")
    assert g.effective_mode() == gt.MODE_OFF
    d = g.judge("voice", "what time is it", stats=MATCHED)
    assert d.admit is True
    assert "off" in g.startup_line().lower()


def test_nothing_enrolled_leaves_enrolment_reachable(tmp_path):
    """First run must not be a brick: with nothing enrolled the gate is
    OFF, so enrolling the first owner needs no owner."""
    reg = Registry(path=tmp_path / "people.json")
    g = gt.OwnerGate(registry=reg, owner="hunter",
                     get_option=lambda k, d=None: "enforce")
    assert g.effective_mode() == gt.MODE_OFF


def test_with_no_instrument_running_the_gate_stands_itself_down(tmp_path):
    """speaker_verify off and no camera is not "nobody is here", it is
    "nothing is measuring". A gate with no legs must never refuse."""
    g = _gate(tmp_path)
    d = g.judge("voice", "what time is it", stats=NOT_RUNNING,
                face_running=False)
    assert d.admit is True
    assert d.how in (gt.HOW_BLIND, gt.HOW_OFF)


def test_the_gate_failing_is_never_the_reason_he_cannot_speak(tmp_path):
    class Exploding:
        usable = True
        people = []

        def roles(self):
            raise RuntimeError("boom")

        def owners(self):
            raise RuntimeError("boom")
    g = gt.OwnerGate(registry=Exploding(), owner="hunter",
                     get_option=lambda k, d=None: "enforce")
    d = g.judge("voice", "what time is it", stats=MATCHED)
    assert d.admit is True
    assert "the gate failed" in d.why


def test_a_get_option_that_raises_does_not_brick_the_turn(tmp_path):
    def boom(key, default=None):
        raise RuntimeError("no config")
    g = gt.OwnerGate(registry=_registry(tmp_path), owner="hunter",
                     get_option=boom)
    assert g.judge("voice", "hello", stats=MATCHED).admit is True


# ------------------------------------------------------------ shadow mode
def test_shadow_refuses_nobody_and_says_what_it_would_have_done(tmp_path):
    """It ships in shadow: his voiceprint corrupted once this week, and a
    day of logged verdicts before anything is refused is cheap."""
    g = _gate(tmp_path, mode="shadow")
    d = g.judge("voice", "what time is it", stats=MATCHED, rejected=True)
    assert d.admit is True
    assert d.would_refuse is True
    assert d.line == ""


def test_the_default_mode_in_the_config_is_shadow():
    from jarvis.assistant_config import DEFAULTS
    assert DEFAULTS["owner"]["mode"] == "shadow"


# ---------------------------------------------------------- enforcing
def test_an_unrecognised_speaker_gets_a_refusal_that_names_the_way_back(
        tmp_path):
    """Replacing app.py's "I only answer to Hunter, sir." -- which named no
    way back in, and that was the whole complaint."""
    d = _gate(tmp_path, phrase=True).judge(
        "voice", "open the garage door", stats=MATCHED, rejected=True)
    assert d.admit is False
    assert "passphrase" in d.line.lower()
    assert d.how == gt.HOW_NOBODY


def test_the_refusal_never_says_a_name_it_does_not_know(tmp_path):
    d = _gate(tmp_path).judge("voice", "hello", stats=MATCHED, rejected=True)
    assert "hunter" not in d.line.lower()


def test_a_rejected_clip_is_rescued_by_the_face(tmp_path):
    """EITHER LEG SUFFICES, and this is the row that proves it: his voice
    is hoarse, the verifier says no, and the camera says it is him."""
    d = _gate(tmp_path).judge("voice", "read me my mail", stats=MATCHED,
                              rejected=True, face="hunter", face_running=True)
    assert d.admit is True and d.how == gt.HOW_FACE and d.who == "hunter"


def test_a_dark_camera_is_never_evidence_against_him(tmp_path):
    d = _gate(tmp_path).judge("voice", "what time is it", stats=MATCHED,
                              face="", face_running=False)
    assert d.admit is True and d.who == "hunter"


# ------------------------------------------------------------- the roles
def test_a_known_person_is_greeted_by_name_and_kept_out_of_his_things(
        tmp_path):
    g = _gate(tmp_path, known=True)
    for open_one in ("what time is it", "what's the weather",
                     "pause the music", "skip this track"):
        d = g.judge("voice", open_one, stats=MATCHED, rejected=True,
                    face="heather", face_running=True)
        assert d.admit is True, open_one
        assert d.role == ROLE_KNOWN and d.who == "heather"
    for his in ("read me my mail", "what's on my calendar", "run my briefing",
                "set the camera to off", "send that to HPCOMPUTER",
                "what did I write in my notes"):
        d = g.judge("voice", his, stats=MATCHED, rejected=True,
                    face="heather", face_running=True)
        assert d.admit is False, his
        assert "Heather" in d.line


def test_a_known_person_cannot_enrol_or_change_a_role(tmp_path):
    reg = _registry(tmp_path, known=True)
    assert reg.may_administer("heather") is False
    assert reg.may_administer("hunter") is True
    assert reg.may_administer("nobody-at-all") is False


# ------------------------------------------------------ the dead-man switch
def test_three_blind_turns_stand_the_gate_down_out_loud(tmp_path):
    """A gate that has had no opinion for three turns running is broken,
    not besieged, and it should stand down rather than hold the door
    against its owner."""
    g = _gate(tmp_path)
    last = None
    for _ in range(gt.DEADMAN_TURNS):
        last = g.judge("voice", "what time is it", stats=NOT_RUNNING,
                       face_running=False)
    assert g.stood_down is True
    assert last.admit is True
    assert "stood the gate down" in (last.line or g.standdown_line()).lower()


def test_a_persistent_stranger_never_disarms_the_gate(tmp_path):
    """Negatives, never abstentions: someone who keeps being refused must
    not be able to talk the gate into standing down."""
    g = _gate(tmp_path)
    for _ in range(gt.DEADMAN_TURNS * 3):
        d = g.judge("voice", "let me in", stats=MATCHED, rejected=True)
        assert d.admit is False
    assert g.stood_down is False


def test_one_good_turn_resets_the_dead_man(tmp_path):
    g = _gate(tmp_path)
    g.judge("voice", "hello", stats=NOT_RUNNING)
    g.judge("voice", "hello", stats=NOT_RUNNING)
    g.judge("voice", "hello", stats=MATCHED)
    g.judge("voice", "hello", stats=NOT_RUNNING)
    assert g.stood_down is False


# ------------------------------------------------------- the passphrase
def test_the_passphrase_works_with_the_camera_off_and_the_gallery_empty(
        tmp_path):
    g = _gate(tmp_path, phrase=True)
    d = g.judge("voice", "Xxx not a real phrase xxx.", stats=MATCHED,
                rejected=True)
    assert d.admit is True and d.how == gt.HOW_PHRASE and d.who == "hunter"


def test_the_passphrase_is_swapped_out_before_anything_can_publish_it(
        tmp_path):
    """The transcript reaches the bus, the history, the transcript pane,
    the turn ledger and commander's own log line. The gate hands back what
    to publish instead."""
    d = _gate(tmp_path, phrase=True).judge(
        "voice", "Xxx not a real phrase xxx.", stats=MATCHED, rejected=True)
    assert d.redact and FAKE_PHRASE not in d.redact
    assert d.redact == gt.REDACTED_TEXT


def test_a_near_miss_is_indistinguishable_from_an_unrelated_sentence(
        tmp_path):
    g = _gate(tmp_path, phrase=True)
    near = g.judge("voice", "xxx not a real phrase yyy", stats=MATCHED,
                   rejected=True)
    other = g.judge("voice", "the quick brown fox jumped over", stats=MATCHED,
                    rejected=True)
    assert (near.admit, near.who, near.role, near.how, near.line) == \
           (other.admit, other.who, other.role, other.how, other.line)


def test_the_key_derivation_stays_off_the_turn_budget(tmp_path):
    """It runs only when recognition already failed AND the words were
    phrase-shaped. An ordinary refused sentence must not pay 20 ms."""
    g = _gate(tmp_path, phrase=True)
    g.judge("voice", "no", stats=MATCHED, rejected=True)
    assert g.kdf_calls == 0
    g.judge("voice", "xxx not a real phrase yyy", stats=MATCHED, rejected=True)
    assert g.kdf_calls == 1
    g.judge("voice", "what time is it", stats=MATCHED)   # he was recognised
    assert g.kdf_calls == 1


def test_the_passphrase_limit_is_a_cool_off_with_a_stated_number(tmp_path):
    g = _gate(tmp_path, phrase=True)
    assert (pp.PHRASE_LIMIT, pp.PHRASE_WINDOW) == (5, 300.0)
    for i in range(pp.PHRASE_LIMIT + 2):
        g.judge("voice", "xxx not a real phrase yyy", stats=MATCHED,
                rejected=True, now=float(i))
    d = g.judge("voice", "Xxx not a real phrase xxx.", stats=MATCHED,
                rejected=True, now=10.0)
    assert d.admit is False
    d = g.judge("voice", "Xxx not a real phrase xxx.", stats=MATCHED,
                rejected=True, now=10.0 + pp.PHRASE_WINDOW + 1)
    assert d.admit is True


# ------------------------------------------------------- the override code
def test_the_override_code_does_nothing_at_all_over_the_microphone(tmp_path):
    """KEYBOARD/SSH ONLY, by construction: it must not be overhearable or
    replayable, so speaking it is worth exactly as much as any other
    sentence -- and not one bit more, not even a hint that it was close."""
    g = _gate(tmp_path, code=True, phrase=True)
    spoken = g.judge("voice", FAKE_CODE, stats=MATCHED, rejected=True)
    unrelated = g.judge("voice", "sausages", stats=MATCHED, rejected=True)
    assert spoken.admit is False
    assert (spoken.admit, spoken.who, spoken.role, spoken.how, spoken.line,
            spoken.why) == (unrelated.admit, unrelated.who, unrelated.role,
                            unrelated.how, unrelated.line, unrelated.why)
    assert g.kdf_calls == 0


def test_the_code_is_checked_only_where_a_keyboard_is(tmp_path):
    reg = _registry(tmp_path, code=True)
    who, why = gt.check_override_code(reg, FAKE_CODE)
    assert who == "hunter", why
    assert gt.check_override_code(reg, "xxx999")[0] == ""
    assert gt.check_override_code(reg, "")[0] == ""


def test_burning_the_passphrase_leaves_the_break_glass_open(tmp_path):
    """The fallback of last resort must not fail exactly when it is
    needed."""
    g = _gate(tmp_path, phrase=True, code=True)
    for i in range(pp.PHRASE_LIMIT + 3):
        g.judge("voice", "xxx not a real phrase yyy", stats=MATCHED,
                rejected=True, now=float(i))
    assert g.judge("voice", "Xxx not a real phrase xxx.", stats=MATCHED,
                   rejected=True, now=9.0).admit is False
    assert gt.check_override_code(g.registry, FAKE_CODE,
                                  attempts=g.code_attempts)[0] == "hunter"


def test_a_registry_with_no_code_set_is_not_a_way_in(tmp_path):
    assert gt.check_override_code(_registry(tmp_path, code=False), "")[0] == ""
    assert gt.check_override_code(_registry(tmp_path, code=False),
                                  FAKE_CODE)[0] == ""


# ------------------------------------------------------------- no leaks
def test_no_secret_reaches_a_log(caplog, tmp_path):
    g = _gate(tmp_path, phrase=True, code=True)
    with caplog.at_level(logging.DEBUG):
        g.judge("voice", "Xxx not a real phrase xxx.", stats=MATCHED,
                rejected=True)
        gt.check_override_code(g.registry, FAKE_CODE)
    assert FAKE_PHRASE.lower() not in caplog.text.lower()
    assert "xxx not a real phrase" not in caplog.text.lower()
    assert FAKE_CODE not in caplog.text


def test_the_registry_file_holds_no_plaintext(tmp_path):
    _registry(tmp_path, phrase=True, code=True)
    raw = (tmp_path / "people.json").read_text()
    assert FAKE_PHRASE not in raw and FAKE_CODE not in raw
    assert "xxx" not in raw.lower()


# ------------------------------------------------------------ the wording
def test_the_three_lines_are_prewarmed_so_a_refusal_is_not_a_silence():
    """An unprewarmed line is a three-second pause -- indistinguishable
    from being ignored, which is the complaint this replaces."""
    src = (__import__("pathlib").Path(gt.__file__).parent / "app.py").read_text()
    assert "gate_mod.PREWARM_LINES" in src
    for line in gt.PREWARM_LINES:
        assert line and isinstance(line, str)


# ------------------------------------------- what the passphrase buys
def test_the_passphrase_opens_the_floor_and_does_not_close_it_at_once(
        tmp_path):
    """Admitting only the turn that CONTAINED the phrase would be a dead
    end of a new shape: the phrase is not a command, so the next thing he
    says is the thing he wanted -- and that clip is rejected exactly as the
    last one was."""
    g = _gate(tmp_path, phrase=True)
    first = g.judge("voice", "Xxx not a real phrase xxx.", stats=MATCHED,
                    rejected=True, now=0.0)
    assert first.admit is True and first.how == gt.HOW_PHRASE
    second = g.judge("voice", "read me my mail", stats=MATCHED,
                     rejected=True, now=5.0)
    assert second.admit is True and second.who == "hunter"
    assert second.how == gt.HOW_GRANT
    assert second.redact == "", "ordinary words must not be redacted"


def test_the_window_the_passphrase_opens_is_five_minutes_and_then_shuts(
        tmp_path):
    """The cost is stated rather than hidden: for GRANT_S seconds anybody
    on the voice path is answered as him."""
    assert gt.GRANT_S == 300.0
    g = _gate(tmp_path, phrase=True)
    g.judge("voice", "Xxx not a real phrase xxx.", stats=MATCHED,
            rejected=True, now=0.0)
    assert g.judge("voice", "hello", stats=MATCHED, rejected=True,
                   now=gt.GRANT_S - 1).admit is True
    assert g.judge("voice", "hello", stats=MATCHED, rejected=True,
                   now=gt.GRANT_S + 1).admit is False


def test_the_window_lives_in_memory_so_a_restart_ends_it(tmp_path):
    g = _gate(tmp_path, phrase=True)
    g.judge("voice", "Xxx not a real phrase xxx.", stats=MATCHED,
            rejected=True, now=0.0)
    fresh = _gate(tmp_path, registry=g.registry)
    assert fresh.judge("voice", "hello", stats=MATCHED, rejected=True,
                       now=1.0).admit is False


def test_judge_never_reaches_the_override_code(monkeypatch, tmp_path):
    """KEYBOARD/SSH ONLY is enforced by the microphone path not having the
    call at all, not by a check inside it."""
    def explode(*a, **k):
        raise AssertionError("the voice path asked about the override code")
    monkeypatch.setattr(gt, "check_override_code", explode)
    g = _gate(tmp_path, code=True, phrase=True)
    for words in (FAKE_CODE, "xxx000 xxx000 xxx000", "let me in please now"):
        g.judge("voice", words, stats=MATCHED, rejected=True)


# --------------------------------- what the face leg may and may not grant
def test_the_face_leg_grants_only_that_persons_own_role(tmp_path):
    """The invariant eye.py writes down -- identity may never GRANT -- is
    deliberately suspended inside this gate, because he said either leg is
    enough. These are the bounds that make it acceptable."""
    g = _gate(tmp_path, known=True)
    # A KNOWN face rescues a dropped clip into KNOWN scope, never his.
    ok = g.judge("voice", "what time is it", stats=MATCHED, rejected=True,
                 face="heather", face_running=True)
    assert ok.admit is True and ok.role == ROLE_KNOWN
    no = g.judge("voice", "read me my mail", stats=MATCHED, rejected=True,
                 face="heather", face_running=True)
    assert no.admit is False


def test_a_face_the_registry_does_not_hold_grants_nothing(tmp_path):
    """A gallery may know a name the registry does not. That is not an
    identity and it must not become one by being the only thing said."""
    d = _gate(tmp_path).judge("voice", "open everything", stats=MATCHED,
                              rejected=True, face="a-stranger",
                              face_running=True)
    assert d.admit is False


def test_the_face_leg_applies_no_bar_of_its_own(tmp_path):
    """eye.Attention.identity is ALREADY past camera.identity_min. Checking
    it again here would be a second copy of a bar -- the same mistake this
    module refuses to make on the voice leg.

    Asked of the code, not the prose: the docstring is allowed to name the
    threshold it declines to re-apply."""
    import ast
    import inspect
    import textwrap
    fn = ast.parse(textwrap.dedent(
        inspect.getsource(gt.OwnerGate._face_leg))).body[0]
    body = fn.body[1:] if isinstance(getattr(fn.body[0], "value", None),
                                     ast.Constant) else fn.body
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant):
                assert sub.value != "camera.identity_min"
                if isinstance(sub.value, float):
                    assert not 0.0 < sub.value < 1.0


def test_the_gate_never_calls_itself_a_lock():
    """Same rule, applied to the module that does the refusing."""
    from tests.test_owner_registry import no_overstatement
    no_overstatement(gt)
