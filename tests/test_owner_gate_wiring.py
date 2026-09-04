"""Where the gate meets the app: one hook, and the passphrase never published.

The methods are exercised as BOUND METHODS on a stand-in, the idiom
tests/test_turn_flow_fixes.py already uses -- no microphone, no lens, no
Whisper, no Tk.

WHY THE HOOK IS WHERE IT IS, and this is the one placement decision that is
not cosmetic. The spoken passphrase arrives as a Whisper transcript.
``_dispatch`` sits downstream of ``bus.publish(UserUtterance(...))``, of
``history.add``, of the transcript pane, of the turn ledger and of
``commander.handle``'s own ``log.info("handle %r source=%s")``. Gating there
would publish and log the phrase in plaintext in at least four places before
the gate had seen a syllable of it. The hook is in ``_process_audio``, and
on the passphrase path the turn ENDS THERE -- the text never reaches any of
them.
"""
from types import SimpleNamespace

import jarvis.app as app_mod
from jarvis import gate as gate_mod
from jarvis import passphrase as pp
from jarvis.identity import ROLE_KNOWN, ROLE_OWNER, Person, Registry

FAKE_PHRASE = "xxx-not-a-real-phrase-xxx"
MATCHED = {"total": 2, "matched": 2, "scores": [0.4, 0.39]}


class FakeTranscriber:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return SimpleNamespace(text=self.text, confidence=-0.2, accepted=True)


def _stand_in(tmp_path, *, mode="enforce", phrase=False, said="hello there",
              known=False):
    reg = Registry(path=tmp_path / "people.json")
    reg.add_person(Person(label="hunter", name="Hunter", role=ROLE_OWNER,
                          voice=True))
    if known:
        reg.add_person(Person(label="heather", name="Heather", role=ROLE_KNOWN,
                              face="heather", consent="typed"))
    if phrase:
        reg.set_secret("hunter", "phrase_hash",
                       pp.hash_secret(pp.normalise_spoken(FAKE_PHRASE)))
    reg.save()
    opts = {"owner.mode": mode, "camera.identity": False}
    a = SimpleNamespace()
    a.spoken = []
    a.abandoned = []
    a.services = None
    a.tts = SimpleNamespace(busy=False)
    a._tts_active = False
    a._last_guest_ts = -1e9
    a._followup_after_speech = False
    a._gate_who = a._gate_how = ""
    a.transcriber = FakeTranscriber(said)
    a.turns = SimpleNamespace(abandon=a.abandoned.append)
    a.get_option = lambda k, d=None: opts.get(k, d)
    a._say = a.spoken.append
    a._music_playing = lambda: False
    a._eye_identity = app_mod.JarvisApp._eye_identity.__get__(a)
    a._face_running = app_mod.JarvisApp._face_running.__get__(a)
    a._refuse_politely = app_mod.JarvisApp._refuse_politely.__get__(a)
    a._gate_rescue = app_mod.JarvisApp._gate_rescue.__get__(a)
    a._gate_rescue_inner = app_mod.JarvisApp._gate_rescue_inner.__get__(a)
    a._gate_admits = app_mod.JarvisApp._gate_admits.__get__(a)
    a._owner_has_phrase = app_mod.JarvisApp._owner_has_phrase.__get__(a)
    a.gate = gate_mod.OwnerGate(registry=reg, owner="hunter",
                                get_option=a.get_option)
    return a


def test_a_rejected_clip_is_still_dropped_when_nothing_can_rescue_it(tmp_path):
    """Today's behaviour is the default: the gate only ever gets to say
    "actually, let it through"."""
    a = _stand_in(tmp_path)
    assert a._gate_rescue(object(), MATCHED, False) is None
    assert a.transcriber.calls == 0, "an ordinary rejected clip pays no decode"


def test_a_rejected_clip_is_refused_out_loud_with_a_way_back_in(tmp_path):
    a = _stand_in(tmp_path, phrase=True)
    a._gate_rescue(object(), MATCHED, False)
    assert a.spoken and "passphrase" in a.spoken[0].lower()


def test_the_refusal_is_said_once_not_twice_in_a_minute(tmp_path):
    """His complaint, verbatim: hearing it twice in a minute. The gate and
    _on_guest share one cooldown and one wording."""
    a = _stand_in(tmp_path, phrase=True)
    for _ in range(4):
        a._gate_rescue(object(), MATCHED, False)
    assert len(a.spoken) == 1


def test_it_stays_quiet_over_music(tmp_path):
    """A "guest" over a bed is far more often HIM, scored down by the music
    under his voice."""
    a = _stand_in(tmp_path, phrase=True)
    a._music_playing = lambda: True
    a._gate_rescue(object(), MATCHED, False)
    assert a.spoken == []


def test_it_stays_quiet_while_jarvis_is_speaking(tmp_path):
    """Under barge-in the listener hears his own voice."""
    a = _stand_in(tmp_path, phrase=True)
    a.tts.busy = True
    a._gate_rescue(object(), MATCHED, False)
    assert a.spoken == []


def test_the_passphrase_ends_the_turn_and_publishes_nothing(tmp_path):
    """The strongest available answer to the leak: on a match the transcript
    reaches nothing at all -- not the bus, not the history, not the pane,
    not the ledger, not commander's log line. The turn is over."""
    a = _stand_in(tmp_path, phrase=True, said="Xxx not a real phrase xxx.")
    assert a._gate_rescue(object(), MATCHED, False) is None
    assert a.spoken == [gate_mod.PHRASE_OK_LINE]
    assert a._followup_after_speech is True
    assert FAKE_PHRASE not in " ".join(a.spoken)


def test_the_phrase_costs_one_decode_and_only_when_it_could_matter(tmp_path):
    a = _stand_in(tmp_path, phrase=True, said="Xxx not a real phrase xxx.")
    a._gate_rescue(object(), MATCHED, False)
    assert a.transcriber.calls == 1
    b = _stand_in(tmp_path, phrase=False)
    b._gate_rescue(object(), MATCHED, False)
    assert b.transcriber.calls == 0


def test_shadow_changes_nothing_at_all_about_a_rejected_clip(tmp_path):
    a = _stand_in(tmp_path, mode="shadow", phrase=True)
    assert a._gate_rescue(object(), MATCHED, False) is None
    assert a.spoken == []
    assert a.transcriber.calls == 0


def test_a_broken_transcriber_leaves_the_clip_exactly_as_it_was(tmp_path):
    a = _stand_in(tmp_path, phrase=True)

    def boom(audio):
        raise RuntimeError("whisper is not loaded")
    a.transcriber.transcribe = boom
    assert a._gate_rescue(object(), MATCHED, False) is None


# ------------------------------------------------------ the admitted path
def test_the_admitted_path_attributes_the_turn(tmp_path):
    a = _stand_in(tmp_path)
    assert a._gate_admits("what time is it", MATCHED) is True
    assert (a._gate_who, a._gate_how) == ("hunter", gate_mod.HOW_VOICE)


def test_a_gate_that_could_not_be_built_admits_everything(tmp_path):
    a = _stand_in(tmp_path)
    a.gate = None
    assert a._gate_admits("anything at all", MATCHED) is True
    assert a._gate_rescue(object(), MATCHED, False) is None


def _watching(a, who):
    """Attach the camera leg the app reads through services.camera_feed.eye,
    the seam _eye_identity already owns. No lens is opened and no frame
    exists: a label and a usable() flag are the whole contract."""
    state = SimpleNamespace(identity=who, usable=lambda **k: True)
    a.services = SimpleNamespace(
        camera_feed=SimpleNamespace(eye=SimpleNamespace(state=lambda: state)))
    a.get_option = lambda k, d=None: {"owner.mode": "enforce",
                                      "camera.identity": True}.get(k, d)
    a.gate.get_option = a.get_option
    return a


def test_a_known_person_is_answered_but_kept_out_of_his_things(tmp_path):
    """The face leg names Heather while the voice leg is not running at all
    -- the shape of a KNOWN person in the room."""
    a = _watching(_stand_in(tmp_path, known=True), "heather")
    assert a._gate_admits("what time is it", {}) is True
    assert (a._gate_who, a._gate_how) == ("heather", gate_mod.HOW_FACE)


def test_a_turn_the_gate_refuses_is_abandoned_in_the_ledger(tmp_path):
    """A refused turn must not sit in turns.jsonl as an unfinished one."""
    a = _watching(_stand_in(tmp_path, known=True), "heather")
    assert a._gate_admits("read me my mail", {}) is False
    assert a.abandoned and a.abandoned[0] == "gate:known"
    assert a.spoken and "Heather" in a.spoken[0]


def test_the_camera_leg_is_off_when_no_feed_is_attached(tmp_path):
    """Nothing on this tree attaches a camera feed, so the face leg is
    unavailable and VOICE CARRIES THE WHOLE THING today. That is a fact to
    state, not a thing to hide behind a default."""
    a = _stand_in(tmp_path)
    assert a._face_running() is False
    assert a._eye_identity() == ""


def test_shadow_does_not_even_let_the_face_leg_rescue_a_clip(tmp_path):
    """The value of shadow is that it is SAFE TO LEAVE ON while the log is
    read, and that is only true if it changes nothing. A face leg that
    started answering clips the speaker filter dropped would be a visible
    change of behaviour he has not asked for yet."""
    a = _watching(_stand_in(tmp_path, mode="shadow"), "hunter")
    a.get_option = lambda k, d=None: {"owner.mode": "shadow",
                                      "camera.identity": True}.get(k, d)
    a.gate.get_option = a.get_option
    assert a._gate_rescue(object(), MATCHED, False) is None
    assert a.transcriber.calls == 0
    assert a.spoken == []


def test_enforcing_the_face_leg_does_rescue_it(tmp_path):
    """...and in enforce it does, which is his "EITHER voice OR face is
    enough" doing the one job that makes the face leg worth having."""
    a = _watching(_stand_in(tmp_path), "hunter")
    out = a._gate_rescue(object(), MATCHED, False)
    assert out is not None and a.transcriber.calls == 1
