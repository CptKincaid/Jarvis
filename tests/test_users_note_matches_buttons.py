"""THE NOTE AND THE BUTTONS MUST BE THE SAME ANSWER, in every wiring.

DEFECT 3, and it is the second time this exact class has been caught in one
day. He found the first version himself: a foot note claiming face enrolment
"still needs a terminal" while jarvis/enrolrun.py had already made that false
for him. That was fixed by DERIVING the note from the seams. The half fixed
was the other side of the sentence -- the BUTTONS were still drawn from
nothing at all. So on a console where ``face_enrol_start`` never landed (the
app and the window merge in either order, and the window can be newer than
the app) the note said "Enrolling a face still needs a terminal" and the
owner's row drew a button labelled "Enrol my face" two inches above it.

THE FIX IS ONE FUNCTION READ TWICE. ``users_page.console_can`` answers "what
can this console actually do", and both the note and ``_build_actions`` ask
it. A capability that is missing cannot then be drawn, and a capability that
is present cannot be denied.

DEFECT 2 IS THE SAME SHAPE ONE PANEL DOWN. ``forget_warning`` promised "each
is a separate button below" whatever was wired, on a DESTRUCTIVE panel, and
said nothing about the fact that forgetting somebody removes the row those
buttons live on -- so the honest reading of the old panel was "press Forget,
then press the two buttons", which is an instruction that cannot be followed.

These tests need no display: they are about which sentences and which button
labels a given wiring produces. The MEASURED half is in
tests/test_users_enrol_layout.py, at both his geometries.
"""
from __future__ import annotations

import itertools

import pytest

from jarvis.ui import users_page as up

SEAMS = ("people_set_phrase", "people_new_code", "face_enrol_start",
         "voice_enrol_start", "people_purge_face", "people_purge_voice",
         "face_enrol_stop", "voice_enrol_stop")

# What the console offers when it can do a thing, keyed by the capability.
BUTTON_FOR = {"face": "Enrol my face", "voice": "Enrol my voice",
              "phrase": "Change phrase", "code": "Send me a new code",
              "purge_face": "Remove face measurements",
              "purge_voice": "Remove voice pool"}

# The sentence the note carries when the console CANNOT do that thing.
DENIAL_FOR = {"face": up.NOTE_FACE_NO, "voice": up.NOTE_VOICE_NO,
              "phrase": up.NOTE_PHRASE_NO, "code": up.NOTE_CODE_NO}


class Svc:
    def __init__(self, *names):
        for name in names:
            setattr(self, name, lambda *a, **k: (True, "done"))


def _subsets(names):
    for n in range(len(names) + 1):
        for combo in itertools.combinations(names, n):
            yield combo


# ================================== A: one function answers, two callers read
def test_the_console_reports_exactly_the_seams_it_was_handed():
    can = up.console_can(Svc("face_enrol_start", "people_purge_voice"))
    assert can["face"] is True
    assert can["voice"] is False
    assert can["purge_voice"] is True
    assert can["purge_face"] is False


def test_a_console_with_no_services_at_all_can_do_nothing():
    for value in up.console_can(None).values():
        assert value is False


def test_a_seam_that_is_present_but_not_callable_is_not_a_capability():
    """A dataclass field that landed as None is the half-wired case that
    actually happens; it must not read as a button."""
    svc = Svc()
    svc.face_enrol_start = None
    assert up.console_can(svc)["face"] is False


# ======================================= B: the note never contradicts a door
@pytest.mark.parametrize("wired", list(_subsets(
    ("face_enrol_start", "voice_enrol_start"))))
def test_the_note_denies_only_what_the_row_will_not_draw(wired):
    """EVERY WIRING, not just the healthy one. The note may say "still needs
    a terminal" only about a thing the owner's row refuses to offer."""
    svc = Svc(*wired)
    can = up.console_can(svc)
    note = "\n".join(up.cannot_do(svc))
    offered = set(up.owner_buttons(can))
    for cap in ("face", "voice"):
        denied = DENIAL_FOR[cap] in note
        drawn = BUTTON_FOR[cap] in offered
        assert denied != drawn, (
            "with %r wired the note %s %s and the row %s a button for it"
            % (wired, "denies" if denied else "allows", cap,
               "draws" if drawn else "does not draw"))


@pytest.mark.parametrize("wired", list(_subsets(SEAMS)))
def test_no_wiring_at_all_produces_a_note_that_its_own_buttons_contradict(
        wired):
    """The whole cross-product, because the defect is a DISAGREEMENT and a
    disagreement only shows up in the combinations nobody thought to try."""
    svc = Svc(*wired)
    can = up.console_can(svc)
    note = "\n".join(up.cannot_do(svc))
    offered = set(up.owner_buttons(can))
    for cap, sentence in DENIAL_FOR.items():
        if sentence in note:
            assert BUTTON_FOR[cap] not in offered, (
                "%r: the note says a terminal is needed for %s and the row "
                "draws %r" % (wired, cap, BUTTON_FOR[cap]))


def test_the_asymmetry_lines_are_only_shown_where_the_button_exists():
    """"this button does ask for the code" is a lie on a console that draws
    no such button."""
    assert up.ENROL_ASYMMETRY[0] not in "\n".join(
        up.foot_lines(Svc("voice_enrol_start")))
    assert up.ENROL_ASYMMETRY[0] in "\n".join(
        up.foot_lines(Svc("face_enrol_start")))


# ================================= C: forgetting does not promise a phantom
class TestTheForgetPanelPromisesOnlyWhatExists:
    """DEFECT 2. The panel is the last thing he reads before a destructive,
    unrecoverable write, so every sentence on it has to be true of THIS
    console and THIS person."""

    BOTH = dict(gallery=("heather",), voices=("heather",))

    def test_it_does_not_promise_a_button_on_a_console_without_the_seam(self):
        can = up.console_can(None)
        text = "\n".join(up.forget_warning("heather", can=can, **self.BOTH))
        assert "button" not in text.lower(), text

    def test_it_names_the_button_when_the_seam_is_actually_wired(self):
        can = up.console_can(Svc("people_purge_face", "people_purge_voice"))
        text = "\n".join(up.forget_warning("heather", can=can, **self.BOTH))
        assert "button" in text.lower()

    def test_a_half_wired_console_promises_one_button_and_hands_over_the_other(
            self):
        can = up.console_can(Svc("people_purge_face"))
        text = "\n".join(up.forget_warning("heather", can=can, **self.BOTH))
        assert up.forget_voice_command("heather") in text, text

    @pytest.mark.parametrize("wired", list(_subsets(
        ("people_purge_face", "people_purge_voice"))))
    @pytest.mark.parametrize("holds", [("gallery",), ("voices",),
                                       ("gallery", "voices"), ()])
    def test_whatever_survives_always_carries_a_way_to_remove_it(
            self, wired, holds):
        """"Say plainly what is left behind AND HOW TO REMOVE IT." The
        command is the half that still works after the row is gone."""
        can = up.console_can(Svc(*wired))
        kw = {name: ("heather",) for name in holds}
        text = "\n".join(up.forget_warning("heather", can=can, **kw))
        if "gallery" in holds:
            assert up.forget_face_command("heather") in text
        if "voices" in holds:
            assert up.forget_voice_command("heather") in text
        if not holds:
            assert "no face measurements and no voice pool" in text

    def test_it_says_that_forgetting_takes_the_buttons_away(self):
        """THE ORDERING IS THE POINT. Those buttons live on the row, and
        Forget removes the row -- so "press Forget then press them" is an
        instruction that cannot be followed, and the old panel implied it."""
        can = up.console_can(Svc("people_purge_face", "people_purge_voice"))
        text = "\n".join(up.forget_warning("heather", can=can, **self.BOTH))
        low = text.lower()
        assert "before" in low
        assert "row" in low

    def test_a_person_with_nothing_stored_is_not_given_a_command_to_run(self):
        can = up.console_can(Svc("people_purge_face", "people_purge_voice"))
        text = "\n".join(up.forget_warning("heather", can=can))
        assert "voice_enrol.py" not in text
        assert "face_enrol.py" not in text


def test_the_voice_command_is_the_delete_form_of_the_shared_builder():
    """ONE way to delete somebody. This is the string scripts/voice_enrol.py
    documents, built by the seam that already builds it."""
    from jarvis import enrolentry
    assert up.forget_voice_command("heather") == \
        enrolentry.voice_command_line("heather", delete=True)
    assert "--delete" in up.forget_voice_command("heather")


# ================= D: the line the app returns after the row is actually gone
class TestTheForgetLineNamesBothStores:
    """The toast is the LAST thing he sees about that person, and by the time
    he reads it the row and its two buttons no longer exist. It said only
    "their face measurements stay in the gallery until that command is run" --
    silent about a voice pool, which after the voice gallery merged is the
    same half-truth one store over."""

    def _line(self, monkeypatch, faces=(), voices=()):
        from jarvis import app as app_mod
        monkeypatch.setattr(app_mod, "gallery_labels", lambda: tuple(faces))
        monkeypatch.setattr(app_mod, "voice_labels", lambda: tuple(voices))
        return app_mod._survivors_line("heather")

    def test_a_voice_pool_is_named_and_not_left_implied(self, monkeypatch):
        line = self._line(monkeypatch, voices=("heather",))
        assert "voice pool" in line
        assert "terminal" in line

    def test_both_stores_are_named_when_both_hold_her(self, monkeypatch):
        line = self._line(monkeypatch, faces=("heather",),
                          voices=("heather",))
        assert "face measurements" in line and "voice pool" in line

    def test_nothing_left_is_said_plainly_rather_than_omitted(self,
                                                              monkeypatch):
        line = self._line(monkeypatch)
        assert "Nothing of theirs is left" in line

    def test_a_gallery_that_cannot_be_read_is_admitted_not_dropped(
            self, monkeypatch):
        """"Forgotten." with a silent omission is the exact shape of sentence
        this lane exists to stop."""
        from jarvis import app as app_mod

        def boom():
            raise OSError("gallery unreadable")

        monkeypatch.setattr(app_mod, "gallery_labels", boom)
        monkeypatch.setattr(app_mod, "voice_labels", lambda: ())
        line = app_mod._survivors_line("heather")
        assert "could not read" in line
        assert "can't say" in line
