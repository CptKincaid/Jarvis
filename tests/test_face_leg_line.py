"""The startup line that says why the camera cannot name anybody.

WHY THIS FILE EXISTS. On 2026-09-05 Hunter was enrolled as the owner with
his real gallery -- ArcFace, 512-D, the model his box actually runs -- and
the startup line told him to RE-ENROL:

    owner-gate: SHADOW -- 1 owner (hunter), voice leg live, face leg
    unavailable (the gallery is 128-D and hunter was enrolled at 512-D;
    re-enrol to bring the face leg back).

Nothing was wrong with his enrolment. ``_face_leg_why`` compared his
``face_dim`` against ``facegallery.EMBED_DIM``, which is a fixed alias for
the SFACE dimension (128) and not the dimension of the model the box is
running. So a CORRECT ArcFace enrolment always read as stale, and the one
thing the line asks for -- re-enrolling -- would not have helped, because
the fresh enrolment would be 512-D too. He had already re-enrolled his face
once this week; being sent to do it again for nothing is the cost of a line
that was never tested.

The comparison belongs against the LIVE backend's ``embed_dim``.
"""
from types import SimpleNamespace

import pytest

from jarvis import facemodels
from jarvis.app import JarvisApp


def _app(face_dim, backend_name):
    """A bare app with one enrolled owner and a chosen face backend."""
    app = object.__new__(JarvisApp)
    person = SimpleNamespace(label="hunter", face="hunter", face_dim=face_dim)
    app.gate = SimpleNamespace(registry=SimpleNamespace(people=[person]))
    opts = {"camera.identity": True, "camera.face_backend": backend_name}
    app.get_option = lambda key, default=None: opts.get(key, default)
    app.assistant = SimpleNamespace(get=lambda k, d=None: opts.get(k, d))
    return app


def test_an_enrolment_at_the_live_models_width_is_not_called_stale():
    """His real case: ArcFace box, ArcFace enrolment. No re-enrol advice."""
    assert facemodels.INSIGHT.embed_dim == 512          # the premise, pinned
    why = _app(512, "insightface")._face_leg_why()
    assert "re-enrol" not in why, why
    assert "512-D" not in why, why


def test_an_sface_enrolment_on_an_arcface_box_still_says_re_enrol():
    """The message must keep working for the case it was written for."""
    why = _app(128, "insightface")._face_leg_why()
    assert "re-enrol" in why
    assert "512-D" in why and "128-D" in why


def test_an_sface_enrolment_on_an_sface_box_is_not_called_stale():
    assert facemodels.OPENCV.embed_dim == 128
    why = _app(128, "opencv")._face_leg_why()
    assert "re-enrol" not in why, why


@pytest.mark.parametrize("backend", ["", None])
def test_the_default_backend_is_resolved_not_assumed(backend):
    """An empty camera.face_backend means the DEFAULT pair, whatever that
    is today -- the line must ask facemodels, not hardcode a width."""
    live = facemodels.backend_for(backend or "").embed_dim
    assert "re-enrol" not in _app(live, backend)._face_leg_why()
    assert "re-enrol" in _app(live + 1, backend)._face_leg_why()


def test_an_unreadable_backend_name_does_not_crash_the_startup_line():
    """_face_leg_why owns a sentence, never an exception: facemodels
    RAISES on an unknown name by design."""
    why = _app(512, "no-such-backend")._face_leg_why()
    assert isinstance(why, str) and why
