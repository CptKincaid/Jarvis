"""Where the face models live, what they are licensed under, and lens geometry.

THE MODELS ARE NOT IN THE REPO AND MUST NOT BE. They are downloaded weights,
they are tens of megabytes, and ``repo/`` has already swept 140 MB of TTS
weights into a commit once. They live under ``~/.aiws_trainer/models/face``,
beside the Silero VAD jit and the voiceprint -- the directory Jarvis already
treats as "downloaded artefacts this machine needs", and, unlike
``~/vss_env/models``, not inside a virtualenv that a ``pip install`` can
recreate out from under them.

LICENCE, WHICH IS LOAD-BEARING ON THIS BOX AND HAS CHANGED. VSS is a
commercial project on the same machine with a ship-gate and a
label-provenance firewall, so a research-licensed model sitting next to it is
how a licence leak happens. FOUR models are now declared here and they are NOT
all commercial-safe:

* YuNet   -- **MIT**, Copyright (c) 2020 Shiqi Yu. opencv_zoo
  ``models/face_detection_yunet/LICENSE``, read 2026-09-02.
* SFace   -- **Apache-2.0**, byte-identical to the canonical text from
  apache.org (both 11358 bytes, sha256 cfc7749b...). opencv_zoo
  ``models/face_recognition_sface/LICENSE``, read 2026-09-02.
* SCRFD-500m (``det_500m.onnx``)  -- **NON-COMMERCIAL RESEARCH ONLY**.
* ArcFace-mbf (``w600k_mbf.onnx``) -- **NON-COMMERCIAL RESEARCH ONLY**.

HOW THAT LAST PAIR WAS VERIFIED, 2026-09-03, and it is not the usual answer.
The habit in this module is to read the PER-MODEL LICENSE file rather than a
repository badge. InsightFace has neither: the repository root carries **no
LICENSE file at all** (the GitHub contents API for the root tree lists none,
and ``/repos/deepinsight/insightface/license`` returns 404), and the weights
ship as bare ``.onnx`` files in a zip with no licence beside them. The only
statement of terms that exists is the first line of
``model_zoo/README.md``, fetched and read on 2026-09-03:

    ":bell:   **ALL models are available for non-commercial research purposes
    only.**"

So there is no permissive grant to fall back on and nothing to read that
would soften it. That sentence IS the licence, it covers the buffalo_s pack
these two files came from, and it is recorded on both models as
``licence="NON-COMMERCIAL RESEARCH ONLY"`` with ``commercial_ok=False``.

WHAT THAT MEANS IN PRACTICE, so nobody has to reconstruct it later. Jarvis is
Hunter's personal assistant and personal research use is what the terms allow;
these weights are fine there. VSS is not, and neither is anything that ships.
``Backend.commercial_ok`` is the flag to test, ``commercial_backends()`` is the
safe list, and the OpenCV pair (YuNet + SFace) remains fully declared and one
config key away precisely so this is a SWAP HE CAN REVERSE, not a deletion.
An earlier version of this docstring said InsightFace weights were "FORBIDDEN
on this machine". That was written when nothing here needed them; it is now
narrower and stated above -- forbidden in anything commercial, allowed in
Jarvis, never silently.

THE OUT-OF-DISTRIBUTION COLLAPSE, measured here 2026-09-02 and the reason
``identity`` may never be trusted on a weak detection. It was re-measured for
ArcFace on 2026-09-03 and ARCFACE IS WORSE, NOT BETTER -- every synthetic
family collapses above the bar for it, where SFace at least left flat colour
partly separated. The table is on ``ARCFACE_MBF`` below, and it means the swap
TIGHTENS this rule rather than relaxing it. SFace's 128-D embedding
is well behaved on faces, but on inputs that are NOT faces it collapses toward
a common direction, and unrelated garbage then matches unrelated garbage far
above the 0.363 "same person" bar:

    input family        mean cosine   fraction >= 0.363
    uniform noise          0.848            1.000
    flat colour            0.923            1.000
    random blobs           0.714            1.000
    smooth gradients       0.660            0.937

So a false-positive box, a motion-blurred crop or a bad alignment does not
produce a LOW score against the gallery -- it produces a CONFIDENT one. The
mitigation is that identity is only ever computed from a detection that
already cleared ``min_conf``; the embedding is not a second opinion on whether
this is a face, and must never be used as one. (Note this is the opposite
failure from the histogram's: ``eye.BODY_MATCH_MIN``'s 0.750 problem is that
non-negative vectors are all similar BY CONSTRUCTION. SFace's components are
~49% negative, so that particular trap does not apply here -- this is a
learned collapse on out-of-distribution input, not an arithmetic artefact.)

Nothing in this module opens a camera, imports cv2 or touches the network.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ZOO = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
       "models")
# The buffalo_s pack, InsightFace v0.7. The zip is the only distribution --
# there is no per-file URL and no LICENSE inside it.
INSIGHTFACE = ("https://github.com/deepinsight/insightface/releases/download/"
               "v0.7")
# The ONE place InsightFace states terms for its weights, fetched and read
# 2026-09-03. There is no LICENSE file in that repository's root.
INSIGHTFACE_LICENCE = ("https://github.com/deepinsight/insightface/blob/master/"
                       "model_zoo/README.md")


@dataclass(frozen=True)
class FaceModel:
    """One downloaded weight file, pinned by hash.

    ``key`` is the ROLE ("detector" / "recogniser"), not the model's name, so
    the two backends' entries are interchangeable in a report and a caller
    reading ``probe()["models"]["detector"]`` keeps working across a swap.
    ``commercial_ok`` is the flag anything that ships must test; it is False
    for both InsightFace weights and the docstring above says why.
    """

    key: str
    filename: str
    sha256: str
    size: int
    licence: str
    url: str
    commercial_ok: bool = True
    licence_source: str = ""


# Pinned to what was downloaded and hashed on 2026-09-02. A changed hash means
# the file is not the one whose licence was read, which is a licence question
# before it is a correctness one -- hence a hard mismatch, not a warning.
YUNET = FaceModel(
    key="detector",
    filename="face_detection_yunet_2023mar.onnx",
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    size=232589,
    licence="MIT",
    url=f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
SFACE = FaceModel(
    key="recogniser",
    filename="face_recognition_sface_2021dec.onnx",
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    size=38696353,
    licence="Apache-2.0",
    url=f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
)

# --------------------------------------------------------------- InsightFace
# buffalo_s, InsightFace v0.7. NON-COMMERCIAL RESEARCH ONLY -- read the
# module docstring before reusing either of these anywhere that ships.
#
# WHY THEY ARE HERE AT ALL, in his words, 2026-09-03: *"he also is recognizing
# me less from the side angle"*. A 128-D SFace embedding is weak in profile.
# w600k_mbf is 512-D, trained on WebFace600K, and is 13.0 MB against SFace's
# 36.9. Sizes and per-call costs measured on this box (aarch64, 20 cores,
# onnxruntime 1.23.2, CPU only -- the GB10 is unreachable from this ORT
# build, whose providers are Azure and CPU):
#
#                       size     1 thr    2 thr    4 thr    dims
#   SFace 2021dec     36.9 MB   20.0 ms  10.2 ms   5.8 ms    128
#   ArcFace mbf       13.0 MB    9.3 ms   5.2 ms   4.5 ms    512
#   YuNet 2023mar      0.2 MB    2.3 ms   1.5 ms   1.4 ms
#   SCRFD 500m @320x192 2.4 MB   3.7 ms   2.5 ms   1.7 ms
#
# The SFace and YuNet rows above are from the 2026-09-02 bench; the SCRFD and
# ArcFace rows were measured on 2026-09-03 on synthetic frames. THE ONLY
# APPLES-TO-APPLES COMPARISON, both pairs through THIS code, 2 threads, a
# synthetic 1280x720 frame, detect including the resize, p50 over 60 calls:
#
#                    detect      embed        sum
#   opencv           2.05 ms    10.41 ms    12.47 ms
#   insightface      2.95 ms     5.53 ms     8.48 ms
#
# NOTE WHICH HALF WINS. The DETECTOR IS SLOWER -- SCRFD 2.95 against YuNet
# 2.05 -- and the whole saving is in the recogniser. "Smaller and faster" is
# true of the pair and false of the detector alone, which matters because the
# detector runs on every frame and the recogniser does not.
#
# THE OOD COLLAPSE IS WORSE HERE, NOT BETTER, and this was measured rather
# than assumed. Same synthetic generator, same 276 pairs per family, both
# models, 2026-09-03 -- mean pairwise cosine between UNRELATED non-face
# crops, and the fraction of those pairs at or above SFace's 0.363
# "same person" bar:
#
#   family              ArcFace mean  frac>=0.363    SFace mean  frac>=0.363
#   uniform noise           0.803        1.000          0.850        1.000
#   flat colour             0.868        1.000          0.635        0.793
#   random blobs            0.734        1.000          0.694        0.996
#   smooth gradients        0.829        1.000          0.694        0.928
#
# So swapping to ArcFace does NOT relax the rule that identity may only be
# computed from a detection that already cleared min_conf -- it tightens it.
# Every family collapses above the bar for ArcFace where SFace at least left
# flat colour partly separated. The embedding is not a second opinion on
# whether the crop is a face, in either model.
SCRFD_500M = FaceModel(
    key="detector",
    filename="det_500m.onnx",
    sha256="5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
    size=2524817,
    licence="NON-COMMERCIAL RESEARCH ONLY",
    url=f"{INSIGHTFACE}/buffalo_s.zip",
    commercial_ok=False,
    licence_source=INSIGHTFACE_LICENCE,
)
ARCFACE_MBF = FaceModel(
    key="recogniser",
    filename="w600k_mbf.onnx",
    sha256="9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
    size=13616099,
    licence="NON-COMMERCIAL RESEARCH ONLY",
    url=f"{INSIGHTFACE}/buffalo_s.zip",
    commercial_ok=False,
    licence_source=INSIGHTFACE_LICENCE,
)

ALL_MODELS = (YUNET, SFACE, SCRFD_500M, ARCFACE_MBF)


@dataclass(frozen=True)
class Backend:
    """One detector + recogniser pair, and what a vector from it means.

    A BACKEND IS THE UNIT, not a model, because the two halves are not
    independent: SCRFD's five landmarks are what ArcFace's alignment consumes,
    and ArcFace's 512 floats are not comparable with SFace's 128. Mixing a
    YuNet box with an ArcFace crop would run and would quietly measure the
    wrong thing, so nothing here lets a caller name one half.

    ``cosine_same`` is None where nothing has been measured. It is None for
    ArcFace on purpose: 0.363 is OpenCV's documented number for SFace and
    applying it to another model's vectors would be a threshold that no longer
    means anything. ``scripts/face_model_compare.py`` is how the real number
    arrives, and it needs his face, so only he can run it.
    """

    name: str
    detector: FaceModel
    recogniser: FaceModel
    embed_dim: int
    embed_model: str          # the tag facegallery stamps on every vector
    commercial_ok: bool
    cosine_same: Optional[float]
    note: str = ""

    def models(self) -> tuple:
        return (self.detector, self.recogniser)


OPENCV = Backend(
    name="opencv",
    detector=YUNET, recogniser=SFACE,
    embed_dim=128, embed_model="sface",
    commercial_ok=True,
    cosine_same=0.363,
    note="YuNet MIT + SFace Apache-2.0; commercial-safe; 128-D.",
)
INSIGHT = Backend(
    name="insightface",
    detector=SCRFD_500M, recogniser=ARCFACE_MBF,
    embed_dim=512, embed_model="arcface_mbf",
    commercial_ok=False,
    cosine_same=None,
    note=("SCRFD-500m + ArcFace-mbf, 512-D. NON-COMMERCIAL RESEARCH ONLY. "
          "No same-person cosine has been measured for this checkpoint on "
          "this machine -- run scripts/face_model_compare.py."),
)
BACKENDS = {b.name: b for b in (OPENCV, INSIGHT)}

# The swap. His symptom was profile recognition, and a 512-D ArcFace embedding
# on a properly aligned crop is the fix that is available without a GPU. The
# OpenCV pair stays fully declared and one config key away
# (``camera.face_backend: "opencv"``), which is what makes this reversible.
DEFAULT_BACKEND = "insightface"
BACKEND_ENV = "JARVIS_FACE_BACKEND"


def backend_for(name: Optional[str] = None) -> Backend:
    """The named backend, or the configured one, or the default.

    Order: the explicit argument, then ``JARVIS_FACE_BACKEND``, then
    ``DEFAULT_BACKEND``. An unknown name RAISES rather than falling back --
    a typo in his config that silently kept the old models is exactly the
    class of silent-fallback bug this whole subsystem is written against.
    """
    want = str(name or "").strip().lower()
    if not want:
        want = str(os.environ.get(BACKEND_ENV, "") or "").strip().lower()
    if not want:
        want = DEFAULT_BACKEND
    try:
        return BACKENDS[want]
    except KeyError:
        raise ValueError(
            "unknown face backend %r; known: %s"
            % (want, ", ".join(sorted(BACKENDS)))) from None


def models(backend: Optional[str] = None) -> tuple:
    """The (detector, recogniser) pair one backend needs on disk."""
    return backend_for(backend).models()


def commercial_backends() -> tuple:
    """The backends whose weights may be used in something that ships.

    VSS is on this machine and has a ship-gate. Anything reusing this module
    from there must pick from here, not from ``BACKENDS``.
    """
    return tuple(b.name for b in BACKENDS.values() if b.commercial_ok)

# The 2026may re-export is the zoo's default but carries SYMBOLIC height/width
# dims for OpenCV 5.x's ONNX Runtime engine. This box has OpenCV 4.12, whose
# DNN backend infers on the exact input shape, so 2023mar is the correct pick
# here. Both load under 4.12; only 2023mar is what the zoo documents for it.
DETECTOR_OPENCV5 = "face_detection_yunet_2026may.onnx"
# int8 variants were downloaded and measured. They are SLOWER on this aarch64
# build -- SFace int8 23.0 ms against fp32 20.9 ms single-threaded -- so they
# are deliberately not the pinned choice.


def model_dir(override: Optional[os.PathLike | str] = None) -> Path:
    """Where the weights live, most specific first.

    ``override`` is ``camera.model_dir`` from his config -- the caller's
    explicit choice, so it wins. JARVIS_FACE_MODEL_DIR is next and lets the
    suite point somewhere harmless (tests/conftest.py forces it, so no test
    can reach the real weights). Otherwise the shipped location.
    """
    if override:
        return Path(override)
    env = os.environ.get("JARVIS_FACE_MODEL_DIR")
    if env:
        return Path(env)
    return Path.home() / ".aiws_trainer" / "models" / "face"


def model_path(model: FaceModel,
               override: Optional[os.PathLike | str] = None) -> Path:
    return model_dir(override) / model.filename


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(model: FaceModel, deep: bool = False,
           model_dir: Optional[os.PathLike | str] = None) -> tuple[bool, str]:
    """(ok, reason). ``deep`` hashes the file; the cheap path checks size,
    which catches the common failure of a git-lfs POINTER (about 130 bytes)
    downloaded instead of the weights."""
    path = model_path(model, model_dir)
    if not path.exists():
        return False, f"missing: {path}"
    actual = path.stat().st_size
    if actual != model.size:
        if actual < 1024:
            return False, (f"{path.name} is {actual} B -- this is a git-lfs "
                           f"pointer, not the model")
        return False, f"{path.name}: size {actual} != expected {model.size}"
    if deep:
        got = sha256_of(path)
        if got != model.sha256:
            return False, f"{path.name}: sha256 {got[:16]}... != pinned"
    return True, "ok"


def available(deep: bool = False,
              model_dir: Optional[os.PathLike | str] = None,
              backend: Optional[str] = None
              ) -> dict[str, tuple[bool, str]]:
    """Per-ROLE readiness for one backend, for a voice_check-style report.

    Keyed by role ("detector", "recogniser") rather than by model, so a
    caller reading this dict does not have to know which pair is live. Which
    pair it is IS reported -- ``facedetect.probe()`` names the backend beside
    these, because "the detector is missing" and "the detector you are not
    using is missing" are different sentences.
    """
    return {m.key: verify(m, deep=deep, model_dir=model_dir)
            for m in models(backend)}


def ready(deep: bool = False,
          model_dir: Optional[os.PathLike | str] = None,
          backend: Optional[str] = None) -> bool:
    """Is the ACTIVE backend's pair usable? Not "are all four present".

    Requiring all four would make a box that has only the OpenCV pair report
    "not ready" while working perfectly, and the reverse after the swap. The
    question is only ever about the pair actually in use.
    """
    return all(ok for ok, _ in available(deep=deep, model_dir=model_dir,
                                         backend=backend).values())


# ------------------------------------------------------------------ geometry
def horizontal_fov_deg(diagonal_fov_deg: float,
                       aspect_w: float = 16.0,
                       aspect_h: float = 9.0) -> float:
    """Horizontal FOV from a DIAGONAL spec, which is how webcams are sold.

    Cameras advertise a diagonal and code wants a horizontal, and the
    conversion is not a ratio of the numbers -- it is a ratio of TANGENTS.
    Getting it wrong is how a 73 deg LifeCam becomes a "73 deg" horizontal
    field that is really 65.6, and every pixel-to-angle claim inherits the
    error.
    """
    if diagonal_fov_deg <= 0 or diagonal_fov_deg >= 180:
        raise ValueError(f"diagonal FOV out of range: {diagonal_fov_deg}")
    diag = math.hypot(aspect_w, aspect_h)
    half = math.tan(math.radians(diagonal_fov_deg) / 2.0) * (aspect_w / diag)
    return 2.0 * math.degrees(math.atan(half))


@dataclass(frozen=True)
class Lens:
    """A real camera's geometry. ``hfov_deg`` HAS NO DEFAULT, on purpose.

    The shipped config's comment reasons from "~90 deg horizontal", which is
    true only of the 98 deg-diagonal camera docs/vision.md section 9
    recommends BUYING. Every camera actually under consideration is narrower
    -- the LifeCam Cinema he already owns is 65.6 deg H, the C930e 82.2 -- and
    a pixel-to-angle map that assumes 90 overstates every off-axis angle by
    up to 1.4x. Requiring the caller to state the real number is the whole
    point of this class.
    """

    width_px: int
    height_px: int
    hfov_deg: float

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("lens resolution must be positive")
        if not 0.0 < self.hfov_deg < 180.0:
            raise ValueError(f"hfov_deg out of range: {self.hfov_deg}")

    @classmethod
    def from_diagonal(cls, width_px: int, height_px: int,
                      diagonal_fov_deg: float) -> "Lens":
        return cls(width_px, height_px,
                   horizontal_fov_deg(diagonal_fov_deg,
                                      float(width_px), float(height_px)))

    def span_cm(self, distance_cm: float) -> float:
        """How wide the frame is, in cm, at that distance."""
        return 2.0 * distance_cm * math.tan(math.radians(self.hfov_deg) / 2.0)

    def px_per_cm(self, distance_cm: float) -> float:
        return self.width_px / self.span_cm(distance_cm)

    def face_px(self, distance_cm: float, face_cm: float = 16.0) -> float:
        """Pixels across a face of ``face_cm`` at ``distance_cm``."""
        return face_cm * self.px_per_cm(distance_cm)

    def detect_face_px(self, distance_cm: float, detect_width: int,
                       face_cm: float = 16.0) -> float:
        """The same face after the downscale the detector actually sees.

        This, not ``face_px``, is the number YuNet's documented 10-300 px
        working range applies to.
        """
        return self.face_px(distance_cm, face_cm) * (detect_width /
                                                     self.width_px)

    def offset_deg(self, px_from_centre: float) -> float:
        """Horizontal angle off the lens axis for a pixel offset.

        Rectilinear (tangent) mapping, not the linear deg-per-pixel
        approximation, which is ~10% wrong at the edge of a 90 deg frame.
        """
        half_px = self.width_px / 2.0
        half_tan = math.tan(math.radians(self.hfov_deg) / 2.0)
        return math.degrees(math.atan(px_from_centre / half_px * half_tan))


# The two cameras actually in play, so no caller has to rediscover the
# tangent conversion. The LifeCam is 720p: a config asking it for 1920x1080
# is asking for a mode it does not have.
LIFECAM_CINEMA = Lens.from_diagonal(1280, 720, 73.0)   # -> 65.64 deg H
C930E = Lens(1920, 1080, 82.2)                          # H spec'd directly
ARDUCAM_IMX462 = Lens.from_diagonal(1920, 1080, 98.0)   # -> 90.1 deg H, the
#                                                         camera §9 says buy


def describe(lens: Lens, distance_cm: float = 95.0,
             detect_width: Optional[int] = 320) -> str:
    """One line for a bring-up log."""
    out = (f"{lens.width_px}x{lens.height_px} hfov {lens.hfov_deg:.1f}deg -> "
           f"span {lens.span_cm(distance_cm):.0f}cm at {distance_cm:.0f}cm, "
           f"face {lens.face_px(distance_cm):.0f}px")
    if detect_width:
        out += (f", {lens.detect_face_px(distance_cm, detect_width):.0f}px "
                f"at detect width {detect_width}")
    return out
