"""Where the face models live, what they are licensed under, and lens geometry.

THE MODELS ARE NOT IN THE REPO AND MUST NOT BE. They are downloaded weights,
they are tens of megabytes, and ``repo/`` has already swept 140 MB of TTS
weights into a commit once. They live under ``~/.aiws_trainer/models/face``,
beside the Silero VAD jit and the voiceprint -- the directory Jarvis already
treats as "downloaded artefacts this machine needs", and, unlike
``~/vss_env/models``, not inside a virtualenv that a ``pip install`` can
recreate out from under them.

LICENCE, WHICH IS LOAD-BEARING ON THIS BOX. VSS is a commercial project on the
same machine with a ship-gate and a label-provenance firewall, so a
research-licensed model sitting next to it is how a licence leak happens. Both
models here were verified by reading the PER-MODEL LICENSE file, not the
repository root:

* YuNet   -- **MIT**, Copyright (c) 2020 Shiqi Yu. opencv_zoo
  ``models/face_detection_yunet/LICENSE``.
* SFace   -- **Apache-2.0**, byte-identical to the canonical text from
  apache.org (both 11358 bytes, sha256 cfc7749b...). opencv_zoo
  ``models/face_recognition_sface/LICENSE``.

Both are permissive and commercial-safe. Neither is InsightFace/buffalo_l,
whose pretrained weights are research-only and are FORBIDDEN on this machine.
The licence texts are copied in beside the weights so the provenance survives
without a network.

THE OUT-OF-DISTRIBUTION COLLAPSE, measured here 2026-09-02 and the reason
``identity`` may never be trusted on a weak detection. SFace's 128-D embedding
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


@dataclass(frozen=True)
class FaceModel:
    """One downloaded weight file, pinned by hash."""

    key: str
    filename: str
    sha256: str
    size: int
    licence: str
    url: str


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
MODELS = (YUNET, SFACE)

# The 2026may re-export is the zoo's default but carries SYMBOLIC height/width
# dims for OpenCV 5.x's ONNX Runtime engine. This box has OpenCV 4.12, whose
# DNN backend infers on the exact input shape, so 2023mar is the correct pick
# here. Both load under 4.12; only 2023mar is what the zoo documents for it.
DETECTOR_OPENCV5 = "face_detection_yunet_2026may.onnx"
# int8 variants were downloaded and measured. They are SLOWER on this aarch64
# build -- SFace int8 23.0 ms against fp32 20.9 ms single-threaded -- so they
# are deliberately not the pinned choice.


def model_dir() -> Path:
    """Where the weights live. JARVIS_FACE_MODEL_DIR lets the suite point
    somewhere harmless; unset in production."""
    env = os.environ.get("JARVIS_FACE_MODEL_DIR")
    if env:
        return Path(env)
    return Path.home() / ".aiws_trainer" / "models" / "face"


def model_path(model: FaceModel) -> Path:
    return model_dir() / model.filename


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(model: FaceModel, deep: bool = False) -> tuple[bool, str]:
    """(ok, reason). ``deep`` hashes the file; the cheap path checks size,
    which catches the common failure of a git-lfs POINTER (about 130 bytes)
    downloaded instead of the weights."""
    path = model_path(model)
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


def available(deep: bool = False) -> dict[str, tuple[bool, str]]:
    """Per-model readiness, for a voice_check-style report."""
    return {m.key: verify(m, deep=deep) for m in MODELS}


def ready(deep: bool = False) -> bool:
    return all(ok for ok, _ in available(deep=deep).values())


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
