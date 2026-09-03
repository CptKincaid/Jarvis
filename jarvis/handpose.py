"""The hand models: where they live, what they are licensed under, or why not.

This is ``jarvis/facemodels.py`` and ``jarvis/facedetect.py`` for hands, in
one file, and it keeps both of their rules. It is also the ONLY module in the
gesture path that touches a pixel, and it drops every buffer it makes before
it returns: what comes out is 21 coordinates and two floats.

THE WEIGHTS ARE NOT IN THE REPO AND MUST NOT BE. ``repo/`` has already
swallowed 140 MB of TTS weights in one commit. They live under
``~/.aiws_trainer/models/hand`` beside the face models -- the directory this
machine already treats as "downloaded artefacts", and, unlike
``~/vss_env/models``, not inside a virtualenv a ``pip install`` can recreate
out from under them.

LICENCE, ESTABLISHED BY READING THE PER-MODEL LICENSE FILE, NOT THE REPO
ROOT. VSS is a commercial project on this same box with a ship-gate and a
label-provenance firewall, so a research-licensed model sitting next to it is
how a licence leak happens. Both files here were checked on 2026-09-03 by
fetching opencv_zoo's own per-model LICENSE:

* palm detection      -- **Apache-2.0**. opencv_zoo
  ``models/palm_detection_mediapipe/LICENSE``, 11358 bytes, sha256
  cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30 --
  BYTE-IDENTICAL to the Apache-2.0 text already verified for SFace in
  ``jarvis/facemodels.py``, which is itself byte-identical to apache.org's.
* handpose estimation -- **Apache-2.0**. opencv_zoo
  ``models/handpose_estimation_mediapipe/LICENSE``, 11357 bytes, sha256
  58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd. It
  differs from the palm file in exactly one byte: a missing trailing
  newline. Verified by diff, not by eye.

BOTH ARE COMMERCIALLY CLEAN. Neither needs the
``AIWS_ALLOW_NONCOMMERCIAL_MODELS`` gate that Cosmos and LocateAnything-3B
sit behind, and neither is MediaPipe's ``hand_landmarker.task`` bundle, which
carries Google's own terms and is deliberately NOT used here. The two licence
texts are copied in beside the weights (``LICENSE.palm_detection.Apache-2.0
.txt``, ``LICENSE.handpose_estimation.Apache-2.0.txt``) with a SHA256SUMS
file, so the provenance survives with no network -- the same arrangement the
face directory already has.

THE PRE/POST-PROCESSING IS ADAPTED FROM opencv_zoo's OWN DEMO CODE
(``mp_palmdet.py``, ``mp_handpose.py``, Apache-2.0, same repository as the
weights), tidied and given the failure behaviour this codebase requires. It
is adapted rather than imported because the originals live in a scratch
directory that is not on the path and is not backed up.

THE 2016 SSD ANCHORS ARE GENERATED, NOT PASTED. opencv_zoo ships them as
2016 float literals -- 2000 lines of the demo file. They are simply the
centres of a 24x24 grid taken twice and a 12x12 grid taken six times, so
they are computed here in four lines. MEASURED: the generated array matches
the pasted literals to within 7.45e-9, which is the literals' own repr
rounding (they are printed to 8 significant figures), and 7.45e-9 of a
normalised anchor is 9.5e-6 px on a 1280-wide frame. The generator is
therefore not an approximation of the shipped anchors; the shipped anchors
are a rounded printout of it.

int8 IS THE PINNED HANDPOSE, AND THAT IS MEASURED ON THIS BOX. From
``~/scratch-gesture/bench_onnx.json``, milliseconds per call:

    threads          1        2        4
    handpose fp32  7.24     4.85     2.79
    handpose int8  1.91     1.29     0.92
    palm detect   12.10     7.71     4.50

So the whole hand stage is 9.00 ms at 2 threads and 5.42 ms at 4, against a
133 ms frame period at the ~7.5 fps the camera actually delivers. Two threads
is the pinned choice, matching ``camera.threads``: this box has already had
one unified-memory power-off with two trainers resident, and 9 ms inside a
133 ms budget does not need the extra contention. (Note this is the opposite
of the face models, where int8 is SLOWER on this aarch64 build -- SFace int8
23.0 ms against fp32 20.9 -- which is why nothing here assumes quantisation
is a win without measuring it.)

ONNX RUNTIME, NOT cv2.dnn, RUNS THE GRAPHS. cv2 has a first-class
``FaceDetectorYN`` for faces and nothing equivalent for hands, so the graph
is run directly; ORT is what the benchmark above measured and is what the
int8 model needs. cv2 is still required, for resize / warpAffine / NMS, and
is imported lazily so this module loads on a box without it.

AN UNUSABLE MODEL RAISES. It never degrades to "no hands". That is the VSS
``region_mode='yunet'`` failure this codebase is built against: a silent
fallback and a working feature produce identical logs. Zero hands with no
exception means the room had no hand in it. Being unable to look is spelled
``HandModelUnavailable`` with a sentence naming the file.

Nothing here opens a camera, and nothing here touches the network.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from jarvis import gesture as gest
from jarvis.logs import get_logger

log = get_logger("handpose")

ZOO = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
       "models")


@dataclass(frozen=True)
class HandModel:
    """One downloaded weight file, pinned by size and hash.

    ``licence_sha256`` pins the LICENSE text that was actually read, so
    ``provenance()`` can say "the licence beside these weights is the one
    this file's verdict was made on" rather than "a file with that name
    exists".
    """

    key: str
    filename: str
    sha256: str
    size: int
    licence: str
    licence_file: str
    url: str
    licence_sha256: str = ""


# The two Apache-2.0 texts as fetched from opencv_zoo on 2026-09-03. The
# palm one is byte-identical to the text jarvis/facemodels.py verified for
# SFace against apache.org; the handpose one differs from it by exactly one
# trailing newline (11357 vs 11358 bytes). Verified by diff, not by eye.
LICENCE_SHA_PALM = (
    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30")
LICENCE_SHA_HANDPOSE = (
    "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd")


# Pinned to what was downloaded, hashed and licence-checked on 2026-09-03. A
# changed hash means this is not the file whose LICENSE was read, which is a
# licence question before it is a correctness one -- so a mismatch is hard,
# not a warning.
PALM_DET = HandModel(
    key="palm",
    filename="palm_detection_mediapipe_2023feb.onnx",
    sha256="78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c",
    size=3905734,
    licence="Apache-2.0",
    licence_file="LICENSE.palm_detection.Apache-2.0.txt",
    url=f"{ZOO}/palm_detection_mediapipe/"
        f"palm_detection_mediapipe_2023feb.onnx",
    licence_sha256=LICENCE_SHA_PALM,
)
HANDPOSE = HandModel(
    key="landmarks",
    filename="handpose_estimation_mediapipe_2023feb_int8.onnx",
    sha256="e97bc1fb83b641954d33424c82b6ade719d0f73250bdb91710ecfd5f7b47e321",
    size=1167628,
    licence="Apache-2.0",
    licence_file="LICENSE.handpose_estimation.Apache-2.0.txt",
    url=f"{ZOO}/handpose_estimation_mediapipe/"
        f"handpose_estimation_mediapipe_2023feb_int8.onnx",
    licence_sha256=LICENCE_SHA_HANDPOSE,
)
# The fp32 landmark model is downloaded too and is 3.8x slower at 2 threads.
# It is here so a self-check can compare the two on identical input if the
# int8 quantisation is ever suspected of moving a landmark; it is not loaded
# in production.
HANDPOSE_FP32 = HandModel(
    key="landmarks_fp32",
    filename="handpose_estimation_mediapipe_2023feb.onnx",
    sha256="db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
    size=4099621,
    licence="Apache-2.0",
    licence_file="LICENSE.handpose_estimation.Apache-2.0.txt",
    url=f"{ZOO}/handpose_estimation_mediapipe/"
        f"handpose_estimation_mediapipe_2023feb.onnx",
    licence_sha256=LICENCE_SHA_HANDPOSE,
)
MODELS = (PALM_DET, HANDPOSE)

DEFAULT_THREADS = 2          # matches camera.threads; see the docstring
PALM_INPUT = (192, 192)
POSE_INPUT = (224, 224)
DETECT_SIZE = (320, 180)     # the downsample both graphs run on (frames lane)
NMS_TOP_K = 5000             # NMS candidate cap, not the hand count
LANDMARK_COLS = 132          # 4 bbox + 63 lm + 63 world + handed + conf
IDX_LM, IDX_WORLD, IDX_HANDED, IDX_CONF = 4, 67, 130, 131


class HandModelUnavailable(RuntimeError):
    """The model cannot be used, and the message says which and why.

    Callers turn this into "we are blind". None of them may turn it into
    zero hands, a different model, or a quietly disabled gesture.
    """


# ------------------------------------------------------------ where/what
def model_dir(override: Optional[os.PathLike | str] = None) -> Path:
    """Where the weights live, most specific first.

    ``override`` is ``camera.gesture.model_dir`` from his config -- the
    caller's explicit choice, so it wins. ``JARVIS_HAND_MODEL_DIR`` is next
    and lets the suite point somewhere harmless. Otherwise the shipped
    location beside the face models.
    """
    if override:
        return Path(override)
    env = os.environ.get("JARVIS_HAND_MODEL_DIR")
    if env:
        return Path(env)
    return Path.home() / ".aiws_trainer" / "models" / "hand"


def model_path(model: HandModel,
               override: Optional[os.PathLike | str] = None) -> Path:
    return model_dir(override) / model.filename


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(model: HandModel, deep: bool = False,
           model_dir: Optional[os.PathLike | str] = None
           ) -> tuple[bool, str]:
    """(ok, reason). ``deep`` hashes the file.

    The cheap path checks the size, which is what catches the common failure:
    the plain raw.githubusercontent URL serves a git-lfs POINTER of about 130
    bytes, and ONNX Runtime's complaint about that is far less useful than
    "this is a pointer, not the model".
    """
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
            return False, (f"{path.name}: sha256 {got[:16]}... != pinned "
                           f"-- this is not the file whose LICENSE was read")
    return True, "ok"


def licence_text_present(model: HandModel,
                         model_dir: Optional[os.PathLike | str] = None
                         ) -> bool:
    """Is the licence copied in beside the weights, offline?"""
    path = globals()["model_dir"](model_dir) / model.licence_file
    return path.exists() and path.stat().st_size > 0


def licence_text_matches(model: HandModel,
                         model_dir: Optional[os.PathLike | str] = None
                         ) -> bool:
    """Is it the SAME licence text the verdict in this file was made on?

    Hashes the 11 KB beside the weights against ``licence_sha256``. A
    present-but-different text (someone re-fetched, or dropped a different
    LICENSE in) is a licence question, and this is what asks it.
    """
    if not model.licence_sha256:
        return False
    path = globals()["model_dir"](model_dir) / model.licence_file
    if not (path.exists() and path.stat().st_size > 0):
        return False
    try:
        return sha256_of(path) == model.licence_sha256
    except OSError:
        return False


def available(deep: bool = False,
              model_dir: Optional[os.PathLike | str] = None
              ) -> dict[str, tuple[bool, str]]:
    return {m.key: verify(m, deep=deep, model_dir=model_dir)
            for m in MODELS}


def ready(deep: bool = False,
          model_dir: Optional[os.PathLike | str] = None) -> bool:
    return all(ok for ok, _ in available(deep=deep,
                                         model_dir=model_dir).values())


# ------------------------------------------------------------- runtimes
def _import_cv2():
    import cv2                       # noqa: PLC0415 - deliberately lazy
    return cv2


def _import_ort():
    import onnxruntime as ort        # noqa: PLC0415 - deliberately lazy
    try:
        # 3 = ERROR. Without this ORT prints a GPU-discovery warning on this
        # aarch64 build every time a session is built, and the pipeline
        # rebuilds its sessions on every console mode change.
        ort.set_default_logger_severity(3)
    except Exception:                                        # noqa: BLE001
        pass
    return ort


def cv2_status() -> dict:
    try:
        cv2 = _import_cv2()
    except Exception as exc:                                 # noqa: BLE001
        return {"available": False, "version": "", "reason": str(exc)}
    return {"available": True,
            "version": str(getattr(cv2, "__version__", "")), "reason": ""}


def ort_status() -> dict:
    try:
        ort = _import_ort()
    except Exception as exc:                                 # noqa: BLE001
        return {"available": False, "version": "", "providers": "",
                "reason": str(exc)}
    try:
        providers = ",".join(ort.get_available_providers())
    except Exception:                                        # noqa: BLE001
        providers = ""
    return {"available": True,
            "version": str(getattr(ort, "__version__", "")),
            "providers": providers, "reason": ""}


def palm_anchors() -> np.ndarray:
    """The 2016 SSD anchor centres, generated rather than pasted.

    Two feature maps -- 24x24 with 2 anchors per cell, then 12x12 with 6 --
    laid out row-major, which is 1152 + 864 = 2016. See the module docstring
    for the measured agreement with opencv_zoo's literals (7.45e-9, their
    own repr rounding). A fresh array every call; ``_anchors()`` holds the
    one the detector uses, at module level, so that no stage object owns an
    ndarray and "no buffer survives a call" can be pinned mechanically.
    """
    rows = []
    for grid, per_cell in ((24, 2), (12, 6)):
        step = 1.0 / grid
        for gy in range(grid):
            cy = (gy + 0.5) * step
            for gx in range(grid):
                rows.extend([((gx + 0.5) * step, cy)] * per_cell)
    return np.asarray(rows, dtype=np.float32)


_ANCHORS: Optional[np.ndarray] = None


def _anchors() -> np.ndarray:
    global _ANCHORS
    if _ANCHORS is None:
        _ANCHORS = palm_anchors()
        _ANCHORS.setflags(write=False)
    return _ANCHORS


def provenance(model_dir: Optional[os.PathLike | str] = None) -> dict:
    """Does the directory carry its own proof of what it holds?

    Reads the ``SHA256SUMS`` file written beside the weights when they were
    fetched and compares each entry to the hash pinned here, and checks that
    both licence texts are present. It does NOT hash the weights -- that is
    ``verify(deep=True)`` -- it checks that the record on disk agrees with
    the record in this file, which is what survives a re-download by someone
    who did not read the LICENSE. Numbers, strings and booleans only.
    """
    root = globals()["model_dir"](model_dir)
    sums_path = root / "SHA256SUMS"
    listed: dict[str, str] = {}
    present = sums_path.exists()
    if present:
        try:
            for line in sums_path.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2 and len(parts[0]) == 64:
                    listed[parts[-1].lstrip("*")] = parts[0].lower()
        except Exception:                                    # noqa: BLE001
            present = False
    models = {}
    for model in (PALM_DET, HANDPOSE, HANDPOSE_FP32):
        got = listed.get(model.filename, "")
        models[model.key] = {
            "file": model.filename, "listed": bool(got),
            "matches_pin": bool(got) and got == model.sha256,
            "licence": model.licence,
            "licence_present": bool(licence_text_present(model, model_dir)),
            "licence_matches": bool(licence_text_matches(model, model_dir)),
        }
    ok = (present and all(m["matches_pin"] and m["licence_matches"]
                          for m in models.values()))
    return {"dir": str(root), "sums_present": bool(present),
            "entries": int(len(listed)), "ok": bool(ok), "models": models}


class _OrtNet:
    """One ONNX graph on the CPU, with the thread count pinned.

    ``inter_op`` is 1 on purpose: these graphs are a single chain, so a
    second op-level thread buys nothing and costs contention on a box that
    also runs a TTS model and an LLM.
    """

    def __init__(self, path: str, threads: int) -> None:
        ort = _import_ort()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads))
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        # Sorted so the output order is the graph's declaration order
        # (Identity, Identity_1, ...) rather than whatever the exporter
        # happened to emit -- the reference post-processing unpacks
        # positionally.
        self.output_names = sorted(o.name
                                   for o in self.session.get_outputs())

    def run(self, blob: np.ndarray) -> list:
        return self.session.run(
            self.output_names,
            {self.input_name: np.ascontiguousarray(
                blob.astype(np.float32))})


def _make_net(path: str, threads: int) -> _OrtNet:
    return _OrtNet(path, threads)


def _verified_path(model: HandModel, deep: bool, model_dir=None) -> str:
    ok, why = verify(model, deep=deep, model_dir=model_dir)
    if not ok:
        raise HandModelUnavailable(
            "%s is unusable (%s). Nothing falls back to anything else; the "
            "gesture path reports that it is blind, never that your hand "
            "was not there." % (model.filename, why))
    return str(model_path(model, model_dir))


# --------------------------------------------------- palm detection stage
class _PalmStage:
    """Finds hands. Adapted from opencv_zoo's MPPalmDet (Apache-2.0).

    Returns one row per palm: 4 box coordinates, 7 palm landmarks, score --
    all in the pixel space of the array handed in, which is what the
    landmark stage needs to crop and rotate.
    """

    def __init__(self, net, score: float, nms: float) -> None:
        self._net = net
        self.score = float(score)
        self.nms = float(nms)

    def _preprocess(self, image):
        cv2 = _import_cv2()
        pad_bias = np.array([0.0, 0.0])
        input_size = np.asarray(PALM_INPUT, dtype=np.float64)
        ratio = float(min(input_size / np.asarray(image.shape[:2],
                                                  dtype=np.float64)))
        if (image.shape[0] != PALM_INPUT[1]
                or image.shape[1] != PALM_INPUT[0]):
            size = (np.asarray(image.shape[:2]) * ratio).astype(np.int32)
            image = cv2.resize(image, (int(size[1]), int(size[0])))
            pad_h = PALM_INPUT[1] - int(size[0])
            pad_w = PALM_INPUT[0] - int(size[1])
            left, top = pad_w // 2, pad_h // 2
            pad_bias[0], pad_bias[1] = left, top
            image = cv2.copyMakeBorder(image, top, pad_h - top, left,
                                       pad_w - left, cv2.BORDER_CONSTANT,
                                       None, (0, 0, 0))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        return image[np.newaxis, ...], (pad_bias / ratio).astype(np.int32)

    def detect(self, image) -> np.ndarray:
        cv2 = _import_cv2()
        h, w = image.shape[:2]
        input_size = np.asarray(PALM_INPUT, dtype=np.float64)
        anchors = _anchors()
        blob, pad_bias = self._preprocess(image)
        out = self._net.run(blob)
        score = 1.0 / (1.0 + np.exp(-out[1][0, :, 0].astype(np.float64)))
        box_delta = out[0][0, :, 0:4]
        lm_delta = out[0][0, :, 4:]
        scale = float(max(w, h))
        cxy = box_delta[:, :2] / input_size
        wh = box_delta[:, 2:] / input_size
        xy1 = (cxy - wh / 2 + anchors) * scale
        xy2 = (cxy + wh / 2 + anchors) * scale
        boxes = np.concatenate([xy1, xy2], axis=1)
        boxes -= [pad_bias[0], pad_bias[1], pad_bias[0], pad_bias[1]]
        keep = cv2.dnn.NMSBoxes(boxes, score, self.score, self.nms,
                                top_k=NMS_TOP_K)
        if len(keep) == 0:
            return np.empty((0, 19), dtype=np.float32)
        keep = np.asarray(keep).ravel()
        lms = lm_delta[keep].reshape(-1, 7, 2) / input_size
        lms = (lms + anchors[keep][:, None, :]) * scale - pad_bias
        return np.c_[boxes[keep].reshape(-1, 4), lms.reshape(-1, 14),
                     score[keep].reshape(-1, 1)].astype(np.float32)


# ------------------------------------------------- landmark (pose) stage
class _PoseStage:
    """21 landmarks. Adapted from opencv_zoo's MPHandPose (Apache-2.0).

    The crop is rotated so the hand stands upright before inference, which
    is why the landmarks come back accurate at any wrist angle -- and why
    the closed-hand scalar in ``jarvis/gesture.py`` can be pose-tolerant at
    all. Every intermediate buffer is a local and is gone on return.
    """

    PALM_BASE, MIDDLE_BASE = 0, 2
    PRE_SHIFT = (0.0, 0.0)
    PRE_ENLARGE = 4.0
    SHIFT = (0.0, -0.4)
    ENLARGE = 3.0

    def __init__(self, net, conf: float) -> None:
        self._net = net
        self.conf = float(conf)

    def _crop_pad(self, image, box, for_rotation: bool):
        cv2 = _import_cv2()
        wh = box[1] - box[0]
        shift = np.asarray(self.PRE_SHIFT if for_rotation else self.SHIFT)
        box = box + shift * wh
        centre = np.sum(box, axis=0) / 2.0
        wh = box[1] - box[0]
        half = wh * (self.PRE_ENLARGE if for_rotation else self.ENLARGE) / 2.0
        box = np.array([centre - half, centre + half]).astype(np.int32)
        box[:, 0] = np.clip(box[:, 0], 0, image.shape[1])
        box[:, 1] = np.clip(box[:, 1], 0, image.shape[0])
        crop = image[box[0][1]:box[1][1], box[0][0]:box[1][0], :]
        if crop.size == 0:
            return None, box, np.array([0, 0])
        side = int(np.linalg.norm(crop.shape[:2]) if for_rotation
                   else max(crop.shape[:2]))
        pad_h, pad_w = side - crop.shape[0], side - crop.shape[1]
        left, top = pad_w // 2, pad_h // 2
        crop = cv2.copyMakeBorder(crop, top, pad_h - top, left,
                                  pad_w - left, cv2.BORDER_CONSTANT, None,
                                  (0, 0, 0))
        return crop, box, box[0] - [left, top]

    def _preprocess(self, image, palm):
        cv2 = _import_cv2()
        box = palm[0:4].reshape(2, 2).astype(np.float64)
        crop, box, bias = self._crop_pad(image, box, True)
        if crop is None:
            return None
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pad_bias = np.asarray(bias, dtype=np.int32)
        box = box - pad_bias
        marks = palm[4:18].reshape(7, 2).astype(np.float64) - pad_bias
        p1, p2 = marks[self.PALM_BASE], marks[self.MIDDLE_BASE]
        radians = math.pi / 2 - math.atan2(-(p2[1] - p1[1]), p2[0] - p1[0])
        radians -= 2 * math.pi * math.floor((radians + math.pi)
                                            / (2 * math.pi))
        angle = math.degrees(radians)
        centre = tuple(float(v) for v in (np.sum(box, axis=0) / 2.0))
        rot = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(crop, rot,
                                 (crop.shape[1], crop.shape[0]))
        homo = np.c_[marks, np.ones(marks.shape[0])]
        turned = np.array([np.dot(homo, rot[0]), np.dot(homo, rot[1])])
        rbox = np.array([np.amin(turned, axis=1), np.amax(turned, axis=1)])
        crop2, rbox, _ = self._crop_pad(rotated, rbox, False)
        if crop2 is None:
            return None
        blob = cv2.resize(crop2, dsize=POSE_INPUT,
                          interpolation=cv2.INTER_AREA).astype(np.float32)
        return blob[np.newaxis, ...] / 255.0, rbox, angle, rot, pad_bias

    def infer(self, image, palm) -> Optional[np.ndarray]:
        cv2 = _import_cv2()
        prepared = self._preprocess(image, palm)
        if prepared is None:
            return None
        blob, rbox, angle, rot, pad_bias = prepared
        lm, conf, handed, world = self._net.run(blob)
        conf = float(conf[0][0])
        if conf < self.conf:
            return None
        lm = lm[0].reshape(-1, 3).astype(np.float64)
        world = world[0].reshape(-1, 3).astype(np.float64)
        input_size = np.asarray(POSE_INPUT, dtype=np.float64)
        scale = float(np.max((rbox[1] - rbox[0]) / input_size))
        lm[:, :2] = (lm[:, :2] - input_size / 2.0) * scale
        lm[:, 2] = lm[:, 2] * scale
        back = cv2.getRotationMatrix2D((0.0, 0.0), angle, 1.0)
        turned = np.dot(lm[:, :2], back[:, :2])
        turned_world = np.dot(world[:, :2], back[:, :2])
        world = np.c_[turned_world, world[:, 2]]
        comp = np.array([[rot[0][0], rot[1][0]], [rot[0][1], rot[1][1]]])
        trans = np.array([rot[0][2], rot[1][2]])
        inv = np.c_[comp, np.array([-np.dot(comp[0], trans),
                                    -np.dot(comp[1], trans)])]
        centre = np.append(np.sum(rbox, axis=0) / 2.0, 1)
        origin = np.array([np.dot(centre, inv[0]), np.dot(centre, inv[1])])
        lm[:, :2] = turned + origin + pad_bias
        return np.r_[np.array([0.0, 0.0, 0.0, 0.0]), lm.reshape(-1),
                     world.reshape(-1), float(handed[0][0]), conf]


# ------------------------------------------------------------ the tracker
@dataclass(frozen=True)
class HandRow:
    """One hand, one frame. The only array in this module's output.

    ``world`` is the network's metric 3-D prediction. It is LOGGED, NOT
    DEPENDED ON: a 3-D closed-hand ratio is perfectly pose-invariant in
    simulation, but that was simulated 3-D, not this network's, and shipping
    a threshold on an unmeasured signal is exactly the mistake this project
    has been damaged by twice. ``handed`` is the model's left/right call: it
    is recorded and NEVER gated on, because it is defined relative to an
    assumed mirroring and is therefore the one field that silently inverts
    when somebody "fixes" the mirror.
    """

    conf: float
    lm: Any
    world: Any = None
    handed: float = 0.0
    palm_diag: float = 0.0


class HandTracker:
    """Palm detection then landmarks, on a frame the caller already has.

    THE CALLER OWNS THE FRAME AND THIS CLASS OWNS NOTHING. It keeps no
    buffer, no crop and no history; ``detect`` is a pure function of the
    array handed in plus the two graphs. That is what lets the gesture stage
    ride ``PreviewPipeline.grab()`` without a second ``cv2.VideoCapture`` --
    which is forbidden, because ``SensingPolicy.attach`` REPLACES a device by
    name and a second feed would silently displace the first, leaving the
    curfew pointing at a lens the app no longer uses.
    """

    input_size = PALM_INPUT

    def __init__(self, model_dir=None, threads: int = DEFAULT_THREADS,
                 palm_score: float = 0.6, nms: float = 0.3,
                 top_k: int = 2, landmark_conf: float = 0.7,
                 deep: bool = False,
                 create: Optional[Callable[..., Any]] = None,
                 detect_size: Optional[tuple] = DETECT_SIZE) -> None:
        maker = create or _make_net
        palm_path = _verified_path(PALM_DET, deep, model_dir)
        pose_path = _verified_path(HANDPOSE, deep, model_dir)
        try:
            palm_net = maker(palm_path, int(threads))
            pose_net = maker(pose_path, int(threads))
        except HandModelUnavailable:
            raise
        except Exception as exc:                             # noqa: BLE001
            raise HandModelUnavailable(
                "ONNX Runtime could not build the hand models from %s "
                "(%s: %s)" % (model_dir or model_path(PALM_DET).parent,
                              type(exc).__name__, exc)) from exc
        self.threads = int(threads)
        self.top_k = max(1, int(top_k))
        self.detect_size = (tuple(int(v) for v in detect_size)
                            if detect_size else None)
        self._palm = _PalmStage(palm_net, palm_score, nms)
        self._pose = _PoseStage(pose_net, landmark_conf)

    def _small(self, frame):
        """The frame the graphs actually see, and the factor back.

        MEASURED (frames lane): the palm detector letterboxes every 16:9
        frame into the same 192x108, so 1280x720 buys nothing over 320x180
        except 5.6 ms; the handpose crop is taken from the SAME small frame
        (landmark error 0.028-0.065 palm-lengths, against a 0.634 open/closed
        separation). The resize is INTER_AREA at 0.04 ms. Both stages run on
        the small frame and the landmarks are scaled back, so what comes out
        is in the pixel space of the frame handed in -- the capture space
        the face's eye_px is already in, which is what makes R honest.
        """
        if not self.detect_size:
            return frame, 1.0, 1.0
        h, w = frame.shape[:2]
        dw, dh = self.detect_size
        if w <= dw and h <= dh:
            return frame, 1.0, 1.0
        cv2 = _import_cv2()
        small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        return small, float(w) / float(dw), float(h) / float(dh)

    def detect(self, frame, origin: tuple = (0.0, 0.0)
               ) -> tuple[HandRow, ...]:
        """0..top_k hands, largest palm_diag first.

        Coordinates come back in the pixel space of ``frame`` whatever size
        the graphs ran at; pass ``origin`` when ``frame`` is a crop of the
        capture buffer and the rows should be in capture pixels. An EMPTY
        TUPLE means no hand was found. Being unable to look raises
        ``HandModelUnavailable`` at construction; the two are never
        confused. Nothing about ``frame`` -- not the downsample, not a
        crop, not a blob -- survives the return.
        """
        small, sx, sy = self._small(frame)
        palms = self._palm.detect(small)
        rows: list[HandRow] = []
        for palm in palms:
            out = self._pose.infer(small, palm)
            if out is None:
                continue
            lm = out[IDX_LM:IDX_WORLD].reshape(21, 3)[:, :2].astype(
                np.float32)
            lm = lm * np.asarray((sx, sy), dtype=np.float32)
            lm = lm + np.asarray(origin, dtype=np.float32)
            world = out[IDX_WORLD:IDX_HANDED].reshape(21, 3).astype(
                np.float32)
            # One definition of the scale, in jarvis/gesture.py. Duplicating
            # the formula here is how the tracker and the state machine end
            # up disagreeing about what a hand-unit is.
            diag = gest.observe_hand(lm, 0.0).palm_diag
            rows.append(HandRow(conf=float(out[IDX_CONF]), lm=lm,
                                world=world, handed=float(out[IDX_HANDED]),
                                palm_diag=float(diag)))
        rows.sort(key=lambda r: r.palm_diag, reverse=True)
        return tuple(rows[:self.top_k])


def probe(model_dir=None, deep: bool = False) -> dict:
    """What is on disk, what it is licensed under, and can it be run.

    Numbers, strings and booleans only -- this goes straight into the
    self-check report he pastes back, and it must pass
    ``visionrig.assert_numbers_only``.
    """
    models = {}
    for model in (PALM_DET, HANDPOSE, HANDPOSE_FP32):
        ok, why = verify(model, deep=deep, model_dir=model_dir)
        path = model_path(model, model_dir)
        models[model.key] = {
            "ok": bool(ok), "reason": "" if ok else why,
            "file": model.filename, "path": str(path),
            "bytes": int(path.stat().st_size) if path.exists() else 0,
            "expected_bytes": model.size, "licence": model.licence,
            "licence_file": model.licence_file,
            "licence_present": bool(licence_text_present(model, model_dir)),
            "licence_matches": bool(licence_text_matches(model, model_dir)),
            "commercial_ok": True,
            "url": model.url,
        }
    return {"dir": str(globals()["model_dir"](model_dir)),
            "ready": bool(ready(deep=deep, model_dir=model_dir)),
            "deep": bool(deep), "threads": DEFAULT_THREADS,
            "detect_size": "%dx%d" % DETECT_SIZE,
            "anchors": int(_anchors().shape[0]),
            "models": models, "provenance": provenance(model_dir),
            "cv2": cv2_status(), "ort": ort_status()}


__all__ = [
    "DEFAULT_THREADS", "DETECT_SIZE", "HANDPOSE", "HANDPOSE_FP32",
    "HandModel", "HandModelUnavailable", "HandRow", "HandTracker",
    "LICENCE_SHA_HANDPOSE", "LICENCE_SHA_PALM", "MODELS", "PALM_DET",
    "available", "cv2_status", "licence_text_matches",
    "licence_text_present", "model_dir", "model_path", "ort_status",
    "palm_anchors", "probe", "provenance", "ready", "sha256_of", "verify",
]
