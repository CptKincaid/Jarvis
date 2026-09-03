"""Loading YuNet and SFace, or saying exactly why not.

THE FAILURE MODE THIS MODULE IS BUILT AGAINST ALREADY SHIPPED ON THIS
MACHINE. VSS has a ``region_mode='yunet'`` path that silently falls back to
head_box; it has never once run, and nobody noticed because a silent
fallback and a working feature produce identical logs. So:

* there is no second detector. An unusable model raises
  ``ModelUnavailable`` with a sentence saying which file and what is wrong
  with it, and callers turn that into "no opinion", never into "no faces";
* the weights are verified against ``jarvis/facemodels.py``'s pinned sizes
  before OpenCV is asked to load them, which catches the git-lfs POINTER the
  plain raw.githubusercontent URL serves -- 130 bytes that OpenCV would
  reject with a less useful message;
* a box with no cv2, or with a cv2 too old for ``FaceDetectorYN``, fails the
  same named way rather than at an import at the top of some other file.

THE DETECTOR'S OWN THRESHOLD IS A FLOOR, NOT THE VERDICT. Setting OpenCV's
``score_threshold`` to his configured ``camera.min_conf`` would mean a face
scoring 0.45 is not reported AT ALL, and the harness would print "no face
seen" when the truth is "your face scores 0.45 against your 0.6 bar". Those
two need different fixes, so the detector floors low and
``jarvis/visionrig.py`` applies the bar.

IDENTITY IS GATED ON THE DETECTION, IN CODE. SFace's embedding collapses on
out-of-distribution input -- unrelated non-face crops match each other at
cosine 0.66-0.92, well above the 0.363 "same person" bar (measured
2026-09-02, see jarvis/facemodels.py) -- so a bad crop scores CONFIDENTLY
against the gallery rather than low. ``SFaceRecogniser.embed`` therefore
refuses a row that did not already clear ``min_conf``. The rig enforces the
same rule at its call site; the duplication is deliberate, because a second
call site will eventually exist and this is not a rule to leave to callers.

cv2 is imported lazily, inside functions, so this module loads on a box with
no OpenCV.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from jarvis import facemodels as fm
from jarvis.logs import get_logger
from jarvis.visionrig import DETECT_COLS, IDX_SCORE

log = get_logger("facedetect")

# The floor OpenCV filters at when the harness is measuring. Low on purpose:
# see the module docstring. Production may pass camera.min_conf once the bar
# has been measured against a real face, but the self-check must not.
PROBE_THRESHOLD = 0.3
DEFAULT_INPUT_SIZE = (320, 180)
DEFAULT_THREADS = 2
NMS_THRESHOLD = 0.3
TOP_K = 50


class ModelUnavailable(RuntimeError):
    """The model cannot be used, and the message says which and why.

    Callers turn this into "no opinion". None of them may turn it into a
    different detector, a different region mode, or zero faces.
    """


def _import_cv2():
    import cv2                       # noqa: PLC0415 - deliberately lazy
    return cv2


def cv2_status() -> dict:
    """Can this box build the two models at all? Strings and booleans."""
    try:
        cv2 = _import_cv2()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "version": "", "detector_api": False,
                "recogniser_api": False, "reason": str(exc)}
    return {"available": True,
            "version": str(getattr(cv2, "__version__", "")),
            "detector_api": hasattr(cv2, "FaceDetectorYN_create"),
            "recogniser_api": hasattr(cv2, "FaceRecognizerSF_create"),
            "reason": ""}


def probe(deep: bool = False, model_dir=None) -> dict:
    """What is on disk, what it is licensed under, and whether cv2 can use it.

    Numbers, strings and booleans only -- this goes straight into the
    self-check report he pastes back.
    """
    models = {}
    for model in fm.MODELS:
        ok, why = fm.verify(model, deep=deep, model_dir=model_dir)
        path = fm.model_path(model, model_dir)
        models[model.key] = {
            "ok": bool(ok), "reason": "" if ok else why,
            "file": model.filename, "path": str(path),
            "bytes": int(path.stat().st_size) if path.exists() else 0,
            "expected_bytes": model.size, "licence": model.licence,
            "url": model.url,
        }
    return {"dir": str(fm.model_dir(model_dir)),
            "ready": all(m["ok"] for m in models.values()),
            "deep": bool(deep), "models": models, "cv2": cv2_status()}


def _verified_path(model, deep: bool, model_dir=None):
    ok, why = fm.verify(model, deep=deep, model_dir=model_dir)
    if not ok:
        raise ModelUnavailable(
            "%s is unusable (%s). Nothing falls back to anything else; the "
            "camera path reports no opinion until this file is right."
            % (model.filename, why))
    return str(fm.model_path(model, model_dir))


# ----------------------------------------------------------------- detector
def _create_yunet(path: str, *, input_size, score_threshold: float,
                  threads: int):
    cv2 = _import_cv2()
    if not hasattr(cv2, "FaceDetectorYN_create"):
        raise ModelUnavailable(
            "cv2 %s has no FaceDetectorYN_create; YuNet needs OpenCV 4.5.4+ "
            "with objdetect (opencv-python, not headless)"
            % getattr(cv2, "__version__", "?"))
    try:
        cv2.setNumThreads(int(threads))
    except Exception:  # noqa: BLE001 - a build without TBB still works
        log.debug("facedetect: setNumThreads refused", exc_info=True)
    return cv2.FaceDetectorYN_create(path, "", tuple(input_size),
                                     float(score_threshold), NMS_THRESHOLD,
                                     TOP_K)


def _cv2_resize(frame, size):
    cv2 = _import_cv2()
    return cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)


class YuNetDetector:
    """The seam ``jarvis/visionrig.Rig`` drives: ``input_size`` + ``detect``.

    The DOWNSCALE lives here rather than in the rig so the rig imports no
    cv2 and runs in the suite. It is also where the aspect decision bites:
    320x180 for a 16:9 capture is an exact ratio in both axes, while 320x240
    is a 1.33x horizontal squash of every face in the frame. The rig checks
    that and says so.
    """

    name = "yunet"

    def __init__(self, net, input_size, resize: Optional[Callable] = None):
        self._net = net
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self._resize = resize or _cv2_resize
        setter = getattr(net, "setInputSize", None)
        if callable(setter):
            setter(self.input_size)

    def detect(self, frame):
        """(N, 15) rows in DETECT pixels, or None. Never an empty guess: a
        detector that failed raises, and the rig counts it as an error."""
        small = self._resize(frame, self.input_size)
        _ok, faces = self._net.detect(small)
        return faces


def load_detector(min_conf: Optional[float] = None,
                  input_size=DEFAULT_INPUT_SIZE,
                  threads: int = DEFAULT_THREADS,
                  score_threshold: Optional[float] = None,
                  deep: bool = False, model_dir=None,
                  create: Optional[Callable[..., Any]] = None) -> YuNetDetector:
    """YuNet, or ``ModelUnavailable`` saying why not."""
    path = _verified_path(fm.YUNET, deep, model_dir)
    floor = PROBE_THRESHOLD if score_threshold is None else float(
        score_threshold)
    if min_conf is not None:
        # Never ABOVE his bar: that is the silent-miss trap in the docstring.
        floor = min(floor, float(min_conf))
    maker = create or _create_yunet
    try:
        net = maker(path, input_size=tuple(input_size), score_threshold=floor,
                    threads=int(threads))
    except ModelUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelUnavailable("OpenCV could not build YuNet from %s (%s: %s)"
                               % (path, type(exc).__name__, exc)) from exc
    return YuNetDetector(net, input_size)


# --------------------------------------------------------------- recogniser
def _create_sface(path: str, **_kw):
    cv2 = _import_cv2()
    if not hasattr(cv2, "FaceRecognizerSF_create"):
        raise ModelUnavailable(
            "cv2 %s has no FaceRecognizerSF_create"
            % getattr(cv2, "__version__", "?"))
    return cv2.FaceRecognizerSF_create(path, "")


class SFaceRecogniser:
    """128-D SFace embeddings, and the gate that keeps them meaningful."""

    name = "sface"

    def __init__(self, net, min_conf: float):
        self._net = net
        self.min_conf = float(min_conf)

    def embed(self, frame, row):
        """The embedding for one detection row, aligned by SFace's own crop.

        Raises on a row under ``min_conf``. That is not defensive
        programming: SFace answers CONFIDENTLY on inputs that are not faces,
        so an embedding taken from a weak detection is worse than no
        embedding -- it is a wrong answer that looks right.

        THE SCORE IS READ FROM ``IDX_SCORE``, NOT FROM ``row[-1]``. This gate
        and the one in ``faceenrol.judge_sample`` are deliberately two gates
        on ONE rule, and that argument only holds if they read the same
        number. They are identical for YuNet's 15 columns and divergent for
        anything longer, in both directions: with a 16-column row a 0.45
        detection the judge would refuse got embedded, and a 0.99 detection
        got dropped. A differently shaped detector must fail LOUDLY here
        rather than have some other column silently read as its confidence.

        ``not (conf >= bar)`` rather than ``conf < bar`` so a non-finite
        score fails SHUT: ``NaN < 0.6`` is False, which made the one gate the
        safety argument rests on fail OPEN on a NaN.
        """
        arr = np.asarray(row).ravel()
        if arr.size != DETECT_COLS:
            raise ValueError(
                "refusing to embed a %d-column detection row: YuNet's is %d "
                "and the confidence is column %d. A row of another shape "
                "would have some other number read as its score."
                % (arr.size, DETECT_COLS, IDX_SCORE))
        conf = float(arr[IDX_SCORE])
        if not (conf >= self.min_conf):
            raise ValueError(
                "refusing to embed a detection scoring %.2f, under the %.2f "
                "bar: SFace collapses on out-of-distribution input and would "
                "return a confident match rather than a low score"
                % (conf, self.min_conf))
        crop = self._net.alignCrop(frame, np.asarray(row, dtype=np.float32))
        return np.asarray(self._net.feature(crop),
                          dtype=np.float32).ravel()


def load_recogniser(min_conf: float = 0.6, deep: bool = False,
                    model_dir=None,
                    create: Optional[Callable[..., Any]] = None
                    ) -> SFaceRecogniser:
    """SFace, or ``ModelUnavailable`` saying why not."""
    path = _verified_path(fm.SFACE, deep, model_dir)
    maker = create or _create_sface
    try:
        net = maker(path)
    except ModelUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelUnavailable("OpenCV could not build SFace from %s (%s: %s)"
                               % (path, type(exc).__name__, exc)) from exc
    return SFaceRecogniser(net, min_conf)
