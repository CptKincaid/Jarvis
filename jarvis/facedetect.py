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

from jarvis import faceinsight as fi
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


def _import_ort():
    import onnxruntime               # noqa: PLC0415 - deliberately lazy
    return onnxruntime


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


def ort_status() -> dict:
    """Can onnxruntime build the InsightFace pair, and on what?

    THE PROVIDER LIST IS THE POINT. This box's onnxruntime is CPU-ONLY --
    ``['AzureExecutionProvider', 'CPUExecutionProvider']``, checked
    2026-09-03 -- so the GB10 is unreachable from here and the model choice
    was made on CPU numbers. Reporting the providers rather than assuming
    them is what keeps a future "why is this slow" from being guesswork.
    """
    try:
        ort = _import_ort()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "version": "", "providers": "",
                "reason": str(exc)}
    try:
        providers = ",".join(str(p) for p in ort.get_available_providers())
    except Exception as exc:  # noqa: BLE001
        providers = ""
        return {"available": True,
                "version": str(getattr(ort, "__version__", "")),
                "providers": providers, "reason": str(exc)}
    return {"available": True,
            "version": str(getattr(ort, "__version__", "")),
            "providers": providers, "reason": ""}


def probe(deep: bool = False, model_dir=None, backend=None) -> dict:
    """What is on disk, what it is licensed under, and whether it can load.

    Numbers, strings and booleans only -- this goes straight into the
    self-check report he pastes back.

    ``models`` is the ACTIVE backend's pair, keyed by role, which is what
    every existing reader expects. ``backends`` says what the other one would
    need, because "the detector is missing" and "the detector you are not
    using is missing" are different sentences and a report that cannot tell
    them apart sends him looking for the wrong file.
    """
    back = fm.backend_for(backend)
    models = {}
    for model in back.models():
        ok, why = fm.verify(model, deep=deep, model_dir=model_dir)
        path = fm.model_path(model, model_dir)
        models[model.key] = {
            "ok": bool(ok), "reason": "" if ok else why,
            "file": model.filename, "path": str(path),
            "bytes": int(path.stat().st_size) if path.exists() else 0,
            "expected_bytes": model.size, "licence": model.licence,
            "licence_source": model.licence_source,
            "commercial_ok": bool(model.commercial_ok),
            "url": model.url,
        }
    others = {}
    for name, other in sorted(fm.BACKENDS.items()):
        others[name] = {
            "ready": bool(fm.ready(deep=deep, model_dir=model_dir,
                                   backend=name)),
            "detector": other.detector.filename,
            "recogniser": other.recogniser.filename,
            "embed_dim": int(other.embed_dim),
            "embed_model": other.embed_model,
            "commercial_ok": bool(other.commercial_ok),
            "licence": other.recogniser.licence,
        }
    return {"dir": str(fm.model_dir(model_dir)),
            "backend": back.name,
            "embed_dim": int(back.embed_dim),
            "embed_model": back.embed_model,
            "commercial_ok": bool(back.commercial_ok),
            "licence": back.recogniser.licence,
            "backend_note": back.note,
            "ready": all(m["ok"] for m in models.values()),
            "deep": bool(deep), "models": models, "backends": others,
            "cv2": cv2_status(), "onnxruntime": ort_status()}


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


# --------------------------------------------------------- SCRFD detector
def _create_ort_session(path: str, threads: int = DEFAULT_THREADS, **_kw):
    """An onnxruntime session on the CPU provider, quietly.

    ``log_severity_level = 3`` because det_500m was exported at 640x640 with
    STATIC output shapes and is run here at 320x192. ORT returns the correct
    dynamic shapes and prints nine "Expected shape ... does not match actual
    shape" warnings per frame while doing it. At 8 fps that is 72 lines a
    second into his log for a condition that is normal and handled --
    ``faceinsight.decode`` checks the row count against the anchor grid and
    raises if they ever really disagree, which is the check that matters.

    CPU ONLY, DELIBERATELY. This box's onnxruntime advertises
    ``['AzureExecutionProvider', 'CPUExecutionProvider']`` -- there is no CUDA
    provider in this build, the GB10 is unreachable from it, and asking for
    one would either raise or silently fall back. The model choice (mbf, not
    r50) was made on that basis; see jarvis/facemodels.py.
    """
    ort = _import_ort()
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    try:
        opts.intra_op_num_threads = int(threads)
    except Exception:  # noqa: BLE001
        log.debug("facedetect: ORT would not take intra_op_num_threads",
                  exc_info=True)
    return ort.InferenceSession(path, opts,
                                providers=["CPUExecutionProvider"])


class ScrfdDetector:
    """SCRFD-500m behind YuNet's seam: ``input_size`` + ``detect(frame)``.

    ``input_size`` IS THE SIZE THE ROWS ARE EXPRESSED IN, not the size the
    graph is fed. The graph needs a multiple of 32 in both axes (strides
    8/16/32) and the shipped detect size is 320x180, so the frame is resized
    to 320x180 exactly as YuNet's is and then PADDED to 320x192 with zeros
    along the bottom. Rows 0..179 hold precisely the pixels a 320x180 resize
    produces, so every coordinate that comes back is already in 320x180 space
    and ``jarvis/visionrig``'s scale_x/scale_y need no adjustment at all.
    Resizing to 192 instead would have added a 1.067x vertical stretch that
    the rig could not see and would have attributed to the face.

    Rows come back in YuNet's exact 15-column layout with the five landmarks
    in YuNet's order, so nothing downstream changes. ``None`` for no faces,
    matching what OpenCV's detector returns, because the rig's "zero faces"
    and "blind detector" paths are different and must stay so.
    """

    name = "scrfd_500m"

    def __init__(self, session, input_size, score_threshold: float,
                 iou_threshold: float = fi.NMS_IOU,
                 resize: Optional[Callable] = None):
        self._session = session
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.padded_size = fi.pad_to_stride(self.input_size[1],
                                            self.input_size[0])
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self._resize = resize or _cv2_resize
        self._input_name = session.get_inputs()[0].name

    def detect(self, frame):
        small = self._resize(frame, self.input_size)
        arr = np.asarray(small)
        ph, pw = self.padded_size
        h, w = arr.shape[0], arr.shape[1]
        if (h, w) != (ph, pw):
            padded = np.zeros((ph, pw, arr.shape[2]), dtype=arr.dtype)
            padded[:h, :w] = arr
        else:
            padded = arr
        blob = fi.model_blob(padded)
        outs = self._session.run(None, {self._input_name: blob})
        boxes, kps, scores = fi.decode(outs, pw, ph, self.score_threshold,
                                       iou_threshold=self.iou_threshold)
        if boxes.shape[0] == 0:
            return None
        return fi.to_detect_rows(boxes, kps, scores)


def load_detector(min_conf: Optional[float] = None,
                  input_size=DEFAULT_INPUT_SIZE,
                  threads: int = DEFAULT_THREADS,
                  score_threshold: Optional[float] = None,
                  deep: bool = False, model_dir=None,
                  create: Optional[Callable[..., Any]] = None,
                  backend: Optional[str] = None):
    """The backend's detector, or ``ModelUnavailable`` saying why not.

    ``backend`` picks the pair; it is resolved by ``facemodels.backend_for``,
    so an unknown name raises rather than quietly leaving the old detector in
    place. Neither branch has a fallback: that is the whole argument of this
    module and a second detector to fall through to would end it.

    THE SCORE FLOOR MEANS THE SAME THING IN BOTH BRANCHES AND THE NUMBER DOES
    NOT. YuNet's and SCRFD's confidences are separate calibrations of separate
    networks; his ``camera.min_conf`` of 0.6 was chosen against YuNet's. It is
    still applied here only as a FLOOR (never above his bar, so a weak face is
    reported and judged rather than vanishing), but whether 0.6 is the right
    bar for SCRFD is UNMEASURED -- it needs his face, and
    ``scripts/face_model_compare.py`` plus a run of the rig is how that number
    arrives.
    """
    back = fm.backend_for(backend)
    floor = PROBE_THRESHOLD if score_threshold is None else float(
        score_threshold)
    if min_conf is not None:
        # Never ABOVE his bar: that is the silent-miss trap in the docstring.
        floor = min(floor, float(min_conf))
    if back.name == "opencv":
        path = _verified_path(fm.YUNET, deep, model_dir)
        maker = create or _create_yunet
        try:
            net = maker(path, input_size=tuple(input_size),
                        score_threshold=floor, threads=int(threads))
        except ModelUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailable(
                "OpenCV could not build YuNet from %s (%s: %s)"
                % (path, type(exc).__name__, exc)) from exc
        return YuNetDetector(net, input_size)

    path = _verified_path(back.detector, deep, model_dir)
    maker = create or _create_ort_session
    try:
        session = maker(path, threads=int(threads))
    except ModelUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelUnavailable(
            "onnxruntime could not build SCRFD from %s (%s: %s)"
            % (path, type(exc).__name__, exc)) from exc
    return ScrfdDetector(session, input_size, floor)


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


class ArcFaceRecogniser:
    """512-D ArcFace embeddings, aligned onto the standard 5-point template.

    THREE THINGS ARE DIFFERENT FROM SFACE AND ALL THREE FAIL SILENTLY.

    1. **The alignment.** SFace does its own ``alignCrop``; ArcFace wants the
       five landmarks warped onto ``faceinsight.ARCFACE_TEMPLATE`` by a
       least-squares SIMILARITY transform -- rotation, uniform scale and
       translation, never a shear. A wrong template still yields 512 finite
       floats that normalise and compare.
    2. **The preprocessing.** 112x112, RGB, NCHW, (x - 127.5) / 128.0. Not
       SFace's, which OpenCV applies internally.
    3. **THE COORDINATE SPACE, which is a bug the SFace path has today.**
       Every call site passes the FULL capture frame with a row in DETECTOR
       pixels (``rows = detector.detect(frame)`` returns 320x180 coordinates;
       ``frame`` is 1280x720). Measured on this box 2026-09-03 with synthetic
       arrays: ``alignCrop`` given that pair reads the frame's TOP-LEFT
       320x180 corner -- a frame bright only there produced a crop of mean
       160.09, a frame bright only where the face really was produced mean
       0.00. So today's SFace embedding is of the wrong part of the picture,
       warped by the face's geometry, which is exactly why it neither raises
       nor scores zero.

       This class scales the landmarks by ``frame_size / input_size`` before
       aligning, so its crop comes from where the face is. ``SFaceRecogniser``
       IS DELIBERATELY NOT CHANGED: his enrolled gallery was built through
       that path, and fixing the query side alone would leave stored vectors
       and live vectors measuring different things -- a silent, total loss of
       recognition. Reverting to the OpenCV backend must give back exactly
       today's behaviour. Fixing SFace is a separate change that costs a
       re-enrolment, and it is his to call.

    ``input_size`` may be None, which means "the row is already in the
    frame's own pixels" and scales by 1. That is the honest default for a
    caller that has not said, and it is what the tests exercise.
    """

    name = "arcface_mbf"
    embed_model = "arcface_mbf"
    dim = fi.EMBED_DIM

    def __init__(self, session, min_conf: float, input_size=None):
        self._session = session
        self.min_conf = float(min_conf)
        self.input_size = (None if input_size is None
                           else (int(input_size[0]), int(input_size[1])))
        self._input_name = session.get_inputs()[0].name

    def scales(self, frame) -> tuple:
        """``(scale_x, scale_y)`` from detector pixels to this frame's."""
        if self.input_size is None:
            return 1.0, 1.0
        arr = np.asarray(frame)
        if arr.ndim < 2:
            return 1.0, 1.0
        return (float(arr.shape[1]) / float(self.input_size[0]),
                float(arr.shape[0]) / float(self.input_size[1]))

    def embed(self, frame, row):
        """The unit-length 512-vector for one detection row.

        The gate is SFace's, word for word and for a stronger reason: ArcFace
        collapses on out-of-distribution input MORE than SFace does, not less
        -- measured 2026-09-03, unrelated non-face crops match each other at
        mean cosine 0.73-0.87 with 100% of pairs above 0.363 in every family
        tested, where SFace left flat colour partly separated
        (jarvis/facemodels.py has the table). The embedding is not a second
        opinion on whether this is a face.

        ``not (conf >= bar)`` rather than ``conf < bar`` so a non-finite score
        fails SHUT: ``NaN < 0.6`` is False, which would open the one gate the
        safety argument rests on.
        """
        arr = np.asarray(row).ravel()
        if arr.size != DETECT_COLS:
            raise ValueError(
                "refusing to embed a %d-column detection row: the detector's "
                "is %d and the confidence is column %d. A row of another "
                "shape would have some other number read as its score."
                % (arr.size, DETECT_COLS, IDX_SCORE))
        conf = float(arr[IDX_SCORE])
        if not (conf >= self.min_conf):
            raise ValueError(
                "refusing to embed a detection scoring %.2f, under the %.2f "
                "bar: ArcFace collapses on out-of-distribution input and "
                "would return a confident match rather than a low score"
                % (conf, self.min_conf))
        sx, sy = self.scales(frame)
        kps = fi.scale_landmarks(arr[4:14].reshape(5, 2), sx, sy)
        matrix = fi.arcface_matrix(kps)
        cv2 = _import_cv2()
        crop = cv2.warpAffine(np.asarray(frame),
                              np.asarray(matrix, dtype=np.float64),
                              (fi.ARCFACE_SIZE, fi.ARCFACE_SIZE),
                              borderValue=0.0)
        blob = fi.arcface_blob(crop)
        out = self._session.run(None, {self._input_name: blob})[0]
        return fi.normalise(np.asarray(out, dtype=np.float32).ravel())


def load_recogniser(min_conf: float = 0.6, deep: bool = False,
                    model_dir=None,
                    create: Optional[Callable[..., Any]] = None,
                    backend: Optional[str] = None,
                    input_size=None):
    """The backend's recogniser, or ``ModelUnavailable`` saying why not.

    ``input_size`` is the DETECTOR's input size and only the ArcFace branch
    uses it -- see ``ArcFaceRecogniser`` for what it is for and why SFace does
    not get the same treatment.
    """
    back = fm.backend_for(backend)
    if back.name == "opencv":
        path = _verified_path(fm.SFACE, deep, model_dir)
        maker = create or _create_sface
        try:
            net = maker(path)
        except ModelUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailable(
                "OpenCV could not build SFace from %s (%s: %s)"
                % (path, type(exc).__name__, exc)) from exc
        return SFaceRecogniser(net, min_conf)

    path = _verified_path(back.recogniser, deep, model_dir)
    maker = create or _create_ort_session
    try:
        session = maker(path, threads=DEFAULT_THREADS)
    except ModelUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelUnavailable(
            "onnxruntime could not build ArcFace from %s (%s: %s)"
            % (path, type(exc).__name__, exc)) from exc
    return ArcFaceRecogniser(session, min_conf, input_size=input_size)
