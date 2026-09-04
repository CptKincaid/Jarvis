"""SCRFD's decode and ArcFace's alignment, as arithmetic over arrays.

THIS MODULE IS WHERE THE SWAP CAN GO WRONG SILENTLY, so it is the module
that holds no model, no camera and no cv2. Everything here is numpy over
numbers, which is what lets the suite pin it on a box with neither weights
nor a lens -- and pinning it is the point, because both halves fail QUIETLY
when they are wrong:

* **SCRFD is not a drop-in for YuNet.** YuNet hands back finished rows. SCRFD
  hands back nine raw tensors -- score, bbox-distance and keypoint-distance
  per stride 8/16/32, two anchors per cell -- and every one of them has to be
  turned into pixels against the right anchor grid. Get the grid wrong by one
  anchor and boxes land in plausible-looking wrong places; get the stride
  multiply wrong and every box is the right shape at the wrong scale. Neither
  raises.
* **ArcFace's alignment is not SFace's ``alignCrop``.** ArcFace wants the face
  warped onto a fixed five-point template by a least-squares similarity
  transform. A wrong template still produces a 512-vector, still normalises,
  still gives cosines in [-1, 1] -- and quietly loses accuracy. There is no
  exception to catch. So the template is written down beside its source, and
  the transform is pinned by round-trips that need no image at all.

THE LANDMARK ORDER IS THE THING THAT MUST NOT DRIFT.
``jarvis/visionrig.landmark_geometry`` and ``landmark_plausibility`` index the
five points POSITIONALLY: point 0 and point 1 are the eyes and their
difference sets the roll and the yaw sign; points 3 and 4 are the mouth
corners and their midpoint sets the mouth-drop test that keeps his wall from
being a face. Swap 0 and 1 and every yaw inverts with nothing to notice it.

YuNet's documented order is *right eye, left eye, nose tip, right mouth
corner, left mouth corner*, where "right" is the PERSON's right -- the LEFT
half of the image for a face looking at the lens. SCRFD's five points are in
the same order, and that is not taken on trust: InsightFace feeds SCRFD's
``kps`` straight into ``face_align.estimate_norm`` against ``arcface_dst``,
and ``arcface_dst``'s own x-coordinates say which is which --

    point 0  x = 38.29   (left half of a 112-wide crop)   image-left eye
    point 1  x = 73.53   (right half)                     image-right eye
    point 2  x = 56.03   y = 71.74                        nose, between them
    point 3  x = 41.55   y = 92.37                        image-left mouth
    point 4  x = 70.73   y = 92.20                        image-right mouth

-- so index 0 is the image-left point in both conventions, index for index.
``tests/test_faceinsight.py`` asserts that ordering from the template numbers
themselves rather than from this paragraph.

SOURCES, both read on 2026-09-03 rather than remembered:
  https://raw.githubusercontent.com/deepinsight/insightface/master/
      python-package/insightface/utils/face_align.py     (arcface_dst)
  https://raw.githubusercontent.com/deepinsight/insightface/master/
      python-package/insightface/model_zoo/scrfd.py      (the decode)

LICENCE: the CODE above is InsightFace's (MIT-style headers, no root LICENSE
file in that repository -- checked 2026-09-03, the root tree has none). The
WEIGHTS this decode is written for are NOT permissive; see
``jarvis/facemodels.py``, which is where that restriction is recorded and
where it is enforced.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# ------------------------------------------------------------------- SCRFD
# det_500m.onnx exports nine tensors: three scores, three bbox distances,
# three keypoint distances, one group per stride. Two anchors share every
# cell. Confirmed on this box 2026-09-03 by running the graph on a synthetic
# 320x192 blob: row counts 1920 / 480 / 120, which is 40x24x2, 20x12x2 and
# 10x6x2 -- exactly strides 8, 16 and 32 with two anchors.
STRIDES = (8, 16, 32)
NUM_ANCHORS = 2
# Every stride must divide the input, so a detect height of 180 cannot be fed
# to the graph as it stands. The detector pads to this multiple; see
# ``pad_to_stride`` for why padding rather than resizing is the right move.
STRIDE_ALIGN = 32
# InsightFace's own defaults. The score floor is NOT set here: the caller
# passes it, because jarvis/facedetect.py's whole argument about a floor
# rather than a verdict applies to this detector exactly as it does to YuNet.
NMS_IOU = 0.4

# ----------------------------------------------------------------- ArcFace
# insightface/utils/face_align.py::arcface_dst, copied digit for digit. This
# is the ONLY thing that makes a 512-vector from this checkpoint comparable
# with another 512-vector from it.
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]], dtype=np.float64)
ARCFACE_SIZE = 112
# blobFromImage(img, 1/128.0, size, (127.5, 127.5, 127.5), swapRB=True) --
# i.e. RGB, NCHW, (x - 127.5) / 128.0. NOT SFace's preprocessing, which the
# OpenCV wrapper does internally and which nothing here may reuse.
ARCFACE_MEAN = 127.5
ARCFACE_STD = 128.0
EMBED_DIM = 512


def pad_to_stride(height: int, width: int, align: int = STRIDE_ALIGN) -> tuple:
    """``(padded_height, padded_width)``, rounded UP to a multiple of ``align``.

    PADDING, NOT RESIZING, and the difference is the whole reason this
    function exists rather than a second detect size. The rig maps detector
    pixels back to capture pixels with one scale per axis, taken from
    ``detector.input_size``. If the frame were resized to 320x192 instead of
    320x180, a 16:9 capture would gain a 1.067x vertical stretch that the rig
    would not know about and would silently attribute to the face. Padding the
    bottom with zeros leaves rows 0..179 holding exactly the pixels a
    320x180 resize produces, so every coordinate the graph returns is already
    in 320x180 space and needs no correction at all.
    """
    h, w = int(height), int(width)
    if h <= 0 or w <= 0:
        raise ValueError("frame size must be positive, got %dx%d" % (w, h))
    a = int(align)
    return (-(-h // a) * a, -(-w // a) * a)


def anchor_centres(height: int, width: int, stride: int,
                   num_anchors: int = NUM_ANCHORS) -> np.ndarray:
    """The (K*num_anchors, 2) grid of cell centres, in INPUT pixels.

    Row order is the graph's: y outer, x inner, then the anchors of one cell
    adjacent. That ordering is not cosmetic -- it is what pairs row i of the
    score tensor with the right cell, and getting it transposed puts every
    detection at the mirror of where it belongs while still looking like a
    plausible set of boxes.
    """
    h, w = int(height) // int(stride), int(width) // int(stride)
    if h <= 0 or w <= 0:
        raise ValueError("stride %d does not fit a %dx%d input"
                         % (stride, width, height))
    ys, xs = np.mgrid[:h, :w]
    centres = np.stack([xs, ys], axis=-1).astype(np.float32) * float(stride)
    centres = centres.reshape((-1, 2))
    n = int(num_anchors)
    if n > 1:
        centres = np.stack([centres] * n, axis=1).reshape((-1, 2))
    return centres


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """(N,2) centres + (N,4) left/top/right/bottom distances -> (N,4) xyxy."""
    points = np.asarray(points, dtype=np.float32)
    distance = np.asarray(distance, dtype=np.float32)
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """(N,2) centres + (N,10) dx/dy offsets -> (N,5,2) points.

    The offsets are SIGNED and relative to the same cell centre for all five
    points, so this is a plain add and not a per-point anchor lookup.
    """
    points = np.asarray(points, dtype=np.float32)
    distance = np.asarray(distance, dtype=np.float32)
    n_pts = distance.shape[1] // 2
    out = np.empty((distance.shape[0], n_pts, 2), dtype=np.float32)
    out[:, :, 0] = points[:, 0:1] + distance[:, 0::2]
    out[:, :, 1] = points[:, 1:2] + distance[:, 1::2]
    return out


def nms(boxes: np.ndarray, scores: np.ndarray,
        iou_threshold: float = NMS_IOU) -> list:
    """Greedy IoU suppression; returns the KEPT indices, best score first.

    Plain numpy rather than ``cv2.dnn.NMSBoxes`` so it is testable with no
    OpenCV, and because the OpenCV call takes xywh where everything here is
    xyxy -- a conversion at exactly the seam where a silent mistake would
    look like "the detector got worse in a crowd".
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).ravel()
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= float(iou_threshold)]
    return keep


def group_outputs(outputs: Sequence[np.ndarray]) -> tuple:
    """Nine raw tensors -> ``(scores, bboxes, kpss)``, each stride-8 first.

    GROUPED BY SHAPE, NOT BY POSITION. The graph's output names are opaque
    ('443', '468', ...) and their order is an export artefact; keying off the
    last dimension (1 = score, 4 = bbox distance, 10 = keypoint distance) and
    then off the row count (largest = finest stride) is a property of what the
    numbers ARE. A re-export that reorders the outputs would silently feed
    bbox distances into the score path under any positional scheme.
    """
    arrs = [np.asarray(o) for o in outputs]
    arrs = [a.reshape(a.shape[-2], a.shape[-1]) if a.ndim == 3 else a
            for a in arrs]
    buckets: dict = {1: [], 4: [], 10: []}
    for a in arrs:
        if a.ndim != 2 or a.shape[1] not in buckets:
            raise ValueError(
                "SCRFD output of shape %r is none of score (*,1), bbox (*,4) "
                "or keypoints (*,10); this graph is not the 9-output "
                "det_500m the decode was written for" % (a.shape,))
        buckets[a.shape[1]].append(a)
    n = len(STRIDES)
    if not all(len(buckets[k]) == n for k in (1, 4, 10)):
        raise ValueError(
            "SCRFD gave %d score / %d bbox / %d keypoint tensors; expected "
            "%d of each (strides %s). A graph with no keypoint head cannot "
            "drive this pipeline: visionrig indexes five landmarks."
            % (len(buckets[1]), len(buckets[4]), len(buckets[10]), n,
               ",".join(str(s) for s in STRIDES)))
    order = lambda group: sorted(group, key=lambda a: -a.shape[0])  # noqa: E731
    return tuple(order(buckets[k]) for k in (1, 4, 10))


def decode(outputs: Sequence[np.ndarray], input_width: int, input_height: int,
           score_threshold: float, strides: Sequence[int] = STRIDES,
           num_anchors: int = NUM_ANCHORS,
           iou_threshold: float = NMS_IOU) -> tuple:
    """Raw SCRFD tensors -> ``(boxes_xyxy, kps, scores)`` after NMS.

    ``boxes_xyxy`` is (N,4), ``kps`` is (N,5,2), ``scores`` is (N,), all in
    INPUT pixels and sorted best first. N may be 0; that is an empty room, and
    the caller -- not this function -- is the one that must never confuse it
    with a blind detector.
    """
    scores_t, bbox_t, kps_t = group_outputs(outputs)
    all_boxes, all_kps, all_scores = [], [], []
    for idx, stride in enumerate(strides):
        score = np.asarray(scores_t[idx], dtype=np.float32).ravel()
        bbox = np.asarray(bbox_t[idx], dtype=np.float32) * float(stride)
        kps = np.asarray(kps_t[idx], dtype=np.float32) * float(stride)
        centres = anchor_centres(input_height, input_width, stride,
                                 num_anchors)
        if centres.shape[0] != score.shape[0]:
            raise ValueError(
                "stride %d: the graph returned %d rows but a %dx%d input has "
                "%d anchors. The input size and the graph disagree, and "
                "decoding anyway would place every face at the wrong cell."
                % (stride, score.shape[0], input_width, input_height,
                   centres.shape[0]))
        hits = np.nonzero(score >= float(score_threshold))[0]
        if hits.size == 0:
            continue
        all_boxes.append(distance2bbox(centres[hits], bbox[hits]))
        all_kps.append(distance2kps(centres[hits], kps[hits]))
        all_scores.append(score[hits])
    if not all_scores:
        return (np.zeros((0, 4), np.float32), np.zeros((0, 5, 2), np.float32),
                np.zeros((0,), np.float32))
    boxes = np.vstack(all_boxes)
    kps = np.vstack(all_kps)
    scores = np.concatenate(all_scores)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], kps[keep], scores[keep]


def to_detect_rows(boxes: np.ndarray, kps: np.ndarray,
                   scores: np.ndarray) -> np.ndarray:
    """``(N,15)`` rows in YuNet's layout, so nothing downstream changes.

    x, y, w, h, then five (x, y) pairs, then the score -- and the five pairs
    are handed over in the order they arrived, which the module docstring
    establishes is already YuNet's. NO REORDERING HAPPENS HERE, deliberately:
    a permutation buried in a conversion helper is precisely the silent
    yaw-inverting bug this file is written against, so if the order were ever
    wrong the fix belongs at the source and not here.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    kps = np.asarray(kps, dtype=np.float32).reshape(-1, 5, 2)
    scores = np.asarray(scores, dtype=np.float32).ravel()
    n = boxes.shape[0]
    if kps.shape[0] != n or scores.shape[0] != n:
        raise ValueError("boxes/kps/scores disagree: %d/%d/%d"
                         % (n, kps.shape[0], scores.shape[0]))
    rows = np.zeros((n, 15), dtype=np.float32)
    rows[:, 0] = boxes[:, 0]
    rows[:, 1] = boxes[:, 1]
    rows[:, 2] = boxes[:, 2] - boxes[:, 0]
    rows[:, 3] = boxes[:, 3] - boxes[:, 1]
    rows[:, 4:14] = kps.reshape(n, 10)
    rows[:, 14] = scores
    return rows


# ------------------------------------------------------- ArcFace alignment
def similarity_transform(src, dst) -> np.ndarray:
    """Least-squares similarity (rotation, uniform scale, translation).

    Umeyama, 1991 -- the same estimator ``skimage.transform.SimilarityTransform``
    runs, reimplemented in fifteen lines of numpy so this module needs neither
    scikit-image nor OpenCV. Uniform scale and rotation ONLY: an affine fit
    would also shear and stretch the face to hit the template, which changes
    the very geometry the embedding is measuring.

    Returns the 2x3 matrix ``M`` with ``dst ~= M @ [x, y, 1]``.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if src.shape != dst.shape or src.shape[0] < 2:
        raise ValueError("need matching point sets of at least 2 points, got "
                         "%r and %r" % (src.shape, dst.shape))
    n = src.shape[0]
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_d, dst_d = src - src_mean, dst - dst_mean
    cov = (dst_d.T @ src_d) / n
    d = np.ones(2, dtype=np.float64)
    if np.linalg.det(cov) < 0:
        d[1] = -1.0
    u, s, vt = np.linalg.svd(cov)
    rank = int(np.linalg.matrix_rank(cov))
    if rank == 0:
        raise ValueError("degenerate landmark set: every point is the same "
                         "place, so there is no transform to estimate")
    if rank == 1:
        if np.linalg.det(u) * np.linalg.det(vt) > 0:
            rot = u @ vt
        else:
            keep = d[1]
            d[1] = -1.0
            rot = u @ np.diag(d) @ vt
            d[1] = keep
    else:
        rot = u @ np.diag(d) @ vt
    var_src = float(src_d.var(axis=0).sum())
    scale = 1.0 if var_src == 0.0 else float(s @ d) / var_src
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rot
    matrix[:, 2] = dst_mean - scale * (rot @ src_mean)
    return matrix


def arcface_matrix(landmarks, image_size: int = ARCFACE_SIZE) -> np.ndarray:
    """The 2x3 warp that puts these five points on the ArcFace template.

    ``image_size`` follows insightface: a multiple of 112 scales the template
    directly; a multiple of 128 scales by size/128 and shifts x by 8*ratio.
    Only 112 is used here, and anything else raises rather than quietly
    producing a crop the checkpoint was not trained on.
    """
    lmk = np.asarray(landmarks, dtype=np.float64).reshape(-1, 2)
    if lmk.shape != (5, 2):
        raise ValueError("ArcFace alignment needs exactly 5 points, got %r"
                         % (lmk.shape,))
    size = int(image_size)
    if size % 112 == 0:
        ratio, diff_x = float(size) / 112.0, 0.0
    elif size % 128 == 0:
        ratio, diff_x = float(size) / 128.0, 8.0 * (float(size) / 128.0)
    else:
        raise ValueError("ArcFace crop size must be a multiple of 112 or 128, "
                         "not %d" % size)
    dst = ARCFACE_TEMPLATE * ratio
    dst = dst + np.array([diff_x, 0.0])
    return similarity_transform(lmk, dst)


def model_blob(image_bgr) -> np.ndarray:
    """An HxWx3 BGR image -> the (1,3,H,W) float32 both graphs want.

    ONE implementation for both models, because both use exactly the same
    preprocessing -- ``blobFromImage(img, 1/128.0, size, (127.5,)*3,
    swapRB=True)`` -- and two copies of an arithmetic that has no way to
    announce a mistake is how they drift apart.

    BGR IN, RGB OUT. cv2 hands frames over as BGR and these checkpoints were
    trained on RGB; insightface gets there with ``swapRB=True`` inside
    ``blobFromImage``. Doing it explicitly here means the channel order is
    visible in a diff instead of being a keyword three layers down -- and a
    swapped-channel embedding is another failure with no exception attached.
    """
    arr = np.asarray(image_bgr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("the graph wants an HxWx3 BGR image, got shape %r"
                         % (arr.shape,))
    rgb = arr[:, :, ::-1].astype(np.float32)
    rgb = (rgb - ARCFACE_MEAN) / ARCFACE_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...],
                                dtype=np.float32)


def arcface_blob(crop_bgr) -> np.ndarray:
    """``model_blob`` plus the size check the recogniser must not skip.

    The detector accepts any padded size; ArcFace accepts exactly 112x112,
    because that is the crop the checkpoint was trained on and a resized
    one still produces 512 plausible floats.
    """
    arr = np.asarray(crop_bgr)
    if arr.ndim == 3 and arr.shape[:2] != (ARCFACE_SIZE, ARCFACE_SIZE):
        raise ValueError("ArcFace wants a %dx%d crop, got %dx%d"
                         % (ARCFACE_SIZE, ARCFACE_SIZE, arr.shape[1],
                            arr.shape[0]))
    return model_blob(arr)


def scale_landmarks(kps, scale_x: float, scale_y: float) -> np.ndarray:
    """Detector-space landmarks -> capture-space landmarks.

    THIS IS NOT DECORATION, IT IS THE BUG THE ARCFACE PATH IS BUILT NOT TO
    INHERIT. Every call site in this app passes the FULL capture frame and a
    row in DETECTOR pixels: ``visionrig.run``, ``eye.FaceIdentity.identify``,
    ``campreview._match`` and ``faceenrol.observe_frame`` all do
    ``recogniser.embed(frame, row)`` where ``rows = detector.detect(frame)``
    returned coordinates in the 320x180 space the detector was handed, while
    ``frame`` is 1280x720. Measured on this box 2026-09-03 with synthetic
    arrays: ``cv2.FaceRecognizerSF.alignCrop`` given that pair reads pixels
    from the frame's TOP-LEFT 320x180 corner (a frame bright only there gave
    a crop of mean 160; a frame bright only where the face actually was gave
    a crop of mean 0.00). The crop is of the wrong part of the picture, and
    it warps whatever is there by the face's geometry -- which is why it does
    not fail loudly.

    ArcFace here scales first, so the crop comes from where the face is.
    """
    arr = np.asarray(kps, dtype=np.float64).reshape(-1, 2).copy()
    arr[:, 0] *= float(scale_x)
    arr[:, 1] *= float(scale_y)
    return arr


def normalise(vec) -> np.ndarray:
    """Unit-length, or the zero vector unchanged.

    ArcFace's raw output is not unit length. ``facegallery.cosine`` normalises
    anyway, so this is not load-bearing for matching -- it is load-bearing for
    STORAGE: normalising once at the source means every stored vector has the
    same magnitude, so a mean or a std over the gallery means something.
    A zero vector is returned as-is rather than divided, for the same reason
    ``facegallery.cosine`` answers 0.0: NaN spreads and reads as "no match"
    in one place and poisons an average in another.
    """
    arr = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or not np.isfinite(norm):
        return arr
    return (arr / norm).astype(np.float32)
