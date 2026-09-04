"""SCRFD's decode and ArcFace's alignment, pinned without a model or a lens.

Every failure this file guards against is SILENT. A transposed anchor grid, a
missing stride multiply, a permuted landmark pair and a wrong alignment
template all produce well-formed output that is simply wrong, so none of them
can be caught by "did it raise" and none of them can be caught by looking --
which is forbidden here anyway. What is pinned:

* the alignment template is InsightFace's ``arcface_dst``, digit for digit,
  and its own x-coordinates prove which point is which eye;
* the similarity transform is exact on a round trip: a known rotate, scale and
  translate applied to the template is undone to within 1e-9;
* the anchor grid is y-outer, x-inner, with the two anchors of a cell
  adjacent, and a planted score at one known cell decodes to one known box;
* the outputs are grouped by SHAPE, so a re-export that reorders them still
  decodes -- and a graph with no keypoint head is refused by name;
* rows come out in YuNet's layout, and driving ``visionrig``'s own geometry
  with them yields the roll, yaw sign and mouth drop a real row would.

No camera, no cv2, no onnxruntime, no weights.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from jarvis import faceinsight as fi
from jarvis import visionrig as vr


# ------------------------------------------------------------- the template
def test_the_template_is_insightfaces_arcface_dst_digit_for_digit():
    """Read 2026-09-03 from insightface/utils/face_align.py. A template that
    drifts from the one the checkpoint was trained against loses accuracy and
    raises nothing."""
    expected = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                [41.5493, 92.3655], [70.7299, 92.2041]]
    assert fi.ARCFACE_TEMPLATE.tolist() == expected
    assert fi.ARCFACE_SIZE == 112
    assert (fi.ARCFACE_MEAN, fi.ARCFACE_STD) == (127.5, 128.0)


def test_the_template_itself_says_which_point_is_which():
    """The landmark ORDER is load-bearing: visionrig indexes these five
    positionally, so a swap inverts every yaw with nothing to notice it. The
    template's own coordinates are the evidence, not a comment."""
    t = fi.ARCFACE_TEMPLATE
    # 0 and 1 are the eyes, 0 on the image-left -- which is the PERSON's
    # right, exactly YuNet's IDX_EYE_R.
    assert t[0][0] < t[1][0]
    # 3 and 4 are the mouth corners, same handedness as the eyes.
    assert t[3][0] < t[4][0]
    # The nose sits between the eyes horizontally and below them vertically.
    assert t[0][0] < t[2][0] < t[1][0]
    assert t[2][1] > t[0][1] and t[2][1] > t[1][1]
    # The mouth is below the nose, which is below the eyes.
    assert t[3][1] > t[2][1] and t[4][1] > t[2][1]


def test_the_index_order_matches_the_one_visionrig_indexes():
    """A structural check, so a future edit to either side breaks here."""
    assert (vr.IDX_EYE_R, vr.IDX_EYE_L, vr.IDX_NOSE) == (4, 6, 8)
    assert (vr.IDX_MOUTH_R, vr.IDX_MOUTH_L) == (10, 12)
    rows = fi.to_detect_rows(np.array([[0.0, 0.0, 112.0, 112.0]]),
                             fi.ARCFACE_TEMPLATE[None, :, :],
                             np.array([0.9]))
    row = rows[0]
    assert row[vr.IDX_EYE_R] < row[vr.IDX_EYE_L]          # image-left first
    assert row[vr.IDX_MOUTH_R] < row[vr.IDX_MOUTH_L]


# ---------------------------------------------------------- the transform
def test_the_transform_of_a_set_onto_itself_is_the_identity():
    m = fi.similarity_transform(fi.ARCFACE_TEMPLATE, fi.ARCFACE_TEMPLATE)
    assert np.allclose(m, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                       atol=1e-9)


def _apply(matrix, points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return pts @ np.asarray(matrix)[:, :2].T + np.asarray(matrix)[:, 2]


@pytest.mark.parametrize("angle_deg,scale,tx,ty",
                         [(0.0, 1.0, 0.0, 0.0), (30.0, 2.5, -17.0, 9.0),
                          (-115.0, 0.3, 300.0, -50.0)])
def test_a_known_similarity_is_undone_exactly(angle_deg, scale, tx, ty):
    """Rotation, uniform scale and translation only. If the estimator were
    fitting a general affine it would also absorb a shear, and the face would
    be stretched onto the template rather than placed on it."""
    th = math.radians(angle_deg)
    rot = np.array([[math.cos(th), -math.sin(th)],
                    [math.sin(th), math.cos(th)]]) * scale
    moved = fi.ARCFACE_TEMPLATE @ rot.T + np.array([tx, ty])
    back = fi.similarity_transform(moved, fi.ARCFACE_TEMPLATE)
    assert np.allclose(_apply(back, moved), fi.ARCFACE_TEMPLATE, atol=1e-9)


def test_a_sheared_face_is_not_fitted_by_a_shear():
    """The estimator must REFUSE to absorb a shear -- it may only rotate and
    scale -- so a sheared input does not land back on the template."""
    shear = np.array([[1.0, 0.45], [0.0, 1.0]])
    moved = fi.ARCFACE_TEMPLATE @ shear.T
    back = fi.similarity_transform(moved, fi.ARCFACE_TEMPLATE)
    residual = np.abs(_apply(back, moved) - fi.ARCFACE_TEMPLATE).max()
    assert residual > 1.0


def test_the_transform_refuses_a_collapsed_landmark_set():
    same = np.zeros((5, 2))
    with pytest.raises(ValueError, match="degenerate"):
        fi.similarity_transform(same, fi.ARCFACE_TEMPLATE)


def test_arcface_matrix_is_the_identity_on_the_template():
    m = fi.arcface_matrix(fi.ARCFACE_TEMPLATE)
    assert np.allclose(_apply(m, fi.ARCFACE_TEMPLATE), fi.ARCFACE_TEMPLATE,
                       atol=1e-9)


def test_arcface_matrix_maps_a_moved_face_back_onto_the_template():
    th = math.radians(22.0)
    rot = np.array([[math.cos(th), -math.sin(th)],
                    [math.sin(th), math.cos(th)]]) * 3.1
    moved = fi.ARCFACE_TEMPLATE @ rot.T + np.array([412.0, 133.0])
    m = fi.arcface_matrix(moved)
    assert np.allclose(_apply(m, moved), fi.ARCFACE_TEMPLATE, atol=1e-8)


def test_arcface_matrix_refuses_a_wrong_point_count_and_a_wrong_size():
    with pytest.raises(ValueError, match="exactly 5 points"):
        fi.arcface_matrix(fi.ARCFACE_TEMPLATE[:4])
    with pytest.raises(ValueError, match="multiple of 112 or 128"):
        fi.arcface_matrix(fi.ARCFACE_TEMPLATE, image_size=100)


def test_a_224_crop_scales_the_template_rather_than_offsetting_it():
    m = fi.arcface_matrix(fi.ARCFACE_TEMPLATE, image_size=224)
    assert np.allclose(_apply(m, fi.ARCFACE_TEMPLATE),
                       fi.ARCFACE_TEMPLATE * 2.0, atol=1e-8)


# -------------------------------------------------------- the preprocessing
def test_the_blob_is_rgb_nchw_and_scaled_the_way_arcface_wants():
    """SFace's preprocessing is OpenCV's business and is NOT this. A crop fed
    with BGR channels, or with SFace's scaling, still yields 512 finite floats
    that normalise and compare -- so nothing downstream can catch it."""
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    crop[:, :, 0] = 10        # B
    crop[:, :, 1] = 127       # G
    crop[:, :, 2] = 250       # R
    blob = fi.arcface_blob(crop)
    assert blob.shape == (1, 3, 112, 112)
    assert blob.dtype == np.float32
    # channel 0 of the blob must be the RED plane, not the blue one
    assert blob[0, 0].min() == pytest.approx((250 - 127.5) / 128.0)
    assert blob[0, 1].min() == pytest.approx((127 - 127.5) / 128.0)
    assert blob[0, 2].min() == pytest.approx((10 - 127.5) / 128.0)


def test_the_blob_refuses_anything_that_is_not_a_112_square():
    with pytest.raises(ValueError, match="112x112"):
        fi.arcface_blob(np.zeros((96, 96, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="HxWx3"):
        fi.arcface_blob(np.zeros((112, 112), dtype=np.uint8))


def test_normalise_is_unit_length_and_leaves_a_zero_vector_alone():
    v = fi.normalise(np.array([3.0, 4.0], dtype=np.float32))
    assert float(np.linalg.norm(v)) == pytest.approx(1.0)
    z = fi.normalise(np.zeros(8, dtype=np.float32))
    assert float(np.linalg.norm(z)) == 0.0
    assert np.all(np.isfinite(z))


# ------------------------------------------------------------ the anchors
def test_padding_rounds_up_to_a_multiple_of_the_coarsest_stride():
    """320x180 is the shipped detect size and 180 is not divisible by 32, so
    the graph cannot be handed it as it stands."""
    assert fi.pad_to_stride(180, 320) == (192, 320)
    assert fi.pad_to_stride(192, 320) == (192, 320)
    assert fi.pad_to_stride(240, 320) == (256, 320)
    with pytest.raises(ValueError):
        fi.pad_to_stride(0, 320)


def test_the_anchor_grid_is_y_outer_x_inner_with_the_pair_adjacent():
    c = fi.anchor_centres(192, 320, 32, num_anchors=2)
    assert c.shape == (10 * 6 * 2, 2)
    # cell (0,0) twice, then cell (1,0) twice -- x advances first.
    assert c[0].tolist() == [0.0, 0.0]
    assert c[1].tolist() == [0.0, 0.0]
    assert c[2].tolist() == [32.0, 0.0]
    # the first row of cells is 10 wide, so index 20 starts the second row
    assert c[20].tolist() == [0.0, 32.0]
    # and one anchor is the same grid without the duplication
    single = fi.anchor_centres(192, 320, 32, num_anchors=1)
    assert single.shape == (60, 2)
    assert single[1].tolist() == [32.0, 0.0]


def test_the_grid_row_count_matches_what_the_graph_returns():
    """Measured on this box 2026-09-03 by running det_500m.onnx on a
    synthetic 320x192 blob: 1920 / 480 / 120 rows."""
    for stride, rows in ((8, 1920), (16, 480), (32, 120)):
        assert fi.anchor_centres(192, 320, stride).shape[0] == rows


def test_distance_decoding_is_the_closed_form():
    pts = np.array([[100.0, 50.0]])
    box = fi.distance2bbox(pts, np.array([[10.0, 5.0, 20.0, 25.0]]))
    assert box.tolist() == [[90.0, 45.0, 120.0, 75.0]]
    kps = fi.distance2kps(pts, np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                          7.0, 8.0, 9.0, 10.0]]))
    assert kps.shape == (1, 5, 2)
    assert kps[0].tolist() == [[101.0, 52.0], [103.0, 54.0], [105.0, 56.0],
                               [107.0, 58.0], [109.0, 60.0]]


# ---------------------------------------------------------------- the NMS
def test_nms_keeps_the_best_of_an_overlapping_pair_and_both_of_a_far_pair():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]],
                     dtype=np.float32)
    scores = np.array([0.6, 0.9, 0.7], dtype=np.float32)
    keep = fi.nms(boxes, scores, 0.4)
    assert keep == [1, 2]          # best first, the 0.6 duplicate suppressed
    assert fi.nms(np.zeros((0, 4)), np.zeros((0,))) == []


def test_nms_keeps_two_faces_that_merely_touch():
    boxes = np.array([[0, 0, 10, 10], [9, 0, 19, 10]], dtype=np.float32)
    keep = fi.nms(boxes, np.array([0.9, 0.8], dtype=np.float32), 0.4)
    assert sorted(keep) == [0, 1]


# ------------------------------------------------------------- the decode
def _blank_outputs(width=320, height=192):
    """Nine all-zero tensors shaped exactly as det_500m returns them."""
    out = []
    for cols in (1, 4, 10):
        for stride in fi.STRIDES:
            n = (height // stride) * (width // stride) * fi.NUM_ANCHORS
            out.append(np.zeros((n, cols), dtype=np.float32))
    return out


def test_a_planted_hit_decodes_to_exactly_the_cell_it_was_planted_in():
    """One score at stride 16, cell (x=5, y=3), anchor 0. The grid is 20 wide,
    so that is row ((3*20 + 5) * 2) = 130, and the centre is (80, 48).
    Distances are in STRIDE units and must be multiplied by 16."""
    outs = _blank_outputs()
    scores16, bbox16, kps16 = outs[1], outs[4], outs[7]
    row = (3 * 20 + 5) * fi.NUM_ANCHORS
    scores16[row, 0] = 0.87
    bbox16[row] = [1.0, 1.0, 1.0, 1.0]
    kps16[row] = [-0.5, -0.5, 0.5, -0.5, 0.0, 0.0, -0.4, 0.6, 0.4, 0.6]
    boxes, kps, scores = fi.decode(outs, 320, 192, 0.3)
    assert boxes.shape == (1, 4) and kps.shape == (1, 5, 2)
    assert scores.tolist() == [pytest.approx(0.87)]
    assert boxes[0].tolist() == [64.0, 32.0, 96.0, 64.0]
    assert kps[0][0].tolist() == [72.0, 40.0]      # 80-8, 48-8
    assert kps[0][2].tolist() == [80.0, 48.0]      # the nose, at the centre


def test_a_hit_under_the_floor_is_not_decoded_and_an_empty_room_is_empty():
    outs = _blank_outputs()
    outs[1][130, 0] = 0.20
    outs[4][130] = [1.0, 1.0, 1.0, 1.0]
    boxes, kps, scores = fi.decode(outs, 320, 192, 0.3)
    assert boxes.shape == (0, 4) and kps.shape == (0, 5, 2)
    assert scores.shape == (0,)


def test_the_stride_multiply_is_not_optional():
    """Without ``* stride`` a stride-32 box would come out 4x too small and
    still look like a face. Same distances at two strides must give boxes
    whose widths differ by exactly the stride ratio."""
    outs = _blank_outputs()
    outs[0][0, 0] = 0.9            # stride 8, cell (0,0)
    outs[3][0] = [1.0, 1.0, 1.0, 1.0]
    b8, _k, _s = fi.decode(outs, 320, 192, 0.3)
    outs = _blank_outputs()
    outs[2][0, 0] = 0.9            # stride 32, cell (0,0)
    outs[5][0] = [1.0, 1.0, 1.0, 1.0]
    b32, _k, _s = fi.decode(outs, 320, 192, 0.3)
    assert (b32[0][2] - b32[0][0]) == 4.0 * (b8[0][2] - b8[0][0])


def test_decode_refuses_an_input_size_the_tensors_do_not_match():
    """The one mistake that would otherwise place every face at a plausible
    but wrong cell: feeding 320x180 to a graph run at 320x192."""
    outs = _blank_outputs(320, 192)
    with pytest.raises(ValueError, match="anchors"):
        fi.decode(outs, 320, 160, 0.3)


def test_outputs_are_grouped_by_shape_not_by_position():
    outs = _blank_outputs()
    outs[1][130, 0] = 0.9
    outs[4][130] = [1.0, 1.0, 1.0, 1.0]
    shuffled = [outs[i] for i in (8, 3, 0, 6, 4, 1, 7, 5, 2)]
    a = fi.decode(outs, 320, 192, 0.3)
    b = fi.decode(shuffled, 320, 192, 0.3)
    assert a[0].tolist() == b[0].tolist()
    assert a[2].tolist() == b[2].tolist()


def test_a_graph_with_no_keypoint_head_is_refused_by_name():
    """A six-output SCRFD export exists and would decode boxes happily. It
    cannot drive this pipeline: visionrig needs five landmarks to tell a face
    from a fist, and identity needs them to align a crop."""
    outs = [o for o in _blank_outputs() if o.shape[1] != 10]
    with pytest.raises(ValueError, match="keypoint"):
        fi.decode(outs, 320, 192, 0.3)


def test_an_unrecognised_output_shape_says_so(caplog):
    with pytest.raises(ValueError, match="none of score"):
        fi.group_outputs([np.zeros((10, 7), dtype=np.float32)] * 9)


# ------------------------------------------- the seam back into visionrig
def test_rows_come_out_in_yunets_layout():
    boxes = np.array([[10.0, 20.0, 50.0, 80.0]], dtype=np.float32)
    kps = np.array([[[20.0, 40.0], [40.0, 40.0], [30.0, 50.0],
                     [22.0, 65.0], [38.0, 65.0]]], dtype=np.float32)
    rows = fi.to_detect_rows(boxes, kps, np.array([0.77], dtype=np.float32))
    assert rows.shape == (1, vr.DETECT_COLS)
    assert rows[0][:4].tolist() == [10.0, 20.0, 40.0, 60.0]   # xywh
    assert rows[0][vr.IDX_SCORE] == pytest.approx(0.77)
    assert rows[0][4:14].tolist() == kps.reshape(10).tolist()


def test_a_frontal_scrfd_row_reads_as_a_face_to_visionrigs_own_geometry():
    """The end of the ordering argument: the same functions that judged
    YuNet's rows must judge these, and reach the right answers."""
    boxes = np.array([[0.0, 0.0, 112.0, 112.0]], dtype=np.float32)
    rows = fi.to_detect_rows(boxes, fi.ARCFACE_TEMPLATE[None, :, :],
                             np.array([0.9], dtype=np.float32))
    t, roll, eye_px, ok = vr.landmark_geometry(rows[0])
    assert ok
    assert abs(roll) < 1.0                      # upright, not 180 deg flipped
    assert eye_px > 30.0
    assert abs(t) < 0.10                        # frontal: nose near the midline
    ratio, drop, shaped = vr.landmark_plausibility(rows[0])
    assert shaped, "the mouth must read as BELOW the eyes"
    assert drop > vr.MOUTH_DROP_MIN_U
    assert 0.2 < ratio < 0.6


def test_a_landmark_swap_would_be_caught_by_that_same_geometry():
    """Proof the previous test can fail: swapping the two eyes flips the roll
    by 180 degrees and turns the mouth drop negative."""
    swapped = fi.ARCFACE_TEMPLATE.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    swapped[[3, 4]] = swapped[[4, 3]]
    rows = fi.to_detect_rows(np.array([[0.0, 0.0, 112.0, 112.0]]),
                             swapped[None, :, :], np.array([0.9]))
    _t, roll, _eye, ok = vr.landmark_geometry(rows[0])
    assert ok and abs(abs(roll) - 180.0) < 1.0
    _ratio, drop, shaped = vr.landmark_plausibility(rows[0])
    assert drop < 0 and not shaped


def test_to_detect_rows_refuses_mismatched_inputs():
    with pytest.raises(ValueError, match="disagree"):
        fi.to_detect_rows(np.zeros((2, 4)), np.zeros((1, 5, 2)),
                          np.zeros((2,)))
