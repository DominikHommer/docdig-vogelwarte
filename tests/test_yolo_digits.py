"""Tests for the YOLO digit reader — pure composition logic, no model needed.

`compose_detections` turns raw (boxes, classes, confidences) arrays into a
digit string; everything here runs without ultralytics or the checkpoint.
"""

import pytest

from libs.yolo_digits import DEFAULT_MODEL_PATHS, YoloDigitReader, compose_detections


def _box(x, w=20, y=10, h=30):
    return [x, y, x + w, y + h]


def test_compose_orders_digits_left_to_right():
    # Detections arrive in arbitrary order; result must follow x-position.
    boxes = [_box(60), _box(10), _box(35)]
    classes = [5, 1, 4]
    confs = [0.9, 0.95, 0.85]
    digits, conf = compose_detections(boxes, classes, confs)
    assert digits == "145"
    assert 0.85 <= conf <= 0.95


def test_compose_drops_low_confidence_detections():
    boxes = [_box(10), _box(40)]
    classes = [7, 3]
    confs = [0.9, 0.1]
    digits, _ = compose_detections(boxes, classes, confs, conf_threshold=0.30)
    assert digits == "7"


def test_compose_empty_input():
    assert compose_detections([], [], []) == ("", 0.0)


def test_compose_all_below_threshold():
    digits, conf = compose_detections([_box(10)], [5], [0.05])
    assert digits == ""
    assert conf == 0.0


def test_compose_resolves_cross_class_overlap_by_confidence():
    # A "1" and a "7" detected on the same stroke — the more confident wins.
    # (YOLO's NMS is per-class, so this duplicate survives to our layer.)
    boxes = [_box(10), _box(12), _box(50)]
    classes = [1, 7, 0]
    confs = [0.6, 0.9, 0.8]
    digits, _ = compose_detections(boxes, classes, confs)
    assert digits == "70"


def test_compose_keeps_adjacent_non_overlapping_digits():
    # Two digits close together but NOT overlapping must both survive.
    boxes = [_box(10, w=18), _box(30, w=18)]
    classes = [4, 2]
    confs = [0.9, 0.9]
    digits, _ = compose_detections(boxes, classes, confs)
    assert digits == "42"


def test_compose_ignores_out_of_range_classes():
    boxes = [_box(10), _box(40)]
    classes = [3, 11]  # 11 is not a digit class
    confs = [0.9, 0.9]
    digits, _ = compose_detections(boxes, classes, confs)
    assert digits == "3"


def test_compose_multi_digit_ring_number():
    # Simulates a bague cell: "142921" written across the cell.
    xs = [5, 30, 55, 80, 105, 130]
    boxes = [_box(x, w=20) for x in xs]
    classes = [1, 4, 2, 9, 2, 1]
    confs = [0.8] * 6
    digits, conf = compose_detections(boxes, classes, confs)
    assert digits == "142921"
    assert conf == pytest.approx(0.8)


def test_reader_without_model_is_noop(tmp_path, monkeypatch):
    # Point the reader at a non-existent checkpoint — read() must degrade
    # gracefully instead of raising.
    monkeypatch.chdir(tmp_path)  # hide ./config/digit_yolo.pt
    reader = YoloDigitReader(model_path=str(tmp_path / "missing.pt"))
    assert reader.ensure_loaded() is False

    import numpy as np

    text, conf = reader.read(np.full((40, 120), 255, dtype=np.uint8))
    assert text == ""
    assert conf == 0.0


def test_default_model_path_is_config_digit_yolo():
    assert DEFAULT_MODEL_PATHS[0] == "./config/digit_yolo.pt"
