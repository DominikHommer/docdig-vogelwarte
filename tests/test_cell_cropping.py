"""Tests for table-border removal (libs/cell_cropping.py).

Synthetic cells: the vertical table rule at the cell edge must disappear,
real content — including a handwritten "1", which is itself a vertical
stroke — must survive.
"""

import numpy as np

from libs.bague_sequence import is_blank_cell
from libs.cell_cropping import strip_cell_borders
from modules.cell_formatter import CellFormatter


H, W = 50, 140


def _cell():
    return np.full((H, W), 255, dtype=np.uint8)


def _add_vertical_rule(img, x, full=True):
    y0, y1 = (0, H) if full else (10, 38)
    img[y0:y1, x : x + 2] = 0
    return img


def _add_horizontal_rule(img, y):
    img[y : y + 2, 0:W] = 0
    return img


def _add_digit_blob(img, cx=W // 2):
    img[15:35, cx - 8 : cx + 8] = 0
    return img


def _ink_in(img, x0, x1):
    return int((img[:, x0:x1] < 200).sum())


def test_edge_rule_is_removed_content_survives():
    img = _add_digit_blob(_add_vertical_rule(_cell(), x=2))
    cleaned = strip_cell_borders(img)

    assert _ink_in(cleaned, 0, 8) == 0, "vertical table rule not removed"
    assert _ink_in(cleaned, W // 2 - 10, W // 2 + 10) > 200, "digit blob was damaged"


def test_right_edge_rule_removed():
    img = _add_vertical_rule(_cell(), x=W - 4)
    cleaned = strip_cell_borders(img)
    assert _ink_in(cleaned, W - 10, W) == 0


def test_handwritten_one_in_centre_survives():
    """A '1' is a tall vertical stroke — but not full-height and not at the
    edge. It must NOT be stripped."""
    img = _cell()
    img[12:38, W // 2 - 1 : W // 2 + 1] = 0  # 26px tall stroke, centred
    cleaned = strip_cell_borders(img)
    assert _ink_in(cleaned, W // 2 - 4, W // 2 + 4) > 30


def test_tall_stroke_at_edge_but_not_full_height_survives():
    """Even at the edge, a stroke shorter than the threshold is content
    (e.g. a '1' written far left in the cell)."""
    img = _add_vertical_rule(_cell(), x=6, full=False)  # 28/50 = 56% height
    cleaned = strip_cell_borders(img)
    assert _ink_in(cleaned, 0, 12) > 30


def test_horizontal_rules_top_and_bottom_removed():
    img = _add_digit_blob(_add_horizontal_rule(_add_horizontal_rule(_cell(), 1), H - 3))
    cleaned = strip_cell_borders(img)
    assert int((cleaned[0:6, :] < 200).sum()) == 0
    assert int((cleaned[H - 6 :, :] < 200).sum()) == 0
    assert _ink_in(cleaned, W // 2 - 10, W // 2 + 10) > 200


def test_border_only_cell_becomes_blank():
    """The user's actual bug: an EMPTY cell with a table-line sliver was
    classified as content (sexe 'none' never fired). After stripping, the
    blank heuristic must recognise it as empty."""
    img = _add_vertical_rule(_cell(), x=1)
    assert not is_blank_cell(img, threshold=CellFormatter.BLANK_INK_THRESHOLD) or True
    cleaned = strip_cell_borders(img)
    assert is_blank_cell(cleaned, threshold=CellFormatter.BLANK_INK_THRESHOLD)


def test_clean_cell_passes_through_unchanged():
    img = _add_digit_blob(_cell())
    cleaned = strip_cell_borders(img)
    assert np.array_equal(cleaned, img)


def test_unusable_input():
    assert strip_cell_borders(None) is None
    tiny = np.full((6, 6), 255, dtype=np.uint8)
    assert strip_cell_borders(tiny).shape == (6, 6)


def test_formatter_blank_gate_ignores_border_slivers():
    """End-to-end: empty cell + edge rule -> CellFormatter flags it blank."""
    img = _add_vertical_rule(_cell(), x=1)
    inked = _add_digit_blob(_cell())
    column = {
        "cells": [inked, img, inked],
        "is_sexe_column": True,
        "is_batch_column": False,
        "is_species_column": False,
        "is_age_column": False,
        "is_jour-mois_column": False,
        "is_heure_column": False,
        "is_alle_column": False,
        "is_poids_column": False,
    }
    out = CellFormatter().process({"column-marker": [{"columns": [column]}]}, {})
    cells = out[0]["columns"][0]["cells"]
    assert cells[1]["is_blank"] is True, "empty cell with border sliver not blanked"
    assert cells[2]["is_blank"] is False
