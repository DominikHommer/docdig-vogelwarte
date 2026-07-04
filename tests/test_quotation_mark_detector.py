"""Tests for QuotationMarkDetector.

We use synthetic numpy images (mostly white vs. mostly inked) so the test
does not depend on real scan data.
"""

import numpy as np

from modules.quotation_mark_detector import QuotationMarkDetector
from .conftest import make_batch_column, make_pages


def _white_cell(shape=(40, 100)):
    return np.full(shape, 255, dtype=np.uint8)


def _inked_cell(shape=(40, 100)):
    img = np.full(shape, 255, dtype=np.uint8)
    img[10:30, 20:80] = 0  # ~30% dark pixels — well above threshold
    return img


def _species_column(cells_payload):
    cells = [{"image": _inked_cell(), "image_raw": _inked_cell(), "skip_ocr": False, "erkannt": "", "score": -1}]
    for img in cells_payload:
        cells.append(
            {"image": img, "image_raw": img, "skip_ocr": False, "erkannt": "", "score": -1}
        )
    return {
        "cells": cells,
        "is_species_column": True,
        "is_batch_column": False,
        "is_sexe_column": False,
        "is_age_column": False,
        "is_jour-mois_column": False,
        "is_heure_column": False,
        "is_alle_column": False,
        "is_poids_column": False,
    }


def test_fully_white_cells_are_blank_not_ditto():
    """A completely inkless cell is EMPTY — 'same as above' needs an actual
    mark. The detector must flag it blank instead of inventing a ditto."""
    pages = make_pages([_species_column([_white_cell(), _inked_cell(), _white_cell()])])
    QuotationMarkDetector().process({"cell-formatter": pages}, {})

    cells = pages[0]["columns"][0]["cells"]
    assert cells[1]["erkannt"] == ""
    assert cells[1]["is_blank"] is True
    assert cells[1]["skip_ocr"] is True  # no recognizer may touch it
    assert cells[2]["skip_ocr"] is False  # inked cell — real content
    assert cells[3]["erkannt"] == ""
    assert cells[3]["is_blank"] is True


def test_header_row_is_never_quote():
    """Cell index 0 is always treated as the header."""
    pages = make_pages([_species_column([])])
    # Replace header with a white cell to be certain we don't mark it.
    pages[0]["columns"][0]["cells"][0]["image"] = _white_cell()

    QuotationMarkDetector().process({"cell-formatter": pages}, {})

    cells = pages[0]["columns"][0]["cells"]
    assert cells[0]["erkannt"] == ""
    assert cells[0]["skip_ocr"] is False


def test_batch_column_skipped_entirely():
    """The bague column should never have ditto detection applied."""
    column = make_batch_column(num_cells=2)
    # Replace data cells with mostly white images.
    for cell in column["cells"][1:]:
        cell["image"] = _white_cell()

    pages = make_pages([column])
    QuotationMarkDetector().process({"cell-formatter": pages}, {})

    for cell in column["cells"][1:]:
        assert cell["erkannt"] == ""
        assert cell["skip_ocr"] is False


# ---------------------------------------------------------------------------
# Shape-based detection for the tintier ditto notations ('', //, ii)
# files-3/meta.json lists four notations: "  ''  //  ii
# ---------------------------------------------------------------------------


def _ditto_mark_cell(kind: str, shape=(48, 160)):
    """Synthetic ditto marks: small ink blobs centred in the cell."""
    img = np.full(shape, 255, dtype=np.uint8)
    h, w = shape
    cx = w // 2
    cy = h // 2
    if kind == '"':
        img[cy - 6 : cy + 2, cx - 4 : cx - 2] = 0
        img[cy - 6 : cy + 2, cx + 2 : cx + 4] = 0
    elif kind == "''":
        img[cy - 8 : cy + 4, cx - 5 : cx - 3] = 0
        img[cy - 8 : cy + 4, cx + 3 : cx + 5] = 0
    elif kind == "//":
        for i in range(12):
            img[cy + 6 - i, cx - 6 + i : cx - 4 + i] = 0
            img[cy + 6 - i, cx + 1 + i : cx + 3 + i] = 0
    elif kind == "ii":
        img[cy - 4 : cy + 6, cx - 5 : cx - 3] = 0
        img[cy - 8 : cy - 6, cx - 5 : cx - 3] = 0
        img[cy - 4 : cy + 6, cx + 3 : cx + 5] = 0
        img[cy - 8 : cy - 6, cx + 3 : cx + 5] = 0
    return img


def _species_word_cell(shape=(48, 160)):
    """Simulates a handwritten species name spanning most of the width."""
    img = np.full(shape, 255, dtype=np.uint8)
    img[16:34, 12:148] = 0
    return img


def test_all_ditto_notations_detected_in_species_column():
    marks = ['"', "''", "//", "ii"]
    pages = make_pages(
        [_species_column([_ditto_mark_cell(kind) for kind in marks])]
    )
    QuotationMarkDetector().process({"cell-formatter": pages}, {})

    cells = pages[0]["columns"][0]["cells"]
    for kind, cell in zip(marks, cells[1:]):
        assert cell["erkannt"] == '"', f"notation {kind!r} not detected as ditto"
        assert cell["skip_ocr"] is True


def test_species_word_is_not_ditto():
    pages = make_pages([_species_column([_species_word_cell()])])
    QuotationMarkDetector().process({"cell-formatter": pages}, {})

    cell = pages[0]["columns"][0]["cells"][1]
    assert cell["erkannt"] == ""
    assert cell["skip_ocr"] is False


def _centred_digit_cell(shape=(48, 160)):
    """A bold handwritten digit: narrow + centred (like a ditto), but with
    more ink than the ratio threshold allows (>5% of the cell)."""
    img = np.full(shape, 255, dtype=np.uint8)
    h, w = shape
    img[12:36, w // 2 - 10 : w // 2 + 10] = 0  # 24x20 = 6.25% ink
    return img


def test_shape_detection_not_applied_to_numeric_columns():
    """A single centred digit on Aile/Poids looks like a ditto blob — the
    shape check must stay off there so real values survive."""
    column = _species_column([_centred_digit_cell()])
    column["is_species_column"] = False
    column["is_alle_column"] = True

    pages = make_pages([column])
    QuotationMarkDetector().process({"cell-formatter": pages}, {})

    cell = pages[0]["columns"][0]["cells"][1]
    assert cell["erkannt"] == ""
    assert cell["skip_ocr"] is False


def test_same_blob_would_be_ditto_in_species_column():
    """Sanity contrast to the numeric test: the identical narrow centred blob
    IS a ditto when the column's regular content is wide (species names)."""
    pages = make_pages([_species_column([_centred_digit_cell()])])
    QuotationMarkDetector().process({"cell-formatter": pages}, {})

    cell = pages[0]["columns"][0]["cells"][1]
    assert cell["erkannt"] == '"'
    assert cell["skip_ocr"] is True
