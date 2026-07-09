"""Blank-cell gate: empty cells must never receive a prediction.

The gate lives in CellFormatter (sets ``is_blank`` + ``skip_ocr`` before any
recognizer runs); every recognizer additionally guards on the flag. These
tests lock both layers.
"""

import numpy as np
import pytest

from modules.cell_formatter import CellFormatter
from modules.digit_recognizer import DigitRecognizer
from modules.htr_vt_recognizer import HtrVtRecognizer
from modules.numeric_consensus import NumericConsensus
from modules.sexe_classifier import SexeClassifier
from modules.trocr import TrOCR


def _white(shape=(48, 160)):
    return np.full(shape, 255, dtype=np.uint8)


def _inked(shape=(48, 160)):
    img = np.full(shape, 255, dtype=np.uint8)
    img[16:34, 12:148] = 0
    return img


def _ditto(shape=(48, 160)):
    """A `"` mark: little ink, but clearly more than scanner noise."""
    img = np.full(shape, 255, dtype=np.uint8)
    h, w = shape
    img[h // 2 - 6 : h // 2 + 2, w // 2 - 4 : w // 2 - 2] = 0
    img[h // 2 - 6 : h // 2 + 2, w // 2 + 2 : w // 2 + 4] = 0
    return img


def _column_of(images, **flags):
    cells = [np.asarray(_inked())] + [np.asarray(img) for img in images]
    column = {
        "cells": cells,
        "is_batch_column": False,
        "is_species_column": False,
        "is_sexe_column": False,
        "is_age_column": False,
        "is_jour-mois_column": False,
        "is_heure_column": False,
        "is_alle_column": False,
        "is_poids_column": False,
    }
    column.update(flags)
    return column


def _run_formatter(column):
    pages = [{"columns": [column]}]
    return CellFormatter().process({"column-marker": pages}, {})


# ---------------------------------------------------------------------------
# The gate itself (CellFormatter)
# ---------------------------------------------------------------------------


def test_formatter_flags_white_cells_blank():
    out = _run_formatter(_column_of([_white(), _inked(), _white()], is_species_column=True))
    cells = out[0]["columns"][0]["cells"]

    assert cells[1]["is_blank"] is True
    assert cells[1]["skip_ocr"] is True
    assert cells[1]["erkannt"] == ""
    assert cells[2]["is_blank"] is False
    assert cells[2]["skip_ocr"] is False
    assert cells[3]["is_blank"] is True


def test_formatter_keeps_ditto_marks_alive():
    """The gate must be strictly below ditto ink — a `"` mark is NOT blank,
    otherwise the QuotationMarkDetector never sees it."""
    out = _run_formatter(_column_of([_ditto()], is_species_column=True))
    cell = out[0]["columns"][0]["cells"][1]
    assert cell["is_blank"] is False
    assert cell["skip_ocr"] is False


# ---------------------------------------------------------------------------
# Recognizer guards — none of them may write into a blank cell
# ---------------------------------------------------------------------------


def _blank_cell():
    return {
        "image": _white(),
        "image_raw": _white(),
        "erkannt": "",
        "score": -1,
        "skip_ocr": True,
        "is_blank": True,
        "verbesserung": "",
    }


def _page_with(column):
    return [{"columns": [column]}]


def test_htr_vt_skips_blank_cells(monkeypatch):
    recognizer = HtrVtRecognizer()
    monkeypatch.setattr(HtrVtRecognizer, "_ensure_loaded", lambda self: True)
    monkeypatch.setattr(HtrVtRecognizer, "_predict", lambda self, img: "Halluzination")

    column = {"cells": [_blank_cell(), _blank_cell()], "is_species_column": True}
    recognizer.process({"cell-formatter": _page_with(column)}, {})

    assert column["cells"][1]["erkannt"] == ""
    assert "predictions" not in column["cells"][1]


def test_trocr_skips_blank_cells(monkeypatch):
    trocr = TrOCR()
    monkeypatch.setattr(TrOCR, "_ensure_loaded", lambda self: True)
    monkeypatch.setattr(
        TrOCR, "_recognise_batch", lambda self, imgs: ["Halluzination"] * len(imgs)
    )

    column = {"cells": [_blank_cell(), _blank_cell()], "is_alle_column": True}
    trocr.process({"cell-formatter": _page_with(column)}, {})

    assert column["cells"][1]["erkannt"] == ""
    assert "predictions" not in column["cells"][1]


def test_digit_recognizer_skips_blank_cells(monkeypatch):
    monkeypatch.setattr(
        DigitRecognizer, "_tesseract_digits", lambda self, img, psm=7: "99"
    )
    monkeypatch.setattr(DigitRecognizer, "_keras_predict", lambda self, img: "99")
    monkeypatch.setattr(DigitRecognizer, "_yolo_predict", lambda self, img: "99")

    column = {"cells": [_blank_cell(), _blank_cell()], "is_poids_column": True}
    DigitRecognizer(use_tesseract=False, use_yolo=False).process(
        {"cell-formatter": _page_with(column)}, {}
    )

    assert column["cells"][1]["erkannt"] == ""
    assert "predictions" not in column["cells"][1]


def test_sexe_classifier_skips_blank_cells(monkeypatch):
    clf = SexeClassifier()
    monkeypatch.setattr(SexeClassifier, "_ensure_loaded", lambda self: True)

    called = []
    monkeypatch.setattr(
        SexeClassifier, "_prepare", lambda self, img: called.append(1) or None
    )

    column = {"cells": [_blank_cell(), _blank_cell()], "is_sexe_column": True}
    clf.process({"cell-formatter": _page_with(column)}, {})

    assert not called, "SexeClassifier must not even preprocess a blank cell"
    assert column["cells"][1]["erkannt"] == ""


def test_consensus_skips_blank_cells():
    cell = _blank_cell()
    cell["predictions"] = {"trocr": "42"}  # stale prediction must be ignored
    column = {
        "cells": [_blank_cell(), cell],
        "is_alle_column": True,
        "is_poids_column": False,
        "is_heure_column": False,
        "is_jour-mois_column": False,
    }
    NumericConsensus().process({"trocr": _page_with(column)}, {})

    assert cell["erkannt"] == ""
    assert cell["score"] == -1


# ---------------------------------------------------------------------------
# End-to-end through the deterministic stages
# ---------------------------------------------------------------------------


def test_blank_cell_survives_formatter_plus_recognizer_chain(monkeypatch):
    """White cell in -> empty cell out, even with recognizers mocked to
    return garbage at every opportunity."""
    out = _run_formatter(_column_of([_white(), _inked()], is_species_column=True))

    monkeypatch.setattr(HtrVtRecognizer, "_ensure_loaded", lambda self: True)
    monkeypatch.setattr(HtrVtRecognizer, "_predict", lambda self, img: "Buchfink")
    HtrVtRecognizer().process({"cell-formatter": out}, {})

    cells = out[0]["columns"][0]["cells"]
    assert cells[1]["erkannt"] == "", "blank cell got a prediction"
    assert cells[2]["predictions"]["htr_vt"] == "Buchfink", (
        "inked cell must still be recognised"
    )
