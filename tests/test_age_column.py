"""Age column: TrOCR reads it, FuzzyMatchingAge snaps onto the closed
vocabulary {Fd, Fnd}.

The decision rule (n/u present -> Fnd, else Fd) was derived from 49
hand-labelled real cells under tests/fixtures/cells/age/ — TrOCR reads the
handwriting at ~90%, HTR-VT (trained on bird names) at 14/49, so the age
column belongs to TrOCR.
"""

import json

import numpy as np
import pytest

from modules.fuzzy_matching import FuzzyMatchingAge
from modules.htr_vt_recognizer import HtrVtRecognizer
from modules.trocr import TrOCR, _TRO_CR_TARGET_FLAGS


def _inked(shape=(48, 160)):
    img = np.full(shape, 255, dtype=np.uint8)
    img[16:34, 60:100] = 0
    return img


def _age_cell(**extra):
    cell = {
        "image": _inked(),
        "image_raw": _inked(),
        "erkannt": "",
        "score": -1,
        "skip_ocr": False,
        "is_blank": False,
        "verbesserung": "",
    }
    cell.update(extra)
    return cell


def _age_column(*cells):
    return {
        "cells": [_age_cell()] + list(cells),
        "is_age_column": True,
        "is_species_column": False,
        "is_batch_column": False,
        "is_sexe_column": False,
        "is_jour-mois_column": False,
        "is_heure_column": False,
        "is_alle_column": False,
        "is_poids_column": False,
    }


def test_age_column_belongs_to_trocr_not_htr():
    """Measured: TrOCR ~90% on labelled age cells, HTR-VT 14/49 (bird-name
    training). The dispatch must reflect that."""
    assert "is_age_column" in _TRO_CR_TARGET_FLAGS
    assert "is_age_column" not in HtrVtRecognizer.DEFAULT_TARGET_FLAGS
    assert "is_species_column" in HtrVtRecognizer.DEFAULT_TARGET_FLAGS


def test_trocr_writes_age_predictions(monkeypatch):
    trocr = TrOCR()
    monkeypatch.setattr(TrOCR, "_ensure_loaded", lambda self: True)
    monkeypatch.setattr(TrOCR, "_recognise", lambda self, img: "Tund")

    column = _age_column(_age_cell(), _age_cell(is_blank=True, skip_ocr=True))
    pages = [{"columns": [column]}]
    trocr.process({"cell-formatter": pages}, {})

    assert column["cells"][1]["predictions"]["trocr"] == "Tund"
    # Blank age cell stays empty (many age rows are blank by design).
    assert "predictions" not in column["cells"][2]
    assert column["cells"][2]["erkannt"] == ""


@pytest.fixture
def age_fuzzy(tmp_path):
    labels = ["Fd", "Fnd", "ad", "juv"]
    p = tmp_path / "age.json"
    p.write_text(json.dumps({label: i for i, label in enumerate(labels)}))
    return FuzzyMatchingAge(class_label_path=str(p), score_threshold=40)


def test_age_fuzzy_snaps_htr_output(age_fuzzy):
    column = _age_column(
        _age_cell(predictions={"htr_vt": "Fnd."}, erkannt="Fnd."),
        _age_cell(predictions={"htr_vt": "fd"}, erkannt="fd"),
    )
    pages = [{"columns": [column]}]
    age_fuzzy.process({"trocr": pages}, {})

    assert column["cells"][1]["erkannt"] == "Fnd"
    assert column["cells"][2]["erkannt"] == "Fd"


def test_age_fuzzy_prefers_htr_prediction_over_provisional(age_fuzzy):
    cell = _age_cell(predictions={"htr_vt": "Fnd"}, erkannt="garbage")
    pages = [{"columns": [_age_column(cell)]}]
    age_fuzzy.process({"trocr": pages}, {})
    assert cell["erkannt"] == "Fnd"


def test_age_fuzzy_leaves_blank_and_ditto_alone(age_fuzzy):
    blank = _age_cell(is_blank=True, skip_ocr=True)
    ditto = _age_cell(erkannt='"', skip_ocr=True)
    pages = [{"columns": [_age_column(blank, ditto)]}]
    age_fuzzy.process({"trocr": pages}, {})

    assert blank["erkannt"] == ""
    assert ditto["erkannt"] == '"'


# ---------------------------------------------------------------------------
# Closed vocabulary: the age column may ONLY ever show Fd / Fnd (or ditto/empty)
# ---------------------------------------------------------------------------


def _run_default_age(*cells):
    pages = [{"columns": [_age_column(*cells)]}]
    FuzzyMatchingAge().process({"trocr": pages}, {})
    return cells


def test_default_vocabulary_is_fd_fnd_only():
    assert tuple(FuzzyMatchingAge().class_labels) == ("Fd", "Fnd")


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Clean reads
        ("Fnd", "Fnd"),
        ("Fd", "Fd"),
        ("Fnd.", "Fnd"),
        ("fd", "Fd"),
        ("FnD", "Fnd"),
        ("Fn", "Fnd"),       # truncated but unambiguous
        # Real TrOCR reads from the labelled fixture cells:
        ("Tund", "Fnd"),
        ("Fud", "Fnd"),      # TrOCR reads the written n as u
        ("Tud", "Fnd"),      # fuzzy alone would mis-snap this to Fd
        ("I'd", "Fd"),
        ("Ird", "Fd"),
        (": Find", "Fnd"),
        ("Vt .", "Fd"),
        # Noise with letter evidence: closed world decides, low score flags it
        ("fnsesm", "Fnd"),
        ("Foto", "Fd"),
        ("End", "Fnd"),
        # No usable evidence -> empty
        ("Buchfink", ""),    # too long to be a misread 2-3 char code
        ("175", ""),         # digits only
    ],
)
def test_age_output_is_whitelisted(raw, expected):
    (cell,) = _run_default_age(_age_cell(predictions={"trocr": raw}, erkannt=raw))
    assert cell["erkannt"] == expected


def test_age_weak_guess_gets_review_score():
    """Closed-world guesses must NOT look confident — the low score puts the
    cell into the UI review flow."""
    (cell,) = _run_default_age(_age_cell(predictions={"trocr": "Tud"}, erkannt="Tud"))
    assert cell["erkannt"] == "Fnd"
    assert cell["score"] == FuzzyMatchingAge.WEAK_GUESS_SCORE
    (clean,) = _run_default_age(_age_cell(predictions={"trocr": "Fnd"}, erkannt="Fnd"))
    assert clean["score"] > FuzzyMatchingAge.WEAK_GUESS_SCORE


def test_age_prefers_trocr_over_htr():
    """TrOCR is the measured-better source — it must win over HTR noise."""
    cell = _age_cell(predictions={"htr_vt": "nnn", "trocr": "Fd"}, erkannt="")
    _run_default_age(cell)
    assert cell["erkannt"] == "Fd"


def test_age_invariant_clears_stray_values_without_predictions():
    """Even when garbage reached `erkannt` through some other path (no
    predictions dict), the closed vocabulary is enforced."""
    (cell,) = _run_default_age(_age_cell(erkannt="zzz999zzz"))
    assert cell["erkannt"] in ("", "Fd", "Fnd")


def test_age_manual_edit_is_never_overwritten():
    cell = _age_cell(erkannt="custom", is_manual_edit=True, skip_ocr=True)
    _run_default_age(cell)
    assert cell["erkannt"] == "custom"


def test_age_every_final_value_in_closed_set():
    """Property-style check over a pile of noisy inputs: the final column
    content is always a subset of {Fd, Fnd, '', '\"'}."""
    noisy = ["Fnd", "fnsesm", "End", "ffnndd", "Fnol", "ad.", "Fd?", "-", "  ", "Fd"]
    cells = [_age_cell(predictions={"htr_vt": t}, erkannt=t) for t in noisy]
    _run_default_age(*cells)
    for cell in cells:
        assert cell["erkannt"] in ("Fd", "Fnd", "", '"'), cell["erkannt"]


# ---------------------------------------------------------------------------
# Real labelled cells (auto-skips when fixtures/models are missing)
# ---------------------------------------------------------------------------

from .conftest import project_root

_AGE_DIR = project_root() / "tests" / "fixtures" / "cells" / "age"
_AGE_LABELS = _AGE_DIR / "labels.json"


@pytest.mark.integration
@pytest.mark.skipif(not _AGE_LABELS.exists(), reason="no labelled age cells")
def test_age_accuracy_on_real_cells(monkeypatch):
    """End-to-end TrOCR + closed-world snap on the 49 hand-labelled cells.

    Measured baseline at introduction: 48/49. The floor of 85% catches
    regressions without breaking on a re-extraction jitter.
    """
    monkeypatch.chdir(project_root())
    import cv2

    from libs.cell_cropping import strip_cell_borders

    labels = {
        k: v for k, v in json.loads(_AGE_LABELS.read_text()).items() if v
    }
    if not labels:
        pytest.skip("labels.json has no labelled cells")

    trocr = TrOCR()
    if not trocr._ensure_loaded():
        pytest.skip("TrOCR model unavailable")
    snapper = FuzzyMatchingAge()

    hits = total = 0
    misses = []
    for name, truth in sorted(labels.items()):
        img = cv2.imread(str(_AGE_DIR / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cleaned = strip_cell_borders(img)
        text = trocr._recognise(cleaned) or ""
        label, _ = snapper._snap(text)
        total += 1
        if label == truth:
            hits += 1
        else:
            misses.append((name, truth, text, label))

    assert total >= 40, f"fixture set unexpectedly small ({total})"
    accuracy = hits / total
    assert accuracy >= 0.85, (
        f"Age accuracy dropped to {accuracy:.0%} ({hits}/{total}). "
        f"First misses: {misses[:5]}"
    )
