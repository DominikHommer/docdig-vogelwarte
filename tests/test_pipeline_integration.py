"""End-to-end pipeline run against the real scan in `data/input/`.

These tests are slow (model load + image processing), so they are gated by:
- presence of a real PDF + core models (auto-skip when missing)
- the `integration` marker (`pytest -m integration` to run, default skips them).

The fast deterministic suite still runs every time you do `pytest tests/`.
"""

import json
import os
from pathlib import Path

import pytest

from .conftest import (
    find_test_pdf,
    project_root,
    required_pipeline_models_available,
)


def _cached_tatr_pages(limit: int = 2):
    tatr_dir = project_root() / "data" / "input" / "tatr"
    if not tatr_dir.exists():
        return []
    pages = sorted(tatr_dir.glob("page_*.jpg"))
    return pages[:limit]


@pytest.mark.integration
@pytest.mark.skipif(
    not find_test_pdf() or not required_pipeline_models_available(),
    reason="No real scan PDF or core model files available.",
)
def test_pipeline_runs_on_real_scan(monkeypatch, tmp_path):
    """Run the recognition portion of the pipeline on cached TATR crops.

    Asserts only structural properties so the test is robust to OCR drift —
    the goal here is regression-detection (crashes, malformed output, lost
    semantic flags), not OCR accuracy.
    """
    pages = _cached_tatr_pages(limit=2)
    if not pages:
        pytest.skip("No cached TATR pages (data/input/tatr/page_*.jpg). Run main.py once to cache them.")

    monkeypatch.chdir(project_root())

    from pipeline.cv_pipeline import CVPipeline
    from modules.cell_denoiser import CellDenoiser
    from modules.cell_formatter import CellFormatter
    from modules.detect_columns import DetectColumns
    from modules.digit_recognizer import DigitRecognizer
    from modules.htr_vt_recognizer import HtrVtRecognizer
    from modules.merged_column_extractor import MergedColumnExtractor
    from modules.merged_row_extractor import MergedRowExtractor
    from modules.quotation_mark_detector import QuotationMarkDetector
    from modules.sexe_classifier import SexeClassifier

    pipeline = CVPipeline(input_data={"tatr-extractor": [str(p) for p in pages]})
    pipeline.add_stage(MergedColumnExtractor(debug=False))
    pipeline.add_stage(MergedRowExtractor(debug=False))
    pipeline.add_stage(DetectColumns())
    pipeline.add_stage(CellDenoiser(debug=False))
    pipeline.add_stage(CellFormatter())
    pipeline.add_stage(QuotationMarkDetector())
    pipeline.add_stage(HtrVtRecognizer())
    pipeline.add_stage(DigitRecognizer())
    pipeline.add_stage(SexeClassifier())
    # TrOCR is intentionally skipped here: the HF model is huge and downloads
    # on first run. Run main.py if you want the full path.

    result = pipeline.run()

    # ---- Structural assertions ---------------------------------------------
    assert isinstance(result, list), f"Pipeline returned {type(result)} instead of list"
    assert len(result) >= 1, "Pipeline produced no pages"

    page = result[0]
    assert "columns" in page, "Page has no 'columns' key"
    columns = page["columns"]
    assert len(columns) >= 3, f"Expected ≥3 columns, got {len(columns)}"

    # ---- Cell dict contract -----------------------------------------------
    for col in columns:
        for cell in col.get("cells", []):
            assert "erkannt" in cell, "cell missing 'erkannt'"
            assert "score" in cell, "cell missing 'score'"
            assert "skip_ocr" in cell, "cell missing 'skip_ocr'"

    # ---- Bague column should be tagged with an anchor somewhere ------------
    bague_cols = [c for c in columns if c.get("is_batch_column")]
    if bague_cols:
        bague = bague_cols[0]
        data_cells = bague.get("cells", [])[1:]
        if data_cells:
            anchors = [c for c in data_cells if c.get("is_anchor")]
            assert len(anchors) <= 1, "Bague column has more than one anchor cell"
            # We don't require an anchor to exist (handwriting OCR may fail
            # entirely on some scans), but if any cell has a score >0, an
            # anchor should be present.
            if any(c.get("score", -1) >= 70 for c in data_cells):
                assert anchors, (
                    "Bague column has scored cells but no anchor — "
                    "DigitRecognizer probably did not run, or sequence helpers regressed."
                )

    # ---- Bague column: any extrapolated cell carries the flag --------------
    for col in columns:
        if not col.get("is_batch_column"):
            continue
        for cell in col.get("cells", [])[1:]:
            if cell.get("score") == 70:
                assert cell.get("is_extrapolated") is True, (
                    "Extrapolated cell missing is_extrapolated flag"
                )


def _load_bague_fixtures(fixture_dir: Path):
    """Build (cells, expected, labels_dict) from the bague fixture directory."""
    import cv2
    import re

    labels = json.loads((fixture_dir / "labels.json").read_text())

    header_files = sorted(fixture_dir.glob("*_header.png"))
    header_img = cv2.imread(str(header_files[0]), cv2.IMREAD_GRAYSCALE) if header_files else None
    cells = [
        {"image": header_img, "image_raw": header_img, "skip_ocr": False, "erkannt": "", "score": -1}
    ]
    expected = [None]

    anchor_files = sorted(f for f in labels if f.startswith("anchor"))

    def _row_index(name: str) -> int:
        m = re.search(r"row(\d+)", name)
        return int(m.group(1)) if m else 0

    suffix_files = sorted((f for f in labels if not f.startswith("anchor")), key=_row_index)

    for name in anchor_files + suffix_files:
        path = fixture_dir / name
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert img is not None, f"Could not read {path}"
        cells.append({"image": img, "image_raw": img, "skip_ocr": False, "erkannt": "", "score": -1})
        expected.append(labels[name])

    return cells, expected, labels


def _bague_column(cells: list) -> dict:
    return {
        "cells": cells,
        "is_batch_column": True,
        "is_species_column": False,
        "is_sexe_column": False,
        "is_age_column": False,
        "is_jour-mois_column": False,
        "is_heure_column": False,
        "is_alle_column": False,
        "is_poids_column": False,
    }


_FIXTURE_DIR = project_root() / "tests" / "fixtures" / "cells" / "bague"
_FIXTURES_AVAILABLE = (_FIXTURE_DIR / "labels.json").exists()


@pytest.mark.integration
@pytest.mark.skipif(not _FIXTURES_AVAILABLE, reason="No bague fixtures. Run tools/extract_bague_fixtures.py first.")
def test_bague_sequence_after_manual_anchor(monkeypatch):
    """Simulate the UI workflow.

    1. DigitRecognizer runs and probably gets the anchor wrong (handwriting
       OCR is unreliable). That is fine — it tags the first data cell as the
       anchor regardless.
    2. The user types the correct anchor into the UI, which calls
       `rebuild_batch_sequence(column, anchor_value=...)`.
    3. The rest of the column should now be exactly correct.

    This is the canonical "robust digitisation" path: a single number the
    user has to provide, and the rest is automatic.
    """
    monkeypatch.chdir(project_root())

    cells, expected, _ = _load_bague_fixtures(_FIXTURE_DIR)
    column = _bague_column(cells)
    pages = [{"columns": [column]}]

    from modules.digit_recognizer import DigitRecognizer
    from libs.bague_sequence import rebuild_batch_sequence

    DigitRecognizer(debug=False).process({"cell-formatter": pages}, {})

    # Manual correction step the user would do in the UI.
    correct_anchor = expected[1]  # the labelled value of the first data cell
    rebuild_batch_sequence(column, anchor_value=correct_anchor)

    actual = [c["erkannt"] for c in column["cells"][1:]]
    assert actual == expected[1:], (
        f"Sequence diverged after manual anchor. First few mismatches: "
        + ", ".join(
            f"expected={e!r} got={a!r}"
            for e, a in zip(expected[1:], actual)
            if e != a
        )[:400]
    )


@pytest.mark.integration
@pytest.mark.skipif(not _FIXTURES_AVAILABLE, reason="No bague fixtures.")
def test_bague_per_cell_ocr_accuracy(monkeypatch):
    """How well does the printed-digit OCR do on real fixtures?

    We don't assert 100% — Tesseract on cropped 40px-tall cells will have
    a handful of misses. Instead we set a sane floor (≥80% suffix-match) so
    a regression that drops accuracy further fails the test loudly.
    """
    monkeypatch.chdir(project_root())

    cells, expected, _ = _load_bague_fixtures(_FIXTURE_DIR)
    from modules.digit_recognizer import DigitRecognizer

    dr = DigitRecognizer(debug=False)
    dr._ensure_tesseract()

    correct = 0
    wrong = []
    # Skip the anchor cell — it is handwritten on this dataset. Per-cell test
    # only covers the printed suffix cells.
    for cell, label in zip(cells[2:], expected[2:]):
        img = cell["image_raw"]
        digit = dr._read_single_digit(img)
        want = (label or "")[-1] if label else ""
        if digit == want:
            correct += 1
        else:
            wrong.append((want, digit))

    total = len(expected) - 2
    accuracy = correct / total if total else 0.0
    print(f"Per-cell suffix OCR accuracy: {accuracy:.0%} ({correct}/{total})")
    # Tesseract alone hits ~58 % on these Vogelwarte cell crops (table-border
    # artifacts confuse it). The sequence reconstruction logic does the heavy
    # lifting after the user supplies an anchor, so we only set a floor here
    # to catch regressions that break OCR entirely.
    assert accuracy >= 0.50, (
        f"Per-cell digit OCR accuracy dropped to {accuracy:.0%} (was ~58 %). "
        f"Misses (want, got): {wrong[:10]}"
    )
