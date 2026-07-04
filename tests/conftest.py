"""Shared test fixtures and path helpers.

Tests should be runnable from a fresh clone. Anything that needs heavy
artifacts (real scans, model checkpoints, hand-extracted cell fixtures) goes
through one of the resolvers below and is automatically skipped when the
file is missing — so the deterministic suite is always green and the
data-driven tests turn on as soon as you drop the files in.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def project_root() -> Path:
    return ROOT


def find_test_pdf() -> Optional[Path]:
    """Return the first real scan PDF available, or None."""
    for pattern in ("data/input/scan_*.pdf", "tests/fixtures/scans/*.pdf"):
        for path in sorted(ROOT.glob(pattern)):
            return path
    return None


def required_pipeline_models_available() -> bool:
    """True only when the *core* model files needed for an end-to-end run exist."""
    needed = [
        "config/denoise_model.keras",
        "config/pubtables1m_detection_detr_r18.pth",
        "config/pubtables1m_structure_detr_r18.pth",
        "config/detection_config.json",
        "config/structure_config.json",
    ]
    return all((ROOT / p).exists() for p in needed)


def bague_fixtures_dir() -> Path:
    return ROOT / "tests" / "fixtures" / "cells" / "bague"


def bague_fixtures_available() -> bool:
    d = bague_fixtures_dir()
    return d.exists() and (d / "labels.json").exists()


def files3_corpus_dir() -> Optional[Path]:
    """Return the path to the multi-page labelled corpus (files-3) if linked."""
    candidate = ROOT / "files-3"
    if candidate.exists() and (candidate / "meta.json").exists():
        return candidate
    return None


def make_batch_column(num_cells: int, header_marker: int = 0) -> dict:
    """Build a synthetic batch column with `num_cells` data rows + 1 header."""
    import numpy as np

    blank = np.zeros((40, 100), dtype=np.uint8)
    cells = [{"image": blank, "image_raw": blank, "skip_ocr": False, "erkannt": "", "score": -1}]
    for _ in range(num_cells):
        cells.append(
            {"image": blank, "image_raw": blank, "skip_ocr": False, "erkannt": "", "score": -1}
        )
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


def make_pages(columns: list) -> list:
    return [{"columns": columns}]
