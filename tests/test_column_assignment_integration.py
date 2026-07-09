"""Integration: schema-based column assignment on the REAL scans in the repo.

This is the regression net for "Bague appeared as UI column 5": on every
available scan the semantic roles must come out exactly once each and in the
printed form order — matching the ground truth in files-3/meta.json
(ui_column_order: bague, espece, sexe, age, jour_mois, heure, aile, poids).
"""

import json
from pathlib import Path

import pytest

from .conftest import project_root, required_pipeline_models_available

ROOT = project_root()
_TATR_PAGES = sorted((ROOT / "data" / "input" / "tatr").glob("page_*.jpg"))
_FILES3 = ROOT / "files-3"

EXPECTED_ROLE_ORDER = [
    "batch", "species", "sexe", "age", "jour_mois", "heure", "aile", "poids",
]

FLAG_FOR_ROLE = {
    "batch": "is_batch_column",
    "species": "is_species_column",
    "sexe": "is_sexe_column",
    "age": "is_age_column",
    "jour_mois": "is_jour-mois_column",
    "heure": "is_heure_column",
    "aile": "is_alle_column",
    "poids": "is_poids_column",
}


def _detected_role_sequence(page: dict) -> list:
    """Role per column (None for unassigned), in extraction order."""
    sequence = []
    for column in page.get("columns", []):
        role = None
        for r, flag in FLAG_FOR_ROLE.items():
            if column.get(flag, False):
                role = r
                break
        sequence.append(role)
    return sequence


def _run_detection(tatr_page_paths, monkeypatch):
    monkeypatch.chdir(ROOT)
    from pipeline.cv_pipeline import CVPipeline
    from modules.merged_column_extractor import MergedColumnExtractor
    from modules.merged_row_extractor import MergedRowExtractor
    from modules.detect_columns import DetectColumns

    pipeline = CVPipeline(
        input_data={"tatr-extractor": [str(p) for p in tatr_page_paths]}
    )
    pipeline.add_stage(MergedColumnExtractor(debug=False))
    pipeline.add_stage(MergedRowExtractor(debug=False))
    pipeline.add_stage(DetectColumns())
    return pipeline.run()


@pytest.mark.integration
@pytest.mark.skipif(not _TATR_PAGES, reason="no cached TATR pages")
@pytest.mark.skipif(
    not required_pipeline_models_available(), reason="core pipeline models missing"
)
def test_all_roles_assigned_in_form_order_on_real_pages(monkeypatch):
    """Every cached scan page: all 8 roles present, each exactly once, in the
    printed order, and Bague on the FIRST assigned column."""
    pages = _run_detection(_TATR_PAGES[:2], monkeypatch)
    assert pages, "detection pipeline returned nothing"

    for page_idx, page in enumerate(pages):
        sequence = _detected_role_sequence(page)
        assigned = [r for r in sequence if r]

        assert assigned == EXPECTED_ROLE_ORDER, (
            f"Seite {page_idx}: Rollen {assigned} != erwartete Reihenfolge "
            f"{EXPECTED_ROLE_ORDER} (Spaltenfolge: {sequence})"
        )
        # The ring column is the first assigned column of the table.
        first_assigned = next(i for i, r in enumerate(sequence) if r)
        assert sequence[first_assigned] == "batch", (
            f"Seite {page_idx}: erste zugeordnete Spalte ist "
            f"{sequence[first_assigned]!r}, nicht die Ringnummer"
        )


@pytest.mark.integration
@pytest.mark.skipif(not _FILES3.exists(), reason="files-3 corpus not present")
@pytest.mark.skipif(
    not required_pipeline_models_available(), reason="core pipeline models missing"
)
def test_ui_column_order_matches_files3_ground_truth(monkeypatch):
    """End check against the labelled corpus: the UI column labels (what the
    CSV export writes) must follow files-3/meta.json ui_column_order."""
    monkeypatch.chdir(ROOT)
    meta = json.loads((_FILES3 / "meta.json").read_text())
    expected_ui = meta["ui_column_order"]  # bague_no, espece, ..., notes

    from pipeline.cv_pipeline import CVPipeline
    from modules.pdf_converter import PdfConverter
    from modules.table_rotator import TableRotator
    from modules.tatr_extraction import TatrExtractor
    from modules.merged_column_extractor import MergedColumnExtractor
    from modules.merged_row_extractor import MergedRowExtractor
    from modules.detect_columns import DetectColumns
    from libs.table_view import visible_columns

    pipeline = CVPipeline()
    pipeline.add_stage(PdfConverter(debug=False))
    pipeline.add_stage(TableRotator(debug=False))
    pipeline.add_stage(TatrExtractor(debug=False))
    pipeline.add_stage(MergedColumnExtractor(debug=False))
    pipeline.add_stage(MergedRowExtractor(debug=False))
    pipeline.add_stage(DetectColumns())
    pages = pipeline.run(input_data=str(_FILES3 / "scan_1972_sample.pdf"))

    label_for_ui = {
        "bague_no": "Bague",
        "espece": "Espèce",
        "sexe": "Sexe",
        "age": "Age",
        "jour_mois": "Jour/Mois",
        "heure": "Heure",
        "aile": "Aile",
        "poids": "Poids",
    }
    expected_labels = [label_for_ui[c] for c in expected_ui if c in label_for_ui]

    for page_idx, page in enumerate(pages):
        # Wrap raw cells like CellFormatter would — visible_columns only
        # needs the flags.
        labels = [label for _, label in visible_columns(page)]
        semantic = [l for l in labels if not l.startswith("Spalte")]
        assert semantic == expected_labels, (
            f"Seite {page_idx}: UI-Spalten {semantic} != Ground-Truth-Reihenfolge "
            f"{expected_labels}"
        )
