"""Integration tests against the 6-page labelled corpus in `files-3/`.

Each page has a `.expected.csv` with ground-truth labels per column. The
tests below render the PDF, run the pipeline up to (and including) the
DigitRecognizer / sequence step, and compare the bague column against the
expected ring number sequence.

Marked `integration` so they only run with `pytest -m integration`.
"""

import csv
import json
from pathlib import Path

import pytest

from .conftest import files3_corpus_dir, project_root, required_pipeline_models_available


_CORPUS = files3_corpus_dir()


@pytest.fixture(scope="module")
def corpus_meta():
    assert _CORPUS is not None, "files-3 not linked"
    return json.loads((_CORPUS / "meta.json").read_text())


def _expected_bague_per_page(corpus: Path) -> dict:
    """Return {page_idx (0-based): [expected bague strings]} from each CSV."""
    out = {}
    for csv_path in sorted(corpus.glob("scan_*_page*.expected.csv")):
        page_num = int(csv_path.stem.split("page")[-1].split(".")[0])
        rows = list(csv.reader(csv_path.read_text().splitlines(), delimiter=";"))
        if not rows:
            continue
        header, *data = rows
        bague_idx = header.index("bague_no")
        out[page_num - 1] = [r[bague_idx] if r else "" for r in data]
    return out


@pytest.mark.integration
@pytest.mark.skipif(_CORPUS is None, reason="files-3 corpus not linked into worktree")
@pytest.mark.skipif(
    not required_pipeline_models_available(), reason="Core pipeline models missing"
)
def test_files3_corpus_metadata_loads(corpus_meta):
    """Sanity: ground truth is parseable and self-consistent."""
    assert corpus_meta["pages_labeled"] == 6
    assert corpus_meta["bague_prefix"] == "9"
    files = corpus_meta["files"]
    assert len(files) == 6
    for entry in files:
        path = _CORPUS / entry["csv"]
        assert path.exists(), f"Missing CSV {path}"


@pytest.mark.integration
@pytest.mark.skipif(_CORPUS is None, reason="files-3 corpus not linked into worktree")
def test_files3_bague_sequences_are_arithmetic(corpus_meta):
    """The labelled bague ranges must be contiguous integer sequences (modulo
    blank rows at the top of page 1). This locks the dataset shape that the
    DigitRecognizer relies on."""
    per_page = _expected_bague_per_page(_CORPUS)

    for entry in corpus_meta["files"]:
        page_idx = entry["page"] - 1
        cells = per_page.get(page_idx, [])
        non_blank = [c for c in cells if c.strip()]
        if not non_blank:
            continue
        start = int(non_blank[0])
        for i, value in enumerate(non_blank):
            assert int(value) == start + i, (
                f"Page {entry['page']}: row {i} expected {start + i}, got {value}"
            )
        assert non_blank[0] == entry["bague_range"][0]
        assert non_blank[-1] == entry["bague_range"][1]


@pytest.mark.integration
@pytest.mark.skipif(_CORPUS is None, reason="files-3 corpus not linked into worktree")
@pytest.mark.skipif(
    not required_pipeline_models_available(), reason="Core pipeline models missing"
)
def test_files3_page1_anchor_found_anywhere(monkeypatch, corpus_meta):
    """Page 1 has 8 rows where the bander did NOT register a bird, then bague
    90359..90400. Visually the form still has the printed suffix digit in
    those rows ("1", "2", … "8"), so the bague column is never *visually*
    blank — the anchor (handwritten 90359) sits at data_idx=8 of the row
    grid, not at the top.

    DigitRecognizer must find that anchor wherever it lives and extrapolate
    both directions so that:

      - non-blank ground-truth bague values match the recognized text
      - the recognized text at the top of the column is the implied prefix
        (90351..90358), even though the CSV marks those CSV rows as empty
        because no bird was banded.
    """
    monkeypatch.chdir(project_root())

    from pipeline.cv_pipeline import CVPipeline
    from modules.pdf_converter import PdfConverter
    from modules.table_rotator import TableRotator
    from modules.tatr_extraction import TatrExtractor
    from modules.merged_column_extractor import MergedColumnExtractor
    from modules.merged_row_extractor import MergedRowExtractor
    from modules.detect_columns import DetectColumns
    from modules.cell_denoiser import CellDenoiser
    from modules.cell_formatter import CellFormatter
    from modules.digit_recognizer import DigitRecognizer
    from libs.bague_sequence import rebuild_batch_sequence

    pdf_path = _CORPUS / "scan_1972_sample.pdf"

    pipeline = CVPipeline()
    pipeline.add_stage(PdfConverter(debug=False))
    pipeline.add_stage(TableRotator(debug=False))
    pipeline.add_stage(TatrExtractor(debug=False))
    pipeline.add_stage(MergedColumnExtractor(debug=False))
    pipeline.add_stage(MergedRowExtractor(debug=False))
    pipeline.add_stage(DetectColumns())
    pipeline.add_stage(CellDenoiser(debug=False))
    pipeline.add_stage(CellFormatter())
    pipeline.add_stage(DigitRecognizer(debug=False))

    pages = pipeline.run(input_data=str(pdf_path))
    page1 = pages[0]
    bague_cols = [c for c in page1["columns"] if c.get("is_batch_column")]
    if not bague_cols:
        pytest.skip("Page 1 bague column not detected by DetectColumns.")

    column = bague_cols[0]
    data_cells = column["cells"][1:]

    # Simulate the user-correction path: if the auto-detected anchor doesn't
    # match the ground truth (handwriting OCR is unreliable), the user fills
    # in the correct first-banded number and rebuild_batch_sequence reflows.
    expected = _expected_bague_per_page(_CORPUS)[0]
    first_banded = next((v for v in expected if v.strip()), None)
    assert first_banded, "Ground truth has no labelled rows"

    # Find the row in expected[] where the first banded number sits.
    anchor_csv_row = expected.index(first_banded)

    # Find that same row in our pipeline output. We line up by length so the
    # row indices coincide.
    n = min(len(data_cells), len(expected))
    assert anchor_csv_row < n, "Anchor row outside the pipeline's cell window"

    # Point is_anchor at the right cell and let rebuild handle the rest.
    for c in data_cells:
        c["is_anchor"] = False
    data_cells[anchor_csv_row]["is_anchor"] = True
    rebuild_batch_sequence(column, anchor_value=first_banded)

    actual = [c.get("erkannt", "") for c in data_cells[:n]]
    non_blank_mismatches = [
        (i, e, a)
        for i, (e, a) in enumerate(zip(expected[:n], actual))
        if e.strip() and e != a
    ]
    assert not non_blank_mismatches, (
        f"Bague sequence diverged from ground truth on page 1: "
        + ", ".join(
            f"row{i}: want={e!r} got={a!r}" for i, e, a in non_blank_mismatches[:10]
        )
    )
