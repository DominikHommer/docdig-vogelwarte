"""End-to-end accuracy against the labelled files-3 corpus.

Runs the FULL recognition pipeline on files-3/scan_1972_sample.pdf and
scores every transcribed column against the hand-labelled expected CSVs.
This is THE yardstick for recognition changes — run before/after.

Usage:
    .venv/bin/python tools/evaluate_corpus.py              # alle 6 Seiten
    .venv/bin/python tools/evaluate_corpus.py --pages 5    # nur Seite 5

Not scored: age / jour_mois / heure — the ground truth deliberately left
them untranscribed (meta.json: columns_left_blank_by_design).
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DITTO_VARIANTS = {'"', "„", "''", "//", "ii", "“", "”"}

# GT column -> (is_*-flag, label) — only the transcribed ones are scored.
SCORED = {
    "bague_no": "is_batch_column",
    "espece": "is_species_column",
    "sexe": "is_sexe_column",
    "aile": "is_alle_column",
    "poids": "is_poids_column",
}


def normalise(value: str) -> str:
    value = unicodedata.normalize("NFC", (value or "").strip())
    if value in DITTO_VARIANTS:
        return '"'
    return value


def load_expected(page: int):
    path = ROOT / "files-3" / f"scan_1972_sample_page{page}.expected.csv"
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    args = parser.parse_args()

    import os

    os.chdir(ROOT)

    from pipeline.cv_pipeline import CVPipeline
    from modules.pdf_converter import PdfConverter
    from modules.table_rotator import TableRotator
    from modules.tatr_extraction import TatrExtractor
    from modules.merged_column_extractor import MergedColumnExtractor
    from modules.merged_row_extractor import MergedRowExtractor
    from modules.detect_columns import DetectColumns
    from modules.cell_formatter import CellFormatter
    from modules.quotation_mark_detector import QuotationMarkDetector
    from modules.htr_vt_recognizer import HtrVtRecognizer
    from modules.digit_recognizer import DigitRecognizer
    from modules.sexe_classifier import SexeClassifier
    from modules.trocr import TrOCR
    from modules.fuzzy_matching import FuzzyMatchingBirdNames, FuzzyMatchingAge
    from modules.numeric_consensus import NumericConsensus
    from libs.bague_sequence import rebuild_batch_sequence

    pipeline = CVPipeline()
    pipeline.add_stage(PdfConverter(debug=False))
    pipeline.add_stage(TableRotator(debug=False))
    pipeline.add_stage(TatrExtractor(debug=False))
    pipeline.add_stage(MergedColumnExtractor(debug=False))
    pipeline.add_stage(MergedRowExtractor(debug=False))
    pipeline.add_stage(DetectColumns())
    pipeline.add_stage(CellFormatter())
    pipeline.add_stage(QuotationMarkDetector())
    pipeline.add_stage(HtrVtRecognizer())
    pipeline.add_stage(DigitRecognizer())
    pipeline.add_stage(SexeClassifier())
    pipeline.add_stage(TrOCR())
    pipeline.add_stage(FuzzyMatchingBirdNames())
    pipeline.add_stage(NumericConsensus())
    pipeline.add_stage(FuzzyMatchingAge())

    pages = pipeline.run(input_data=str(ROOT / "files-3" / "scan_1972_sample.pdf"))

    grand = {}
    for page_no in args.pages:
        if page_no > len(pages):
            continue
        page = pages[page_no - 1]
        expected = load_expected(page_no)

        # Simulate the one manual step of the intended workflow: the user
        # corrects the bague anchor once per page (UI does the same).
        truth_bague = [r.get("bague_no", "") for r in expected]
        first_truth = next((b for b in truth_bague if b.strip()), None)
        for column in page.get("columns", []):
            if column.get("is_batch_column") and first_truth:
                anchor_row = truth_bague.index(first_truth) + 1
                cells = column.get("cells", [])
                if anchor_row < len(cells):
                    for c in cells[1:]:
                        c["is_anchor"] = False
                    cells[anchor_row]["is_anchor"] = True
                    rebuild_batch_sequence(column, anchor_value=first_truth)

        print(f"\n===== Seite {page_no} =====")
        for gt_key, flag in SCORED.items():
            column = next(
                (c for c in page.get("columns", []) if c.get(flag)), None
            )
            hits = total = 0
            misses = []
            for row_i, exp_row in enumerate(expected, start=1):
                truth = normalise(exp_row.get(gt_key, ""))
                cells = column.get("cells", []) if column else []
                got = (
                    normalise(cells[row_i].get("erkannt", ""))
                    if row_i < len(cells)
                    else ""
                )
                total += 1
                if got == truth:
                    hits += 1
                elif len(misses) < 5:
                    misses.append((row_i, truth, got))
            pct = 100.0 * hits / max(total, 1)
            grand.setdefault(gt_key, [0, 0])
            grand[gt_key][0] += hits
            grand[gt_key][1] += total
            print(f"  {gt_key:10} {hits:3}/{total:<3} ({pct:5.1f}%)  "
                  + "; ".join(f"#{i} soll={t!r} ist={g!r}" for i, t, g in misses))

    print("\n===== Gesamt =====")
    for gt_key, (hits, total) in grand.items():
        print(f"  {gt_key:10} {hits:3}/{total:<3} ({100.0 * hits / max(total, 1):5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
