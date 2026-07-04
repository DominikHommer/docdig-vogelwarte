"""Extract bague (ring number) cells from real scans into test fixtures.

Usage:

    python3 tools/extract_bague_fixtures.py            # uses cached TATR pages
    python3 tools/extract_bague_fixtures.py --pages 3  # how many pages

What it does:

1. Runs MergedColumnExtractor + MergedRowExtractor on the cached TATR pages
   under ``data/input/tatr/`` (or falls back to running TATR on the PDF).
2. Identifies the bague column via ``DetectColumns`` (or as a heuristic, the
   left-most column if Tesseract isn't available).
3. Writes every cell to ``tests/fixtures/cells/bague/`` as a PNG and creates
   a ``labels.json`` template with empty strings — fill it in by hand and the
   ``test_digit_recognizer_on_real_cells`` test will pick the data up.

This is a one-off helper, not part of the runtime pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _ensure_cwd_at_root() -> None:
    os.chdir(ROOT)


def _resolve_tatr_pages(limit: int) -> list[Path]:
    cached = sorted((ROOT / "data" / "input" / "tatr").glob("page_*.jpg"))
    return cached[:limit]


def _identify_batch_column(columns: list) -> int | None:
    """Return the column index marked as `is_batch_column`, or fall back to 0."""
    for idx, col in enumerate(columns):
        if col.get("is_batch_column"):
            return idx
    return 0 if columns else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pages", type=int, default=2, help="How many cached TATR pages to use"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "cells" / "bague",
        help="Where to write the cells + labels.json",
    )
    args = parser.parse_args()

    _ensure_cwd_at_root()

    tatr_pages = _resolve_tatr_pages(args.pages)
    if not tatr_pages:
        print(
            "[extract_bague_fixtures] No cached TATR pages. Run `python3 src/main.py` "
            "once to populate data/input/tatr/."
        )
        return 1

    print(f"[extract_bague_fixtures] Using {len(tatr_pages)} cached page(s).")

    from pipeline.cv_pipeline import CVPipeline
    from modules.merged_column_extractor import MergedColumnExtractor
    from modules.merged_row_extractor import MergedRowExtractor
    try:
        from modules.detect_columns import DetectColumns
    except ImportError:
        DetectColumns = None  # tesseract not installed -> heuristic fallback

    pipeline = CVPipeline(input_data={"tatr-extractor": [str(p) for p in tatr_pages]})
    pipeline.add_stage(MergedColumnExtractor(debug=False))
    pipeline.add_stage(MergedRowExtractor(debug=False))
    if DetectColumns is not None:
        pipeline.add_stage(DetectColumns())

    result = pipeline.run()
    if not result:
        print("[extract_bague_fixtures] Pipeline returned nothing — aborting.")
        return 1

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    labels: dict[str, str] = {}

    cells_written = 0
    for page_idx, page in enumerate(result):
        columns = page.get("columns", [])
        if not columns:
            continue

        bague_idx = _identify_batch_column(columns)
        if bague_idx is None:
            continue

        col = columns[bague_idx]
        cells = col.get("cells", [])
        for cell_idx, cell in enumerate(cells):
            # Cells may be raw ndarrays (no DetectColumns) or dicts.
            if isinstance(cell, dict):
                img = cell.get("image_raw")
                if img is None:
                    img = cell.get("image")
            else:
                img = cell

            if img is None:
                continue

            if cell_idx == 0:
                name = f"page_{page_idx}_header.png"
                cv2.imwrite(str(out_dir / name), img)
                continue  # header doesn't get a label
            if cell_idx == 1:
                name = f"anchor_p{page_idx}_row{cell_idx}.png"
            else:
                name = f"suffix_p{page_idx}_row{cell_idx}.png"

            cv2.imwrite(str(out_dir / name), img)
            labels.setdefault(name, "")
            cells_written += 1

    labels_path = out_dir / "labels.json"
    # Preserve any labels the user may have already filled in.
    if labels_path.exists():
        try:
            existing = json.loads(labels_path.read_text())
            for k, v in existing.items():
                if v:
                    labels[k] = v
        except Exception:
            pass

    labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False))

    print(f"[extract_bague_fixtures] Wrote {cells_written} cells to {out_dir}")
    print(f"[extract_bague_fixtures] Label template: {labels_path}")
    print(
        "[extract_bague_fixtures] Fill in expected_text per filename, then run\n"
        "    pytest -m integration tests/test_pipeline_integration.py::test_digit_recognizer_on_real_cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
