"""Compare the digit-recognition backends against labelled fixture cells.

Runs Tesseract, the Keras CRNN and the YOLO detector *individually* on every
labelled cell in ``tests/fixtures/cells/bague/`` and prints a per-cell table
plus per-backend accuracy — so you can judge each voice on real data and
decide which backends to enable (see DOCDIG_DIGIT_BACKENDS).

Usage:
    .venv/bin/python tools/compare_digit_backends.py
    .venv/bin/python tools/compare_digit_backends.py --no-strip   # without border cropping
    .venv/bin/python tools/compare_digit_backends.py --cells DIR --labels FILE

Notes:
- The bague suffix cells contain a single printed digit; a backend "hits"
  when its *last* digit matches the label's last digit (leading artefacts
  from neighbouring columns don't count against it) and "exact" when the
  full string matches the expected suffix digit.
- Blank verdicts come from the same gate the pipeline uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from libs.bague_sequence import is_blank_cell  # noqa: E402
from libs.cell_cropping import strip_cell_borders  # noqa: E402
from libs.yolo_digits import YoloDigitReader  # noqa: E402
from modules.digit_recognizer import DigitRecognizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells", type=Path, default=ROOT / "tests" / "fixtures" / "cells" / "bague"
    )
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument(
        "--no-strip", action="store_true", help="skip table-border removal"
    )
    args = parser.parse_args()

    labels_path = args.labels or (args.cells / "labels.json")
    if not labels_path.exists():
        print(f"No labels found at {labels_path}")
        return 1
    labels = {
        k: str(v) for k, v in json.loads(labels_path.read_text()).items() if v
    }
    if not labels:
        print("labels.json contains no labelled cells")
        return 1

    recognizer = DigitRecognizer()  # all backends, lazy-loaded
    yolo = YoloDigitReader()

    stats = {
        name: {"exact": 0, "last": 0, "empty": 0}
        for name in ("tesseract", "keras", "yolo")
    }
    rows = []
    tested = 0

    for name in sorted(labels):
        img_path = args.cells / name
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if not args.no_strip:
            cleaned = strip_cell_borders(img)
            if cleaned is not None:
                img = cleaned

        truth = labels[name]
        # Mirror the pipeline's blank gate (CellFormatter): strict threshold,
        # small margins on the already-stripped crop.
        blank = (
            is_blank_cell(img, threshold=0.004, margin_y=3, margin_x=3)
            if not args.no_strip
            else is_blank_cell(img, threshold=0.004)
        )
        if blank:
            rows.append((name, truth, "<blank>", "<blank>", "<blank>"))
            continue

        tested += 1
        preds = {
            "tesseract": recognizer._tesseract_digits(img, psm=7),
            "keras": recognizer._keras_predict(img),
            "yolo": yolo.read(img)[0],
        }
        for backend, text in preds.items():
            if not text:
                stats[backend]["empty"] += 1
                continue
            # Suffix cells carry one printed digit; anchor cells the full number.
            if text == truth or text == truth[-1]:
                stats[backend]["exact"] += 1
            if text[-1] == truth[-1]:
                stats[backend]["last"] += 1

        rows.append(
            (name, truth, preds["tesseract"] or "-", preds["keras"] or "-", preds["yolo"] or "-")
        )

    name_w = max(len(r[0]) for r in rows) if rows else 20
    print(f"\n{'cell':<{name_w}}  {'truth':>8}  {'tesseract':>10}  {'crnn':>8}  {'yolo':>8}")
    print("-" * (name_w + 42))
    for name, truth, tess, keras, yolo_text in rows:
        print(f"{name:<{name_w}}  {truth:>8}  {tess:>10}  {keras:>8}  {yolo_text:>8}")

    print(f"\nTested cells (non-blank): {tested}")
    print(f"{'backend':<10} {'exact':>7} {'last-digit':>11} {'no output':>10}")
    for backend, s in stats.items():
        print(
            f"{backend:<10} {s['exact']:>4}/{tested:<3} {s['last']:>7}/{tested:<3} {s['empty']:>7}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
