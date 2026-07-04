"""Compare Fd/Fnd discriminators on the labelled age cells.

Reproduces the measurement that decided the age-column design (2026-07):

    backend                    accuracy
    HTR-VT text + n-rule       14/49   (trained on bird names -> noise)
    TrOCR text + n-rule        44/49
    TrOCR text + n|u-rule      48/49   <- shipped (FuzzyMatchingAge._snap)

Usage:
    .venv/bin/python tools/compare_age_discriminators.py

Regenerate the fixture cells (after relabelling, adjust labels.json by hand):
    see the extraction snippet in tests/fixtures/cells/age/ or extract via
    MergedColumn/RowExtractor + DetectColumns on cached TATR pages.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from libs.cell_cropping import strip_cell_borders  # noqa: E402
from modules.fuzzy_matching import FuzzyMatchingAge  # noqa: E402


def main() -> int:
    age_dir = ROOT / "tests" / "fixtures" / "cells" / "age"
    labels_path = age_dir / "labels.json"
    if not labels_path.exists():
        print(f"No labels at {labels_path}")
        return 1
    labels = {k: v for k, v in json.loads(labels_path.read_text()).items() if v}
    print(f"{len(labels)} labelled cells")

    images = {}
    for name in sorted(labels):
        img = cv2.imread(str(age_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images[name] = strip_cell_borders(img)

    snapper = FuzzyMatchingAge()

    # --- TrOCR (primary reader) -----------------------------------------
    from modules.trocr import TrOCR

    trocr = TrOCR()
    if trocr._ensure_loaded():
        hits = 0
        misses = []
        for name, truth in sorted(labels.items()):
            text = trocr._recognise(images[name]) or ""
            label, score = snapper._snap(text)
            if label == truth:
                hits += 1
            else:
                misses.append((name, truth, text, label, score))
        print(f"\nTrOCR + closed-world snap: {hits}/{len(labels)}")
        for m in misses:
            print("  miss:", m)
    else:
        print("TrOCR unavailable — skipped")

    # --- HTR-VT (for reference) ------------------------------------------
    from modules.htr_vt_recognizer import HtrVtRecognizer

    htr = HtrVtRecognizer()
    if htr._ensure_loaded():
        hits = 0
        for name, truth in sorted(labels.items()):
            text = htr._predict(htr._to_pil_grayscale(images[name])) or ""
            label, _ = snapper._snap(text)
            hits += label == truth
        print(f"HTR-VT + closed-world snap: {hits}/{len(labels)} (reference)")
    else:
        print("HTR-VT unavailable — skipped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
