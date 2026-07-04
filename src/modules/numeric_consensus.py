"""Consensus scoring for the numeric columns (Aile, Poids, Heure, Jour/Mois).

Up to three recognizers run on these cells:

- ``predictions["digit"]`` — DigitRecognizer's Tesseract/CRNN reading
- ``predictions["yolo"]``  — YOLOv8 per-digit detector, composed left-to-right
- ``predictions["trocr"]`` — Microsoft TrOCR (generic handwriting OCR)

This stage normalises all available predictions and majority-votes:

- All present voices agree                  -> winner, score 100
- 2 of 3 agree                              -> majority value, score 90
- All disagree                              -> best-shaped value, score 60,
                                               losers preserved as `alternatives`
- Only one voice returned anything          -> that value, score 60/55/50
                                               (digit / yolo / trocr)
- Nothing usable                            -> cell stays blank, score -1

For the *decimal* columns (Aile, Poids) values are compared after numeric
normalisation ("10,5" == "10.5" == "010.5"). For the *compound* columns
(Heure "14:30", Jour/Mois "15.8") the digit sequence decides agreement
("14:30" == "1430") and the display value keeps the richest formatting.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from libs.columns import is_compound_column, is_numeric_column
from .module_base import Module


_NUMERIC_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _normalise(text: str) -> Optional[str]:
    """Pick the first numeric run, swap commas for dots, drop everything else."""
    if not text:
        return None
    match = _NUMERIC_RE.search(text)
    if not match:
        return None
    cleaned = match.group(0).replace(",", ".")
    # Strip leading zeros for a stable comparison, but keep one digit if value is "0".
    try:
        if "." in cleaned:
            return cleaned.lstrip("0") or "0"
        return str(int(cleaned))
    except ValueError:
        return cleaned


def _digits_key(text: str) -> Optional[str]:
    """Digit sequence of a compound value — '14:30' and '1430' compare equal."""
    if not text:
        return None
    digits = "".join(c for c in text if c.isdigit())
    return digits or None


def _compound_display(text: str) -> Optional[str]:
    """Keep digits and the separators that are meaningful on the forms."""
    if not text:
        return None
    cleaned = "".join(c for c in text if c.isdigit() or c in ".:,/-").strip(".:,/-")
    return cleaned or None


def _shape_score(text: Optional[str]) -> int:
    """Heuristic: how likely is this a clean number?"""
    if not text:
        return 0
    if re.fullmatch(r"\d{1,3}(\.\d{1,2})?", text):
        return 80
    if re.fullmatch(r"\d+", text):
        return 60
    return 20


class NumericConsensus(Module):
    """Majority voting over DigitRecognizer + YOLO + TrOCR predictions."""

    ALL_AGREE = 100
    MAJORITY = 90
    DISAGREE_SCORE = 60
    # Solo scores mirror how much we trust each backend on this material.
    SOLO_SCORES = {"digit": 60, "yolo": 55, "trocr": 50}
    # Voting order doubles as the tie-break order.
    SOURCES = ("digit", "yolo", "trocr")

    def __init__(self, debug: bool = False):
        super().__init__("numeric-consensus")
        self.debug = debug

    def get_preconditions(self) -> List[str]:
        # Whatever ran last in the recognizer chain will do — we read every
        # prediction off the cell dict itself, the precondition is mostly to
        # make sure *some* upstream populated the cells.
        return [
            "trocr",
            "digit-recognizer",
            "htr-vt-recognizer",
            "cell-formatter",
        ]

    def _resolve_pages(self, data: dict):
        for key in self.get_preconditions():
            value = data.get(key)
            if value is not None:
                return value
        return None

    def _consensus(
        self, predictions: Dict[str, str], compound: bool = False
    ) -> Tuple[str, int, list]:
        """Vote over the available predictions.

        Returns (erkannt, score, alternatives). ``compound=True`` compares on
        digit sequences and preserves separator formatting in the output.
        """
        key_fn = _digits_key if compound else _normalise
        display_fn = _compound_display if compound else _normalise

        # source -> (comparison key, display value); skip unusable predictions.
        votes: List[Tuple[str, str, str]] = []  # (source, key, display)
        for source in self.SOURCES:
            raw = predictions.get(source, "")
            key = key_fn(raw)
            display = display_fn(raw)
            if key and display:
                votes.append((source, key, display))

        if not votes:
            return "", -1, []

        if len(votes) == 1:
            source, _, display = votes[0]
            return display, self.SOLO_SCORES.get(source, 50), []

        # Tally by comparison key. Display value: the longest one in the
        # winning bucket (keeps "14:30" over "1430" for compound columns).
        buckets: Dict[str, List[Tuple[str, str]]] = {}
        for source, key, display in votes:
            buckets.setdefault(key, []).append((source, display))

        ranked = sorted(
            buckets.items(),
            key=lambda kv: (
                len(kv[1]),
                max(_shape_score(d) for _, d in kv[1]),
            ),
            reverse=True,
        )
        winner_key, winner_votes = ranked[0]
        winner_display = max((d for _, d in winner_votes), key=len)
        alternatives = [
            max((d for _, d in members), key=len)
            for key, members in ranked[1:]
        ]

        if len(winner_votes) == len(votes):
            return winner_display, self.ALL_AGREE, []
        if len(winner_votes) >= 2:
            return winner_display, self.MAJORITY, alternatives
        # Every voice disagrees — ranked[0] already prefers the best shape.
        return winner_display, self.DISAGREE_SCORE, alternatives

    def process(self, data: dict, config: dict):
        pages = self._resolve_pages(data)
        if pages is None:
            return []
        if isinstance(pages, dict):
            pages = [pages]
        if not isinstance(pages, list):
            return []

        for page_idx, page in enumerate(pages):
            for col_idx, column in enumerate(page.get("columns", [])):
                if not is_numeric_column(column):
                    continue
                compound = is_compound_column(column)
                for c_idx, cell in enumerate(column.get("cells", [])):
                    if c_idx == 0 or cell.get("is_blank") or cell.get("skip_ocr"):
                        continue

                    predictions = cell.get("predictions", {})
                    if not any(predictions.get(s) for s in self.SOURCES):
                        continue

                    erkannt, score, alternatives = self._consensus(
                        predictions, compound=compound
                    )
                    if erkannt:
                        cell["erkannt"] = erkannt
                        cell["score"] = score
                        if alternatives:
                            cell["alternatives"] = alternatives
                        cell["skip_ocr"] = True

                    if self.debug:
                        print(
                            f"[NumericConsensus] page={page_idx} col={col_idx} row={c_idx}: "
                            f"{ {s: predictions.get(s, '') for s in self.SOURCES} } "
                            f"-> {erkannt!r}/{score}"
                        )

        print("\nNumeric consensus finished!\n")
        return pages
