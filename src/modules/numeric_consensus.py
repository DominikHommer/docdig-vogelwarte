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
from libs.column_schema import ROLE_TO_FLAG, load_schema
from .module_base import Module


def _value_ranges_by_flag() -> dict:
    """{is_*_column flag: (lo, hi)} for schema roles with a value_range."""
    ranges = {}
    try:
        for col in load_schema().get("columns", []):
            vr = col.get("value_range")
            flag = ROLE_TO_FLAG.get(col.get("role"))
            if vr and flag:
                ranges[flag] = (float(vr[0]), float(vr[1]))
    except Exception:
        pass
    return ranges


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

    # Winner lies outside the column's plausible value range (schema):
    IMPLAUSIBLE_SCORE = 30

    def __init__(self, debug: bool = False):
        super().__init__("numeric-consensus")
        self.debug = debug
        self._value_ranges = _value_ranges_by_flag()

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

        # Decimal reconciliation (decimal columns only): Tesseract and YOLO
        # cannot see the decimal point — they read "10.5" as "105" and used
        # to OUTVOTE TrOCR's correct "10.5" two to one. When values share the
        # same digit sequence and exactly one dotted variant exists, they are
        # the SAME reading — merge onto the dotted one.
        if not compound:
            dotted_by_digits: Dict[str, set] = {}
            for _, key, display in votes:
                if "." in display:
                    dotted_by_digits.setdefault(
                        _digits_key(display) or "", set()
                    ).add(display)
            merged_votes = []
            for source, key, display in votes:
                digits = _digits_key(display) or ""
                dotted = dotted_by_digits.get(digits, set())
                if "." not in display and len(dotted) == 1:
                    display = next(iter(dotted))
                    key = _normalise(display) or key
                merged_votes.append((source, key, display))
            votes = merged_votes

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

    @staticmethod
    def _in_range(value: str, value_range: Tuple[float, float]) -> bool:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return value_range[0] <= number <= value_range[1]

    def _apply_value_range(
        self, column: dict, erkannt: str, score: int, alternatives: list
    ) -> Tuple[str, int, list]:
        """Plausibility arbitration (Aile in mm, Poids in g — from the schema).

        Neighbour-column bleed produces values like '785' or '412' on cells
        whose truth is 78 / 9.5. If the winner is outside the plausible range
        but an alternative fits, the alternative wins (winner kept as
        alternative); if nothing fits, the value stays but is flagged red.
        """
        value_range = None
        for flag, vr in self._value_ranges.items():
            if column.get(flag, False):
                value_range = vr
                break
        if value_range is None or not erkannt:
            return erkannt, score, alternatives

        if self._in_range(erkannt, value_range):
            return erkannt, score, alternatives

        for alt in alternatives:
            if self._in_range(alt, value_range):
                rest = [a for a in alternatives if a != alt]
                return alt, self.DISAGREE_SCORE, rest + [erkannt]

        return erkannt, min(score, self.IMPLAUSIBLE_SCORE), alternatives

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
                    if not compound:
                        erkannt, score, alternatives = self._apply_value_range(
                            column, erkannt, score, alternatives
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
