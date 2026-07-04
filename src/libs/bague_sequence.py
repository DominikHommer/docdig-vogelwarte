"""Shared helpers for reasoning about the bague (ring number) column.

Both `DigitRecognizer` and the Streamlit UI rebuild bague sequences from a
single anchor cell, so the logic lives here to keep the two in sync.

Convention on the cell dictionaries used by the bague column:

- The first non-header data cell carries `is_anchor=True`. Its `erkannt` text
  is the full ring number ("142921").
- Every other data cell carries `is_extrapolated=True` while it holds a value
  derived from the anchor (`start + offset`). A manual edit clears that flag,
  so subsequent rebuilds will not overwrite the user's correction.
- `score` reflects confidence: 100 (anchor or OCR-confirmed), 70 (extrapolated
  only), 30 (per-cell OCR without anchor), -1 (no value).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def is_blank_cell(
    image,
    threshold: float = 0.015,
    margin_y: Optional[int] = None,
    margin_x: Optional[int] = None,
) -> bool:
    """True when the cell crop has almost no dark pixels in its centre — i.e.
    the row was never filled in by the bander.

    The default centre crop removes table borders that would otherwise count
    as "content" and trick the detector. When the caller already stripped the
    borders (libs/cell_cropping.strip_cell_borders), pass small ``margin_y``/
    ``margin_x`` values — otherwise content sitting near the cell edge (e.g.
    the printed suffix digit on bague forms) is cropped away and the cell is
    falsely reported blank.
    """
    if image is None:
        return True
    arr = np.asarray(image)
    if arr.size == 0:
        return True
    if arr.ndim == 3:
        arr = (
            0.299 * arr[:, :, 0]
            + 0.587 * arr[:, :, 1]
            + 0.114 * arr[:, :, 2]
        )
    if arr.dtype != np.uint8:
        mx = float(arr.max()) if arr.size else 1.0
        mn = float(arr.min()) if arr.size else 0.0
        if 0.0 <= mn and mx <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Strip a margin to ignore the top/bottom/left/right table borders.
    h, w = arr.shape[:2]
    if h < 20 or w < 20:
        return False
    if margin_y is None:
        margin_y = max(4, h // 6)
    if margin_x is None:
        margin_x = max(4, w // 8)
    inner = arr[margin_y : h - margin_y, margin_x : w - margin_x]
    if inner.size == 0:
        return False
    non_white = float((inner < 200).sum()) / float(inner.size)
    return non_white < threshold


def parse_anchor(text: Optional[str]) -> Optional[int]:
    """Best-effort parse of an anchor cell's free-text value to an int."""
    if not text:
        return None
    digits = "".join(c for c in str(text) if c.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def mark_manual_edit(cell: Dict) -> None:
    """Call this when the user manually changes a cell — protects it from rebuild.

    ``is_manual_edit`` is the rebuild-protection flag; ``is_extrapolated`` is
    independent and only governs the UI marker. The two used to be one flag,
    which broke rebuild after a successful per-cell OCR cross-check.
    """
    cell["is_manual_edit"] = True
    cell["is_extrapolated"] = False
    cell["skip_ocr"] = True


def rebuild_batch_sequence(
    column: Dict,
    anchor_value: Optional[str] = None,
    *,
    overwrite_manual: bool = False,
) -> Dict[str, int]:
    """
    Re-sequence a batch column from its anchor.

    Args:
        column: a column dict from the pipeline. Must contain ``cells``.
        anchor_value: optional override for the anchor. If ``None`` we read
            the value from the cell currently marked ``is_anchor`` (falling
            back to the first non-header cell).
        overwrite_manual: when ``True``, every non-header cell is rewritten
            regardless of ``is_extrapolated``. Use sparingly — meant for an
            explicit "Reset Bague" UI action.

    Returns:
        Stats dict with keys ``anchor`` (parsed int or None), ``filled``,
        ``preserved``, ``empty``.
    """
    cells = column.get("cells", [])
    stats = {"anchor": None, "filled": 0, "preserved": 0, "empty": 0}

    # Find the anchor cell (first non-header cell that is_anchor, else data[0]).
    anchor_idx = None
    for c_idx, cell in enumerate(cells):
        if c_idx == 0:
            continue
        if cell.get("is_anchor"):
            anchor_idx = c_idx
            break
    if anchor_idx is None:
        # Fall back: first non-header cell is the anchor by definition.
        for c_idx, cell in enumerate(cells):
            if c_idx == 0:
                continue
            cell["is_anchor"] = True
            anchor_idx = c_idx
            break
    if anchor_idx is None:
        return stats  # no data cells

    anchor_cell = cells[anchor_idx]
    if anchor_value is not None:
        anchor_cell["erkannt"] = anchor_value
        anchor_cell["is_extrapolated"] = False  # the anchor itself is never "extrapolated"
        anchor_cell["skip_ocr"] = True

    anchor_text = (anchor_cell.get("erkannt") or "").strip()
    start = parse_anchor(anchor_text)
    stats["anchor"] = start
    if start is None:
        # Without a parseable anchor we cannot extrapolate. Leave cells alone.
        return stats

    # Preserve the anchor's printed width so "001" stays zero-padded, not "1".
    digit_chars = [c for c in anchor_text if c.isdigit()]
    width = max(len(digit_chars), len(str(start)))

    def _fill(cell: Dict, target: str) -> None:
        is_manual = cell.get("is_manual_edit", False)
        if (not is_manual) or overwrite_manual:
            cell["erkannt"] = target
            cell["is_extrapolated"] = True
            cell["is_manual_edit"] = False
            cell["skip_ocr"] = True
            if cell.get("score", -1) < 70:
                cell["score"] = 70
            stats["filled"] += 1
        else:
            # Manual edit — leave alone.
            stats["preserved"] += 1
            if not (cell.get("erkannt") or "").strip():
                stats["empty"] += 1

    # Forward from the anchor (anchor itself included).
    for offset, cell in enumerate(cells[anchor_idx:]):
        target = str(start + offset)
        if len(target) < width:
            target = target.zfill(width)

        if offset == 0:
            cell["erkannt"] = target
            cell["score"] = max(cell.get("score", -1), 100)
            cell["skip_ocr"] = True
            cell["is_anchor"] = True
            cell["is_extrapolated"] = False
            stats["filled"] += 1
            continue

        _fill(cell, target)

    # Backwards from the anchor — the anchor can sit anywhere in the column
    # (blank top rows, mid-column printed number), so the rows above must be
    # re-sequenced too, otherwise a corrected anchor leaves stale values there.
    for back, cell in enumerate(reversed(cells[1:anchor_idx]), start=1):
        value = start - back
        if value <= 0:
            break
        target = str(value)
        if len(target) < width:
            target = target.zfill(width)
        _fill(cell, target)

    return stats


def annotate_initial_sequence(
    column: Dict,
    sequence: List[str],
    per_cell_observations: List[Optional[str]],
    blank_mask: Optional[List[bool]] = None,
    anchor_data_idx: int = 0,
) -> None:
    """
    Apply the initial sequence to a freshly-extracted batch column.

    Args:
        column: column dict from the pipeline.
        sequence: one full ring number per non-blank data cell, in order.
        per_cell_observations: single-digit OCR per data cell (length == number
            of data cells, includes entries even for blank cells).
        blank_mask: boolean per data cell — True means "leave this row empty
            (the bander did not fill it in)". Defaults to all-False.
        anchor_data_idx: index (in data-cell space) of the cell that yielded
            the anchor — this gets the ⚓ marker.
    """
    cells = column.get("cells", [])
    data_cells = [c for c_idx, c in enumerate(cells) if c_idx > 0]
    if blank_mask is None:
        blank_mask = [False] * len(data_cells)

    seq_iter = iter(sequence)
    for data_idx, (cell, is_blank, observed) in enumerate(
        zip(data_cells, blank_mask, per_cell_observations)
    ):
        if is_blank:
            cell["erkannt"] = ""
            cell["is_anchor"] = False
            cell["is_extrapolated"] = False
            cell["is_blank"] = True
            cell["skip_ocr"] = True
            cell["score"] = -1
            continue

        cell["is_blank"] = False
        try:
            full_number = next(seq_iter)
        except StopIteration:
            cell["erkannt"] = ""
            cell["is_anchor"] = False
            cell["is_extrapolated"] = False
            cell["score"] = -1
            continue

        cell["erkannt"] = full_number
        cell["skip_ocr"] = True
        cell["is_anchor"] = data_idx == anchor_data_idx
        if cell["is_anchor"]:
            cell["is_extrapolated"] = False
            cell["score"] = 100
        else:
            confirmed = bool(
                observed
                and observed.isdigit()
                and observed[-1] == full_number[-1]
            )
            cell["is_extrapolated"] = not confirmed
            cell["score"] = 100 if confirmed else 70


def annotate_per_cell_fallback(
    column: Dict,
    per_cell_observations: List[Optional[str]],
) -> None:
    """No-anchor fallback: every cell gets whatever per-cell OCR returned."""
    cells = column.get("cells", [])
    data_cells = [c for c_idx, c in enumerate(cells) if c_idx > 0]

    for offset, (cell, observed) in enumerate(zip(data_cells, per_cell_observations)):
        cell["is_anchor"] = offset == 0
        cell["is_extrapolated"] = False
        if observed and observed.isdigit():
            cell["erkannt"] = observed
            cell["score"] = 30
            cell["skip_ocr"] = True
