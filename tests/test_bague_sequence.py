"""Tests for the bague (ring number) sequence helpers.

These tests are pure-Python and need no fixtures, models, or sample scans.
They guarantee that the sequence reconstruction logic — the centerpiece of
the robust bague digitisation — keeps doing the right thing even when the
implementation changes.
"""

from libs.bague_sequence import (
    annotate_initial_sequence,
    annotate_per_cell_fallback,
    mark_manual_edit,
    parse_anchor,
    rebuild_batch_sequence,
)


# ---------------------------------------------------------------------------
# parse_anchor
# ---------------------------------------------------------------------------


def test_parse_anchor_plain_digits():
    assert parse_anchor("142921") == 142921


def test_parse_anchor_with_noise():
    assert parse_anchor("14292 1") == 142921
    assert parse_anchor("Bague 142921") == 142921
    assert parse_anchor(" 142921 ") == 142921


def test_parse_anchor_returns_none_for_garbage():
    assert parse_anchor("") is None
    assert parse_anchor(None) is None
    assert parse_anchor("abc") is None


# ---------------------------------------------------------------------------
# rebuild_batch_sequence
# ---------------------------------------------------------------------------


def _make_column(num_data_cells: int, anchor_value: str = "") -> dict:
    cells = [{"erkannt": "", "score": -1, "skip_ocr": False}]  # header
    for i in range(num_data_cells):
        cells.append(
            {
                "erkannt": "" if i > 0 else anchor_value,
                "score": -1,
                "skip_ocr": bool(anchor_value) if i == 0 else False,
                "is_anchor": i == 0,
                "is_extrapolated": i > 0,
            }
        )
    return {"cells": cells, "is_batch_column": True}


def test_rebuild_from_anchor_text():
    col = _make_column(5, anchor_value="142921")

    stats = rebuild_batch_sequence(col)
    assert stats["anchor"] == 142921
    assert stats["filled"] == 5
    assert stats["preserved"] == 0

    erkannt = [c["erkannt"] for c in col["cells"]]
    assert erkannt == ["", "142921", "142922", "142923", "142924", "142925"]


def test_rebuild_handles_wrap_around():
    """9 → 0 must increment the full number, not just the suffix."""
    col = _make_column(12, anchor_value="142929")

    rebuild_batch_sequence(col)
    erkannt = [c["erkannt"] for c in col["cells"][1:]]
    assert erkannt == [
        "142929", "142930", "142931", "142932", "142933", "142934",
        "142935", "142936", "142937", "142938", "142939", "142940",
    ]


def test_rebuild_extrapolates_backwards_from_mid_column_anchor():
    """The anchor can sit anywhere (anchor-anywhere detection). Rows ABOVE it
    must be re-sequenced too — a corrected anchor must never leave stale
    values in the cells before it."""
    col = _make_column(7)
    # Move the anchor to data cell index 3 (cells[4]) with garbage above.
    for cell in col["cells"][1:]:
        cell["is_anchor"] = False
        cell["erkannt"] = "stale"
    col["cells"][4]["is_anchor"] = True

    rebuild_batch_sequence(col, anchor_value="144705")

    erkannt = [c["erkannt"] for c in col["cells"][1:]]
    assert erkannt == [
        "144702", "144703", "144704", "144705", "144706", "144707", "144708"
    ]


def test_rebuild_backwards_respects_manual_edits():
    col = _make_column(4)
    for cell in col["cells"][1:]:
        cell["is_anchor"] = False
    col["cells"][3]["is_anchor"] = True
    col["cells"][1]["erkannt"] = "MANUELL"
    col["cells"][1]["is_manual_edit"] = True

    stats = rebuild_batch_sequence(col, anchor_value="500")

    assert col["cells"][1]["erkannt"] == "MANUELL"
    assert col["cells"][2]["erkannt"] == "499"
    assert col["cells"][3]["erkannt"] == "500"
    assert col["cells"][4]["erkannt"] == "501"
    assert stats["preserved"] == 1


def test_rebuild_backwards_stops_at_zero():
    """No negative or zero ring numbers when the anchor is tiny."""
    col = _make_column(5)
    for cell in col["cells"][1:]:
        cell["is_anchor"] = False
        cell["erkannt"] = "stale"
    col["cells"][4]["is_anchor"] = True

    rebuild_batch_sequence(col, anchor_value="2")

    erkannt = [c["erkannt"] for c in col["cells"][1:]]
    # backwards: 1 fits, 0 and below do not -> cells stay untouched there
    assert erkannt[2] == "1"
    assert erkannt[3] == "2"
    assert erkannt[4] == "3"
    assert erkannt[0] == "stale" and erkannt[1] == "stale"


def test_rebuild_preserves_manual_edits():
    col = _make_column(6, anchor_value="100")

    # User manually fixed row 3.
    col["cells"][3]["erkannt"] = "999"
    mark_manual_edit(col["cells"][3])

    stats = rebuild_batch_sequence(col)
    erkannt = [c["erkannt"] for c in col["cells"][1:]]
    assert erkannt == ["100", "101", "999", "103", "104", "105"]
    assert stats["preserved"] == 1
    assert stats["filled"] == 5  # anchor + 4 extrapolated


def test_rebuild_with_new_anchor_value_overwrites():
    """If we pass a fresh anchor value (UI edit) the column should re-flow."""
    col = _make_column(4, anchor_value="100")
    rebuild_batch_sequence(col)  # initial fill

    stats = rebuild_batch_sequence(col, anchor_value="555")
    assert stats["anchor"] == 555
    erkannt = [c["erkannt"] for c in col["cells"][1:]]
    assert erkannt == ["555", "556", "557", "558"]


def test_rebuild_overwrite_manual_flag_resets_everything():
    col = _make_column(4, anchor_value="100")
    col["cells"][2]["erkannt"] = "MANUAL"
    mark_manual_edit(col["cells"][2])

    rebuild_batch_sequence(col, overwrite_manual=True)
    erkannt = [c["erkannt"] for c in col["cells"][1:]]
    assert erkannt == ["100", "101", "102", "103"]


def test_rebuild_without_anchor_does_nothing():
    col = _make_column(3, anchor_value="")
    stats = rebuild_batch_sequence(col)
    assert stats["anchor"] is None
    assert stats["filled"] == 0


def test_rebuild_pads_leading_zeros_when_widths_match():
    col = _make_column(3, anchor_value="001")
    rebuild_batch_sequence(col)
    erkannt = [c["erkannt"] for c in col["cells"][1:]]
    assert erkannt == ["001", "002", "003"]


# ---------------------------------------------------------------------------
# annotate_initial_sequence / annotate_per_cell_fallback
# ---------------------------------------------------------------------------


def test_annotate_initial_sequence_flags():
    col = _make_column(4)
    sequence = ["100", "101", "102", "103"]
    # Per-cell OCR confirmed every second cell.
    per_cell = ["0", "1", None, "3"]

    annotate_initial_sequence(col, sequence, per_cell)

    cells = col["cells"][1:]
    assert cells[0]["is_anchor"] is True
    assert cells[0]["is_extrapolated"] is False
    assert cells[0]["score"] == 100

    # Cell 1: confirmed by per-cell OCR
    assert cells[1]["is_anchor"] is False
    assert cells[1]["is_extrapolated"] is False
    assert cells[1]["score"] == 100

    # Cell 2: no per-cell observation -> extrapolated only
    assert cells[2]["is_extrapolated"] is True
    assert cells[2]["score"] == 70

    # Cell 3: per-cell OCR confirms suffix "3"
    assert cells[3]["is_extrapolated"] is False
    assert cells[3]["score"] == 100


def test_annotate_per_cell_fallback_no_anchor():
    col = _make_column(4)
    per_cell = ["2", "3", None, "5"]

    annotate_per_cell_fallback(col, per_cell)

    cells = col["cells"][1:]
    assert cells[0]["erkannt"] == "2"
    assert cells[1]["erkannt"] == "3"
    assert cells[2]["erkannt"] == ""  # None observation stays blank
    assert cells[3]["erkannt"] == "5"
    assert all(not c.get("is_extrapolated") for c in cells)


# ---------------------------------------------------------------------------
# mark_manual_edit
# ---------------------------------------------------------------------------


def test_mark_manual_edit_clears_extrapolated_flag():
    cell = {"is_extrapolated": True, "skip_ocr": False}
    mark_manual_edit(cell)
    assert cell["is_extrapolated"] is False
    assert cell["skip_ocr"] is True
