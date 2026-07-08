"""Tests for the delta-based edit application (libs/editing.py) and the
letter-prefix handling in ring numbers."""

import pytest

from libs.bague_sequence import rebuild_batch_sequence, split_anchor
from libs.editing import apply_cell_edit, apply_editor_deltas


def _cell(erkannt="", **extra):
    cell = {"erkannt": erkannt, "score": -1, "skip_ocr": False}
    cell.update(extra)
    return cell


def _column(flag, values, anchor_at=None):
    cells = [_cell()]  # header
    for i, v in enumerate(values, start=1):
        cells.append(_cell(v, is_anchor=(i == anchor_at)))
    column = {
        "cells": cells,
        "is_batch_column": False,
        "is_species_column": False,
        "is_sexe_column": False,
        "is_age_column": False,
        "is_jour-mois_column": False,
        "is_heure_column": False,
        "is_alle_column": False,
        "is_poids_column": False,
    }
    column[flag] = True
    return column


# ---------------------------------------------------------------------------
# split_anchor — letter prefixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("90401", ("", 90401, 5)),
        ("A90401", ("A", 90401, 5)),
        ("a90401", ("A", 90401, 5)),
        ("AB 001", ("AB", 1, 3)),
        ("  A90401  ", ("A", 90401, 5)),
        ("A-90401", ("A", 90401, 5)),
        ("", None),
        ("nur text", None),
    ],
)
def test_split_anchor(text, expected):
    assert split_anchor(text) == expected


def test_rebuild_keeps_letter_prefix():
    """Jan: anchor 'A90401' must not degrade to '90401' — and the whole
    column continues as A90402, A90403, ..."""
    column = _column("is_batch_column", ["", "", ""], anchor_at=1)
    rebuild_batch_sequence(column, anchor_value="A90401")
    erkannt = [c["erkannt"] for c in column["cells"][1:]]
    assert erkannt == ["A90401", "A90402", "A90403"]


def test_rebuild_prefix_with_zero_padding_and_backwards():
    column = _column("is_batch_column", ["", "", ""], anchor_at=2)
    rebuild_batch_sequence(column, anchor_value="B008")
    erkannt = [c["erkannt"] for c in column["cells"][1:]]
    assert erkannt == ["B007", "B008", "B009"]


# ---------------------------------------------------------------------------
# apply_cell_edit
# ---------------------------------------------------------------------------


def test_edit_normal_cell_marks_manual():
    column = _column("is_alle_column", ["68"])
    page = {"columns": [column]}
    result = apply_cell_edit(page, 0, 1, "72")
    assert result.changed
    cell = column["cells"][1]
    assert cell["erkannt"] == "72"
    assert cell["is_manual_edit"] is True
    assert cell["skip_ocr"] is True
    assert cell["score"] == 100


def test_edit_identical_value_is_noop():
    column = _column("is_alle_column", ["68"])
    page = {"columns": [column]}
    result = apply_cell_edit(page, 0, 1, "68")
    assert not result.changed
    assert "is_manual_edit" not in column["cells"][1]


def test_edit_accepts_ditto_mark():
    """Jan: Gänsefüßchen must be enterable and survive verbatim."""
    column = _column("is_species_column", ["Buchfink"])
    page = {"columns": [column]}
    result = apply_cell_edit(page, 0, 1, '"')
    assert result.changed
    assert column["cells"][1]["erkannt"] == '"'
    assert result.unknown_species is None  # ditto is not an unknown species


def test_edit_anchor_rebuilds_column_with_prefix():
    column = _column("is_batch_column", ["90401", "90402", "90403"], anchor_at=1)
    page = {"columns": [column]}
    result = apply_cell_edit(page, 0, 1, "A90401")
    assert result.anchor_rebuilt
    erkannt = [c["erkannt"] for c in column["cells"][1:]]
    assert erkannt == ["A90401", "A90402", "A90403"]


def test_edit_unknown_species_reported():
    column = _column("is_species_column", ["Buchfink"])
    page = {"columns": [column]}
    result = apply_cell_edit(page, 0, 1, "Phantasievogel", species_catalog=["Buchfink"])
    assert result.unknown_species == "Phantasievogel"


# ---------------------------------------------------------------------------
# apply_editor_deltas — st.data_editor edited_rows semantics
# ---------------------------------------------------------------------------


def _two_column_page():
    return {
        "columns": [
            _column("is_batch_column", ["90401", "90402", "90403"], anchor_at=1),
            _column("is_alle_column", ["68", "72", "75"]),
        ]
    }


VISIBLE = [(0, "Bague"), (1, "Aile")]


def test_deltas_applied_via_display_positions():
    page = _two_column_page()
    # Unfiltered view: display position p corresponds to row p+1.
    deltas = {1: {"Aile": "99"}}
    apply_editor_deltas(page, deltas, displayed_index=[1, 2, 3], visible=VISIBLE)
    assert page["columns"][1]["cells"][2]["erkannt"] == "99"


def test_deltas_respect_filtered_index():
    """With the review filter active, display position 0 can be row 3."""
    page = _two_column_page()
    deltas = {0: {"Aile": "11"}}
    apply_editor_deltas(page, deltas, displayed_index=[3], visible=VISIBLE)
    assert page["columns"][1]["cells"][3]["erkannt"] == "11"
    assert page["columns"][1]["cells"][1]["erkannt"] == "68"


def test_deltas_are_idempotent():
    page = _two_column_page()
    deltas = {0: {"Aile": "99"}}
    r1 = apply_editor_deltas(page, deltas, displayed_index=[1, 2, 3], visible=VISIBLE)
    r2 = apply_editor_deltas(page, deltas, displayed_index=[1, 2, 3], visible=VISIBLE)
    assert any(r.changed for r in r1)
    assert not any(r.changed for r in r2), "re-applying the same delta must be a no-op"


def test_deltas_ignore_helper_columns():
    page = _two_column_page()
    deltas = {0: {"✓": "🟢", "📷 Aile": "data:image/png;base64,x"}}
    results = apply_editor_deltas(page, deltas, displayed_index=[1], visible=VISIBLE)
    assert not any(r.changed for r in results)
