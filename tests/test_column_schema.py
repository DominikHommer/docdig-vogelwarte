"""Tests for schema-based column assignment (libs/column_schema.py).

The "real page" cases use header texts + widths measured on the actual
scan_1972 pages in this repo (data/input/tatr, page 0) — including the
columns whose header OCR comes back empty.
"""

import pytest

from libs.column_schema import (
    ROLE_TO_FLAG,
    assign_roles,
    flags_for_role,
    keyword_score,
    load_schema,
    width_score,
)


# Measured on data/input/tatr/page_25.jpg (scan_1972_CdB_10): 11 extracted
# columns, 5 readable headers, Espèce/Age/Heure headers unreadable.
REAL_HEADERS = [
    "Bague No", "", "d'Sexe ®", "", "Jour Mois", "", "Alle", "Poids",
    "", "", "falsser en blanc:",
]
REAL_WIDTHS = [129, 386, 104, 156, 159, 105, 139, 112, 87, 87, 129]

EXPECTED_ROLES = [
    "batch", "species", "sexe", "age", "jour_mois", "heure", "aile", "poids",
    None, None, None,
]


def test_real_page_layout_fully_assigned():
    """The flagship case: every semantic column lands on the right position,
    even the three whose header OCR failed (inferred from order + width)."""
    assert assign_roles(REAL_HEADERS, REAL_WIDTHS) == EXPECTED_ROLES


def test_bague_can_never_move_to_a_later_column():
    """Jan's bug: 'Bague' showed up as UI column 5. With monotone alignment
    a later junk header cannot steal an earlier role when the real anchor
    matched before it."""
    headers = list(REAL_HEADERS)
    headers[5] = "No"  # OCR junk that substring-matched 'bague no' before
    roles = assign_roles(headers, REAL_WIDTHS)
    assert roles[0] == "batch"
    assert roles[5] != "batch"


def test_junk_short_header_gives_no_keyword_evidence():
    """The old bug: 'no' substring-matched 'bague no' anywhere. A lone 'No'
    fragment must contribute (almost) nothing as keyword evidence — only
    width+order may place a role on such a column."""
    schema = load_schema()
    batch = next(c for c in schema["columns"] if c["role"] == "batch")
    assert keyword_score("No", batch["keywords"]) < 0.15
    assert keyword_score("N0", batch["keywords"]) < 0.15


def test_junk_header_cannot_outrank_a_real_anchor():
    """When the real 'Bague No' header exists anywhere, a 'No' fragment on
    another column must never win the batch role."""
    headers = ["Bague No", "No", "", ""]
    widths = [100, 100, 300, 100]
    roles = assign_roles(headers, widths)
    assert roles[0] == "batch"
    assert roles[1] != "batch"


def test_missing_column_is_skipped_not_shifted():
    """If a column was not extracted, later roles must NOT shift left onto
    the wrong neighbours."""
    headers = ["Bague No", "", "d'Sexe", "Jour Mois", "Heure", "Aile", "Poids"]
    #                       age missing ^
    widths = [129, 386, 104, 159, 105, 139, 112]
    roles = assign_roles(headers, widths)
    assert roles == ["batch", "species", "sexe", "jour_mois", "heure", "aile", "poids"]


def test_ocr_misreads_still_match():
    headers = ["Bagve N0", "Espèce", "Sexe", "Age", "Jour Mo1s", "Heure", "Aiie", "Polds"]
    widths = [129, 386, 104, 156, 159, 105, 139, 112]
    roles = assign_roles(headers, widths)
    assert roles == [
        "batch", "species", "sexe", "age", "jour_mois", "heure", "aile", "poids"
    ]


def test_widths_alone_never_make_the_wide_species_column_a_bague():
    """All headers unreadable: the 386px species column must not become the
    ring column — its width share is far outside the bague range."""
    headers = [""] * len(REAL_WIDTHS)
    roles = assign_roles(headers, REAL_WIDTHS)
    batch_positions = [i for i, r in enumerate(roles) if r == "batch"]
    assert 1 not in batch_positions  # the wide column
    species_positions = [i for i, r in enumerate(roles) if r == "species"]
    assert species_positions in ([], [1])


def test_each_role_at_most_once():
    roles = assign_roles(REAL_HEADERS, REAL_WIDTHS)
    assigned = [r for r in roles if r]
    assert len(assigned) == len(set(assigned))


def test_roles_preserve_schema_order():
    schema_order = [c["role"] for c in load_schema()["columns"]]
    roles = [r for r in assign_roles(REAL_HEADERS, REAL_WIDTHS) if r]
    assert roles == [r for r in schema_order if r in roles]


def test_keyword_score_basics():
    schema = load_schema()
    bague = next(c for c in schema["columns"] if c["role"] == "batch")
    assert keyword_score("Bague No", bague["keywords"]) > 0.9
    assert keyword_score("", bague["keywords"]) == 0.0
    assert keyword_score("Poids", bague["keywords"]) < 0.5


def test_width_score_band():
    assert width_score(0.08, [0.04, 0.14]) == 1.0
    assert width_score(0.30, [0.04, 0.14]) < 0.5
    assert width_score(0.0, [0.04, 0.14]) == 0.0


# ---------------------------------------------------------------------------
# UI bridge: editable rows <-> schema, user override persistence
# ---------------------------------------------------------------------------

from libs import column_schema as cs
from libs.column_schema import (
    reset_schema,
    rows_to_schema,
    save_schema,
    schema_to_rows,
)


def test_rows_roundtrip_preserves_schema():
    schema = load_schema()
    rows = schema_to_rows(schema)
    rebuilt = rows_to_schema(rows, base_schema=schema)
    assert [c["role"] for c in rebuilt["columns"]] == [
        c["role"] for c in schema["columns"]
    ]
    assert rebuilt["columns"][0]["keywords"] == schema["columns"][0]["keywords"]


def test_rows_reorder_and_deactivate():
    schema = load_schema()
    rows = schema_to_rows(schema)
    # Move poids to the front and deactivate heure.
    for row in rows:
        if row["_role"] == "poids":
            row["Pos"] = 0
        if row["_role"] == "heure":
            row["Aktiv"] = False
    rebuilt = rows_to_schema(rows, base_schema=schema)
    assert rebuilt["columns"][0]["role"] == "poids"
    heure = next(c for c in rebuilt["columns"] if c["role"] == "heure")
    assert heure["enabled"] is False

    # Deactivated roles never get assigned.
    roles = assign_roles(REAL_HEADERS, REAL_WIDTHS, schema=rebuilt)
    assert "heure" not in roles


def test_rows_keyword_edit_flows_into_matching():
    schema = load_schema()
    rows = schema_to_rows(schema)
    for row in rows:
        if row["_role"] == "batch":
            row["Header-Stichwörter"] = "ringnummer, serie"
    rebuilt = rows_to_schema(rows, base_schema=schema)
    batch = rebuilt["columns"][0]
    assert batch["keywords"] == ["ringnummer", "serie"]
    assert keyword_score("Ringnummer", batch["keywords"]) > 0.9


def test_custom_override_load_save_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "CUSTOM_SCHEMA_PATH", tmp_path / "custom.json")
    base = load_schema()
    assert not cs.CUSTOM_SCHEMA_PATH.exists()

    modified = dict(base)
    modified["columns"] = list(base["columns"])
    modified["min_assign_score"] = 0.5
    save_schema(modified)
    assert cs.CUSTOM_SCHEMA_PATH.exists()
    assert load_schema()["min_assign_score"] == 0.5

    assert reset_schema() is True
    assert load_schema().get("min_assign_score") != 0.5
    assert reset_schema() is False


def test_flags_for_role_complete_and_exclusive():
    flags = flags_for_role("batch")
    assert flags["is_batch_column"] is True
    assert sum(flags.values()) == 1
    assert set(flags.keys()) == set(ROLE_TO_FLAG.values())
    assert sum(flags_for_role(None).values()) == 0
