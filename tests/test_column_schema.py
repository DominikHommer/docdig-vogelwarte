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


def test_flags_for_role_complete_and_exclusive():
    flags = flags_for_role("batch")
    assert flags["is_batch_column"] is True
    assert sum(flags.values()) == 1
    assert set(flags.keys()) == set(ROLE_TO_FLAG.values())
    assert sum(flags_for_role(None).values()) == 0
