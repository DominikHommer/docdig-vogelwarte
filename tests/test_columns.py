"""Tests for the shared column-flag helpers (libs/columns.py)."""

from libs.columns import (
    ALL_COLUMN_FLAGS,
    COLUMN_FLAG_TO_LABEL,
    NUMERIC_COLUMN_FLAGS,
    column_label,
    is_compound_column,
    is_numeric_column,
)


def _col(**flags):
    base = {flag: False for flag in ALL_COLUMN_FLAGS}
    base.update(flags)
    return base


def test_every_form_column_has_a_label():
    # Order matters: this is the order the columns appear on the forms.
    assert list(COLUMN_FLAG_TO_LABEL.values()) == [
        "Bague",
        "Espèce",
        "Sexe",
        "Age",
        "Jour/Mois",
        "Heure",
        "Aile",
        "Poids",
    ]


def test_numeric_columns_cover_all_number_bearing_types():
    assert set(NUMERIC_COLUMN_FLAGS) == {
        "is_alle_column",
        "is_poids_column",
        "is_heure_column",
        "is_jour-mois_column",
    }


def test_is_numeric_column():
    assert is_numeric_column(_col(is_alle_column=True))
    assert is_numeric_column(_col(is_poids_column=True))
    assert is_numeric_column(_col(is_heure_column=True))
    assert is_numeric_column(_col(**{"is_jour-mois_column": True}))
    assert not is_numeric_column(_col(is_species_column=True))
    assert not is_numeric_column(_col(is_batch_column=True))  # sequence logic instead
    assert not is_numeric_column(_col())


def test_is_compound_column():
    assert is_compound_column(_col(is_heure_column=True))
    assert is_compound_column(_col(**{"is_jour-mois_column": True}))
    assert not is_compound_column(_col(is_alle_column=True))
    assert not is_compound_column(_col(is_poids_column=True))


def test_column_label_semantic():
    assert column_label(_col(is_batch_column=True)) == "Bague"
    assert column_label(_col(is_poids_column=True)) == "Poids"


def test_column_label_fallback_shows_every_column():
    # Untagged columns must still get a label — the UI shows all columns.
    assert column_label(_col(), index=0) == "Spalte 1"
    assert column_label(_col(), index=8) == "Spalte 9"
    assert column_label(_col()) == "Unbekannt"
