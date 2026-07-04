"""Single source of truth for the semantic column flags set by DetectColumns.

Every stage that dispatches on column type (DigitRecognizer, TrOCR,
NumericConsensus, the Streamlit UI) imports from here so that adding a new
column type is a one-file change.
"""

# Flag -> human-readable label, in the order the columns appear on the
# Vogelwarte "Liste de baguement" forms (see files-3/meta.json:ui_column_order).
COLUMN_FLAG_TO_LABEL = {
    "is_batch_column": "Bague",
    "is_species_column": "Espèce",
    "is_sexe_column": "Sexe",
    "is_age_column": "Age",
    "is_jour-mois_column": "Jour/Mois",
    "is_heure_column": "Heure",
    "is_alle_column": "Aile",
    "is_poids_column": "Poids",
}

ALL_COLUMN_FLAGS = tuple(COLUMN_FLAG_TO_LABEL.keys())

# Columns whose cells contain plain decimal numbers (one value per cell).
DECIMAL_COLUMN_FLAGS = (
    "is_alle_column",
    "is_poids_column",
)

# Columns whose cells contain digit groups with a separator ("15.8", "14:30").
# Consensus compares these on their digit sequence, not the exact string.
COMPOUND_COLUMN_FLAGS = (
    "is_heure_column",
    "is_jour-mois_column",
)

# Every column where the content is (mostly) digits and the digit recognizers
# should run. The bague column is handled separately because of its
# sequence structure (see DigitRecognizer._process_batch_column).
NUMERIC_COLUMN_FLAGS = DECIMAL_COLUMN_FLAGS + COMPOUND_COLUMN_FLAGS


def is_numeric_column(column: dict) -> bool:
    return any(column.get(flag, False) for flag in NUMERIC_COLUMN_FLAGS)


def is_compound_column(column: dict) -> bool:
    return any(column.get(flag, False) for flag in COMPOUND_COLUMN_FLAGS)


def column_label(column: dict, index: int = None) -> str:
    """Human-readable label for a column.

    Flagged columns get their semantic name; unknown columns fall back to
    "Spalte N" (1-based) so the UI can still show them.
    """
    for flag, label in COLUMN_FLAG_TO_LABEL.items():
        if column.get(flag, False):
            return label
    if index is None:
        return "Unbekannt"
    return f"Spalte {index + 1}"
