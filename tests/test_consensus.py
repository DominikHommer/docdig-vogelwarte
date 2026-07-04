"""Tests for the multi-model consensus stages.

- `FuzzyMatchingBirdNames` for Espèce (HTR-VT + TrOCR + class_indices.json).
- `NumericConsensus` for Aile / Poids / Heure / Jour-Mois
  (DigitRecognizer + YOLO + TrOCR, majority voting).
"""

import pytest


# ---------------------------------------------------------------------------
# Numeric consensus
# ---------------------------------------------------------------------------


from modules.numeric_consensus import NumericConsensus, _digits_key, _normalise


def test_normalise_strips_units_and_unifies_decimal():
    assert _normalise("12g") == "12"
    assert _normalise("12.5") == "12.5"
    assert _normalise("12,5") == "12.5"
    assert _normalise(" 75 ") == "75"
    assert _normalise("Poids: 13.0g") == "13.0"
    assert _normalise("") is None
    assert _normalise("abc") is None


def test_digits_key_ignores_separators():
    assert _digits_key("14:30") == "1430"
    assert _digits_key("15.8") == "158"
    assert _digits_key("1430") == "1430"
    assert _digits_key("") is None
    assert _digits_key("::") is None


def _numeric_column(*cell_preds, flag="is_alle_column"):
    """Build a numeric column. Each item is a dict of source -> prediction."""
    cells = [{"image": None, "image_raw": None, "skip_ocr": False, "erkannt": "", "score": -1}]
    for preds in cell_preds:
        cells.append(
            {
                "image": None,
                "image_raw": None,
                "skip_ocr": False,
                "erkannt": "",
                "score": -1,
                "predictions": {k: v for k, v in preds.items() if v is not None},
            }
        )
    column = {
        "cells": cells,
        "is_alle_column": False,
        "is_poids_column": False,
        "is_batch_column": False,
        "is_species_column": False,
        "is_sexe_column": False,
        "is_age_column": False,
        "is_jour-mois_column": False,
        "is_heure_column": False,
    }
    column[flag] = True
    return column


def _aile_column(*pairs):
    return _numeric_column(*({"digit": d, "trocr": t} for d, t in pairs))


def _run_numeric_consensus(column):
    pages = [{"columns": [column]}]
    NumericConsensus().process({"trocr": pages}, {})
    return [c["erkannt"] for c in column["cells"][1:]], [c["score"] for c in column["cells"][1:]]


def test_numeric_agreement_scores_100():
    column = _aile_column(("68", "68"))
    erk, scores = _run_numeric_consensus(column)
    assert erk == ["68"]
    assert scores == [100]


def test_numeric_agreement_after_normalisation():
    column = _aile_column(("12.5", "12,5"))
    erk, _ = _run_numeric_consensus(column)
    assert erk == ["12.5"]


def test_numeric_disagreement_prefers_clean_shape():
    column = _aile_column(("68", "6B"))
    erk, scores = _run_numeric_consensus(column)
    assert erk == ["68"]
    assert scores[0] == NumericConsensus.DISAGREE_SCORE


def test_numeric_only_one_source():
    column = _aile_column(("75", None), (None, "12"))
    erk, scores = _run_numeric_consensus(column)
    assert erk == ["75", "12"]
    # First cell -> digit-only (60), second -> trocr-only (50)
    assert scores == [60, 50]


def test_numeric_both_empty_keeps_blank():
    column = _aile_column((None, None))
    erk, scores = _run_numeric_consensus(column)
    assert erk == [""]
    assert scores == [-1]


# --- three-voice majority voting (digit + yolo + trocr) --------------------


def test_three_voices_all_agree_scores_100():
    column = _numeric_column({"digit": "68", "yolo": "68", "trocr": "68"})
    erk, scores = _run_numeric_consensus(column)
    assert erk == ["68"]
    assert scores == [NumericConsensus.ALL_AGREE]


def test_two_of_three_majority_wins():
    column = _numeric_column({"digit": "68", "yolo": "68", "trocr": "63"})
    erk, scores = _run_numeric_consensus(column)
    assert erk == ["68"]
    assert scores == [NumericConsensus.MAJORITY]
    assert column["cells"][1]["alternatives"] == ["63"]


def test_yolo_and_trocr_outvote_digit():
    column = _numeric_column({"digit": "88", "yolo": "68", "trocr": "68"})
    erk, scores = _run_numeric_consensus(column)
    assert erk == ["68"]
    assert scores == [NumericConsensus.MAJORITY]
    assert column["cells"][1]["alternatives"] == ["88"]


def test_three_way_disagreement_keeps_all_alternatives():
    column = _numeric_column({"digit": "68", "yolo": "63", "trocr": "88"})
    erk, scores = _run_numeric_consensus(column)
    assert scores == [NumericConsensus.DISAGREE_SCORE]
    cell = column["cells"][1]
    assert erk[0] in ("68", "63", "88")
    assert len(cell["alternatives"]) == 2


def test_yolo_solo_scores_between_digit_and_trocr():
    column = _numeric_column({"yolo": "42"})
    erk, scores = _run_numeric_consensus(column)
    assert erk == ["42"]
    assert scores == [NumericConsensus.SOLO_SCORES["yolo"]]


# --- compound columns (Heure, Jour/Mois) ------------------------------------


def test_heure_agreement_on_digit_sequence_keeps_separator():
    # TrOCR reads "14:30", YOLO can only see digits "1430" -> same value.
    column = _numeric_column(
        {"yolo": "1430", "trocr": "14:30"}, flag="is_heure_column"
    )
    erk, scores = _run_numeric_consensus(column)
    assert erk == ["14:30"]
    assert scores == [NumericConsensus.ALL_AGREE]


def test_jour_mois_agreement_on_digit_sequence():
    column = _numeric_column(
        {"digit": "158", "trocr": "15.8"}, flag="is_jour-mois_column"
    )
    erk, scores = _run_numeric_consensus(column)
    assert erk == ["15.8"]
    assert scores == [NumericConsensus.ALL_AGREE]


def test_decimal_column_does_not_use_digit_sequence_matching():
    # On Aile "12.5" and "125" are different values — no false agreement.
    column = _numeric_column({"digit": "125", "trocr": "12.5"})
    erk, scores = _run_numeric_consensus(column)
    assert scores == [NumericConsensus.DISAGREE_SCORE]


# ---------------------------------------------------------------------------
# Species fuzzy consensus
# ---------------------------------------------------------------------------


from modules.fuzzy_matching import FuzzyMatchingBirdNames


@pytest.fixture
def fuzzy(tmp_path):
    """A FuzzyMatchingBirdNames with a small, controllable class list."""
    import json

    labels = ["Buchfink", "Bergfink", "Erlenzeisig", "Zitronenzeisig"]
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(labels))
    return FuzzyMatchingBirdNames(class_label_path=str(p), score_threshold=50)


def _species_cell(htr=None, trocr=None):
    preds = {}
    if htr is not None:
        preds["htr_vt"] = htr
    if trocr is not None:
        preds["trocr"] = trocr
    return {
        "image": None,
        "image_raw": None,
        "skip_ocr": False,
        "erkannt": "",
        "score": -1,
        "predictions": preds,
    }


def _species_page(*cell_specs):
    cells = [_species_cell()] + [_species_cell(**spec) for spec in cell_specs]
    return [
        {
            "columns": [
                {
                    "cells": cells,
                    "is_species_column": True,
                    "is_batch_column": False,
                    "is_sexe_column": False,
                    "is_age_column": False,
                    "is_jour-mois_column": False,
                    "is_heure_column": False,
                    "is_alle_column": False,
                    "is_poids_column": False,
                }
            ]
        }
    ]


def test_species_both_agree_high_score(fuzzy):
    pages = _species_page({"htr": "Buchfink", "trocr": "Buchfink"})
    fuzzy.process({"trocr": pages}, {})
    cell = pages[0]["columns"][0]["cells"][1]
    assert cell["erkannt"] == "Buchfink"
    assert cell["score"] >= 90
    assert cell["consensus"] == "both_agree"
    assert "alternatives" not in cell or not cell["alternatives"]


def test_species_disagreement_keeps_alternative(fuzzy):
    pages = _species_page({"htr": "Buchfink", "trocr": "Bergfink"})
    fuzzy.process({"trocr": pages}, {})
    cell = pages[0]["columns"][0]["cells"][1]
    assert cell["erkannt"] in ("Buchfink", "Bergfink")
    # The "loser" prediction is preserved as alternative.
    assert cell.get("alternatives")
    assert cell["consensus"] in ("htr_preferred", "trocr_preferred")


def test_species_typo_corrected_to_class_list(fuzzy):
    """Both models off but matching the same target -> still gets it right."""
    pages = _species_page({"htr": "Buchffnk", "trocr": "Bucfink"})
    fuzzy.process({"trocr": pages}, {})
    cell = pages[0]["columns"][0]["cells"][1]
    assert cell["erkannt"] == "Buchfink"
    assert cell["consensus"] == "both_agree"


def test_species_only_htr(fuzzy):
    pages = _species_page({"htr": "Erlenzeisig"})
    fuzzy.process({"trocr": pages}, {})
    cell = pages[0]["columns"][0]["cells"][1]
    assert cell["erkannt"] == "Erlenzeisig"
    assert cell["consensus"] == "htr_only"


def test_species_ditto_mark_left_alone(fuzzy):
    pages = _species_page({"htr": "Buchfink", "trocr": "Buchfink"})
    cell = pages[0]["columns"][0]["cells"][1]
    cell["erkannt"] = '"'
    cell["skip_ocr"] = True
    fuzzy.process({"trocr": pages}, {})
    assert cell["erkannt"] == '"', "Ditto mark should be preserved"
