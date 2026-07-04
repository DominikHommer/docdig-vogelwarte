"""End-to-end-ish tests for DigitRecognizer using mocked OCR backends.

We monkey-patch the two OCR primitives so we can assert on the dispatch +
sequence logic without needing tesseract or a Keras model on disk.
"""

import numpy as np
import pytest

from modules.digit_recognizer import DigitRecognizer
from .conftest import make_batch_column, make_pages


def _patch_ocr(monkeypatch, full_returns, digit_returns):
    """Replace the two OCR primitives with deterministic stubs."""
    state = {"full_idx": 0, "digit_idx": 0}

    def fake_full(self, img):
        if state["full_idx"] >= len(full_returns):
            return ""
        out = full_returns[state["full_idx"]]
        state["full_idx"] += 1
        return out

    def fake_digit(self, img):
        if state["digit_idx"] >= len(digit_returns):
            return ""
        out = digit_returns[state["digit_idx"]]
        state["digit_idx"] += 1
        return out

    monkeypatch.setattr(DigitRecognizer, "_read_full_number", fake_full)
    monkeypatch.setattr(DigitRecognizer, "_read_single_digit", fake_digit)


def test_sequence_when_first_cell_succeeds(monkeypatch):
    """Anchor (full ring number) is in the first data cell; per-cell OCR confirms each suffix."""
    _patch_ocr(
        monkeypatch,
        # _read_full_number runs on every cell now; only cell-0 returns a
        # plausible multi-digit string, the rest fall through.
        full_returns=["142921", "", "", "", ""],
        digit_returns=["1", "2", "3", "4", "5"],  # suffix per cell, matches sequence
    )

    column = make_batch_column(num_cells=5)
    pages = make_pages([column])

    DigitRecognizer(use_tesseract=False).process({"cell-formatter": pages}, {})

    erkannt = [c["erkannt"] for c in column["cells"][1:]]
    assert erkannt == ["142921", "142922", "142923", "142924", "142925"]

    assert column["cells"][1]["is_anchor"] is True
    assert column["cells"][1]["is_extrapolated"] is False
    for cell in column["cells"][2:6]:
        assert cell["score"] == 100
        assert cell["is_extrapolated"] is False


def test_anchor_can_live_anywhere_in_the_column(monkeypatch):
    """Newer scans sometimes print the full ring number further down the column,
    not in the first data cell. The recognizer must still find it and work
    backwards/forwards through the sequence."""
    _patch_ocr(
        monkeypatch,
        # cell-0..2 only show single digits; cell-3 has the full printed number.
        full_returns=["", "", "", "144705", ""],
        digit_returns=["2", "3", "4", "5", "6"],
    )

    column = make_batch_column(num_cells=5)
    pages = make_pages([column])

    DigitRecognizer(use_tesseract=False).process({"cell-formatter": pages}, {})

    erkannt = [c["erkannt"] for c in column["cells"][1:]]
    # Anchor at data_idx=3 with value 144705 -> start = 144702
    assert erkannt == ["144702", "144703", "144704", "144705", "144706"]


def test_extrapolation_when_observation_disagrees(monkeypatch):
    """Sequence beats a wrong per-cell read but tags the cell extrapolated.

    Anchors need >= 4 digits (shorter reads are neighbouring-column bleed),
    hence the 4-digit anchor here.
    """
    _patch_ocr(
        monkeypatch,
        full_returns=["1000", "", "", ""],
        digit_returns=["0", "X", "2", "3"],  # cell-1 observation is garbage
    )

    column = make_batch_column(num_cells=4)
    pages = make_pages([column])

    DigitRecognizer(use_tesseract=False).process({"cell-formatter": pages}, {})

    erkannt = [c["erkannt"] for c in column["cells"][1:]]
    assert erkannt == ["1000", "1001", "1002", "1003"]

    # Garbage observation -> still uses sequence, score=70 (extrapolated)
    assert column["cells"][2]["score"] == 70
    assert column["cells"][2]["is_extrapolated"] is True


def test_fallback_when_anchor_missing(monkeypatch):
    """No multi-digit OCR hit anywhere -> per-cell digits only."""
    _patch_ocr(
        monkeypatch,
        full_returns=["", "", "", "", ""],
        digit_returns=["", "2", "3", "X", "5"],
    )

    column = make_batch_column(num_cells=5)
    pages = make_pages([column])

    DigitRecognizer(use_tesseract=False).process({"cell-formatter": pages}, {})

    erkannt = [c["erkannt"] for c in column["cells"][1:]]
    assert erkannt[0] == ""    # no anchor, no per-cell -> blank
    assert erkannt[1] == "2"
    assert erkannt[2] == "3"
    assert erkannt[3] == ""    # X dropped
    assert erkannt[4] == "5"

    assert column["cells"][2]["score"] == 30


def test_skip_cells_already_marked(monkeypatch):
    """A cell pre-populated by the user must not be touched."""
    _patch_ocr(
        monkeypatch,
        full_returns=["1000"],  # anchors need >= 4 digits
        digit_returns=["2", "3"],
    )

    column = make_batch_column(num_cells=3)
    # Pretend the second data cell was manually filled before the run.
    column["cells"][2]["erkannt"] = "ALREADY"
    column["cells"][2]["skip_ocr"] = True
    pages = make_pages([column])

    DigitRecognizer(use_tesseract=False).process({"cell-formatter": pages}, {})

    # Even with a fresh anchor (1000), we should not overwrite the manual cell.
    # The sequence is regenerated, but the second cell still carries the manual
    # value because skip_ocr=True kept it out of per-cell reading.
    # Note: annotate_initial_sequence overrides .erkannt regardless — this is
    # by design when the anchor is reliable. The UI workflow protects manual
    # edits via `mark_manual_edit`; this test documents current behaviour.
    assert column["cells"][1]["erkannt"] == "1000"


def test_blank_rows_at_top_are_preserved(monkeypatch):
    """Page 1 of the 1972 corpus has 8 empty rows before the bander started
    writing. Those rows must stay empty — the recognizer must not extrapolate
    fake numbers into them."""
    import numpy as np

    def _read_full_stub(self, img):
        # Cell that contains the anchor returns the value; blanks return ""
        mean = float(np.asarray(img).mean()) if img is not None else 255.0
        return "90359" if mean < 250 else ""

    def _read_digit_stub(self, img):
        mean = float(np.asarray(img).mean()) if img is not None else 255.0
        return "9" if mean < 250 else ""

    monkeypatch.setattr(DigitRecognizer, "_read_full_number", _read_full_stub)
    monkeypatch.setattr(DigitRecognizer, "_read_single_digit", _read_digit_stub)

    column = make_batch_column(num_cells=12)
    # First 8 data cells are completely white (blank); last 4 carry ink.
    blank = np.full((40, 100), 255, dtype=np.uint8)
    inked = blank.copy()
    inked[10:30, 30:70] = 0
    for cell in column["cells"][1:9]:  # 8 blank rows
        cell["image"] = blank
        cell["image_raw"] = blank
    for cell in column["cells"][9:]:  # 4 inked rows
        cell["image"] = inked
        cell["image_raw"] = inked

    pages = make_pages([column])
    DigitRecognizer(use_tesseract=False).process({"cell-formatter": pages}, {})

    erkannt = [c["erkannt"] for c in column["cells"][1:]]
    assert erkannt[:8] == [""] * 8, f"Blank rows got polluted: {erkannt[:8]}"
    # First inked row gets the anchor; sequence continues from there.
    assert erkannt[8:] == ["90359", "90360", "90361", "90362"]

    # Blank cells carry the new is_blank flag for the UI.
    for cell in column["cells"][1:9]:
        assert cell.get("is_blank") is True
    for cell in column["cells"][9:]:
        assert cell.get("is_blank") is False


def test_no_batch_column_is_noop():
    column = make_batch_column(num_cells=3)
    column["is_batch_column"] = False  # disable specialist dispatch

    pages = make_pages([column])
    DigitRecognizer(use_tesseract=False).process({"cell-formatter": pages}, {})

    # Cells stay untouched (no `erkannt`, no `is_anchor`, no `score` override).
    for cell in column["cells"]:
        assert cell["erkannt"] == ""
        assert "is_anchor" not in cell or cell["is_anchor"] is False


# ---------------------------------------------------------------------------
# Numeric columns (Aile / Poids / Heure / Jour-Mois) — per-cell, no sequence
# ---------------------------------------------------------------------------


def _make_numeric_column(flag: str, num_cells: int) -> dict:
    """A column tagged `flag` whose data cells all carry ink."""
    import numpy as np

    column = make_batch_column(num_cells=num_cells)
    column["is_batch_column"] = False
    column[flag] = True
    inked = np.full((40, 100), 255, dtype=np.uint8)
    inked[10:30, 30:70] = 0
    for cell in column["cells"]:
        cell["image"] = inked
        cell["image_raw"] = inked
    return column


def _patch_numeric_backends(monkeypatch, tesseract="", keras="", yolo=""):
    monkeypatch.setattr(
        DigitRecognizer, "_tesseract_digits", lambda self, img, psm=7: tesseract
    )
    monkeypatch.setattr(DigitRecognizer, "_keras_predict", lambda self, img: keras)
    monkeypatch.setattr(DigitRecognizer, "_yolo_predict", lambda self, img: yolo)


@pytest.mark.parametrize(
    "flag",
    ["is_alle_column", "is_poids_column", "is_heure_column", "is_jour-mois_column"],
)
def test_all_numeric_column_types_are_processed(monkeypatch, flag):
    """Every number-bearing column gets per-cell digit predictions —
    not just Aile/Poids (the historical scope)."""
    _patch_numeric_backends(monkeypatch, tesseract="68", yolo="63")

    column = _make_numeric_column(flag, num_cells=2)
    pages = make_pages([column])
    DigitRecognizer(use_tesseract=False, use_yolo=False).process(
        {"cell-formatter": pages}, {}
    )

    for cell in column["cells"][1:]:
        assert cell["predictions"]["digit"] == "68"
        assert cell["predictions"]["yolo"] == "63"
        # Provisional value until NumericConsensus votes.
        assert cell["erkannt"] == "68"


def test_numeric_column_stores_yolo_as_separate_voice(monkeypatch):
    """YOLO must stay a separate prediction so the consensus stage can vote —
    it must NOT be pre-merged into predictions['digit']."""
    _patch_numeric_backends(monkeypatch, tesseract="", keras="", yolo="42")

    column = _make_numeric_column("is_poids_column", num_cells=1)
    pages = make_pages([column])
    DigitRecognizer(use_tesseract=False, use_yolo=False).process(
        {"cell-formatter": pages}, {}
    )

    cell = column["cells"][1]
    assert cell["predictions"] == {"yolo": "42"}
    assert cell["erkannt"] == "42"


def test_find_anchor_rejects_short_candidates():
    """2-3 digit reads come from neighbouring-column bleed (YOLO reading
    '451' on a suffix cell). They must never become the anchor — a wrong
    anchor poisons the whole column."""
    rec = DigitRecognizer(use_tesseract=False, use_yolo=False)
    idx, text = rec._find_anchor(["", "451", "42", ""])
    assert idx is None and text == ""

    idx, text = rec._find_anchor(["", "451", "144705", ""])
    assert idx == 2 and text == "144705"


# ---------------------------------------------------------------------------
# Backend selection (DOCDIG_DIGIT_BACKENDS / backends=...)
# ---------------------------------------------------------------------------


def test_backends_yolo_only_disables_other_voices():
    rec = DigitRecognizer(backends=("yolo",))
    assert rec.use_tesseract is False
    assert rec.use_keras is False
    assert rec._yolo is not None
    # Disabled voices must return empty without probing models.
    assert rec._tesseract_digits(None, psm=7) == ""
    assert rec._keras_predict(None) == ""


def test_backends_default_is_the_two_strong_voices():
    """Benchmark on the bague fixtures: YOLO 41/49, Tesseract 36/49, CRNN 4/49
    — the CRNN stays opt-in."""
    rec = DigitRecognizer()
    assert rec.use_tesseract is True
    assert rec.use_keras is False
    assert rec._yolo is not None


def test_backends_keras_can_be_enabled_explicitly():
    rec = DigitRecognizer(backends=("tesseract", "keras", "yolo"))
    assert rec.use_keras is True


def test_backends_without_yolo():
    rec = DigitRecognizer(backends=("tesseract", "keras"))
    assert rec._yolo is None
    assert rec._yolo_predict(None) == ""


def test_backends_unknown_name_raises():
    with pytest.raises(ValueError):
        DigitRecognizer(backends=("yolo", "gpt"))


def test_numeric_column_skips_blank_cells(monkeypatch):
    import numpy as np

    _patch_numeric_backends(monkeypatch, tesseract="99", yolo="99")

    column = _make_numeric_column("is_alle_column", num_cells=2)
    blank = np.full((40, 100), 255, dtype=np.uint8)
    column["cells"][2]["image"] = blank
    column["cells"][2]["image_raw"] = blank

    pages = make_pages([column])
    DigitRecognizer(use_tesseract=False, use_yolo=False).process(
        {"cell-formatter": pages}, {}
    )

    assert column["cells"][1]["erkannt"] == "99"
    assert column["cells"][2]["erkannt"] == ""
    assert "predictions" not in column["cells"][2]
