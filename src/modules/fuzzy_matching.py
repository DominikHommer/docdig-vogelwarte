from .module_base import Module
from typing import List, Dict
import os
import json
from rapidfuzz import process, fuzz
from rapidfuzz import utils as fuzz_utils


_OCR_INPUT_KEYS = (
    "trocr",
    "htr-vt-recognizer",
    "sexe-classifier",
    "digit-recognizer",
    "predictor",
    "predictor-dummy",
    "quotationmark-detector",
    "cell-formatter",
)


def _load_class_labels(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            # support {label: idx} (sorted by idx) or {"classes": [...]}
            if "classes" in data and isinstance(data["classes"], list):
                return [str(x) for x in data["classes"]]
            try:
                ordered = sorted(data.items(), key=lambda kv: kv[1])
                return [str(k) for k, _ in ordered]
            except Exception:
                return [str(k) for k in data.keys()]
    except Exception as e:
        print(f"[FuzzyMatching] Could not load labels from {path}: {e}")
    return []


def _skip_fuzzy(cell: Dict) -> bool:
    text = (cell.get("erkannt") or "").strip()
    # Ditto / quotation marker — keep verbatim so downstream consumers can
    # back-fill from the preceding row.
    if text == '"':
        return True
    # Skip only if BOTH the provisional value AND the per-source predictions
    # are empty. Otherwise the consensus stage still has something to work
    # with even when no upstream model wrote to `erkannt` yet.
    if not text and not cell.get("predictions"):
        return True
    return False


def _resolve_input(data: dict) -> List[Dict]:
    for key in _OCR_INPUT_KEYS:
        value = data.get(key)
        if value is not None:
            if isinstance(value, dict):
                return [value]
            if isinstance(value, list):
                return value
    raise ValueError("No valid input found for fuzzy matching.")


def _load_species_catalog(path: str) -> List[str]:
    """Read both the canonical file *and* the user-extensions file."""
    try:
        # Optional dependency on the catalog helpers — keep it lazy so older
        # call sites that pass a single path keep working.
        from libs.species_catalog import load_catalog as _load_extended

        from pathlib import Path

        return _load_extended(path=Path(path))
    except Exception:
        return _load_class_labels(path)


class FuzzyMatchingBirdNames(Module):
    """Dual-source consensus + fuzzy correction for the Espèce column.

    Reads both `cell["predictions"]["htr_vt"]` (Felix' Vision Transformer) and
    `cell["predictions"]["trocr"]` (Microsoft's generic OCR), fuzzy-matches each
    against the labelled species list, then picks a consensus:

    - Both models agree on the same species   -> score = min(100, avg + 15)
    - One has a much higher fuzzy match score -> use that, score = score - 10
    - Both fail to match anything plausible   -> leave blank with score 20

    The "loser" prediction is preserved in `cell["alternatives"]` so the UI
    can show it to the user as a second option.
    """

    AGREEMENT_BONUS = 15
    DISAGREEMENT_PENALTY = 10
    MIN_SOLO_FLOOR = 30

    # Below this fuzzy score the raw OCR text is kept (red, for review)
    # instead of FORCING the nearest catalog species — measured on page 5 of
    # the corpus: species missing from the catalog (Waldlaubsänger,
    # Berglaubsänger) were silently snapped to wrong species at ~50-60.
    DEFAULT_SCORE_THRESHOLD = 66

    def __init__(
        self,
        class_label_path: str = "./config/class_indices.json",
        score_threshold: int = DEFAULT_SCORE_THRESHOLD,
    ):
        super().__init__("fuzzy-corrector-Species")
        self.class_label_path = class_label_path
        self.class_labels = _load_species_catalog(class_label_path)
        self.score_threshold = score_threshold

    def get_preconditions(self) -> List[str]:
        return list(_OCR_INPUT_KEYS)

    def reload_catalog(self) -> int:
        """Pull in any user-added entries from class_indices_custom.json.

        Called by the UI right after the user has added a new species; the
        next pipeline run picks them up automatically too because we re-read
        on instantiation.
        """
        self.class_labels = _load_species_catalog(self.class_label_path)
        return len(self.class_labels)

    def _fuzzy(self, text: str):
        text = (text or "").strip()
        if not text or not self.class_labels:
            return "", 0
        try:
            best, score, _ = process.extractOne(
                text, self.class_labels, scorer=fuzz.token_sort_ratio
            )
            if score < self.score_threshold:
                return "", int(score)
            return best, int(score)
        except Exception:
            return "", 0

    def _consensus(self, htr_text: str, trocr_text: str):
        """Return (erkannt, score, alternatives_list, consensus_label)."""
        htr_match, htr_score = self._fuzzy(htr_text)
        trocr_match, trocr_score = self._fuzzy(trocr_text)

        if htr_match and htr_match == trocr_match:
            avg = (htr_score + trocr_score) / 2
            return htr_match, min(100, int(avg + self.AGREEMENT_BONUS)), [], "both_agree"

        if htr_match and trocr_match:
            # Both produced a match, but they disagree.
            if htr_score >= trocr_score:
                return (
                    htr_match,
                    max(int(htr_score - self.DISAGREEMENT_PENALTY), self.MIN_SOLO_FLOOR),
                    [trocr_match],
                    "htr_preferred",
                )
            return (
                trocr_match,
                max(int(trocr_score - self.DISAGREEMENT_PENALTY), self.MIN_SOLO_FLOOR),
                [htr_match],
                "trocr_preferred",
            )

        if htr_match:
            return htr_match, max(int(htr_score - 5), self.MIN_SOLO_FLOOR), [], "htr_only"

        if trocr_match:
            return trocr_match, max(int(trocr_score - 5), self.MIN_SOLO_FLOOR), [], "trocr_only"

        # No match anywhere — keep whatever raw OCR text we have so the user
        # has something to start from.
        raw = (htr_text or trocr_text or "").strip()
        return raw, 20 if raw else -1, [], "no_match"

    def process(self, data: dict, config: dict) -> List[Dict]:
        pages = _resolve_input(data)

        for page in pages:
            for column in page.get("columns", []):
                if not column.get("is_species_column", False):
                    continue

                for cell_idx, cell in enumerate(column.get("cells", [])):
                    if cell_idx == 0:
                        continue
                    if _skip_fuzzy(cell):
                        continue

                    predictions = cell.get("predictions", {})
                    htr_text = predictions.get("htr_vt", "")
                    trocr_text = predictions.get("trocr", "")
                    if not (htr_text or trocr_text):
                        # No model managed to read this cell — leave blank.
                        if not cell.get("erkannt"):
                            cell["score"] = -1
                        continue

                    erkannt, score, alternatives, label = self._consensus(htr_text, trocr_text)
                    cell["erkannt"] = erkannt
                    cell["score"] = score
                    cell["consensus"] = label
                    if alternatives:
                        cell["alternatives"] = alternatives
                    cell["skip_ocr"] = True  # final value — downstream stages leave it alone

        return pages


class FuzzyMatchingAge(Module):
    """Enforce the closed age vocabulary — on these forms ONLY "Fd" and "Fnd"
    exist.

    Source text comes from HTR-VT (`predictions["htr_vt"]`) with TrOCR as a
    fallback. The output is a hard whitelist: a cell in the age column ends
    up as one of ``allowed_labels``, a ditto mark, or empty — never raw OCR
    garbage like "fnsesm".

    Decision rule for the binary Fd/Fnd case — derived from 49 hand-labelled
    real cells (tests/fixtures/cells/age/, tools/compare_age_discriminators):
    TrOCR reads the handwriting as things like "Fd", "Tund", "Fud", "I'd",
    "Ird". The single reliable discriminator is the presence of an 'n' or
    'u' (TrOCR often reads the written n as u) — that rule scores 48/49 on
    the labelled set, while plain fuzzy matching mis-snaps "Fud"->"Fd" and
    HTR-VT text is unusable (14/49).

    So for short reads with letters: 'n'/'u' present -> Fnd, else -> Fd.
    The score reflects certainty: full fuzzy agreement keeps its real score,
    a bare closed-world guess gets 45 (-> red review flag in the UI).
    Long/letterless junk falls back to plain fuzzy matching and ends up
    empty. Output is always in {allowed labels, '"', ""}.
    """

    DEFAULT_ALLOWED = ("Fd", "Fnd")
    WEAK_GUESS_SCORE = 45
    # Noise reads longer than this are not "a misread short code" anymore.
    MAX_GUESS_LEN = 6

    def __init__(
        self,
        allowed_labels=None,
        class_label_path: str = None,
        score_threshold: int = 62,
    ):
        super().__init__("fuzzy-corrector-Age")
        # Priority: explicit labels > explicit label file > user-editable
        # catalog (config/age_options.json via the sidebar) > built-in Fd/Fnd.
        if allowed_labels is not None:
            self.class_labels = list(allowed_labels)
        elif class_label_path:
            loaded = _load_class_labels(class_label_path)
            self.class_labels = loaded or list(self.DEFAULT_ALLOWED)
        else:
            try:
                from libs.value_catalog import recognition_labels

                self.class_labels = recognition_labels("age") or list(self.DEFAULT_ALLOWED)
            except Exception:
                self.class_labels = list(self.DEFAULT_ALLOWED)
        self.score_threshold = score_threshold
        # The closed-world guess only makes sense for the binary Fd/Fnd case.
        self._binary_fd_fnd = set(self.class_labels) == {"Fd", "Fnd"}

    def get_preconditions(self) -> List[str]:
        return list(_OCR_INPUT_KEYS) + ["fuzzy-corrector-Species"]

    def _fuzzy(self, text: str):
        best_match, score, _ = process.extractOne(
            text,
            self.class_labels,
            scorer=fuzz.ratio,
            processor=fuzz_utils.default_process,
        )
        return best_match, int(score)

    def _snap(self, text: str):
        """Return (label, score) — label is '' when nothing plausible matches."""
        text = (text or "").strip()
        if not text or not self.class_labels:
            return "", 0

        # Binary closed world: decide Fd vs Fnd by the n/u evidence — this
        # beats fuzzy matching, which mis-snaps reads like "Fud" to "Fd".
        if self._binary_fd_fnd:
            processed = fuzz_utils.default_process(text)
            has_letters = any(c.isalpha() for c in processed)
            if processed and has_letters and len(processed) <= self.MAX_GUESS_LEN:
                label = "Fnd" if any(c in processed for c in "nu") else "Fd"
                # Score: keep the fuzzy score when it agrees (clean reads like
                # "fd" -> 100, "tund" -> 75); bare guesses stay review-worthy.
                fuzzy_label, fuzzy_score = self._fuzzy(text)
                if fuzzy_label == label and fuzzy_score >= self.score_threshold:
                    return label, fuzzy_score
                return label, self.WEAK_GUESS_SCORE

        # Long / letterless junk (and custom label sets): plain fuzzy.
        best_match, score = self._fuzzy(text)
        if score >= self.score_threshold:
            return best_match, score
        return "", score

    def process(self, data: dict, config: dict) -> List[Dict]:
        pages = _resolve_input(data)
        allowed = set(self.class_labels)

        for page in pages:
            for column in page.get("columns", []):
                if not column.get("is_age_column", False):
                    continue

                for cell_idx, cell in enumerate(column.get("cells", [])):
                    if cell_idx == 0:
                        continue
                    if cell.get("is_blank", False):  # empty cell stays empty
                        continue
                    current = (cell.get("erkannt") or "").strip()
                    if current == '"':  # ditto — resolved downstream
                        continue
                    if cell.get("is_manual_edit", False):
                        continue  # never overwrite the user

                    predictions = cell.get("predictions", {})
                    # TrOCR is the primary age reader (~90% on labelled
                    # cells); HTR-VT only as a last resort.
                    text = (
                        predictions.get("trocr")
                        or predictions.get("htr_vt")
                        or current
                    )

                    if text:
                        label, score = self._snap(text)
                        cell["erkannt"] = label
                        cell["score"] = score if label else -1
                    elif current and current not in allowed:
                        # Closed-vocabulary invariant: whatever slipped in
                        # through another path gets cleared, not shown.
                        cell["erkannt"] = ""
                        cell["score"] = -1

        print("\nFuzzy Matching (Age) finished!\n")
        return pages
