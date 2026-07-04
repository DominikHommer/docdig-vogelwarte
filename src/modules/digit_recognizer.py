"""Robust recognizer for the bague (batch) column.

The bague column has a very predictable structure on the Vogelwarte forms:

- The first data cell holds a handwritten prefix (e.g. "14292") followed by a
  printed suffix digit (e.g. "1"). The cell content is therefore the full ring
  number ("142921").
- Every subsequent cell only shows the next printed suffix digit in sequence —
  "2", "3", ..., "9", "0", "1", ... wrapping around modulo 10. The implied
  full number increases by 1 per row.

We exploit this structure: once the first cell is known, the rest can be
extrapolated arithmetically. Per-cell OCR is still performed as a cross-check
and as a fallback if the prefix cannot be parsed.

Strategies, in order of preference:
1. **Tesseract** (digits-only whitelist) — very good on printed digits, used
   for every cell.
2. **Trained Keras CRNN+CTC** model (optional, `config/digit_model.keras`) —
   used for handwritten parts, mainly in the very first cell.
3. **Trained YOLOv8 digit detector** (optional, `config/digit_yolo.pt`) —
   detects individual handwritten digits and composes them left-to-right;
   third voice next to Tesseract and the CRNN.
4. **Sequence extrapolation** — anchored on the first cell's full number,
   fills the rest and confirms with per-cell OCR where possible.

On the free-standing numeric columns (Aile, Poids, Heure, Jour/Mois) there is
no sequence structure; every backend's raw prediction is stored under
``cell["predictions"]`` and the NumericConsensus stage does majority voting
against TrOCR downstream.

If none of these can produce a number, the cell is left blank so the user can
type it in via the UI.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from libs.bague_sequence import (
    annotate_initial_sequence,
    annotate_per_cell_fallback,
    is_blank_cell,
)
from libs.cell_cropping import strip_cell_borders
from libs.columns import is_numeric_column
from libs.yolo_digits import YoloDigitReader
from .module_base import Module


def _clean_crop(image):
    """Border-stripped version of a cell crop (falls back to the original).

    Vertical slivers of the printed table grid read as "1" in every digit
    backend — strip them before recognition.
    """
    if image is None:
        return None
    cleaned = strip_cell_borders(image)
    return cleaned if cleaned is not None else image


def _to_uint8_grayscale(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.size == 0:
        return None
    arr = np.nan_to_num(arr, nan=255.0, posinf=255.0, neginf=0.0)

    if arr.ndim == 3:
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        if arr.shape[-1] == 3:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif arr.ndim != 2:
        return None

    if arr.dtype != np.uint8:
        mx = float(arr.max()) if arr.size else 1.0
        mn = float(arr.min()) if arr.size else 0.0
        if 0.0 <= mn and mx <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _digits_only(text: str) -> str:
    return "".join(c for c in (text or "") if c.isdigit())


class DigitRecognizer(Module):
    DEFAULT_MODEL_PATHS = [
        "./config/digit_model.keras",
        "./config/digit/digit_model.keras",
    ]

    VALID_BACKENDS = ("tesseract", "keras", "yolo")
    # Default: the two strong voices. On the labelled bague fixtures the CRNN
    # (trained on DIDA handwriting) reads printed digits at ~8% accuracy while
    # Tesseract hits 73% and YOLO 84% — keras stays opt-in via
    # DOCDIG_DIGIT_BACKENDS=tesseract,keras,yolo.
    DEFAULT_BACKENDS = ("tesseract", "yolo")

    def __init__(
        self,
        model_path: Optional[str] = None,
        yolo_model_path: Optional[str] = None,
        img_size=(160, 40),  # (width, height) — matches Digit2.ipynb
        characters: str = "0123456789",
        use_tesseract: bool = True,
        use_yolo: bool = True,
        backends: Optional[Tuple[str, ...]] = None,
        debug: bool = False,
    ):
        """``backends`` selects which OCR voices run, e.g. ``("yolo",)`` for a
        YOLO-only setup (also reachable via the DOCDIG_DIGIT_BACKENDS env var
        in the Streamlit app). Default: Tesseract + YOLO."""
        super().__init__("digit-recognizer")
        self.model_path = model_path
        self.img_width, self.img_height = img_size
        self.characters = characters

        if backends is None:
            backends = self.DEFAULT_BACKENDS
        backends = tuple(b.strip().lower() for b in backends if b.strip())
        unknown = set(backends) - set(self.VALID_BACKENDS)
        if unknown:
            raise ValueError(
                f"Unknown digit backends {sorted(unknown)} — valid: {self.VALID_BACKENDS}"
            )
        self.backends = backends
        self.use_tesseract = use_tesseract and "tesseract" in backends
        self.use_keras = "keras" in backends
        use_yolo = use_yolo and "yolo" in backends
        self.debug = debug

        self._tf = None
        self._model = None
        self._num_to_char = None
        self._keras_state: Optional[bool] = None  # None=not probed, True/False=resolved
        self._tesseract = None
        self._tesseract_state: Optional[bool] = None
        self._yolo = (
            YoloDigitReader(model_path=yolo_model_path, debug=debug) if use_yolo else None
        )

    def get_preconditions(self) -> List[str]:
        return [
            "cell-formatter",
            "quotationmark-detector",
            "htr-vt-recognizer",
        ]

    # ------------------------------------------------------------------
    # Lazy backend loaders
    # ------------------------------------------------------------------
    def _ensure_tesseract(self) -> bool:
        if self._tesseract_state is not None:
            return self._tesseract_state
        if not self.use_tesseract:
            self._tesseract_state = False
            return False
        try:
            import pytesseract  # type: ignore
            self._tesseract = pytesseract
            self._tesseract_state = True
            return True
        except ImportError as e:
            print(f"[DigitRecognizer] pytesseract unavailable: {e}")
            self._tesseract_state = False
            return False

    def _resolve_keras_path(self) -> Optional[str]:
        if self.model_path and os.path.exists(self.model_path):
            return self.model_path
        for path in self.DEFAULT_MODEL_PATHS:
            if os.path.exists(path):
                return path
        return None

    def _ensure_keras(self) -> bool:
        if self._keras_state is not None:
            return self._keras_state

        model_path = self._resolve_keras_path()
        if not model_path:
            if self.debug:
                print("[DigitRecognizer] No digit_model.keras — relying on Tesseract only.")
            self._keras_state = False
            return False

        try:
            import tensorflow as tf
        except ImportError as e:
            print(f"[DigitRecognizer] tensorflow missing: {e}")
            self._keras_state = False
            return False

        try:
            model = tf.keras.models.load_model(model_path, compile=False)

            # Strip training-only head (CTC loss lambda) if present.
            if isinstance(model.input, list) and len(model.input) > 1:
                image_inputs = [inp for inp in model.input if "image" in inp.name.lower()]
                if not image_inputs:
                    image_inputs = [model.input[0]]
                inference_output = None
                for layer in reversed(model.layers):
                    if hasattr(layer, "activation") and getattr(layer.activation, "__name__", "") == "softmax":
                        inference_output = layer.output
                        break
                if inference_output is None:
                    print("[DigitRecognizer] Could not isolate inference output from training graph.")
                    self._keras_state = False
                    return False
                model = tf.keras.Model(inputs=image_inputs[0], outputs=inference_output)

            char_to_num = tf.keras.layers.StringLookup(
                vocabulary=list(self.characters),
                mask_token=None,
                num_oov_indices=0,
            )
            num_to_char = tf.keras.layers.StringLookup(
                vocabulary=char_to_num.get_vocabulary(),
                mask_token=None,
                num_oov_indices=0,
                invert=True,
            )

            self._tf = tf
            self._model = model
            self._num_to_char = num_to_char
            self._keras_state = True
            print(f"[DigitRecognizer] Loaded digit model from {model_path}.")
            return True
        except Exception as e:
            print(f"[DigitRecognizer] Failed to load digit model: {e}")
            self._keras_state = False
            return False

    # ------------------------------------------------------------------
    # OCR primitives
    # ------------------------------------------------------------------
    def _tesseract_digits(self, image, psm: int) -> str:
        if not self._ensure_tesseract():
            return ""
        arr = _to_uint8_grayscale(image)
        if arr is None:
            return ""
        # Cell crops on Vogelwarte forms have a thin table border at the top
        # and bottom and a tiny digit floating in the centre. Tesseract works
        # much better when we strip those borders away and upscale the result.
        prepared = self._preprocess_for_digits(arr)
        try:
            config = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
            text = self._tesseract.image_to_string(prepared, config=config).strip()
            return _digits_only(text)
        except Exception as e:
            if self.debug:
                print(f"[DigitRecognizer] tesseract psm={psm} error: {e}")
            return ""

    @staticmethod
    def _preprocess_for_digits(arr: np.ndarray) -> np.ndarray:
        """Crop vertical borders, pad with whitespace, upscale.

        This is the empirically-tuned recipe that gets Tesseract from ~35 %
        accuracy to ~60 % on the Vogelwarte suffix cells. Bigger gains need a
        trained digit model.
        """
        h, w = arr.shape[:2]
        if h < 12 or w < 12:
            return arr
        pad_y = max(3, h // 8)
        cropped = arr[pad_y : h - pad_y, :]
        padded = cv2.copyMakeBorder(cropped, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        scaled = cv2.resize(padded, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        return scaled

    def _yolo_predict(self, image) -> str:
        if self._yolo is None:
            return ""
        text, _ = self._yolo.read(image)
        return _digits_only(text)

    def _keras_predict(self, image) -> str:
        if not self.use_keras:
            return ""
        if not self._ensure_keras():
            return ""
        arr = _to_uint8_grayscale(image)
        if arr is None:
            return ""
        try:
            resized = cv2.resize(arr, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
            batched = np.expand_dims(np.expand_dims(resized.astype(np.float32) / 255.0, axis=-1), axis=0)
            preds = self._model.predict(batched, verbose=0)

            input_len = np.ones(preds.shape[0]) * preds.shape[1]
            decoded = self._tf.keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0]
            res = decoded[0]
            res = self._tf.gather(res, self._tf.where(res != -1))[:, 0]
            text = self._tf.strings.reduce_join(self._num_to_char(res)).numpy().decode("utf-8", errors="ignore")
            return _digits_only(text)
        except Exception as e:
            if self.debug:
                print(f"[DigitRecognizer] keras predict error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Cell-level recognition
    # ------------------------------------------------------------------
    @staticmethod
    def _plausibility(text: str) -> int:
        """How plausible is this as a full ring number? 4–7 digits is ideal."""
        n = len(text)
        if 4 <= n <= 7:
            return 2
        if 2 <= n <= 8:
            return 1
        return 0

    @classmethod
    def _pick_most_plausible(cls, *candidates: str) -> str:
        """Best non-empty candidate; ties go to the earlier backend."""
        candidates = [c for c in candidates if c]
        if not candidates:
            return ""
        return max(candidates, key=cls._plausibility)

    def _read_full_number(self, image) -> str:
        """First data cell: handwritten prefix + printed suffix. Goal: 4–7 digits."""
        # Tesseract (psm 7 = single line), CRNN and YOLO all get a vote.
        return self._pick_most_plausible(
            self._tesseract_digits(image, psm=7),
            self._keras_predict(image),
            self._yolo_predict(image),
        )

    def _find_anchor(self, per_cell_full: List[str]) -> Tuple[Optional[int], str]:
        """Pick the cell most likely to hold the full ring number.

        The user has confirmed that the printed prefix can sit *anywhere* in
        the bague column — not always row 1. We pick the cell whose OCR text
        gives the longest plausible (4–7 digit) number; ties go to the
        earliest row so downstream sequence math stays simple.
        """
        best_idx: Optional[int] = None
        best_text = ""
        best_score = -1
        for idx, text in enumerate(per_cell_full):
            digits = "".join(c for c in (text or "") if c.isdigit())
            n = len(digits)
            # Plausibility scoring: 4–7 digits is the sweet spot for ring
            # numbers. Shorter reads are NOT anchor material — 2-3 digit
            # strings routinely come from neighbouring-column bleed (e.g.
            # YOLO reading "451" on a suffix cell) and a wrong anchor poisons
            # the entire column. Better no anchor (-> header parse / per-cell
            # fallback) than a wrong one.
            if 4 <= n <= 7:
                score = 100 - abs(5 - n)  # prefer ~5 digit prefixes
            else:
                continue
            if score > best_score:
                best_score = score
                best_idx = idx
                best_text = digits
        return best_idx, best_text

    def _read_header_for_first_number(self, image) -> str:
        """The header cell on Vogelwarte forms often contains BOTH the
        "Bague No" label *and* the first ring number underneath
        (e.g. "14470 ¹"). When that happens the first *data* cell only shows
        the suffix of the second ring number ("2"), and naive anchoring would
        produce a single-digit start. So we also OCR the header and look for
        any digit run of length ≥ 4 — that is the first full ring number.

        Returns "" when nothing usable is found.
        """
        if image is None:
            return ""
        # psm 6 = uniform block of text, picks up text that has multiple lines.
        candidates = []
        for psm in (6, 11, 7):
            text = self._tesseract_digits(image, psm=psm)
            if not text:
                continue
            # Pick the longest digit run >= 4 chars.
            run = ""
            best = ""
            for c in text + " ":
                if c.isdigit():
                    run += c
                else:
                    if len(run) > len(best):
                        best = run
                    run = ""
            if len(best) >= 4:
                candidates.append(best)
        if not candidates:
            return ""
        # Prefer the longest plausible candidate.
        candidates.sort(key=lambda s: (4 <= len(s) <= 7, len(s)), reverse=True)
        return candidates[0]

    def _read_single_digit(self, image) -> str:
        """Subsequent cells: should be a single printed digit."""
        # Try multiple PSMs — different cells respond to different layouts
        # after preprocessing.
        for psm in (10, 13, 8, 7):
            digits = self._tesseract_digits(image, psm=psm)
            if digits:
                return digits[-1]
        keras_text = self._keras_predict(image)
        return keras_text[-1] if keras_text else ""

    # ------------------------------------------------------------------
    # Sequence reconciliation
    # ------------------------------------------------------------------
    def _build_sequence(
        self,
        first_full: str,
        per_cell_digits: List[Optional[str]],
    ) -> Tuple[Optional[List[str]], dict]:
        """
        Given the first cell's full number and per-cell single-digit OCR for
        every subsequent data cell, build an aligned sequence of full numbers.

        Returns:
            (sequence, stats) where sequence is a list of strings (one per
            non-header cell, indexed from 0 = first data cell) or None if no
            anchor could be established.
        """
        stats = {"verified": 0, "mismatches": 0, "extrapolated": 0}

        try:
            start = int(first_full)
        except (TypeError, ValueError):
            return None, stats

        if start <= 0:
            return None, stats

        sequence: List[str] = []
        for offset, observed in enumerate(per_cell_digits):
            expected_num = start + offset
            expected_str = str(expected_num)
            sequence.append(expected_str)

            if offset == 0:
                # First data cell — anchor itself; we already trust it.
                stats["verified"] += 1
                continue

            if observed and observed.isdigit() and observed[-1] == expected_str[-1]:
                stats["verified"] += 1
            elif observed:
                stats["mismatches"] += 1
            else:
                stats["extrapolated"] += 1

        return sequence, stats

    def _try_anchor_from_sequence(self, per_cell_digits: List[Optional[str]]) -> Optional[str]:
        """
        Fallback: if `_read_full_number` failed but `per_cell_digits` exposes a
        clean arithmetic sequence "1,2,3,...", we can infer the suffix of cell-1
        but not the prefix. Returns None unless we can fully reconstruct the
        number (which we can't without the prefix).

        Kept as a hook for future enhancement (e.g. user-provided prefix).
        """
        return None

    # ------------------------------------------------------------------
    # Pipeline entry
    # ------------------------------------------------------------------
    def _resolve_pages(self, data: dict):
        for key in self.get_preconditions():
            value = data.get(key)
            if value is not None:
                return value
        return None

    def process(self, data: dict, config: dict) -> List[dict]:
        pages = self._resolve_pages(data)
        if pages is None:
            return []
        if isinstance(pages, dict):
            pages = [pages]
        if not isinstance(pages, list):
            return []

        # Warm up Tesseract / Keras now so the user gets a single clear message
        # rather than scattered noise from each cell.
        self._ensure_tesseract()
        # Don't force-load Keras unless we hit a cell.

        for page_idx, page in enumerate(pages):
            for col_idx, column in enumerate(page.get("columns", [])):
                cells = column.get("cells", [])
                if not cells:
                    continue

                if column.get("is_batch_column", False):
                    self._process_batch_column(cells, page_idx=page_idx, col_idx=col_idx)
                elif is_numeric_column(column):
                    self._process_numeric_column(cells, page_idx=page_idx, col_idx=col_idx)

        print("\nDigit recognizer finished!\n")
        return pages

    def _process_numeric_column(self, cells: List[dict], page_idx: int, col_idx: int):
        """Aile / Poids / Heure / Jour-Mois: independent handwritten numbers.

        No sequence reasoning here — read every non-blank cell with every
        available backend and store each raw prediction separately
        (``predictions["digit"]`` = Tesseract/CRNN, ``predictions["yolo"]`` =
        YOLO detector). The downstream NumericConsensus stage does majority
        voting together with TrOCR's prediction.
        """
        for c_idx, cell in enumerate(cells):
            if c_idx == 0:  # header
                continue
            if cell.get("is_blank", False) or cell.get("skip_ocr", False):
                continue
            source = cell.get("image_raw")
            if source is None:
                source = cell.get("image")
            source = _clean_crop(source)
            # Borders already stripped -> small margins, else edge content
            # (like the printed suffix digit) would be cropped out.
            if source is None or is_blank_cell(source, margin_y=3, margin_x=3):
                continue

            # One inference per backend; YOLO stays a separate voice so the
            # consensus stage can vote instead of us pre-merging here.
            text = self._pick_most_plausible(
                self._tesseract_digits(source, psm=7),
                self._keras_predict(source),
            )
            yolo_text = self._yolo_predict(source)
            if not (text or yolo_text):
                continue

            predictions = cell.setdefault("predictions", {})
            if text:
                predictions["digit"] = text
            if yolo_text:
                predictions["yolo"] = yolo_text
            if not (cell.get("erkannt") or "").strip():
                cell["erkannt"] = text or yolo_text
                cell["score"] = 70

            if self.debug:
                print(
                    f"[DigitRecognizer] page={page_idx} col={col_idx} row={c_idx} "
                    f"numeric -> digit={text!r} yolo={yolo_text!r}"
                )

    def _process_batch_column(self, cells: List[dict], page_idx: int, col_idx: int):
        data_cells = []  # list[(global_index, cell)]
        for c_idx, cell in enumerate(cells):
            if c_idx == 0:
                continue
            data_cells.append((c_idx, cell))

        if not data_cells:
            return

        # 1. Per-cell OCR on every data cell. We also probe each cell for
        # "blank-ness" so we can leave empty rows empty instead of
        # extrapolating bogus numbers into the dead space at the top or
        # bottom of partially-filled forms.
        per_cell_full: List[str] = []
        per_cell_digits: List[Optional[str]] = []
        blank_mask: List[bool] = []
        for gidx, cell in data_cells:
            source = cell.get("image_raw")
            if source is None:
                source = cell.get("image")
            source = _clean_crop(source)
            blank = is_blank_cell(source, margin_y=3, margin_x=3)
            blank_mask.append(blank)
            if blank or cell.get("skip_ocr", False) or cell.get("erkannt"):
                per_cell_full.append("")
                per_cell_digits.append(None)
                continue
            per_cell_full.append(self._read_full_number(source))
            per_cell_digits.append(self._read_single_digit(source) or None)

        # 2. Locate the anchor: the cell whose multi-digit OCR returned the
        # longest plausible ring-number candidate (≥4 digits). Falls back to
        # parsing the header cell when no data cell yielded one (this covers
        # the historical "first ring number cropped into the header" layout).
        anchor_data_idx, anchor_full = self._find_anchor(per_cell_full)

        if anchor_data_idx is None:
            header_img = cells[0].get("image_raw")
            if header_img is None:
                header_img = cells[0].get("image")
            header_img = _clean_crop(header_img)
            header_full = self._read_header_for_first_number(header_img)
            if header_full and len(header_full) >= 4:
                try:
                    anchor_full = str(int(header_full) + 1)
                    anchor_data_idx = 0  # header value belongs *before* the first data cell
                    if self.debug:
                        print(
                            f"[DigitRecognizer] page={page_idx} col={col_idx}: "
                            f"anchor sourced from header ({header_full!r}) -> "
                            f"first data cell expected to be {anchor_full!r}"
                        )
                except ValueError:
                    pass

        if self.debug and anchor_data_idx is not None:
            print(
                f"[DigitRecognizer] page={page_idx} col={col_idx}: "
                f"anchor at data_idx={anchor_data_idx} value={anchor_full!r}"
            )

        # 3. Build the sequence over non-blank cells only.
        non_blank_indices = [i for i, blank in enumerate(blank_mask) if not blank]
        sequence: Optional[List[str]] = None
        stats = {}
        anchor_rank = 0
        if anchor_full and anchor_full.isdigit() and anchor_data_idx is not None:
            try:
                anchor_int = int(anchor_full)
                # rank == position of the anchor among non-blank cells
                if anchor_data_idx in non_blank_indices:
                    anchor_rank = non_blank_indices.index(anchor_data_idx)
                start = anchor_int - anchor_rank
                if start > 0:
                    sequence, stats = self._build_sequence(
                        first_full=str(start),
                        per_cell_digits=[per_cell_digits[i] for i in non_blank_indices],
                    )
            except ValueError:
                pass

        # Locate the parent column dict so we can use the annotation helpers.
        # We have `cells` but the annotators take the column. Reconstruct.
        column = {"cells": cells}

        if sequence is None:
            if self.debug:
                print(
                    f"[DigitRecognizer] page={page_idx} col={col_idx}: "
                    f"no anchor; using raw per-cell OCR."
                )
            annotate_per_cell_fallback(column, per_cell_digits)
            return

        annotate_initial_sequence(
            column,
            sequence,
            per_cell_digits,
            blank_mask=blank_mask,
            anchor_data_idx=anchor_data_idx if anchor_data_idx is not None else 0,
        )

        if self.debug:
            print(
                f"[DigitRecognizer] page={page_idx} col={col_idx}: "
                f"anchor={sequence[0]} cells={len(sequence)} stats={stats}"
            )
