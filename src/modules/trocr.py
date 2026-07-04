from .module_base import Module
import numpy as np
from typing import List, Dict, Iterable, Optional

from PIL import Image
import os
import cv2

from libs.cell_cropping import strip_cell_borders
from libs.columns import NUMERIC_COLUMN_FLAGS, is_numeric_column

# TrOCR runs on these column types:
#   - is_species_column: as a second opinion next to HTR-VT (fuzzy matching
#     reconciles both predictions into a consensus value).
#   - is_age_column: PRIMARY reader. Measured on 49 labelled cells: TrOCR
#     reads the Fd/Fnd handwriting at ~90% while HTR-VT (trained on bird
#     names) produces noise (14/49) — FuzzyMatchingAge snaps the output.
#   - every numeric column (Aile, Poids, Heure, Jour/Mois): cross-check
#     against DigitRecognizer + YOLO via NumericConsensus.
_TRO_CR_TARGET_FLAGS = ("is_species_column", "is_age_column") + NUMERIC_COLUMN_FLAGS


class TrOCR(Module):
    """TrOCR runs on the Espèce column plus all numeric columns by default.

    Loads the HuggingFace model lazily so that an unreachable model hub or a
    missing checkpoint does not bring down the entire pipeline. If loading
    fails the module becomes a no-op — cells stay blank for the user to fill
    in via the UI.

    Pass ``target_column_flags=("is_alle_column", "is_poids_column", ...)`` to
    extend the set, or ``target_column_flags=()`` to run on every column that
    no specialist claimed (legacy behaviour).
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-stage1",
        output_path: str = "data/output/trocr_output.xlsx",
        target_column_flags: Iterable[str] = _TRO_CR_TARGET_FLAGS,
        debug: bool = False,
        debug_folder: str = "debug/debug_trocr/",
    ):
        super().__init__("trocr")
        self.debug = debug
        self.debug_folder = debug_folder
        self.target_column_flags = tuple(target_column_flags)
        self.model_name = model_name
        self.output_path = output_path
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

        self.processor = None
        self.model = None
        self.device = None
        self._torch = None
        self._available: Optional[bool] = None

    def _ensure_loaded(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as e:
            print(f"[TrOCR] dependencies missing: {e}")
            self._available = False
            return False

        try:
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self._torch = torch
            self._available = True
            print(f"[TrOCR] Loaded {self.model_name} (device={self.device}).")
            return True
        except Exception as e:
            print(f"[TrOCR] Failed to load {self.model_name}: {e}")
            self._available = False
            return False

    def get_preconditions(self) -> List[str]:
        # Accept any upstream that produced cell dicts. Mutations happen in place,
        # so reading whichever key is present works equivalently.
        return [
            "sexe-classifier",
            "digit-recognizer",
            "htr-vt-recognizer",
            "quotationmark-detector",
            "cell-formatter",
        ]

    def _to_rgb(self, image) -> Optional[np.ndarray]:
        if image is None:
            return None
        img = np.asarray(image)
        if img.size == 0:
            return None

        img = np.nan_to_num(img, nan=255.0, posinf=255.0, neginf=0.0)

        if img.ndim == 2:
            if img.dtype != np.uint8:
                mx = float(img.max()) if img.size else 1.0
                mn = float(img.min()) if img.size else 0.0
                if 0.0 <= mn and mx <= 1.0:
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if img.ndim == 3:
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            if img.dtype != np.uint8:
                mx = float(img.max()) if img.size else 1.0
                mn = float(img.min()) if img.size else 0.0
                if 0.0 <= mn and mx <= 1.0:
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            return img

        return None

    def _recognise(self, image) -> Optional[str]:
        if not self._ensure_loaded():
            return None
        rgb = self._to_rgb(image)
        if rgb is None:
            return None
        try:
            pil_image = Image.fromarray(rgb).convert("RGB")
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                generated_ids = self.model.generate(**inputs)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text
        except Exception as e:
            if self.debug:
                print(f"[TrOCR] inference error: {e}")
            return None

    def process(self, data: dict, config: dict) -> List[Dict]:
        valid_keys = self.get_preconditions()
        input_key = next((k for k in valid_keys if data.get(k) is not None), None)
        if input_key is None:
            raise ValueError("No valid input found for TrOCR.")

        pages = data[input_key]
        if isinstance(pages, dict):
            pages = [pages]
        if not isinstance(pages, list):
            return []

        for page_idx, page in enumerate(pages):
            columns = page.get("columns", [])
            if not isinstance(columns, list):
                continue

            for col_idx, column in enumerate(columns):
                cells = column.get("cells", [])

                if self.debug and cells and isinstance(cells[0], dict):
                    first_img = cells[0].get("image")
                    if first_img is not None:
                        debug_img = self._to_rgb(first_img)
                        if debug_img is not None:
                            debug_path = os.path.join(
                                self.debug_folder, f"page_{page_idx}_col_{col_idx}.png"
                            )
                            cv2.imwrite(debug_path, debug_img)

                # If a target list is set, the column must match it; otherwise
                # run on every column (legacy behaviour).
                if self.target_column_flags and not any(
                    column.get(flag, False) for flag in self.target_column_flags
                ):
                    # A specialist recognizer is responsible. Cells it could not
                    # populate (model missing) will be left blank — the user can
                    # still fill them in via the UI.
                    if self.debug:
                        print(f"[TrOCR] skipping column {col_idx} (specialist column)")
                    continue

                for cell_idx, cell in enumerate(cells):
                    if cell_idx == 0:  # header row
                        continue
                    if cell.get("is_blank", False):
                        # Empty cell (CellFormatter blank gate) — TrOCR would
                        # hallucinate text into it, so never run.
                        continue
                    if cell.get("skip_ocr", False):
                        # quote / ditto markers — leave the decision verbatim
                        continue

                    # Use the raw cell crop so TrOCR sees the same data as
                    # the specialist recognizers (dark text on white).
                    source = cell.get("image_raw")
                    if source is None:
                        source = cell.get("image")

                    # Numeric + age columns: strip table-grid slivers first —
                    # a vertical rule at the edge reads as "1"/"I" otherwise.
                    if is_numeric_column(column) or column.get("is_age_column", False):
                        cleaned = strip_cell_borders(source)
                        if cleaned is not None:
                            source = cleaned

                    text = self._recognise(source)
                    if text is None:
                        continue

                    # Write into the predictions dict — downstream stages
                    # (FuzzyMatchingBirdNames, NumericConsensus) take over from
                    # there and decide the final `erkannt`/`score`.
                    cell.setdefault("predictions", {})["trocr"] = text

                    # Provisional erkannt so the cell isn't empty if no
                    # consensus stage runs (e.g. column without a specialist).
                    if not (cell.get("erkannt") or "").strip():
                        cell["erkannt"] = text
                        cell["score"] = 50

                    if self.debug:
                        print(f"[TrOCR] page={page_idx} col={col_idx} row={cell_idx} -> {text!r}")

        print("\nOCR (TrOCR) finished!\n")
        return pages
