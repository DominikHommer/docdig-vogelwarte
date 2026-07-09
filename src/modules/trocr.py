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

        # Batch size for inference — bigger is faster but uses more RAM.
        # DOCDIG_TROCR_BATCH lets a memory-tight host dial it down.
        try:
            self.batch_size = max(1, int(os.environ.get("DOCDIG_TROCR_BATCH", "16")))
        except ValueError:
            self.batch_size = 16

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
            # NOTE: do NOT raise torch threads here — measured, more threads
            # made generate() SLOWER (58s -> 85s/page) due to contention on
            # the small per-token tensors. The default thread count wins.
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
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
        results = self._recognise_batch([image])
        return results[0] if results else None

    def _recognise_batch(self, images: List) -> List[Optional[str]]:
        """Run TrOCR on many crops at once.

        Batching is the single biggest speed win here: generate() has a large
        fixed overhead per call, so 250 single cells/page cost ~66s while the
        same cells in batches of DOCDIG_TROCR_BATCH cost a fraction. Output is
        identical to per-cell calls (same images, same model). Unreadable
        crops keep their slot as None so indices line up with the caller.
        """
        if not self._ensure_loaded():
            return [None] * len(images)

        rgbs = [self._to_rgb(img) for img in images]
        valid = [(i, rgb) for i, rgb in enumerate(rgbs) if rgb is not None]
        out: List[Optional[str]] = [None] * len(images)
        if not valid:
            return out

        for start in range(0, len(valid), self.batch_size):
            chunk = valid[start : start + self.batch_size]
            pil_batch = [Image.fromarray(rgb).convert("RGB") for _, rgb in chunk]
            try:
                inputs = self.processor(images=pil_batch, return_tensors="pt").to(
                    self.device
                )
                with self._torch.no_grad():
                    # Cell contents are short (ring numbers <=7 chars, species
                    # names ~20). Capping new tokens stops generate() from
                    # decoding up to its default length on every cell —
                    # genuinely shorter work, output unchanged for real cells.
                    generated_ids = self.model.generate(
                        **inputs, max_new_tokens=24, num_beams=1
                    )
                    texts = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                for (idx, _), text in zip(chunk, texts):
                    out[idx] = text
            except Exception as e:
                if self.debug:
                    print(f"[TrOCR] batch inference error: {e}")
        return out

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

        # Collect every cell that needs TrOCR across ALL pages first, then run
        # them through the model in batches — one generate() call per batch
        # instead of one per cell.
        work_cells = []   # cell dicts to write back into
        work_images = []  # the matching preprocessed crops
        for page in pages:
            columns = page.get("columns", [])
            if not isinstance(columns, list):
                continue
            for column in columns:
                if self.target_column_flags and not any(
                    column.get(flag, False) for flag in self.target_column_flags
                ):
                    continue  # a specialist recognizer owns this column
                for cell_idx, cell in enumerate(column.get("cells", [])):
                    if cell_idx == 0 or cell.get("is_blank") or cell.get("skip_ocr"):
                        continue
                    source = cell.get("image_raw")
                    if source is None:
                        source = cell.get("image")
                    if is_numeric_column(column) or column.get("is_age_column", False):
                        cleaned = strip_cell_borders(source)
                        if cleaned is not None:
                            source = cleaned
                    work_cells.append(cell)
                    work_images.append(source)

        if work_cells:
            texts = self._recognise_batch(work_images)
            for cell, text in zip(work_cells, texts):
                if text is None:
                    continue
                cell.setdefault("predictions", {})["trocr"] = text
                if not (cell.get("erkannt") or "").strip():
                    cell["erkannt"] = text
                    cell["score"] = 50

        print(f"\nOCR (TrOCR) finished! ({len(work_cells)} Zellen, "
              f"Batchgröße {self.batch_size})\n")
        return pages
