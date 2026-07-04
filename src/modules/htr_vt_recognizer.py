"""HTR-VT recognizer for the species and age columns.

Uses the vendored HTR-VT (`src/vendor/htr_vt`) trained on bird-name line crops.
The model expects the original cell crop (`image_raw`), not the denoised /
inverted version produced by `CellFormatter`.

The age column carries a tiny vocabulary on these forms (essentially "Fd" /
"Fnd"); HTR-VT reads the raw text and `FuzzyMatchingAge` snaps it onto
`config/age_classes.json` downstream.
"""

import json
import os
import re
import sys
from collections import OrderedDict
from typing import List, Optional

import numpy as np

from .module_base import Module


class HtrVtRecognizer(Module):
    """Run HTR-VT on cells flagged as species columns."""

    DEFAULT_MODEL_PATHS = [
        "./config/htr_vt/best_WER.pth",
        "./config/htr_vt/best_CER.pth",
        "./FelixUpload/inferencedata/digC/best_WER.pth",
        "./FelixUpload/inferencedata/digC/best_CER.pth",
    ]

    DEFAULT_ALPHABET_PATHS = [
        "./config/htr_vt/alphabet.json",
    ]

    DEFAULT_TRAIN_LIST_CANDIDATES = [
        ("./config/htr_vt/train.ln", "./config/htr_vt/lines/"),
        ("./FelixUpload/inferencedata/digC/train.ln", "./FelixUpload/inferencedata/digC/lines/"),
    ]

    # Columns this recognizer reads. Species only: HTR-VT is trained on bird
    # names — measured on 49 labelled age cells it emits noise (14/49 via the
    # n-heuristic) while TrOCR reads the same cells at ~90%, so the age
    # column belongs to TrOCR (+ FuzzyMatchingAge).
    DEFAULT_TARGET_FLAGS = ("is_species_column",)

    def __init__(
        self,
        model_path: Optional[str] = None,
        alphabet_path: Optional[str] = None,
        train_list_path: Optional[str] = None,
        train_data_path: Optional[str] = None,
        nb_cls: int = 80,
        img_size=(256, 64),
        device: Optional[str] = None,
        target_column_flags=DEFAULT_TARGET_FLAGS,
        debug: bool = False,
    ):
        super().__init__("htr-vt-recognizer")
        self.model_path = model_path
        self.alphabet_path = alphabet_path
        self.train_list_path = train_list_path
        self.train_data_path = train_data_path
        self.nb_cls = nb_cls
        self.img_size = tuple(img_size)
        self.device_str = device
        self.target_column_flags = tuple(target_column_flags)
        self.debug = debug

        self._model = None
        self._converter = None
        self._torch = None
        self._device = None
        self._transform = None
        self._available = None  # tri-state: None=not probed, True/False=resolved

    def get_preconditions(self) -> List[str]:
        return ["cell-formatter"]

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, explicit: Optional[str], candidates: List[str]) -> Optional[str]:
        if explicit and os.path.exists(explicit):
            return explicit
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _load_alphabet(self) -> Optional[List[str]]:
        alphabet_path = self._resolve_path(self.alphabet_path, self.DEFAULT_ALPHABET_PATHS)
        if alphabet_path:
            try:
                with open(alphabet_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "characters" in data:
                    return list(data["characters"])
                if isinstance(data, list):
                    return list(data)
            except Exception as e:
                print(f"[HtrVtRecognizer] Failed to load alphabet from {alphabet_path}: {e}")

        # Fall back to extracting from the training label list.
        train_list = self.train_list_path
        train_data = self.train_data_path
        if not (train_list and train_data and os.path.exists(train_list) and os.path.exists(train_data)):
            for candidate_list, candidate_data in self.DEFAULT_TRAIN_LIST_CANDIDATES:
                if os.path.exists(candidate_list) and os.path.exists(candidate_data):
                    train_list, train_data = candidate_list, candidate_data
                    break

        if not (train_list and train_data and os.path.exists(train_list) and os.path.exists(train_data)):
            return None

        try:
            with open(train_list, "r", encoding="utf-8") as f:
                stems = [line.strip() for line in f if line.strip()]
            chars: set = set()
            for stem in stems:
                txt = os.path.join(train_data, os.path.splitext(stem)[0] + ".txt")
                if not os.path.exists(txt):
                    continue
                with open(txt, "r", encoding="utf-8") as t:
                    chars.update(" ".join(t.read().split()))
            alphabet = sorted(chars)
            if alphabet:
                # Cache for future loads.
                cache_path = "./config/htr_vt/alphabet.json"
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump({"characters": alphabet}, f, ensure_ascii=False)
                except Exception as e:
                    print(f"[HtrVtRecognizer] Could not cache alphabet to {cache_path}: {e}")
            return alphabet
        except Exception as e:
            print(f"[HtrVtRecognizer] Failed to derive alphabet from {train_list}: {e}")
            return None

    def _ensure_loaded(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            import torch
            from torchvision import transforms
        except ImportError as e:
            print(f"[HtrVtRecognizer] torch/torchvision missing: {e}")
            self._available = False
            return False

        model_path = self._resolve_path(self.model_path, self.DEFAULT_MODEL_PATHS)
        if not model_path:
            print("[HtrVtRecognizer] No HTR-VT checkpoint found. Skipping species OCR.")
            self._available = False
            return False

        alphabet = self._load_alphabet()
        if not alphabet:
            print("[HtrVtRecognizer] Could not load HTR-VT alphabet. Skipping species OCR.")
            self._available = False
            return False

        # Add the vendor path so the relative imports inside HTR-VT resolve.
        vendor_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vendor"))
        if vendor_root not in sys.path:
            sys.path.insert(0, vendor_root)

        try:
            from htr_vt.model import HTR_VT  # type: ignore
            from htr_vt.utils import utils as htr_utils  # type: ignore
        except Exception as e:
            print(f"[HtrVtRecognizer] Could not import vendored HTR-VT: {e}")
            self._available = False
            return False

        try:
            device = torch.device(self.device_str) if self.device_str else torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )

            model = HTR_VT.create_model(nb_cls=self.nb_cls, img_size=list(self.img_size)[::-1])
            ckpt = torch.load(model_path, map_location="cpu")
            state_dict = ckpt.get("state_dict_ema", ckpt)

            cleaned = OrderedDict()
            pattern = re.compile("module.")
            for k, v in state_dict.items():
                cleaned[re.sub(pattern, "", k)] = v
            model.load_state_dict(cleaned, strict=True)
            model.to(device).eval()

            converter = htr_utils.CTCLabelConverter(alphabet)
            transform_fn = transforms.Compose(
                [
                    transforms.Resize((self.img_size[1], self.img_size[0])),
                    transforms.ToTensor(),
                ]
            )

            self._torch = torch
            self._device = device
            self._model = model
            self._converter = converter
            self._transform = transform_fn
            self._available = True
            print(f"[HtrVtRecognizer] Loaded HTR-VT from {model_path} (device={device}).")
            return True
        except Exception as e:
            print(f"[HtrVtRecognizer] Failed to initialise HTR-VT model: {e}")
            self._available = False
            return False

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def _to_pil_grayscale(self, image):
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
                # Convert RGB -> grayscale via luminance approximation.
                arr = (
                    0.299 * arr[:, :, 0]
                    + 0.587 * arr[:, :, 1]
                    + 0.114 * arr[:, :, 2]
                )
        elif arr.ndim != 2:
            return None

        if arr.dtype != np.uint8:
            mx = float(arr.max()) if arr.size else 1.0
            mn = float(arr.min()) if arr.size else 0.0
            if 0.0 <= mn and mx <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        from PIL import Image

        return Image.fromarray(arr).convert("L")

    def _predict(self, pil_img) -> Optional[str]:
        if pil_img is None:
            return None
        try:
            tensor = self._transform(pil_img).unsqueeze(0).to(self._device)
            with self._torch.no_grad():
                preds = self._model(tensor).float()
                preds_size = self._torch.IntTensor([preds.size(1)])
                preds = preds.permute(1, 0, 2).log_softmax(2)
                _, preds_index = preds.max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                texts = self._converter.decode(preds_index.data, preds_size.data)
            return texts[0] if texts else ""
        except Exception as e:
            if self.debug:
                print(f"[HtrVtRecognizer] Inference error: {e}")
            return None

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------
    def process(self, data: dict, config: dict) -> List[dict]:
        pages = data.get("cell-formatter", [])
        if isinstance(pages, dict):
            pages = [pages]
        if not isinstance(pages, list):
            return []

        if not self._ensure_loaded():
            return pages  # No-op fallback: leave cells for downstream OCR.

        for page in pages:
            for column in page.get("columns", []):
                if not any(column.get(flag, False) for flag in self.target_column_flags):
                    continue

                for cell_idx, cell in enumerate(column.get("cells", [])):
                    if cell_idx == 0:  # header row stays untouched
                        continue
                    if cell.get("is_blank", False):  # empty cell — nothing to read
                        continue
                    if cell.get("skip_ocr", False):
                        continue
                    if cell.get("erkannt"):  # already filled (e.g. by quotation detector)
                        continue

                    source_img = cell.get("image_raw")
                    if source_img is None:
                        source_img = cell.get("image")

                    pil_img = self._to_pil_grayscale(source_img)
                    text = self._predict(pil_img)

                    if text is None or not text.strip():
                        # Let TrOCR / fuzzy step have a shot if HTR-VT had nothing to say.
                        continue

                    # Store as a per-source prediction so the consensus stages
                    # (FuzzyMatchingBirdNames / FuzzyMatchingAge) take over.
                    cell.setdefault("predictions", {})["htr_vt"] = text
                    # Provisional erkannt so the UI shows something even before
                    # the consensus stage runs; the fuzzy stage will overwrite it.
                    if not cell.get("erkannt"):
                        cell["erkannt"] = text
                        cell["score"] = 70
                    if self.debug:
                        print(f"[HtrVtRecognizer] cell -> {text!r}")

        print("\nHTR-VT finished!\n")
        return pages
