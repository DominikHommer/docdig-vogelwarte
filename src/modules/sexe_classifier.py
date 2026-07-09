"""CNN classifier for the sexe column (m / w / none).

Reads a Keras model trained on 150x60 grayscale crops with 3 output classes,
trained from ``M-W-Classification/dataset/`` (class subfolders ``m``, ``none``,
``w`` — see M-W-Classification/cnn.ipynb or tools/train_sexe_model.py).
Missing model -> no-op (TrOCR fallback).
"""

import json
import os
from typing import List, Optional

import cv2
import numpy as np

from libs.bague_sequence import is_blank_cell
from libs.cell_cropping import strip_cell_borders
from .module_base import Module


class SexeClassifier(Module):
    DEFAULT_MODEL_PATHS = [
        "./config/sexe_model.keras",
        "./config/sexe/sexe_model.keras",
        "./config/mw_model.keras",
    ]

    DEFAULT_CLASSES_PATHS = [
        "./config/sexe_classes.json",
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        classes_path: Optional[str] = None,
        classes: Optional[List[str]] = None,
        target_size=(150, 60),  # (height, width) — matches ImageDataGenerator target_size
        # 3-class softmax: 0.4 is barely above uniform (0.33). Require a real
        # majority — below that the cell stays blank for manual review instead
        # of risking a silent wrong value.
        score_threshold: float = 0.5,
        debug: bool = False,
    ):
        super().__init__("sexe-classifier")
        self.model_path = model_path
        self.classes_path = classes_path
        # Default: alphabetical ordering used by flow_from_directory over
        # M-W-Classification/dataset/{m,none,w}. Overridden by
        # config/sexe_classes.json when present.
        self.classes = classes or ["m", "none", "w"]
        self.target_size = tuple(target_size)
        self.score_threshold = score_threshold
        self.debug = debug

        self._tf = None
        self._model = None
        self._available = None

    def get_preconditions(self) -> List[str]:
        return ["cell-formatter"]

    # ------------------------------------------------------------------
    def _resolve_model_path(self) -> Optional[str]:
        if self.model_path and os.path.exists(self.model_path):
            return self.model_path
        for path in self.DEFAULT_MODEL_PATHS:
            if os.path.exists(path):
                return path
        return None

    def _load_classes(self):
        for path in [self.classes_path, *self.DEFAULT_CLASSES_PATHS]:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.classes = list(data)
                    return
                if isinstance(data, dict):
                    # Accept either {"classes": [...]} or {label: idx} (Keras style)
                    if "classes" in data and isinstance(data["classes"], list):
                        self.classes = list(data["classes"])
                        return
                    if all(isinstance(v, int) for v in data.values()):
                        ordered = sorted(data.items(), key=lambda kv: kv[1])
                        self.classes = [k for k, _ in ordered]
                        return
            except Exception as e:
                print(f"[SexeClassifier] Could not parse classes file {path}: {e}")

    def _ensure_loaded(self) -> bool:
        if self._available is not None:
            return self._available

        model_path = self._resolve_model_path()
        if not model_path:
            print("[SexeClassifier] No sexe model found. Skipping sexe column.")
            self._available = False
            return False

        try:
            import tensorflow as tf
        except ImportError as e:
            print(f"[SexeClassifier] tensorflow missing: {e}")
            self._available = False
            return False

        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            self._load_classes()
            self._tf = tf
            self._model = model
            self._available = True
            print(f"[SexeClassifier] Loaded sexe model from {model_path}, classes={self.classes}.")
            return True
        except Exception as e:
            print(f"[SexeClassifier] Failed to load sexe model: {e}")
            self._available = False
            return False

    # ------------------------------------------------------------------
    def _prepare(self, image) -> Optional[np.ndarray]:
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

        h, w = self.target_size  # (height, width)
        try:
            resized = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
        except cv2.error:
            return None

        prepared = resized.astype(np.float32) / 255.0
        prepared = np.expand_dims(prepared, axis=-1)
        return np.expand_dims(prepared, axis=0)

    def _label_for(self, preds: np.ndarray):
        idx = int(np.argmax(preds))
        score = float(np.max(preds))
        if idx >= len(self.classes):
            return None, score
        return self.classes[idx], score

    # ------------------------------------------------------------------
    def process(self, data: dict, config: dict) -> List[dict]:
        pages = data.get("cell-formatter", [])
        if isinstance(pages, dict):
            pages = [pages]
        if not isinstance(pages, list):
            return []

        if not self._ensure_loaded():
            return pages

        for page in pages:
            for column in page.get("columns", []):
                if not column.get("is_sexe_column", False):
                    continue

                for cell_idx, cell in enumerate(column.get("cells", [])):
                    if cell_idx == 0:
                        continue
                    if cell.get("is_blank", False):  # empty cell stays empty
                        continue
                    if cell.get("skip_ocr", False):
                        continue
                    if cell.get("erkannt"):
                        continue

                    source_img = cell.get("image_raw")
                    if source_img is None:
                        source_img = cell.get("image")

                    # Stage-independent blank guard on the border-stripped
                    # crop: an empty cell with a table-line sliver must stay
                    # empty — the CNN would otherwise classify the sliver.
                    cleaned = strip_cell_borders(source_img)
                    if cleaned is not None and is_blank_cell(
                        cleaned, threshold=0.004, margin_y=3, margin_x=3
                    ):
                        cell["is_blank"] = True
                        cell["skip_ocr"] = True
                        cell["erkannt"] = ""
                        cell["score"] = -1
                        continue

                    batched = self._prepare(source_img)
                    if batched is None:
                        continue

                    try:
                        preds = self._model.predict(batched, verbose=0)
                    except Exception as e:
                        if self.debug:
                            print(f"[SexeClassifier] predict error: {e}")
                        continue

                    label, score = self._label_for(preds[0])
                    if label is None:
                        continue
                    if score < self.score_threshold:
                        if self.debug:
                            print(f"[SexeClassifier] low confidence ({score:.2f}) -> {label!r}")
                        continue

                    # "none" == the bander left the cell empty; show it blank.
                    cell.setdefault("predictions", {})["sexe_cnn"] = label
                    cell["erkannt"] = "" if label == "none" else label
                    cell["score"] = int(round(score * 100))
                    cell["skip_ocr"] = True

        print("\nSexe classifier finished!\n")
        return pages
