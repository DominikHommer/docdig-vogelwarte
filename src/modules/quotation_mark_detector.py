from .module_base import Module
from typing import List, Dict, Optional
import cv2
import numpy as np

from libs.bague_sequence import is_blank_cell
from modules.cell_formatter import CellFormatter


class QuotationMarkDetector(Module):
    """Detects "ditto" cells (same value as the row above).

    Two signals:

    1. **Ink ratio** (all non-skipped columns): almost no dark pixels — the
       classic `"` mark barely leaves ink.
    2. **Ink shape** (wide-content columns only, default: Espèce): the corpus
       (files-3/meta.json) uses four notations — `"`, `''`, `//`, `ii`. The
       tintier ones exceed the ratio threshold, but they all share a shape: a
       small ink blob centred in a cell whose regular content (a species
       name) spans most of the width. Only enabled where that width contrast
       holds; on numeric columns a single digit would look identical.
    """

    def __init__(
        self,
        threshold=0.05,
        skip_column_flags=("is_batch_column",),
        shape_column_flags=("is_species_column",),
        shape_max_width_ratio=0.35,
        shape_max_ink_ratio=0.15,
    ):
        super().__init__("quotationmark-detector")
        self.threshold = threshold
        # Columns where "ditto" marks make no semantic sense (e.g. unique ring numbers).
        self.skip_column_flags = tuple(skip_column_flags)
        # Columns whose regular content is wide — small centred marks are dittos.
        self.shape_column_flags = tuple(shape_column_flags)
        self.shape_max_width_ratio = shape_max_width_ratio
        self.shape_max_ink_ratio = shape_max_ink_ratio

    def get_preconditions(self) -> List[str]:
        return ['cell-formatter']

    @staticmethod
    def _to_uint8_gray(image) -> Optional[np.ndarray]:
        if image is None:
            return None
        img = np.asarray(image)
        if img.size == 0:
            return None
        img = np.nan_to_num(img, nan=255.0, posinf=255.0, neginf=0.0)
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            if img.dtype != np.uint8:
                mx = float(img.max()) if img.size else 1.0
                mn = float(img.min()) if img.size else 0.0
                if 0.0 <= mn and mx <= 1.0:
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            if img.dtype != np.uint8:
                mx = float(img.max()) if img.size else 1.0
                mn = float(img.min()) if img.size else 0.0
                if 0.0 <= mn and mx <= 1.0:
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            return None
        return img

    def detect_by_shape(self, image) -> bool:
        """True when the ink forms a small, horizontally centred blob.

        Covers `''`, `//`, `ii` — too much ink for the ratio check, but their
        footprint is a fraction of the cell width. Table borders are stripped
        before measuring so they don't count as ink.
        """
        img = self._to_uint8_gray(image)
        if img is None:
            return False
        h, w = img.shape[:2]
        if h < 16 or w < 40:
            return False

        margin_y = max(3, h // 8)
        margin_x = max(3, w // 16)
        inner = img[margin_y : h - margin_y, margin_x : w - margin_x]
        if inner.size == 0:
            return False

        ink = inner < 200
        ink_count = int(ink.sum())
        if ink_count < 8:
            return False  # effectively blank — the ratio check owns this case
        if ink_count / ink.size > self.shape_max_ink_ratio:
            return False  # too much ink to be a ditto mark

        ys, xs = np.nonzero(ink)
        bbox_w = int(xs.max() - xs.min() + 1)
        iw = inner.shape[1]
        if bbox_w > self.shape_max_width_ratio * iw:
            return False  # spans the cell — real content

        # Horizontally centred: blob centre within the middle half of the cell.
        cx = float(xs.mean())
        return abs(cx - iw / 2.0) <= iw * 0.25

    def detect_quotation_marks(self, image, relative_non_white_threshold: float) -> bool:
        # Interpolation and normalization already done in cell_formatter.py
        #image = (image - image.min()) / (image.max() - image.min())
        #image = 1.0 - image
        #image = (image * 255).astype(np.uint8)

        # Debug outputAdd commentMore actions
        #print(f"Image dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")
        #cv2.imshow("Quotation Mark Candidate", image)
        #cv2.waitKey(0)  # Press a key to close
        #cv2.destroyAllWindows()

        if image is None:
            return False

        img = np.asarray(image)
        if img.size == 0:
            return False

        img = np.nan_to_num(img, nan=255.0, posinf=255.0, neginf=0.0)

        # Ensure grayscale
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            if img.dtype != np.uint8:
                mx = float(img.max()) if img.size else 1.0
                mn = float(img.min()) if img.size else 0.0
                if 0.0 <= mn and mx <= 1.0:
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            if img.dtype != np.uint8:
                mx = float(img.max()) if img.size else 1.0
                mn = float(img.min()) if img.size else 0.0
                if 0.0 <= mn and mx <= 1.0:
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            # Unsupported shape
            return False

        non_white_pixels = int(np.sum(img < 250))  # allow small tolerance
        total_pixels = int(img.shape[0] * img.shape[1])
        if total_pixels == 0:
            return False
        relative_non_white = non_white_pixels / total_pixels

        #print(f" NonWhite: {non_white_pixels}, relative: {relative_non_white}")

        return relative_non_white < relative_non_white_threshold

    def process(self, data: dict, config: dict) -> List[Dict]:
        pages = data.get("cell-formatter", [])
        if isinstance(pages, dict):
            pages = [pages]

        output = []

        for page in pages:
            processed_page = {"columns": []}

            columns = page.get("columns", [])
            if not isinstance(columns, list):
                columns = []

            for col_idx, column in enumerate(columns):
                cells = column.get("cells", [])

                processed_cells = []

                # Dispatch semantically: only skip columns that explicitly forbid ditto detection.
                if any(column.get(flag, False) for flag in self.skip_column_flags):
                    processed_page["columns"].append(column)
                    continue

                # Shape-based detection only where regular content is wide.
                use_shape = any(
                    column.get(flag, False) for flag in self.shape_column_flags
                )

                for cell_idx, cell in enumerate(cells):

                    if cell_idx == 0: # if header
                        processed_cells.append(cell)
                        continue

                    # Blank cells (flagged by CellFormatter) are EMPTY, not
                    # ditto — "same as above" needs an actual mark.
                    if cell.get("is_blank", False) or cell.get("skip_ocr", False):
                        processed_cells.append(cell)
                        continue

                    image = cell.get("image")
                    if image is None:
                        processed_cells.append(cell)
                        continue

                    # Stage-independent blank guard (same heuristic as the
                    # CellFormatter gate): no ink at all -> empty, not ditto.
                    if is_blank_cell(image, threshold=CellFormatter.BLANK_INK_THRESHOLD):
                        cell["is_blank"] = True
                        cell["skip_ocr"] = True
                        processed_cells.append(cell)
                        continue

                    is_quote = self.detect_quotation_marks(image, self.threshold)
                    if not is_quote and use_shape:
                        is_quote = self.detect_by_shape(image)

                    if is_quote:
                        print("Detected QuotationMark")

                    cell["erkannt"] = '"' if is_quote else cell.get("erkannt", "")
                    cell["score"] = 100 if is_quote else cell.get("score", -1)
                    cell["skip_ocr"] = is_quote

                    processed_cells.append(cell)

                processed_page["columns"].append({
                    "cells": processed_cells if processed_cells else cells,
                    "is_batch_column": column.get("is_batch_column", False),
                    "is_species_column": column.get("is_species_column", False),
                    "is_sexe_column": column.get("is_sexe_column", False),
                    "is_age_column": column.get("is_age_column", False),
                    "is_jour-mois_column": column.get("is_jour-mois_column", False),
                    "is_heure_column": column.get("is_heure_column", False),
                    "is_alle_column": column.get("is_alle_column", False),
                    "is_poids_column": column.get("is_poids_column", False)
                })

            output.append(processed_page)

        print("\nQuotationmark-Detector finished!\n")
        return output
