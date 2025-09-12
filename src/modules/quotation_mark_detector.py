from .module_base import Module
from typing import List, Dict
import cv2
import numpy as np


class QuotationMarkDetector(Module):
    def __init__(self, threshold=0.05):
        super().__init__("quotationmark-detector")
        self.threshold = threshold

    def get_preconditions(self) -> List[str]:
        return ['cell-formatter']

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

                if not col_idx == 1:
                    processed_page["columns"].append(column)
                    continue

                for cell_idx, cell in enumerate(cells):

                    if cell_idx == 0: # if header
                        processed_cells.append(cell)
                        continue

                    image = cell.get("image")
                    if image is None:
                        processed_cells.append(cell)
                        continue

                    is_quote = self.detect_quotation_marks(image, self.threshold)

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
