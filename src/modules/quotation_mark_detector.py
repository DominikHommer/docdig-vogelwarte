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

        non_white_pixels = np.sum(image < 250)  # allow small tolerance
        total_pixels = image.shape[0] * image.shape[1]
        relative_non_white = non_white_pixels / total_pixels

        #print(f" NonWhite: {non_white_pixels}, relative: {relative_non_white}")

        return relative_non_white < relative_non_white_threshold

    def process(self, data: dict, config: dict) -> List[Dict]:
        pages = data["cell-formatter"]
        if isinstance(pages, dict):
            pages = [pages]

        output = []

        for page in pages:
            processed_page = {"columns": []}

            for col_idx, column in enumerate(page["columns"]):
                cells = column["cells"]

                processed_cells = []

                if not col_idx == 1:
                    processed_page["columns"].append(column)
                    continue

                for cell_idx, cell in enumerate(cells):

                    if cell_idx == 0: # if header
                        processed_cells.append(cell)
                        continue

                    image = cell["image"]
                    is_quote = self.detect_quotation_marks(image, self.threshold)

                    if is_quote:
                        print("Detected QuotationMark")

                    cell["erkannt"] = '"' if is_quote else cell.get("erkannt", "")
                    cell["score"] = 100 if is_quote else cell.get("score", -1)
                    cell["skip_ocr"] = is_quote

                    processed_cells.append(cell)

                processed_page["columns"].append({
                    "cells": processed_cells,
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
