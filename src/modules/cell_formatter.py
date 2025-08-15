from .module_base import Module
from typing import List, Dict
import numpy as np

class CellFormatter(Module):
    def __init__(self):
        super().__init__("cell-formatter")

    def get_preconditions(self) -> List[str]:
        return ['cell-denoiser']

    def process(self, data: dict, config: dict) -> List[Dict]:
        pages = data["cell-denoiser"]
        output = []

        for page in pages:
            formatted_page = {"columns": []}

            for column in page["columns"]:
                formatted_column = []

                for cell_image in column["cells"]:

                    image = cell_image if isinstance(cell_image, np.ndarray) else cell_image["image"]

                    if image is None:
                        formatted_column.append({
                            "image": None,
                            "erkannt": "",
                            "score": -1,
                            "verbesserung": "",
                            "skip_ocr": False
                        })
                        continue

                    # Interpolate, and normalize image
                    image = (image - image.min()) / (image.max() - image.min())
                    image = 1.0 - image
                    image = (image * 255).astype(np.uint8)


                    formatted_column.append({
                        "image": image,
                        "erkannt": "",
                        "score": -1,
                        "verbesserung": "",
                        "skip_ocr": False
                    })

                formatted_page["columns"].append({
                    "cells": formatted_column,
                    "is_batch_column": column.get("is_batch_column", False),
                    "is_species_column": column.get("is_species_column", False),
                    "is_sexe_column": column.get("is_sexe_column", False),
                    "is_age_column": column.get("is_age_column", False),
                    "is_jour-mois_column": column.get("is_jour-mois_column", False),
                    "is_heure_column": column.get("is_heure_column", False),
                    "is_alle_column": column.get("is_alle_column", False),
                    "is_poids_column": column.get("is_poids_column", False)
                })

            output.append(formatted_page)

        print("\nAll Cells Formatted!\n")

        return output
