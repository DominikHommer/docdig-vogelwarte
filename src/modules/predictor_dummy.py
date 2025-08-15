from .module_base import Module
import numpy as np
from typing import List, Dict


class PredictorDummy(Module):
    def __init__(self):
        super().__init__("predictor-dummy")

    def get_preconditions(self) -> List[str]:
        return ['cell-denoiser']

    def process(self, data: dict, config: dict) -> List[Dict]:
        denoised_pages = data['cell-denoiser']
        dummy_predictions = []

        for page in denoised_pages:
            dummy_page = {"columns": []}

            for col in page["columns"]:
                col_cells = []

                for img in col:
                    # Jede Zelle besteht aus dem Bild + 2 Textzeilen
                    cell = {
                        "image": img,
                        "erkannt": "",
                        "verbesserung": ""
                    }
                    col_cells.append(cell)

                dummy_page["columns"].append(col_cells)

            dummy_predictions.append(dummy_page)

        return dummy_predictions
