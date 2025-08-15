import os
import numpy as np
import cv2
from .module_base import Module
import tensorflow
from tensorflow.keras.models import load_model

def weighted_mse(y_true, y_pred):
    img_height, img_width = y_true.shape[1], y_true.shape[2]
    
    x = np.linspace(0, 1, img_width)
    y = np.linspace(0, 1, img_height)
    xv, yv = np.meshgrid(x, y)
    
    sigma = 0.3
    mask = np.exp(-((xv - 0.5)**2 + (yv - 0.5)**2) / (2 * sigma**2))
    mask = tensorflow.convert_to_tensor(mask, dtype=tensorflow.float32)
    mask = tensorflow.expand_dims(mask, axis=-1)
    
    error = tensorflow.square(y_true - y_pred)
    weighted_error = error * mask
    return tensorflow.reduce_mean(weighted_error)

class CellDenoiserResult:
    columns: list[list[np.ndarray]]

class CellDenoiser(Module):
    def __init__(self, debug=False, debug_folder="debug/debug_cell_denoiser/"):
        super().__init__("cell-denoiser")
        
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['column-reorderer']
    
    def process(self, data: dict, config: dict) -> list:
        pages: list = data.get('column-reorderer')

        model = load_model(config["denoise"]["model"], custom_objects={"weighted_mse": weighted_mse})
    
        result = []
        for p_i, page in enumerate(pages):
            page_data = {"columns": []}

            for col_nr, col in enumerate(page["columns"]):
                denoised_cells = []

                for row_nr, cell in enumerate(col["cells"]):
                    img = cell["image"]
                    if img is None:
                        denoised_cells.append(cell)
                        continue

                    o_h, o_w = img.shape
                    img_resized = cv2.resize(img, (384, 80))
                    img_resized = np.expand_dims(img_resized, axis=-1)
                    img_resized = np.expand_dims(img_resized, axis=0)

                    # Predict denoised image
                    output = model.predict(cv2.bitwise_not(img_resized))
                    output = np.squeeze(output)
                    output = cv2.resize(output, (o_w, o_h))

                    if self.debug:
                        debug_path = os.path.join(self.debug_folder, f"page_{p_i}_column_{col_nr}_row_{row_nr}.jpg")
                        cv2.imwrite(debug_path, output)

                    # Save denoised image back into the cell
                    cell["image"] = output
                    denoised_cells.append(cell)

                # Append column with metadata and denoised cells
                page_data["columns"].append({
                    "cells": denoised_cells,
                    "is_batch_column": col.get("is_batch_column", False),
                    "is_species_column": col.get("is_species_column", False),
                    "is_sexe_column": col.get("is_sexe_column", False),
                    "is_age_column": col.get("is_age_column", False),
                    "is_jour-mois_column": col.get("is_jour-mois_column", False),
                    "is_heure_column": col.get("is_heure_column", False),
                    "is_alle_column": col.get("is_alle_column", False),
                    "is_poids_column": col.get("is_poids_column", False)
                })

            result.append(page_data)

        print("\nAll Cells denoised!\n")
        return result