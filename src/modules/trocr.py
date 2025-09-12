from .module_base import Module
import numpy as np
from typing import List, Dict
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os
import pandas as pd
import cv2

class TrOCR(Module):
    def __init__(self, model_name="microsoft/trocr-base-stage1", output_path="data/output/trocr_output.xlsx", debug=False, debug_folder="debug/debug_cell_denoiser/"):
        super().__init__("trocr")

        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)
            
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.output_path = output_path

    def get_preconditions(self) -> List[str]:
        return ['quotationmark-detector', 'cell-formatter']

    def _to_rgb(self, image):
        if image is None:
            return None
        img = np.asarray(image)
        if img.size == 0:
            return None

        img = np.nan_to_num(img, nan=255.0, posinf=255.0, neginf=0.0)

        if img.ndim == 2:
            # Graustufe -> RGB
            if img.dtype != np.uint8:
                mx = float(img.max()) if img.size else 1.0
                mn = float(img.min()) if img.size else 0.0
                if 0.0 <= mn and mx <= 1.0:
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if img.ndim == 3:
            # RGBA -> RGB
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            # Datentyp normalisieren
            if img.dtype != np.uint8:
                mx = float(img.max()) if img.size else 1.0
                mn = float(img.min()) if img.size else 0.0
                if 0.0 <= mn and mx <= 1.0:
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            return img

        # Unsupported shape
        return None

    def process(self, data: dict, config: dict) -> List[Dict]:
        valid_keys = self.get_preconditions()
        input_key = next((k for k in valid_keys if k in data), None)
        if input_key is None:
            raise ValueError("No valid input found for TrOCR.")

        pages = data[input_key]
        if isinstance(pages, dict):
            pages = [pages]
        if not isinstance(pages, list):
            return []

        output = []

        for page_idx, page in enumerate(pages):
            processed_page = {"columns": []}

            columns = page.get("columns", [])
            if not isinstance(columns, list):
                columns = []

            for col_idx, column in enumerate(columns):
                cells = column.get("cells", [])
                processed_column = []

                # Speichere Beispielbild für Debug-Zwecke
                if cells and isinstance(cells[0], dict) and "image" in cells[0] and cells[0]["image"] is not None:
                    if self.debug:
                        debug_img = self._to_rgb(cells[0]["image"])
                        if debug_img is not None:
                            debug_path = os.path.join(self.debug_folder, f"test_png_{col_idx}.png")
                            cv2.imwrite(debug_path, debug_img)

                if not (col_idx == 1 or col_idx == 3 or col_idx == 4 or col_idx == 5):
                    print(f"Skipping OCR for column {col_idx}.")
                    processed_page["columns"].append(column)
                    continue

                for cell_idx, cell in enumerate(cells):

                    if cell_idx == 0:  # if header
                        processed_column.append(cell)
                        continue

                    if cell.get("skip_ocr", False):
                        processed_column.append(cell)
                        print("OCR skipped due to quotationmark")
                    else:
                        image = cell.get("image")
                        rgb = self._to_rgb(image)
                        if rgb is None:
                            processed_column.append(cell)
                            continue

                        pil_image = Image.fromarray(rgb).convert("RGB")

                        #print(f"Image dtype: {rgb.dtype}, min: {rgb.min()}, max: {rgb.max()}")
                        #cv2.imshow("Quotation Mark Candidate", rgb)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()

                        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

                        with torch.no_grad():
                            generated_ids = self.model.generate(**inputs)
                            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                        cell["erkannt"] = text
                        cell["score"] = 0
                        processed_column.append(cell)
                        print(f"Text erkannt: {text}")

                processed_page["columns"].append({
                    "cells": processed_column if processed_column else cells,
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

        print("\nOCR finished!\n")
        return output
