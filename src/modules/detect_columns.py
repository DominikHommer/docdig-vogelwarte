import numpy as np
import pytesseract
import cv2

from .module_base import Module


def crop_center(image, width, height):
    h, w = image.shape[:2]
    x_start = max((w - width) // 2, 0)
    y_start = max((h - height) // 2, 0)
    return image[y_start:y_start + height, x_start:x_start + width]

class DetectColumns(Module):
    def __init__(self,
                 debug: bool = False,
                 batch_keywords: list[str] = ["bague", "no", "n°", "numéro", "nummer"],
                 species_keywords: list[str] = ["espèce", "espece", "art"],
                 sexe_keywords: list[str] = ["sexe", "d'aexe"],
                 age_keywords: list[str] = ["age"],
                 jourmois_keywords: list[str] = ["jour mois", "jour", "mois"],
                 heure_keywords: list[str] = ["heure"],
                 alle_keywords: list[str] = ["alle", "aile", "aiie"],
                 poids_keywords: list[str] = ["poids", "polds"]):
        super().__init__("column-marker")
        self.debug = debug
        self.batch_keywords = set(k.lower() for k in batch_keywords)
        self.species_keywords = set(k.lower() for k in species_keywords)
        self.sexe_keywords = set(k.lower() for k in sexe_keywords)
        self.age_keywords = set(k.lower() for k in age_keywords)
        self.jourmois_keywords = set(k.lower() for k in jourmois_keywords)
        self.heure_keywords = set(k.lower() for k in heure_keywords)
        self.alle_keywords = set(k.lower() for k in alle_keywords)
        self.poids_keywords = set(k.lower() for k in poids_keywords)

    def get_preconditions(self) -> list[str]:
        return ["row-extractor"]

    def process(self, data: dict, config: dict) -> list[dict]:
        pages = data["row-extractor"]
        output = []

        for _, page in enumerate(pages):
            processed_page = {"columns": []}

            for _, column_cells in enumerate(page["columns"]):
                is_batch = False
                is_spezies = False
                is_sexe = False
                is_age = False
                is_jourmois = False
                is_heure = False
                is_alle = False
                is_poids = False

                if not column_cells or not isinstance(column_cells[0], np.ndarray):
                    processed_page["columns"].append({
                        "cells": [],
                        "is_batch_column": False,
                        "is_species_column": False,
                        "is_sexe_column": False,
                        "is_age_column": False,
                        "is_jour-mois_column": False,
                        "is_heure_column": False,
                        "is_alle_column": False,
                        "is_poids_column": False
                    })

                    continue

                first_cell_img = column_cells[0]
                cropped = crop_center(first_cell_img, width=200, height=50)

                # OCR attempts
                text_cropped = pytesseract.image_to_string(cropped, lang="fra").strip().lower()
                text_full = pytesseract.image_to_string(first_cell_img, lang="fra").strip().lower()

                # Display the detected Text
                #print(f"Cropped: {text_cropped}, Full: {text_full}")

                # Check batch column
                if any(k in text_cropped for k in self.batch_keywords) or \
                   any(k in text_full for k in self.batch_keywords):
                    is_batch = True
                    print(f"Found Batch Column!")

                # Check species column
                if any(k in text_cropped for k in self.species_keywords) or \
                   any(k in text_full for k in self.species_keywords) or \
                   first_cell_img.shape[1] >= 280:
                    is_spezies = True
                    print(f"Found Species Column!")

                # Check sexe column
                if any(k in text_cropped for k in self.sexe_keywords) or \
                   any(k in text_full for k in self.sexe_keywords):
                    is_sexe = True
                    print(f"Found Sexe Column!")

                # Check age column
                if any(k in text_cropped for k in self.age_keywords) or \
                   any(k in text_full for k in self.age_keywords):
                    is_age = True
                    print(f"Found Age Column!")

                # Check Jour Mois column
                if any(k in text_cropped for k in self.jourmois_keywords) or \
                   any(k in text_full for k in self.jourmois_keywords):
                    is_jourmois = True
                    print(f"Found Jour Mois Column!")

                # Check Heure column
                if any(k in text_cropped for k in self.heure_keywords) or \
                   any(k in text_full for k in self.heure_keywords):
                    is_heure = True
                    print(f"Found Heure Column!")

                # Check Alle column
                if any(k in text_cropped for k in self.alle_keywords) or \
                   any(k in text_full for k in self.alle_keywords):
                    is_alle = True
                    print(f"Found Alle Column!")

                # Check Poids column
                if any(k in text_cropped for k in self.poids_keywords) or \
                   any(k in text_full for k in self.poids_keywords):
                    is_poids = True
                    print(f"Found Poids Column!")

                # Convert ndarray to dicts
                cell_dicts = [{"image": img, "skip_ocr": False} for img in column_cells]

                processed_page["columns"].append({
                    "cells": cell_dicts,
                    "is_batch_column": is_batch,
                    "is_species_column": is_spezies,
                    "is_sexe_column": is_sexe,
                    "is_age_column": is_age,
                    "is_jour-mois_column": is_jourmois,
                    "is_heure_column": is_heure,
                    "is_alle_column": is_alle,
                    "is_poids_column": is_poids
                })

            output.append(processed_page)
            print("\nColumn Detector finished!\n")

        return output
