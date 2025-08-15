from .module_base import Module
from typing import List, Dict
import os
import json
from rapidfuzz import process, fuzz

class FuzzyMatchingBirdNames(Module):
    def __init__(self, class_label_path: str = "./config/class_indices.json", score_threshold: int = 10):
        super().__init__("fuzzy-corrector-Species")
        self.class_labels = []
        self.score_threshold = score_threshold

        if os.path.exists(class_label_path):
            with open(class_label_path, "r", encoding="utf-8") as f:
                name_to_idx = json.load(f)
                # Extract just the class names
                self.class_labels = list(name_to_idx.keys())

    def get_preconditions(self) -> List[str]:
        # Accepts output from any detector module with same structure
        return ['trocr', 'predictor', 'predictor-dummy']

    def process(self, data: dict, config: dict) -> List[Dict]:

        valid_keys = self.get_preconditions()
        input_key = next((k for k in data if k in valid_keys), None)
        if input_key is None:
            raise ValueError("No valid input found for fuzzy matching.")

        pages = data[input_key]
        if isinstance(pages, dict):
            pages = [pages]

        output = []

        for page in pages:
            processed_page = {"columns": []}

            for column in page["columns"]:
                is_species_column = column.get("is_species_column", False)
                cells = column.get("cells", [])
                processed_cells = []

                if not is_species_column:
                    # Spalte nicht bearbeiten, einfach übernehmen
                    processed_page["columns"].append(column)
                    continue

                for cell_idx, cell in enumerate(cells):

                    if cell_idx == 0:  # if header
                        processed_cells.append(cell)
                        continue

                    if cell.get("skip_ocr", False):
                        print("Skipped due to questionmark fuzzy")
                        processed_cells.append(cell)
                    else:
                        text = cell.get("erkannt", "")
                        best_match, score, _ = process.extractOne(
                            text,
                            self.class_labels,
                            scorer=fuzz.token_sort_ratio
                        ) if self.class_labels else ("", 0, None)

                        correction = best_match if score >= self.score_threshold else ""
                        cell["erkannt"] = correction
                        cell["score"] = score

                        processed_cells.append(cell)

                # FIXME: This is used everywhere and very repetitive. Can be improved by a lot by just merging the arrays...
                processed_page["columns"].append({
                    "cells": processed_cells,
                    "is_batch_column": column.get("is_batch_column", False),
                    "is_species_column": is_species_column,
                    "is_sexe_column": column.get("is_sexe_column", False),
                    "is_age_column": column.get("is_age_column", False),
                    "is_jour-mois_column": column.get("is_jour-mois_column", False),
                    "is_heure_column": column.get("is_heure_column", False),
                    "is_alle_column": column.get("is_alle_column", False),
                    "is_poids_column": column.get("is_poids_column", False)
                })

            output.append(processed_page)

        return output


class FuzzyMatchingAge(Module):
    def __init__(self, class_label_path: str = "./config/age_classes.json", score_threshold: int = 10):
        super().__init__("fuzzy-corrector-Age")
        self.class_labels = []
        self.score_threshold = score_threshold

        if os.path.exists(class_label_path):
            with open(class_label_path, "r", encoding="utf-8") as f:
                name_to_idx = json.load(f)
                # Extract just the class names
                self.class_labels = list(name_to_idx.keys())

    def get_preconditions(self) -> List[str]:
        # Accepts output from any detector module with same structure
        return ['trocr', 'predictor', 'predictor-dummy', 'fuzzy-corrector-species']

    def process(self, data: dict, config: dict) -> List[Dict]:

        valid_keys = self.get_preconditions()
        input_key = next((k for k in data if k in valid_keys), None)
        if input_key is None:
            raise ValueError("No valid input found for fuzzy matching.")

        pages = data[input_key]
        if isinstance(pages, dict):
            pages = [pages]

        output = []

        for page in pages:
            processed_page = {"columns": []}

            for col_idx, column in enumerate(page["columns"]):
                cells = column.get("cells", [])
                processed_cells = []

                if not col_idx == 3:
                    # Spalte nicht bearbeiten, einfach übernehmen
                    processed_page["columns"].append(column)
                    continue

                for cell_idx, cell in enumerate(cells):

                    if cell_idx == 0:  # if header
                        processed_cells.append(cell)
                        continue

                    if cell.get("skip_ocr", False):
                        print("Skipped due to questionmark fuzzy")
                        processed_cells.append(cell)
                    else:
                        text = cell.get("erkannt", "")
                        best_match, score, _ = process.extractOne(
                            text,
                            self.class_labels,
                            scorer=fuzz.token_sort_ratio
                        ) if self.class_labels else ("", 0, None)

                        correction = best_match if score >= self.score_threshold else ""
                        cell["erkannt"] = correction
                        cell["score"] = score

                        processed_cells.append(cell)

                processed_page["columns"].append({
                    "cells": processed_cells,
                    "is_batch_column": column.get("is_batch_column", False),
                    "is_species_column": column.get("is_species_column", False),
                    "is_sexe_column": column.get("is_sexe_column", False),
                    "is_age_column": True,
                    "is_jour-mois_column": column.get("is_jour-mois_column", False),
                    "is_heure_column": column.get("is_heure_column", False),
                    "is_alle_column": column.get("is_alle_column", False),
                    "is_poids_column": column.get("is_poids_column", False)
                })

            output.append(processed_page)

        print("\nFuzzy Matching finished!\n")
        return output
