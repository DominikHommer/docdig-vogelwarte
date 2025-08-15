from .module_base import Module
from typing import List, Dict
import copy


class ReorderColumns(Module):
    def __init__(self):
        super().__init__("column-reorderer")

        # Define expected order
        self.expected_roles = [
            "is_batch_column",
            "is_species_column",
            "is_sexe_column",
            "is_age_column",
            "is_jour-mois_column",
            "is_heure_column",
            "is_alle_column",
            "is_poids_column",
            "empty_column1",
            "empty_column2",
            "laisser-en-blanc_column",
        ]

    def get_preconditions(self) -> List[str]:
        return ["column-marker"]

    def _get_column_role(self, column: dict) -> str | None:
        """Return the first detected role in this column."""
        for role in self.expected_roles:
            if column.get(role, False):
                return role
        return None

    def process(self, data: dict, config: dict) -> List[Dict]:
        pages = data["column-marker"]
        output = []

        for page in pages:
            columns = page["columns"]
            role_to_index = {}


            # Map detected roles to their current column index
            for i, col in enumerate(columns):
                role = self._get_column_role(col)
                if role and role not in role_to_index:
                    role_to_index[role] = i

            # Detect alignment issue
            misaligned = False
            for expected_idx, expected_role in enumerate(self.expected_roles):
                actual_idx = role_to_index.get(expected_role)
                if actual_idx is not None and actual_idx != expected_idx:
                    misaligned = True
                    print(f"Some missalignement!")
                    break

            if not misaligned:
                output.append({"columns": columns})
                continue

            # Align columns by inserting placeholders for missing roles
            for expected_idx, expected_role in enumerate(self.expected_roles):
                if expected_idx < len(columns):
                    current_role = self._get_column_role(columns[expected_idx])
                    if current_role == expected_role:
                        continue  # Role matches, nothing to do

                    # Check if expected role appears later
                    found_later = any(
                        self._get_column_role(col) == expected_role
                        for col in columns[expected_idx + 1:]
                    )
                    if found_later:
                        continue  # It's just misaligned, will be reordered later

                    # expected role is missing AND next role is one step early
                    next_role = self._get_column_role(columns[expected_idx])
                    expected_idx_later = self.expected_roles.index(
                        next_role) if next_role in self.expected_roles else -1

                    if expected_idx_later == expected_idx + 1:
                        print(f"Inserting placeholder for missing column: {expected_role} at index {expected_idx}")
                        placeholder_cells = [{"image": None, "skip_ocr": False} for _ in range(50)]
                        placeholder = {
                            "cells": placeholder_cells,
                            "is_batch_column": False,
                            "is_species_column": False,
                            "is_sexe_column": False,
                            "is_age_column": False,
                            "is_jour-mois_column": False,
                            "is_heure_column": False,
                            "is_alle_column": False,
                            "is_poids_column": False,
                        }
                        placeholder[expected_role] = True
                        columns.insert(expected_idx, placeholder)

            # Rebuild role mapping after placeholder insertion
            role_to_index = {}
            for i, col in enumerate(columns):
                role = self._get_column_role(col)
                if role and role not in role_to_index:
                    role_to_index[role] = i
                else:
                    role = self.expected_roles[i]
                    role_to_index[role] = i


            # Reorder: build new list of columns in correct role order
            reordered = []
            for expected_role in self.expected_roles:
                idx = role_to_index.get(expected_role)
                if idx is not None:
                    print(f"Updated Column: {expected_role}, at idx: {idx}")
                    reordered.append(columns[idx])

            output.append({"columns": reordered})

        print("\nColumns Reordered!\n")
        return output
