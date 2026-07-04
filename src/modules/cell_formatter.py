"""Cell formatter — normalises the cell-dict schema after the row extractor.

The CellDenoiser used to run upstream of this stage and overwrite ``image``.
We no longer denoise (it hurt downstream OCR more than it helped), so this
module is now essentially a schema validator: every cell has the fields the
rest of the pipeline relies on (``image``, ``image_raw``, ``erkannt``,
``score``, ``skip_ocr``, ``verbesserung``) and images stay dark-on-light
(black text on white background) — no inversion, no normalisation.

This is also where the **blank-cell gate** lives: cells that carry
essentially no ink are flagged ``is_blank`` and ``skip_ocr=True`` here, so no
downstream recognizer (HTR-VT, TrOCR, Digit, Sexe) ever hallucinates a value
into an empty cell. The threshold is deliberately much stricter than the
ditto detector's — a real `"` mark leaves ~10x more ink than scanner noise,
so ditto cells pass through and the QuotationMarkDetector still sees them.
"""

from typing import List, Dict
import numpy as np

from libs.bague_sequence import is_blank_cell
from libs.cell_cropping import strip_cell_borders
from .module_base import Module


def _coerce_uint8(image):
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.size == 0:
        return None
    if arr.dtype != np.uint8:
        mx = float(arr.max()) if arr.size else 1.0
        mn = float(arr.min()) if arr.size else 0.0
        if 0.0 <= mn and mx <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class CellFormatter(Module):
    # Fraction of dark pixels (in the border-stripped centre) below which a
    # cell counts as empty. Must stay well below the ink of a `"` ditto mark.
    BLANK_INK_THRESHOLD = 0.004

    def __init__(self, blank_ink_threshold: float = None):
        super().__init__("cell-formatter")
        self.blank_ink_threshold = (
            blank_ink_threshold
            if blank_ink_threshold is not None
            else self.BLANK_INK_THRESHOLD
        )

    def get_preconditions(self) -> List[str]:
        # Accept output either directly from DetectColumns (no denoiser) or
        # from a denoiser if someone wires it back in.
        return ["column-marker", "column-reorderer", "cell-denoiser"]

    def process(self, data: dict, config: dict) -> List[Dict]:
        pages = (
            data.get("column-marker")
            or data.get("column-reorderer")
            or data.get("cell-denoiser")
            or []
        )
        if isinstance(pages, dict):
            pages = [pages]
        if not isinstance(pages, list):
            return []

        output = []
        for page in pages:
            formatted_page = {"columns": []}

            for column in page.get("columns", []):
                formatted_column = []

                for cell in column.get("cells", []):
                    is_ndarray = isinstance(cell, np.ndarray)
                    image = cell if is_ndarray else cell.get("image")
                    image_raw = None if is_ndarray else cell.get("image_raw", image)

                    image = _coerce_uint8(image)
                    image_raw = _coerce_uint8(image_raw)

                    if image is None and image_raw is None:
                        formatted_column.append(
                            {
                                "image": None,
                                "image_raw": None,
                                "erkannt": "" if is_ndarray else (cell.get("erkannt") or ""),
                                "score": -1 if is_ndarray else cell.get("score", -1),
                                "skip_ocr": False if is_ndarray else cell.get("skip_ocr", False),
                                "verbesserung": "" if is_ndarray else cell.get("verbesserung", ""),
                            }
                        )
                        continue

                    if image is None:
                        image = image_raw
                    if image_raw is None:
                        image_raw = image

                    # Blank gate: essentially inkless cells never reach a
                    # recognizer — they stay empty instead of hallucinated.
                    # Judge on the border-stripped crop, so a sliver of the
                    # printed table grid doesn't count as "content". Margins
                    # stay small — the stripping already removed the borders,
                    # and content near the edge (e.g. the printed bague
                    # suffix digit) must still count as ink.
                    blank = is_blank_cell(
                        strip_cell_borders(image_raw),
                        threshold=self.blank_ink_threshold,
                        margin_y=3,
                        margin_x=3,
                    )

                    formatted = {
                        "image": image,
                        "image_raw": image_raw,
                        "erkannt": "" if is_ndarray else (cell.get("erkannt") or ""),
                        "score": -1 if is_ndarray else cell.get("score", -1),
                        "skip_ocr": False if is_ndarray else cell.get("skip_ocr", False),
                        "verbesserung": "" if is_ndarray else cell.get("verbesserung", ""),
                        # Preserve flags set by upstream (DetectColumns etc.) if cell is a dict.
                        **(
                            {
                                k: v
                                for k, v in cell.items()
                                if k
                                not in {
                                    "image",
                                    "image_raw",
                                    "erkannt",
                                    "score",
                                    "skip_ocr",
                                    "verbesserung",
                                }
                            }
                            if not is_ndarray
                            else {}
                        ),
                    }
                    formatted["is_blank"] = blank
                    if blank:
                        formatted["skip_ocr"] = True
                        formatted["erkannt"] = ""
                        formatted["score"] = -1
                    formatted_column.append(formatted)

                formatted_page["columns"].append(
                    {
                        "cells": formatted_column,
                        "is_batch_column": column.get("is_batch_column", False),
                        "is_species_column": column.get("is_species_column", False),
                        "is_sexe_column": column.get("is_sexe_column", False),
                        "is_age_column": column.get("is_age_column", False),
                        "is_jour-mois_column": column.get("is_jour-mois_column", False),
                        "is_heure_column": column.get("is_heure_column", False),
                        "is_alle_column": column.get("is_alle_column", False),
                        "is_poids_column": column.get("is_poids_column", False),
                    }
                )

            output.append(formatted_page)

        print("\nCell formatter finished (no inversion).\n")
        return output
