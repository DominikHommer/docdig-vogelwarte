"""Tag extracted columns with their semantic role — schema-based.

Previous approach (fragile, caused "Bague appears as UI column 5"): every
column was classified independently by substring keyword matching on its
header OCR. Header OCR fails on several columns per page and short keywords
matched into the wrong columns.

Current approach: OCR all headers, measure all widths, then align the whole
column sequence against the predefined form schema
(config/column_schema.json) with a monotone assignment — see
libs/column_schema.py. Columns whose header is unreadable are carried by
their position between confidently matched anchors plus their width.
"""

from typing import Optional

import numpy as np
import pytesseract

from libs.column_schema import ROLE_TO_FLAG, assign_roles, flags_for_role, load_schema
from .module_base import Module


def crop_center(image, width, height):
    h, w = image.shape[:2]
    x_start = max((w - width) // 2, 0)
    y_start = max((h - height) // 2, 0)
    return image[y_start:y_start + height, x_start:x_start + width]


class DetectColumns(Module):
    def __init__(self, debug: bool = False, schema_path: Optional[str] = None):
        super().__init__("column-marker")
        self.debug = debug
        self.schema = load_schema(schema_path)

    def get_preconditions(self) -> list[str]:
        return ["row-extractor"]

    def _header_text(self, header_img) -> str:
        """OCR the header cell — centre crop first (cuts neighbour bleed),
        full cell as fallback; the longer plausible read wins."""
        try:
            cropped = crop_center(header_img, width=200, height=50)
            text_cropped = pytesseract.image_to_string(cropped, lang="fra").strip()
        except Exception:
            text_cropped = ""
        try:
            text_full = pytesseract.image_to_string(header_img, lang="fra").strip()
        except Exception:
            text_full = ""
        return text_full if len(text_full) >= len(text_cropped) else text_cropped

    def process(self, data: dict, config: dict) -> list[dict]:
        pages = data["row-extractor"]
        output = []

        for page_idx, page in enumerate(pages):
            processed_page = {"columns": []}

            columns = page["columns"]
            header_texts = []
            widths = []
            for column_cells in columns:
                if not column_cells or not isinstance(column_cells[0], np.ndarray):
                    header_texts.append("")
                    widths.append(0.0)
                    continue
                header = column_cells[0]
                header_texts.append(self._header_text(header))
                widths.append(float(header.shape[1]))

            roles = assign_roles(header_texts, widths, schema=self.schema)

            if self.debug:
                for i, (text, w, role) in enumerate(zip(header_texts, widths, roles)):
                    print(
                        f"[DetectColumns] page={page_idx} col={i}: "
                        f"width={w:.0f} header={text!r} -> {role or '-'}"
                    )

            assigned = [r for r in roles if r]
            print(
                f"[DetectColumns] Seite {page_idx}: {len(assigned)}/{len(ROLE_TO_FLAG)} "
                f"Rollen zugeordnet ({', '.join(assigned) if assigned else 'keine'})."
            )

            for column_cells, role in zip(columns, roles):
                flags = flags_for_role(role)

                if not column_cells or not isinstance(column_cells[0], np.ndarray):
                    processed_page["columns"].append({"cells": [], **flags})
                    continue

                # Convert ndarray to dicts. `image_raw` preserves the original
                # cell crop for models trained on non-denoised input.
                cell_dicts = [
                    {"image": img, "image_raw": img, "skip_ocr": False}
                    for img in column_cells
                ]
                processed_page["columns"].append({"cells": cell_dicts, **flags})

            output.append(processed_page)

        print("\nColumn Detector finished!\n")
        return output
