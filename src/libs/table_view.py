"""Pure table-view logic for the editor UI — no Streamlit imports.

Everything the Streamlit app needs to turn pipeline predictions into an
editable DataFrame (and back into a CSV) lives here so it can be unit-tested
without a browser session.
"""

from __future__ import annotations

import base64
import csv
import io
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from libs.columns import COLUMN_FLAG_TO_LABEL, column_label

VISIBLE_COLUMN_ORDER = list(COLUMN_FLAG_TO_LABEL.values())


def visible_columns(page: dict) -> List[Tuple[int, str]]:
    """[(col_idx_in_predictions, label), ...] in display order.

    Every extracted column is included: semantically tagged columns get their
    form label, untagged ones show up as "Spalte N" at their scan position.
    Duplicate labels get a " (2)" suffix so DataFrame columns stay unique.
    """
    labelled = []
    seen: Dict[str, int] = {}
    for i, col in enumerate(page.get("columns", [])):
        label = column_label(col, index=i)
        if label in seen:
            seen[label] += 1
            label = f"{label} ({seen[label]})"
        else:
            seen[label] = 1
        labelled.append((i, label))
    order = {name: rank for rank, name in enumerate(VISIBLE_COLUMN_ORDER)}
    labelled.sort(key=lambda pair: (order.get(pair[1], 999), pair[0]))
    return labelled


def cell_image_data_uri(image, max_h: int = 36, max_w: int = 220) -> str:
    """Base64 thumbnail for `st.column_config.ImageColumn` ('' when unusable)."""
    if image is None:
        return ""
    try:
        import cv2

        arr = np.asarray(image)
        if arr.size == 0:
            return ""
        if arr.dtype != np.uint8:
            mx = float(arr.max()) if arr.size else 1.0
            if 0.0 <= float(arr.min()) and mx <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[:, :, :3]

        h, w = arr.shape[:2]
        scale = min(max_h / max(h, 1), max_w / max(w, 1), 1.0)
        if scale < 1.0:
            arr = cv2.resize(
                arr,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if not ok:
            return ""
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
    except Exception:
        return ""


def page_to_dataframe(
    page: dict, include_thumbnails: bool = True
) -> Tuple[pd.DataFrame, List[Tuple[int, str]]]:
    """Build (df, [(col_idx, label), ...]) for one page.

    ``include_thumbnails=True`` adds a 📷 image column before each value
    column. The CSV export MUST use ``include_thumbnails=False`` — base64
    image URIs do not belong in a data export.
    """
    columns = page.get("columns", [])
    visible = visible_columns(page)
    if not visible:
        return pd.DataFrame(), []

    max_data_rows = max((len(col.get("cells", [])) - 1) for col in columns) if columns else 0
    max_data_rows = max(max_data_rows, 0)

    ordered_keys: list[str] = []
    data: dict = {}

    for col_i, label in visible:
        if include_thumbnails:
            img_key = f"📷 {label}"
            ordered_keys.append(img_key)
            data[img_key] = []
        ordered_keys.append(label)
        data[label] = []

    for row_i in range(1, max_data_rows + 1):
        for col_i, label in visible:
            cells = columns[col_i].get("cells", [])
            cell = cells[row_i] if row_i < len(cells) else None
            value = (cell.get("erkannt") if cell else "") or ""
            data[label].append(value.strip())
            if include_thumbnails:
                img_key = f"📷 {label}"
                source = cell.get("image_raw") if cell else None
                if source is None and cell is not None:
                    source = cell.get("image")
                data[img_key].append(cell_image_data_uri(source))

    df = pd.DataFrame(data, columns=ordered_keys)
    df.index = pd.RangeIndex(start=1, stop=len(df) + 1, name="#")
    return df, visible


def compute_row_confidence(
    page: dict, visible: Sequence[Tuple[int, str]], num_rows: int
) -> List[Tuple[str, int, bool]]:
    """Per data row: (emoji, min_score, has_alternatives)."""
    result = []
    columns = page.get("columns", [])
    for row_offset in range(num_rows):
        row_scores = []
        has_alt = False
        for col_i, _ in visible:
            cells = columns[col_i].get("cells", []) if col_i < len(columns) else []
            cell = cells[row_offset + 1] if row_offset + 1 < len(cells) else None
            if cell is None:
                continue
            score = int(cell.get("score", -1) or -1)
            if score >= 0:
                row_scores.append(score)
            if cell.get("alternatives"):
                has_alt = True
        if not row_scores:
            result.append(("⬜", -1, has_alt))
        else:
            ms = min(row_scores)
            if ms >= 95:
                result.append(("🟢", ms, has_alt))
            elif ms >= 60:
                result.append(("🟡", ms, has_alt))
            else:
                result.append(("🔴", ms, has_alt))
    return result


def build_csv_bytes(predictions: Sequence[Optional[dict]]) -> bytes:
    """One CSV over all pages — value columns only, never thumbnails."""
    output = io.StringIO()
    writer = csv.writer(output, delimiter=";")
    for page_idx, page in enumerate(predictions):
        if page is None:
            continue
        df, visible = page_to_dataframe(page, include_thumbnails=False)
        writer.writerow([f"Seite {page_idx + 1}"])
        writer.writerow([label for _, label in visible])
        for _, row in df.iterrows():
            writer.writerow(row.tolist())
        writer.writerow([])
    return output.getvalue().encode("utf-8-sig")
