"""Table-border removal for cell crops.

Cell crops from the row/column extractors often carry slivers of the printed
table grid: a vertical rule at the left/right edge (which digit recognizers
love to read as a "1") or a horizontal rule at top/bottom. This module strips
those *before* any recognizer runs.

Approach (standard OCR preprocessing): binarise, extract long thin lines with
elongated morphological structuring elements, and whiten them out. Two
safeguards keep real content intact:

- Only *near-full-height* vertical strokes count as table lines (a
  handwritten "1" is tall, but not as tall as the crop itself).
- Only strokes close to the cell edge are removed — a table rule bleeding
  into the crop sits at the margin, handwriting sits in the middle.
"""

from typing import Optional

import cv2
import numpy as np


def _to_uint8_gray(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.size == 0:
        return None
    arr = np.nan_to_num(arr, nan=255.0, posinf=255.0, neginf=0.0)
    if arr.ndim == 3:
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        if arr.shape[-1] == 3:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if arr.ndim != 2:
        return None
    if arr.dtype != np.uint8:
        mx = float(arr.max()) if arr.size else 1.0
        mn = float(arr.min()) if arr.size else 0.0
        if 0.0 <= mn and mx <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def strip_cell_borders(
    image,
    vertical_min_height: float = 0.72,
    horizontal_min_width: float = 0.55,
    edge_zone: float = 0.22,
    dilate_px: int = 2,
):
    """Whiten table-grid lines in a cell crop; return the cleaned grayscale.

    Args:
        image: cell crop (grayscale or RGB, uint8 or float).
        vertical_min_height: a vertical stroke must span at least this
            fraction of the crop height to count as a table line. Keep well
            above typical digit height (a handwritten "1" is ~40–60%).
        horizontal_min_width: same for horizontal strokes vs. crop width.
        edge_zone: only strokes whose centre lies within this fraction from
            the left/right (vertical) or top/bottom (horizontal) edge are
            removed. Content in the middle is never touched.
        dilate_px: widen the line mask to catch anti-aliasing halos.

    Returns the cleaned grayscale image (uint8), or the input coerced to
    grayscale when nothing had to be removed. Returns None for unusable input.
    """
    gray = _to_uint8_gray(image)
    if gray is None:
        return None

    h, w = gray.shape[:2]
    if h < 12 or w < 12:
        return gray

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask = np.zeros_like(binary)

    # --- vertical table rules -------------------------------------------
    v_len = max(3, int(round(h * vertical_min_height)))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    if v_lines.any():
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(v_lines)
        for i in range(1, num):
            cx = centroids[i][0]
            if cx <= w * edge_zone or cx >= w * (1.0 - edge_zone):
                mask[labels == i] = 255

    # --- horizontal table rules ------------------------------------------
    h_len = max(3, int(round(w * horizontal_min_width)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    if h_lines.any():
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(h_lines)
        for i in range(1, num):
            cy = centroids[i][1]
            if cy <= h * edge_zone or cy >= h * (1.0 - edge_zone):
                mask[labels == i] = 255

    if not mask.any():
        return gray

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)

    cleaned = gray.copy()
    cleaned[mask > 0] = 255
    return cleaned
