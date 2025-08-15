import os
import cv2
import numpy as np
import shutil
import math
import statistics

from libs.cv_helpers import getVerticalLines
from .module_base import Module

import numpy as np
from statistics import median
from typing import List, Dict, Any

def compute_mean_widths(all_widths: List[List[float]]) -> np.ndarray:
    # Matrix mit NaNs auffüllen, um Spalten-Mittel zu berechnen
    n_pages = len(all_widths)
    max_len = max(len(w) for w in all_widths)
    mat = np.full((n_pages, max_len), np.nan, dtype=float)
    for i, w in enumerate(all_widths):
        mat[i, :len(w)] = w
    return np.nanmean(mat, axis=0)

def unify_page_columns(
    page: Dict[str, Any],
    mean_widths: np.ndarray,
    target_n: int,
    width_tolerance: float = 0.5
) -> Dict[str, Any]:
    """
    Ordnet jede Ziel-Spalte (0..target_n-1) der Page zu, indem
    die Spalte mit der nächstliegenden Breite ausgewählt wird.
    Liegt der Unterschied über (mean_widths[j] * width_tolerance),
    wird eine leere Spalte eingefügt.
    """
    w_rgb   = page['columns_rgb']
    w_gray  = page['columns_gray']
    w_width = page['split_widths']

    used_idxs = set()
    new_rgb, new_gray, new_widths = [], [], []

    for j in range(target_n):
        # Differenzen nur für nicht-verwendete Spalten
        diffs = [
            abs(w_width[k] - mean_widths[j]) if k not in used_idxs else np.inf
            for k in range(len(w_width))
        ]
        best_k = int(np.argmin(diffs))
        if diffs[best_k] <= mean_widths[j] * width_tolerance:
            used_idxs.add(best_k)
            new_rgb.append(w_rgb[best_k])
            new_gray.append(w_gray[best_k])
            new_widths.append(w_width[best_k])
        else:
            # Kein passender Fit → leere Spalte
            new_rgb.append([])
            new_gray.append([])
            new_widths.append([])

    return {
        'columns_rgb':   new_rgb,
        'columns_gray':  new_gray,
        'split_widths':  new_widths,
    }

def unify_all_pages(result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 1) Alle Breiten sammeln
    all_widths = [page['split_widths'] for page in result]

    # 2) Mean-Widths und Ziel-Anzahl bestimmen
    mean_widths = compute_mean_widths(all_widths)
    target_n    = int(median(len(w) for w in all_widths))

    # 3) Jede Page vereinheitlichen
    return [
        unify_page_columns(page, mean_widths, target_n)
        for page in result
    ]


class ColumnExtractorResult:
    columns_rgb: list[np.ndarray]
    columns_gray: list[np.ndarray]
    split_widths: list[float]

class ColumnExtractor(Module):
    def __init__(self,
                 minFoundColumns: int = 5,
                 try_experimental_unify: bool = False,
                 debug: bool = False,
                 debug_folder: str = "debug/debug_column_extractor/"):
        super().__init__("column-extractor")
        self.debug = debug
        self.debug_folder = debug_folder

        self.try_experimental_unify = try_experimental_unify
        self.xThres = 40
        self.minFoundLines = 2
        self.minFoundColumns = minFoundColumns

        if self.debug:
            if os.path.exists(self.debug_folder):
                shutil.rmtree(self.debug_folder)

            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['tatr-extractor']

    def process(self, data: dict, config: dict) -> list[str]:
        file_paths: list[str] = data['tatr-extractor']

        result: list[ColumnExtractorResult] = []
        for page_i, path in enumerate(file_paths):
            base_img = cv2.imread(path, cv2.IMREAD_COLOR)
            gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            gray = cv2.bitwise_not(gray_img)

            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            iHeight, iWidth = thresh.shape[:2]
            iHeightPart = math.floor(iHeight / 8)
            if iHeightPart % 2 == 0:
                iHeightPart = iHeightPart + 1

            blur = cv2.GaussianBlur(thresh, (5, iHeightPart), 0)
            blur = cv2.GaussianBlur(blur, (3, iHeightPart), 0)
            blur = cv2.GaussianBlur(blur, (1, iHeightPart), 0)

            threshRGB = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
            threshCopy = np.copy(threshRGB)

            verticalLines = getVerticalLines(blur, self.xThres, self.minFoundLines)

            # Skip image if no vertical lines found
            if not verticalLines or len(verticalLines) < self.minFoundColumns:
                page: ColumnExtractorResult = {
                    'columns_rgb': [],
                    'columns_gray': [],
                    'split_widths': [],
                }

                result.append(page)

                continue
            
            xSplits = [[0, 0]]

            for i in range(len(verticalLines)):
                vLine = verticalLines[i]
                x1, y1, x2, y2 = vLine[0]

                # This basically gets the maximum (possible) size of the column
                start = 0
                if (i > 0):
                    pLine = verticalLines[i - 1]

                    px1 = pLine[0][0]
                    px2 = pLine[0][2]

                    start = px1
                    if (px2 < px1):
                        start = px2

                end = x1
                if (x2 > end):
                    end = x2

                xSplits.append([start, end])
                if self.debug:
                    cv2.line(threshCopy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self.debug:
                cv2.imwrite(f"{os.path.dirname(self.debug_folder)}/page_{page_i}_columns.jpg", threshCopy)

            xSplits.append([xSplits[-1][-1], iWidth])
            xSplits = np.sort(xSplits, axis=0)

            doneSplits = []
            doneSplitsRGB = []
            splitWidths = []
            for splits in xSplits:
                start, end = splits

                if start == 0:
                    continue

                if end - start <= self.xThres:
                    continue

                splitWidths.append(end - start)
                doneSplits.append(thresh[:, start:end + 15])
                doneSplitsRGB.append(base_img[:, start:end + 15])

            page: ColumnExtractorResult = {
                'columns_rgb': doneSplitsRGB,
                'columns_gray': doneSplits,
                'split_widths': splitWidths,
            }

            result.append(page)

            if self.debug:
                for j, col_img in enumerate(doneSplitsRGB):
                    cv2.imwrite(f"{os.path.dirname(self.debug_folder)}/page_{page_i}_column_{j}.jpg", col_img)

        if not self.try_experimental_unify:
            return result

        # Do we really need this?
        # Maybe we can map after transcription based on the header
        return unify_all_pages(result)

