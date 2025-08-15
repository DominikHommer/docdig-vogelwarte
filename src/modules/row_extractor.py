import os
import cv2
import numpy as np
import shutil
import math
import statistics

from libs.cv_helpers import getYStartEndForLine
from .module_base import Module

class RowExtractorResult:
    columns: list[list[np.ndarray]]

class RowExtractor(Module):
    def __init__(self,
                 debug: bool = False,
                 debug_folder: str = "debug/debug_row_extractor/"):
        super().__init__("row-extractor") 
        self.debug = debug
        self.debug_folder = debug_folder

        self.xThres = 40
        self.minFoundLines = 2

        if self.debug:
            if os.path.exists(self.debug_folder):
                shutil.rmtree(self.debug_folder)

            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['column-extractor']

    def process(self, data: dict, config: dict) -> list[str]:
        pages: list = data['column-extractor']

        results = []
        for page_i, page in enumerate(pages):
            page_data: RowExtractorResult = {
                "columns": [list() for _ in range(len(page['columns_gray']))]
            }

            for col_nr, col in enumerate(page['columns_gray']):
                copyTest = np.copy(col)
                copyRGB = np.copy(col)

                if len(copyTest.shape) < 2:
                    page_data['columns'][col_nr] = []

                    continue

                iHeight, iWidth = copyTest.shape[:2]

                copyTest = cv2.bitwise_not(copyTest)

                if iWidth % 2 == 0:
                    iWidth = iWidth + 1

                iWidthPart = math.floor(iWidth / 4)
                if iWidthPart % 2 == 0:
                    iWidthPart = iWidthPart + 1

                colBlur = cv2.GaussianBlur(copyTest, (iWidth, 5), 0)
                colBlur = cv2.GaussianBlur(copyTest, (iWidth, 3), 0)
                colBlur = cv2.GaussianBlur(copyTest, (iWidth, 1), 0)

                colDenoised = cv2.fastNlMeansDenoising(colBlur, None, 30)
                colCanny = cv2.Canny(colDenoised, 10, 200)

                houghThreshold = 100
                minLineLength = 35

                # Small cells should have "smaller" thresholds to detect lines
                if iWidth < 200:
                    houghThreshold = 50
                    minLineLength = 10

                lines = cv2.HoughLinesP(colCanny, 1, np.pi / 180, threshold=houghThreshold, minLineLength=minLineLength, maxLineGap=2)

                # Skip image if no horizontal lines found
                if lines is None:
                    page_data['columns'][col_nr] = []

                    continue

                # We define a 30px y-Threshold, which "decides" if a line is indeed a horizontal detected line
                yThres = 30
                horizontalLines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    # Skip value, as it is not a horizontal line
                    if not ((y1 + yThres) > y2 and (y1 - yThres) < y2):
                        continue

                    shouldAdd = True
                    for i in range(len(horizontalLines)):
                        hLine = horizontalLines[i]
                        h_x1, h_y1, h_x2, h_y2 = hLine[0]

                        # Check if "line" Vector is above / under "hLine" Vector and in its y-threshold
                        # Update horizontalLines List accordingly
                        if (x1 < h_x1) and ((h_y1 + yThres) > y1 and (h_y1 - yThres) < y1):
                            horizontalLines[i] = [[x1, y1, h_x2, h_y2], hLine[1]+1]

                            shouldAdd = False
                        elif (x2 > h_x2) and ((h_y2 + yThres) > y2 and (h_y2 - yThres) < y2):
                            horizontalLines[i] = [[h_x1, h_y1, x2, y2], hLine[1]+1]

                            shouldAdd = False

                    if shouldAdd == True:
                        horizontalLines.append([line[0], 0])

                minFoundLines = 1
                if iWidth < 200:
                    minFoundLines = 0

                ySplits = [[0, 0]]

                horizontalLines = list(filter(lambda l: l[1] >= minFoundLines, horizontalLines))

                for i in range(len(horizontalLines)):
                    hLine = horizontalLines[i]
                    x1, y1, x2, y2 = hLine[0]

                    # This basically gets the maximum (possible) height of the row
                    start, end = getYStartEndForLine(i, horizontalLines)
                    
                    ySplits.append([start, end])
                    cv2.line(copyRGB,(x1,y1),(x2,y2),(0,255,0),2)

                if self.debug:
                    cv2.imwrite(f"{os.path.dirname(self.debug_folder)}/page_{page_i}_column_{col_nr}_rows.jpg", copyRGB)

                ySplits.append([ySplits[-1][-1], iHeight])
                ySplits = np.sort(ySplits, axis=0)

                realSplits = []
                heights = []

                for splits in ySplits:
                    start, end = splits

                    if start == 0:
                        continue

                    height = end - start
                    if height <= yThres:
                        continue
                    
                    heights.append(height)

                heightMedian = statistics.median(heights)
                
                for splits in ySplits:
                    start, end = splits

                    if start == 0:
                        continue

                    height = end - start
                    if height <= yThres:
                        continue
                    
                    if height >= heightMedian * 1.75:
                        while height > heightMedian * 1.75:
                            newEnd = int(start + heightMedian)

                            realSplits.append([start, newEnd])

                            start = newEnd
                            height = end - start
                    
                        realSplits.append([start, end])
                    else:
                        realSplits.append(splits)

                ySplits = realSplits

                doneRowSplits = []
                for splits in ySplits:
                    start, end = splits

                    if start == 0:
                        continue

                    if end - start <= yThres:
                        continue

                    doneRowSplits.append(copyTest[start:end + 15,:])
                
                if self.debug:
                    for j, row_img in enumerate(doneRowSplits):
                        cv2.imwrite(f"{os.path.dirname(self.debug_folder)}/page_{page_i}_column_{col_nr}_row_{j}.jpg", row_img)

                page_data['columns'][col_nr] = doneRowSplits
            
            results.append(page_data)

        return results

