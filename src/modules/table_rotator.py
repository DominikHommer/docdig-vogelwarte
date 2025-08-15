import os
import cv2
import numpy as np
import shutil
import math

from libs.cv_helpers import getVerticalLines, rotateImg
from .module_base import Module

class TableRotator(Module):
    def __init__(self,
                 output_folder: str = "data/input/rotated_pages/",
                 debug: bool = False,
                 debug_folder: str = "debug/debug_table_rotator/"):
        super().__init__("table-rotator")
        self.debug = debug
        self.debug_folder = debug_folder
        self.output_folder = output_folder

        self.xThres = 40
        self.minFoundLines = 3

        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

        os.makedirs(self.output_folder, exist_ok=True)
        if self.debug:
            if os.path.exists(self.debug_folder):
                shutil.rmtree(self.debug_folder)

            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['pdf-converter']

    def process(self, data: dict, config: dict) -> list[str]:
        file_paths: list[str] = data['pdf-converter']

        out = []
        for path in file_paths:
            base_img = cv2.imread(path, cv2.IMREAD_COLOR)
            gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            gray = cv2.bitwise_not(gray_img)

            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            iHeight, _ = thresh.shape[:2]
            iHeightPart = math.floor(iHeight / 8)
            if iHeightPart % 2 == 0:
                iHeightPart = iHeightPart + 1

            blur = cv2.GaussianBlur(thresh, (1, 31), 0)
            blur = cv2.GaussianBlur(blur, (3, 1), 0)
            blur = cv2.GaussianBlur(blur, (3, 21), 0)
            blur = cv2.GaussianBlur(blur, (3, 31), 0)
            blur = cv2.GaussianBlur(blur, (3, 1), 0)

            vl = getVerticalLines(blur, self.xThres, self.minFoundLines)
            if not vl:
                return

            img_rotated = rotateImg(base_img, vl)
            out.append(img_rotated)

        result = {
            'file_paths': [],
            'imgs': [],
        }

        for i, img in enumerate(out):
            path = f"{os.path.dirname(self.output_folder)}/page_{i}.jpg"
            cv2.imwrite(path, img)

            result['file_paths'].append(path)
            result['imgs'].append(img)

            if self.debug:
                cv2.imwrite(f"{os.path.dirname(self.debug_folder)}/page_{i}.jpg", img)
                print(f"[TableRotator] rotated page {i}")

        return result

