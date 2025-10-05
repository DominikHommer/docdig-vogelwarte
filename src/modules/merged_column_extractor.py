import os
import cv2
import numpy as np
import shutil
import math
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from libs.cv_helpers import getYStartEndForLine
from .module_base import Module

class MergedColumnExtractorResult:
    columns_rgb: list[np.ndarray]
    columns_gray: list[np.ndarray]
    split_widths: list[float]

class MergedColumnExtractor(Module):
    def __init__(self,
                 min_page_occurrence_ratio: float = 0.4,  # Spalten müssen auf mindestens % der Seiten vorkommen
                 clustering_tolerance: int = 30,  # Pixel-Toleranz für Spalten-Clustering
                 xThresh: int = 60,  # Minimale Breite einer Spalte
                 debug: bool = False,
                 debug_folder: str = "debug/debug_merged_column_extractor/"):
        super().__init__("merged-column-extractor")
        self.debug = debug
        self.debug_folder = debug_folder
        
        self.min_page_occurrence_ratio = min_page_occurrence_ratio
        self.clustering_tolerance = clustering_tolerance
        self.xThresh = xThresh
        
        self.fld_distance_threshold = 1.414213562
        self.fld_canny_th1 = 50.0
        self.fld_canny_th2 = 50.0
        self.fld_canny_aperture_size = 3
        self.fld_do_merge = False

        if self.debug:
            if os.path.exists(self.debug_folder):
                shutil.rmtree(self.debug_folder)
            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['tatr-extractor']
    
    def _detect_vertical_lines_fld(self, gray_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        height = gray_img.shape[0]
        length_threshold = int(height * 0.85)

        fld = cv2.ximgproc.createFastLineDetector(
            length_threshold=length_threshold,
            distance_threshold=self.fld_distance_threshold,
            canny_th1=self.fld_canny_th1,
            canny_th2=self.fld_canny_th2,
            canny_aperture_size=self.fld_canny_aperture_size,
            do_merge=self.fld_do_merge
        )
        
        lines = fld.detect(gray_img)
        
        if lines is None:
            return []
        
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Winkel der Linie
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            if 80 <= angle <= 100:
                vertical_lines.append((int(x1), int(y1), int(x2), int(y2)))
        
        return vertical_lines
    
    def _merge_nearby_lines(self, lines: List[Tuple[int, int, int, int]], tolerance: int) -> List[int]:
        if not lines:
            return []
        
        x_positions = [(line[0] + line[2]) // 2 for line in lines]
        x_positions.sort()
        
        merged = []
        current_group = [x_positions[0]]
        
        for x in x_positions[1:]:
            if x - current_group[-1] <= tolerance:
                current_group.append(x)
            else:
                merged.append(int(np.mean(current_group)))
                current_group = [x]
        
        if current_group:
            merged.append(int(np.mean(current_group)))
        
        return merged
    
    def _create_column_template(self, all_page_columns: List[List[int]], total_pages: int, mean_image_width: int) -> List[int]:
        column_occurrences = defaultdict(int)
        column_positions = defaultdict(list)
        
        for page_columns in all_page_columns:
            for col_x in page_columns:
                found_cluster = False
                for cluster_x in list(column_occurrences.keys()):
                    if abs(col_x - cluster_x) <= self.clustering_tolerance:
                        column_occurrences[cluster_x] += 1
                        column_positions[cluster_x].append(col_x)
                        found_cluster = True
                        break
                
                if not found_cluster:
                    column_occurrences[col_x] = 1
                    column_positions[col_x] = [col_x]
        
        min_occurrences = int(total_pages * self.min_page_occurrence_ratio)
        template_columns = []
        
        for cluster_x, count in column_occurrences.items():
            if count >= min_occurrences:
                avg_x = int(np.mean(column_positions[cluster_x]))
                template_columns.append(avg_x)
        
        template_columns.sort()
        
        if template_columns:
            filtered_columns = []
            for i in range(len(template_columns)):
                if i == 0:
                    # Erste Spalte: vom linken Rand bis zur ersten Grenze
                    width = template_columns[0]
                elif i == len(template_columns) - 1:
                    # Letzte Spalte: von letzter Grenze bis zum Bildrand
                    width = mean_image_width - template_columns[i]
                else:
                    # Mittlere Spalten
                    if len(filtered_columns) > 0:
                        width = template_columns[i] - filtered_columns[-1]
                    else:
                        width = template_columns[i]
                
                if width >= self.xThresh:
                    filtered_columns.append(template_columns[i])
                elif self.debug:
                    print(f"  Removed column boundary at x={template_columns[i]} (resulting column width {width} < {self.xThresh})")
            
            template_columns = filtered_columns
        
        if self.debug:
            print(f"Column Template Statistics:")
            print(f"  Total pages: {total_pages}")
            print(f"  Min occurrence ratio: {self.min_page_occurrence_ratio}")
            print(f"  Min occurrences needed: {min_occurrences}")
            print(f"  Min column width (xThresh): {self.xThresh}")
            print(f"  Final template columns: {len(template_columns)}")
            for i, col_x in enumerate(template_columns):
                print(f"    Column boundary {i}: x={col_x}")
        
        return template_columns
    
    def _apply_template_to_page(self, base_img: np.ndarray, gray_img: np.ndarray, 
                                template_columns: List[int]) -> Dict[str, Any]:
        if not template_columns:
            return {
                'columns_rgb': [],
                'columns_gray': [],
                'split_widths': [],
            }
        
        _, iWidth = gray_img.shape[:2]
        
        columns_rgb = []
        columns_gray = []
        split_widths = []
        
        # Füge erste Spalte hinzu (vom Rand bis zur ersten Template-Spalte)
        if template_columns[0] > 0:
            end_x = min(template_columns[0] + 5, iWidth)
            columns_rgb.append(base_img[:, 0:end_x])
            columns_gray.append(cv2.bitwise_not(gray_img[:, 0:end_x]))
            split_widths.append(end_x)
        
        # Spalten zwischen Template-Positionen
        for i in range(len(template_columns) - 1):
            start_x = max(0, template_columns[i] - 5)
            end_x = min(template_columns[i + 1] + 5, iWidth)
            
            if end_x - start_x > 0:
                columns_rgb.append(base_img[:, start_x:end_x])
                columns_gray.append(cv2.bitwise_not(gray_img[:, start_x:end_x]))
                split_widths.append(end_x - start_x)
        
        # Letzte Spalte (von letzter Template-Position bis zum Rand)
        if template_columns[-1] < iWidth:
            start_x = max(0, template_columns[-1] - 5)
            columns_rgb.append(base_img[:, start_x:iWidth])
            columns_gray.append(cv2.bitwise_not(gray_img[:, start_x:iWidth]))
            split_widths.append(iWidth - start_x)
        
        return {
            'columns_rgb': columns_rgb,
            'columns_gray': columns_gray,
            'split_widths': split_widths,
        }
    
    def process(self, data: dict, config: dict) -> list[str]:
        file_paths: list[str] = data['tatr-extractor']
        
        all_page_columns = []
        page_images = []
        image_widths = []  # Sammle Bildbreiten
        
        for page_i, path in enumerate(file_paths):
            base_img = cv2.imread(path, cv2.IMREAD_COLOR)
            gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            page_images.append((base_img, gray_img))
            image_widths.append(gray_img.shape[1])  # Speichere Bildbreite
            
            gray = cv2.bitwise_not(gray_img)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            iHeight, _ = thresh.shape[:2]
            iHeightPart = math.floor(iHeight / 8)
            if iHeightPart % 2 == 0:
                iHeightPart = iHeightPart + 1
            
            blur = cv2.GaussianBlur(thresh, (5, iHeightPart), 0)
            blur = cv2.GaussianBlur(blur, (3, iHeightPart), 0)
            blur = cv2.GaussianBlur(blur, (1, iHeightPart), 0)
            
            vertical_lines = self._detect_vertical_lines_fld(blur)
            column_positions = self._merge_nearby_lines(vertical_lines, self.clustering_tolerance)
            all_page_columns.append(column_positions)
            
            if self.debug:
                debug_img = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
                for line in vertical_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                for col_x in column_positions:
                    cv2.line(debug_img, (col_x, 0), (col_x, iHeight), (255, 0, 0), 3)
                
                cv2.imwrite(f"{self.debug_folder}/page_{page_i}_detected_lines.jpg", debug_img)
                print(f"Page {page_i}: Found {len(column_positions)} columns")
        
        # Berechne mittlere Bildbreite
        mean_image_width = int(np.mean(image_widths)) if image_widths else 0
        
        # Erstelle Template mit mittlerer Bildbreite
        template_columns = self._create_column_template(all_page_columns, len(file_paths), mean_image_width)
        
        if self.debug and template_columns:
            template_img = page_images[0][0].copy()
            iHeight = template_img.shape[0]
            for i, col_x in enumerate(template_columns):
                cv2.line(template_img, (col_x, 0), (col_x, iHeight), (0, 0, 255), 2)
                cv2.putText(template_img, f"C{i}", (col_x - 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(f"{self.debug_folder}/column_template.jpg", template_img)
        
        result = []
        for page_i, (base_img, gray_img) in enumerate(page_images):
            page_result = self._apply_template_to_page(base_img, gray_img, template_columns)
            result.append(page_result)
            
            if self.debug:
                for j, col_img in enumerate(page_result['columns_rgb']):
                    if len(col_img) > 0:
                        cv2.imwrite(f"{self.debug_folder}/page_{page_i}_column_{j}.jpg", col_img)
        
        return result
