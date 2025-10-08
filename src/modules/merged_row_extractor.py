import os
import cv2
import numpy as np
import shutil
import math
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from libs.cv_helpers import getYStartEndForLine
from .module_base import Module

class MergedRowExtractorResult:
    cells: list[list[list[np.ndarray]]]  # [page][column][row]

class MergedRowExtractor(Module):
    def __init__(self,
                 min_page_occurrence_ratio: float = 0.4,  # Reihen müssen auf mindestens % der Seiten vorkommen
                 clustering_tolerance: int = 15,  # Pixel-Toleranz für Reihen-Clustering
                 min_row_height: int = 40,  # Minimale Höhe einer Reihe
                 debug: bool = False,
                 debug_folder: str = "debug/debug_merged_row_extractor/"):
        super().__init__("row-extractor")
        self.debug = debug
        self.debug_folder = debug_folder
        
        self.min_page_occurrence_ratio = min_page_occurrence_ratio
        self.clustering_tolerance = clustering_tolerance
        self.min_row_height = min_row_height
        
        # FastLineDetector Parameter für horizontale Linien
        self.fld_distance_threshold = 1.414213562
        self.fld_canny_th1 = 30.0
        self.fld_canny_th2 = 30.0
        self.fld_canny_aperture_size = 3
        self.fld_do_merge = False

        if self.debug:
            if os.path.exists(self.debug_folder):
                shutil.rmtree(self.debug_folder)
            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['merged-column-extractor', 'tatr-extractor']
    
    def _detect_horizontal_lines_fld(self, gray_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Verwendet FastLineDetector um horizontale Linien auf der gesamten Seite zu erkennen
        """
        _, width = gray_img.shape[:2]
        length_threshold = int(width * 0.5)  # Linien müssen mindestens 50% der Seitenbreite haben

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
        
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Winkel der Linie berechnen
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            # Horizontale Linien: Winkel nahe 0° oder 180°
            if angle <= 10 or (170 <= angle <= 190):
                horizontal_lines.append((int(x1), int(y1), int(x2), int(y2)))
        
        return horizontal_lines
    
    def _merge_nearby_lines(self, lines: List[Tuple[int, int, int, int]], tolerance: int) -> List[int]:
        """
        Führt nahe beieinander liegende horizontale Linien zusammen
        Gibt Y-Positionen zurück
        """
        if not lines:
            return []
        
        # Y-Positionen extrahieren (Mittelwert von y1 und y2)
        y_positions = [(line[1] + line[3]) // 2 for line in lines]
        y_positions.sort()
        
        merged = []
        current_group = [y_positions[0]]
        
        for y in y_positions[1:]:
            if y - current_group[-1] <= tolerance:
                current_group.append(y)
            else:
                merged.append(int(np.mean(current_group)))
                current_group = [y]
        
        if current_group:
            merged.append(int(np.mean(current_group)))
        
        return merged
    
    def _create_row_template(self, all_page_rows: List[List[int]], total_pages: int, mean_image_height: int) -> List[int]:
        """
        Erstellt ein globales Reihen-Template basierend auf allen Seiten
        """
        row_occurrences = defaultdict(int)
        row_positions = defaultdict(list)
        
        # Sammle und clustere Reihen-Positionen von allen Seiten
        for page_rows in all_page_rows:
            for row_y in page_rows:
                found_cluster = False
                for cluster_y in list(row_occurrences.keys()):
                    if abs(row_y - cluster_y) <= self.clustering_tolerance:
                        row_occurrences[cluster_y] += 1
                        row_positions[cluster_y].append(row_y)
                        found_cluster = True
                        break
                
                if not found_cluster:
                    row_occurrences[row_y] = 1
                    row_positions[row_y] = [row_y]
        
        # Filtere Reihen basierend auf Mindest-Vorkommen
        min_occurrences = int(total_pages * self.min_page_occurrence_ratio)
        template_rows = []
        
        for cluster_y, count in row_occurrences.items():
            if count >= min_occurrences:
                avg_y = int(np.mean(row_positions[cluster_y]))
                template_rows.append(avg_y)
        
        template_rows.sort()
        
        # Filtere zu kleine Reihen raus
        if template_rows:
            filtered_rows = []
            for i in range(len(template_rows)):
                if i == 0:
                    # Erste Reihe: vom oberen Rand bis zur ersten Grenze
                    height = template_rows[0]
                elif i == len(template_rows) - 1:
                    # Letzte Reihe: von letzter Grenze bis zum Bildrand
                    height = mean_image_height - template_rows[i]
                else:
                    # Mittlere Reihen
                    if len(filtered_rows) > 0:
                        height = template_rows[i] - filtered_rows[-1]
                    else:
                        height = template_rows[i]
                
                if height >= self.min_row_height:
                    filtered_rows.append(template_rows[i])
                elif self.debug:
                    print(f"  Removed row boundary at y={template_rows[i]} (resulting row height {height} < {self.min_row_height})")
            
            template_rows = filtered_rows
        
        if self.debug:
            print(f"Row Template Statistics:")
            print(f"  Total pages: {total_pages}")
            print(f"  Min occurrence ratio: {self.min_page_occurrence_ratio}")
            print(f"  Min occurrences needed: {min_occurrences}")
            print(f"  Min row height: {self.min_row_height}")
            print(f"  Final template rows: {len(template_rows)}")
            for i, row_y in enumerate(template_rows):
                print(f"    Row boundary {i}: y={row_y}")
        
        return template_rows
    
    def _apply_template_to_page(self, page_columns: Dict[str, Any], template_rows: List[int]) -> Dict[str, Any]:
        """
        Wendet das Reihen-Template auf eine Seite an und schneidet alle Spalten entsprechend zu
        """
        if not template_rows:
            return {
                'columns': [[] for _ in page_columns['columns_gray']]
            }
        
        cells = []
        
        # Für jede Spalte
        for col_idx, (col_rgb, col_gray) in enumerate(zip(page_columns['columns_rgb'], page_columns['columns_gray'])):
            column_cells = []
            
            if len(col_gray.shape) < 2:
                cells.append([])
                continue
            
            iHeight, _ = col_gray.shape[:2]
            
            # Erstelle Reihen-Boundaries mit Template
            row_boundaries = []
            
            # Füge erste Reihe hinzu (vom Rand bis zur ersten Template-Position)
            if template_rows[0] > 0:
                row_boundaries.append((0, min(template_rows[0] + 5, iHeight)))
            
            # Reihen zwischen Template-Positionen
            for i in range(len(template_rows) - 1):
                start_y = max(0, template_rows[i] - 5)
                end_y = min(template_rows[i + 1] + 5, iHeight)
                
                if end_y - start_y >= self.min_row_height:
                    row_boundaries.append((start_y, end_y))
            
            # Letzte Reihe (von letzter Template-Position bis zum Rand)
            if template_rows[-1] < iHeight:
                start_y = max(0, template_rows[-1] - 5)
                if iHeight - start_y >= self.min_row_height:
                    row_boundaries.append((start_y, iHeight))
            
            # Schneide Zellen aus
            for start_y, end_y in row_boundaries:
                # Verwende col_gray direkt (bereits invertiert vom column extractor)
                cell = cv2.bitwise_not(col_gray[start_y:end_y, :])
                column_cells.append(cell)
            
            cells.append(column_cells)
        
        return {'columns': cells}
    
    def process(self, data: dict, config: dict) -> list[str]:
        """
        Verarbeitet alle Seiten mit globalem Reihen-Template
        """
        column_pages: list = data.get('merged-column-extractor', [])
        file_paths: list[str] = data.get('tatr-extractor', [])
        
        if not column_pages or not file_paths:
            print("Warning: No input data found for MergedRowExtractor")
            return []
        
        all_page_rows = []
        image_heights = []
        
        # Phase 1: Erkenne horizontale Linien auf allen Seiten
        for page_i, path in enumerate(file_paths):
            gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            if gray_img is None:
                print(f"Warning: Could not read image {path}")
                continue
            
            image_heights.append(gray_img.shape[0])
            
            # Vorbereitung des Bildes für Linienerkennung
            gray = cv2.bitwise_not(gray_img)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            iHeight, iWidth = thresh.shape[:2]
            
            # Blur für bessere Linienerkennung (horizontal ausgerichtet)
            iWidthPart = math.floor(iWidth / 4)
            if iWidthPart % 2 == 0:
                iWidthPart = iWidthPart + 1
            
            blur = cv2.GaussianBlur(thresh, (iWidthPart, 5), 0)
            blur = cv2.GaussianBlur(blur, (iWidthPart, 3), 0)
            blur = cv2.GaussianBlur(blur, (iWidthPart, 1), 0)
            
            # Erkenne horizontale Linien
            horizontal_lines = self._detect_horizontal_lines_fld(blur)
            row_positions = self._merge_nearby_lines(horizontal_lines, self.clustering_tolerance)
            all_page_rows.append(row_positions)
            
            if self.debug:
                debug_img = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
                
                # Zeichne erkannte Linien
                for line in horizontal_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Zeichne gemergede Positionen
                for row_y in row_positions:
                    cv2.line(debug_img, (0, row_y), (iWidth, row_y), (255, 0, 0), 3)
                
                cv2.imwrite(f"{self.debug_folder}/page_{page_i}_detected_rows.jpg", debug_img)
                print(f"Page {page_i}: Found {len(row_positions)} rows")
        
        # Phase 2: Erstelle globales Reihen-Template
        mean_image_height = int(np.mean(image_heights)) if image_heights else 0
        template_rows = self._create_row_template(all_page_rows, len(file_paths), mean_image_height)
        
        # Debug: Visualisiere Template
        if self.debug and template_rows and file_paths:
            template_img = cv2.imread(file_paths[0], cv2.IMREAD_COLOR)
            if template_img is not None:
                iHeight, iWidth = template_img.shape[:2]
                for i, row_y in enumerate(template_rows):
                    cv2.line(template_img, (0, row_y), (iWidth, row_y), (0, 0, 255), 2)
                    cv2.putText(template_img, f"R{i}", (10, row_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite(f"{self.debug_folder}/_row_template.jpg", template_img)
        
        # Phase 3: Wende Template auf alle Seiten an
        results = []
        for page_i, page_columns in enumerate(column_pages):
            page_result = self._apply_template_to_page(page_columns, template_rows)
            results.append(page_result)
            
            # Debug: Speichere zugeschnittene Zellen
            if self.debug:
                for col_idx, column_cells in enumerate(page_result['columns']):
                    for row_idx, cell in enumerate(column_cells):
                        if len(cell) > 0:
                            cv2.imwrite(f"{self.debug_folder}/page_{page_i}_col_{col_idx}_row_{row_idx}.jpg", cell)
        
        if self.debug:
            print(f"\nMergedRowExtractor Summary:")
            print(f"  Processed {len(results)} pages")
            if results:
                print(f"  Columns per page: {len(results[0]['columns'])}")
                if results[0]['columns']:
                    print(f"  Rows per column: {len(results[0]['columns'][0])}")
        
        return results
