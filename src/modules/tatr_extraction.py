import os
import shutil
from .module_base import Module
from libs.table_detector import TableDetector

class TatrExtractor(Module):
    def __init__(self,
                 output_folder: str = "data/input/tatr/",
                 debug: bool = False,
                 debug_folder: str = "debug/debug_tatr_extractor/"):
        super().__init__("tatr-extractor")
        self.debug = debug
        self.debug_folder = debug_folder
        self.output_folder = output_folder

        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

        os.makedirs(self.output_folder, exist_ok=True)
        if self.debug:
            if os.path.exists(self.debug_folder):
                shutil.rmtree(self.debug_folder)

            os.makedirs(self.debug_folder, exist_ok=True)

    def get_preconditions(self) -> list[str]:
        return ['table-rotator']

    def process(self, data: dict, config: dict) -> list[str]:
        rotator_data: dict = data['table-rotator']

        file_paths = rotator_data['file_paths']

        detector = TableDetector(
            config['tatr']['detection_config'],
            config['tatr']['detection_model'],
            config['tatr']['structure_config'],
            config['tatr']['structure_model']
        )

        out = []
        for path in file_paths:
            saved_crops = detector.detect_tables(path, self.output_folder)

            out.append(saved_crops[0])
        
        return out

