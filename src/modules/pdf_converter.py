import os
import shutil
from .module_base import Module
import fleep
from pdf2image import convert_from_path

class PdfConverter(Module):
    def __init__(self,
                 output_folder: str = "data/input/pages/",
                 debug: bool = False,
                 debug_folder: str = "debug/debug_pdf_converter/"):
        super().__init__("pdf-converter")
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
        return ['input']

    def _is_pdf(self, file_path) -> bool:
        with open(file_path, "rb") as file:
            info = fleep.get(file.read(128))

            return info.extension_matches("pdf")
        
        return False
    
    def process(self, data: dict, config: dict) -> list[str]:
        input_path: str = data["input"]

        if self._is_pdf(input_path):
            images = convert_from_path(input_path)
            
            inputs = []
            for i, img in enumerate(images):
                path = f"{os.path.dirname(self.output_folder)}/page_{i}.jpg"
                img.save(path)

                if self.debug:
                    img.save(f"{os.path.dirname(self.debug_folder)}/page_{i}.jpg")
                    print(f"[PdfConverter] saved page {i}")

                inputs.append(path)

            return inputs
        
        return [input_path]

