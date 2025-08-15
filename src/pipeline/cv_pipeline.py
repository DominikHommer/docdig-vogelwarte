import os
import cv2
import fleep
from pdf2image import convert_from_path

from .pipeline import Pipeline

class CVPipeline(Pipeline):
    """
    Stellt eine modulare Pipeline zusammen, in der verschiedene Verarbeitungsschritte
    (Klassen mit einer process()-Methode) sequentiell ausgeführt werden.
    """
    def __init__(self, input_data: dict = {}):
        super().__init__(input_data)
    
    def run_and_extract(self, input_path: str):
        return self.run(input_path)