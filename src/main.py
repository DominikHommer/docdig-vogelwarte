from pipeline.cv_pipeline import CVPipeline
import os

from modules.pdf_converter import PdfConverter
from modules.tatr_extraction import TatrExtractor
from modules.table_rotator import TableRotator
from modules.merged_column_extractor import MergedColumnExtractor
from modules.merged_row_extractor import MergedRowExtractor
from modules.cell_formatter import CellFormatter
from modules.quotation_mark_detector import QuotationMarkDetector
from modules.htr_vt_recognizer import HtrVtRecognizer
from modules.digit_recognizer import DigitRecognizer
from modules.sexe_classifier import SexeClassifier
from modules.trocr import TrOCR
from modules.fuzzy_matching import FuzzyMatchingBirdNames, FuzzyMatchingAge
from modules.numeric_consensus import NumericConsensus
from modules.detect_columns import DetectColumns

input_image_path = os.path.join("data", "input", "scan_1972_CdB_19_20231125163614-1-3.pdf")

input_data = {}

pipeline = CVPipeline(input_data=input_data)

pipeline.add_stage(PdfConverter(debug=False))
pipeline.add_stage(TableRotator(debug=False))
pipeline.add_stage(TatrExtractor(debug=False))
pipeline.add_stage(MergedColumnExtractor(debug=False))
pipeline.add_stage(MergedRowExtractor(debug=False))
pipeline.add_stage(DetectColumns())
pipeline.add_stage(CellFormatter())
pipeline.add_stage(QuotationMarkDetector())
pipeline.add_stage(HtrVtRecognizer())
pipeline.add_stage(DigitRecognizer())
pipeline.add_stage(SexeClassifier())
pipeline.add_stage(TrOCR())                # second opinion for Espèce + Aile + Poids
pipeline.add_stage(FuzzyMatchingBirdNames())  # consensus + fuzzy for Espèce
pipeline.add_stage(NumericConsensus())     # consensus for Aile + Poids
pipeline.add_stage(FuzzyMatchingAge())

out = pipeline.run(input_data=input_image_path)

print(len(out[0]['columns']))
