from pipeline.cv_pipeline import CVPipeline
import os

from modules.pdf_converter import PdfConverter
from modules.tatr_extraction import TatrExtractor
from modules.table_rotator import TableRotator
from modules.column_extractor import ColumnExtractor
from modules.merged_column_extractor import MergedColumnExtractor
from modules.row_extractor import RowExtractor
from modules.cell_denoiser import CellDenoiser
from modules.cell_formatter import CellFormatter
from modules.quotation_mark_detector import QuotationMarkDetector
from modules.trocr import TrOCR
from modules.fuzzy_matching import FuzzyMatchingBirdNames
from modules.predictor_dummy import PredictorDummy
from modules.detect_columns import DetectColumns
from modules.reorder_columns import ReorderColumns

input_image_path = os.path.join("data", "input", "scan_1972_CdB_19_20231125163614-1-3.pdf")

input_data = {}
### Uncomment if extracted table structure images already exists
#tatr = []
#for i in range(25, 35):
#    tatr.append(f'data/input/tatr/page_{i}.jpg')
#
#input_data = {
#    'tatr-extractor': tatr
#}

pipeline = CVPipeline(input_data=input_data)

## Uncomment to convert pdf to jpgs and extract table structure
pipeline.add_stage(PdfConverter(debug=False))
pipeline.add_stage(TableRotator(debug=False))
pipeline.add_stage(TatrExtractor(debug=False))
pipeline.add_stage(MergedColumnExtractor(debug=True))
pipeline.add_stage(RowExtractor(debug=True, useFastLine=False))
#pipeline.add_stage(DetectColumns(debug=True))
#pipeline.add_stage(ReorderColumns())
#pipeline.add_stage(CellDenoiser(debug=True))
#pipeline.add_stage(CellFormatter())
#pipeline.add_stage(QuotationMarkDetector())
#pipeline.add_stage(TrOCR())
#pipeline.add_stage(FuzzyMatchingBirdNames())
out = pipeline.run(input_data=input_image_path)

print(len(out[0]['columns']))