import os
import math
from PIL import Image
from vendor.table_transformer.src.inference import TableExtractionPipeline

class TableDetector:
    def __init__(
        self,
        det_config_path: str,
        det_model_path: str,
        str_config_path: str,
        str_model_path: str,
        device: str = "cpu",
        crops_folder: str = "crops",
        padding: int = 10
    ):
        self.det_config_path = det_config_path
        self.det_model_path = det_model_path
        self.str_config_path = str_config_path
        self.str_model_path = str_model_path
        self.device = device
        self.crops_folder = crops_folder
        self.padding = padding
        
        self.pipe = TableExtractionPipeline(
            det_config_path=self.det_config_path,
            det_model_path=self.det_model_path,
            det_device=self.device,
            str_config_path=self.str_config_path,
            str_model_path=self.str_model_path,
            str_device=self.device
        )
        os.makedirs(self.crops_folder, exist_ok=True)

    def detect_tables(self, img_path: str, output_folder: str):
        img = Image.open(img_path).convert("RGB")

        detections = self.pipe.detect(
            img=img,
            tokens=None,
            out_objects=True
        )

        obj_list = detections.get("objects", [])
        saved_files = []

        for i, obj in enumerate(obj_list, start=1):
            label = obj["label"]
            score = obj["score"]
            (xmin, ymin, xmax, ymax) = obj["bbox"]

            print(f"[INFO] Objekt #{i}: Label={label}, Score={score:.3f}, BBox={obj['bbox']}")

            left = max(0, math.floor(xmin) - self.padding)
            top = max(0, math.floor(ymin) - self.padding)
            right = min(img.width, math.ceil(xmax) + self.padding)
            bottom = min(img.height, math.ceil(ymax) + self.padding)

            cropped_img = img.crop((left, top, right, bottom))
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            crop_filename = f"{base_name}.jpg"
            crop_path = os.path.join(output_folder, crop_filename)
            cropped_img.save(crop_path)
            print(f"[INFO] → Gespeichert als {crop_path}")
            saved_files.append(crop_path)

        return saved_files

    def process_directory(self, input_folder: str):
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith(".jpg"):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, input_folder)
                    output_subfolder = os.path.join(self.crops_folder, relative_path)
                    os.makedirs(output_subfolder, exist_ok=True)
                    self.detect_tables(full_path, output_subfolder)
