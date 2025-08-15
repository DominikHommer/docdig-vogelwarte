import os
import json
import numpy as np
import cv2
from typing import List
from tensorflow.keras.models import load_model
from .module_base import Module


class Predictor(Module):
    def __init__(
        self,
        model_path: str = "./config/classifier_model.keras",
        label_path: str = "./config/class_indices.json"
    ):
        super().__init__("predictor")
        self.model = load_model(model_path)
        self.target_size = (150, 22)

        self.class_labels = None
        if label_path and os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                name_to_idx = json.load(f)

            # Erzeuge korrektes Mapping: index → label
            idx_to_label = {int(v): k for k, v in name_to_idx.items()}
            # Sortiere nach Index (0 bis N-1)
            self.class_labels = [idx_to_label[i] for i in range(len(idx_to_label))]

    def get_preconditions(self) -> List[str]:
        return ['cell-denoiser']

    def process(self, data: dict, config: dict) -> List[dict]:
        denoised_pages = data['cell-denoiser']
        predictions = []

        for page in denoised_pages:
            page_prediction = {"columns": []}

            for col in page["columns"]:
                col_preds = []

                for img in col:
                    img_resized = cv2.resize(img, self.target_size)
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
                    img_resized = np.expand_dims(img_resized, axis=0)
                    img_resized = img_resized.astype("float32") / 255.0

                    pred = self.model.predict(img_resized)
                    pred_class = int(np.argmax(pred, axis=1)[0])

                    if self.class_labels:
                        label = self.class_labels[pred_class]
                        col_preds.append(label)
                    else:
                        col_preds.append(pred_class)

                page_prediction["columns"].append(col_preds)

            predictions.append(page_prediction)

        return predictions
