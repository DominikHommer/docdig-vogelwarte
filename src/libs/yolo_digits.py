"""YOLO-based digit reader (third voice of the numeric ensemble).

Wraps the YOLOv8 detector trained on single handwritten digits
(10 classes, "0"–"9"; see /config/digit_yolo.pt, trained in
ml_spielereien/runs/detect/train5). A cell crop goes in, every detected
digit comes back as a box; we order the boxes left-to-right and join the
class labels into the number string.

The composition logic (`compose_detections`) is a pure function over plain
arrays so it can be unit-tested without ultralytics or the model file.
The model itself is loaded lazily — a missing checkpoint or a missing
`ultralytics` install turns the reader into a no-op.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MODEL_PATHS = [
    "./config/digit_yolo.pt",
    "./config/digit/digit_yolo.pt",
]


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compose_detections(
    boxes: Sequence[Sequence[float]],
    classes: Sequence[int],
    confidences: Sequence[float],
    conf_threshold: float = 0.30,
    overlap_iou: float = 0.45,
) -> Tuple[str, float]:
    """Turn per-digit detections into a number string.

    Args:
        boxes: xyxy boxes, one per detection.
        classes: class index per detection (0–9 == the digit itself).
        confidences: detector confidence per detection.
        conf_threshold: detections below this are dropped.
        overlap_iou: when two *different-class* boxes overlap more than this,
            only the more confident one survives. (YOLO's built-in NMS is
            per-class, so a "1" box and a "7" box can sit on the same stroke.)

    Returns:
        (digits, confidence) — the left-to-right digit string and the mean
        confidence of the surviving detections. ("", 0.0) when nothing
        survives.
    """
    kept: List[Tuple[Tuple[float, float, float, float], int, float]] = []
    for box, cls, conf in zip(boxes, classes, confidences):
        if conf < conf_threshold:
            continue
        if not (0 <= int(cls) <= 9):
            continue
        kept.append((tuple(float(v) for v in box), int(cls), float(conf)))

    if not kept:
        return "", 0.0

    # Cross-class de-duplication: strongest detection wins its territory.
    kept.sort(key=lambda item: item[2], reverse=True)
    surviving: List[Tuple[Tuple[float, float, float, float], int, float]] = []
    for candidate in kept:
        if any(_iou(candidate[0], other[0]) > overlap_iou for other in surviving):
            continue
        surviving.append(candidate)

    # Left-to-right reading order by x-centre.
    surviving.sort(key=lambda item: (item[0][0] + item[0][2]) / 2.0)

    digits = "".join(str(cls) for _, cls, _ in surviving)
    confidence = float(np.mean([conf for _, _, conf in surviving]))
    return digits, confidence


class YoloDigitReader:
    """Lazy wrapper around the trained YOLOv8 digit detector."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.30,
        overlap_iou: float = 0.45,
        imgsz: int = 640,
        debug: bool = False,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.overlap_iou = overlap_iou
        self.imgsz = imgsz
        self.debug = debug

        self._model = None
        self._available: Optional[bool] = None

    def _resolve_model_path(self) -> Optional[str]:
        import os

        if self.model_path and os.path.exists(self.model_path):
            return self.model_path
        for path in DEFAULT_MODEL_PATHS:
            if os.path.exists(path):
                return path
        return None

    def ensure_loaded(self) -> bool:
        if self._available is not None:
            return self._available

        model_path = self._resolve_model_path()
        if not model_path:
            if self.debug:
                print("[YoloDigitReader] No digit_yolo.pt found — YOLO voice disabled.")
            self._available = False
            return False

        try:
            from ultralytics import YOLO  # noqa: WPS433 (lazy import by design)
        except ImportError as e:
            print(f"[YoloDigitReader] ultralytics not installed: {e}")
            self._available = False
            return False

        try:
            self._model = YOLO(model_path)
            self._available = True
            print(f"[YoloDigitReader] Loaded YOLO digit model from {model_path}.")
            return True
        except Exception as e:
            print(f"[YoloDigitReader] Failed to load YOLO model: {e}")
            self._available = False
            return False

    @staticmethod
    def _to_grayscale(image) -> Optional[np.ndarray]:
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.size == 0:
            return None
        arr = np.nan_to_num(arr, nan=255.0, posinf=255.0, neginf=0.0)
        if arr.dtype != np.uint8:
            mx = float(arr.max()) if arr.size else 1.0
            mn = float(arr.min()) if arr.size else 0.0
            if 0.0 <= mn and mx <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3:
            if arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            if arr.shape[-1] == 3:
                import cv2

                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if arr.ndim != 2:
            return None
        return arr

    @staticmethod
    def _prepare(image, canvas: int = 640, target_ink_height: int = 140) -> Optional[np.ndarray]:
        """Match the model's training distribution.

        The detector was trained on 640x640 images of WHITE digits on a BLACK
        background, digit height ~0.22 of the canvas (see
        ml_spielereien/datasets/handwritten_dataset). Cell crops from the
        pipeline are the opposite: small, dark ink on white paper, with table
        borders. Convert: strip borders, binarise + invert, scale the ink to
        the training digit height, paste centred on a black canvas.
        """
        import cv2

        gray = YoloDigitReader._to_grayscale(image)
        if gray is None:
            return None

        h, w = gray.shape[:2]
        if h < 8 or w < 8:
            return None

        # Strip the printed table borders around the cell.
        margin_y = max(2, h // 8)
        margin_x = max(2, w // 16)
        inner = gray[margin_y : h - margin_y, margin_x : w - margin_x]
        if inner.size == 0:
            return None

        # Ink -> white on black, like the training data.
        _, binary = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        ys, xs = np.nonzero(binary)
        if len(ys) < 10:  # essentially blank
            return None
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        ink = binary[y0:y1, x0:x1]

        ink_h, ink_w = ink.shape[:2]
        scale = target_ink_height / max(ink_h, 1)
        # Keep the strip inside the canvas (long ring numbers are wide).
        scale = min(scale, (canvas - 40) / max(ink_w, 1), 16.0)
        if scale <= 0:
            return None
        ink = cv2.resize(
            ink,
            (max(1, int(round(ink_w * scale))), max(1, int(round(ink_h * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )

        result = np.zeros((canvas, canvas), dtype=np.uint8)
        ih, iw = ink.shape[:2]
        oy = max(0, (canvas - ih) // 2)
        ox = max(0, (canvas - iw) // 2)
        result[oy : oy + ih, ox : ox + iw] = ink[: canvas - oy, : canvas - ox]
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def read(self, image) -> Tuple[str, float]:
        """Detect digits in a cell crop. Returns ("", 0.0) when unavailable."""
        if not self.ensure_loaded():
            return "", 0.0
        bgr = self._prepare(image)
        if bgr is None:
            return "", 0.0
        try:
            results = self._model.predict(
                bgr,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                verbose=False,
            )
        except Exception as e:
            if self.debug:
                print(f"[YoloDigitReader] inference error: {e}")
            return "", 0.0

        if not results:
            return "", 0.0
        boxes_obj = getattr(results[0], "boxes", None)
        if boxes_obj is None or boxes_obj.xyxy is None or len(boxes_obj) == 0:
            return "", 0.0

        boxes = boxes_obj.xyxy.cpu().numpy().tolist()
        classes = boxes_obj.cls.cpu().numpy().astype(int).tolist()
        confs = boxes_obj.conf.cpu().numpy().tolist()

        return compose_detections(
            boxes,
            classes,
            confs,
            conf_threshold=self.conf_threshold,
            overlap_iou=self.overlap_iou,
        )
