"""Frame -> YOLO detections with crops."""
from __future__ import annotations

from typing import List

import numpy as np

from models.yolo_model import YoloModel


class Detector:
    def __init__(self, model: YoloModel | None = None):
        self.model = model or YoloModel()

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.25) -> List[dict]:
        dets = self.model.predict(frame_bgr, conf=conf)
        out = []
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            crop = frame_bgr[max(0, y1):y2, max(0, x1):x2]
            out.append({**d, "crop": crop})
        return out
