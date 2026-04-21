"""Weighted-vote ensemble across YOLO, CNN, and RF."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from data.dataset import get_class_mapping
from data.preprocess import HandKeypointExtractor
from models.cnn_model import CNNModel
from models.rf_model import RFModel
from models.yolo_model import YoloModel

DEFAULT_WEIGHTS = {"yolo": 0.5, "cnn": 0.3, "rf": 0.2}


class Ensemble:
    def __init__(self, weights: Optional[dict] = None, num_classes: int = 106):
        self.w = weights or DEFAULT_WEIGHTS
        self.num_classes = num_classes
        self.classes = get_class_mapping()
        self.yolo = YoloModel()
        self.cnn = CNNModel(num_classes=num_classes)
        self.rf = RFModel()
        self.kp = HandKeypointExtractor()
        self._loaded = {"yolo": False, "cnn": False, "rf": False}

    def load_all(self) -> dict:
        from models.yolo_model import DEFAULT_WEIGHTS as YW
        from models.cnn_model import DEFAULT_WEIGHTS as CW
        from models.rf_model import DEFAULT_WEIGHTS as RW
        if Path(YW).exists():
            self.yolo.load(YW)
            self._loaded["yolo"] = True
        if Path(CW).exists():
            self.cnn.load(CW)
            self._loaded["cnn"] = True
        if Path(RW).exists():
            self.rf.load(RW)
            self._loaded["rf"] = True
        return self._loaded

    def _vote(self, scores: dict) -> dict:
        agg = np.zeros(self.num_classes, dtype=np.float32)
        per_model = {}
        active_weight = 0.0
        for name, (cls_id, conf) in scores.items():
            if cls_id is None:
                continue
            w = self.w.get(name, 0.0)
            if 0 <= cls_id < self.num_classes:
                agg[cls_id] += w * conf
            per_model[name] = {"class_id": cls_id, "label": self.classes.get(cls_id, str(cls_id)), "confidence": conf}
            active_weight += w
        if agg.sum() == 0:
            return {"label": None, "class_id": None, "confidence": 0.0, "per_model": per_model}
        best = int(agg.argmax())
        denom = active_weight if active_weight > 0 else 1.0
        return {
            "label": self.classes.get(best, str(best)),
            "class_id": best,
            "confidence": float(agg[best] / denom),
            "per_model": per_model,
        }

    def predict(self, frame_bgr: np.ndarray) -> dict:
        """Run full ensemble on a BGR frame. Returns final label + breakdown + detections."""
        dets = self.yolo.predict(frame_bgr) if self._loaded["yolo"] else []
        if dets:
            # take most confident detection
            det = max(dets, key=lambda d: d["confidence"])
            x1, y1, x2, y2 = det["box"]
            crop = frame_bgr[max(0, y1):y2, max(0, x1):x2]
            yolo_score = (det["class_id"], det["confidence"])
        else:
            crop = frame_bgr
            yolo_score = (None, 0.0)
            det = None

        # CNN
        if self._loaded["cnn"] and crop.size > 0:
            cnn_out = self.cnn.predict(crop)
            cnn_score = (cnn_out["class_id"], cnn_out["confidence"])
        else:
            cnn_score = (None, 0.0)

        # RF via keypoints
        if self._loaded["rf"] and crop.size > 0:
            feats = self.kp.extract(crop)
            if feats is not None:
                rf_out = self.rf.predict(feats)
                rf_score = (rf_out["class_id"], rf_out["confidence"])
            else:
                rf_score = (None, 0.0)
        else:
            rf_score = (None, 0.0)

        result = self._vote({"yolo": yolo_score, "cnn": cnn_score, "rf": rf_score})
        result["detection"] = det
        result["detections"] = dets
        return result


if __name__ == "__main__":
    e = Ensemble()
    print("ensemble constructed; loaded:", e.load_all())
