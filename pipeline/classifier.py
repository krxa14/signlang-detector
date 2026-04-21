"""Crop -> ensemble prediction."""
from __future__ import annotations

import numpy as np

from models.ensemble import Ensemble


class Classifier:
    def __init__(self, ensemble: Ensemble | None = None):
        self.ensemble = ensemble or Ensemble()
        self.ensemble.load_all()

    def classify(self, frame_bgr: np.ndarray) -> dict:
        res = self.ensemble.predict(frame_bgr)
        return {
            "label": res["label"],
            "confidence": res["confidence"],
            "model_breakdown": res["per_model"],
            "detection": res.get("detection"),
            "detections": res.get("detections", []),
        }
