"""YOLOv8 detection wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "yolo_best.pt"


class YoloModel:
    def __init__(self, weights: Optional[Path] = None):
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        if weights and Path(weights).exists():
            self.model = YOLO(str(weights))
        else:
            base = PROJECT_ROOT / "yolov8n.pt"
            self.model = YOLO(str(base) if base.exists() else "yolov8n.pt")

    def train(self, data_yaml: Path, epochs: int = 5, imgsz: int = 320, batch: int = 16, fraction: float = 0.2) -> dict:
        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            fraction=fraction,
            device="cpu",
            project=str(PROJECT_ROOT / "runs"),
            name="yolo_train",
            exist_ok=True,
            verbose=True,
            patience=3,
            workers=0,
        )
        # copy best to stable location
        best = Path(self.model.trainer.best) if hasattr(self.model, "trainer") else None
        if best and best.exists():
            import shutil
            shutil.copy(best, DEFAULT_WEIGHTS)
        return {"best": str(DEFAULT_WEIGHTS)}

    def predict(self, frame: np.ndarray, conf: float = 0.25) -> List[dict]:
        res = self.model.predict(frame, conf=conf, device="cpu", verbose=False)
        out: List[dict] = []
        if not res:
            return out
        r = res[0]
        names = r.names
        if r.boxes is None:
            return out
        for box, cls, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box.tolist()
            cls_i = int(cls)
            out.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "class_id": cls_i,
                "label": names.get(cls_i, str(cls_i)),
                "confidence": float(c),
            })
        return out

    def load(self, path: Path):
        self.model = YOLO(str(path))


if __name__ == "__main__":
    m = YoloModel()
    print("yolo loaded")
