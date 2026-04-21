"""Evaluate all three models + write results/evaluation.csv."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import DATASET_ROOT, DATA_YAML, get_class_mapping  # noqa: E402
from data.preprocess import HandKeypointExtractor  # noqa: E402
from models.cnn_model import CNNModel, DEFAULT_WEIGHTS as CNN_W  # noqa: E402
from models.rf_model import RFModel, DEFAULT_WEIGHTS as RF_W, KEYPOINT_CSV  # noqa: E402
from models.yolo_model import YoloModel, DEFAULT_WEIGHTS as YW  # noqa: E402

RESULTS_CSV = PROJECT_ROOT / "results" / "evaluation.csv"


def _test_samples(max_samples: int = 200):
    test_imgs = DATASET_ROOT / "test" / "images"
    test_lbls = DATASET_ROOT / "test" / "labels"
    samples = []
    lbl_stems = {p.stem: p for p in test_lbls.glob("*.txt")}
    for img in test_imgs.glob("*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = lbl_stems.get(img.stem)
        if not lbl:
            continue
        with open(lbl) as f:
            line = f.readline().strip().split()
        if not line:
            continue
        cls = int(line[0])
        xc, yc, w, h = map(float, line[1:5])
        samples.append((img, cls, (xc, yc, w, h)))
        if len(samples) >= max_samples:
            break
    return samples


def _crop(img_bgr, box):
    xc, yc, w, h = box
    H, W = img_bgr.shape[:2]
    x1 = max(0, int((xc - w / 2) * W))
    y1 = max(0, int((yc - h / 2) * H))
    x2 = min(W, int((xc + w / 2) * W))
    y2 = min(H, int((yc + h / 2) * H))
    if x2 <= x1 or y2 <= y1:
        return img_bgr
    return img_bgr[y1:y2, x1:x2]


def eval_yolo():
    if not Path(YW).exists():
        return {"mAP50": None, "accuracy": None, "f1": None, "avg_ms": None, "note": "no weights"}
    m = YoloModel(weights=Path(YW))
    try:
        metrics = m.model.val(data=str(DATA_YAML), device="cpu", split="test", verbose=False)
        map50 = float(metrics.box.map50)
    except Exception as e:
        map50 = None
        print(f"  yolo val err: {e}")
    samples = _test_samples(150)
    y_true, y_pred, lat = [], [], []
    for img_path, cls, _ in samples:
        img = cv2.imread(str(img_path))
        t = time.time()
        dets = m.predict(img, conf=0.1)
        lat.append((time.time() - t) * 1000)
        if dets:
            det = max(dets, key=lambda d: d["confidence"])
            y_pred.append(det["class_id"])
        else:
            y_pred.append(-1)
        y_true.append(cls)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"mAP50": map50, "accuracy": float(acc), "f1": float(f1), "avg_ms": float(np.mean(lat)) if lat else None}


def eval_cnn():
    if not Path(CNN_W).exists():
        return {"mAP50": None, "accuracy": None, "f1": None, "avg_ms": None, "note": "no weights"}
    m = CNNModel(num_classes=len(get_class_mapping()))
    m.load(CNN_W)
    samples = _test_samples(200)
    y_true, y_pred, lat = [], [], []
    for img_path, cls, box in samples:
        img = cv2.imread(str(img_path))
        crop = _crop(img, box)
        t = time.time()
        pred = m.predict(crop)
        lat.append((time.time() - t) * 1000)
        y_pred.append(pred["class_id"])
        y_true.append(cls)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"mAP50": None, "accuracy": float(acc), "f1": float(f1), "avg_ms": float(np.mean(lat))}


def eval_rf():
    if not Path(RF_W).exists() or not Path(KEYPOINT_CSV).exists():
        return {"mAP50": None, "accuracy": None, "f1": None, "avg_ms": None, "note": "no weights/csv"}
    m = RFModel()
    m.load(RF_W)
    ext = HandKeypointExtractor()
    samples = _test_samples(200)
    y_true, y_pred, lat = [], [], []
    for img_path, cls, box in samples:
        img = cv2.imread(str(img_path))
        crop = _crop(img, box)
        t = time.time()
        feats = ext.extract(crop)
        if feats is None:
            lat.append((time.time() - t) * 1000)
            continue
        pred = m.predict(feats)
        lat.append((time.time() - t) * 1000)
        y_pred.append(pred["class_id"])
        y_true.append(cls)
    ext.close()
    if not y_pred:
        return {"mAP50": None, "accuracy": 0.0, "f1": 0.0, "avg_ms": float(np.mean(lat)) if lat else None, "note": "no keypoints detected"}
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"mAP50": None, "accuracy": float(acc), "f1": float(f1), "avg_ms": float(np.mean(lat))}


def main():
    print("Evaluating YOLO...")
    y = eval_yolo()
    print("Evaluating CNN...")
    c = eval_cnn()
    print("Evaluating RF...")
    r = eval_rf()

    rows = [
        {"Model": "YOLOv8", **y},
        {"Model": "CNN", **c},
        {"Model": "RF", **r},
    ]
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Model", "mAP50", "accuracy", "f1", "avg_ms", "note"])
        w.writeheader()
        for row in rows:
            row.setdefault("note", "")
            w.writerow({k: row.get(k) for k in w.fieldnames})

    df = pd.DataFrame(rows)
    print("\n=== Evaluation ===")
    print(df.to_string(index=False))
    print(f"\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
