"""MediaPipe hand keypoint extraction + keypoint CSV dataset builder."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from data.dataset import DATASET_ROOT, PROJECT_ROOT

KEYPOINT_CSV = PROJECT_ROOT / "data" / "keypoints.csv"
MIN_CONFIDENCE = 0.7

mp_hands = mp.solutions.hands


class HandKeypointExtractor:
    def __init__(self, min_detection_confidence: float = MIN_CONFIDENCE):
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
        )

    def extract(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        if image_bgr is None or image_bgr.size == 0:
            return None
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0].landmark
        feats = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
        return feats  # shape (63,)

    def close(self):
        self.hands.close()


def extract_keypoints(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    ex = HandKeypointExtractor()
    try:
        return ex.extract(image_bgr)
    finally:
        ex.close()


def _crop_from_yolo(img: np.ndarray, lbl_path: Path):
    with open(lbl_path) as f:
        line = f.readline().strip().split()
    if not line:
        return None, None
    cls = int(line[0])
    xc, yc, w, h = map(float, line[1:5])
    H, W = img.shape[:2]
    x1 = max(0, int((xc - w / 2) * W))
    y1 = max(0, int((yc - h / 2) * H))
    x2 = min(W, int((xc + w / 2) * W))
    y2 = min(H, int((yc + h / 2) * H))
    if x2 <= x1 or y2 <= y1:
        return None, None
    return img[y1:y2, x1:x2], cls


def build_keypoint_dataset(limit: Optional[int] = None, verbose: bool = True) -> dict:
    """Iterate train crops, extract MediaPipe keypoints, write CSV."""
    img_dir = DATASET_ROOT / "train" / "images"
    lbl_dir = DATASET_ROOT / "train" / "labels"
    KEYPOINT_CSV.parent.mkdir(parents=True, exist_ok=True)

    extractor = HandKeypointExtractor()
    n_written = 0
    n_skipped = 0
    with open(KEYPOINT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"f{i}" for i in range(63)] + ["label"]
        writer.writerow(header)
        stems = {p.stem: p for p in lbl_dir.glob("*.txt")}
        img_paths = [p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if limit:
            img_paths = img_paths[:limit]
        for i, img_path in enumerate(img_paths):
            lbl = stems.get(img_path.stem)
            if not lbl:
                n_skipped += 1
                continue
            img = cv2.imread(str(img_path))
            crop, cls = _crop_from_yolo(img, lbl)
            if crop is None:
                n_skipped += 1
                continue
            feats = extractor.extract(crop)
            if feats is None:
                n_skipped += 1
                continue
            writer.writerow(list(feats) + [cls])
            n_written += 1
            if verbose and (i + 1) % 500 == 0:
                print(f"  processed {i+1}/{len(img_paths)} written={n_written} skipped={n_skipped}")
    extractor.close()
    return {"written": n_written, "skipped": n_skipped, "csv": str(KEYPOINT_CSV)}


if __name__ == "__main__":
    info = build_keypoint_dataset(limit=200)
    print(info)
