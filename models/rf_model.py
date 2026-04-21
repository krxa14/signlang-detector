"""Random Forest classifier over 63-dim MediaPipe keypoints."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "rf_model.joblib"
KEYPOINT_CSV = PROJECT_ROOT / "data" / "keypoints.csv"


class RFModel:
    def __init__(self, n_estimators: int = 200):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
        self.trained = False

    def train(self, csv_path: Optional[Path] = None) -> dict:
        csv_path = Path(csv_path or KEYPOINT_CSV)
        df = pd.read_csv(csv_path)
        if df.empty:
            raise RuntimeError(f"keypoint csv empty: {csv_path}")
        X = df[[f"f{i}" for i in range(63)]].values
        y = df["label"].astype(int).values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
        self.clf.fit(Xtr, ytr)
        self.trained = True
        preds = self.clf.predict(Xte)
        acc = accuracy_score(yte, preds)
        f1 = f1_score(yte, preds, average="macro", zero_division=0)
        self.save(DEFAULT_WEIGHTS)
        return {"accuracy": float(acc), "f1_macro": float(f1), "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "weights": str(DEFAULT_WEIGHTS)}

    def predict(self, feats: np.ndarray) -> dict:
        if not self.trained:
            raise RuntimeError("RF not trained/loaded")
        feats = feats.reshape(1, -1)
        probs = self.clf.predict_proba(feats)[0]
        classes = self.clf.classes_
        idx = int(probs.argmax())
        return {"class_id": int(classes[idx]), "confidence": float(probs[idx]), "probs": probs, "classes": classes}

    def get_feature_importance(self) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("RF not trained/loaded")
        return self.clf.feature_importances_

    def save(self, path: Optional[Path] = None):
        path = Path(path or DEFAULT_WEIGHTS)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path)

    def load(self, path: Optional[Path] = None):
        path = Path(path or DEFAULT_WEIGHTS)
        self.clf = joblib.load(path)
        self.trained = True


if __name__ == "__main__":
    m = RFModel()
    print("rf ready")
