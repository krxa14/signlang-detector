"""Standalone webcam demo using the full ensemble."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ensemble import Ensemble
from pipeline.translator import SignTranslator


def main():
    ens = Ensemble()
    loaded = ens.load_all()
    print(f"Loaded: {loaded}")

    trans = SignTranslator()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")
    prev = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = ens.predict(frame)
            if res["label"]:
                trans.add_sign(res["label"])
            for d in res.get("detections") or []:
                x1, y1, x2, y2 = d["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (88, 166, 255), 2)
                cv2.putText(frame, f"{d['label']} {d['confidence']:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (88, 166, 255), 2)
            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, trans.get_buffer_sentence(), (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("SignLang Ensemble", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
