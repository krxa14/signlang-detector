"""Sequentially train YOLO, CNN, RF. CPU-friendly defaults."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import DATA_YAML, fix_valid_split, get_class_mapping, get_dataloaders  # noqa: E402
from data.preprocess import build_keypoint_dataset  # noqa: E402
from models.cnn_model import CNNModel  # noqa: E402
from models.rf_model import RFModel  # noqa: E402
from models.yolo_model import YoloModel  # noqa: E402


def _fmt(d):
    return json.dumps(d, indent=2, default=str)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo-epochs", type=int, default=3)
    p.add_argument("--yolo-imgsz", type=int, default=320)
    p.add_argument("--yolo-batch", type=int, default=16)
    p.add_argument("--yolo-fraction", type=float, default=0.15)
    p.add_argument("--cnn-epochs", type=int, default=2)
    p.add_argument("--cnn-batch", type=int, default=64)
    p.add_argument("--kp-limit", type=int, default=1500)
    p.add_argument("--skip-yolo", action="store_true")
    p.add_argument("--skip-cnn", action="store_true")
    p.add_argument("--skip-rf", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    print("=" * 70); print("[1/5] Fix dataset split"); print("=" * 70)
    split_info = fix_valid_split()
    print(_fmt(split_info))

    print("=" * 70); print("[2/5] Extract MediaPipe keypoints -> CSV"); print("=" * 70)
    kp_info = build_keypoint_dataset(limit=args.kp_limit)
    print(_fmt(kp_info))

    results = {"split": split_info, "keypoints": kp_info}

    if not args.skip_yolo:
        print("=" * 70); print(f"[3/5] Train YOLOv8n epochs={args.yolo_epochs} imgsz={args.yolo_imgsz} fraction={args.yolo_fraction}"); print("=" * 70)
        y = YoloModel()
        yinfo = y.train(DATA_YAML, epochs=args.yolo_epochs, imgsz=args.yolo_imgsz, batch=args.yolo_batch, fraction=args.yolo_fraction)
        results["yolo"] = yinfo
        print(_fmt(yinfo))
    else:
        print("[3/5] skipped")

    if not args.skip_cnn:
        print("=" * 70); print(f"[4/5] Train CNN epochs={args.cnn_epochs} batch={args.cnn_batch}"); print("=" * 70)
        tl, vl = get_dataloaders(batch_size=args.cnn_batch, num_workers=0)
        print(f"train batches: {len(tl)} | val batches: {len(vl)}")
        cnn = CNNModel(num_classes=len(get_class_mapping()))
        cinfo = cnn.train(tl, vl, epochs=args.cnn_epochs)
        results["cnn"] = cinfo
        print(_fmt({k: v for k, v in cinfo.items() if k != "history"}))
    else:
        print("[4/5] skipped")

    if not args.skip_rf:
        print("=" * 70); print("[5/5] Train Random Forest on keypoints"); print("=" * 70)
        rf = RFModel(n_estimators=200)
        rinfo = rf.train()
        results["rf"] = rinfo
        print(_fmt(rinfo))
    else:
        print("[5/5] skipped")

    elapsed = time.time() - t0
    print("=" * 70); print(f"Training complete in {elapsed:.1f}s"); print("=" * 70)
    print("Final summary:")
    print(_fmt(results))
    results_path = PROJECT_ROOT / "results" / "train_summary.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


if __name__ == "__main__":
    main()
