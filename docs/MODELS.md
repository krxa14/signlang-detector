# Model Cards

## YOLOv8n (detection)

- **Backbone**: ultralytics YOLOv8-nano (3.4M params, 9.6 GFLOPs).
- **Input**: variable-size BGR frame, resized to `imgsz` (default 256).
- **Output**: list of `{box xyxy, class_id, confidence}` over 106 classes.
- **Training**: `data=American-sign-language-2/data.yaml`, CPU, 5 epochs, fraction=0.12, batch=16. ~18 min on Apple M4.
- **Reported (this repo)**: mAP50 0.0396 on the test split. Underfit — see README "Notes".
- **Weights**: `models/weights/yolo_best.pt` (copied from `runs/yolo_train/weights/best.pt`).

## SignCNN (classification)

- **Architecture**: 3 × (Conv → BN → ReLU → MaxPool) with 32→64→128 filters, Dropout 0.3, FC 256 → 106.
- **Input**: 64×64 RGB crop (from YOLO box at inference time, from YOLO labels at training time).
- **Loss/optim**: cross-entropy + Adam (lr 1e-3).
- **Training**: 1 epoch on ~14k crops, batch 64, CPU. ~1 min.
- **Reported (this repo)**: val acc 0.575, test acc 0.565, F1 0.471, 1.3 ms/inference.
- **Weights**: `models/weights/cnn_best.pt` (state dict + `num_classes`).

## RandomForest on MediaPipe keypoints

- **Features**: 21 hand landmarks × (x, y, z) = 63 floats, extracted via MediaPipe Hands (`min_detection_confidence=0.7`).
- **Model**: `RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)`.
- **Training data**: `data/keypoints.csv` (built by `data/preprocess.py:build_keypoint_dataset`); ~435 rows after the MediaPipe confidence gate on 1200 sampled crops.
- **Reported (this repo)**: holdout acc 0.506, test acc 0.429, F1 0.279, 22.7 ms/inference (dominated by MediaPipe).
- **Weights**: `models/weights/rf_model.joblib`.
- **Interpretability**: `RFModel.get_feature_importance()` returns the 63-dim importance vector — visualised in `data/analysis.ipynb`.

## Ensemble

- **Weighted vote** in `models/ensemble.py`, weights `{yolo: 0.5, cnn: 0.3, rf: 0.2}`.
- For each class, accumulates `Σ wᵢ · confᵢ` over models that produced a prediction; picks argmax. Ensemble confidence = winning score divided by sum of active weights.
- Robust to any single model failing: missing detections / no MediaPipe hand simply zero out that branch.
