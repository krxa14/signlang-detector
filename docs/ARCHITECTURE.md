# Architecture

This document explains how a single video frame becomes a translated sentence.

## End-to-end data flow

```
[webcam frame, 320x240 BGR]
        │
        ▼
┌─────────────────────────┐
│ pipeline.detector.Detector
│   YOLOv8n (models/yolo_model.py)
│   conf threshold 0.25
└──────────┬──────────────┘
           │ list[{box, crop, label, conf, class_id}]
           ▼
┌──────────────────────────────────────────────────────────────┐
│ models.ensemble.Ensemble.predict                            │
│   highest-conf detection → crop                             │
│   ┌─────────┐    ┌─────────┐    ┌────────────────────┐      │
│   │ YOLO    │    │  CNN    │    │ MediaPipe → RF     │      │
│   │ class+c │    │ 64×64   │    │ 21×3 keypoints     │      │
│   └────┬────┘    └────┬────┘    └─────────┬──────────┘      │
│        │ 0.5         │ 0.3              │ 0.2              │
│        └────────┬────┴───────┬──────────┘                    │
│                 ▼ Σ wᵢ·confᵢ per class_id                    │
│            argmax → final label, ensemble confidence         │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────┐
│ pipeline.translator.SignTranslator  │
│   rolling buffer (5)                │
│   contiguous-letter-merge → words   │
│   list of words → English sentence  │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ pipeline.regional.translate         │
│   deep-translator → Telugu, Tamil   │
│   graceful fallback on offline      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ database.crud.log_prediction        │
│   SQLite (Prediction, Session)      │
│   feeds /analytics                  │
└─────────────────────────────────────┘
```

## Module boundaries

- **data/**: dataset surgery (train↔valid split fix), CNN DataLoaders, MediaPipe keypoint CSV builder. No model code.
- **models/**: each of YOLO, CNN, RF wraps `train`/`predict`/`save`/`load`. `ensemble.py` is the only place weights are combined.
- **pipeline/**: thin glue. Detector ↔ Classifier ↔ Translator ↔ Regional. No model internals leak through.
- **api/**: HTTP / WebSocket surface. Routes never touch SQLAlchemy directly — they go through `database/crud.py`.
- **database/**: SQLAlchemy models + CRUD + engine. Engine is `connect_args={"check_same_thread": False}` so `/stream` can write from worker threads.
- **frontend/**: vanilla JS, no build step. `stream.js` owns the WebSocket; `analytics.js` owns the polling loop and Chart.js bar chart.

## Why an ensemble?

Each model has a different failure mode on this dataset:

| Model | Strength | Weakness |
|-------|----------|----------|
| YOLOv8 | Localizes hand + classifies in one pass | 106-class detection needs lots of epochs |
| CNN   | Fast, robust to MediaPipe failure | Sees the whole crop incl. background |
| RF    | Tiny, deterministic, interpretable feature importance | Useless when MediaPipe doesn't detect a hand |

The weighted vote (0.5/0.3/0.2) lets YOLO drive when it's confident, lets CNN backstop when YOLO is uncertain, and lets RF break ties using pose geometry rather than pixels.

## Translator design

The 106 classes are a mix of **letters** (A–Z) and **whole words** (`pizza`, `thank-you`, …). The translator therefore:

1. Filters consecutive duplicates (a held sign shouldn't add 10 copies).
2. Concatenates contiguous single letters into a word (`H,E,L,L,O` → `HELLO`).
3. Joins everything else with spaces.
4. Hands the resulting English string to `deep-translator` (Google) for Telugu/Tamil.

The whole sentence accumulates across frames; `/reset-sentence` clears it.

## Database schema

```
predictions
  id pk · timestamp · input_image_path · predicted_label · confidence
  model_used · correct (nullable) · session_id fk

sessions
  id pk · start_time · end_time · total_predictions · dominant_label

feedback
  id pk · prediction_id fk · timestamp · is_correct · corrected_label · notes
```

`get_accuracy_over_time` uses SQLAlchemy 2.x `case((cond, val), else_=...)` (note: `case` is imported from `sqlalchemy`, not `sqlalchemy.func`).

## Why CPU-only?

The user's environment (Apple M4) has no CUDA. The whole stack is configured with `device="cpu"` for ultralytics, `torch.device("cpu")` for the CNN, and `n_jobs=-1` for sklearn. To enable GPU, change those three lines and rebuild Docker with a CUDA base image.
