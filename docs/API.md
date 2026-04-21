# API Reference

Base URL: `http://localhost:8000`

Interactive Swagger UI: `http://localhost:8000/docs`

## `GET /health`

Liveness + per-model load status.

```json
{
  "status": "ok",
  "models_loaded": {"yolo": true, "cnn": true, "rf": true},
  "num_classes": 106
}
```

## `POST /predict`

Body:

```json
{
  "image_b64": "data:image/jpeg;base64,...",
  "languages": ["telugu", "tamil"]
}
```

Response: `label`, `confidence`, `model_breakdown` (per-model), `detections` (YOLO boxes), `translations` (English/Telugu/Tamil), `sentence` (rolling sentence), `prediction_id`.

The `data:` URI prefix is optional. `languages` defaults to `["telugu", "tamil"]`.

## `WS /stream`

Send JSON frames `{"image_b64": "..."}`; receive prediction JSON for each.
Recommended cadence: 1–2 fps (CPU). The frontend caps at one frame every 800 ms.

## `GET /analytics`

```json
{
  "total_predictions": 132,
  "avg_confidence": 0.41,
  "top_labels": [{"label": "A", "count": 17}, ...],
  "recent": [{"id": 132, "predicted_label": "A", "confidence": 0.5, ...}],
  "accuracy_over_time": [{"day": "2026-04-20", "total": 50, "correct": 22, "accuracy": 0.44}]
}
```

## `POST /feedback`

```json
{"prediction_id": 17, "is_correct": false, "corrected_label": "B"}
```

Marks `predictions.correct` and inserts a `feedback` row.

## `POST /reset-sentence`

Clears the rolling sentence buffer used by `/predict` and `/stream`.
