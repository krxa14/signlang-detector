"""POST /predict route."""
from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from PIL import Image
from sqlalchemy.orm import Session as OrmSession

from api.schemas import FeedbackRequest, PredictRequest, PredictResponse
from api.state import ensemble, translator
from database.crud import log_prediction, submit_feedback
from database.db import get_db
from pipeline.regional import translate

router = APIRouter()


def _decode_image(b64: str) -> np.ndarray:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}")
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: OrmSession = Depends(get_db)):
    frame = _decode_image(req.image_b64)
    result = ensemble.predict(frame)
    label = result.get("label")
    conf = float(result.get("confidence") or 0.0)

    if label:
        translator.add_sign(label)
    sentence = translator.get_sentence() or (label or "")
    translations = translate(sentence, languages=req.languages or ["telugu", "tamil"]) if sentence else {}

    pred_id = None
    if label:
        pred = log_prediction(db, label=label, confidence=conf)
        pred_id = pred.id

    breakdown = {}
    for name, info in (result.get("per_model") or {}).items():
        breakdown[name] = {
            "class_id": info.get("class_id"),
            "label": info.get("label"),
            "confidence": info.get("confidence", 0.0),
        }

    dets = []
    for d in result.get("detections") or []:
        dets.append({
            "box": d["box"],
            "label": d["label"],
            "confidence": d["confidence"],
            "class_id": d["class_id"],
        })

    return PredictResponse(
        label=label,
        confidence=conf,
        model_breakdown=breakdown,
        detections=dets,
        translations=translations,
        sentence=sentence,
        prediction_id=pred_id,
    )


@router.post("/feedback")
def feedback(req: FeedbackRequest, db: OrmSession = Depends(get_db)):
    fb = submit_feedback(db, req.prediction_id, req.is_correct, req.corrected_label, req.notes)
    return {"id": fb.id, "ok": True}


@router.post("/reset-sentence")
def reset_sentence():
    translator.reset()
    return {"ok": True}
