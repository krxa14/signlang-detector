"""WebSocket /stream route: receives base64 frames, returns predictions."""
from __future__ import annotations

import base64
import json
from io import BytesIO

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image

from api.state import ensemble, translator
from database.crud import log_prediction
from database.db import SessionLocal
from pipeline.regional import translate

router = APIRouter()


def _decode(b64: str) -> np.ndarray:
    if "," in b64 and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(BytesIO(raw)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


@router.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            try:
                payload = json.loads(msg)
            except Exception:
                payload = {"image_b64": msg}
            b64 = payload.get("image_b64", "")
            if not b64:
                await ws.send_json({"error": "missing image_b64"})
                continue
            try:
                frame = _decode(b64)
            except Exception as e:
                await ws.send_json({"error": f"decode: {e}"})
                continue
            result = ensemble.predict(frame)
            label = result.get("label")
            conf = float(result.get("confidence") or 0.0)
            if label:
                translator.add_sign(label)
                db = SessionLocal()
                try:
                    log_prediction(db, label=label, confidence=conf)
                finally:
                    db.close()
            sentence = translator.get_sentence() or (label or "")
            translations = translate(sentence, ["telugu", "tamil"]) if sentence else {}
            dets = [
                {"box": d["box"], "label": d["label"], "confidence": d["confidence"], "class_id": d["class_id"]}
                for d in (result.get("detections") or [])
            ]
            await ws.send_json({
                "label": label,
                "confidence": conf,
                "sentence": sentence,
                "translations": translations,
                "detections": dets,
            })
    except WebSocketDisconnect:
        return
