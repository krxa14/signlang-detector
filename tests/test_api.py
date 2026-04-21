"""FastAPI endpoint tests."""
from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app


def _b64_blank():
    img = Image.new("RGB", (320, 240), (128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "models_loaded" in body


def test_predict_blank():
    with TestClient(app) as c:
        r = c.post("/predict", json={"image_b64": _b64_blank()})
        assert r.status_code == 200
        body = r.json()
        assert "label" in body
        assert "confidence" in body


def test_analytics():
    with TestClient(app) as c:
        r = c.get("/analytics")
        assert r.status_code == 200
        body = r.json()
        for k in ["total_predictions", "avg_confidence", "top_labels", "recent", "accuracy_over_time"]:
            assert k in body
