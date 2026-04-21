"""GET /analytics route."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as OrmSession

from api.schemas import AnalyticsResponse, PredictionOut
from database.crud import get_accuracy_over_time, get_recent_predictions, get_summary
from database.db import get_db

router = APIRouter()


@router.get("/analytics", response_model=AnalyticsResponse)
def analytics(db: OrmSession = Depends(get_db)):
    summary = get_summary(db)
    recent = [PredictionOut.model_validate(p) for p in get_recent_predictions(db, 20)]
    trend = get_accuracy_over_time(db, days=7)
    return AnalyticsResponse(
        total_predictions=summary["total_predictions"],
        avg_confidence=summary["avg_confidence"],
        top_labels=summary["top_labels"],
        recent=recent,
        accuracy_over_time=trend,
    )
