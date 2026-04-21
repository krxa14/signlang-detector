"""DB CRUD helpers."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import case, func
from sqlalchemy.orm import Session as OrmSession

from database.models import Feedback, Prediction, Session


def log_prediction(db: OrmSession, label: str, confidence: float, model_used: str = "ensemble",
                   image_path: Optional[str] = None, session_id: Optional[int] = None) -> Prediction:
    pred = Prediction(
        predicted_label=label,
        confidence=confidence,
        model_used=model_used,
        input_image_path=image_path,
        session_id=session_id,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


def get_recent_predictions(db: OrmSession, n: int = 20) -> List[Prediction]:
    return db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(n).all()


def get_accuracy_over_time(db: OrmSession, days: int = 7) -> List[dict]:
    cutoff = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(
            func.date(Prediction.timestamp).label("day"),
            func.count(Prediction.id).label("total"),
            func.sum(case((Prediction.correct.is_(True), 1), else_=0)).label("correct"),
        )
        .filter(Prediction.timestamp >= cutoff)
        .group_by("day")
        .all()
    )
    out = []
    for day, total, correct in rows:
        total = total or 0
        correct = correct or 0
        out.append({"day": str(day), "total": int(total), "correct": int(correct), "accuracy": (correct / total) if total else None})
    return out


def submit_feedback(db: OrmSession, prediction_id: int, is_correct: bool,
                    corrected_label: Optional[str] = None, notes: Optional[str] = None) -> Feedback:
    fb = Feedback(prediction_id=prediction_id, is_correct=is_correct, corrected_label=corrected_label, notes=notes)
    db.add(fb)
    pred = db.get(Prediction, prediction_id)
    if pred:
        pred.correct = is_correct
    db.commit()
    db.refresh(fb)
    return fb


def get_label_counts(db: OrmSession, limit: int = 10) -> List[dict]:
    rows = (
        db.query(Prediction.predicted_label, func.count(Prediction.id))
        .group_by(Prediction.predicted_label)
        .order_by(func.count(Prediction.id).desc())
        .limit(limit)
        .all()
    )
    return [{"label": r[0], "count": int(r[1])} for r in rows]


def get_summary(db: OrmSession) -> dict:
    total = db.query(func.count(Prediction.id)).scalar() or 0
    avg_conf = db.query(func.avg(Prediction.confidence)).scalar()
    labels = get_label_counts(db, limit=10)
    return {
        "total_predictions": int(total),
        "avg_confidence": float(avg_conf) if avg_conf is not None else 0.0,
        "top_labels": labels,
    }


def start_session(db: OrmSession) -> Session:
    s = Session()
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def end_session(db: OrmSession, session_id: int) -> Optional[Session]:
    s = db.get(Session, session_id)
    if not s:
        return None
    preds = db.query(Prediction).filter(Prediction.session_id == session_id).all()
    s.end_time = datetime.utcnow()
    s.total_predictions = len(preds)
    if preds:
        counter = Counter(p.predicted_label for p in preds)
        s.dominant_label = counter.most_common(1)[0][0]
    db.commit()
    db.refresh(s)
    return s
