"""SQLAlchemy models."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String

from database.db import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    input_image_path = Column(String, nullable=True)
    predicted_label = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    model_used = Column(String, default="ensemble")
    correct = Column(Boolean, nullable=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=True)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_predictions = Column(Integer, default=0)
    dominant_label = Column(String, nullable=True)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_correct = Column(Boolean, nullable=False)
    corrected_label = Column(String, nullable=True)
    notes = Column(String, nullable=True)
