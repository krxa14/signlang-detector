"""Pydantic schemas for the API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded JPEG/PNG image (with or without data: prefix)")
    languages: Optional[List[str]] = Field(default=None, description="Target regional languages")


class ModelScore(BaseModel):
    class_id: Optional[int] = None
    label: Optional[str] = None
    confidence: float = 0.0


class Detection(BaseModel):
    box: List[int]
    label: str
    confidence: float
    class_id: int


class PredictResponse(BaseModel):
    label: Optional[str]
    confidence: float
    model_breakdown: Dict[str, ModelScore] = {}
    detections: List[Detection] = []
    translations: Dict[str, str] = {}
    sentence: str = ""
    prediction_id: Optional[int] = None


class PredictionOut(BaseModel):
    id: int
    timestamp: datetime
    predicted_label: str
    confidence: float
    model_used: str
    correct: Optional[bool] = None

    class Config:
        from_attributes = True


class AnalyticsResponse(BaseModel):
    total_predictions: int
    avg_confidence: float
    top_labels: List[Dict[str, Any]]
    recent: List[PredictionOut]
    accuracy_over_time: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    num_classes: int


class FeedbackRequest(BaseModel):
    prediction_id: int
    is_correct: bool
    corrected_label: Optional[str] = None
    notes: Optional[str] = None
