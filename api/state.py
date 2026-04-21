"""Shared API state: ensemble + translator singletons."""
from __future__ import annotations

from models.ensemble import Ensemble
from pipeline.translator import SignTranslator

ensemble = Ensemble()
translator = SignTranslator()
_loaded_status = {"yolo": False, "cnn": False, "rf": False}


def load_models() -> dict:
    global _loaded_status
    _loaded_status = ensemble.load_all()
    return _loaded_status


def models_status() -> dict:
    return dict(_loaded_status)
