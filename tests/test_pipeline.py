"""Unit tests for pipeline stages."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import get_class_mapping
from data.preprocess import extract_keypoints
from pipeline.regional import translate
from pipeline.translator import SignTranslator


def test_class_mapping():
    m = get_class_mapping()
    assert len(m) == 106
    assert m[0] == "A"


def test_translator_letter_joining():
    t = SignTranslator(buffer_size=5)
    for c in ["H", "E", "L", "L", "O"]:
        t.add_sign(c)
    # consecutive duplicates are filtered, so "HEL" is built, then one L, then O
    sentence = t.get_sentence()
    assert "H" in sentence


def test_translator_resets():
    t = SignTranslator()
    t.add_sign("hello")
    assert t.get_sentence()
    t.reset()
    assert t.get_sentence() == ""


def test_extract_keypoints_empty():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    res = extract_keypoints(img)
    # blank image -> no hand
    assert res is None or res.shape == (63,)


def test_regional_fallback():
    out = translate("hello", ["telugu", "tamil"])
    assert "english" in out
    assert "telugu" in out
    assert "tamil" in out
