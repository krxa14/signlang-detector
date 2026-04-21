"""English -> Telugu/Tamil translation via deep-translator with graceful fallback."""
from __future__ import annotations

from typing import Dict, List

try:
    from deep_translator import GoogleTranslator
    _HAS_TRANSLATOR = True
except Exception:  # pragma: no cover
    _HAS_TRANSLATOR = False

_LANG_MAP = {"telugu": "te", "tamil": "ta", "english": "en"}


def translate(text: str, languages: List[str] | None = None) -> Dict[str, str]:
    languages = languages or ["telugu", "tamil"]
    out: Dict[str, str] = {"english": text}
    if not text:
        for lang in languages:
            out[lang] = ""
        return out
    if not _HAS_TRANSLATOR:
        for lang in languages:
            out[lang] = text  # fallback: echo
        out["_warning"] = "deep-translator not installed; returning echo"
        return out
    for lang in languages:
        code = _LANG_MAP.get(lang.lower())
        if not code:
            out[lang] = text
            continue
        try:
            out[lang] = GoogleTranslator(source="en", target=code).translate(text)
        except Exception as e:
            out[lang] = text
            out[f"_{lang}_error"] = str(e)
    return out


if __name__ == "__main__":
    print(translate("hello world"))
