"""Rolling-buffer sign-sequence -> English sentence translator."""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

# Letter-to-word helpers: if buffer contains contiguous single letters, join them.
_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class SignTranslator:
    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        self.buffer: Deque[str] = deque(maxlen=buffer_size)
        self.sentence_tokens: List[str] = []
        self._last_added: Optional[str] = None

    def add_sign(self, label: Optional[str]):
        if not label:
            return
        if label == self._last_added:
            return
        self.buffer.append(label)
        self._last_added = label
        self.sentence_tokens.append(label)

    def _compose(self, tokens: List[str]) -> str:
        out: List[str] = []
        current_word: List[str] = []
        for t in tokens:
            if t in _LETTERS:
                current_word.append(t)
            else:
                if current_word:
                    out.append("".join(current_word))
                    current_word = []
                out.append(t)
        if current_word:
            out.append("".join(current_word))
        return " ".join(out).replace(" thank-you", " thank you").strip()

    def get_sentence(self) -> str:
        return self._compose(self.sentence_tokens)

    def get_buffer_sentence(self) -> str:
        return self._compose(list(self.buffer))

    def reset(self):
        self.buffer.clear()
        self.sentence_tokens.clear()
        self._last_added = None
