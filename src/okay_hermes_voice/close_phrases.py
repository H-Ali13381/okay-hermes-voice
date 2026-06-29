"""Shared local voice-session close phrase normalization."""
from __future__ import annotations

import re

DEFAULT_CLOSE_PHRASES = (
    "close",
    "please close",
    "close voice",
    "close voice mode",
    "close conversation",
    "close hermes",
    "stop",
    "stop voice",
    "stop listening",
    "end conversation",
    "end voice mode",
    "that's all",
    "that is all",
    "goodbye",
    "bye",
    "cancel",
)


def normalize_close_phrase(text: str) -> str:
    """Normalize STT text for exact local voice-session close commands."""
    normalized = (text or "").casefold()
    normalized = re.sub(r"[^\w\s']+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for prefix in ("okay hermes ", "ok hermes ", "hey hermes ", "hermes "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
            break
    return normalized


LOCAL_CLOSE_PHRASES = {normalize_close_phrase(phrase) for phrase in DEFAULT_CLOSE_PHRASES}


__all__ = ["DEFAULT_CLOSE_PHRASES", "LOCAL_CLOSE_PHRASES", "normalize_close_phrase"]
