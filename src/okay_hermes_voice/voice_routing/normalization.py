"""Local voice-command transcript normalization."""
from __future__ import annotations

import re


def normalize_voice_command(text: str) -> str:
    """Normalize STT text for exact local voice-control commands."""
    normalized = (text or "").casefold()
    normalized = re.sub(r"[^\w\s']+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for prefix in ("okay hermes ", "ok hermes ", "hey hermes ", "hermes "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
            break
    return normalized


__all__ = ["normalize_voice_command"]
