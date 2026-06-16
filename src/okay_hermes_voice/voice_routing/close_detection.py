"""Explicit close-command detection."""
from __future__ import annotations

from typing import Any, Dict

from ..daemon_config import DEFAULT_CONFIG
from .normalization import normalize_voice_command


def is_close_transcript(transcript: str, cfg: Dict[str, Any]) -> bool:
    """Return True only for explicit voice-session close commands."""
    normalized = normalize_voice_command(transcript)
    phrases = cfg.get("conversation_close_phrases") or DEFAULT_CONFIG["conversation_close_phrases"]
    normalized_phrases = {normalize_voice_command(str(phrase)) for phrase in phrases if str(phrase).strip()}
    return normalized in normalized_phrases


__all__ = ["is_close_transcript"]
