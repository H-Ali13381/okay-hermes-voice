"""Timing helpers for activation turns."""
from __future__ import annotations

from typing import Any, Dict


def elapsed_seconds(time_module: Any, started: float) -> float:
    """Return a monotonic non-negative elapsed duration for observability fields."""
    return max(0.0, time_module.monotonic() - started)


def merge_speak_timing(turn_timing: Dict[str, Any], speak_result: Any, fallback_seconds: float) -> None:
    """Copy TTS/playback timing returned by playback.speak_response into a turn record."""
    turn_timing["speak_seconds"] = fallback_seconds
    if not isinstance(speak_result, dict):
        return
    for key in (
        "tts_enabled",
        "tts_success",
        "playback_success",
        "tts_seconds",
        "playback_seconds",
        "speak_seconds",
        "tts_file_path",
    ):
        if key in speak_result:
            turn_timing[key] = speak_result[key]
