"""Popup lifecycle timing helpers."""
from __future__ import annotations

from typing import Any, Mapping

DEFAULT_FINAL_KEEP_OPEN_SECONDS = 45.0


def final_keep_open_seconds(state: Mapping[str, Any]) -> float:
    """Return final-state keep-open seconds, preserving explicit zero."""
    raw = state.get("keep_open_seconds", DEFAULT_FINAL_KEEP_OPEN_SECONDS)
    if raw is None or raw == "":
        raw = DEFAULT_FINAL_KEEP_OPEN_SECONDS
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_FINAL_KEEP_OPEN_SECONDS


__all__ = ["DEFAULT_FINAL_KEEP_OPEN_SECONDS", "final_keep_open_seconds"]
