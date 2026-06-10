"""Numeric seconds-field coercion."""
from __future__ import annotations

from typing import Any, Optional


def numeric_seconds(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0.0 else None


__all__ = ["numeric_seconds"]
