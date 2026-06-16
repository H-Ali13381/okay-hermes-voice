"""Copy *_seconds fields into a summary bucket."""
from __future__ import annotations

from typing import Any, Dict

from .metrics import numeric_seconds


def add_seconds_fields(bucket: Dict[str, Any], timing: Any) -> None:
    if not isinstance(timing, dict):
        return
    for key, value in timing.items():
        if not str(key).endswith("_seconds"):
            continue
        numeric = numeric_seconds(value)
        if numeric is not None:
            bucket["timing_values"][str(key)].append(numeric)


__all__ = ["add_seconds_fields"]
