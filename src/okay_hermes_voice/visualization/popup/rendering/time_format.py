"""Timestamp formatting for popup state."""
from __future__ import annotations

import datetime as _dt
from typing import Any


def format_time(epoch: Any) -> str:
    try:
        return _dt.datetime.fromtimestamp(float(epoch)).strftime("%H:%M:%S")
    except Exception:
        return "--:--:--"


__all__ = ["format_time"]
