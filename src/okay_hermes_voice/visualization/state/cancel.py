"""Cancellation reads from popup visualization state."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .read import read_visualization_state


def is_visualization_cancel_requested(path: Optional[Path]) -> bool:
    return bool(read_visualization_state(path).get("cancel_requested"))


def visualization_cancel_reason(path: Optional[Path]) -> str:
    state = read_visualization_state(path)
    return str(state.get("cancel_reason") or "terminal_cancel")


__all__ = ["is_visualization_cancel_requested", "visualization_cancel_reason"]
