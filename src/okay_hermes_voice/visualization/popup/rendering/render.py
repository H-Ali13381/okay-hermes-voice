"""String-only popup render compatibility wrapper."""
from __future__ import annotations

from typing import Any, Dict

from .frame import render_frame


def render(state: Dict[str, Any], tick: int, final_seen_at: float | None, scroll_offset: int = 0) -> str:
    return render_frame(state, tick, final_seen_at, scroll_offset)[0]


__all__ = ["render"]
