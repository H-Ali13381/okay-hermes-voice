"""Popup viewport scroll-key handling."""
from __future__ import annotations

from ..rendering import term_height


def apply_scroll_key(scroll_offset: int, key: str | None) -> int:
    if key in {"up", "page_up"}:
        amount = 1 if key == "up" else max(3, term_height() - 6)
        return max(0, scroll_offset - amount)
    if key in {"down", "page_down"}:
        amount = 1 if key == "down" else max(3, term_height() - 6)
        return scroll_offset + amount
    if key == "home":
        return 0
    if key == "end":
        return 10**9
    return scroll_offset


__all__ = ["apply_scroll_key"]
