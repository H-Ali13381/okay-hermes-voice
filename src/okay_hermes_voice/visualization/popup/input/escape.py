"""Normalize escape sequences into popup control keys."""
from __future__ import annotations

from ..constants import SCROLL_KEY_SEQUENCES


def normalized_escape_key(sequence: str) -> str | None:
    if sequence in SCROLL_KEY_SEQUENCES:
        return SCROLL_KEY_SEQUENCES[sequence]
    if sequence.startswith("[<") and sequence.endswith(("M", "m")):
        try:
            button = int(sequence[3:-1].split(";", 1)[0])
        except ValueError:
            return None
        base_button = button & ~0b11100
        if base_button == 64:
            return "up"
        if base_button == 65:
            return "down"
        return None
    if sequence.startswith("[M") and len(sequence) >= 6:
        base_button = (ord(sequence[3]) - 32) & ~0b11100
        if base_button == 64:
            return "up"
        if base_button == 65:
            return "down"
    return None


__all__ = ["normalized_escape_key"]
