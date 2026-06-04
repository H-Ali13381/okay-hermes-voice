"""Keyboard, mouse-wheel, and terminal-control handling for the popup."""
from __future__ import annotations

import select
import sys

from .constants import SCROLL_KEY_CHARS, SCROLL_KEY_SEQUENCES
from .rendering import term_height


def write_terminal_control(sequence: str) -> None:
    sys.stdout.write(sequence)
    sys.stdout.flush()


def normalized_escape_key(sequence: str) -> str | None:
    if sequence in SCROLL_KEY_SEQUENCES:
        return SCROLL_KEY_SEQUENCES[sequence]

    if sequence.startswith("\033[<") and sequence.endswith(("M", "m")):
        try:
            button = int(sequence[3:-1].split(";", 1)[0])
        except ValueError:
            return None
        # SGR mouse mode reports wheel as buttons 64/65. Modifier bits may be
        # added, so strip Shift/Meta/Ctrl bits before matching.
        base_button = button & ~0b11100
        if base_button == 64:
            return "up"
        if base_button == 65:
            return "down"
        return None

    if sequence.startswith("\033[M") and len(sequence) >= 6:
        base_button = (ord(sequence[3]) - 32) & ~0b11100
        if base_button == 64:
            return "up"
        if base_button == 65:
            return "down"
    return None


def read_keypress() -> str | None:
    """Read one non-blocking TUI keypress from stdin, normalized for scrolling."""
    try:
        if not sys.stdin.isatty():
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        char = sys.stdin.read(1)
        if char == "\033":
            sequence = char
            for _ in range(32):
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if not ready:
                    break
                sequence += sys.stdin.read(1)
                if sequence in SCROLL_KEY_SEQUENCES:
                    break
                if sequence.startswith("\033[<") and sequence.endswith(("M", "m")):
                    break
                if sequence.startswith("\033[M") and len(sequence) >= 6:
                    break
            return normalized_escape_key(sequence)
        return SCROLL_KEY_CHARS.get(char)
    except Exception:
        return None


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


__all__ = ["apply_scroll_key", "normalized_escape_key", "read_keypress", "write_terminal_control"]
