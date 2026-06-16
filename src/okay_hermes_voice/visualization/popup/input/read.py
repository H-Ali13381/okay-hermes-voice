"""Non-blocking popup keypress reads."""
from __future__ import annotations

import select
import sys

from ..constants import SCROLL_KEY_CHARS, SCROLL_KEY_SEQUENCES
from .escape import normalized_escape_key


def read_keypress() -> str | None:
    """Read one non-blocking TUI keypress from stdin, normalized for scrolling."""
    try:
        if not sys.stdin.isatty():
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        char = sys.stdin.read(1)
        if char == "":
            sequence = char
            for _ in range(32):
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if not ready:
                    break
                sequence += sys.stdin.read(1)
                if sequence in SCROLL_KEY_SEQUENCES:
                    break
                if sequence.startswith("[<") and sequence.endswith(("M", "m")):
                    break
                if sequence.startswith("[M") and len(sequence) >= 6:
                    break
            return normalized_escape_key(sequence)
        return SCROLL_KEY_CHARS.get(char)
    except Exception:
        return None


__all__ = ["read_keypress"]
