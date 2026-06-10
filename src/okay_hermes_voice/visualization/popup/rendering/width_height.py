"""Terminal width/height convenience accessors."""
from __future__ import annotations

from .size import term_size


def term_width() -> int:
    return term_size()[0]


def term_height() -> int:
    return term_size()[1]


__all__ = ["term_height", "term_width"]
