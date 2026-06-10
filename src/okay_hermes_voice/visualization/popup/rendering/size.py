"""Terminal viewport size policy."""
from __future__ import annotations

import shutil
from typing import Tuple


def term_size() -> Tuple[int, int]:
    size = shutil.get_terminal_size((96, 28))
    width = max(60, min(size.columns, 140))
    height = max(10, size.lines)
    return width, height


__all__ = ["term_size"]
