"""Text wrapping for popup sections."""
from __future__ import annotations

import textwrap
from typing import List

from ..constants import BOLD, RESET


def wrapped_block(title: str, text: str, width: int, color: str = "") -> List[str]:
    if not text:
        return []
    body_width = max(20, width - 6)
    lines = [f"{BOLD}{color}{title}{RESET}"]
    for para in str(text).strip().splitlines() or [""]:
        if not para.strip():
            lines.append("")
            continue
        lines.extend("  " + line for line in textwrap.wrap(para, width=body_width, replace_whitespace=False))
    return lines


__all__ = ["wrapped_block"]
