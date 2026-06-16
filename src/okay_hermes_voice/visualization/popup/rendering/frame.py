"""Full popup frame rendering and viewport math."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ..constants import DIM, RESET
from .body import render_body_lines
from .footer import render_status_footer_lines
from .header import render_header_lines
from .size import term_size


def render_frame(state: Dict[str, Any], tick: int, final_seen_at: float | None, scroll_offset: int = 0) -> Tuple[str, int, int]:
    width, height = term_size()
    header = render_header_lines(state, tick, width)
    body = render_body_lines(state, width)
    status_footer = render_status_footer_lines(state, final_seen_at)

    visible_header_rows = max(0, len(header) - 1)
    body_height = max(1, height - visible_header_rows - len(status_footer))
    scrollable = len(body) > body_height
    scroll_footer = []
    if scrollable:
        scroll_footer = [f"{DIM}Scroll: ↑/k ↓/j PgUp/PgDn Home/g End/G · body {min(len(body), scroll_offset + 1)}-{min(len(body), scroll_offset + body_height)}/{len(body)}{RESET}"]
    footer = scroll_footer + status_footer
    body_height = max(1, height - visible_header_rows - len(footer))
    max_scroll = max(0, len(body) - body_height)
    clamped_scroll = max(0, min(scroll_offset, max_scroll))
    visible_body = body[clamped_scroll : clamped_scroll + body_height]

    if scrollable:
        footer[0] = f"{DIM}Scroll: ↑/k ↓/j PgUp/PgDn Home/g End/G · body {clamped_scroll + 1}-{min(len(body), clamped_scroll + body_height)}/{len(body)}{RESET}"

    lines = header + visible_body + footer + [""]
    return "\n".join(lines), clamped_scroll, max_scroll


__all__ = ["render_frame"]
