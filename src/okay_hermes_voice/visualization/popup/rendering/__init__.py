"""Terminal popup rendering facade."""
from __future__ import annotations

from . import size as _size_mod
from .body import render_body_lines
from .footer import render_status_footer_lines
from .frame import render_frame
from .header import render_header_lines
from .pipeline import render_pipeline_lines
from .render import render
from .size import term_size
from .time_format import format_time
from .width_height import term_height, term_width
from .wrapping import wrapped_block

shutil = _size_mod.shutil

__all__ = [
    "format_time",
    "render",
    "render_body_lines",
    "render_frame",
    "render_header_lines",
    "render_pipeline_lines",
    "render_status_footer_lines",
    "term_height",
    "term_size",
    "term_width",
    "wrapped_block",
]
