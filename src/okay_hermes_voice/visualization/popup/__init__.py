"""Public facade for the terminal popup visualizer.

Boundary:
- callers import stable popup rendering/control names from this package
- state-file writes, rendering, input, and app-loop mechanics live in leaves
"""
from __future__ import annotations

from .app import main, run
from .input import apply_scroll_key, normalized_escape_key, read_keypress, write_terminal_control
from .rendering import (
    format_time,
    render,
    render_body_lines,
    render_frame,
    render_header_lines,
    render_pipeline_lines,
    render_status_footer_lines,
    term_height,
    term_size,
    term_width,
    wrapped_block,
)
from .state import load_state, render_fingerprint_state, request_cancel, state_fingerprint

__all__ = [
    "apply_scroll_key",
    "format_time",
    "load_state",
    "main",
    "normalized_escape_key",
    "read_keypress",
    "render",
    "render_body_lines",
    "render_fingerprint_state",
    "render_frame",
    "render_header_lines",
    "render_pipeline_lines",
    "render_status_footer_lines",
    "request_cancel",
    "run",
    "state_fingerprint",
    "term_height",
    "term_size",
    "term_width",
    "wrapped_block",
    "write_terminal_control",
]
