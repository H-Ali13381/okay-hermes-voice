#!/usr/bin/env python3
"""Compatibility facade and script entrypoint for the popup visualizer.

Boundary:
- ``okay_hermes_voice.voice_activation_popup`` remains the historical import
  path and executable script target
- terminal rendering/control mechanics live under ``visualization.popup``
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    # The daemon launches this file by path. Add the package's ``src`` root so
    # absolute package imports still work under script-style execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from okay_hermes_voice.visualization.popup import (  # type: ignore[no-redef]
        apply_scroll_key,
        format_time,
        load_state,
        main,
        normalized_escape_key,
        read_keypress,
        render,
        render_body_lines,
        render_fingerprint_state,
        render_frame,
        render_header_lines,
        render_pipeline_lines,
        render_status_footer_lines,
        request_cancel,
        run,
        state_fingerprint,
        term_height,
        term_size,
        term_width,
        wrapped_block,
        write_terminal_control,
    )
else:
    from .visualization.popup import (
        apply_scroll_key,
        format_time,
        load_state,
        main,
        normalized_escape_key,
        read_keypress,
        render,
        render_body_lines,
        render_fingerprint_state,
        render_frame,
        render_header_lines,
        render_pipeline_lines,
        render_status_footer_lines,
        request_cancel,
        run,
        state_fingerprint,
        term_height,
        term_size,
        term_width,
        wrapped_block,
        write_terminal_control,
    )

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


if __name__ == "__main__":
    raise SystemExit(main())
