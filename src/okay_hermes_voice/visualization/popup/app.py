"""Terminal popup application loop and CLI entrypoint."""
from __future__ import annotations

import argparse
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Iterable

from .constants import (
    ALT_SCREEN_ENTER,
    ALT_SCREEN_EXIT,
    FINAL_STATUSES,
    MOUSE_CAPTURE_ENTER,
    MOUSE_CAPTURE_EXIT,
    POLL_INTERVAL_SECONDS,
    SPINNER,
)
from .input import apply_scroll_key, read_keypress, write_terminal_control
from .rendering import render_frame
from .state import load_state, request_cancel, state_fingerprint


def run(path: Path) -> int:
    final_seen_at: float | None = None
    last_rendered_fingerprint: str | None = None
    last_rendered_scroll_offset: int | None = None
    last_rendered_spinner_frame: int | None = None
    scroll_offset = 0
    tick = 0
    stdin_fd: int | None = None
    original_tty_attrs = None
    write_terminal_control(ALT_SCREEN_ENTER + MOUSE_CAPTURE_ENTER)
    try:
        try:
            if sys.stdin.isatty():
                fd = sys.stdin.fileno()
                original_attrs = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                stdin_fd = fd
                original_tty_attrs = original_attrs
        except Exception:
            stdin_fd = None
            original_tty_attrs = None

        while True:
            state = load_state(path)
            status = str(state.get("status") or "listening")
            fingerprint = state_fingerprint(state)
            scroll_offset = apply_scroll_key(scroll_offset, read_keypress())

            if status in FINAL_STATUSES and final_seen_at is None:
                final_seen_at = time.monotonic()

            # Repainting identical frames every poll makes kitty keep jumping to the
            # bottom and fills scrollback with duplicate dashboards. Poll often, but
            # only write when the daemon-published visible state, the TUI viewport,
            # or the active-state spinner frame changes. The alternate screen keeps
            # full-frame redraws out of normal scrollback; the viewport makes long
            # output scroll inside the popup.
            frame, clamped_scroll_offset, _max_scroll = render_frame(state, tick, final_seen_at, scroll_offset)
            spinner_frame = tick % len(SPINNER) if status not in FINAL_STATUSES else None
            if (
                fingerprint != last_rendered_fingerprint
                or clamped_scroll_offset != last_rendered_scroll_offset
                or spinner_frame != last_rendered_spinner_frame
            ):
                sys.stdout.write(frame)
                sys.stdout.flush()
                last_rendered_fingerprint = fingerprint
                last_rendered_scroll_offset = clamped_scroll_offset
                last_rendered_spinner_frame = spinner_frame
            scroll_offset = clamped_scroll_offset

            if final_seen_at is not None:
                keep_open = float(state.get("keep_open_seconds") or 45.0)
                if keep_open > 0 and time.monotonic() - final_seen_at >= keep_open:
                    break

            tick += 1
            time.sleep(POLL_INTERVAL_SECONDS)
    finally:
        if stdin_fd is not None and original_tty_attrs is not None:
            try:
                termios.tcsetattr(stdin_fd, termios.TCSADRAIN, original_tty_attrs)
            except Exception:
                pass
        write_terminal_control(MOUSE_CAPTURE_EXIT + ALT_SCREEN_EXIT)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render Okay Hermes wakeword state in a terminal window")
    parser.add_argument("--state", required=True, help="Path to daemon-written JSON state file")
    args = parser.parse_args(list(argv) if argv is not None else None)
    state_path = Path(args.state).expanduser()
    try:
        return run(state_path)
    except KeyboardInterrupt:
        request_cancel(state_path, reason="ctrl_c")
        return 0


__all__ = ["main", "run"]
