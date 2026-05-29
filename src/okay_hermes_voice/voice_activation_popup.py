#!/usr/bin/env python3
"""Small terminal visualizer for the Okay Hermes wakeword daemon.

The daemon writes a JSON state file while it records, transcribes, thinks,
and speaks. This script runs inside a popped-up terminal window and renders
that state as a lightweight CLI dashboard. It intentionally does not invoke
Hermes itself; the daemon keeps using its warm in-process agent for latency.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import select
import shutil
import sys
import termios
import textwrap
import time
import tty
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

FINAL_STATUSES = {"done", "error", "cancelled"}
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
STATUS_LABELS = {
    "listening": "Listening for your request",
    "transcribing": "Transcribing speech",
    "thinking": "Hermes is thinking",
    "speaking": "Speaking response",
    "cancel_requested": "Stopping voice session",
    "done": "Done",
    "error": "Needs attention",
    "cancelled": "Cancelled",
}
STATUS_COLORS = {
    "listening": "\033[36m",
    "transcribing": "\033[35m",
    "thinking": "\033[33m",
    "speaking": "\033[32m",
    "cancel_requested": "\033[31m",
    "done": "\033[32m",
    "error": "\033[31m",
    "cancelled": "\033[31m",
}
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
ALT_SCREEN_ENTER = "\033[?1049h\033[?25l"
ALT_SCREEN_EXIT = "\033[?25h\033[?1049l"
MOUSE_CAPTURE_ENTER = "\033[?1007h\033[?1000h\033[?1006h"
MOUSE_CAPTURE_EXIT = "\033[?1006l\033[?1000l\033[?1007l"
POLL_INTERVAL_SECONDS = 0.2
SCROLL_KEY_SEQUENCES = {
    "\033[A": "up",
    "\033[B": "down",
    "\033[5~": "page_up",
    "\033[6~": "page_down",
    "\033[H": "home",
    "\033[F": "end",
    "\033OH": "home",
    "\033OF": "end",
}
SCROLL_KEY_CHARS = {
    "k": "up",
    "j": "down",
    "g": "home",
    "G": "end",
}
RENDER_FINGERPRINT_IGNORED_KEYS = {
    # These fields are useful for logs/daemon bookkeeping, but repainting only
    # because they changed makes the terminal scrollback fill with duplicate
    # visible frames.
    "activation_archive",
    "cancel_reason",
    "cancel_requested",
    "cancel_requested_at",
    "completed_at",
    "current_turn",
    "interaction_ack_text",
    "updated_at",
    "visualization_launch_error",
    "visualization_terminal",
}


def load_state(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "listening", "message": "Waiting for wakeword state…"}
    except Exception as exc:
        return {"status": "error", "error": f"Could not read state file: {exc}"}


def request_cancel(path: Path, reason: str = "ctrl_c") -> None:
    """Request cancellation of the active daemon-owned voice session.

    The popup is only a visualizer, so Ctrl-C cannot signal the systemd daemon
    directly through process ancestry.  Instead it atomically marks the shared
    state file; the daemon polls that file while recording/following up.
    """
    path = Path(path).expanduser()
    try:
        state: Dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    state = loaded
            except Exception:
                state = {}
        state.update(
            {
                "status": "cancel_requested",
                "message": "Ctrl-C pressed in the Hermes Voice window. Stopping this voice session…",
                "cancel_requested": True,
                "cancel_reason": reason,
                "cancel_requested_at": time.time(),
                "error": "",
            }
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        print(f"Could not request voice-session cancellation: {exc}", file=sys.stderr)


def term_size() -> Tuple[int, int]:
    size = shutil.get_terminal_size((96, 28))
    width = max(60, min(size.columns, 140))
    height = max(10, size.lines)
    return width, height


def term_width() -> int:
    return term_size()[0]


def term_height() -> int:
    return term_size()[1]


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


def format_time(epoch: Any) -> str:
    try:
        return _dt.datetime.fromtimestamp(float(epoch)).strftime("%H:%M:%S")
    except Exception:
        return "--:--:--"


def render_fingerprint_state(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: render_fingerprint_state(item)
            for key, item in value.items()
            if key not in RENDER_FINGERPRINT_IGNORED_KEYS
        }
    if isinstance(value, list):
        return [render_fingerprint_state(item) for item in value]
    return value


def state_fingerprint(state: Dict[str, Any]) -> str:
    """Stable identity for deciding whether a popup frame needs repainting."""
    try:
        return json.dumps(
            render_fingerprint_state(state),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        return repr(render_fingerprint_state(state))


def render_header_lines(state: Dict[str, Any], tick: int, width: int) -> List[str]:
    status = str(state.get("status") or "listening")
    color = STATUS_COLORS.get(status, "")
    label = STATUS_LABELS.get(status, status.replace("_", " ").title())
    spinner = "✓" if status == "done" else "!" if status == "error" else SPINNER[tick % len(SPINNER)]
    title = str(state.get("title") or "Hermes Voice")
    probability = state.get("probability")
    activated_at = state.get("activated_at")
    updated_at = state.get("updated_at")
    rule = "═" * min(width, 96)
    meta = [f"activated {format_time(activated_at)}", f"updated {format_time(updated_at)}"]
    if isinstance(probability, (int, float)):
        meta.append(f"wake score {probability:.3f}")
    return [
        "\033[2J\033[H",
        f"{BOLD}{color}{title}{RESET}",
        f"{DIM}{rule}{RESET}",
        f"{color}{spinner} {label}{RESET}",
        f"{DIM}{' · '.join(meta)}{RESET}",
    ]


def render_body_lines(state: Dict[str, Any], width: int) -> List[str]:
    status = str(state.get("status") or "listening")
    color = STATUS_COLORS.get(status, "")
    lines: List[str] = []

    message = state.get("message")
    if message:
        lines.append("")
        lines.extend(wrapped_block("Status", str(message), width, color))

    turns_raw = state.get("turns")
    turns = turns_raw if isinstance(turns_raw, list) else []
    if turns:
        lines.append("")
        lines.append(f"{BOLD}\033[34mConversation{RESET}")
        body_width = max(20, width - 6)
        for idx, turn in enumerate(turns[-8:], start=max(1, len(turns) - 7)):
            if not isinstance(turn, dict):
                continue
            transcript_text = str(turn.get("transcript") or "").strip()
            response_text = str(turn.get("response") or "").strip()
            if transcript_text:
                for line in textwrap.wrap(f"You: {transcript_text}", width=body_width, replace_whitespace=False):
                    lines.append(f"  \033[36m{line}{RESET}")
            if response_text:
                for line in textwrap.wrap(f"Hermes: {response_text}", width=body_width, replace_whitespace=False):
                    lines.append(f"  \033[32m{line}{RESET}")
            if idx != len(turns):
                lines.append("")

    transcript = str(state.get("transcript") or "").strip()
    response = str(state.get("response") or "").strip()
    latest_turn = turns[-1] if turns and isinstance(turns[-1], dict) else {}
    latest_is_rendered = bool(
        latest_turn
        and transcript
        and response
        and transcript == str(latest_turn.get("transcript") or "").strip()
        and response == str(latest_turn.get("response") or "").strip()
    )
    if transcript and not latest_is_rendered:
        lines.append("")
        lines.extend(wrapped_block("Request", transcript, width, "\033[36m"))

    if response and not latest_is_rendered:
        lines.append("")
        lines.extend(wrapped_block("Hermes", response, width, "\033[32m"))

    error = str(state.get("error") or "").strip()
    if error:
        lines.append("")
        lines.extend(wrapped_block("Error", error, width, "\033[31m"))

    return lines


def render_status_footer_lines(state: Dict[str, Any], final_seen_at: float | None) -> List[str]:
    status = str(state.get("status") or "listening")
    if status not in FINAL_STATUSES:
        return [f"{DIM}Press Ctrl-C here to stop this voice session and close this window.{RESET}"]

    keep_open = float(state.get("keep_open_seconds") or 45.0)
    if final_seen_at is not None and keep_open > 0:
        remaining = max(0, int(round(keep_open - (time.monotonic() - final_seen_at))))
        return [f"{DIM}Closing in {remaining}s. Ctrl-C closes now.{RESET}"]
    if keep_open <= 0:
        return [f"{DIM}Complete. Ctrl-C closes this window.{RESET}"]
    return []


def render_frame(
    state: Dict[str, Any],
    tick: int,
    final_seen_at: float | None,
    scroll_offset: int = 0,
) -> Tuple[str, int, int]:
    width, height = term_size()
    header = render_header_lines(state, tick, width)
    body = render_body_lines(state, width)
    status_footer = render_status_footer_lines(state, final_seen_at)

    # The clear-screen escape is not a visible row. Keep the header pinned and
    # scroll only the body, so long Hermes responses behave like a real TUI pane
    # instead of relying on kitty's normal scrollback.
    visible_header_rows = max(0, len(header) - 1)
    body_height = max(1, height - visible_header_rows - len(status_footer))
    scrollable = len(body) > body_height
    scroll_footer = []
    if scrollable:
        scroll_footer = [
            (
                f"{DIM}Scroll: ↑/k ↓/j PgUp/PgDn Home/g End/G · body "
                f"{min(len(body), scroll_offset + 1)}-"
                f"{min(len(body), scroll_offset + body_height)}/{len(body)}{RESET}"
            )
        ]
    footer = scroll_footer + status_footer
    body_height = max(1, height - visible_header_rows - len(footer))
    max_scroll = max(0, len(body) - body_height)
    clamped_scroll = max(0, min(scroll_offset, max_scroll))
    visible_body = body[clamped_scroll : clamped_scroll + body_height]

    if scrollable:
        footer[0] = (
            f"{DIM}Scroll: ↑/k ↓/j PgUp/PgDn Home/g End/G · body "
            f"{clamped_scroll + 1}-{min(len(body), clamped_scroll + body_height)}/{len(body)}{RESET}"
        )

    lines = header + visible_body + footer + [""]
    return "\n".join(lines), clamped_scroll, max_scroll


def render(state: Dict[str, Any], tick: int, final_seen_at: float | None, scroll_offset: int = 0) -> str:
    return render_frame(state, tick, final_seen_at, scroll_offset)[0]


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


def run(path: Path) -> int:
    final_seen_at: float | None = None
    last_rendered_fingerprint: str | None = None
    last_rendered_scroll_offset: int | None = None
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
            # only write when the daemon-published visible state or the TUI viewport
            # changes. The alternate screen keeps full-frame redraws out of normal
            # scrollback; the viewport makes long output scroll inside the popup.
            frame, clamped_scroll_offset, _max_scroll = render_frame(state, tick, final_seen_at, scroll_offset)
            if fingerprint != last_rendered_fingerprint or clamped_scroll_offset != last_rendered_scroll_offset:
                sys.stdout.write(frame)
                sys.stdout.flush()
                last_rendered_fingerprint = fingerprint
                last_rendered_scroll_offset = clamped_scroll_offset
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


if __name__ == "__main__":
    raise SystemExit(main())
