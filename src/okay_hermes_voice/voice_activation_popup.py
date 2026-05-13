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
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


def term_width() -> int:
    return max(60, min(shutil.get_terminal_size((96, 28)).columns, 140))


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


def render(state: Dict[str, Any], tick: int, final_seen_at: float | None) -> str:
    width = term_width()
    status = str(state.get("status") or "listening")
    color = STATUS_COLORS.get(status, "")
    label = STATUS_LABELS.get(status, status.replace("_", " ").title())
    spinner = "✓" if status == "done" else "!" if status == "error" else SPINNER[tick % len(SPINNER)]
    title = str(state.get("title") or "Hermes Voice")
    probability = state.get("probability")
    activated_at = state.get("activated_at")
    updated_at = state.get("updated_at")

    rule = "═" * min(width, 96)
    lines: List[str] = [
        "\033[2J\033[H",
        f"{BOLD}{color}{title}{RESET}",
        f"{DIM}{rule}{RESET}",
        f"{color}{spinner} {label}{RESET}",
    ]
    meta = [f"activated {format_time(activated_at)}", f"updated {format_time(updated_at)}"]
    if isinstance(probability, (int, float)):
        meta.append(f"wake score {probability:.3f}")
    lines.append(f"{DIM}{' · '.join(meta)}{RESET}")

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

    if status not in FINAL_STATUSES:
        lines.append("")
        lines.append(f"{DIM}Press Ctrl-C here to stop this voice session and close this window.{RESET}")
    else:
        keep_open = float(state.get("keep_open_seconds") or 45.0)
        if final_seen_at is not None and keep_open > 0:
            remaining = max(0, int(round(keep_open - (time.monotonic() - final_seen_at))))
            lines.append("")
            lines.append(f"{DIM}Closing in {remaining}s. Ctrl-C closes now.{RESET}")
        elif keep_open <= 0:
            lines.append("")
            lines.append(f"{DIM}Complete. Ctrl-C closes this window.{RESET}")

    lines.append("")
    return "\n".join(lines)


def run(path: Path) -> int:
    final_seen_at: float | None = None
    tick = 0
    while True:
        state = load_state(path)
        status = str(state.get("status") or "listening")
        if status in FINAL_STATUSES and final_seen_at is None:
            final_seen_at = time.monotonic()
        sys.stdout.write(render(state, tick, final_seen_at))
        sys.stdout.flush()

        if final_seen_at is not None:
            keep_open = float(state.get("keep_open_seconds") or 45.0)
            if keep_open > 0 and time.monotonic() - final_seen_at >= keep_open:
                break
        tick += 1
        time.sleep(0.2)
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
