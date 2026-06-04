"""Terminal popup rendering and viewport layout."""
from __future__ import annotations

import datetime as _dt
import shutil
import textwrap
import time
from typing import Any, Dict, List, Tuple

from .constants import (
    BOLD,
    DIM,
    FINAL_STATUSES,
    PIPELINE_STAGE_INDEX,
    PIPELINE_STAGES,
    RESET,
    SPINNER,
    STATUS_COLORS,
    STATUS_LABELS,
)


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


def render_pipeline_lines(state: Dict[str, Any]) -> List[str]:
    stage = str(state.get("pipeline_stage") or state.get("status") or "")
    if stage not in PIPELINE_STAGE_INDEX:
        return []
    current_index = PIPELINE_STAGE_INDEX[stage]
    done = str(state.get("status") or "") == "done"
    parts = []
    for idx, (stage_id, label) in enumerate(PIPELINE_STAGES):
        if done or idx < current_index:
            parts.append(f"\033[32m✓ {label}{RESET}")
        elif idx == current_index:
            color = STATUS_COLORS.get(stage_id, "")
            parts.append(f"{BOLD}{color}● {label}{RESET}")
        else:
            parts.append(f"{DIM}○ {label}{RESET}")
    return ["", f"{BOLD}\033[34mPipeline{RESET}", "  " + " → ".join(parts)]


def render_body_lines(state: Dict[str, Any], width: int) -> List[str]:
    status = str(state.get("status") or "listening")
    color = STATUS_COLORS.get(status, "")
    lines: List[str] = []

    lines.extend(render_pipeline_lines(state))

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
