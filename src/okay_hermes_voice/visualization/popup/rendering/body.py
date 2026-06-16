"""Body rendering for popup status, turns, response, and errors."""
from __future__ import annotations

import textwrap
from typing import Any, Dict, List

from ..constants import BOLD, RESET, STATUS_COLORS
from .pipeline import render_pipeline_lines
from .wrapping import wrapped_block


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
        lines.append(f"{BOLD}[34mConversation{RESET}")
        body_width = max(20, width - 6)
        for idx, turn in enumerate(turns[-8:], start=max(1, len(turns) - 7)):
            if not isinstance(turn, dict):
                continue
            transcript_text = str(turn.get("transcript") or "").strip()
            response_text = str(turn.get("response") or "").strip()
            if transcript_text:
                for line in textwrap.wrap(f"You: {transcript_text}", width=body_width, replace_whitespace=False):
                    lines.append(f"  [36m{line}{RESET}")
            if response_text:
                for line in textwrap.wrap(f"Hermes: {response_text}", width=body_width, replace_whitespace=False):
                    lines.append(f"  [32m{line}{RESET}")
            if idx != len(turns):
                lines.append("")

    transcript = str(state.get("transcript") or "").strip()
    response = str(state.get("response") or "").strip()
    latest_turn = turns[-1] if turns and isinstance(turns[-1], dict) else {}
    latest_is_rendered = bool(latest_turn and transcript and response and transcript == str(latest_turn.get("transcript") or "").strip() and response == str(latest_turn.get("response") or "").strip())
    if transcript and not latest_is_rendered:
        lines.append("")
        lines.extend(wrapped_block("Request", transcript, width, "[36m"))
    if response and not latest_is_rendered:
        lines.append("")
        lines.extend(wrapped_block("Hermes", response, width, "[32m"))

    error = str(state.get("error") or "").strip()
    if error:
        lines.append("")
        lines.extend(wrapped_block("Error", error, width, "[31m"))
    return lines


__all__ = ["render_body_lines"]
