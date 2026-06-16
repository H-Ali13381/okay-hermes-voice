"""Header rendering for the popup frame."""
from __future__ import annotations

from typing import Any, Dict, List

from ..constants import BOLD, DIM, RESET, SPINNER, STATUS_COLORS, STATUS_LABELS
from .time_format import format_time


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
    return ["[2J[H", f"{BOLD}{color}{title}{RESET}", f"{DIM}{rule}{RESET}", f"{color}{spinner} {label}{RESET}", f"{DIM}{' · '.join(meta)}{RESET}"]


__all__ = ["render_header_lines"]
