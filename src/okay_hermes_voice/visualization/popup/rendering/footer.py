"""Footer rendering for popup lifecycle state."""
from __future__ import annotations

import time
from typing import Any, Dict, List

from ..constants import DIM, FINAL_STATUSES, RESET
from ..lifecycle import final_keep_open_seconds


def render_status_footer_lines(state: Dict[str, Any], final_seen_at: float | None) -> List[str]:
    status = str(state.get("status") or "listening")
    if status not in FINAL_STATUSES:
        return [f"{DIM}Press Ctrl-C here to stop this voice session and close this window.{RESET}"]

    keep_open = final_keep_open_seconds(state)
    if final_seen_at is not None and keep_open > 0:
        remaining = max(0, int(round(keep_open - (time.monotonic() - final_seen_at))))
        return [f"{DIM}Closing in {remaining}s. Ctrl-C closes now.{RESET}"]
    if keep_open <= 0:
        return [f"{DIM}Complete. Ctrl-C closes this window.{RESET}"]
    return []


__all__ = ["render_status_footer_lines"]
