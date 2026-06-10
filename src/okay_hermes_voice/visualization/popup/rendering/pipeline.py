"""Pipeline progress rendering."""
from __future__ import annotations

from typing import Any, Dict, List

from ..constants import BOLD, DIM, PIPELINE_STAGE_INDEX, PIPELINE_STAGES, RESET, STATUS_COLORS


def render_pipeline_lines(state: Dict[str, Any]) -> List[str]:
    stage = str(state.get("pipeline_stage") or state.get("status") or "")
    if stage not in PIPELINE_STAGE_INDEX:
        return []
    current_index = PIPELINE_STAGE_INDEX[stage]
    done = str(state.get("status") or "") == "done"
    parts = []
    for idx, (stage_id, label) in enumerate(PIPELINE_STAGES):
        if done or idx < current_index:
            parts.append(f"[32m✓ {label}{RESET}")
        elif idx == current_index:
            color = STATUS_COLORS.get(stage_id, "")
            parts.append(f"{BOLD}{color}● {label}{RESET}")
        else:
            parts.append(f"{DIM}○ {label}{RESET}")
    return ["", f"{BOLD}[34mPipeline{RESET}", "  " + " → ".join(parts)]


__all__ = ["render_pipeline_lines"]
