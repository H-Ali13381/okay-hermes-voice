"""Visualization and archive publishing helpers for activation flow."""
from __future__ import annotations

from typing import Any, Dict, List

from .constants import PIPELINE_STAGE_MESSAGES
from .services import ActivationFlowServices


def publish_pipeline_stage(
    deps: ActivationFlowServices,
    visual_state: Any,
    stage: str,
    message: str | None = None,
    **updates: Any,
) -> None:
    """Publish a descriptive, user-visible voice pipeline stage to the popup."""
    deps.update_visualization_state(
        visual_state,
        status=stage,
        pipeline_stage=stage,
        message=message or PIPELINE_STAGE_MESSAGES.get(stage, stage),
        **updates,
    )


def publish_turn_timing(
    deps: ActivationFlowServices,
    visual_state: Any,
    activation_archive: Any,
    turn_timings: List[Dict[str, Any]],
    turn_timing: Dict[str, Any],
    archive_turns: List[Dict[str, Any]],
) -> None:
    """Expose the latest turn timing to the popup and durable activation archive."""
    stable_timing = dict(turn_timing)
    turn_timings.append(stable_timing)
    deps.update_visualization_state(
        visual_state,
        latest_turn_timing=stable_timing,
        turn_timings=turn_timings,
    )
    deps.update_activation_archive_metadata(
        activation_archive,
        latest_turn_timing=stable_timing,
        turn_timings=turn_timings,
        turns=archive_turns,
    )
