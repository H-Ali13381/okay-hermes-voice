"""Final-transcript routing and acknowledgement dispatch."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..interaction_router import AckTemplate, VoiceRequestPlan


def route_transcribed_request(
    cfg: Dict[str, Any],
    transcript: str,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[VoiceRequestPlan]:
    """Plan routing for a final STT transcript and play any immediate acknowledgement."""
    from . import plan_interaction_route, play_interaction_ack
    plan = plan_interaction_route(cfg, transcript)
    if plan is None:
        return None
    if plan.route.ack_template_id is not AckTemplate.NONE:
        play_interaction_ack(cfg, plan.route.ack_template_id, cancel_check=cancel_check, block=False)
    return plan


__all__ = ["route_transcribed_request"]
