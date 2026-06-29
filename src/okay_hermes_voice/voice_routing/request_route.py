"""Final-transcript routing and acknowledgement dispatch."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..interaction_router import AckTemplate, VoiceRequestPlan


def route_transcribed_request(
    cfg: Dict[str, Any],
    transcript: str,
    cancel_check: Optional[Callable[[], bool]] = None,
    *,
    loop_ack_until_cancelled: bool = False,
) -> Optional[VoiceRequestPlan]:
    """Plan routing for a final STT transcript and play any immediate acknowledgement."""
    from . import plan_interaction_route, play_interaction_ack

    provisional_ack_started = False
    if loop_ack_until_cancelled:
        provisional_ack_started = play_interaction_ack(
            cfg,
            AckTemplate.GOT_IT,
            cancel_check=cancel_check,
            block=False,
            loop_until_cancelled=True,
        )

    plan = plan_interaction_route(cfg, transcript)
    if plan is None:
        return None
    if not provisional_ack_started and plan.route.ack_template_id is not AckTemplate.NONE:
        play_interaction_ack(
            cfg,
            plan.route.ack_template_id,
            cancel_check=cancel_check,
            block=False,
            loop_until_cancelled=loop_ack_until_cancelled,
        )
    return plan


__all__ = ["route_transcribed_request"]
