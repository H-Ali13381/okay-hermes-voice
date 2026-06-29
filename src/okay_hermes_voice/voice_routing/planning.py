"""Classify a transcript and log the planned interaction route."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..daemon_config import LOG
from ..interaction_router import VoiceRequestPlan
from .router_config import interaction_router_config_from_daemon_config


def plan_interaction_route(cfg: Dict[str, Any], transcript: str) -> Optional[VoiceRequestPlan]:
    """Classify a transcript and choose the deterministic voice route."""
    router_cfg = interaction_router_config_from_daemon_config(cfg)
    if not router_cfg.router_enabled:
        return None
    started = time.monotonic()
    from . import plan_voice_request
    plan = plan_voice_request(transcript, router_cfg)
    LOG.info(
        "Interaction router lane=%s target=%s ack=%s reason=%s confidence=%.2f latency=%.3fs router_reason=%s",
        plan.route_lane.value,
        plan.route.target.value,
        plan.route.ack_template_id.value,
        plan.route.reason,
        plan.decision.confidence,
        time.monotonic() - started,
        plan.decision.brief_reason,
    )
    return plan


__all__ = ["plan_interaction_route"]
