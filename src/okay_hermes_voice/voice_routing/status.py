"""User-visible route status text."""
from __future__ import annotations

from typing import Optional

from ..interaction_router import AckTemplate, VoiceRequestPlan


def interaction_ack_text(plan: Optional[VoiceRequestPlan]) -> str:
    if plan is None or plan.route.ack_template_id is AckTemplate.NONE:
        return ""
    from . import ACK_TEXT
    return ACK_TEXT.get(plan.route.ack_template_id, "")


def routed_request_status_message(plan: Optional[VoiceRequestPlan]) -> str:
    route_target = plan.route.target.value if plan else "heavy_agent"
    route_label = route_target.replace("_", " ")
    ack_text = interaction_ack_text(plan)
    if ack_text:
        return f"{ack_text} Request routed to {route_label}. Handling it now…"
    return f"Request routed to {route_label}. Handling it now…"


__all__ = ["interaction_ack_text", "routed_request_status_message"]
