"""Execute the chosen voice request route."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..daemon_config import LOG
from ..interaction_router import RouteTarget, VoiceRequestPlan


def answer_routed_request(
    cfg: Dict[str, Any],
    transcript: str,
    plan: Optional[VoiceRequestPlan],
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]], str]:
    """Execute the selected route, falling back to heavy Hermes when needed."""
    history = list(conversation_history or [])
    if plan and plan.route.target is RouteTarget.SAFETY_FLOW:
        return "I can’t help with that request.", history, "safety_flow"
    if plan and plan.route.target is RouteTarget.ASK_CLARIFICATION:
        return "Could you clarify what you want me to do?", history, "ask_clarification"
    if plan and plan.route.target is RouteTarget.SMALL_MODEL:
        from . import answer_with_small_model, interaction_router_config_from_daemon_config
        router_cfg = interaction_router_config_from_daemon_config(cfg)
        response = answer_with_small_model(transcript, router_cfg)
        if response:
            history.extend([
                {"role": "user", "content": transcript},
                {"role": "assistant", "content": response},
            ])
            return response, history, "small_model"
        LOG.info("Small-model route produced no response; falling back to heavy Hermes")
    from . import ask_hermes_turn
    response, history = ask_hermes_turn(cfg, transcript, history, cancel_check=cancel_check)
    return response, history, "heavy_agent"


__all__ = ["answer_routed_request"]
