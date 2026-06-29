"""Choose the deterministic execution route for a router decision."""
from __future__ import annotations

from ..interaction_ack_cache import AckTemplate
from ..interaction_types import InteractionRouterConfig, RequestComplexity, RouteTarget, RouterDecision, ToolRisk, VoiceRoute
from .ack_default import _ack_or_default
from .close_phrases import LOCAL_CLOSE_PHRASES, normalize_close_phrase


def choose_voice_route(
    transcript: str,
    decision: RouterDecision,
    cfg: InteractionRouterConfig,
) -> VoiceRoute:
    normalized = normalize_close_phrase(transcript)
    if normalized in LOCAL_CLOSE_PHRASES:
        return VoiceRoute(RouteTarget.IMMEDIATE_ONLY, AckTemplate.NONE, "local_close_phrase")

    if decision.request_complexity is RequestComplexity.UNSAFE or decision.route_target is RouteTarget.SAFETY_FLOW:
        return VoiceRoute(RouteTarget.SAFETY_FLOW, AckTemplate.NONE, "safety_flow")

    if decision.confidence < cfg.router_min_confidence:
        return VoiceRoute(RouteTarget.HEAVY_AGENT, AckTemplate.GOT_IT, "low_router_confidence")

    if decision.route_target is RouteTarget.HEAVY_AGENT:
        return VoiceRoute(RouteTarget.HEAVY_AGENT, _ack_or_default(decision), "router_heavy_agent")

    if decision.tool_risk in {ToolRisk.SIDE_EFFECT, ToolRisk.IRREVERSIBLE, ToolRisk.UNKNOWN}:
        return VoiceRoute(RouteTarget.HEAVY_AGENT, _ack_or_default(decision), "side_effect_or_unknown_tool_risk")

    if decision.requires_tools or decision.requires_memory or decision.requires_external_data:
        return VoiceRoute(RouteTarget.HEAVY_AGENT, _ack_or_default(decision), "requires_heavy_capability")

    if decision.route_target is RouteTarget.ASK_CLARIFICATION:
        return VoiceRoute(RouteTarget.ASK_CLARIFICATION, AckTemplate.NONE, "router_clarification")

    if decision.route_target is RouteTarget.SMALL_MODEL:
        if decision.request_complexity is not RequestComplexity.SIMPLE:
            return VoiceRoute(RouteTarget.HEAVY_AGENT, _ack_or_default(decision), "non_simple_small_model_suggestion")
        if decision.tool_risk is not ToolRisk.NONE:
            return VoiceRoute(RouteTarget.HEAVY_AGENT, _ack_or_default(decision), "tool_risk_small_model_suggestion")
        if not cfg.small_model_enabled:
            return VoiceRoute(RouteTarget.HEAVY_AGENT, _ack_or_default(decision), "small_model_disabled")
        return VoiceRoute(RouteTarget.SMALL_MODEL, decision.ack_template_id, "router_small_model")

    if decision.route_target is RouteTarget.IMMEDIATE_ONLY:
        return VoiceRoute(RouteTarget.IMMEDIATE_ONLY, decision.ack_template_id, "router_immediate_only")

    return VoiceRoute(RouteTarget.HEAVY_AGENT, _ack_or_default(decision), "router_heavy_agent")


__all__ = ["choose_voice_route"]
