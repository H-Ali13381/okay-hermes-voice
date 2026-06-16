"""Plan a voice request by classifying and choosing a route."""
from __future__ import annotations

from ..interaction_types import InteractionRouterConfig, VoiceRequestPlan
from .choose import choose_voice_route


def plan_voice_request(transcript: str, cfg: InteractionRouterConfig) -> VoiceRequestPlan:
    from . import classify_request
    decision = classify_request(transcript, cfg)
    route = choose_voice_route(transcript, decision, cfg)
    return VoiceRequestPlan(transcript=transcript, decision=decision, route=route)


__all__ = ["plan_voice_request"]
