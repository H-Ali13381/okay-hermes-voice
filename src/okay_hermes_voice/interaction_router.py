"""Compatibility facade for voice interaction routing.

The implementation lives in focused modules; this file preserves the original
`okay_hermes_voice.interaction_router` import surface.
"""
from __future__ import annotations

from .interaction_ack_cache import ACK_AUDIO_SUFFIXES, ACK_TEXT, AcknowledgementCache
from .interaction_clients import (
    answer_with_small_model,
    build_router_messages,
    build_small_model_messages,
    classify_request,
    classify_with_client,
)
from .interaction_routes import LOCAL_CLOSE_PHRASES, _ack_or_default, choose_voice_route, plan_voice_request
from .interaction_types import (
    AckTemplate,
    InteractionRouterConfig,
    RequestComplexity,
    RouteTarget,
    RouterDecision,
    ToolRisk,
    VoiceRequestPlan,
    VoiceRoute,
    _as_bool,
    _clamp_confidence,
    _enum_value,
)

__all__ = [
    "ACK_AUDIO_SUFFIXES",
    "ACK_TEXT",
    "AcknowledgementCache",
    "AckTemplate",
    "InteractionRouterConfig",
    "LOCAL_CLOSE_PHRASES",
    "RequestComplexity",
    "RouteTarget",
    "RouterDecision",
    "ToolRisk",
    "VoiceRequestPlan",
    "VoiceRoute",
    "answer_with_small_model",
    "build_router_messages",
    "build_small_model_messages",
    "choose_voice_route",
    "classify_request",
    "classify_with_client",
    "plan_voice_request",
]
