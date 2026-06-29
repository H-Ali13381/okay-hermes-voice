"""Deterministic local classifications that should never hit a provider."""
from __future__ import annotations

from typing import Optional

from ..close_phrases import LOCAL_CLOSE_PHRASES, normalize_close_phrase
from ..interaction_types import AckTemplate, RequestComplexity, RouteTarget, RouterDecision, ToolRisk

_LOCAL_SIMPLE_CHAT_PHRASES = {
    "hello",
    "hi",
    "hey",
    "hey there",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how's it going",
    "hows it going",
    "thank you",
    "thanks",
    "thanks hermes",
}


def _simple_chat_decision() -> RouterDecision:
    return RouterDecision(
        request_complexity=RequestComplexity.SIMPLE,
        route_target=RouteTarget.SMALL_MODEL,
        ack_template_id=AckTemplate.NONE,
        tool_risk=ToolRisk.NONE,
        confidence=1.0,
        brief_reason="local_simple_chat",
    )


def classify_local_request(transcript: str) -> Optional[RouterDecision]:
    """Return a high-confidence local decision when a transcript is deterministic."""
    normalized = normalize_close_phrase(transcript)
    if normalized in LOCAL_CLOSE_PHRASES:
        return RouterDecision(
            request_complexity=RequestComplexity.IMMEDIATE,
            route_target=RouteTarget.IMMEDIATE_ONLY,
            ack_template_id=AckTemplate.NONE,
            tool_risk=ToolRisk.NONE,
            confidence=1.0,
            brief_reason="local_close_phrase",
        )
    if normalized in _LOCAL_SIMPLE_CHAT_PHRASES:
        return _simple_chat_decision()
    return None


__all__ = ["classify_local_request"]
