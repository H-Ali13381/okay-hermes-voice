"""Local and engine-backed router classification entrypoint."""
from __future__ import annotations

from ..interaction_types import InteractionRouterConfig, RouterDecision
from .intent_engine import DEFAULT_INTENT_ENGINE, IntentClassificationEngine
from .local_classification import classify_local_request


def classify_request(
    transcript: str,
    cfg: InteractionRouterConfig,
    *,
    engine: IntentClassificationEngine | None = None,
) -> RouterDecision:
    local_decision = classify_local_request(transcript)
    if local_decision is not None:
        return local_decision
    return (engine or DEFAULT_INTENT_ENGINE).classify(transcript, cfg)


__all__ = ["classify_request"]
