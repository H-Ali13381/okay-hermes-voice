"""Swappable intent classification engines for voice routing."""
from __future__ import annotations

from typing import Protocol

from ..interaction_types import InteractionRouterConfig, RouterDecision
from .router_call import classify_with_client
from .router_client_cache import RouterClientCache, router_client_cache


class IntentClassificationEngine(Protocol):
    """Interface for transcript-to-router-decision engines.

    The default implementation is LLM-backed today. A future BERT/ONNX intent
    model should implement this same contract and can be selected at this seam.
    """

    def classify(self, transcript: str, cfg: InteractionRouterConfig) -> RouterDecision:
        """Classify a final transcript into the router decision schema."""
        ...


class LlmIntentClassificationEngine:
    """LLM-backed intent classifier used until a local BERT-style router is ready."""

    def __init__(self, client_cache: RouterClientCache = router_client_cache) -> None:
        self.client_cache = client_cache

    def classify(self, transcript: str, cfg: InteractionRouterConfig) -> RouterDecision:
        if not cfg.router_enabled:
            return RouterDecision(brief_reason="router_disabled")
        cached = self.client_cache.get(cfg)
        if cached is not None:
            return classify_with_client(cached.client, cached.resolved_model, transcript, cfg)
        try:
            client, model = self.client_cache.resolve(cfg)
        except Exception as exc:
            return RouterDecision(brief_reason=f"provider_resolution_failed:{type(exc).__name__}")
        if client is None or not model:
            return RouterDecision(brief_reason="provider_resolution_returned_none")
        self.client_cache.store(cfg, client, model)
        return classify_with_client(client, model, transcript, cfg)


DEFAULT_INTENT_ENGINE = LlmIntentClassificationEngine()


__all__ = [
    "DEFAULT_INTENT_ENGINE",
    "IntentClassificationEngine",
    "LlmIntentClassificationEngine",
]
