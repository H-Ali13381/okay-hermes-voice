"""Router and small-model client-call facade."""
from __future__ import annotations

from .intent_engine import IntentClassificationEngine, LlmIntentClassificationEngine
from .router_call import classify_with_client
from .router_classification import classify_request
from .router_messages import build_router_messages
from .router_prewarm import prewarm_interaction_router
from .small_model_answer import answer_with_small_model
from .small_model_messages import build_small_model_messages

__all__ = [
    "IntentClassificationEngine",
    "LlmIntentClassificationEngine",
    "answer_with_small_model",
    "build_router_messages",
    "build_small_model_messages",
    "classify_request",
    "classify_with_client",
    "prewarm_interaction_router",
]
