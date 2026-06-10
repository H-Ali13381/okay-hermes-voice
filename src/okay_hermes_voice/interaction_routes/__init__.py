"""Deterministic route selection facade for classified voice requests."""
from __future__ import annotations

from ..interaction_clients import classify_request
from .ack_default import _ack_or_default
from .choose import choose_voice_route
from .close_phrases import LOCAL_CLOSE_PHRASES
from .plan import plan_voice_request

__all__ = [
    "LOCAL_CLOSE_PHRASES",
    "_ack_or_default",
    "choose_voice_route",
    "classify_request",
    "plan_voice_request",
]
