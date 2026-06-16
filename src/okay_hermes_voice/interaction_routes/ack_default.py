"""Acknowledgement defaults for routed voice requests."""
from __future__ import annotations

from ..interaction_ack_cache import AckTemplate
from ..interaction_types import RouterDecision


def _ack_or_default(decision: RouterDecision) -> AckTemplate:
    if decision.ack_template_id is AckTemplate.NONE:
        return AckTemplate.GOT_IT
    return decision.ack_template_id


__all__ = ["_ack_or_default"]
