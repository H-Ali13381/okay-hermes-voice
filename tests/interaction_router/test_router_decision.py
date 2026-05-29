from __future__ import annotations

import json

from okay_hermes_voice.interaction_router import (
    AckTemplate,
    RequestComplexity,
    RouteTarget,
    RouterDecision,
    ToolRisk,
)


def test_parse_invalid_json_falls_back_to_unclear_heavy():
    decision = RouterDecision.parse("not json")

    assert decision.request_complexity is RequestComplexity.UNCLEAR
    assert decision.route_target is RouteTarget.HEAVY_AGENT
    assert decision.ack_template_id is AckTemplate.GOT_IT
    assert decision.confidence == 0.0
    assert "invalid_json" in decision.brief_reason

def test_parse_unknown_values_fall_back_safely():
    decision = RouterDecision.parse(
        json.dumps(
            {
                "request_complexity": "definitely_easy",
                "route_target": "do_everything",
                "ack_template_id": "promise_success",
                "tool_risk": "whatever",
                "confidence": 2.0,
            }
        )
    )

    assert decision.request_complexity is RequestComplexity.UNCLEAR
    assert decision.route_target is RouteTarget.HEAVY_AGENT
    assert decision.ack_template_id is AckTemplate.GOT_IT
    assert decision.tool_risk is ToolRisk.UNKNOWN
    assert decision.confidence == 1.0
