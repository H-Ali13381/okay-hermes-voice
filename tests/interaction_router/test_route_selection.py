from __future__ import annotations

from okay_hermes_voice.interaction_router import (
    AckTemplate,
    InteractionRouterConfig,
    RequestComplexity,
    RouteTarget,
    RouterDecision,
    ToolRisk,
    choose_voice_route,
    plan_voice_request,
)


def test_close_phrase_is_immediate_only_even_without_router_confidence():
    cfg = InteractionRouterConfig()
    decision = RouterDecision(confidence=0.0)

    route = choose_voice_route("close voice mode", decision, cfg)

    assert route.target is RouteTarget.IMMEDIATE_ONLY
    assert route.ack_template_id is AckTemplate.NONE
    assert route.reason == "local_close_phrase"

def test_low_confidence_routes_to_heavy_agent_with_ack():
    cfg = InteractionRouterConfig(router_min_confidence=0.7)
    decision = RouterDecision(
        request_complexity=RequestComplexity.SIMPLE,
        route_target=RouteTarget.SMALL_MODEL,
        confidence=0.4,
    )

    route = choose_voice_route("what is the weather", decision, cfg)

    assert route.target is RouteTarget.HEAVY_AGENT
    assert route.ack_template_id is AckTemplate.GOT_IT
    assert route.reason == "low_router_confidence"

def test_small_model_requires_simple_safe_enabled_decision():
    cfg = InteractionRouterConfig(small_model_enabled=True)
    decision = RouterDecision(
        request_complexity=RequestComplexity.SIMPLE,
        route_target=RouteTarget.SMALL_MODEL,
        ack_template_id=AckTemplate.NONE,
        tool_risk=ToolRisk.NONE,
        confidence=0.95,
    )

    route = choose_voice_route("say a one sentence fun fact", decision, cfg)

    assert route.target is RouteTarget.SMALL_MODEL
    assert route.ack_template_id is AckTemplate.NONE
    assert route.reason == "router_small_model"

def test_complex_small_model_suggestion_routes_to_heavy_agent():
    cfg = InteractionRouterConfig(small_model_enabled=True)
    decision = RouterDecision(
        request_complexity=RequestComplexity.COMPLEX,
        route_target=RouteTarget.SMALL_MODEL,
        ack_template_id=AckTemplate.CHECKING,
        tool_risk=ToolRisk.NONE,
        confidence=0.99,
    )

    route = choose_voice_route("inspect the repository and fix the tests", decision, cfg)

    assert route.target is RouteTarget.HEAVY_AGENT
    assert route.ack_template_id is AckTemplate.CHECKING
    assert route.reason == "non_simple_small_model_suggestion"

def test_read_only_tool_risk_small_model_suggestion_routes_to_heavy_agent():
    cfg = InteractionRouterConfig(small_model_enabled=True)
    decision = RouterDecision(
        request_complexity=RequestComplexity.SIMPLE,
        route_target=RouteTarget.SMALL_MODEL,
        ack_template_id=AckTemplate.CHECKING,
        tool_risk=ToolRisk.READ_ONLY,
        confidence=0.99,
    )

    route = choose_voice_route("what files changed", decision, cfg)

    assert route.target is RouteTarget.HEAVY_AGENT
    assert route.ack_template_id is AckTemplate.CHECKING
    assert route.reason == "tool_risk_small_model_suggestion"

def test_plan_voice_request_returns_route_and_decision(monkeypatch):
    cfg = InteractionRouterConfig()
    decision = RouterDecision(
        request_complexity=RequestComplexity.COMPLEX,
        route_target=RouteTarget.HEAVY_AGENT,
        ack_template_id=AckTemplate.CHECKING,
        tool_risk=ToolRisk.READ_ONLY,
        confidence=0.9,
    )
    monkeypatch.setattr(
        "okay_hermes_voice.interaction_routes.classify_request",
        lambda transcript, cfg: decision,
    )

    plan = plan_voice_request("inspect the repo", cfg)

    assert plan.transcript == "inspect the repo"
    assert plan.decision is decision
    assert plan.route.target is RouteTarget.HEAVY_AGENT
    assert plan.route.ack_template_id is AckTemplate.CHECKING
