from __future__ import annotations

import json
import sys
import types

from okay_hermes_voice.interaction_router import (
    InteractionRouterConfig,
    RouteTarget,
    classify_request,
    classify_with_client,
)
from okay_hermes_voice.interaction_clients.router_prewarm import (
    clear_prewarmed_router,
    prewarm_interaction_router,
)

from .fakes import FakeClient


def test_classify_with_client_requests_json_and_parses_result():
    client = FakeClient(
        json.dumps(
            {
                "request_complexity": "complex",
                "route_target": "heavy_agent",
                "ack_template_id": "checking",
                "tool_risk": "read_only",
                "confidence": 0.9,
            }
        )
    )
    cfg = InteractionRouterConfig(router_model="test-model", router_timeout_seconds=1.25)

    decision = classify_with_client(client, "test-model", "fix the tests", cfg)

    assert decision.route_target is RouteTarget.HEAVY_AGENT
    kwargs = client.chat.completions.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] <= 220
    assert kwargs["response_format"] == {"type": "json_object"}

def test_classify_request_uses_hermes_provider_client(monkeypatch):
    fake_client = FakeClient(
        json.dumps(
            {
                "request_complexity": "simple",
                "route_target": "small_model",
                "ack_template_id": "none",
                "tool_risk": "none",
                "confidence": 0.95,
            }
        )
    )

    monkeypatch.setitem(
        sys.modules,
        "agent.auxiliary_client",
        types.SimpleNamespace(
            resolve_provider_client=lambda provider, model=None: (fake_client, model)
        ),
    )

    cfg = InteractionRouterConfig(
        router_provider="openrouter", router_model="google/gemini-2.5-flash-lite"
    )
    decision = classify_request("tell me a fun fact", cfg)

    assert decision.route_target is RouteTarget.SMALL_MODEL
    assert fake_client.chat.completions.kwargs["model"] == "google/gemini-2.5-flash-lite"

def test_classify_request_reuses_prewarmed_router_client(monkeypatch):
    fake_client = FakeClient(
        json.dumps(
            {
                "request_complexity": "simple",
                "route_target": "small_model",
                "ack_template_id": "none",
                "tool_risk": "none",
                "confidence": 0.95,
            }
        )
    )
    resolve_calls = []

    def resolve_provider_client(provider, model=None):
        resolve_calls.append((provider, model))
        return fake_client, model

    monkeypatch.setitem(
        sys.modules,
        "agent.auxiliary_client",
        types.SimpleNamespace(resolve_provider_client=resolve_provider_client),
    )
    cfg = InteractionRouterConfig(router_provider="openrouter", router_model="router-model")

    try:
        assert prewarm_interaction_router(cfg) is True
        assert classify_request("tell me a fun fact", cfg).route_target is RouteTarget.SMALL_MODEL
        assert classify_request("say hello", cfg).route_target is RouteTarget.SMALL_MODEL
    finally:
        clear_prewarmed_router()

    assert resolve_calls == [("openrouter", "router-model")]
    kwargs = fake_client.chat.completions.kwargs
    assert kwargs is not None
    assert kwargs["model"] == "router-model"

def test_answer_with_small_model_uses_configured_small_model_client(monkeypatch):
    from okay_hermes_voice.interaction_router import answer_with_small_model

    fake_client = FakeClient("A tiny answer.")
    monkeypatch.setitem(
        sys.modules,
        "agent.auxiliary_client",
        types.SimpleNamespace(
            resolve_provider_client=lambda provider, model=None: (fake_client, model)
        ),
    )
    cfg = InteractionRouterConfig(
        small_model_provider="openrouter",
        small_model_model="google/gemini-2.5-flash-lite",
    )

    response = answer_with_small_model("tell me a tiny fact", cfg)

    assert response == "A tiny answer."
    kwargs = fake_client.chat.completions.kwargs
    assert kwargs["model"] == "google/gemini-2.5-flash-lite"
    assert kwargs["temperature"] == 0.2
