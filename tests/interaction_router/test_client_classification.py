from __future__ import annotations

import json
import sys
import types

from okay_hermes_voice.interaction_router import (
    AckTemplate,
    InteractionRouterConfig,
    RequestComplexity,
    RouteTarget,
    RouterDecision,
    ToolRisk,
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

class ScriptedCompletions:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class ScriptedClient:
    def __init__(self, *outcomes):
        self.chat = types.SimpleNamespace(completions=ScriptedCompletions(*outcomes))


def test_classify_with_client_retries_timeout_typeerror_without_timeout_kwarg():
    client = ScriptedClient(
        TypeError("unexpected keyword argument 'timeout'"),
        types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps({"confidence": 0.8})))]
        ),
    )
    cfg = InteractionRouterConfig(router_timeout_seconds=1.25)

    decision = classify_with_client(client, "test-model", "hello", cfg)

    assert decision.confidence == 0.8
    assert len(client.chat.completions.calls) == 2
    assert client.chat.completions.calls[0]["timeout"] == 1.25
    assert "timeout" not in client.chat.completions.calls[1]


def test_classify_with_client_falls_back_on_non_timeout_typeerror():
    client = ScriptedClient(TypeError("bad messages"))

    decision = classify_with_client(client, "test-model", "hello", InteractionRouterConfig())

    assert decision.brief_reason == "router_call_failed:TypeError"


def test_classify_with_client_falls_back_on_generic_exception():
    client = ScriptedClient(RuntimeError("network down"))

    decision = classify_with_client(client, "test-model", "hello", InteractionRouterConfig())

    assert decision.brief_reason == "router_call_failed:RuntimeError"


def test_classify_with_client_marks_invalid_response_shape():
    client = ScriptedClient(types.SimpleNamespace())

    decision = classify_with_client(client, "test-model", "hello", InteractionRouterConfig())

    assert decision.brief_reason.startswith("router_response_invalid:")


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

def test_classify_request_handles_local_close_without_provider(monkeypatch):
    resolve_calls = []

    def resolve_provider_client(provider, model=None):
        resolve_calls.append((provider, model))
        raise AssertionError("local close routing should not resolve a provider client")

    monkeypatch.setitem(
        sys.modules,
        "agent.auxiliary_client",
        types.SimpleNamespace(resolve_provider_client=resolve_provider_client),
    )
    clear_prewarmed_router()

    decision = classify_request(
        "close voice mode",
        InteractionRouterConfig(router_provider="openrouter", router_model="router-model"),
    )

    assert resolve_calls == []
    assert decision.route_target is RouteTarget.IMMEDIATE_ONLY
    assert decision.ack_template_id is AckTemplate.NONE
    assert decision.confidence == 1.0
    assert decision.brief_reason == "local_close_phrase"


def test_classify_request_handles_obvious_local_simple_chat_without_provider(monkeypatch):
    resolve_calls = []

    def resolve_provider_client(provider, model=None):
        resolve_calls.append((provider, model))
        raise AssertionError("obvious simple chat should not resolve a router provider client")

    monkeypatch.setitem(
        sys.modules,
        "agent.auxiliary_client",
        types.SimpleNamespace(resolve_provider_client=resolve_provider_client),
    )
    clear_prewarmed_router()

    for transcript in ("hello", "how are you", "thanks"):
        decision = classify_request(
            transcript,
            InteractionRouterConfig(router_provider="openrouter", router_model="router-model"),
        )
        assert decision.request_complexity is RequestComplexity.SIMPLE
        assert decision.route_target is RouteTarget.SMALL_MODEL
        assert decision.ack_template_id is AckTemplate.NONE
        assert decision.tool_risk is ToolRisk.NONE
        assert decision.confidence == 1.0
        assert decision.brief_reason == "local_simple_chat"

    assert resolve_calls == []



def test_classify_request_accepts_swappable_intent_engine():
    class FakeIntentEngine:
        def __init__(self):
            self.calls = []

        def classify(self, transcript, cfg):
            self.calls.append((transcript, cfg))
            return RouterDecision(
                route_target=RouteTarget.SMALL_MODEL,
                ack_template_id=AckTemplate.NONE,
                confidence=0.98,
                brief_reason="fake_engine",
            )

    cfg = InteractionRouterConfig(router_provider="openrouter", router_model="router-model")
    engine = FakeIntentEngine()

    decision = classify_request("tell me a fun fact", cfg, engine=engine)

    assert engine.calls == [("tell me a fun fact", cfg)]
    assert decision.route_target is RouteTarget.SMALL_MODEL
    assert decision.brief_reason == "fake_engine"


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
        router_timeout_seconds=1.25,
        small_model_timeout_seconds=4.5,
    )

    response = answer_with_small_model("tell me a tiny fact", cfg)

    assert response == "A tiny answer."
    kwargs = fake_client.chat.completions.kwargs
    assert kwargs["model"] == "google/gemini-2.5-flash-lite"
    assert kwargs["temperature"] == 0.2
    assert kwargs["timeout"] == 4.5
