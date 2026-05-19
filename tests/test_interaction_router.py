import json
import sys
import types
from pathlib import Path

from okay_hermes_voice.interaction_router import (
    ACK_TEXT,
    AckTemplate,
    AcknowledgementCache,
    InteractionRouterConfig,
    RequestComplexity,
    RouteTarget,
    RouterDecision,
    ToolRisk,
    build_router_messages,
    choose_voice_route,
    classify_request,
    classify_with_client,
    plan_voice_request,
)


def test_router_config_defaults_are_conservative():
    cfg = InteractionRouterConfig()

    assert cfg.router_enabled is True
    assert cfg.router_provider == "openrouter"
    assert cfg.router_model == "google/gemini-2.5-flash-lite"
    assert cfg.router_timeout_seconds == 1.5
    assert cfg.router_min_confidence == 0.70
    assert cfg.small_model_enabled is False
    assert cfg.ack_cache_enabled is True


def test_router_config_from_mapping_strips_blank_values():
    cfg = InteractionRouterConfig.from_mapping(
        {
            "router_provider": " deepseek ",
            "router_model": " deepseek/deepseek-v4-flash ",
            "router_timeout_seconds": "2.25",
            "router_min_confidence": "0.8",
            "small_model_enabled": "yes",
        }
    )

    assert cfg.router_provider == "deepseek"
    assert cfg.router_model == "deepseek/deepseek-v4-flash"
    assert cfg.router_timeout_seconds == 2.25
    assert cfg.router_min_confidence == 0.8
    assert cfg.small_model_enabled is True


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


def test_build_router_messages_mentions_json_and_forbids_solving():
    messages = build_router_messages("inspect the repo and fix tests")
    joined = "\n".join(message["content"] for message in messages)

    assert "JSON" in joined
    assert "Do not solve" in joined
    assert "inspect the repo and fix tests" in joined


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse(self.content)


class FakeChat:
    def __init__(self, content: str):
        self.completions = FakeCompletions(content)


class FakeClient:
    def __init__(self, content: str):
        self.chat = FakeChat(content)


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
        "okay_hermes_voice.interaction_router.classify_request",
        lambda transcript, cfg: decision,
    )

    plan = plan_voice_request("inspect the repo", cfg)

    assert plan.transcript == "inspect the repo"
    assert plan.decision is decision
    assert plan.route.target is RouteTarget.HEAVY_AGENT
    assert plan.route.ack_template_id is AckTemplate.CHECKING


def test_ack_cache_generates_missing_audio_once(tmp_path):
    generated: list[str] = []

    def fake_tts(text: str, out_path: Path) -> Path:
        generated.append(text)
        out_path.write_bytes(b"fake audio")
        return out_path

    cache = AcknowledgementCache(tmp_path, tts_generator=fake_tts, audio_player=lambda path: True)

    first = cache.ensure(AckTemplate.GOT_IT)
    second = cache.ensure(AckTemplate.GOT_IT)

    assert ACK_TEXT[AckTemplate.GOT_IT] == "Okay, I’m on it."
    assert first == second
    assert first.exists()
    assert generated == ["Okay, I’m on it."]


def test_ack_cache_preserves_provider_suffix_and_ignores_mislabeled_wav(tmp_path):
    generated: list[str] = []
    stale = tmp_path / "got_it.wav"
    stale.write_bytes(b"OggS\x00mislabeled opus cache")

    def fake_tts(text: str, out_path: Path) -> Path:
        generated.append(text)
        actual_path = out_path.with_suffix(".ogg")
        actual_path.write_bytes(b"OggS\x00fresh opus cache")
        return actual_path

    cache = AcknowledgementCache(tmp_path, tts_generator=fake_tts, audio_player=lambda path: True)

    first = cache.ensure(AckTemplate.GOT_IT)
    second = cache.ensure(AckTemplate.GOT_IT)

    assert first == tmp_path / "got_it.ogg"
    assert second == first
    assert first.read_bytes().startswith(b"OggS")
    assert generated == ["Okay, I’m on it."]


def test_ack_cache_play_uses_existing_audio(tmp_path):
    played: list[Path] = []
    path = tmp_path / "got_it.wav"
    path.write_bytes(b"fake audio")

    cache = AcknowledgementCache(
        tmp_path,
        tts_generator=lambda text, out_path: out_path,
        audio_player=lambda p: played.append(p) or True,
    )

    assert cache.play(AckTemplate.GOT_IT) is True
    assert played == [path]
