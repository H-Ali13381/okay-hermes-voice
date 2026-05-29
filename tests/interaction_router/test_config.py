from __future__ import annotations

from okay_hermes_voice.interaction_router import InteractionRouterConfig


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
