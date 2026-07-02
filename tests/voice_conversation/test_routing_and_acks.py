from __future__ import annotations

import json
import threading
import time

from okay_hermes_voice import interaction_router as router
from okay_hermes_voice import voice_routing as routing
from okay_hermes_voice import wakeword_daemon as wake
from okay_hermes_voice.voice_routing import ack_playback


def test_interaction_router_config_from_daemon_config_maps_prefixed_keys():
    cfg = dict(wake.DEFAULT_CONFIG)
    cfg.update(
        {
            "interaction_router_enabled": True,
            "interaction_router_provider": "deepseek",
            "interaction_router_model": "deepseek/deepseek-v4-flash",
            "interaction_router_timeout_seconds": 2.25,
            "interaction_router_small_model_timeout_seconds": 4.5,
            "interaction_router_min_confidence": 0.8,
            "interaction_router_small_model_enabled": True,
            "interaction_router_ack_cache_enabled": False,
            "interaction_router_ack_cache_dir": "~/tmp/acks",
        }
    )

    router_cfg = routing.interaction_router_config_from_daemon_config(cfg)

    assert router_cfg.router_enabled is True
    assert router_cfg.router_provider == "deepseek"
    assert router_cfg.router_model == "deepseek/deepseek-v4-flash"
    assert router_cfg.router_timeout_seconds == 2.25
    assert router_cfg.small_model_timeout_seconds == 4.5
    assert router_cfg.router_min_confidence == 0.8
    assert router_cfg.small_model_enabled is True
    assert router_cfg.ack_cache_enabled is False
    assert router_cfg.ack_cache_dir == "~/tmp/acks"

def test_default_config_routes_simple_safe_requests_to_small_model(monkeypatch):
    def fake_plan_voice_request(transcript, cfg):
        assert cfg.small_model_enabled is True
        decision = router.RouterDecision(
            request_complexity=router.RequestComplexity.SIMPLE,
            route_target=router.RouteTarget.SMALL_MODEL,
            ack_template_id=router.AckTemplate.NONE,
            tool_risk=router.ToolRisk.NONE,
            confidence=0.95,
        )
        route = router.choose_voice_route(transcript, decision, cfg)
        return router.VoiceRequestPlan(transcript, decision, route)

    monkeypatch.setattr(routing, "plan_voice_request", fake_plan_voice_request)

    plan = routing.plan_interaction_route(dict(wake.DEFAULT_CONFIG), "tell me a fun fact")

    assert plan is not None
    assert plan.route.target is router.RouteTarget.SMALL_MODEL
    assert plan.route.reason == "router_small_model"


def test_plan_interaction_route_returns_none_when_disabled(monkeypatch):
    called = False

    def fake_plan_voice_request(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(routing, "plan_voice_request", fake_plan_voice_request)

    assert routing.plan_interaction_route({"interaction_router_enabled": False}, "hello") is None
    assert called is False

def test_play_interaction_ack_uses_ack_cache(monkeypatch, tmp_path):
    played = []

    class FakeAckCache:
        def __init__(self, cache_dir, *, tts_generator, audio_player):
            self.cache_dir = cache_dir
            self.tts_generator = tts_generator
            self.audio_player = audio_player

        def play(self, template_id):
            played.append((self.cache_dir, template_id))
            return True

    monkeypatch.setattr(routing, "AcknowledgementCache", FakeAckCache)

    cfg = {
        "interaction_router_ack_cache_dir": str(tmp_path),
        "interaction_router_ack_cache_enabled": True,
    }

    assert routing.play_interaction_ack(cfg, wake.AckTemplate.CHECKING) is True
    assert played == [(tmp_path, wake.AckTemplate.CHECKING)]

def test_generate_ack_tts_preserves_tts_provider_suffix(monkeypatch, tmp_path):
    source = tmp_path / "source.ogg"
    source.write_bytes(b"OggS\x00provider audio")

    monkeypatch.setattr(
        routing,
        "text_to_speech_tool",
        lambda text: json.dumps({"success": True, "file_path": str(source)}),
    )

    target = routing._generate_ack_tts("Okay, I’m on it.", tmp_path / "got_it.wav")

    assert target == tmp_path / "got_it.ogg"
    assert target.read_bytes() == source.read_bytes()

def test_route_transcribed_request_plays_immediate_ack(monkeypatch):
    decision = router.RouterDecision(confidence=0.9)
    route = router.VoiceRoute(
        wake.RouteTarget.HEAVY_AGENT,
        wake.AckTemplate.CHECKING,
        "router_heavy_agent",
    )
    plan = wake.VoiceRequestPlan("inspect the repo", decision, route)
    played = []

    monkeypatch.setattr(routing, "plan_interaction_route", lambda cfg, transcript: plan)
    monkeypatch.setattr(
        routing,
        "play_interaction_ack",
        lambda cfg, template, **kwargs: played.append((template, kwargs.get("block"))) or True,
    )

    assert routing.route_transcribed_request({}, "inspect the repo") is plan
    assert played == [(wake.AckTemplate.CHECKING, False)]


def test_route_transcribed_request_schedules_provisional_ack_before_planning(monkeypatch):
    decision = router.RouterDecision(confidence=0.9)
    route = router.VoiceRoute(
        wake.RouteTarget.HEAVY_AGENT,
        wake.AckTemplate.CHECKING,
        "router_heavy_agent",
    )
    plan = wake.VoiceRequestPlan("inspect the repo", decision, route)
    events = []

    def fake_plan_interaction_route(_cfg, _transcript):
        events.append("plan_start")
        assert events == [("ack", wake.AckTemplate.GOT_IT, False, True), "plan_start"]
        return plan

    monkeypatch.setattr(routing, "plan_interaction_route", fake_plan_interaction_route)
    monkeypatch.setattr(
        routing,
        "play_interaction_ack",
        lambda cfg, template, **kwargs: events.append(
            (template and "ack", template, kwargs.get("block"), kwargs.get("loop_until_cancelled"))
        ) or True,
    )

    assert routing.route_transcribed_request({}, "inspect the repo", loop_ack_until_cancelled=True) is plan
    assert events == [("ack", wake.AckTemplate.GOT_IT, False, True), "plan_start"]


def test_route_transcribed_request_can_schedule_looping_ack(monkeypatch):
    decision = router.RouterDecision(confidence=0.9)
    route = router.VoiceRoute(
        wake.RouteTarget.HEAVY_AGENT,
        wake.AckTemplate.CHECKING,
        "router_heavy_agent",
    )
    plan = wake.VoiceRequestPlan("inspect the repo", decision, route)
    played = []

    monkeypatch.setattr(routing, "plan_interaction_route", lambda cfg, transcript: plan)
    monkeypatch.setattr(
        routing,
        "play_interaction_ack",
        lambda cfg, template, **kwargs: played.append(
            (template, kwargs.get("block"), kwargs.get("loop_until_cancelled"))
        ) or True,
    )

    assert routing.route_transcribed_request({}, "inspect the repo", loop_ack_until_cancelled=True) is plan
    assert played == [(wake.AckTemplate.GOT_IT, False, True)]


def test_play_interaction_ack_repeats_until_cancelled(monkeypatch):
    calls = 0
    finished = threading.Event()

    def fake_play(_cfg, template_id, *, cancel_check=None):
        nonlocal calls
        assert template_id is wake.AckTemplate.CHECKING
        assert cancel_check is not None
        calls += 1
        if calls >= 3:
            finished.set()
        return True

    monkeypatch.setattr(ack_playback, "_play_interaction_ack_sync", fake_play)

    assert routing.play_interaction_ack(
        {},
        wake.AckTemplate.CHECKING,
        cancel_check=lambda: finished.is_set(),
        block=False,
        loop_until_cancelled=True,
    ) is True

    assert finished.wait(timeout=1.0)
    assert calls >= 3


def test_route_transcribed_request_schedules_heavy_ack_without_blocking(monkeypatch, tmp_path):
    decision = router.RouterDecision(confidence=0.9)
    route = router.VoiceRoute(
        wake.RouteTarget.HEAVY_AGENT,
        wake.AckTemplate.CHECKING,
        "router_heavy_agent",
    )
    plan = wake.VoiceRequestPlan("inspect the repo", decision, route)
    ack_started = threading.Event()
    ack_can_finish = threading.Event()
    ack_finished = threading.Event()

    class SlowAckCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def play(self, template_id):
            assert template_id is wake.AckTemplate.CHECKING
            ack_started.set()
            ack_can_finish.wait(timeout=1.0)
            ack_finished.set()
            return True

    monkeypatch.setattr(routing, "plan_interaction_route", lambda cfg, transcript: plan)
    monkeypatch.setattr(routing, "AcknowledgementCache", SlowAckCache)

    started = time.monotonic()
    result = routing.route_transcribed_request(
        {
            "interaction_router_ack_cache_dir": str(tmp_path),
            "interaction_router_ack_cache_enabled": True,
        },
        "inspect the repo",
    )
    elapsed = time.monotonic() - started

    try:
        assert result is plan
        assert elapsed < 0.2
        assert ack_started.wait(timeout=0.2)
    finally:
        ack_can_finish.set()
        ack_finished.wait(timeout=1.0)

def test_route_transcribed_request_skips_none_ack(monkeypatch):
    decision = router.RouterDecision(confidence=0.9)
    route = router.VoiceRoute(
        wake.RouteTarget.IMMEDIATE_ONLY,
        wake.AckTemplate.NONE,
        "router_immediate_only",
    )
    plan = wake.VoiceRequestPlan("close", decision, route)

    monkeypatch.setattr(routing, "plan_interaction_route", lambda cfg, transcript: plan)
    monkeypatch.setattr(
        routing,
        "play_interaction_ack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ack should not play")),
    )

    assert routing.route_transcribed_request({}, "close") is plan

def test_answer_routed_request_immediate_only_bypasses_heavy_hermes_call(monkeypatch):
    route = router.VoiceRoute(
        wake.RouteTarget.IMMEDIATE_ONLY,
        wake.AckTemplate.NONE,
        "router_immediate_only",
    )
    plan = wake.VoiceRequestPlan("close voice mode", router.RouterDecision(confidence=1.0), route)
    history = [{"role": "user", "content": "prior"}]
    monkeypatch.setattr(
        routing,
        "ask_hermes_turn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("heavy agent should not run")),
    )

    response, returned_history, source = routing.answer_routed_request({}, "close voice mode", plan, history)

    assert response is None
    assert returned_history == history
    assert source == "immediate_only"


def test_answer_routed_request_uses_small_model_and_updates_history(monkeypatch):
    route = router.VoiceRoute(
        wake.RouteTarget.SMALL_MODEL,
        wake.AckTemplate.NONE,
        "router_small_model",
    )
    plan = wake.VoiceRequestPlan("tell me a tiny fact", router.RouterDecision(confidence=0.95), route)
    seen = {}

    def fake_answer_with_small_model(transcript, cfg):
        seen["transcript"] = transcript
        seen["provider"] = cfg.small_model_provider
        seen["model"] = cfg.small_model_model
        return "Tiny answer."

    monkeypatch.setattr(routing, "answer_with_small_model", fake_answer_with_small_model)
    monkeypatch.setattr(
        routing,
        "ask_hermes_turn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("heavy agent should not run")),
    )

    response, history, source = routing.answer_routed_request({}, "tell me a tiny fact", plan, [])

    assert response == "Tiny answer."
    assert source == "small_model"
    assert seen == {
        "transcript": "tell me a tiny fact",
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash-lite",
    }
    assert history[-2:] == [
        {"role": "user", "content": "tell me a tiny fact"},
        {"role": "assistant", "content": "Tiny answer."},
    ]

def test_answer_routed_request_falls_back_to_heavy_agent(monkeypatch):
    route = router.VoiceRoute(
        wake.RouteTarget.HEAVY_AGENT,
        wake.AckTemplate.CHECKING,
        "router_heavy_agent",
    )
    plan = wake.VoiceRequestPlan("inspect the repo", router.RouterDecision(confidence=0.95), route)
    monkeypatch.setattr(routing, "ask_hermes_turn", lambda cfg, transcript, history, **_kwargs: ("Heavy answer.", [*history, {"role": "assistant", "content": "Heavy answer."}]))

    response, history, source = routing.answer_routed_request({}, "inspect the repo", plan, [])

    assert response == "Heavy answer."
    assert source == "heavy_agent"
    assert history == [{"role": "assistant", "content": "Heavy answer."}]


def test_command_recording_config_uses_poll_timeout_while_heavy_delegation_pending():
    cfg = {
        "conversation_mode_enabled": True,
        "conversation_followup_start_timeout_seconds": 0.0,
        "heavy_agent_delegation_followup_start_timeout_seconds": 1.25,
        "_heavy_agent_delegation_pending": True,
    }

    turn_cfg = routing.command_recording_config_for_turn(cfg, 2)

    assert turn_cfg["speech_start_timeout_seconds"] == 1.25


def test_answer_routed_request_dispatches_heavy_agent_delegation(monkeypatch):
    route = router.VoiceRoute(
        wake.RouteTarget.HEAVY_AGENT,
        wake.AckTemplate.CHECKING,
        "router_heavy_agent",
    )
    plan = wake.VoiceRequestPlan("debug the repo", router.RouterDecision(confidence=0.95), route)
    dispatched = {}

    def fake_dispatch(cfg, transcript, history):
        dispatched["cfg"] = cfg
        dispatched["transcript"] = transcript
        dispatched["history"] = list(history)
        return "deleg_123"

    monkeypatch.setattr(routing, "dispatch_heavy_agent_delegation", fake_dispatch)
    monkeypatch.setattr(
        routing,
        "ask_hermes_turn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("heavy agent should not block inline")),
    )

    response, history, source = routing.answer_routed_request(
        {"heavy_agent_delegation_enabled": True},
        "debug the repo",
        plan,
        [{"role": "user", "content": "prior"}],
    )

    assert source == "heavy_agent_delegation"
    assert isinstance(response, str)
    assert "working" in response.lower()
    assert "deleg_123" in response
    assert dispatched["transcript"] == "debug the repo"
    assert dispatched["history"] == [{"role": "user", "content": "prior"}]
    assert history[-2:] == [
        {"role": "user", "content": "debug the repo"},
        {"role": "assistant", "content": response},
    ]


def test_answer_routed_request_falls_back_when_delegation_rejected(monkeypatch):
    route = router.VoiceRoute(
        wake.RouteTarget.HEAVY_AGENT,
        wake.AckTemplate.CHECKING,
        "router_heavy_agent",
    )
    plan = wake.VoiceRequestPlan("debug the repo", router.RouterDecision(confidence=0.95), route)

    monkeypatch.setattr(routing, "dispatch_heavy_agent_delegation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        routing,
        "ask_hermes_turn",
        lambda cfg, transcript, history, **_kwargs: ("Heavy answer.", [*history, {"role": "assistant", "content": "Heavy answer."}]),
    )

    response, history, source = routing.answer_routed_request(
        {"heavy_agent_delegation_enabled": True}, "debug the repo", plan, []
    )

    assert response == "Heavy answer."
    assert source == "heavy_agent"
    assert history == [{"role": "assistant", "content": "Heavy answer."}]
