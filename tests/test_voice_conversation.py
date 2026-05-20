from __future__ import annotations

import json
import signal
import sys
import threading
import time
import types
import wave
from pathlib import Path

from okay_hermes_voice import interaction_router as router
from okay_hermes_voice import wakeword_daemon as wake
from okay_hermes_voice import voice_activation_popup as popup

import numpy as np

def test_interaction_router_config_from_daemon_config_maps_prefixed_keys():
    cfg = dict(wake.DEFAULT_CONFIG)
    cfg.update(
        {
            "interaction_router_enabled": True,
            "interaction_router_provider": "deepseek",
            "interaction_router_model": "deepseek/deepseek-v4-flash",
            "interaction_router_timeout_seconds": 2.25,
            "interaction_router_min_confidence": 0.8,
            "interaction_router_small_model_enabled": True,
            "interaction_router_ack_cache_enabled": False,
            "interaction_router_ack_cache_dir": "~/tmp/acks",
        }
    )

    router_cfg = wake.interaction_router_config_from_daemon_config(cfg)

    assert router_cfg.router_enabled is True
    assert router_cfg.router_provider == "deepseek"
    assert router_cfg.router_model == "deepseek/deepseek-v4-flash"
    assert router_cfg.router_timeout_seconds == 2.25
    assert router_cfg.router_min_confidence == 0.8
    assert router_cfg.small_model_enabled is True
    assert router_cfg.ack_cache_enabled is False
    assert router_cfg.ack_cache_dir == "~/tmp/acks"


def test_plan_interaction_route_returns_none_when_disabled(monkeypatch):
    called = False

    def fake_plan_voice_request(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(wake, "plan_voice_request", fake_plan_voice_request)

    assert wake.plan_interaction_route({"interaction_router_enabled": False}, "hello") is None
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

    monkeypatch.setattr(wake, "AcknowledgementCache", FakeAckCache)

    cfg = {
        "interaction_router_ack_cache_dir": str(tmp_path),
        "interaction_router_ack_cache_enabled": True,
    }

    assert wake.play_interaction_ack(cfg, wake.AckTemplate.CHECKING) is True
    assert played == [(tmp_path, wake.AckTemplate.CHECKING)]


def test_generate_ack_tts_preserves_tts_provider_suffix(monkeypatch, tmp_path):
    source = tmp_path / "source.ogg"
    source.write_bytes(b"OggS\x00provider audio")

    monkeypatch.setattr(
        wake,
        "text_to_speech_tool",
        lambda text: json.dumps({"success": True, "file_path": str(source)}),
    )

    target = wake._generate_ack_tts("Okay, I’m on it.", tmp_path / "got_it.wav")

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

    monkeypatch.setattr(wake, "plan_interaction_route", lambda cfg, transcript: plan)
    monkeypatch.setattr(
        wake,
        "play_interaction_ack",
        lambda cfg, template, **kwargs: played.append((template, kwargs.get("block"))) or True,
    )

    assert wake.route_transcribed_request({}, "inspect the repo") is plan
    assert played == [(wake.AckTemplate.CHECKING, False)]


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

    monkeypatch.setattr(wake, "plan_interaction_route", lambda cfg, transcript: plan)
    monkeypatch.setattr(wake, "AcknowledgementCache", SlowAckCache)

    started = time.monotonic()
    result = wake.route_transcribed_request(
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

    monkeypatch.setattr(wake, "plan_interaction_route", lambda cfg, transcript: plan)
    monkeypatch.setattr(
        wake,
        "play_interaction_ack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ack should not play")),
    )

    assert wake.route_transcribed_request({}, "close") is plan


def test_handle_activation_surfaces_routing_ack_text_in_popup(monkeypatch, tmp_path):
    state_path = tmp_path / "voice_state.json"
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")
    decision = router.RouterDecision(confidence=0.0)
    route = router.VoiceRoute(
        wake.RouteTarget.HEAVY_AGENT,
        wake.AckTemplate.GOT_IT,
        "low_router_confidence",
    )
    plan = wake.VoiceRequestPlan("what is the weather", decision, route)

    def fake_launch_visualization(_cfg, probability):
        wake.update_visualization_state(
            state_path,
            status="listening",
            probability=probability,
            cancel_requested=False,
            cancel_reason="",
        )
        return state_path

    def fake_answer_routed_request(_cfg, transcript, routed_plan, history, *, cancel_check):
        assert transcript == "what is the weather"
        assert routed_plan is plan
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["interaction_ack_text"] == "Okay, I’m on it."
        assert "Okay, I’m on it." in state["message"]
        assert "heavy agent" in state["message"]
        return "spoken answer", history, "heavy_agent"

    wake.STOP.clear()
    monkeypatch.setattr(wake, "save_activation_archive", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wake, "launch_visualization", fake_launch_visualization)
    monkeypatch.setattr(wake, "maybe_beep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wake, "record_command", lambda *_args, **_kwargs: command_path)
    monkeypatch.setattr(wake, "archive_command_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wake, "transcribe_command", lambda *_args, **_kwargs: "what is the weather")
    monkeypatch.setattr(wake, "route_transcribed_request", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(wake, "answer_routed_request", fake_answer_routed_request)
    monkeypatch.setattr(wake, "speak_response", lambda *_args, **_kwargs: None)

    wake.handle_activation({"conversation_mode_enabled": False}, {"probability": 0.9})


def test_answer_routed_request_uses_small_model_and_updates_history(monkeypatch):
    route = router.VoiceRoute(
        wake.RouteTarget.SMALL_MODEL,
        wake.AckTemplate.NONE,
        "router_small_model",
    )
    plan = wake.VoiceRequestPlan("tell me a tiny fact", router.RouterDecision(confidence=0.95), route)
    monkeypatch.setattr(wake, "answer_with_small_model", lambda transcript, cfg: "Tiny answer.")
    monkeypatch.setattr(
        wake,
        "ask_hermes_turn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("heavy agent should not run")),
    )

    response, history, source = wake.answer_routed_request({}, "tell me a tiny fact", plan, [])

    assert response == "Tiny answer."
    assert source == "small_model"
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
    monkeypatch.setattr(wake, "ask_hermes_turn", lambda cfg, transcript, history, **_kwargs: ("Heavy answer.", [*history, {"role": "assistant", "content": "Heavy answer."}]))

    response, history, source = wake.answer_routed_request({}, "inspect the repo", plan, [])

    assert response == "Heavy answer."
    assert source == "heavy_agent"
    assert history == [{"role": "assistant", "content": "Heavy answer."}]


def test_close_transcript_matches_only_explicit_session_close_commands():
    cfg = {
        "conversation_close_phrases": [
            "close",
            "close voice mode",
            "close conversation",
            "stop listening",
            "end conversation",
        ]
    }

    assert wake.is_close_transcript("close", cfg)
    assert wake.is_close_transcript("Okay Hermes, close voice mode.", cfg)
    assert wake.is_close_transcript("Hermes stop listening", cfg)
    assert wake.is_close_transcript("end conversation", cfg)

    assert not wake.is_close_transcript("close the browser", cfg)
    assert not wake.is_close_transcript("can you close the window after this", cfg)
    assert not wake.is_close_transcript("what is the closest coffee shop", cfg)


def test_command_recording_config_for_followup_disables_start_timeout_without_mutating_original():
    cfg = {
        "speech_start_timeout_seconds": 15.0,
        "conversation_followup_start_timeout_seconds": 0.0,
    }

    first = wake.command_recording_config_for_turn(cfg, turn_index=1)
    followup = wake.command_recording_config_for_turn(cfg, turn_index=2)

    assert first["speech_start_timeout_seconds"] == 15.0
    assert followup["speech_start_timeout_seconds"] == 0.0
    assert cfg["speech_start_timeout_seconds"] == 15.0
    assert followup is not cfg


def test_configured_hermes_toolsets_preserves_explicit_wakeword_override():
    assert wake.configured_hermes_toolsets({"hermes_toolsets": ["web"]}) == ["web"]


def test_configured_hermes_toolsets_inherits_cli_platform_tools_when_unset(monkeypatch):
    from hermes_cli import config as hermes_config
    from hermes_cli import tools_config

    fake_cfg = {"platform_toolsets": {"cli": ["terminal", "file", "skills"]}}
    monkeypatch.setattr(hermes_config, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda cfg, platform: set(cfg["platform_toolsets"][platform]))

    assert wake.configured_hermes_toolsets({"hermes_toolsets": None}) == ["file", "skills", "terminal"]
    assert wake.configured_hermes_toolsets({"hermes_toolsets": ""}) == ["file", "skills", "terminal"]


def test_warm_hermes_agent_loads_soul_identity_without_project_context(monkeypatch):
    created_kwargs = {}

    class FakeAIAgent:
        def __init__(self, **kwargs):
            created_kwargs.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        types.SimpleNamespace(
            resolve_runtime_provider=lambda requested=None, target_model=None: {
                "provider": "openai-codex",
                "api_key": "token",
                "base_url": "https://example.invalid",
                "api_mode": "codex_responses",
                "credential_pool": None,
            }
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.oneshot",
        types.SimpleNamespace(
            _create_session_db_for_oneshot=lambda: object(),
            _oneshot_clarify_callback=lambda *_args, **_kwargs: "pick a default",
        ),
    )
    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAIAgent))
    wake._HERMES_AGENT_CACHE.clear()

    wake.get_warm_hermes_agent(
        {"hermes_max_iterations": 90, "hermes_load_soul_identity": True},
        provider=None,
        model="gpt-5.5",
        toolsets=["skills"],
    )

    assert created_kwargs["skip_context_files"] is True
    assert created_kwargs["load_soul_identity"] is True
    assert created_kwargs["max_iterations"] == 90
    wake._HERMES_AGENT_CACHE.clear()


def test_ask_hermes_turn_preserves_voice_conversation_history(monkeypatch):
    calls = {}

    class FakeAgent:
        def run_conversation(self, prompt, conversation_history=None, persist_user_message=None):
            calls["prompt"] = prompt
            calls["conversation_history"] = list(conversation_history or [])
            calls["persist_user_message"] = persist_user_message
            return {
                "final_response": "second answer",
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": persist_user_message},
                    {"role": "assistant", "content": "second answer"},
                ],
            }

    monkeypatch.setattr(wake, "configured_hermes_runtime_selection", lambda cfg: (None, "gpt-5.5"))
    monkeypatch.setattr(wake, "configured_hermes_toolsets", lambda cfg: ["skills"])
    monkeypatch.setattr(wake, "get_warm_hermes_agent", lambda cfg, provider, model, toolsets: FakeAgent())

    previous_history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    response, updated_history = wake.ask_hermes_turn(
        {
            "hermes_inprocess": True,
            "hermes_warm_agent": True,
            "hermes_prompt_prefix": "VOICE PREFIX",
        },
        "follow-up question",
        previous_history,
    )

    assert response == "second answer"
    assert calls["prompt"] == "VOICE PREFIX\n\nTranscript:\nfollow-up question"
    assert calls["conversation_history"] == previous_history
    assert calls["persist_user_message"] == "follow-up question"
    assert updated_history[-2:] == [
        {"role": "user", "content": "follow-up question"},
        {"role": "assistant", "content": "second answer"},
    ]


def test_ask_hermes_turn_interrupts_warm_agent_when_cancel_requested(monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.interrupted = False
            self.interrupt_message = None

        def run_conversation(self, prompt, conversation_history=None, persist_user_message=None):
            deadline = time.monotonic() + 0.25
            while time.monotonic() < deadline and not self.interrupted:
                time.sleep(0.01)
            return {
                "final_response": "late answer",
                "messages": list(conversation_history or []),
                "interrupted": self.interrupted,
            }

        def interrupt(self, message=None):
            self.interrupted = True
            self.interrupt_message = message

    fake_agent = FakeAgent()
    monkeypatch.setattr(wake, "configured_hermes_runtime_selection", lambda cfg: (None, "gpt-5.5"))
    monkeypatch.setattr(wake, "configured_hermes_toolsets", lambda cfg: ["skills"])
    monkeypatch.setattr(wake, "get_warm_hermes_agent", lambda cfg, provider, model, toolsets: fake_agent)

    response, history = wake.ask_hermes_turn(
        {
            "hermes_inprocess": True,
            "hermes_warm_agent": True,
            "hermes_cancel_poll_seconds": 0.01,
            "hermes_interrupt_wait_seconds": 1.0,
        },
        "please do a long task",
        [{"role": "user", "content": "before"}],
        cancel_check=lambda: True,
    )

    assert response is None
    assert history == [{"role": "user", "content": "before"}]
    assert fake_agent.interrupted is True
    assert fake_agent.interrupt_message == "Voice session cancelled"


def test_ask_hermes_turn_kills_subprocess_group_when_cancel_requested(monkeypatch):
    popen_kwargs = {}
    killpg_calls = []

    class FakeProc:
        pid = 43210
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -signal.SIGTERM
            return self.returncode

        def communicate(self):
            return "partial stdout", "partial stderr"

    fake_proc = FakeProc()

    def fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return fake_proc

    monkeypatch.setattr(wake, "configured_hermes_runtime_selection", lambda cfg: (None, None))
    monkeypatch.setattr(wake, "configured_hermes_toolsets", lambda cfg: None)
    monkeypatch.setattr(wake.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        wake.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Hermes subprocess must be cancellable Popen, not run")),
    )
    monkeypatch.setattr(wake.os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))

    response, history = wake.ask_hermes_turn(
        {
            "hermes_inprocess": False,
            "hermes_bin": "hermes",
            "hermes_source": "wakeword",
            "hermes_timeout_seconds": 30,
            "hermes_cancel_poll_seconds": 0.01,
            "hermes_interrupt_wait_seconds": 1.0,
        },
        "run a long shell command",
        [],
        cancel_check=lambda: True,
    )

    assert response is None
    assert history == []
    assert popen_kwargs["start_new_session"] is True
    assert killpg_calls == [(fake_proc.pid, signal.SIGTERM)]


def test_append_visualization_turn_preserves_conversation_history(tmp_path):
    state_path = tmp_path / "voice_state.json"
    wake.update_visualization_state(
        state_path,
        status="thinking",
        message="first turn",
        transcript="",
        response="",
        turns=[],
    )

    wake.append_visualization_turn(state_path, transcript="First request", response="First response")
    wake.append_visualization_turn(state_path, transcript="Second request", response="Second response")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["transcript"] == "Second request"
    assert state["response"] == "Second response"
    assert [turn["transcript"] for turn in state["turns"]] == ["First request", "Second request"]
    assert [turn["response"] for turn in state["turns"]] == ["First response", "Second response"]


def test_popup_render_shows_multiple_conversation_turns():
    rendered = popup.render(
        {
            "title": "Hermes Voice",
            "status": "listening",
            "message": "Listening for follow-up",
            "activated_at": 1,
            "updated_at": 2,
            "turns": [
                {"transcript": "First request", "response": "First response"},
                {"transcript": "Second request", "response": "Second response"},
            ],
        },
        tick=0,
        final_seen_at=None,
    )

    assert "Conversation" in rendered
    assert "You: First request" in rendered
    assert "Hermes: First response" in rendered
    assert "You: Second request" in rendered
    assert "Hermes: Second response" in rendered


def test_popup_ctrl_c_request_marks_state_for_daemon_cancel(tmp_path):
    state_path = tmp_path / "voice_state.json"
    state_path.write_text(json.dumps({"status": "listening", "message": "Listening"}), encoding="utf-8")

    popup.request_cancel(state_path, reason="ctrl_c")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["cancel_requested"] is True
    assert state["cancel_reason"] == "ctrl_c"
    assert state["status"] == "cancel_requested"
    assert "cancel_requested_at" in state


def test_daemon_detects_visualization_cancel_request(tmp_path):
    state_path = tmp_path / "voice_state.json"
    wake.update_visualization_state(state_path, status="listening")

    assert not wake.is_visualization_cancel_requested(state_path)

    popup.request_cancel(state_path, reason="ctrl_c")

    assert wake.is_visualization_cancel_requested(state_path)


def test_record_command_returns_none_without_opening_stream_when_cancelled(monkeypatch):
    def fail_if_opened(*_args, **_kwargs):
        raise AssertionError("InputStream should not open after cancellation")

    cfg = dict(wake.DEFAULT_CONFIG)
    cfg.update({"speech_start_timeout_seconds": 1.0, "block_seconds": 0.1})
    monkeypatch.setattr(wake.sd, "InputStream", fail_if_opened)

    assert wake.record_command(cfg, cancel_check=lambda: True) is None


def test_play_tts_file_terminates_player_when_cancel_requested(monkeypatch):
    fake_proc = types.SimpleNamespace(
        returncode=None,
        terminated=False,
        killed=False,
        stderr="",
        stdout="",
    )

    def poll():
        return fake_proc.returncode

    def terminate():
        fake_proc.terminated = True
        fake_proc.returncode = -15

    def kill():
        fake_proc.killed = True
        fake_proc.returncode = -9

    def wait(timeout=None):
        return fake_proc.returncode

    fake_proc.poll = poll
    fake_proc.terminate = terminate
    fake_proc.kill = kill
    fake_proc.wait = wait

    monkeypatch.setattr(wake.subprocess, "Popen", lambda *_args, **_kwargs: fake_proc)
    monkeypatch.setattr(
        wake.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("playback should use cancellable Popen")),
    )
    monkeypatch.setattr(wake.time, "sleep", lambda *_args, **_kwargs: None)

    cancel_checks = iter([False, False, True])

    ok = wake.play_tts_file(
        {"playback_sink": "@DEFAULT_SINK@", "playback_volume": 1.0},
        "/tmp/response.wav",
        cancel_check=lambda: next(cancel_checks, True),
    )

    assert ok is False
    assert fake_proc.terminated is True


def test_handle_activation_passes_popup_cancel_check_to_response_playback(monkeypatch, tmp_path):
    state_path = tmp_path / "voice_state.json"
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")

    def fake_launch_visualization(_cfg, probability):
        wake.update_visualization_state(
            state_path,
            status="listening",
            probability=probability,
            cancel_requested=False,
            cancel_reason="",
        )
        return state_path

    def fake_speak_response(_cfg, text, *, cancel_check):
        assert text == "spoken answer"
        assert cancel_check() is False
        popup.request_cancel(state_path, reason="ctrl_c")
        assert cancel_check() is True

    wake.STOP.clear()
    monkeypatch.setattr(wake, "save_activation_archive", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wake, "launch_visualization", fake_launch_visualization)
    monkeypatch.setattr(wake, "maybe_beep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wake, "record_command", lambda *_args, **_kwargs: command_path)
    monkeypatch.setattr(wake, "archive_command_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wake, "transcribe_command", lambda *_args, **_kwargs: "what is up")
    monkeypatch.setattr(wake, "route_transcribed_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wake, "answer_routed_request", lambda *_args, **_kwargs: ("spoken answer", [], "heavy_agent"))
    monkeypatch.setattr(wake, "speak_response", fake_speak_response)

    wake.handle_activation({"conversation_mode_enabled": False}, {"probability": 0.9})

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "cancelled"
    assert state["cancel_reason"] == "ctrl_c"


def test_handle_activation_passes_popup_cancel_check_to_heavy_agent_execution(monkeypatch, tmp_path):
    state_path = tmp_path / "voice_state.json"
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")

    def fake_launch_visualization(_cfg, probability):
        wake.update_visualization_state(
            state_path,
            status="listening",
            probability=probability,
            cancel_requested=False,
            cancel_reason="",
        )
        return state_path

    def fake_answer_routed_request(_cfg, transcript, plan, history, *, cancel_check):
        assert transcript == "start a complex task"
        assert cancel_check() is False
        popup.request_cancel(state_path, reason="ctrl_c")
        assert cancel_check() is True
        return None, history, "heavy_agent"

    wake.STOP.clear()
    monkeypatch.setattr(wake, "save_activation_archive", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wake, "launch_visualization", fake_launch_visualization)
    monkeypatch.setattr(wake, "maybe_beep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wake, "record_command", lambda *_args, **_kwargs: command_path)
    monkeypatch.setattr(wake, "archive_command_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wake, "transcribe_command", lambda *_args, **_kwargs: "start a complex task")
    monkeypatch.setattr(wake, "route_transcribed_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wake, "answer_routed_request", fake_answer_routed_request)
    monkeypatch.setattr(wake, "speak_response", lambda *_args, **_kwargs: None)

    wake.handle_activation({"conversation_mode_enabled": False}, {"probability": 0.9})

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "cancelled"
    assert state["cancel_reason"] == "ctrl_c"


def test_default_wakeword_model_matches_public_artifact():
    assert wake.DEFAULT_CONFIG["model_path"].endswith(
        "okay-hermes-repcnn-onnx/wakeword.onnx"
    )
    assert wake.DEFAULT_CONFIG["threshold"] == 0.6973556280136108


def test_save_activation_archive_writes_wake_clip_and_metadata(tmp_path):
    cfg = {
        "activation_archive_dir": str(tmp_path / "activations"),
        "sample_rate": 16000,
        "window_seconds": 3.0,
        "threshold": 0.6973556280136108,
        "model_path": "/models/okay-hermes.onnx",
    }
    waveform = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    activation = {
        "probability": 0.92,
        "scores": [0.88, 0.92],
        "waveform": waveform,
        "sample_rate": 16000,
        "detected_at": 1234.5,
    }

    archive = wake.save_activation_archive(cfg, activation)

    assert archive is not None
    wav_path = Path(archive["wake_wav_path"])
    meta_path = Path(archive["metadata_path"])
    assert wav_path.parent == tmp_path / "activations"
    assert wav_path.exists()
    assert meta_path.exists()
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 16000
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["probability"] == 0.92
    assert metadata["scores"] == [0.88, 0.92]
    assert metadata["sample_rate"] == 16000
    assert metadata["status"] == "wake_detected"
    assert metadata["model_path"] == "/models/okay-hermes.onnx"


def test_update_activation_archive_metadata_merges_turn_details(tmp_path):
    meta_path = tmp_path / "activation.json"
    meta_path.write_text(json.dumps({"status": "wake_detected", "turns": []}), encoding="utf-8")
    archive = {"metadata_path": str(meta_path)}

    wake.update_activation_archive_metadata(
        archive,
        status="completed",
        turns=[{"turn": 1, "transcript": "hello", "response": "hi"}],
        close_reason="close_phrase",
    )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["turns"] == [{"turn": 1, "transcript": "hello", "response": "hi"}]
    assert metadata["close_reason"] == "close_phrase"
