from __future__ import annotations

import json

import pytest

from okay_hermes_voice import activation_flow as flow
from okay_hermes_voice import interaction_router as router
from okay_hermes_voice import voice_activation_popup as popup
from okay_hermes_voice import wakeword_daemon as wake


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
        flow.update_visualization_state(
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

    flow.STOP.clear()
    monkeypatch.setattr(flow, "save_activation_archive", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "launch_visualization", fake_launch_visualization)
    monkeypatch.setattr(flow, "maybe_beep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "record_command", lambda *_args, **_kwargs: command_path)
    monkeypatch.setattr(flow, "archive_command_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "transcribe_command", lambda *_args, **_kwargs: "what is the weather")
    monkeypatch.setattr(flow, "route_transcribed_request", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(flow, "answer_routed_request", fake_answer_routed_request)
    monkeypatch.setattr(flow, "speak_response", lambda *_args, **_kwargs: None)

    flow.handle_activation({"conversation_mode_enabled": False}, {"probability": 0.9})


def test_handle_activation_passes_popup_cancel_check_to_response_playback(monkeypatch, tmp_path):
    state_path = tmp_path / "voice_state.json"
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")

    def fake_launch_visualization(_cfg, probability):
        flow.update_visualization_state(
            state_path,
            status="listening",
            probability=probability,
            cancel_requested=False,
            cancel_reason="",
        )
        return state_path

    def fake_speak_response(_cfg, text, *, cancel_check, stage_callback=None):
        assert text == "spoken answer"
        assert cancel_check() is False
        popup.request_cancel(state_path, reason="ctrl_c")
        assert cancel_check() is True

    flow.STOP.clear()
    monkeypatch.setattr(flow, "save_activation_archive", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "launch_visualization", fake_launch_visualization)
    monkeypatch.setattr(flow, "maybe_beep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "record_command", lambda *_args, **_kwargs: command_path)
    monkeypatch.setattr(flow, "archive_command_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "transcribe_command", lambda *_args, **_kwargs: "what is up")
    monkeypatch.setattr(flow, "route_transcribed_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "answer_routed_request", lambda *_args, **_kwargs: ("spoken answer", [], "heavy_agent"))
    monkeypatch.setattr(flow, "speak_response", fake_speak_response)

    result = flow.handle_activation({"conversation_mode_enabled": False}, {"probability": 0.9})

    assert result == flow.VOICE_SESSION_CANCELLED
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "cancelled"
    assert state["cancel_reason"] == "ctrl_c"

def test_handle_activation_passes_popup_cancel_check_to_heavy_agent_execution(monkeypatch, tmp_path):
    state_path = tmp_path / "voice_state.json"
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")

    def fake_launch_visualization(_cfg, probability):
        flow.update_visualization_state(
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

    flow.STOP.clear()
    monkeypatch.setattr(flow, "save_activation_archive", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "launch_visualization", fake_launch_visualization)
    monkeypatch.setattr(flow, "maybe_beep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "record_command", lambda *_args, **_kwargs: command_path)
    monkeypatch.setattr(flow, "archive_command_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "transcribe_command", lambda *_args, **_kwargs: "start a complex task")
    monkeypatch.setattr(flow, "route_transcribed_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "answer_routed_request", fake_answer_routed_request)
    monkeypatch.setattr(flow, "speak_response", lambda *_args, **_kwargs: None)

    result = flow.handle_activation({"conversation_mode_enabled": False}, {"probability": 0.9})

    assert result == flow.VOICE_SESSION_CANCELLED
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "cancelled"
    assert state["cancel_reason"] == "ctrl_c"


def test_handle_activation_updates_popup_pipeline_stages(monkeypatch, tmp_path):
    state_path = tmp_path / "voice_state.json"
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")
    stages = []

    original_update = flow.update_visualization_state

    def capture_update(path, **updates):
        original_update(path, **updates)
        if path != state_path:
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        stage = state.get("pipeline_stage")
        if stage and (not stages or stages[-1] != stage):
            stages.append(stage)

    def fake_launch_visualization(_cfg, probability):
        capture_update(
            state_path,
            status="wake",
            pipeline_stage="wake",
            message="wake: wakeword detected",
            probability=probability,
            cancel_requested=False,
            cancel_reason="",
        )
        return state_path

    def fake_speak_response(_cfg, text, *, cancel_check, stage_callback):
        assert text == "spoken answer"
        assert cancel_check() is False
        stage_callback("tts")
        stage_callback("playback")
        return {"tts_success": True, "playback_success": True, "tts_seconds": 0.1, "playback_seconds": 0.2}

    flow.STOP.clear()
    monkeypatch.setattr(flow, "update_visualization_state", capture_update)
    monkeypatch.setattr(flow, "save_activation_archive", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "launch_visualization", fake_launch_visualization)
    monkeypatch.setattr(flow, "maybe_beep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "record_command", lambda *_args, **_kwargs: command_path)
    monkeypatch.setattr(flow, "archive_command_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "transcribe_command", lambda *_args, **_kwargs: "summarize this")
    monkeypatch.setattr(flow, "route_transcribed_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "answer_routed_request", lambda *_args, **_kwargs: ("spoken answer", [], "heavy_agent"))
    monkeypatch.setattr(flow, "speak_response", fake_speak_response)

    result = flow.handle_activation({"conversation_mode_enabled": False}, {"probability": 0.9})

    assert result == flow.VOICE_SESSION_COMPLETED
    assert stages[:7] == ["wake", "record", "transcript", "route", "answer", "tts", "playback"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "done"
    assert state["pipeline_stage"] == "playback"


def test_handle_activation_records_phase_zero_timing_in_popup_and_archive(monkeypatch, tmp_path):
    state_path = tmp_path / "voice_state.json"
    metadata_path = tmp_path / "activation.json"
    metadata_path.write_text("{}", encoding="utf-8")
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")
    archived_command_path = tmp_path / "archived-command.wav"

    clock = {"now": 100.0}

    def advance(seconds: float) -> None:
        clock["now"] += seconds

    def fake_launch_visualization(_cfg, probability):
        flow.update_visualization_state(
            state_path,
            status="listening",
            probability=probability,
            cancel_requested=False,
            cancel_reason="",
        )
        return state_path

    def fake_record_command(_cfg, *, cancel_check):
        assert cancel_check() is False
        advance(1.25)
        return command_path

    def fake_transcribe_command(path):
        assert path == command_path
        advance(0.5)
        return "time this request"

    def fake_route_transcribed_request(_cfg, transcript, *, cancel_check):
        assert transcript == "time this request"
        assert cancel_check() is False
        advance(0.25)
        return None

    def fake_answer_routed_request(_cfg, transcript, plan, history, *, cancel_check):
        assert transcript == "time this request"
        assert plan is None
        assert cancel_check() is False
        advance(2.0)
        return "spoken answer", history, "heavy_agent"

    def fake_speak_response(_cfg, text, *, cancel_check, stage_callback=None):
        assert text == "spoken answer"
        assert cancel_check() is False
        advance(1.5)
        return {
            "tts_success": True,
            "playback_success": True,
            "tts_seconds": 0.4,
            "playback_seconds": 1.1,
            "speak_seconds": 1.5,
            "tts_file_path": "/tmp/spoken-answer.wav",
        }

    flow.STOP.clear()
    monkeypatch.setattr(flow.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(flow.time, "time", lambda: 52.0)
    monkeypatch.setattr(flow, "save_activation_archive", lambda *_args, **_kwargs: {"metadata_path": str(metadata_path)})
    monkeypatch.setattr(flow, "launch_visualization", fake_launch_visualization)
    monkeypatch.setattr(flow, "maybe_beep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flow, "record_command", fake_record_command)
    monkeypatch.setattr(flow, "archive_command_audio", lambda *_args, **_kwargs: str(archived_command_path))
    monkeypatch.setattr(flow, "transcribe_command", fake_transcribe_command)
    monkeypatch.setattr(flow, "route_transcribed_request", fake_route_transcribed_request)
    monkeypatch.setattr(flow, "answer_routed_request", fake_answer_routed_request)
    monkeypatch.setattr(flow, "speak_response", fake_speak_response)

    flow.handle_activation({"conversation_mode_enabled": False}, {"probability": 0.9, "detected_at": 50.0})

    state = json.loads(state_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    timing = state["latest_turn_timing"]
    assert state["voice_session_timing"]["wake_to_handle_seconds"] == pytest.approx(2.0)
    assert metadata["voice_session_timing"] == state["voice_session_timing"]
    assert timing["turn"] == 1
    assert timing["wake_to_record_start_seconds"] == pytest.approx(2.0)
    assert timing["record_seconds"] == pytest.approx(1.25)
    assert timing["transcribe_seconds"] == pytest.approx(0.5)
    assert timing["route_seconds"] == pytest.approx(0.25)
    assert timing["answer_seconds"] == pytest.approx(2.0)
    assert timing["tts_seconds"] == pytest.approx(0.4)
    assert timing["playback_seconds"] == pytest.approx(1.1)
    assert timing["speak_seconds"] == pytest.approx(1.5)
    assert timing["turn_seconds"] == pytest.approx(5.5)
    assert state["turn_timings"] == [timing]
    assert metadata["latest_turn_timing"] == timing
    assert metadata["turn_timings"] == [timing]
    assert metadata["turns"][0]["timings"] == timing
