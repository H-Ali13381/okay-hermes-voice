from __future__ import annotations

import json

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

    def fake_speak_response(_cfg, text, *, cancel_check):
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

    flow.handle_activation({"conversation_mode_enabled": False}, {"probability": 0.9})

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

    flow.handle_activation({"conversation_mode_enabled": False}, {"probability": 0.9})

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "cancelled"
    assert state["cancel_reason"] == "ctrl_c"
