from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from okay_hermes_voice import interaction_router as router
from okay_hermes_voice import voice_activation_popup as popup
from okay_hermes_voice import wakeword_daemon as wake
from okay_hermes_voice.activation.flow import (
    VOICE_SESSION_CANCELLED,
    VOICE_SESSION_COMPLETED,
    ActivationFlowServices,
    handle_activation_impl,
)
from okay_hermes_voice.activation_archive import command_audio_metadata_fields, update_activation_archive_metadata
from okay_hermes_voice.audio.recording_result import CommandRecording
from okay_hermes_voice.visualization import (
    append_visualization_turn,
    finish_cancelled_voice_session,
    is_visualization_cancel_requested,
    update_visualization_state,
    visualization_cancel_reason,
)
from okay_hermes_voice.voice_routing.close_detection import is_close_transcript
from okay_hermes_voice.voice_routing.status import interaction_ack_text, routed_request_status_message
from okay_hermes_voice.activation.flow.session_setup import start_activation_session


class _TestLog:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def _fail(message: str):
    def raise_assertion(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(message)

    return raise_assertion


def _launch_to_state(state_path):
    def launch(_cfg, probability, *, on_process_started=None):
        if on_process_started:
            on_process_started()
        update_visualization_state(
            state_path,
            status="listening",
            probability=probability,
            cancel_requested=False,
            cancel_reason="",
        )
        return state_path

    return launch


def _services(**overrides: Any) -> ActivationFlowServices:
    values = {
        "archive_command_audio": lambda *_args, **_kwargs: None,
        "command_audio_metadata_fields": command_audio_metadata_fields,
        "save_activation_archive": lambda *_args, **_kwargs: None,
        "update_activation_archive_metadata": update_activation_archive_metadata,
        "record_command": lambda *_args, **_kwargs: None,
        "transcribe_command": lambda *_args, **_kwargs: "",
        "log": _TestLog(),
        "stop": threading.Event(),
        "maybe_beep": lambda *_args, **_kwargs: None,
        "speak_response": lambda *_args, **_kwargs: None,
        "append_visualization_turn": append_visualization_turn,
        "finish_cancelled_voice_session": finish_cancelled_voice_session,
        "is_visualization_cancel_requested": is_visualization_cancel_requested,
        "launch_visualization": _fail("launch_visualization was not configured"),
        "update_visualization_state": update_visualization_state,
        "visualization_cancel_reason": visualization_cancel_reason,
        "answer_routed_request": lambda *_args, **_kwargs: ("spoken answer", [], "heavy_agent"),
        "command_recording_config_for_turn": lambda cfg, _turn: cfg,
        "interaction_ack_text": interaction_ack_text,
        "is_close_transcript": is_close_transcript,
        "route_transcribed_request": lambda *_args, **_kwargs: None,
        "routed_request_status_message": routed_request_status_message,
        "time": time,
    }
    values.update(overrides)
    return ActivationFlowServices(**values)


def _handle(services: ActivationFlowServices, cfg: dict[str, Any], activation: Any) -> str:
    services.stop.clear()
    return handle_activation_impl(services, cfg, activation)


def test_wake_beep_starts_when_popup_process_starts_not_after_launch_returns(tmp_path):
    events: list[str] = []
    state_path = tmp_path / "voice_state.json"

    def launch(_cfg, probability, *, on_process_started=None):
        events.append("launch-start")
        assert probability == 0.9
        if on_process_started:
            on_process_started()
        events.append("launch-end")
        update_visualization_state(state_path, status="wake")
        return state_path

    services = _services(
        launch_visualization=launch,
        maybe_beep=lambda _cfg, frequency, count: events.append(f"beep-{frequency}-{count}"),
    )

    start_activation_session(services, {}, {"probability": 0.9, "detected_at": 1.0})

    assert events == ["launch-start", "beep-880-1", "launch-end"]


def test_handle_activation_surfaces_routing_ack_text_in_popup(tmp_path):
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

    def fake_answer_routed_request(_cfg, transcript, routed_plan, history, *, cancel_check):
        assert transcript == "what is the weather"
        assert routed_plan is plan
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["interaction_ack_text"] == "Okay, I’m on it."
        assert "Okay, I’m on it." in state["message"]
        assert "heavy agent" in state["message"]
        return "spoken answer", history, "heavy_agent"

    services = _services(
        launch_visualization=_launch_to_state(state_path),
        record_command=lambda *_args, **_kwargs: command_path,
        transcribe_command=lambda *_args, **_kwargs: "what is the weather",
        route_transcribed_request=lambda *_args, **_kwargs: plan,
        answer_routed_request=fake_answer_routed_request,
    )

    _handle(services, {"conversation_mode_enabled": False}, {"probability": 0.9})


def test_handle_activation_stops_looping_ack_when_answer_is_ready(tmp_path):
    state_path = tmp_path / "voice_state.json"
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")
    ack_cancel_check = None
    route_kwargs = None
    decision = router.RouterDecision(confidence=0.0)
    route = router.VoiceRoute(
        wake.RouteTarget.HEAVY_AGENT,
        wake.AckTemplate.GOT_IT,
        "low_router_confidence",
    )
    plan = wake.VoiceRequestPlan("what is the weather", decision, route)

    def fake_route_transcribed_request(_cfg, _transcript, **kwargs):
        nonlocal ack_cancel_check, route_kwargs
        route_kwargs = kwargs
        ack_cancel_check = kwargs["cancel_check"]
        assert ack_cancel_check() is False
        return plan

    def fake_answer_routed_request(_cfg, _transcript, _plan, history, *, cancel_check):
        assert ack_cancel_check is not None
        assert ack_cancel_check() is False
        assert cancel_check() is False
        return "spoken answer", history, "heavy_agent"

    def fake_speak_response(_cfg, _text, **_kwargs):
        assert ack_cancel_check is not None
        assert ack_cancel_check() is True

    services = _services(
        launch_visualization=_launch_to_state(state_path),
        record_command=lambda *_args, **_kwargs: command_path,
        transcribe_command=lambda *_args, **_kwargs: "what is the weather",
        route_transcribed_request=fake_route_transcribed_request,
        answer_routed_request=fake_answer_routed_request,
        speak_response=fake_speak_response,
    )

    _handle(services, {"conversation_mode_enabled": False}, {"probability": 0.9})

    assert route_kwargs is not None
    assert route_kwargs["loop_ack_until_cancelled"] is True


def test_handle_activation_passes_popup_cancel_check_to_response_playback(tmp_path):
    state_path = tmp_path / "voice_state.json"
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")

    def fake_speak_response(_cfg, text, *, cancel_check, stage_callback=None):
        assert text == "spoken answer"
        assert cancel_check() is False
        popup.request_cancel(state_path, reason="ctrl_c")
        assert cancel_check() is True

    services = _services(
        launch_visualization=_launch_to_state(state_path),
        record_command=lambda *_args, **_kwargs: command_path,
        transcribe_command=lambda *_args, **_kwargs: "what is up",
        answer_routed_request=lambda *_args, **_kwargs: ("spoken answer", [], "heavy_agent"),
        speak_response=fake_speak_response,
    )

    result = _handle(services, {"conversation_mode_enabled": False}, {"probability": 0.9})

    assert result == VOICE_SESSION_CANCELLED
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "cancelled"
    assert state["cancel_reason"] == "ctrl_c"


def test_handle_activation_honors_cancel_during_close_ack_playback(tmp_path):
    state_path = tmp_path / "voice_state.json"
    metadata_path = tmp_path / "activation.json"
    metadata_path.write_text("{}", encoding="utf-8")
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")

    def fake_speak_response(_cfg, text, *, cancel_check, stage_callback=None):
        assert text == "Closing voice mode."
        assert cancel_check() is False
        popup.request_cancel(state_path, reason="ctrl_c")
        assert cancel_check() is True
        if stage_callback:
            stage_callback("playback")
        return {"tts_success": True, "playback_success": False, "speak_seconds": 0.1}

    services = _services(
        save_activation_archive=lambda *_args, **_kwargs: {"metadata_path": str(metadata_path)},
        launch_visualization=_launch_to_state(state_path),
        record_command=lambda *_args, **_kwargs: command_path,
        transcribe_command=lambda *_args, **_kwargs: "close",
        route_transcribed_request=_fail("close path should not route"),
        answer_routed_request=_fail("close path should not answer"),
        speak_response=fake_speak_response,
    )

    result = _handle(
        services,
        {
            "conversation_mode_enabled": True,
            "conversation_close_ack": "Closing voice mode.",
            "conversation_close_phrases": ["close"],
        },
        {"probability": 0.9},
    )

    assert result == VOICE_SESSION_CANCELLED
    state = json.loads(state_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert state["status"] == "cancelled"
    assert state["cancel_reason"] == "ctrl_c"
    assert metadata["status"] == "cancelled_by_terminal"
    assert metadata["cancel_reason"] == "ctrl_c"


def test_handle_activation_transcript_only_mode_skips_route_answer_and_tts(tmp_path):
    state_path = tmp_path / "voice_state.json"
    metadata_path = tmp_path / "activation.json"
    metadata_path.write_text("{}", encoding="utf-8")
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")
    archived_command_path = tmp_path / "archived-command.wav"

    services = _services(
        save_activation_archive=lambda *_args, **_kwargs: {"metadata_path": str(metadata_path)},
        launch_visualization=_launch_to_state(state_path),
        record_command=lambda *_args, **_kwargs: command_path,
        archive_command_audio=lambda *_args, **_kwargs: str(archived_command_path),
        transcribe_command=lambda *_args, **_kwargs: "how are you doing today",
        route_transcribed_request=_fail("transcript-only path should not route"),
        answer_routed_request=_fail("transcript-only path should not answer"),
        speak_response=_fail("transcript-only path should not speak"),
    )

    result = _handle(services, {"conversation_mode_enabled": False, "transcript_only_mode": True}, {"probability": 0.9})

    assert result == VOICE_SESSION_COMPLETED
    state = json.loads(state_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert state["status"] == "done"
    assert state["message"] == "Transcript captured; shadow comparison complete."
    assert state["transcript"] == "how are you doing today"
    assert metadata["status"] == "transcript_only_completed"
    assert metadata["close_reason"] == "transcript_only"
    assert metadata["latest_transcript"] == "how are you doing today"
    assert metadata["turns"][0]["response_source"] == "transcript_only"
    assert metadata["turns"][0]["timings"]["route_seconds"] == 0.0
    assert metadata["turns"][0]["timings"]["answer_seconds"] == 0.0
    assert metadata["turns"][0]["timings"]["speak_seconds"] == 0.0


def test_handle_activation_uses_live_recording_transcript_without_post_wav_stt(tmp_path):
    state_path = tmp_path / "voice_state.json"
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")
    seen = []

    def fake_answer_routed_request(_cfg, transcript, plan, history, *, cancel_check):
        seen.append(transcript)
        return "spoken answer", history, "heavy_agent"

    services = _services(
        launch_visualization=_launch_to_state(state_path),
        record_command=lambda *_args, **_kwargs: CommandRecording(
            path=command_path,
            live_transcript="streamed nemotron transcript",
        ),
        transcribe_command=_fail("post-WAV STT should not run after live Nemotron"),
        answer_routed_request=fake_answer_routed_request,
        speak_response=lambda *_args, **_kwargs: {"speak_seconds": 0.0},
    )

    result = _handle(services, {"conversation_mode_enabled": False}, {"probability": 0.9})

    assert result == VOICE_SESSION_COMPLETED
    assert seen == ["streamed nemotron transcript"]


def test_handle_activation_passes_popup_cancel_check_to_heavy_agent_execution(tmp_path):
    state_path = tmp_path / "voice_state.json"
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")

    def fake_answer_routed_request(_cfg, transcript, plan, history, *, cancel_check):
        assert transcript == "start a complex task"
        assert cancel_check() is False
        popup.request_cancel(state_path, reason="ctrl_c")
        assert cancel_check() is True
        return None, history, "heavy_agent"

    services = _services(
        launch_visualization=_launch_to_state(state_path),
        record_command=lambda *_args, **_kwargs: command_path,
        transcribe_command=lambda *_args, **_kwargs: "start a complex task",
        answer_routed_request=fake_answer_routed_request,
    )

    result = _handle(services, {"conversation_mode_enabled": False}, {"probability": 0.9})

    assert result == VOICE_SESSION_CANCELLED
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "cancelled"
    assert state["cancel_reason"] == "ctrl_c"


def test_handle_activation_updates_popup_pipeline_stages(tmp_path):
    state_path = tmp_path / "voice_state.json"
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")
    stages = []

    def capture_update(path, **updates):
        update_visualization_state(path, **updates)
        if path != state_path:
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        stage = state.get("pipeline_stage")
        if stage and (not stages or stages[-1] != stage):
            stages.append(stage)

    def fake_launch_visualization(_cfg, probability, *, on_process_started=None):
        if on_process_started:
            on_process_started()
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

    services = _services(
        launch_visualization=fake_launch_visualization,
        update_visualization_state=capture_update,
        record_command=lambda *_args, **_kwargs: command_path,
        transcribe_command=lambda *_args, **_kwargs: "summarize this",
        answer_routed_request=lambda *_args, **_kwargs: ("spoken answer", [], "heavy_agent"),
        speak_response=fake_speak_response,
    )

    result = _handle(services, {"conversation_mode_enabled": False}, {"probability": 0.9})

    assert result == VOICE_SESSION_COMPLETED
    assert stages[:7] == ["wake", "record", "transcript", "route", "answer", "tts", "playback"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "done"
    assert state["pipeline_stage"] == "playback"


def test_handle_activation_records_phase_zero_timing_in_popup_and_archive(tmp_path):
    state_path = tmp_path / "voice_state.json"
    metadata_path = tmp_path / "activation.json"
    metadata_path.write_text("{}", encoding="utf-8")
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")
    archived_command_path = tmp_path / "archived-command.wav"
    clock = {"now": 100.0}

    class FakeTime:
        @staticmethod
        def monotonic() -> float:
            return clock["now"]

        @staticmethod
        def time() -> float:
            return 52.0

    def advance(seconds: float) -> None:
        clock["now"] += seconds

    def fake_record_command(_cfg, *, cancel_check):
        assert cancel_check() is False
        advance(1.25)
        return command_path

    def fake_transcribe_command(path, cfg):
        assert path == command_path
        assert cfg["conversation_mode_enabled"] is False
        advance(0.5)
        return "time this request"

    def fake_route_transcribed_request(_cfg, transcript, *, cancel_check, loop_ack_until_cancelled=False):
        assert transcript == "time this request"
        assert loop_ack_until_cancelled is True
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

    services = _services(
        save_activation_archive=lambda *_args, **_kwargs: {"metadata_path": str(metadata_path)},
        launch_visualization=_launch_to_state(state_path),
        record_command=fake_record_command,
        archive_command_audio=lambda *_args, **_kwargs: str(archived_command_path),
        transcribe_command=fake_transcribe_command,
        route_transcribed_request=fake_route_transcribed_request,
        answer_routed_request=fake_answer_routed_request,
        speak_response=fake_speak_response,
        time=FakeTime,
    )

    _handle(services, {"conversation_mode_enabled": False}, {"probability": 0.9, "detected_at": 50.0})

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
