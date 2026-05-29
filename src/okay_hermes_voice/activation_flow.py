"""Wake activation conversation orchestration."""
from __future__ import annotations

import time
from typing import Any, Dict, List

from .activation_archive import (
    archive_command_audio,
    command_audio_metadata_fields,
    save_activation_archive,
    update_activation_archive_metadata,
)
from .audio_io import record_command, transcribe_command
from .daemon_config import LOG, STOP
from .playback import maybe_beep, speak_response
from .visualization import (
    append_visualization_turn,
    finish_cancelled_voice_session,
    is_visualization_cancel_requested,
    launch_visualization,
    update_visualization_state,
    visualization_cancel_reason,
)
from .voice_routing import (
    answer_routed_request,
    command_recording_config_for_turn,
    interaction_ack_text,
    is_close_transcript,
    route_transcribed_request,
    routed_request_status_message,
)


def handle_activation(cfg: Dict[str, Any], activation: Any) -> None:
    if isinstance(activation, dict):
        probability = float(activation.get("probability") or 0.0)
    else:
        probability = float(activation)
        activation = {"probability": probability, "scores": [probability], "detected_at": time.time()}
    LOG.info("Handling wake activation; probability=%.6f", probability)
    activation_archive = save_activation_archive(cfg, activation)
    visual_state = launch_visualization(cfg, probability)
    if activation_archive:
        update_visualization_state(visual_state, activation_archive=activation_archive)
    maybe_beep(cfg, frequency=880, count=1)

    conversation_enabled = bool(cfg.get("conversation_mode_enabled", True))
    max_turns = max(1, int(cfg.get("conversation_max_turns") or 50))
    turn_index = 1
    archive_turns: List[Dict[str, Any]] = []
    hermes_history: List[Dict[str, Any]] = []

    def voice_cancel_requested() -> bool:
        return is_visualization_cancel_requested(visual_state)

    def stop_if_cancelled() -> bool:
        if not voice_cancel_requested():
            return False
        finish_cancelled_voice_session(
            visual_state,
            activation_archive,
            archive_turns,
            visualization_cancel_reason(visual_state),
        )
        return True

    while not STOP.is_set() and turn_index <= max_turns:
        first_turn = turn_index == 1
        listening_message = (
            "Wakeword detected. Listening for your request…"
            if first_turn
            else "Listening for a follow-up. Say “close” to end voice mode."
        )
        update_visualization_state(
            visual_state,
            status="listening",
            message=listening_message,
            current_turn=turn_index,
            error="",
        )
        update_activation_archive_metadata(
            activation_archive,
            status="listening" if first_turn else "listening_followup",
            current_turn=turn_index,
        )
        if not first_turn and cfg.get("conversation_followup_beep_enabled", False):
            maybe_beep(cfg, frequency=660, count=1)

        if stop_if_cancelled():
            return

        command_path = record_command(command_recording_config_for_turn(cfg, turn_index), cancel_check=voice_cancel_requested)
        if stop_if_cancelled():
            return
        if not command_path:
            if first_turn or not conversation_enabled:
                update_visualization_state(
                    visual_state,
                    status="error",
                    message="Wakeword detected, but no spoken request was captured.",
                    error="No speech heard before the start timeout.",
                )
                update_activation_archive_metadata(
                    activation_archive,
                    status="no_first_command",
                    close_reason="no_speech_after_wake",
                    turns=archive_turns,
                )
                maybe_beep(cfg, frequency=330, count=2)
                return
            update_visualization_state(
                visual_state,
                status="listening",
                message="I did not catch that. Still listening; say “close” to end voice mode.",
                error="No usable speech captured for the follow-up turn.",
            )
            update_activation_archive_metadata(
                activation_archive,
                status="followup_no_audio",
                current_turn=turn_index,
                turns=archive_turns,
            )
            maybe_beep(cfg, frequency=330, count=1)
            continue

        archived_command_path = archive_command_audio(cfg, activation_archive, command_path, turn_index)
        update_visualization_state(
            visual_state,
            status="transcribing",
            message="Speech captured. Transcribing now…",
            current_turn=turn_index,
        )
        update_activation_archive_metadata(
            activation_archive,
            status="transcribing",
            current_turn=turn_index,
            **command_audio_metadata_fields(archived_command_path, command_path, latest=True),
        )
        transcript = transcribe_command(command_path)
        if stop_if_cancelled():
            return
        if not transcript:
            if first_turn or not conversation_enabled:
                update_visualization_state(
                    visual_state,
                    status="error",
                    message="Speech was captured, but STT did not produce a usable request.",
                    error="Empty, failed, or hallucinated transcript.",
                )
                update_activation_archive_metadata(
                    activation_archive,
                    status="first_command_not_transcribed",
                    close_reason="stt_empty_or_failed",
                    **command_audio_metadata_fields(archived_command_path, command_path, latest=True),
                    turns=archive_turns,
                )
                maybe_beep(cfg, frequency=330, count=2)
                return
            update_visualization_state(
                visual_state,
                status="listening",
                message="I could not transcribe that follow-up. Still listening; say “close” to end voice mode.",
                error="Empty, failed, or hallucinated transcript.",
            )
            update_activation_archive_metadata(
                activation_archive,
                status="followup_not_transcribed",
                current_turn=turn_index,
                **command_audio_metadata_fields(archived_command_path, command_path, latest=True),
                turns=archive_turns,
            )
            maybe_beep(cfg, frequency=330, count=1)
            continue

        if conversation_enabled and is_close_transcript(transcript, cfg):
            ack = str(cfg.get("conversation_close_ack") or "").strip()
            archive_turns.append({
                "turn": turn_index,
                "transcript": transcript,
                "response": ack,
                **command_audio_metadata_fields(archived_command_path, command_path),
                "closed_session": True,
            })
            if ack:
                append_visualization_turn(visual_state, transcript=transcript, response=ack)
                update_visualization_state(
                    visual_state,
                    status="speaking",
                    message="Closing voice mode…",
                    transcript=transcript,
                    response=ack,
                    error="",
                )
                speak_response(cfg, ack, cancel_check=voice_cancel_requested)
            update_visualization_state(
                visual_state,
                status="done",
                message="Voice conversation closed.",
                transcript=transcript,
                response=ack,
                error="",
            )
            update_activation_archive_metadata(
                activation_archive,
                status="closed_by_voice_command",
                close_reason="close_phrase",
                latest_transcript=transcript,
                turns=archive_turns,
            )
            LOG.info("Voice conversation closed by transcript: %r", transcript)
            return

        interaction_plan = route_transcribed_request(cfg, transcript, cancel_check=voice_cancel_requested)
        if stop_if_cancelled():
            return

        update_visualization_state(
            visual_state,
            status="thinking",
            message=routed_request_status_message(interaction_plan),
            transcript=transcript,
            response="",
            error="",
            current_turn=turn_index,
            interaction_ack_text=interaction_ack_text(interaction_plan),
        )
        router_metadata: Dict[str, Any] = {}
        if interaction_plan:
            router_metadata = {
                "interaction_route_target": interaction_plan.route.target.value,
                "interaction_route_reason": interaction_plan.route.reason,
                "interaction_ack_template": interaction_plan.route.ack_template_id.value,
                "interaction_ack_text": interaction_ack_text(interaction_plan),
                "interaction_router_confidence": interaction_plan.decision.confidence,
                "interaction_router_reason": interaction_plan.decision.brief_reason,
            }
        update_activation_archive_metadata(
            activation_archive,
            status="thinking",
            current_turn=turn_index,
            latest_transcript=transcript,
            **router_metadata,
            **command_audio_metadata_fields(archived_command_path, command_path, latest=True),
        )
        if stop_if_cancelled():
            return
        response, hermes_history, response_source = answer_routed_request(
            cfg,
            transcript,
            interaction_plan,
            hermes_history,
            cancel_check=voice_cancel_requested,
        )
        if stop_if_cancelled():
            return
        if response:
            archive_turns.append({
                "turn": turn_index,
                "transcript": transcript,
                "response": response,
                "response_source": response_source,
                **command_audio_metadata_fields(archived_command_path, command_path),
            })
            update_activation_archive_metadata(
                activation_archive,
                status="speaking",
                latest_transcript=transcript,
                latest_response=response,
                latest_response_source=response_source,
                turns=archive_turns,
            )
            append_visualization_turn(visual_state, transcript=transcript, response=response)
            update_visualization_state(
                visual_state,
                status="speaking",
                message=(
                    "Hermes responded. Speaking now; then I’ll keep listening…"
                    if conversation_enabled
                    else "Hermes responded. Generating and playing voice output…"
                ),
                transcript=transcript,
                response=response,
                error="",
                current_turn=turn_index,
            )
            if stop_if_cancelled():
                return
            speak_response(cfg, response, cancel_check=voice_cancel_requested)
            if stop_if_cancelled():
                return
            if not conversation_enabled:
                update_visualization_state(
                    visual_state,
                    status="done",
                    message="Voice request complete.",
                    transcript=transcript,
                    response=response,
                    error="",
                )
                update_activation_archive_metadata(
                    activation_archive,
                    status="completed",
                    close_reason="single_turn_complete",
                    turns=archive_turns,
                )
                return
            update_activation_archive_metadata(
                activation_archive,
                status="awaiting_followup",
                turns=archive_turns,
                current_turn=turn_index + 1,
            )
            turn_index += 1
            continue

        update_visualization_state(
            visual_state,
            status="error",
            message="Hermes did not return a response.",
            transcript=transcript,
            error="No response from Hermes.",
        )
        update_activation_archive_metadata(
            activation_archive,
            status="hermes_no_response",
            close_reason="no_response",
            latest_transcript=transcript,
            turns=archive_turns,
        )
        maybe_beep(cfg, frequency=330, count=2)
        return

    if not STOP.is_set():
        update_visualization_state(
            visual_state,
            status="done",
            message=f"Voice conversation hit the safety limit of {max_turns} turns.",
            error="Say the wakeword again to start a new voice session.",
        )
        update_activation_archive_metadata(
            activation_archive,
            status="max_turns_reached",
            close_reason="max_turns",
            turns=archive_turns,
        )
