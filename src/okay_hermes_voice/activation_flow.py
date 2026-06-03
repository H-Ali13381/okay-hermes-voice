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

VOICE_SESSION_COMPLETED = "completed"
VOICE_SESSION_CANCELLED = "cancelled"

# Small, stable stage names make the popup and the archived timing fields line
# up: ``route_seconds`` measures the code that runs while ``pipeline_stage`` is
# ``route``, ``answer_seconds`` lines up with ``answer``, and so on.
# Human-readable messages can change; these keys should stay boring.
PIPELINE_STAGE_MESSAGES = {
    "record": "record: recording your request now…",
    "transcript": "transcript: speech captured; converting it to text…",
    "route": "route: choosing the right handler for this request…",
    "answer": "answer: Hermes is producing the response…",
    "tts": "TTS: generating spoken audio…",
    "playback": "playback: playing the spoken response…",
}


def _publish_pipeline_stage(visual_state: Any, stage: str, message: str | None = None, **updates: Any) -> None:
    """Publish one named voice-pipeline stage to the popup state file.

    ``status`` keeps compatibility with the existing popup state shape, while
    ``pipeline_stage`` gives tests and reviewers a stable stage vocabulary.
    """
    update_visualization_state(
        visual_state,
        status=stage,
        pipeline_stage=stage,
        message=message or PIPELINE_STAGE_MESSAGES.get(stage, stage),
        **updates,
    )


def _elapsed_seconds(started: float) -> float:
    """Return a monotonic duration for in-process work.

    Do not use wall-clock deltas for these fields; NTP/timezone/system-clock
    changes should not make an STT or playback duration negative.
    """
    return max(0.0, time.monotonic() - started)


def _merge_speak_timing(turn_timing: Dict[str, Any], speak_result: Any, fallback_seconds: float) -> None:
    """Copy TTS/playback timing returned by playback.speak_response into a turn record."""
    # ``speak_response`` now returns a dict, but older tests/doubles may still
    # return ``None``. Keep a coarse end-to-end speak duration in that case.
    turn_timing["speak_seconds"] = fallback_seconds
    if not isinstance(speak_result, dict):
        return
    for key in (
        "tts_enabled",
        "tts_success",
        "playback_success",
        "tts_seconds",
        "playback_seconds",
        "speak_seconds",
        "tts_file_path",
    ):
        if key in speak_result:
            turn_timing[key] = speak_result[key]


def _publish_turn_timing(
    visual_state: Any,
    activation_archive: Any,
    turn_timings: List[Dict[str, Any]],
    turn_timing: Dict[str, Any],
    archive_turns: List[Dict[str, Any]],
) -> None:
    """Expose the latest turn timing to the popup and durable activation archive."""
    # Copy before publishing so subsequent mutations cannot silently change the
    # snapshot shown in the popup or written into the archive JSON.
    stable_timing = dict(turn_timing)
    turn_timings.append(stable_timing)
    update_visualization_state(
        visual_state,
        latest_turn_timing=stable_timing,
        turn_timings=turn_timings,
    )
    update_activation_archive_metadata(
        activation_archive,
        latest_turn_timing=stable_timing,
        turn_timings=turn_timings,
        turns=archive_turns,
    )


def handle_activation(cfg: Dict[str, Any], activation: Any) -> str:
    """Run one wake-triggered voice session and return its terminal outcome.

    Timing model:
    - ``voice_session_timing`` covers the detector-to-handler boundary. It uses
      wall-clock seconds because the wake detector already records
      ``detected_at`` as ``time.time()``.
    - Per-turn phase timings use ``time.monotonic()`` because they measure work
      inside this process and should not move if the system clock jumps.
    """
    if isinstance(activation, dict):
        probability = float(activation.get("probability") or 0.0)
    else:
        probability = float(activation)
        activation = {"probability": probability, "scores": [probability], "detected_at": time.time()}
    # ``detected_at`` is wall-clock detector metadata. Keep wall time only for
    # wake-to-handle/wake-to-record measurements; use monotonic clocks after
    # control enters this handler.
    activation_detected_at = float(activation.get("detected_at") or time.time())
    handle_started_at = time.time()
    voice_session_timing = {
        "schema_version": 1,
        "activation_detected_at": activation_detected_at,
        "handle_started_at": handle_started_at,
        "wake_to_handle_seconds": max(0.0, handle_started_at - activation_detected_at),
    }
    LOG.info("Handling wake activation; probability=%.6f", probability)
    activation_archive = save_activation_archive(cfg, activation)
    visual_state = launch_visualization(cfg, probability)
    update_visualization_state(visual_state, voice_session_timing=voice_session_timing)
    update_activation_archive_metadata(activation_archive, voice_session_timing=voice_session_timing)
    if activation_archive:
        update_visualization_state(visual_state, activation_archive=activation_archive)
    maybe_beep(cfg, frequency=880, count=1)

    conversation_enabled = bool(cfg.get("conversation_mode_enabled", True))
    max_turns = max(1, int(cfg.get("conversation_max_turns") or 50))
    turn_index = 1
    archive_turns: List[Dict[str, Any]] = []
    turn_timings: List[Dict[str, Any]] = []
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
        turn_started = time.monotonic()
        # Built incrementally as the request moves through record -> STT ->
        # router -> answer -> TTS/playback, then published once the turn has a
        # response or close acknowledgement.
        turn_timing: Dict[str, Any] = {"turn": turn_index}
        record_message = (
            "record: wakeword detected; recording your request now…"
            if first_turn
            else "record: recording a follow-up. Say “close” to end voice mode."
        )
        _publish_pipeline_stage(
            visual_state,
            "record",
            record_message,
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
            return VOICE_SESSION_CANCELLED

        record_started = time.monotonic()
        record_wall_started = time.time()
        # This is the only per-turn field that intentionally crosses the
        # detector/handler boundary, so it shares the detector's wall clock.
        turn_timing["wake_to_record_start_seconds"] = max(0.0, record_wall_started - activation_detected_at)
        command_path = record_command(command_recording_config_for_turn(cfg, turn_index), cancel_check=voice_cancel_requested)
        turn_timing["record_seconds"] = _elapsed_seconds(record_started)
        if stop_if_cancelled():
            return VOICE_SESSION_CANCELLED
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
                return VOICE_SESSION_COMPLETED
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
        _publish_pipeline_stage(
            visual_state,
            "transcript",
            "transcript: speech captured; converting it to text…",
            current_turn=turn_index,
        )
        update_activation_archive_metadata(
            activation_archive,
            status="transcribing",
            current_turn=turn_index,
            **command_audio_metadata_fields(archived_command_path, command_path, latest=True),
        )
        transcribe_started = time.monotonic()
        transcript = transcribe_command(command_path)
        turn_timing["transcribe_seconds"] = _elapsed_seconds(transcribe_started)
        if stop_if_cancelled():
            return VOICE_SESSION_CANCELLED
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
                return VOICE_SESSION_COMPLETED
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
            # A close phrase skips the router and Hermes answer path by design.
            # Store explicit zeroes so archive consumers can rely on a stable
            # per-turn schema instead of treating missing keys as special cases.
            turn_timing["route_seconds"] = 0.0
            turn_timing["answer_seconds"] = 0.0
            archive_turns.append({
                "turn": turn_index,
                "transcript": transcript,
                "response": ack,
                **command_audio_metadata_fields(archived_command_path, command_path),
                "closed_session": True,
            })
            if ack:
                append_visualization_turn(visual_state, transcript=transcript, response=ack)

                def publish_close_speech_stage(stage: str) -> None:
                    # ``speak_response`` owns the real TTS/playback boundary;
                    # this callback mirrors that boundary into popup state.
                    message = (
                        "TTS: generating the close acknowledgement…"
                        if stage == "tts"
                        else "playback: playing the close acknowledgement…"
                    )
                    _publish_pipeline_stage(
                        visual_state,
                        stage,
                        message,
                        transcript=transcript,
                        response=ack,
                        error="",
                        current_turn=turn_index,
                    )

                speak_started = time.monotonic()
                speak_result = speak_response(
                    cfg,
                    ack,
                    cancel_check=voice_cancel_requested,
                    stage_callback=publish_close_speech_stage,
                )
                _merge_speak_timing(turn_timing, speak_result, _elapsed_seconds(speak_started))
            else:
                turn_timing["speak_seconds"] = 0.0
            turn_timing["turn_seconds"] = _elapsed_seconds(turn_started)
            if archive_turns:
                archive_turns[-1]["timings"] = dict(turn_timing)
            _publish_turn_timing(visual_state, activation_archive, turn_timings, turn_timing, archive_turns)
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
            return VOICE_SESSION_COMPLETED

        _publish_pipeline_stage(
            visual_state,
            "route",
            "route: transcript ready; choosing the right handler…",
            transcript=transcript,
            response="",
            error="",
            current_turn=turn_index,
        )
        route_started = time.monotonic()
        interaction_plan = route_transcribed_request(cfg, transcript, cancel_check=voice_cancel_requested)
        turn_timing["route_seconds"] = _elapsed_seconds(route_started)
        if stop_if_cancelled():
            return VOICE_SESSION_CANCELLED

        _publish_pipeline_stage(
            visual_state,
            "answer",
            f"answer: {routed_request_status_message(interaction_plan)}",
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
            return VOICE_SESSION_CANCELLED
        answer_started = time.monotonic()
        response, hermes_history, response_source = answer_routed_request(
            cfg,
            transcript,
            interaction_plan,
            hermes_history,
            cancel_check=voice_cancel_requested,
        )
        turn_timing["answer_seconds"] = _elapsed_seconds(answer_started)
        if stop_if_cancelled():
            return VOICE_SESSION_CANCELLED
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

            def publish_speech_stage(stage: str) -> None:
                # Keep popup stage transitions tied to the playback helper, not
                # guessed here, so failures/cancellations are attributed to the
                # phase that was actually running.
                message = (
                    "TTS: Hermes answered; generating spoken audio…"
                    if stage == "tts"
                    else (
                        "playback: playing the response; then I’ll keep listening…"
                        if conversation_enabled
                        else "playback: playing the response…"
                    )
                )
                _publish_pipeline_stage(
                    visual_state,
                    stage,
                    message,
                    transcript=transcript,
                    response=response,
                    error="",
                    current_turn=turn_index,
                )

            if stop_if_cancelled():
                return VOICE_SESSION_CANCELLED
            speak_started = time.monotonic()
            speak_result = speak_response(
                cfg,
                response,
                cancel_check=voice_cancel_requested,
                stage_callback=publish_speech_stage,
            )
            _merge_speak_timing(turn_timing, speak_result, _elapsed_seconds(speak_started))
            turn_timing["turn_seconds"] = _elapsed_seconds(turn_started)
            if archive_turns:
                archive_turns[-1]["timings"] = dict(turn_timing)
            _publish_turn_timing(visual_state, activation_archive, turn_timings, turn_timing, archive_turns)
            if stop_if_cancelled():
                return VOICE_SESSION_CANCELLED
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
                return VOICE_SESSION_COMPLETED
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
        return VOICE_SESSION_COMPLETED

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
    return VOICE_SESSION_COMPLETED
