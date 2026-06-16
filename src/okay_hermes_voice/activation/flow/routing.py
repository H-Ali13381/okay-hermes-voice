"""Route, answer, and speak normal activation requests."""
from __future__ import annotations

from typing import Any, Dict, List

from .constants import VOICE_SESSION_CANCELLED, VOICE_SESSION_COMPLETED
from .services import ActivationFlowServices
from .timing import elapsed_seconds, merge_speak_timing
from .visualization import publish_pipeline_stage, publish_turn_timing


class RoutedTurnHandler:
    def __init__(
        self,
        deps: ActivationFlowServices,
        cfg: Dict[str, Any],
        visual_state: Any,
        activation_archive: Any,
        archive_turns: List[Dict[str, Any]],
        turn_timings: List[Dict[str, Any]],
        voice_cancel_requested: Any,
        stop_if_cancelled: Any,
    ):
        self.deps = deps
        self.cfg = cfg
        self.visual_state = visual_state
        self.activation_archive = activation_archive
        self.archive_turns = archive_turns
        self.turn_timings = turn_timings
        self.voice_cancel_requested = voice_cancel_requested
        self.stop_if_cancelled = stop_if_cancelled

    def answer(
        self,
        turn_index: int,
        transcript: str,
        turn_started: float,
        turn_timing: Dict[str, Any],
        archived_command_path: Any,
        command_path: Any,
        hermes_history: List[Dict[str, Any]],
        conversation_enabled: bool,
    ) -> tuple[str, List[Dict[str, Any]], bool]:
        publish_pipeline_stage(
            self.deps,
            self.visual_state,
            "route",
            "route: transcript ready; choosing the right handler…",
            transcript=transcript,
            response="",
            error="",
            current_turn=turn_index,
        )
        route_started = self.deps.time.monotonic()
        interaction_plan = self.deps.route_transcribed_request(self.cfg, transcript, cancel_check=self.voice_cancel_requested)
        turn_timing["route_seconds"] = elapsed_seconds(self.deps.time, route_started)
        if self.stop_if_cancelled():
            return VOICE_SESSION_CANCELLED, hermes_history, False

        publish_pipeline_stage(
            self.deps,
            self.visual_state,
            "answer",
            f"answer: {self.deps.routed_request_status_message(interaction_plan)}",
            transcript=transcript,
            response="",
            error="",
            current_turn=turn_index,
            interaction_ack_text=self.deps.interaction_ack_text(interaction_plan),
        )
        router_metadata: Dict[str, Any] = {}
        if interaction_plan:
            router_metadata = {
                "interaction_route_target": interaction_plan.route.target.value,
                "interaction_route_reason": interaction_plan.route.reason,
                "interaction_ack_template": interaction_plan.route.ack_template_id.value,
                "interaction_ack_text": self.deps.interaction_ack_text(interaction_plan),
                "interaction_router_confidence": interaction_plan.decision.confidence,
                "interaction_router_reason": interaction_plan.decision.brief_reason,
            }
        self.deps.update_activation_archive_metadata(
            self.activation_archive,
            status="thinking",
            current_turn=turn_index,
            latest_transcript=transcript,
            **router_metadata,
            **self.deps.command_audio_metadata_fields(archived_command_path, command_path, latest=True),
        )
        if self.stop_if_cancelled():
            return VOICE_SESSION_CANCELLED, hermes_history, False
        answer_started = self.deps.time.monotonic()
        response, hermes_history, response_source = self.deps.answer_routed_request(
            self.cfg,
            transcript,
            interaction_plan,
            hermes_history,
            cancel_check=self.voice_cancel_requested,
        )
        turn_timing["answer_seconds"] = elapsed_seconds(self.deps.time, answer_started)
        if self.stop_if_cancelled():
            return VOICE_SESSION_CANCELLED, hermes_history, False
        if not response:
            self.deps.update_visualization_state(
                self.visual_state,
                status="error",
                message="Hermes did not return a response.",
                transcript=transcript,
                error="No response from Hermes.",
            )
            self.deps.update_activation_archive_metadata(
                self.activation_archive,
                status="hermes_no_response",
                close_reason="no_response",
                latest_transcript=transcript,
                turns=self.archive_turns,
            )
            self.deps.maybe_beep(self.cfg, frequency=330, count=2)
            return VOICE_SESSION_COMPLETED, hermes_history, False

        self.archive_turns.append({
            "turn": turn_index,
            "transcript": transcript,
            "response": response,
            "response_source": response_source,
            **self.deps.command_audio_metadata_fields(archived_command_path, command_path),
        })
        self.deps.update_activation_archive_metadata(
            self.activation_archive,
            status="speaking",
            latest_transcript=transcript,
            latest_response=response,
            latest_response_source=response_source,
            turns=self.archive_turns,
        )
        self.deps.append_visualization_turn(self.visual_state, transcript=transcript, response=response)

        def publish_speech_stage(stage: str) -> None:
            message = (
                "TTS: Hermes answered; generating spoken audio…"
                if stage == "tts"
                else (
                    "playback: playing the response; then I’ll keep listening…"
                    if conversation_enabled
                    else "playback: playing the response…"
                )
            )
            publish_pipeline_stage(
                self.deps,
                self.visual_state,
                stage,
                message,
                transcript=transcript,
                response=response,
                error="",
                current_turn=turn_index,
            )

        if self.stop_if_cancelled():
            return VOICE_SESSION_CANCELLED, hermes_history, False
        speak_started = self.deps.time.monotonic()
        speak_result = self.deps.speak_response(
            self.cfg,
            response,
            cancel_check=self.voice_cancel_requested,
            stage_callback=publish_speech_stage,
        )
        merge_speak_timing(turn_timing, speak_result, elapsed_seconds(self.deps.time, speak_started))
        turn_timing["turn_seconds"] = elapsed_seconds(self.deps.time, turn_started)
        if self.archive_turns:
            self.archive_turns[-1]["timings"] = dict(turn_timing)
        publish_turn_timing(self.deps, self.visual_state, self.activation_archive, self.turn_timings, turn_timing, self.archive_turns)
        if self.stop_if_cancelled():
            return VOICE_SESSION_CANCELLED, hermes_history, False
        if not conversation_enabled:
            self.deps.update_visualization_state(
                self.visual_state,
                status="done",
                message="Voice request complete.",
                transcript=transcript,
                response=response,
                error="",
            )
            self.deps.update_activation_archive_metadata(
                self.activation_archive,
                status="completed",
                close_reason="single_turn_complete",
                turns=self.archive_turns,
            )
            return VOICE_SESSION_COMPLETED, hermes_history, False
        self.deps.update_activation_archive_metadata(
            self.activation_archive,
            status="awaiting_followup",
            turns=self.archive_turns,
            current_turn=turn_index + 1,
        )
        return VOICE_SESSION_COMPLETED, hermes_history, True
