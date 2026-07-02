"""Terminal and retry outcomes for activation turns."""
from __future__ import annotations

from typing import Any, Dict, List

from .constants import VOICE_SESSION_CANCELLED, VOICE_SESSION_COMPLETED
from .services import ActivationFlowServices
from .timing import elapsed_seconds, merge_speak_timing
from .visualization import publish_pipeline_stage, publish_turn_timing


class TurnOutcomeHandler:
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

    def no_recording(self, turn_index: int, first_turn: bool, conversation_enabled: bool) -> str | None:
        if self.cfg.get("_heavy_agent_delegation_pending", False) and conversation_enabled and not first_turn:
            self.deps.update_visualization_state(
                self.visual_state,
                status="thinking",
                message="The heavy agent is still working. I’m listening briefly for updates, then I’ll check again.",
                error="",
            )
            self.deps.update_activation_archive_metadata(
                self.activation_archive,
                status="waiting_for_heavy_agent",
                current_turn=turn_index,
                turns=self.archive_turns,
            )
            return None
        if first_turn or not conversation_enabled:
            self.deps.update_visualization_state(
                self.visual_state,
                status="error",
                message="Wakeword detected, but no spoken request was captured.",
                error="No speech heard before the start timeout.",
            )
            self.deps.update_activation_archive_metadata(
                self.activation_archive,
                status="no_first_command",
                close_reason="no_speech_after_wake",
                turns=self.archive_turns,
            )
            self.deps.maybe_beep(self.cfg, frequency=330, count=2)
            return VOICE_SESSION_COMPLETED
        self.deps.update_visualization_state(
            self.visual_state,
            status="listening",
            message="I did not catch that. Still listening; say “close” to end voice mode.",
            error="No usable speech captured for the follow-up turn.",
        )
        self.deps.update_activation_archive_metadata(
            self.activation_archive,
            status="followup_no_audio",
            current_turn=turn_index,
            turns=self.archive_turns,
        )
        self.deps.maybe_beep(self.cfg, frequency=330, count=1)
        return None

    def no_transcript(
        self,
        turn_index: int,
        first_turn: bool,
        conversation_enabled: bool,
        archived_command_path: Any,
        command_path: Any,
    ) -> str | None:
        if first_turn or not conversation_enabled:
            self.deps.update_visualization_state(
                self.visual_state,
                status="error",
                message="Speech was captured, but STT did not produce a usable request.",
                error="Empty, failed, or hallucinated transcript.",
            )
            self.deps.update_activation_archive_metadata(
                self.activation_archive,
                status="first_command_not_transcribed",
                close_reason="stt_empty_or_failed",
                **self.deps.command_audio_metadata_fields(archived_command_path, command_path, latest=True),
                turns=self.archive_turns,
            )
            self.deps.maybe_beep(self.cfg, frequency=330, count=2)
            return VOICE_SESSION_COMPLETED
        self.deps.update_visualization_state(
            self.visual_state,
            status="listening",
            message="I could not transcribe that follow-up. Still listening; say “close” to end voice mode.",
            error="Empty, failed, or hallucinated transcript.",
        )
        self.deps.update_activation_archive_metadata(
            self.activation_archive,
            status="followup_not_transcribed",
            current_turn=turn_index,
            **self.deps.command_audio_metadata_fields(archived_command_path, command_path, latest=True),
            turns=self.archive_turns,
        )
        self.deps.maybe_beep(self.cfg, frequency=330, count=1)
        return None

    def transcript_only(
        self,
        turn_index: int,
        transcript: str,
        turn_started: float,
        turn_timing: Dict[str, Any],
        archived_command_path: Any,
        command_path: Any,
    ) -> str:
        turn_timing["route_seconds"] = 0.0
        turn_timing["answer_seconds"] = 0.0
        turn_timing["speak_seconds"] = 0.0
        turn_timing["turn_seconds"] = elapsed_seconds(self.deps.time, turn_started)
        self.archive_turns.append({
            "turn": turn_index,
            "transcript": transcript,
            "response": "",
            "response_source": "transcript_only",
            "timings": dict(turn_timing),
            **self.deps.command_audio_metadata_fields(archived_command_path, command_path),
        })
        publish_turn_timing(self.deps, self.visual_state, self.activation_archive, self.turn_timings, turn_timing, self.archive_turns)
        self.deps.update_visualization_state(
            self.visual_state,
            status="done",
            message="Transcript captured; shadow comparison complete.",
            transcript=transcript,
            response="",
            error="",
        )
        self.deps.update_activation_archive_metadata(
            self.activation_archive,
            status="transcript_only_completed",
            close_reason="transcript_only",
            latest_transcript=transcript,
            turns=self.archive_turns,
            current_turn=turn_index,
            **self.deps.command_audio_metadata_fields(archived_command_path, command_path, latest=True),
        )
        self.deps.log.info("Transcript-only voice session completed: %r", transcript)
        return VOICE_SESSION_COMPLETED

    def close_command(
        self,
        turn_index: int,
        transcript: str,
        turn_started: float,
        turn_timing: Dict[str, Any],
        archived_command_path: Any,
        command_path: Any,
    ) -> str:
        ack = str(self.cfg.get("conversation_close_ack") or "").strip()
        turn_timing["route_seconds"] = 0.0
        turn_timing["answer_seconds"] = 0.0
        self.archive_turns.append({
            "turn": turn_index,
            "transcript": transcript,
            "response": ack,
            **self.deps.command_audio_metadata_fields(archived_command_path, command_path),
            "closed_session": True,
        })
        if ack:
            self.deps.append_visualization_turn(self.visual_state, transcript=transcript, response=ack)

            def publish_close_speech_stage(stage: str) -> None:
                message = (
                    "TTS: generating the close acknowledgement…"
                    if stage == "tts"
                    else "playback: playing the close acknowledgement…"
                )
                publish_pipeline_stage(
                    self.deps,
                    self.visual_state,
                    stage,
                    message,
                    transcript=transcript,
                    response=ack,
                    error="",
                    current_turn=turn_index,
                )

            speak_started = self.deps.time.monotonic()
            speak_result = self.deps.speak_response(
                self.cfg,
                ack,
                cancel_check=self.voice_cancel_requested,
                stage_callback=publish_close_speech_stage,
            )
            merge_speak_timing(turn_timing, speak_result, elapsed_seconds(self.deps.time, speak_started))
            if self.stop_if_cancelled():
                return VOICE_SESSION_CANCELLED
        else:
            turn_timing["speak_seconds"] = 0.0
        turn_timing["turn_seconds"] = elapsed_seconds(self.deps.time, turn_started)
        if self.archive_turns:
            self.archive_turns[-1]["timings"] = dict(turn_timing)
        publish_turn_timing(self.deps, self.visual_state, self.activation_archive, self.turn_timings, turn_timing, self.archive_turns)
        self.deps.update_visualization_state(
            self.visual_state,
            status="done",
            message="Voice conversation closed.",
            transcript=transcript,
            response=ack,
            error="",
        )
        self.deps.update_activation_archive_metadata(
            self.activation_archive,
            status="closed_by_voice_command",
            close_reason="close_phrase",
            latest_transcript=transcript,
            turns=self.archive_turns,
        )
        self.deps.log.info("Voice conversation closed by transcript: %r", transcript)
        return VOICE_SESSION_COMPLETED
