"""Record and transcribe one activation turn."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .services import ActivationFlowServices
from .timing import elapsed_seconds
from .visualization import publish_pipeline_stage


@dataclass(frozen=True)
class TurnInput:
    transcript: str
    command_path: Path
    archived_command_path: Any


class TurnInputHandler:
    def __init__(
        self,
        deps: ActivationFlowServices,
        cfg: Dict[str, Any],
        visual_state: Any,
        activation_archive: Any,
        activation_detected_at: float,
        voice_cancel_requested: Any,
    ):
        self.deps = deps
        self.cfg = cfg
        self.visual_state = visual_state
        self.activation_archive = activation_archive
        self.activation_detected_at = activation_detected_at
        self.voice_cancel_requested = voice_cancel_requested

    def publish_recording_start(self, turn_index: int, first_turn: bool) -> None:
        record_message = (
            "record: wakeword detected; recording your request now…"
            if first_turn
            else "record: recording a follow-up. Say “close” to end voice mode."
        )
        publish_pipeline_stage(
            self.deps,
            self.visual_state,
            "record",
            record_message,
            current_turn=turn_index,
            error="",
        )
        self.deps.update_activation_archive_metadata(
            self.activation_archive,
            status="listening" if first_turn else "listening_followup",
            current_turn=turn_index,
        )
        if not first_turn and self.cfg.get("conversation_followup_beep_enabled", False):
            self.deps.maybe_beep(self.cfg, frequency=660, count=1)

    def record_and_transcribe(self, turn_index: int, turn_timing: Dict[str, Any]) -> Optional[TurnInput]:
        record_started = self.deps.time.monotonic()
        record_wall_started = self.deps.time.time()
        turn_timing["wake_to_record_start_seconds"] = max(0.0, record_wall_started - self.activation_detected_at)
        recording = self.deps.record_command(
            self.deps.command_recording_config_for_turn(self.cfg, turn_index),
            cancel_check=self.voice_cancel_requested,
        )
        turn_timing["record_seconds"] = elapsed_seconds(self.deps.time, record_started)
        if not recording:
            return None

        raw_command_path = getattr(recording, "path", None)
        command_path = raw_command_path if isinstance(raw_command_path, Path) else Path(str(raw_command_path or recording))
        live_transcript = str(getattr(recording, "live_transcript", "") or "").strip()
        turn_timing["live_stt_during_recording"] = bool(live_transcript)
        archived_command_path = self.deps.archive_command_audio(self.cfg, self.activation_archive, command_path, turn_index)
        publish_pipeline_stage(
            self.deps,
            self.visual_state,
            "transcript",
            "transcript: speech captured; converting it to text…",
            current_turn=turn_index,
        )
        self.deps.update_activation_archive_metadata(
            self.activation_archive,
            status="transcribing",
            current_turn=turn_index,
            **self.deps.command_audio_metadata_fields(archived_command_path, command_path, latest=True),
        )
        transcribe_started = self.deps.time.monotonic()
        if live_transcript:
            transcript = live_transcript
            turn_timing["transcribe_seconds"] = 0.0
            turn_timing["live_stt_used"] = True
            self.deps.log.info("Using live STT transcript captured during recording: %s", transcript)
        else:
            transcript = self.deps.transcribe_command(command_path, self.cfg)
            turn_timing["transcribe_seconds"] = elapsed_seconds(self.deps.time, transcribe_started)
            turn_timing["live_stt_used"] = False
        return TurnInput(
            transcript=transcript,
            command_path=command_path,
            archived_command_path=archived_command_path,
        )
