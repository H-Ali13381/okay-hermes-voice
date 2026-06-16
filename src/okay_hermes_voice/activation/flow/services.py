"""Production services required by the activation-flow runner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActivationFlowServices:
    """Ports used by activation orchestration.

    The public facade builds this from the real archive, audio, playback,
    visualization, and routing modules. The runner receives one explicit
    service object so the orchestration package does not import those outer
    systems directly.
    """

    archive_command_audio: Any
    command_audio_metadata_fields: Any
    save_activation_archive: Any
    update_activation_archive_metadata: Any
    record_command: Any
    transcribe_command: Any
    log: Any
    stop: Any
    maybe_beep: Any
    speak_response: Any
    append_visualization_turn: Any
    finish_cancelled_voice_session: Any
    is_visualization_cancel_requested: Any
    launch_visualization: Any
    update_visualization_state: Any
    visualization_cancel_reason: Any
    answer_routed_request: Any
    command_recording_config_for_turn: Any
    interaction_ack_text: Any
    is_close_transcript: Any
    route_transcribed_request: Any
    routed_request_status_message: Any
    time: Any
