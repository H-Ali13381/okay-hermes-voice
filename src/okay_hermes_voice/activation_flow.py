"""Public wake activation conversation facade."""
from __future__ import annotations

import time
from typing import Any, Dict

from .activation.flow import (
    ActivationFlowServices,
    VOICE_SESSION_CANCELLED,
    VOICE_SESSION_COMPLETED,
    handle_activation_impl,
)
from .daemon_config import LOG, STOP


def _activation_flow_services() -> ActivationFlowServices:
    """Assemble the production service graph for one activation session."""
    from .activation_archive import (
        archive_command_audio,
        command_audio_metadata_fields,
        save_activation_archive,
        update_activation_archive_metadata,
    )
    from .audio import record_command, transcribe_command
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

    return ActivationFlowServices(
        archive_command_audio=archive_command_audio,
        command_audio_metadata_fields=command_audio_metadata_fields,
        save_activation_archive=save_activation_archive,
        update_activation_archive_metadata=update_activation_archive_metadata,
        record_command=record_command,
        transcribe_command=transcribe_command,
        log=LOG,
        stop=STOP,
        maybe_beep=maybe_beep,
        speak_response=speak_response,
        append_visualization_turn=append_visualization_turn,
        finish_cancelled_voice_session=finish_cancelled_voice_session,
        is_visualization_cancel_requested=is_visualization_cancel_requested,
        launch_visualization=launch_visualization,
        update_visualization_state=update_visualization_state,
        visualization_cancel_reason=visualization_cancel_reason,
        answer_routed_request=answer_routed_request,
        command_recording_config_for_turn=command_recording_config_for_turn,
        interaction_ack_text=interaction_ack_text,
        is_close_transcript=is_close_transcript,
        route_transcribed_request=route_transcribed_request,
        routed_request_status_message=routed_request_status_message,
        time=time,
    )


def handle_activation(cfg: Dict[str, Any], activation: Any) -> str:
    return handle_activation_impl(_activation_flow_services(), cfg, activation)


__all__ = [
    "VOICE_SESSION_CANCELLED",
    "VOICE_SESSION_COMPLETED",
    "handle_activation",
]
