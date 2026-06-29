"""Voice command routing, acknowledgements, and response dispatch facade."""
from __future__ import annotations

from ..hermes_runtime import ask_hermes_turn
from ..interaction_router import (
    ACK_TEXT,
    AckTemplate,
    AcknowledgementCache,
    InteractionRouterConfig,
    RouteTarget,
    VoiceRequestPlan,
    answer_with_small_model,
    plan_voice_request,
)
from ..playback import play_tts_file, speak_response
from .ack_generation import _generate_ack_tts
from .ack_playback import play_interaction_ack
from .ack_playback_sync import _play_interaction_ack_sync
from .answer import answer_routed_request
from .close_detection import is_close_transcript
from .normalization import normalize_voice_command
from .planning import plan_interaction_route
from .recording_config import command_recording_config_for_turn
from .request_route import route_transcribed_request
from .router_config import interaction_router_config_from_daemon_config
from .status import interaction_ack_text, routed_request_status_message


def text_to_speech_tool(text: str) -> str:
    """Resolve Hermes TTS lazily so importing voice routing does not require Hermes tools."""
    from tools.tts_tool import text_to_speech_tool as _text_to_speech_tool
    return _text_to_speech_tool(text)


__all__ = [
    "answer_routed_request",
    "command_recording_config_for_turn",
    "interaction_ack_text",
    "interaction_router_config_from_daemon_config",
    "is_close_transcript",
    "normalize_voice_command",
    "plan_interaction_route",
    "play_interaction_ack",
    "route_transcribed_request",
    "routed_request_status_message",
    "text_to_speech_tool",
]
