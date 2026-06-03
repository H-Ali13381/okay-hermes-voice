"""Voice-session stage vocabulary and popup publishing helpers.

These names are the review/debugging spine for Phase 0 observability. The
activation flow owns the business sequence; this module owns the stable labels
shown in the popup while that sequence runs.
"""
from __future__ import annotations

from typing import Any

from .visualization import update_visualization_state


# These keys intentionally match timing field stems. When the popup says
# ``pipeline_stage == "answer"``, the matching archive field is
# ``answer_seconds``. The text can be edited for UX; the keys are the API.
PIPELINE_STAGE_MESSAGES = {
    "record": "record: recording your request now…",
    "transcript": "transcript: speech captured; converting it to text…",
    "route": "route: choosing the right handler for this request…",
    "answer": "answer: Hermes is producing the response…",
    "tts": "TTS: generating spoken audio…",
    "playback": "playback: playing the spoken response…",
}


def record_stage_message(first_turn: bool) -> str:
    """Return the recording message for initial wake vs follow-up turns."""
    if first_turn:
        return "record: wakeword detected; recording your request now…"
    return "record: recording a follow-up. Say “close” to end voice mode."


def close_ack_speech_stage_message(stage: str) -> str:
    """Describe the TTS/playback phase for a close-phrase acknowledgement."""
    if stage == "tts":
        return "TTS: generating the close acknowledgement…"
    return "playback: playing the close acknowledgement…"


def response_speech_stage_message(stage: str, *, conversation_enabled: bool) -> str:
    """Describe the TTS/playback phase for a normal Hermes response."""
    if stage == "tts":
        return "TTS: Hermes answered; generating spoken audio…"
    if conversation_enabled:
        return "playback: playing the response; then I’ll keep listening…"
    return "playback: playing the response…"


def publish_pipeline_stage(visual_state: Any, stage: str, message: str | None = None, **updates: Any) -> None:
    """Write one stable pipeline stage into the popup state file.

    ``status`` remains for the existing popup state contract. ``pipeline_stage``
    is the durable observability key used by tests and archives to align UI
    state with phase timings.
    """
    update_visualization_state(
        visual_state,
        status=stage,
        pipeline_stage=stage,
        message=message or PIPELINE_STAGE_MESSAGES.get(stage, stage),
        **updates,
    )
