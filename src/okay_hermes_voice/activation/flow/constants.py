"""Activation-flow status constants."""
from __future__ import annotations

VOICE_SESSION_COMPLETED = "completed"
VOICE_SESSION_CANCELLED = "cancelled"

PIPELINE_STAGE_MESSAGES = {
    "record": "record: recording your request now…",
    "transcript": "transcript: speech captured; converting it to text…",
    "route": "route: choosing the right handler for this request…",
    "answer": "answer: Hermes is producing the response…",
    "tts": "TTS: generating spoken audio…",
    "playback": "playback: playing the spoken response…",
}
