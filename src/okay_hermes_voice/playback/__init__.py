"""Public playback facade.

Boundary:
- callers import stable TTS/playback helpers from here
- response generation and audio process mechanics live in playback/response.py
- keep this file as an import map, not a second playback implementation
"""
from __future__ import annotations

from . import response
from .response import (
    maybe_beep,
    play_tts_file,
    speak_response,
)

__all__ = [
    "maybe_beep",
    "play_tts_file",
    "response",
    "speak_response",
]
