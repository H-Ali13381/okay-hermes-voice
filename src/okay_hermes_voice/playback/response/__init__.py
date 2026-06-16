"""TTS generation and cancellable audio playback facade."""
from __future__ import annotations

import subprocess as subprocess
import time as time

from tools.tts_tool import text_to_speech_tool

from .beep import maybe_beep
from .cancellation import _playback_cancel_requested
from .fallback import _play_hermes_audio_file_with_cancel
from .output import _collect_playback_output
from .play_file import play_tts_file
from .sinks import _configured_playback_sinks
from .speak import speak_response
from .termination import _terminate_playback_process
from .wait_process import _wait_playback_process
from .wait_processes import _wait_playback_processes

__all__ = [
    "maybe_beep",
    "play_tts_file",
    "speak_response",
]
