"""Public facade for audio input, wake inference, WAV, and STT helpers.

Boundary:
- callers import stable audio operations from this package
- device streams, wake inference, recording, transcription, and WAV mechanics
  live in semantic leaf modules
"""
from __future__ import annotations

from .devices import list_devices
from .recording import record_command
from .smoke import smoke_test
from .transcription import prewarm_stt, transcribe_command
from .wake import model_session, run_wake_inference, wait_for_wake
from .waveform import float_waveform_to_int16, rms_int16
from .wav import write_wav_int16, write_wav_int16_to_path

__all__ = [
    "float_waveform_to_int16",
    "list_devices",
    "model_session",
    "prewarm_stt",
    "record_command",
    "rms_int16",
    "run_wake_inference",
    "smoke_test",
    "transcribe_command",
    "wait_for_wake",
    "write_wav_int16",
    "write_wav_int16_to_path",
]
