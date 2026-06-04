"""Compatibility facade for the audio subsystem.

Boundary:
- legacy callers may keep importing ``okay_hermes_voice.audio_io``
- implementation lives under ``okay_hermes_voice.audio`` leaf modules
"""
from __future__ import annotations

from .audio import (
    float_waveform_to_int16,
    list_devices,
    model_session,
    prewarm_stt,
    record_command,
    rms_int16,
    run_wake_inference,
    smoke_test,
    transcribe_command,
    wait_for_wake,
    write_wav_int16,
    write_wav_int16_to_path,
)
from .audio.recording import _cancel_check_requested

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
