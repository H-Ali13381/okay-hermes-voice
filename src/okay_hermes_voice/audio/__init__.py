"""Public facade for audio input, wake inference, WAV, and STT helpers.

Boundary:
- callers import stable audio operations from this package
- device streams, wake inference, recording, transcription, and WAV mechanics
  live in semantic leaf modules
- heavy audio backends are imported only when their operation is requested
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CommandRecording": ".recording_result",
    "float_waveform_to_int16": ".waveform",
    "list_devices": ".devices",
    "model_session": ".wake",
    "prewarm_stt": ".transcription",
    "record_command": ".recording",
    "rms_int16": ".waveform",
    "run_wake_inference": ".wake",
    "smoke_test": ".smoke",
    "transcribe_command": ".transcription",
    "wait_for_wake": ".wake",
    "write_wav_int16": ".wav",
    "write_wav_int16_to_path": ".wav",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
