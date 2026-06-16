"""Compatibility facade for the audio subsystem.

Boundary:
- legacy callers may keep importing ``okay_hermes_voice.audio_io``
- implementation lives under ``okay_hermes_voice.audio`` leaf modules
- heavy audio backends are imported only when their operation is requested
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "float_waveform_to_int16": ".audio",
    "list_devices": ".audio",
    "model_session": ".audio",
    "prewarm_stt": ".audio",
    "record_command": ".audio",
    "rms_int16": ".audio",
    "run_wake_inference": ".audio",
    "smoke_test": ".audio",
    "transcribe_command": ".audio",
    "wait_for_wake": ".audio",
    "write_wav_int16": ".audio",
    "write_wav_int16_to_path": ".audio",
    "_cancel_check_requested": ".audio.recording",
}

__all__ = sorted(name for name in _EXPORTS if not name.startswith("_"))


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __package__)
    value = getattr(module, name)
    globals()[name] = value
    return value
