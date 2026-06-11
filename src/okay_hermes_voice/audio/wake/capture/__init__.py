"""Wakeword audio block capture backends."""
from __future__ import annotations

from .arecord import iter_arecord_blocks
from .blocks import iter_wake_audio_blocks
from .sounddevice import iter_sounddevice_blocks

__all__ = ["iter_arecord_blocks", "iter_sounddevice_blocks", "iter_wake_audio_blocks"]
