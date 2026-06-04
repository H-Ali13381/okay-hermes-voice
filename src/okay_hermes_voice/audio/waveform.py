"""Waveform conversion and level measurement for wakeword audio."""
from __future__ import annotations

import math

import numpy as np


def rms_int16(block: np.ndarray) -> float:
    arr = block.astype(np.float32, copy=False).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(arr * arr))))


def float_waveform_to_int16(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    clipped = np.clip(waveform, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


__all__ = ["float_waveform_to_int16", "rms_int16"]
