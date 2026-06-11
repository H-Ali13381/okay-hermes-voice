"""Configured wakeword audio capture backend selection."""
from __future__ import annotations

from typing import Any, Dict, Iterator

import numpy as np

from .arecord import iter_arecord_blocks
from .sounddevice import iter_sounddevice_blocks


def iter_wake_audio_blocks(cfg: Dict[str, Any], *, sample_rate: int, block_samples: int) -> Iterator[np.ndarray]:
    """Yield wakeword audio blocks using the configured capture backend."""
    backend = str(cfg.get("wake_audio_backend") or "sounddevice").strip().lower()
    if backend == "arecord":
        yield from iter_arecord_blocks(
            sample_rate=sample_rate,
            block_samples=block_samples,
            device=str(cfg.get("wake_audio_device") or "default"),
        )
        return
    if backend != "sounddevice":
        raise ValueError(f"Unsupported wake_audio_backend: {backend}")
    yield from iter_sounddevice_blocks(sample_rate=sample_rate, block_samples=block_samples)
