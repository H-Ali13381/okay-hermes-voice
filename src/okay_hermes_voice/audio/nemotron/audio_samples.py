"""Audio loading helpers for Nemotron streaming."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_mono_audio_samples(path: Path) -> Any:
    from nemo.collections.asr.parts.preprocessing.segment import get_samples  # type: ignore[import-not-found]

    audio = np.asarray(get_samples(str(Path(path).expanduser())))
    if audio.ndim == 2:
        # NeMo's get_samples() transposes multichannel files to (channels, time),
        # but CacheAwareStreamingAudioBuffer.preprocess_audio expects (time,).
        channel_axis = 0 if audio.shape[0] <= audio.shape[1] else 1
        audio = audio.mean(axis=channel_axis)
    else:
        audio = audio.reshape(-1)
    return audio.astype("float32", copy=False)
