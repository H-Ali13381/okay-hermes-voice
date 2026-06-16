"""Audio loading helpers for Parakeet streaming."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_mono_audio_samples(path: Path) -> Any:
    from nemo.collections.asr.parts.preprocessing.segment import get_samples  # type: ignore[import-not-found]

    return np.asarray(get_samples(str(path)), dtype=np.float32).reshape(-1)
