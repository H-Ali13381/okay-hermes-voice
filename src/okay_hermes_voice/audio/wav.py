"""WAV persistence for captured wakeword and command audio."""
from __future__ import annotations

import tempfile
import time
import wave
from pathlib import Path

import numpy as np


def write_wav_int16(audio: np.ndarray, sample_rate: int, prefix: str = "wake_command") -> Path:
    out_dir = Path(tempfile.gettempdir()) / "hermes_voice_wakeword"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    write_wav_int16_to_path(audio, sample_rate, path)
    return path


def write_wav_int16_to_path(audio: np.ndarray, sample_rate: int, path: Path) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.int16).reshape(-1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return path


__all__ = ["write_wav_int16", "write_wav_int16_to_path"]
