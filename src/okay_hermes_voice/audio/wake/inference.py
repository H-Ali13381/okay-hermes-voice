"""Single-window wakeword inference."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort


def run_wake_inference(session: ort.InferenceSession, input_name: str, output_name: str, waveform: np.ndarray) -> float:
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32, copy=False)
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    probability = session.run([output_name], {input_name: waveform[None, :]})[0][0]
    return float(probability)


__all__ = ["run_wake_inference"]
