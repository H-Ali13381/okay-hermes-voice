"""CLI smoke-test path for wakeword model and zero-audio inference."""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np

from ..daemon_config import setup_logging
from .wake import model_session, run_wake_inference


def smoke_test(cfg: Dict[str, Any]) -> int:
    setup_logging(cfg, verbose=True)
    session, input_name, output_name = model_session(cfg["model_path"])
    sample_rate = int(cfg["sample_rate"])
    window_samples = int(float(cfg["window_seconds"]) * sample_rate)
    zeros = np.zeros(window_samples, dtype=np.float32)
    prob = run_wake_inference(session, input_name, output_name, zeros)
    print(json.dumps({
        "ok": True,
        "model_path": cfg["model_path"],
        "zero_audio_probability": prob,
        "threshold": float(cfg["threshold"]),
        "input_name": input_name,
        "output_name": output_name,
    }, indent=2))
    return 0


__all__ = ["smoke_test"]
