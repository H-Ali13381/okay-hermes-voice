"""Continuous wakeword inference loop."""
from __future__ import annotations

import collections
import time
from typing import Any, Deque, Dict, Optional

import numpy as np

from ...daemon_config import LOG, STOP
from .capture import iter_wake_audio_blocks
from .inference import run_wake_inference


def wait_for_wake(cfg: Dict[str, Any], session: Any, input_name: str, output_name: str) -> Optional[Dict[str, Any]]:
    sample_rate = int(cfg["sample_rate"])
    window_samples = int(float(cfg["window_seconds"]) * sample_rate)
    block_samples = int(float(cfg["block_seconds"]) * sample_rate)
    inference_interval = float(cfg["inference_interval_seconds"])
    threshold = float(cfg["threshold"])
    consecutive = max(1, int(cfg["trigger_consecutive_windows"]))

    rolling: Deque[float] = collections.deque(maxlen=window_samples)
    recent: Deque[float] = collections.deque(maxlen=consecutive)
    last_inference = 0.0

    backend = str(cfg.get("wake_audio_backend") or "sounddevice")
    LOG.info("Listening for wakeword: threshold=%.6f consecutive=%d backend=%s", threshold, consecutive, backend)
    for block in iter_wake_audio_blocks(cfg, sample_rate=sample_rate, block_samples=block_samples):
        if STOP.is_set():
            break

        rolling.extend(float(x) for x in block)
        if len(rolling) < window_samples:
            continue

        now = time.monotonic()
        if now - last_inference < inference_interval:
            continue
        last_inference = now

        waveform = np.fromiter(rolling, dtype=np.float32, count=window_samples)
        probability = run_wake_inference(session, input_name, output_name, waveform)
        recent.append(probability)
        if probability >= threshold * 0.75:
            LOG.debug("Wake score %.6f", probability)
        if len(recent) == consecutive and all(score >= threshold for score in recent):
            scores = [float(score) for score in recent]
            LOG.info("Wakeword detected: scores=%s", [round(s, 6) for s in scores])
            return {
                "probability": max(scores),
                "scores": scores,
                "waveform": waveform.copy(),
                "sample_rate": sample_rate,
                "detected_at": time.time(),
            }
    return None


__all__ = ["wait_for_wake"]
