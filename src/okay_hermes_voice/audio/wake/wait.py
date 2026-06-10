"""Continuous wakeword inference loop."""
from __future__ import annotations

import collections
import contextlib
import queue
import time
from typing import Any, Deque, Dict, Optional

import numpy as np
import onnxruntime as ort

from ...daemon_config import LOG, STOP
from .inference import run_wake_inference


def wait_for_wake(cfg: Dict[str, Any], session: ort.InferenceSession, input_name: str, output_name: str) -> Optional[Dict[str, Any]]:
    sample_rate = int(cfg["sample_rate"])
    window_samples = int(float(cfg["window_seconds"]) * sample_rate)
    block_samples = int(float(cfg["block_seconds"]) * sample_rate)
    inference_interval = float(cfg["inference_interval_seconds"])
    threshold = float(cfg["threshold"])
    consecutive = max(1, int(cfg["trigger_consecutive_windows"]))

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
    rolling: Deque[float] = collections.deque(maxlen=window_samples)
    recent: Deque[float] = collections.deque(maxlen=consecutive)
    last_inference = 0.0

    import sounddevice as sd  # type: ignore[import-not-found]

    def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        if status:
            LOG.debug("Wake audio callback status: %s", status)
        block = np.asarray(indata[:, 0], dtype=np.float32).copy()
        with contextlib.suppress(queue.Full):
            audio_q.put_nowait(block)

    LOG.info("Listening for wakeword: threshold=%.6f consecutive=%d", threshold, consecutive)
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=block_samples, callback=callback):
        while not STOP.is_set():
            try:
                block = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

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
