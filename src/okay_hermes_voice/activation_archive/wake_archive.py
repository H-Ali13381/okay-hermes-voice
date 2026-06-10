"""Persist the wakeword audio window that triggered an activation."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..audio.waveform import float_waveform_to_int16
from ..audio.wav import write_wav_int16_to_path
from ..daemon_config import DEFAULT_CONFIG, LOG


def save_activation_archive(cfg: Dict[str, Any], activation: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Save the triggering wakeword audio window and metadata for review."""
    if not cfg.get("save_activation_audio", True):
        return None
    try:
        waveform = activation.get("waveform")
        if waveform is None:
            LOG.warning("Activation archive requested but activation has no waveform")
            return None
        sample_rate = int(activation.get("sample_rate") or cfg.get("sample_rate") or 16000)
        probability = float(activation.get("probability") or 0.0)
        detected_at = float(activation.get("detected_at") or time.time())
        scores = [float(score) for score in (activation.get("scores") or [probability])]
        out_dir = Path(str(cfg.get("activation_archive_dir") or DEFAULT_CONFIG["activation_archive_dir"])).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        whole = time.strftime("%Y%m%d_%H%M%S", time.localtime(detected_at))
        millis = int((detected_at - int(detected_at)) * 1000)
        stem = f"activation_{whole}_{millis:03d}_p{probability:.3f}_{os.getpid()}_{time.monotonic_ns()}"
        wav_path = out_dir / f"{stem}.wav"
        metadata_path = out_dir / f"{stem}.json"
        audio = float_waveform_to_int16(np.asarray(waveform, dtype=np.float32))
        write_wav_int16_to_path(audio, sample_rate, wav_path)
        metadata: Dict[str, Any] = {
            "status": "wake_detected",
            "detected_at": detected_at,
            "detected_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(detected_at)),
            "probability": probability,
            "scores": scores,
            "threshold": float(cfg.get("threshold") or 0.0),
            "trigger_consecutive_windows": int(cfg.get("trigger_consecutive_windows") or len(scores) or 1),
            "sample_rate": sample_rate,
            "window_seconds": float(len(audio) / sample_rate) if sample_rate else 0.0,
            "wake_wav_path": str(wav_path),
            "metadata_path": str(metadata_path),
            "model_path": str(cfg.get("model_path") or ""),
            "command_wav_paths": [],
            "turns": [],
        }
        tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, metadata_path)
        LOG.info("Saved activation archive: %s", wav_path)
        return {"wake_wav_path": str(wav_path), "metadata_path": str(metadata_path), "stem": stem}
    except Exception as exc:
        LOG.warning("Could not save activation archive: %s", exc)
        return None


__all__ = ["save_activation_archive"]
