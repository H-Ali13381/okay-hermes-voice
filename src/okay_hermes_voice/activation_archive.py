"""Activation archive WAV and metadata persistence."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .audio_io import float_waveform_to_int16, write_wav_int16_to_path
from .daemon_config import DEFAULT_CONFIG, LOG

def _activation_timestamp(detected_at: float) -> str:
    whole = time.strftime("%Y%m%d_%H%M%S", time.localtime(detected_at))
    millis = int((detected_at - int(detected_at)) * 1000)
    return f"{whole}_{millis:03d}"


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
        stem = f"activation_{_activation_timestamp(detected_at)}_p{probability:.3f}_{os.getpid()}_{time.monotonic_ns()}"
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


def update_activation_archive_metadata(archive: Optional[Dict[str, Any]], **updates: Any) -> None:
    """Merge metadata updates into the activation archive JSON."""
    if not archive:
        return
    try:
        raw_metadata_path = archive.get("metadata_path")
        if not raw_metadata_path:
            return
        metadata_path = Path(str(raw_metadata_path)).expanduser()
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with contextlib.suppress(Exception):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(updates)
        metadata["updated_at"] = time.time()
        metadata["updated_at_local"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(metadata["updated_at"]))
        tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, metadata_path)
    except Exception as exc:
        LOG.warning("Could not update activation archive metadata: %s", exc)


def archive_command_audio(cfg: Dict[str, Any], archive: Optional[Dict[str, Any]], command_path: Path, turn_index: int) -> Optional[str]:
    if not archive or not cfg.get("activation_save_command_audio", True):
        return None
    try:
        raw_metadata_path = archive.get("metadata_path")
        if not raw_metadata_path:
            return None
        metadata_path = Path(str(raw_metadata_path)).expanduser()
        stem = str(archive.get("stem") or metadata_path.stem)
        dest = metadata_path.with_name(f"{stem}_turn{turn_index:02d}_command.wav")
        shutil.copy2(str(command_path), str(dest))
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with contextlib.suppress(Exception):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths = metadata.get("command_wav_paths")
        if not isinstance(paths, list):
            paths = []
        paths.append(str(dest))
        update_activation_archive_metadata(archive, command_wav_paths=paths)
        return str(dest)
    except Exception as exc:
        LOG.warning("Could not archive command audio %s: %s", command_path, exc)
        return None


def command_audio_metadata_fields(archived_command_path: Optional[str], command_path: Path, latest: bool = False) -> Dict[str, str]:
    if archived_command_path:
        return {"latest_command_wav_path" if latest else "command_wav_path": archived_command_path}
    return {"temp_command_wav_path": str(command_path)}
