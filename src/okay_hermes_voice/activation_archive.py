"""Activation archive WAV and metadata persistence."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _round_metric(value: float) -> float:
    """Round metric values enough for stable JSON output without hiding useful deltas."""
    return round(float(value), 6)


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return _round_metric(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    interpolated = ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction
    return _round_metric(interpolated)


def _numeric_seconds(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0.0 else None


def _counter_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def _timing_stats(values: List[float]) -> Dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": _round_metric(ordered[0]),
        "mean": _round_metric(sum(ordered) / len(ordered)),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": _round_metric(ordered[-1]),
    }


def _new_summary_bucket() -> Dict[str, Any]:
    return {
        "archive_count": 0,
        "turn_count": 0,
        "status_counts": Counter(),
        "cancel_reason_counts": Counter(),
        "response_source_counts": Counter(),
        "route_target_counts": Counter(),
        "timing_values": defaultdict(list),
        "benchmark_category": "",
    }


def _add_seconds_fields(bucket: Dict[str, Any], timing: Any) -> None:
    if not isinstance(timing, dict):
        return
    for key, value in timing.items():
        if not str(key).endswith("_seconds"):
            continue
        numeric = _numeric_seconds(value)
        if numeric is not None:
            bucket["timing_values"][str(key)].append(numeric)


def _turn_timing_records(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = metadata.get("turns")
    records: List[Dict[str, Any]] = []
    if isinstance(turns, list):
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            timing = turn.get("timings")
            if isinstance(timing, dict):
                records.append({"turn": turn, "timing": timing})
    if records:
        return records

    turn_timings = metadata.get("turn_timings")
    if isinstance(turn_timings, list):
        for timing in turn_timings:
            if isinstance(timing, dict):
                records.append({"turn": {}, "timing": timing})
    if records:
        return records

    latest = metadata.get("latest_turn_timing")
    if isinstance(latest, dict):
        records.append({"turn": {}, "timing": latest})
    return records


def _add_archive_to_bucket(bucket: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    bucket["archive_count"] += 1
    status = str(metadata.get("status") or "unknown")
    bucket["status_counts"][status] += 1
    cancel_reason = metadata.get("cancel_reason")
    if cancel_reason:
        bucket["cancel_reason_counts"][str(cancel_reason)] += 1
    route_target = metadata.get("interaction_route_target")
    if route_target:
        bucket["route_target_counts"][str(route_target)] += 1

    category = metadata.get("benchmark_category")
    if category and not bucket["benchmark_category"]:
        bucket["benchmark_category"] = str(category)

    _add_seconds_fields(bucket, metadata.get("voice_session_timing"))

    for record in _turn_timing_records(metadata):
        bucket["turn_count"] += 1
        turn = record["turn"]
        timing = record["timing"]
        response_source = turn.get("response_source") or timing.get("response_source")
        if response_source:
            bucket["response_source_counts"][str(response_source)] += 1
        _add_seconds_fields(bucket, timing)


def _finalize_summary_bucket(bucket: Dict[str, Any], *, include_category: bool = False) -> Dict[str, Any]:
    finalized: Dict[str, Any] = {
        "archive_count": int(bucket["archive_count"]),
        "turn_count": int(bucket["turn_count"]),
        "status_counts": _counter_dict(bucket["status_counts"]),
        "cancel_reason_counts": _counter_dict(bucket["cancel_reason_counts"]),
        "response_source_counts": _counter_dict(bucket["response_source_counts"]),
        "route_target_counts": _counter_dict(bucket["route_target_counts"]),
        "timing_fields": {
            field: _timing_stats(values)
            for field, values in sorted(bucket["timing_values"].items())
            if values
        },
    }
    if include_category:
        finalized["benchmark_category"] = bucket["benchmark_category"]
    return finalized


def summarize_activation_archives(archive_dir: str | Path) -> Dict[str, Any]:
    """Summarize Phase 0 latency telemetry from activation archive metadata JSON files."""
    root = Path(archive_dir).expanduser()
    overall = _new_summary_bucket()
    by_preset: Dict[str, Dict[str, Any]] = {}
    invalid_json_count = 0

    for metadata_path in sorted(root.glob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            invalid_json_count += 1
            continue
        if not isinstance(metadata, dict):
            invalid_json_count += 1
            continue

        preset = str(metadata.get("benchmark_preset") or "uncategorized")
        bucket = by_preset.setdefault(preset, _new_summary_bucket())
        _add_archive_to_bucket(overall, metadata)
        _add_archive_to_bucket(bucket, metadata)

    summary = {
        "schema_version": 1,
        "archive_dir": str(root),
        "archive_count": int(overall["archive_count"]),
        "turn_count": int(overall["turn_count"]),
        "invalid_json_count": invalid_json_count,
        "status_counts": _counter_dict(overall["status_counts"]),
        "cancel_reason_counts": _counter_dict(overall["cancel_reason_counts"]),
        "response_source_counts": _counter_dict(overall["response_source_counts"]),
        "route_target_counts": _counter_dict(overall["route_target_counts"]),
        "timing_fields": {
            field: _timing_stats(values)
            for field, values in sorted(overall["timing_values"].items())
            if values
        },
        "by_preset": {
            preset: _finalize_summary_bucket(bucket, include_category=True)
            for preset, bucket in sorted(by_preset.items())
        },
    }
    return summary


def format_activation_latency_summary(summary: Dict[str, Any]) -> str:
    """Render an activation latency summary as compact terminal text."""
    lines = [
        "Activation archive latency summary",
        f"Archive dir: {summary.get('archive_dir', '')}",
        f"Archives: {summary.get('archive_count', 0)} | Turns: {summary.get('turn_count', 0)} | Invalid JSON: {summary.get('invalid_json_count', 0)}",
    ]
    status_counts = summary.get("status_counts") or {}
    if status_counts:
        lines.append("Statuses: " + ", ".join(f"{key}={value}" for key, value in status_counts.items()))

    timing_fields = summary.get("timing_fields") or {}
    preferred_fields = [
        "wake_to_handle_seconds",
        "wake_to_record_start_seconds",
        "record_seconds",
        "transcribe_seconds",
        "route_seconds",
        "answer_seconds",
        "tts_seconds",
        "playback_seconds",
        "speak_seconds",
        "turn_seconds",
    ]
    ordered_fields = [field for field in preferred_fields if field in timing_fields]
    ordered_fields.extend(field for field in sorted(timing_fields) if field not in ordered_fields)
    if ordered_fields:
        lines.append("")
        lines.append("Timing fields (seconds):")
        for field in ordered_fields:
            stats = timing_fields[field]
            lines.append(
                f"  {field}: count={stats['count']} mean={stats['mean']} p50={stats['p50']} p95={stats['p95']} max={stats['max']}"
            )

    by_preset = summary.get("by_preset") or {}
    if by_preset:
        lines.append("")
        lines.append("Presets:")
        for preset, bucket in by_preset.items():
            category = bucket.get("benchmark_category") or ""
            timing = (bucket.get("timing_fields") or {}).get("turn_seconds") or {}
            suffix = f" category={category}" if category else ""
            if timing:
                lines.append(
                    f"  {preset}: archives={bucket['archive_count']} turns={bucket['turn_count']} turn_mean={timing['mean']} turn_p95={timing['p95']}{suffix}"
                )
            else:
                lines.append(f"  {preset}: archives={bucket['archive_count']} turns={bucket['turn_count']}{suffix}")
    return "\n".join(lines) + "\n"
