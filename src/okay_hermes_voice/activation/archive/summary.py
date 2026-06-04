"""Activation archive latency summary and terminal formatting helpers."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    cancel_reason = metadata.get("cancel_reason") or metadata.get("close_reason")
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

    cancel_reason_counts = summary.get("cancel_reason_counts") or {}
    if cancel_reason_counts:
        lines.append("Cancel reasons: " + ", ".join(f"{key}={value}" for key, value in cancel_reason_counts.items()))

    response_source_counts = summary.get("response_source_counts") or {}
    if response_source_counts:
        lines.append("Response sources: " + ", ".join(f"{key}={value}" for key, value in response_source_counts.items()))

    route_target_counts = summary.get("route_target_counts") or {}
    if route_target_counts:
        lines.append("Route targets: " + ", ".join(f"{key}={value}" for key, value in route_target_counts.items()))

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
