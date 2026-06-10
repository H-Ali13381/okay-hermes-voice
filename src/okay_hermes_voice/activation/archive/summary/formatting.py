"""Terminal formatting for activation archive latency summaries."""
from __future__ import annotations

from typing import Any, Dict


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


__all__ = ["format_activation_latency_summary"]
