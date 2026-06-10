"""Summarize activation archive metadata files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .archive_bucket import add_archive_to_bucket
from .bucket import finalize_summary_bucket, new_summary_bucket
from .counts import counter_dict
from .timing_stats import timing_stats


def summarize_activation_archives(archive_dir: str | Path) -> Dict[str, Any]:
    """Summarize Phase 0 latency telemetry from activation archive metadata JSON files."""
    root = Path(archive_dir).expanduser()
    overall = new_summary_bucket()
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
        bucket = by_preset.setdefault(preset, new_summary_bucket())
        add_archive_to_bucket(overall, metadata)
        add_archive_to_bucket(bucket, metadata)

    return {
        "schema_version": 1,
        "archive_dir": str(root),
        "archive_count": int(overall["archive_count"]),
        "turn_count": int(overall["turn_count"]),
        "invalid_json_count": invalid_json_count,
        "status_counts": counter_dict(overall["status_counts"]),
        "cancel_reason_counts": counter_dict(overall["cancel_reason_counts"]),
        "response_source_counts": counter_dict(overall["response_source_counts"]),
        "route_target_counts": counter_dict(overall["route_target_counts"]),
        "timing_fields": {
            field: timing_stats(values)
            for field, values in sorted(overall["timing_values"].items())
            if values
        },
        "by_preset": {
            preset: finalize_summary_bucket(bucket, include_category=True)
            for preset, bucket in sorted(by_preset.items())
        },
    }


__all__ = ["summarize_activation_archives"]
