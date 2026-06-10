"""Summary bucket lifecycle for activation archives."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict

from .counts import counter_dict
from .timing_stats import timing_stats


def new_summary_bucket() -> Dict[str, Any]:
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


def finalize_summary_bucket(bucket: Dict[str, Any], *, include_category: bool = False) -> Dict[str, Any]:
    finalized: Dict[str, Any] = {
        "archive_count": int(bucket["archive_count"]),
        "turn_count": int(bucket["turn_count"]),
        "status_counts": counter_dict(bucket["status_counts"]),
        "cancel_reason_counts": counter_dict(bucket["cancel_reason_counts"]),
        "response_source_counts": counter_dict(bucket["response_source_counts"]),
        "route_target_counts": counter_dict(bucket["route_target_counts"]),
        "timing_fields": {
            field: timing_stats(values)
            for field, values in sorted(bucket["timing_values"].items())
            if values
        },
    }
    if include_category:
        finalized["benchmark_category"] = bucket["benchmark_category"]
    return finalized


__all__ = ["finalize_summary_bucket", "new_summary_bucket"]
