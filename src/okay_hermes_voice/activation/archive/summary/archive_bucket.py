"""Add one archive metadata document to a summary bucket."""
from __future__ import annotations

from typing import Any, Dict

from .seconds_fields import add_seconds_fields
from .turn_records import turn_timing_records


def add_archive_to_bucket(bucket: Dict[str, Any], metadata: Dict[str, Any]) -> None:
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

    add_seconds_fields(bucket, metadata.get("voice_session_timing"))

    for record in turn_timing_records(metadata):
        bucket["turn_count"] += 1
        turn = record["turn"]
        timing = record["timing"]
        response_source = turn.get("response_source") or timing.get("response_source")
        if response_source:
            bucket["response_source_counts"][str(response_source)] += 1
        add_seconds_fields(bucket, timing)


__all__ = ["add_archive_to_bucket"]
