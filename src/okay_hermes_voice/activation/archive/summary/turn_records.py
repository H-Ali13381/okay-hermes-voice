"""Extract turn timing records from archive metadata."""
from __future__ import annotations

from typing import Any, Dict, List


def turn_timing_records(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
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


__all__ = ["turn_timing_records"]
