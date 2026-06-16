"""Timing field aggregate statistics."""
from __future__ import annotations

from typing import Any, Dict, List

from .metrics import percentile, round_metric


def timing_stats(values: List[float]) -> Dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round_metric(ordered[0]),
        "mean": round_metric(sum(ordered) / len(ordered)),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "max": round_metric(ordered[-1]),
    }


__all__ = ["timing_stats"]
