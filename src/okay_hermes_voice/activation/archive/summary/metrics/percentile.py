"""Percentile interpolation for summary timing fields."""
from __future__ import annotations

from typing import List

from .rounding import round_metric


def percentile(values: List[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round_metric(ordered[0])
    position = (len(ordered) - 1) * percentile_value
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    interpolated = ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction
    return round_metric(interpolated)


__all__ = ["percentile"]
