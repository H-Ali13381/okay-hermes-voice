"""Metric rounding policy."""
from __future__ import annotations


def round_metric(value: float) -> float:
    """Round metric values enough for stable JSON output without hiding useful deltas."""
    return round(float(value), 6)


__all__ = ["round_metric"]
