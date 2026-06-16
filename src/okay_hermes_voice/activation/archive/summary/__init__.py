"""Activation archive latency summary facade."""
from __future__ import annotations

from .formatting import format_activation_latency_summary
from .summarize import summarize_activation_archives

__all__ = ["format_activation_latency_summary", "summarize_activation_archives"]
