"""Cancellable Hermes subprocess timing config."""
from __future__ import annotations

from typing import Any, Dict


def _hermes_cancel_poll_seconds(cfg: Dict[str, Any]) -> float:
    return max(0.01, min(float(cfg.get("hermes_cancel_poll_seconds") or 0.1), 1.0))


def _hermes_interrupt_wait_seconds(cfg: Dict[str, Any]) -> float:
    return max(0.1, min(float(cfg.get("hermes_interrupt_wait_seconds") or 5.0), 60.0))


__all__ = ["_hermes_cancel_poll_seconds", "_hermes_interrupt_wait_seconds"]
