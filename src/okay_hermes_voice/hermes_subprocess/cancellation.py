"""Cancellation polling for Hermes execution."""
from __future__ import annotations

from typing import Callable, Optional

from ..daemon_config import LOG, STOP


def _execution_cancel_requested(cancel_check: Optional[Callable[[], bool]]) -> bool:
    """Return True when the voice session has requested cancellation."""
    if STOP.is_set():
        return True
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception as exc:
        LOG.warning("Hermes execution cancel check failed: %s", exc)
        return False


__all__ = ["_execution_cancel_requested"]
