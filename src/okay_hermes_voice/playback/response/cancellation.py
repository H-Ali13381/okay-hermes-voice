"""Playback cancellation checks."""
from __future__ import annotations

from typing import Callable, Optional

from ...daemon_config import LOG, STOP


def _playback_cancel_requested(cancel_check: Optional[Callable[[], bool]]) -> bool:
    if STOP.is_set():
        return True
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception as exc:
        LOG.warning("Playback cancel check failed: %s", exc)
        return False


__all__ = ["_playback_cancel_requested"]
