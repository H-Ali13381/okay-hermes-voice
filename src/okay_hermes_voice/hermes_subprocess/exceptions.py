"""Internal Hermes runtime fall-through sentinels."""
from __future__ import annotations


class _UseSubprocessForCancellation(RuntimeError):
    """Internal sentinel for falling through to process-group cancellable Hermes."""


__all__ = ["_UseSubprocessForCancellation"]
