"""Wakeword model loading and inference facade."""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "model_session": ".session",
    "run_wake_inference": ".inference",
    "wait_for_wake": ".wait",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
