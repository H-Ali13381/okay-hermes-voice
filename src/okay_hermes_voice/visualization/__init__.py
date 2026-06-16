"""Public visualization facade.

Boundary:
- daemon orchestration imports visualization state and launch helpers from here
- implementation mechanics live in semantic visualization submodules
- lazy exports keep popup-only imports from loading archive/audio dependencies
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "VISUALIZATION_LAUNCH_GRACE_SECONDS": ("launcher", "VISUALIZATION_LAUNCH_GRACE_SECONDS"),
    "append_visualization_turn": ("state", "append_visualization_turn"),
    "finish_cancelled_voice_session": ("state", "finish_cancelled_voice_session"),
    "is_visualization_cancel_requested": ("state", "is_visualization_cancel_requested"),
    "launch_visualization": ("launcher", "launch_visualization"),
    "read_visualization_state": ("state", "read_visualization_state"),
    "update_visualization_state": ("state", "update_visualization_state"),
    "visualization_cancel_reason": ("state", "visualization_cancel_reason"),
    "visualization_test": ("launcher", "visualization_test"),
}
_SUBMODULES = {"launcher", "state"}

__all__ = [
    "VISUALIZATION_LAUNCH_GRACE_SECONDS",
    "append_visualization_turn",
    "finish_cancelled_voice_session",
    "is_visualization_cancel_requested",
    "launcher",
    "launch_visualization",
    "read_visualization_state",
    "state",
    "update_visualization_state",
    "visualization_cancel_reason",
    "visualization_test",
]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(f"{__name__}.{module_name}")
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
