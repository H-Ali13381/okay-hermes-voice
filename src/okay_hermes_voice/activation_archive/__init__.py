"""Public activation archive facade.

Boundary:
- callers import stable archive operations from this package
- each archive operation lives in its own semantic leaf module
- latency summary implementation remains under activation.archive.summary
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "archive_command_audio": ".command_audio_archive",
    "command_audio_metadata_fields": ".command_audio_fields",
    "format_activation_latency_summary": "..activation.archive.summary",
    "save_activation_archive": ".wake_archive",
    "summarize_activation_archives": "..activation.archive.summary",
    "update_activation_archive_metadata": ".metadata_update",
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
