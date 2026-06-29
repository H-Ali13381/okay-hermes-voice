"""Optional audible cue playback."""
from __future__ import annotations

import contextlib
from typing import Any, Dict


def _play_beep(**kwargs: Any) -> None:
    from tools.voice_mode import play_beep
    play_beep(**kwargs)


def maybe_beep(cfg: Dict[str, Any], frequency: int = 880, count: int = 1) -> None:
    if not cfg.get("beep_enabled", True):
        return
    with contextlib.suppress(Exception):
        _play_beep(frequency=frequency, duration=0.10, count=count)


__all__ = ["maybe_beep"]
