"""Hermes voice-mode playback fallback."""
from __future__ import annotations

import contextlib
import threading
from typing import Any, Callable, Dict, Optional

from ...daemon_config import LOG
from .cancellation import _playback_cancel_requested


def _play_hermes_audio_file_with_cancel(file_path: str, cancel_check: Optional[Callable[[], bool]]) -> bool:
    from tools.voice_mode import play_audio_file, stop_playback

    if cancel_check is None:
        return bool(play_audio_file(str(file_path)))

    done = threading.Event()
    result: Dict[str, Any] = {"ok": False}

    def _run() -> None:
        try:
            result["ok"] = bool(play_audio_file(str(file_path)))
        except Exception as exc:
            result["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    while not done.wait(0.05):
        if _playback_cancel_requested(cancel_check):
            with contextlib.suppress(Exception):
                stop_playback()
            done.wait(2.0)
            LOG.info("Hermes fallback playback cancelled")
            return False
    if result.get("error"):
        LOG.warning("Hermes fallback playback failed: %s", result["error"])
        return False
    return bool(result.get("ok"))


__all__ = ["_play_hermes_audio_file_with_cancel"]
