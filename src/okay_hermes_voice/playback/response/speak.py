"""Public spoken-response TTS generation pipeline."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Optional

from ...daemon_config import LOG
from .cancellation import _playback_cancel_requested


def speak_response(
    cfg: Dict[str, Any],
    text: str,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    stage_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    started = time.monotonic()
    timing: Dict[str, Any] = {
        "tts_enabled": bool(cfg.get("tts_enabled", True)),
        "tts_success": False,
        "playback_success": False,
        "tts_seconds": 0.0,
        "playback_seconds": 0.0,
        "speak_seconds": 0.0,
        "tts_file_path": "",
    }

    def finish() -> Dict[str, Any]:
        timing["speak_seconds"] = max(0.0, time.monotonic() - started)
        return timing

    def notify_stage(stage: str) -> None:
        if stage_callback is None:
            return
        try:
            stage_callback(stage)
        except Exception as exc:
            LOG.warning("Voice popup stage callback failed for %s: %s", stage, exc)

    if not timing["tts_enabled"]:
        LOG.info("TTS disabled; response not spoken")
        return finish()
    max_chars = int(cfg.get("max_spoken_response_chars") or 2500)
    spoken = text.strip()
    if len(spoken) > max_chars:
        spoken = spoken[:max_chars].rstrip() + "… The full response is in the wakeword log."
    LOG.info("Generating TTS (%d chars)", len(spoken))
    notify_stage("tts")
    tts_started = time.monotonic()
    from . import text_to_speech_tool, play_tts_file
    result_raw = text_to_speech_tool(spoken)
    timing["tts_seconds"] = max(0.0, time.monotonic() - tts_started)
    try:
        result = json.loads(result_raw)
    except Exception:
        LOG.error("TTS returned non-JSON: %r", result_raw)
        return finish()
    if not result.get("success"):
        LOG.error("TTS failed: %s", result.get("error") or result)
        return finish()
    timing["tts_success"] = True
    file_path = result.get("file_path")
    if not file_path:
        LOG.error("TTS response missing file_path: %s", result)
        return finish()
    timing["tts_file_path"] = str(file_path)
    LOG.info("Playing TTS: %s", file_path)
    notify_stage("playback")
    playback_started = time.monotonic()
    ok = play_tts_file(cfg, str(file_path), cancel_check=cancel_check)
    timing["playback_seconds"] = max(0.0, time.monotonic() - playback_started)
    timing["playback_success"] = bool(ok)
    if not ok and not _playback_cancel_requested(cancel_check):
        LOG.error("Audio playback failed: %s", file_path)
    return finish()


__all__ = ["speak_response"]
