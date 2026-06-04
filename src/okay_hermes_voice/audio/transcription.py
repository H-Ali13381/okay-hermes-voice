"""Speech-to-text transcription and STT prewarming."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from tools.voice_mode import is_whisper_hallucination, transcribe_recording

from ..daemon_config import LOG
from .wav import write_wav_int16


def transcribe_command(path: Path) -> Optional[str]:
    LOG.info("Transcribing command: %s", path)
    result = transcribe_recording(str(path))
    if not result.get("success"):
        LOG.error("STT failed: %s", result.get("error") or result)
        return None
    transcript = (result.get("transcript") or "").strip()
    if is_whisper_hallucination(transcript):
        LOG.info("Filtered empty/hallucinated transcript: %r", transcript)
        return None
    LOG.info("Transcript: %s", transcript)
    return transcript


def prewarm_stt(cfg: Dict[str, Any]) -> None:
    """Load Hermes' STT stack once so the first real wake request is not delayed."""
    if not cfg.get("prewarm_stt_on_start", True):
        return
    try:
        sample_rate = int(cfg["sample_rate"])
        silence = np.zeros(int(sample_rate * 0.25), dtype=np.int16)
        path = write_wav_int16(silence, sample_rate, prefix="wake_stt_prewarm")
        LOG.info("Prewarming STT with %s", path)
        result = transcribe_recording(str(path))
        LOG.info("STT prewarm result: %s", result)
    except Exception as exc:
        LOG.warning("STT prewarm failed; continuing: %s", exc)


__all__ = ["prewarm_stt", "transcribe_command"]
