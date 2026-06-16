"""Speech-to-text transcription and STT prewarming."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from tools.voice_mode import is_whisper_hallucination, transcribe_recording

from ..daemon_config import LOG
from .nemotron_stt import is_nemotron_provider, prewarm_nemotron_streaming, transcribe_nemotron_streaming
from .parakeet_stt import is_parakeet_provider, prewarm_parakeet_streaming, transcribe_parakeet_streaming
from .wav import write_wav_int16


def transcribe_command(path: Path, cfg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    cfg = cfg or {}
    provider = str(cfg.get("stt_provider") or "hermes").strip() or "hermes"
    LOG.info("Transcribing command with STT provider %s: %s", provider, path)
    try:
        if is_parakeet_provider(provider):
            result = transcribe_parakeet_streaming(path, cfg)
        elif is_nemotron_provider(provider):
            result = transcribe_nemotron_streaming(path, cfg)
        else:
            result = transcribe_recording(str(path))
    except Exception as exc:
        LOG.exception("STT provider %s failed: %s", provider, exc)
        return None

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
    """Load the configured STT stack once so the first wake request is not delayed."""
    if not cfg.get("prewarm_stt_on_start", True):
        return
    provider = str(cfg.get("stt_provider") or "hermes").strip() or "hermes"
    try:
        if is_parakeet_provider(provider):
            LOG.info("Prewarming Parakeet Unified streaming STT")
            prewarm_parakeet_streaming(cfg)
            return
        if is_nemotron_provider(provider):
            LOG.info("Prewarming Nemotron streaming STT")
            prewarm_nemotron_streaming(cfg)
            return

        sample_rate = int(cfg["sample_rate"])
        silence = np.zeros(int(sample_rate * 0.25), dtype=np.int16)
        path = write_wav_int16(silence, sample_rate, prefix="wake_stt_prewarm")
        LOG.info("Prewarming Hermes STT with %s", path)
        result = transcribe_recording(str(path))
        LOG.info("STT prewarm result: %s", result)
    except Exception as exc:
        LOG.warning("STT prewarm failed; continuing: %s", exc)


__all__ = ["prewarm_stt", "transcribe_command"]
