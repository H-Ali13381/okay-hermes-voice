"""Public Parakeet STT facade for optional streaming ASR."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from .parakeet_config import (
    PARAKEET_UNIFIED_EN_MODEL,
    PROVIDER_NAME,
    ParakeetStreamingConfig,
    is_parakeet_provider,
    parakeet_config_from_daemon_config,
)
from .parakeet_provider import ParakeetLiveStreamingSession, ParakeetStreamingTranscriber

_TRANSCRIBER_CACHE: Dict[Tuple[Any, ...], ParakeetStreamingTranscriber] = {}


def get_parakeet_transcriber(cfg: Dict[str, Any]) -> ParakeetStreamingTranscriber:
    provider_cfg = parakeet_config_from_daemon_config(cfg)
    transcriber = _TRANSCRIBER_CACHE.get(provider_cfg.cache_key)
    if transcriber is None:
        transcriber = ParakeetStreamingTranscriber(provider_cfg)
        _TRANSCRIBER_CACHE[provider_cfg.cache_key] = transcriber
    return transcriber


def transcribe_parakeet_streaming(path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    transcript = get_parakeet_transcriber(cfg).transcribe_file(path)
    return {"success": True, "transcript": transcript, "provider": PROVIDER_NAME, "streaming": True}


class _StartParakeetLiveStreaming:
    def __call__(self, cfg: Dict[str, Any]) -> ParakeetLiveStreamingSession:
        return get_parakeet_transcriber(cfg).start_live_session()


class _PrewarmParakeetStreaming:
    def __call__(self, cfg: Dict[str, Any]) -> None:
        get_parakeet_transcriber(cfg).load()


start_parakeet_live_streaming = _StartParakeetLiveStreaming()
prewarm_parakeet_streaming = _PrewarmParakeetStreaming()

__all__ = [
    "PARAKEET_UNIFIED_EN_MODEL",
    "PROVIDER_NAME",
    "ParakeetStreamingConfig",
    "ParakeetStreamingTranscriber",
    "get_parakeet_transcriber",
    "is_parakeet_provider",
    "parakeet_config_from_daemon_config",
    "prewarm_parakeet_streaming",
    "start_parakeet_live_streaming",
    "transcribe_parakeet_streaming",
]
