"""Public Nemotron STT facade for optional cache-aware streaming ASR."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from .nemotron_config import (
    NEMOTRON_EN_MODEL,
    PROVIDER_NAME,
    NemotronStreamingConfig,
    is_nemotron_provider,
    nemotron_config_from_daemon_config,
)
from .nemotron_provider import NemotronLiveStreamingSession, NemotronStreamingTranscriber

_TRANSCRIBER_CACHE: Dict[Tuple[Any, ...], NemotronStreamingTranscriber] = {}


def get_nemotron_transcriber(cfg: Dict[str, Any]) -> NemotronStreamingTranscriber:
    provider_cfg = nemotron_config_from_daemon_config(cfg)
    transcriber = _TRANSCRIBER_CACHE.get(provider_cfg.cache_key)
    if transcriber is None:
        transcriber = NemotronStreamingTranscriber(provider_cfg)
        _TRANSCRIBER_CACHE[provider_cfg.cache_key] = transcriber
    return transcriber


def transcribe_nemotron_streaming(path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Transcribe a captured WAV through Nemotron's cache-aware streaming path."""

    transcript = get_nemotron_transcriber(cfg).transcribe_file(path)
    return {"success": True, "transcript": transcript, "provider": PROVIDER_NAME, "streaming": True}


class _StartNemotronLiveStreaming:
    def __call__(self, cfg: Dict[str, Any]) -> NemotronLiveStreamingSession:
        return get_nemotron_transcriber(cfg).start_live_session()


start_nemotron_live_streaming = _StartNemotronLiveStreaming()


class _PrewarmNemotronStreaming:
    def __call__(self, cfg: Dict[str, Any]) -> None:
        get_nemotron_transcriber(cfg).load()


prewarm_nemotron_streaming = _PrewarmNemotronStreaming()

__all__ = [
    "NEMOTRON_EN_MODEL",
    "PROVIDER_NAME",
    "NemotronStreamingConfig",
    "NemotronStreamingTranscriber",
    "get_nemotron_transcriber",
    "is_nemotron_provider",
    "nemotron_config_from_daemon_config",
    "prewarm_nemotron_streaming",
    "start_nemotron_live_streaming",
    "transcribe_nemotron_streaming",
]
