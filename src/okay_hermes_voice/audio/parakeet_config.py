"""Parakeet provider config values."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

PARAKEET_UNIFIED_EN_MODEL = "nvidia/parakeet-unified-en-0.6b"
PROVIDER_NAME = "parakeet_unified_streaming"
_PROVIDER_ALIASES = {
    PROVIDER_NAME,
    "parakeet",
    "parakeet_unified",
    "parakeet-unified",
    "parakeet-streaming",
    "parakeet_unified_streaming",
}


@dataclass(frozen=True)
class ParakeetStreamingConfig:
    """Runtime options for NVIDIA Parakeet Unified English streaming ASR."""

    model_name: str = PARAKEET_UNIFIED_EN_MODEL
    model_path: str = ""
    device: str = "auto"
    left_context_secs: float = 2.0
    chunk_secs: float = 0.56
    right_context_secs: float = 0.56
    amp: bool = True
    amp_dtype: str = "float16"
    cudnn_enabled: bool = False

    @property
    def cache_key(self) -> Tuple[Any, ...]:
        return (
            self.model_path,
            self.model_name,
            self.device,
            self.left_context_secs,
            self.chunk_secs,
            self.right_context_secs,
            self.amp,
            self.amp_dtype,
            self.cudnn_enabled,
        )


def is_parakeet_provider(provider: Any) -> bool:
    return str(provider or "").strip().lower() in _PROVIDER_ALIASES


def parakeet_config_from_daemon_config(cfg: Dict[str, Any]) -> ParakeetStreamingConfig:
    return ParakeetStreamingConfig(
        model_name=str(cfg.get("parakeet_model_name") or PARAKEET_UNIFIED_EN_MODEL),
        model_path=str(cfg.get("parakeet_model_path") or ""),
        device=str(cfg.get("parakeet_device") or "auto"),
        left_context_secs=float(cfg.get("parakeet_left_context_secs", 2.0)),
        chunk_secs=float(cfg.get("parakeet_chunk_secs", 0.56)),
        right_context_secs=float(cfg.get("parakeet_right_context_secs", 0.56)),
        amp=bool(cfg.get("parakeet_amp", True)),
        amp_dtype=str(cfg.get("parakeet_amp_dtype") or "float16"),
        cudnn_enabled=bool(cfg.get("parakeet_cudnn_enabled", False)),
    )


__all__ = [
    "PARAKEET_UNIFIED_EN_MODEL",
    "PROVIDER_NAME",
    "ParakeetStreamingConfig",
    "is_parakeet_provider",
    "parakeet_config_from_daemon_config",
]
