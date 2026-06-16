"""Nemotron provider config values."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

NEMOTRON_EN_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
PROVIDER_NAME = "nemotron_en_streaming"
_PROVIDER_ALIASES = {PROVIDER_NAME, "nemotron", "nemotron_en", "nemotron-streaming", "nemotron_en_streaming"}


@dataclass(frozen=True)
class NemotronStreamingConfig:
    """Runtime options for the English-only Nemotron streaming provider."""

    model_name: str = NEMOTRON_EN_MODEL
    model_path: str = ""
    device: str = "auto"
    att_context_size: Tuple[int, int] = (70, 13)
    amp: bool = True
    amp_dtype: str = "float16"
    cudnn_enabled: bool = False
    online_normalization: bool = False
    pad_and_drop_preencoded: bool = False

    @property
    def cache_key(self) -> Tuple[Any, ...]:
        return (
            self.model_path,
            self.model_name,
            self.device,
            self.att_context_size,
            self.amp,
            self.amp_dtype,
            self.cudnn_enabled,
            self.online_normalization,
            self.pad_and_drop_preencoded,
        )


def is_nemotron_provider(provider: Any) -> bool:
    return str(provider or "").strip().lower() in _PROVIDER_ALIASES


def nemotron_config_from_daemon_config(cfg: Dict[str, Any]) -> NemotronStreamingConfig:
    raw_context = cfg.get("nemotron_att_context_size", (70, 13))
    if isinstance(raw_context, str):
        parts = [part.strip() for part in raw_context.strip("[] ").split(",") if part.strip()]
        context = tuple(int(part) for part in parts)
    else:
        context = tuple(int(part) for part in raw_context)
    if len(context) != 2:
        raise ValueError("nemotron_att_context_size must contain exactly two integers, e.g. [70, 13]")

    return NemotronStreamingConfig(
        model_name=str(cfg.get("nemotron_model_name") or NEMOTRON_EN_MODEL),
        model_path=str(cfg.get("nemotron_model_path") or ""),
        device=str(cfg.get("nemotron_device") or "auto"),
        att_context_size=(context[0], context[1]),
        amp=bool(cfg.get("nemotron_amp", True)),
        amp_dtype=str(cfg.get("nemotron_amp_dtype") or "float16"),
        cudnn_enabled=bool(cfg.get("nemotron_cudnn_enabled", False)),
        online_normalization=bool(cfg.get("nemotron_online_normalization", False)),
        pad_and_drop_preencoded=bool(cfg.get("nemotron_pad_and_drop_preencoded", False)),
    )


__all__ = [
    "NEMOTRON_EN_MODEL",
    "PROVIDER_NAME",
    "NemotronStreamingConfig",
    "is_nemotron_provider",
    "nemotron_config_from_daemon_config",
]
