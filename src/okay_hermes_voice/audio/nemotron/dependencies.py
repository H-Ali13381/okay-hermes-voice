"""Optional Nemotron dependency loading."""
from __future__ import annotations

from typing import Any, Tuple


def load_nemotron_dependencies() -> Tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        import nemo.collections.asr as nemo_asr  # type: ignore[import-not-found]
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional NeMo install.
        raise RuntimeError(
            "Nemotron streaming STT requires NVIDIA NeMo and PyTorch. Install it in the Hermes/okay-hermes "
            "environment, for example: pip install Cython packaging && "
            "pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'"
        ) from exc
    return torch, nemo_asr, CacheAwareStreamingAudioBuffer
