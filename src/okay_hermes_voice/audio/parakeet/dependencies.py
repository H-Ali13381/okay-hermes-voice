"""Optional Parakeet dependency loading."""
from __future__ import annotations

from typing import Any, Tuple


def load_parakeet_dependencies() -> Tuple[Any, Any, Any, Any, Any]:
    import torch  # type: ignore[import-not-found]
    import nemo.collections.asr as nemo_asr  # type: ignore[import-not-found]
    from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses  # type: ignore[import-not-found]
    from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer  # type: ignore[import-not-found]

    return torch, nemo_asr, StreamingBatchedAudioBuffer, ContextSize, batched_hyps_to_hypotheses
