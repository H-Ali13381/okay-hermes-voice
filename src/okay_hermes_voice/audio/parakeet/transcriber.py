"""Parakeet streaming transcriber lifecycle."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ...daemon_config import LOG
from ..parakeet_config import ParakeetStreamingConfig
from .audio_samples import load_mono_audio_samples
from .dependencies import load_parakeet_dependencies
from .session import ParakeetLiveStreamingSession


class ParakeetStreamingTranscriber:
    """Load Parakeet Unified and expose live chunked streaming sessions."""

    def __init__(self, cfg: ParakeetStreamingConfig):
        self.cfg = cfg
        self._loaded = False
        self._torch = None
        self._nemo_asr = None
        self._stream_buffer_cls = None
        self._context_size_cls = None
        self._batched_hyps_to_hypotheses = None
        self._model = None
        self._device = None
        self._amp_dtype = None

    def load(self) -> None:
        if self._loaded:
            return
        torch, nemo_asr, stream_buffer_cls, context_size_cls, batched_hyps_to_hypotheses = load_parakeet_dependencies()
        device_name = self.cfg.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        torch.backends.cudnn.enabled = bool(self.cfg.cudnn_enabled)
        LOG.info("Loading Parakeet Unified streaming STT model: %s on %s", self.cfg.model_path or self.cfg.model_name, device)

        if self.cfg.model_path:
            model = nemo_asr.models.ASRModel.restore_from(restore_path=self.cfg.model_path, map_location=device)
        else:
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.cfg.model_name, map_location=device)
        if hasattr(model, "freeze"):
            model.freeze()
        model = model.to(device=device)
        model.eval()
        if hasattr(model, "change_decoding_strategy") and hasattr(model, "cfg") and hasattr(model.cfg, "decoding"):
            model.change_decoding_strategy(model.cfg.decoding)
        if hasattr(model, "preprocessor") and hasattr(model.preprocessor, "featurizer"):
            model.preprocessor.featurizer.dither = 0.0
            model.preprocessor.featurizer.pad_to = 0

        self._torch = torch
        self._nemo_asr = nemo_asr
        self._stream_buffer_cls = stream_buffer_cls
        self._context_size_cls = context_size_cls
        self._batched_hyps_to_hypotheses = batched_hyps_to_hypotheses
        self._model = model
        self._device = device
        self._amp_dtype = self._resolve_amp_dtype(torch, self.cfg.amp_dtype)
        self._loaded = True

    def transcribe_file(self, path: Path) -> str:
        session = self.start_live_session()
        audio = load_mono_audio_samples(path)
        chunk_samples = max(1, int(session.sample_rate * self.cfg.chunk_secs))
        for offset in range(0, len(audio), chunk_samples):
            session.accept_audio(audio[offset : offset + chunk_samples])
        return session.finalize()

    def start_live_session(self) -> ParakeetLiveStreamingSession:
        self.load()
        assert self._torch is not None
        assert self._model is not None
        assert self._device is not None
        assert self._stream_buffer_cls is not None
        assert self._context_size_cls is not None
        return ParakeetLiveStreamingSession(self)

    @property
    def amp_context(self) -> Any:
        assert self._torch is not None
        assert self._device is not None
        amp_device = "cuda" if getattr(self._device, "type", str(self._device)) == "cuda" else "cpu"
        return self._torch.amp.autocast(amp_device, dtype=self._amp_dtype, enabled=bool(self.cfg.amp))

    @staticmethod
    def _resolve_amp_dtype(torch: Any, name: str) -> Any:
        return torch.bfloat16 if str(name).lower() == "bfloat16" else torch.float16
