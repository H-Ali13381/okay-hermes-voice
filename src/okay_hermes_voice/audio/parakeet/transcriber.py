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
            self._ensure_streaming_decoding_strategy(model)
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
        self.load()
        if not self._supports_live_decoding():
            return self._transcribe_file_batch(path)
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

    def _supports_live_decoding(self) -> bool:
        return self._model_has_live_decoding(self._model)

    @staticmethod
    def _model_has_live_decoding(model: Any) -> bool:
        decoding = getattr(getattr(model, "decoding", None), "decoding", None)
        return hasattr(decoding, "decoding_computer")

    def _ensure_streaming_decoding_strategy(self, model: Any) -> None:
        if self._model_has_live_decoding(model):
            return
        decoding_cfg = getattr(getattr(model, "cfg", None), "decoding", None)
        if decoding_cfg is None or not hasattr(model, "change_decoding_strategy"):
            return
        model_type = str(getattr(decoding_cfg, "model_type", "")).lower()
        if model_type != "tdt":
            return
        from omegaconf import open_dict  # type: ignore[import-not-found]

        streaming_cfg = decoding_cfg.copy()
        with open_dict(streaming_cfg):
            streaming_cfg.strategy = "greedy_batch"
            streaming_cfg.fused_batch_size = -1
            streaming_cfg.tdt_include_token_duration = False
            streaming_cfg.greedy.loop_labels = True
            streaming_cfg.greedy.preserve_alignments = False
        LOG.info("Switching Parakeet TDT decoding to greedy_batch label-looping for live streaming")
        model.change_decoding_strategy(streaming_cfg)

    def _transcribe_file_batch(self, path: Path) -> str:
        assert self._model is not None
        LOG.info("Parakeet model does not expose live decoding; using batch transcription for %s", path)
        output = self._model.transcribe([str(path)], batch_size=1, return_hypotheses=False)
        first = output[0] if isinstance(output, (list, tuple)) else output
        if hasattr(first, "text"):
            return str(first.text).strip()
        if isinstance(first, dict) and "text" in first:
            return str(first["text"]).strip()
        return str(first).strip()

    @staticmethod
    def _resolve_amp_dtype(torch: Any, name: str) -> Any:
        return torch.bfloat16 if str(name).lower() == "bfloat16" else torch.float16
