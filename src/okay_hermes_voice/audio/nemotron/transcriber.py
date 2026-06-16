"""Nemotron streaming transcriber lifecycle."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

from ...daemon_config import LOG
from ..nemotron_config import NemotronStreamingConfig
from .audio_samples import load_mono_audio_samples
from .dependencies import load_nemotron_dependencies
from .session import NemotronLiveStreamingSession


class NemotronStreamingTranscriber:
    """Lazy in-process NeMo transcriber using cache-aware streaming chunks."""

    def __init__(self, cfg: NemotronStreamingConfig):
        self.cfg = cfg
        self._loaded = False
        self._torch = None
        self._buffer_cls = None
        self._model = None
        self._device = None
        self._amp_dtype = None

    def load(self) -> None:
        if self._loaded:
            return

        torch, nemo_asr, buffer_cls = load_nemotron_dependencies()
        torch.backends.cudnn.enabled = bool(self.cfg.cudnn_enabled)
        device = self._resolve_torch_device(torch, self.cfg.device)
        model_ref = self.cfg.model_path or self.cfg.model_name
        LOG.info("Loading Nemotron streaming STT model: %s on %s", model_ref, device)

        if self.cfg.model_path:
            model = nemo_asr.models.ASRModel.restore_from(
                restore_path=str(Path(self.cfg.model_path).expanduser()),
                map_location=device,
            )
        else:
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.cfg.model_name,
                map_location=device,
            )

        self._configure_model(model)
        model = model.to(device=device)
        model.eval()

        self._torch = torch
        self._buffer_cls = buffer_cls
        self._model = model
        self._device = device
        self._amp_dtype = self._resolve_amp_dtype(torch, self.cfg.amp_dtype)
        self._loaded = True

    def transcribe_file(self, path: Path) -> str:
        self.load()
        assert self._torch is not None
        assert self._buffer_cls is not None
        assert self._model is not None
        assert self._device is not None

        session = self.start_live_session()
        self._append_audio_file(session.buffer, path)
        transcript = session.finalize()
        LOG.info("Nemotron streaming transcript: %s", transcript)
        return transcript

    def start_live_session(self) -> NemotronLiveStreamingSession:
        """Create an incremental Nemotron session that can consume mic blocks live."""
        self.load()
        assert self._torch is not None
        assert self._buffer_cls is not None
        assert self._model is not None
        assert self._device is not None

        buffer = self._buffer_cls(
            model=self._model,
            online_normalization=self.cfg.online_normalization,
            pad_and_drop_preencoded=self.cfg.pad_and_drop_preencoded,
        )
        return NemotronLiveStreamingSession(self, buffer)

    @property
    def amp_context(self) -> Any:
        assert self._torch is not None
        assert self._device is not None
        amp_device = "cuda" if getattr(self._device, "type", str(self._device)) == "cuda" else "cpu"
        return self._torch.amp.autocast(amp_device, dtype=self._amp_dtype, enabled=bool(self.cfg.amp))

    @staticmethod
    def _append_audio_file(buffer: Any, path: Path) -> None:
        audio = load_mono_audio_samples(path)
        buffer.append_audio(audio, stream_id=-1)

    def _perform_streaming(self, streaming_buffer: Any) -> List[str]:
        torch = self._torch
        model = self._model
        assert torch is not None
        assert model is not None

        batch_size = len(streaming_buffer.streams_length)
        cache_last_channel, cache_last_time, cache_last_channel_len = model.encoder.get_initial_cache_state(
            batch_size=batch_size
        )
        previous_hypotheses = None
        previous_pred_out = None
        transcribed_texts: List[str] = []

        for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(streaming_buffer)):
            with torch.inference_mode():
                chunk_audio = chunk_audio.to(torch.float32)
                with torch.no_grad():
                    (
                        previous_pred_out,
                        transcribed_texts,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                    ) = model.conformer_stream_step(
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=streaming_buffer.is_buffer_empty(),
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=previous_pred_out,
                        drop_extra_pre_encoded=self._drop_extra_pre_encoded(step_num),
                        return_transcription=True,
                    )

        return self._extract_transcriptions(transcribed_texts)

    def _configure_model(self, model: Any) -> None:
        if hasattr(model.encoder, "set_default_att_context_size"):
            model.encoder.set_default_att_context_size(att_context_size=list(self.cfg.att_context_size))
        if hasattr(model, "change_decoding_strategy") and hasattr(model, "decoding"):
            model.change_decoding_strategy(model.cfg.decoding)

    def _drop_extra_pre_encoded(self, step_num: int) -> int:
        if step_num == 0 and not self.cfg.pad_and_drop_preencoded:
            return 0
        model = self._model
        assert model is not None
        return int(getattr(model.encoder.streaming_cfg, "drop_extra_pre_encoded", 0))

    @staticmethod
    def _extract_transcriptions(hypotheses: Any) -> List[str]:
        return [str(getattr(hyp, "text", hyp)) for hyp in (hypotheses or [])]

    @staticmethod
    def _resolve_torch_device(torch: Any, requested: str) -> Any:
        requested = str(requested or "auto").lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(requested)

    @staticmethod
    def _resolve_amp_dtype(torch: Any, name: str) -> Any:
        return torch.bfloat16 if str(name).lower() == "bfloat16" else torch.float16
