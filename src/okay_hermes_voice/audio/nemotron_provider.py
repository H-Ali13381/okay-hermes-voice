"""In-process Nemotron cache-aware streaming transcriber."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from ..daemon_config import LOG
from .nemotron_config import NemotronStreamingConfig


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

        torch, nemo_asr, buffer_cls = _load_nemotron_dependencies()
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

    def start_live_session(self) -> "NemotronLiveStreamingSession":
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
        audio = _load_mono_audio_samples(path)
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


class NemotronLiveStreamingSession:
    """Incremental Nemotron streaming session fed by live microphone blocks."""

    def __init__(self, transcriber: NemotronStreamingTranscriber, buffer: Any):
        self.transcriber = transcriber
        self.buffer = buffer
        self._stream_id: Optional[int] = None
        self._cache_last_channel = None
        self._cache_last_time = None
        self._cache_last_channel_len = None
        self._previous_hypotheses = None
        self._previous_pred_out = None
        self._step_num = 0
        self._latest_texts: List[str] = []

    def accept_int16(self, block: Any) -> str:
        """Append one live int16 microphone block and consume newly available chunks."""
        audio = self._int16_block_to_float32_audio(block)
        return self.accept_audio(audio)

    def accept_audio(self, audio: Any) -> str:
        audio_array = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio_array.size == 0:
            return self.latest_transcript
        _, _, stream_id = self.buffer.append_audio(
            audio_array,
            stream_id=-1 if self._stream_id is None else self._stream_id,
        )
        resolved_stream_id = int(stream_id)
        self._stream_id = 0 if resolved_stream_id < 0 else resolved_stream_id
        self._consume_available(final=False)
        return self.latest_transcript

    def finalize(self) -> str:
        self._consume_available(final=True)
        transcript = self.latest_transcript
        LOG.info("Nemotron live streaming transcript: %s", transcript)
        return transcript

    @property
    def latest_transcript(self) -> str:
        return (self._latest_texts[0] if self._latest_texts else "").strip()

    def _ensure_cache_state(self) -> bool:
        if getattr(self.buffer, "streams_length", None) is None:
            return False
        if self._cache_last_channel is not None:
            return True
        model = self.transcriber._model
        assert model is not None
        batch_size = len(self.buffer.streams_length)
        (
            self._cache_last_channel,
            self._cache_last_time,
            self._cache_last_channel_len,
        ) = model.encoder.get_initial_cache_state(batch_size=batch_size)
        return True

    def _consume_available(self, *, final: bool) -> None:
        if not self._ensure_cache_state():
            return
        torch = self.transcriber._torch
        model = self.transcriber._model
        assert torch is not None
        assert model is not None

        with self.transcriber.amp_context:
            if not hasattr(self.buffer, "buffer") or not hasattr(self.buffer, "buffer_idx"):
                for chunk_audio, chunk_lengths in iter(self.buffer):
                    self._consume_chunk(torch, model, chunk_audio, chunk_lengths, final=final)
                return
            while self._has_chunk_to_process(final=final):
                iterator = iter(self.buffer)
                try:
                    chunk_audio, chunk_lengths = next(iterator)
                except StopIteration:
                    break
                self._consume_chunk(torch, model, chunk_audio, chunk_lengths, final=final)

    def _consume_chunk(self, torch: Any, model: Any, chunk_audio: Any, chunk_lengths: Any, *, final: bool) -> None:
        with torch.inference_mode():
            chunk_audio = chunk_audio.to(torch.float32)
            with torch.no_grad():
                (
                    self._previous_pred_out,
                    transcribed_texts,
                    self._cache_last_channel,
                    self._cache_last_time,
                    self._cache_last_channel_len,
                    self._previous_hypotheses,
                ) = model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=self._cache_last_channel,
                    cache_last_time=self._cache_last_time,
                    cache_last_channel_len=self._cache_last_channel_len,
                    keep_all_outputs=bool(final and self.buffer.is_buffer_empty()),
                    previous_hypotheses=self._previous_hypotheses,
                    previous_pred_out=self._previous_pred_out,
                    drop_extra_pre_encoded=self.transcriber._drop_extra_pre_encoded(self._step_num),
                    return_transcription=True,
                )
        self._step_num += 1
        extracted = self.transcriber._extract_transcriptions(transcribed_texts)
        if extracted:
            self._latest_texts = extracted

    def _has_chunk_to_process(self, *, final: bool) -> bool:
        if getattr(self.buffer, "buffer", None) is None:
            return False
        remaining = int(self.buffer.buffer.size(-1) - self.buffer.buffer_idx)
        if remaining <= 0:
            return False
        if final:
            return True
        return remaining >= self._next_chunk_size()

    def _next_chunk_size(self) -> int:
        chunk_size = self.buffer.streaming_cfg.chunk_size
        if self.buffer.buffer_idx == 0 and isinstance(chunk_size, list):
            index = 1 if self.buffer.pad_and_drop_preencoded else 0
            return int(chunk_size[index])
        if isinstance(chunk_size, list):
            return int(chunk_size[1])
        return int(chunk_size)

    @staticmethod
    def _int16_block_to_float32_audio(block: Any) -> np.ndarray:
        audio = np.asarray(block)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if np.issubdtype(audio.dtype, np.integer):
            return (audio.astype(np.float32) / 32768.0).reshape(-1)
        return audio.astype(np.float32, copy=False).reshape(-1)


def _load_mono_audio_samples(path: Path) -> Any:
    from nemo.collections.asr.parts.preprocessing.segment import get_samples  # type: ignore[import-not-found]
    import numpy as np

    audio = np.asarray(get_samples(str(Path(path).expanduser())))
    if audio.ndim == 2:
        # NeMo's get_samples() transposes multichannel files to (channels, time),
        # but CacheAwareStreamingAudioBuffer.preprocess_audio expects (time,).
        channel_axis = 0 if audio.shape[0] <= audio.shape[1] else 1
        audio = audio.mean(axis=channel_axis)
    else:
        audio = audio.reshape(-1)
    return audio.astype("float32", copy=False)


def _load_nemotron_dependencies() -> Tuple[Any, Any, Any]:
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


__all__ = ["NemotronLiveStreamingSession", "NemotronStreamingTranscriber"]
