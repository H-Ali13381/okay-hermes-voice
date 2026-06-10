"""Live Nemotron streaming session."""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from ...daemon_config import LOG


class NemotronLiveStreamingSession:
    """Incremental Nemotron streaming session fed by live microphone blocks."""

    def __init__(self, transcriber: Any, buffer: Any):
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
