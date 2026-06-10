"""Live Parakeet streaming session."""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ...daemon_config import LOG


class ParakeetLiveStreamingSession:
    """Live stateful chunked RNNT session fed by microphone blocks."""

    def __init__(self, transcriber: Any):
        self.transcriber = transcriber
        self.torch = transcriber._torch
        self.model = transcriber._model
        assert self.torch is not None
        assert self.model is not None
        self.sample_rate, self.context_encoder_frames, self.context_samples, self.encoder_frame_samples = self._compute_contexts()
        self.buffer = transcriber._stream_buffer_cls(
            batch_size=1,
            context_samples=self.context_samples,
            dtype=self.torch.float32,
            device=transcriber._device,
        )
        self.audio = self.torch.empty((1, 0), dtype=self.torch.float32, device=transcriber._device)
        self.left_sample = 0
        self.right_sample = self.context_samples.chunk + self.context_samples.right
        self.state = None
        self.current_batched_hyps = None
        self.latest_text = ""
        self._configure_att_context()

    def accept_int16(self, block: Any) -> str:
        return self.accept_audio(self._int16_block_to_float32_audio(block))

    def accept_audio(self, audio: Any) -> str:
        audio_array = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio_array.size == 0:
            return self.latest_transcript
        audio_tensor = self.torch.as_tensor(audio_array, dtype=self.torch.float32, device=self.audio.device).unsqueeze(0)
        self.audio = self.torch.cat((self.audio, audio_tensor), dim=1)
        self._consume_available(final=False)
        return self.latest_transcript

    def finalize(self) -> str:
        self._consume_available(final=True)
        LOG.info("Parakeet live streaming transcript: %s", self.latest_transcript)
        return self.latest_transcript

    @property
    def latest_transcript(self) -> str:
        return self.latest_text.strip()

    def _compute_contexts(self) -> Tuple[int, Any, Any, int]:
        cfg = self.model.cfg
        sample_rate = int(cfg.preprocessor.sample_rate)
        feature_stride_sec = float(cfg.preprocessor.window_stride)
        features_per_sec = 1.0 / feature_stride_sec
        subsampling_factor = int(self.model.encoder.subsampling_factor)
        features_frame_samples = self._make_divisible_by(int(sample_rate * feature_stride_sec), subsampling_factor)
        encoder_frame_samples = features_frame_samples * subsampling_factor
        context_cls = self.transcriber._context_size_cls
        provider_cfg = self.transcriber.cfg
        encoder_frames = context_cls(
            left=int(provider_cfg.left_context_secs * features_per_sec / subsampling_factor),
            chunk=max(1, int(provider_cfg.chunk_secs * features_per_sec / subsampling_factor)),
            right=int(provider_cfg.right_context_secs * features_per_sec / subsampling_factor),
        )
        context_samples = context_cls(
            left=encoder_frames.left * encoder_frame_samples,
            chunk=encoder_frames.chunk * encoder_frame_samples,
            right=encoder_frames.right * encoder_frame_samples,
        )
        return sample_rate, encoder_frames, context_samples, encoder_frame_samples

    def _configure_att_context(self) -> None:
        encoder_cfg = getattr(getattr(self.model, "cfg", None), "encoder", None)
        if getattr(encoder_cfg, "att_context_style", None) != "chunked_limited_with_rc":
            return
        context = [
            self.context_encoder_frames.left,
            self.context_encoder_frames.chunk,
            self.context_encoder_frames.right,
        ]
        supported = getattr(self.model.encoder, "att_context_size_all", None)
        if supported and context not in supported:
            LOG.debug("Skipping unsupported Parakeet att_context_size=%s; supported=%s", context, supported)
            return
        self.model.encoder.set_default_att_context_size(att_context_size=context)

    def _consume_available(self, *, final: bool) -> None:
        total = int(self.audio.shape[1])
        while self.left_sample < total and (final or total >= self.right_sample):
            chunk_end = min(self.right_sample, total)
            is_last = bool(final and chunk_end >= total)
            chunk_length = max(0, chunk_end - self.left_sample)
            if chunk_length <= 0:
                break
            self._consume_chunk(chunk_end, chunk_length, is_last=is_last)
            self.left_sample = chunk_end
            self.right_sample = self.right_sample + self.context_samples.chunk

    def _consume_chunk(self, chunk_end: int, chunk_length: int, *, is_last: bool) -> None:
        torch = self.torch
        chunk_lengths = torch.tensor([chunk_length], dtype=torch.long, device=self.audio.device)
        is_last_batch = torch.tensor([is_last], dtype=torch.bool, device=self.audio.device)
        self.buffer.add_audio_batch_(
            self.audio[:, self.left_sample : chunk_end],
            audio_lengths=chunk_lengths,
            is_last_chunk=is_last,
            is_last_chunk_batch=is_last_batch,
        )
        with self.transcriber.amp_context:
            with torch.no_grad():
                encoder_output, encoder_output_len = self.model(
                    input_signal=self.buffer.samples,
                    input_signal_length=self.buffer.context_size_batch.total(),
                )
                encoder_output = encoder_output.transpose(1, 2)
                encoder_context = self.buffer.context_size.subsample(factor=self.encoder_frame_samples)
                encoder_context_batch = self.buffer.context_size_batch.subsample(factor=self.encoder_frame_samples)
                encoder_output = encoder_output[:, encoder_context.left :]
                output_len = torch.where(is_last_batch, encoder_output_len - encoder_context_batch.left, encoder_context_batch.chunk)
                chunk_hyps, self.state = self.model.decoding.decoding.decoding_computer(
                    x=encoder_output,
                    out_len=output_len,
                    prev_batched_state=self.state,
                    multi_biasing_ids=None,
                )
        if self.current_batched_hyps is None:
            self.current_batched_hyps = chunk_hyps
        else:
            self.current_batched_hyps.merge_(chunk_hyps)
        self._refresh_latest_text()

    def _refresh_latest_text(self) -> None:
        converter = self.transcriber._batched_hyps_to_hypotheses
        assert converter is not None
        hyps = converter(self.current_batched_hyps, batch_size=1)
        if hyps:
            self.latest_text = self.model.tokenizer.ids_to_text(hyps[0].y_sequence.tolist())

    @staticmethod
    def _make_divisible_by(num: int, factor: int) -> int:
        return (num // factor) * factor

    @staticmethod
    def _int16_block_to_float32_audio(block: Any) -> np.ndarray:
        audio = np.asarray(block)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if np.issubdtype(audio.dtype, np.integer):
            return (audio.astype(np.float32) / 32768.0).reshape(-1)
        return audio.astype(np.float32, copy=False).reshape(-1)
