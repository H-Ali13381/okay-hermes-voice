"""Command recording loop and cancellation boundary."""
from __future__ import annotations

import collections
import contextlib
import queue
import time
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Union

import numpy as np

from ..daemon_config import LOG, STOP
from .nemotron_config import is_nemotron_provider
from .nemotron_stt import start_nemotron_live_streaming
from .parakeet_config import is_parakeet_provider
from .parakeet_stt import start_parakeet_live_streaming
from .recording_result import CommandRecording
from .waveform import rms_int16
from .wav import write_wav_int16

_SD = None


class _SoundDeviceBackend:
    def __call__(self) -> Any:
        global _SD
        if _SD is None:
            import sounddevice as sd  # type: ignore[import-not-found]

            _SD = sd
        return _SD


_sounddevice = _SoundDeviceBackend()


def _cancel_check_requested(cancel_check: Optional[Callable[[], bool]]) -> bool:
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception as exc:
        LOG.warning("Voice-session cancel check failed: %s", exc)
        return False


def record_command(cfg: Dict[str, Any], cancel_check: Optional[Callable[[], bool]] = None) -> Optional[Union[Path, CommandRecording]]:
    sample_rate = int(cfg["sample_rate"])
    block_samples = int(float(cfg["block_seconds"]) * sample_rate)
    threshold = float(cfg["speech_rms_threshold"])
    start_timeout = float(cfg["speech_start_timeout_seconds"])
    silence_duration = float(cfg["speech_silence_duration_seconds"])
    streamed_text_idle_duration = float(cfg.get("streamed_text_idle_duration_seconds", 0.0) or 0.0)
    max_seconds = float(cfg["max_command_seconds"])
    min_seconds = float(cfg["min_command_seconds"])
    start_consecutive_blocks = max(1, int(cfg.get("speech_start_consecutive_blocks") or 1))

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=128)
    chunks: List[np.ndarray] = []
    preroll: Deque[np.ndarray] = collections.deque(maxlen=max(1, int(0.4 / float(cfg["block_seconds"]))))
    live_session = None
    live_transcript = ""
    last_streamed_text = ""
    last_streamed_text_update_time: Optional[float] = None
    started = False
    start_time = time.monotonic()
    speech_start_time: Optional[float] = None
    last_voice_time: Optional[float] = None
    consecutive_voice_blocks = 0

    def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        if status:
            LOG.debug("Command audio callback status: %s", status)
        block = np.asarray(indata[:, 0], dtype=np.int16).copy()
        with contextlib.suppress(queue.Full):
            audio_q.put_nowait(block)

    def note_live_transcript(transcript: str, timestamp: float) -> None:
        nonlocal last_streamed_text, last_streamed_text_update_time
        normalized = " ".join((transcript or "").split())
        if normalized and normalized != last_streamed_text:
            last_streamed_text = normalized
            last_streamed_text_update_time = timestamp

    if _cancel_check_requested(cancel_check):
        LOG.info("Command recording cancelled before audio stream opened")
        return None
    if bool(cfg.get("parakeet_live_streaming", False)) and is_parakeet_provider(cfg.get("stt_provider")):
        LOG.info("Starting live Parakeet streaming STT before command capture")
        live_session = start_parakeet_live_streaming(cfg)
    elif bool(cfg.get("nemotron_live_streaming", False)) and is_nemotron_provider(cfg.get("stt_provider")):
        LOG.info("Starting live Nemotron streaming STT before command capture")
        live_session = start_nemotron_live_streaming(cfg)

    LOG.info("Recording command; speech_rms_threshold=%.1f silence=%.1fs", threshold, silence_duration)
    with _sounddevice().InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=block_samples,
        callback=callback,
    ):
        while not STOP.is_set():
            if _cancel_check_requested(cancel_check):
                LOG.info("Command recording cancelled by active voice-session request")
                return None
            now = time.monotonic()
            if started and speech_start_time is not None and now - speech_start_time > max_seconds:
                LOG.info("Command recording hit max %.1fs", max_seconds)
                break
            if not started and start_timeout > 0 and now - start_time > start_timeout:
                LOG.info("No speech after wakeword for %.1fs", start_timeout)
                return None
            try:
                block = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            now = time.monotonic()

            level = rms_int16(block)
            has_voice = level >= threshold
            if not started:
                preroll.append(block)
                if has_voice:
                    consecutive_voice_blocks += 1
                    last_voice_time = now
                    if consecutive_voice_blocks >= start_consecutive_blocks:
                        started = True
                        speech_start_time = now - (start_consecutive_blocks - 1) * float(cfg["block_seconds"])
                        chunks.extend(list(preroll))
                        if live_session is not None:
                            for buffered_block in preroll:
                                live_transcript = live_session.accept_int16(buffered_block)
                                note_live_transcript(live_transcript, now)
                        LOG.info(
                            "Speech started; rms=%.1f consecutive_blocks=%d",
                            level,
                            consecutive_voice_blocks,
                        )
                else:
                    consecutive_voice_blocks = 0
                continue

            if has_voice:
                last_voice_time = now
            chunks.append(block)
            if live_session is not None:
                live_transcript = live_session.accept_int16(block)
                note_live_transcript(live_transcript, now)
            using_streamed_text_endpointing = (
                live_session is not None and streamed_text_idle_duration > 0 and last_streamed_text_update_time is not None
            )
            if using_streamed_text_endpointing:
                assert last_streamed_text_update_time is not None
                if now - last_streamed_text_update_time >= streamed_text_idle_duration:
                    LOG.info("Speech ended after %.1fs without streamed text changes", streamed_text_idle_duration)
                    break
            if not using_streamed_text_endpointing and last_voice_time is not None and now - last_voice_time >= silence_duration:
                LOG.info("Speech ended after %.1fs silence", silence_duration)
                break

    if not chunks or speech_start_time is None:
        return None
    audio = np.concatenate(chunks).astype(np.int16, copy=False)
    duration = audio.size / sample_rate
    if duration < min_seconds:
        LOG.info("Ignoring too-short command: %.2fs", duration)
        return None
    path = write_wav_int16(audio, sample_rate)
    LOG.info("Command WAV saved: %s (%.2fs)", path, duration)
    if live_session is not None:
        live_transcript = live_session.finalize() or live_transcript
        return CommandRecording(path=path, live_transcript=live_transcript)
    return path


__all__ = ["CommandRecording", "record_command"]
