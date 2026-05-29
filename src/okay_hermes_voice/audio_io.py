"""Wakeword, recording, and speech-to-text audio helpers."""
from __future__ import annotations

import collections
import contextlib
import json
import math
import queue
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import sounddevice as sd

from tools.voice_mode import is_whisper_hallucination, transcribe_recording

from .daemon_config import DEFAULT_CONFIG, LOG, STOP, setup_logging

def model_session(model_path: str) -> Tuple[ort.InferenceSession, str, str]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Wakeword ONNX model not found: {path}")
    opts = ort.SessionOptions()
    # Keep an always-on listener cheap: the RepCNN model is tiny, and letting
    # ONNX Runtime fan out over every CPU core costs more than it saves here.
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    shape = session.get_inputs()[0].shape
    LOG.info("Loaded wakeword model: %s input=%s shape=%s output=%s", path, input_name, shape, output_name)
    return session, input_name, output_name


def run_wake_inference(session: ort.InferenceSession, input_name: str, output_name: str, waveform: np.ndarray) -> float:
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32, copy=False)
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    probability = session.run([output_name], {input_name: waveform[None, :]})[0][0]
    return float(probability)


def rms_int16(block: np.ndarray) -> float:
    arr = block.astype(np.float32, copy=False).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(arr * arr))))


def write_wav_int16(audio: np.ndarray, sample_rate: int, prefix: str = "wake_command") -> Path:
    out_dir = Path(tempfile.gettempdir()) / "hermes_voice_wakeword"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    write_wav_int16_to_path(audio, sample_rate, path)
    return path


def write_wav_int16_to_path(audio: np.ndarray, sample_rate: int, path: Path) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.int16).reshape(-1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return path


def float_waveform_to_int16(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    clipped = np.clip(waveform, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def wait_for_wake(cfg: Dict[str, Any], session: ort.InferenceSession, input_name: str, output_name: str) -> Optional[Dict[str, Any]]:
    sample_rate = int(cfg["sample_rate"])
    window_samples = int(float(cfg["window_seconds"]) * sample_rate)
    block_samples = int(float(cfg["block_seconds"]) * sample_rate)
    inference_interval = float(cfg["inference_interval_seconds"])
    threshold = float(cfg["threshold"])
    consecutive = max(1, int(cfg["trigger_consecutive_windows"]))

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
    rolling: Deque[float] = collections.deque(maxlen=window_samples)
    recent: Deque[float] = collections.deque(maxlen=consecutive)
    last_inference = 0.0

    def callback(indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        del frames, time_info
        if status:
            LOG.debug("Wake audio callback status: %s", status)
        block = np.asarray(indata[:, 0], dtype=np.float32).copy()
        with contextlib.suppress(queue.Full):
            audio_q.put_nowait(block)

    LOG.info("Listening for wakeword: threshold=%.6f consecutive=%d", threshold, consecutive)
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_samples,
        callback=callback,
    ):
        while not STOP.is_set():
            try:
                block = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            rolling.extend(float(x) for x in block)
            if len(rolling) < window_samples:
                continue

            now = time.monotonic()
            if now - last_inference < inference_interval:
                continue
            last_inference = now

            waveform = np.fromiter(rolling, dtype=np.float32, count=window_samples)
            probability = run_wake_inference(session, input_name, output_name, waveform)
            recent.append(probability)
            if probability >= threshold * 0.75:
                LOG.debug("Wake score %.6f", probability)
            if len(recent) == consecutive and all(score >= threshold for score in recent):
                scores = [float(score) for score in recent]
                LOG.info("Wakeword detected: scores=%s", [round(s, 6) for s in scores])
                return {
                    "probability": max(scores),
                    "scores": scores,
                    "waveform": waveform.copy(),
                    "sample_rate": sample_rate,
                    "detected_at": time.time(),
                }
    return None


def _cancel_check_requested(cancel_check: Optional[Callable[[], bool]]) -> bool:
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception as exc:
        LOG.warning("Voice-session cancel check failed: %s", exc)
        return False


def record_command(cfg: Dict[str, Any], cancel_check: Optional[Callable[[], bool]] = None) -> Optional[Path]:
    sample_rate = int(cfg["sample_rate"])
    block_samples = int(float(cfg["block_seconds"]) * sample_rate)
    threshold = float(cfg["speech_rms_threshold"])
    start_timeout = float(cfg["speech_start_timeout_seconds"])
    silence_duration = float(cfg["speech_silence_duration_seconds"])
    max_seconds = float(cfg["max_command_seconds"])
    min_seconds = float(cfg["min_command_seconds"])
    start_consecutive_blocks = max(1, int(cfg.get("speech_start_consecutive_blocks") or 1))

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=128)
    chunks: List[np.ndarray] = []
    preroll: Deque[np.ndarray] = collections.deque(maxlen=max(1, int(0.4 / float(cfg["block_seconds"]))))
    started = False
    start_time = time.monotonic()
    speech_start_time: Optional[float] = None
    last_voice_time: Optional[float] = None
    consecutive_voice_blocks = 0

    def callback(indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        del frames, time_info
        if status:
            LOG.debug("Command audio callback status: %s", status)
        block = np.asarray(indata[:, 0], dtype=np.int16).copy()
        with contextlib.suppress(queue.Full):
            audio_q.put_nowait(block)

    if _cancel_check_requested(cancel_check):
        LOG.info("Command recording cancelled before audio stream opened")
        return None

    LOG.info("Recording command; speech_rms_threshold=%.1f silence=%.1fs", threshold, silence_duration)
    with sd.InputStream(
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
            if last_voice_time is not None and now - last_voice_time >= silence_duration:
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
    return path


def transcribe_command(path: Path) -> Optional[str]:
    LOG.info("Transcribing command: %s", path)
    result = transcribe_recording(str(path))
    if not result.get("success"):
        LOG.error("STT failed: %s", result.get("error") or result)
        return None
    transcript = (result.get("transcript") or "").strip()
    if is_whisper_hallucination(transcript):
        LOG.info("Filtered empty/hallucinated transcript: %r", transcript)
        return None
    LOG.info("Transcript: %s", transcript)
    return transcript


def prewarm_stt(cfg: Dict[str, Any]) -> None:
    """Load Hermes' STT stack once so the first real wake request is not delayed."""
    if not cfg.get("prewarm_stt_on_start", True):
        return
    try:
        sample_rate = int(cfg["sample_rate"])
        silence = np.zeros(int(sample_rate * 0.25), dtype=np.int16)
        path = write_wav_int16(silence, sample_rate, prefix="wake_stt_prewarm")
        LOG.info("Prewarming STT with %s", path)
        result = transcribe_recording(str(path))
        LOG.info("STT prewarm result: %s", result)
    except Exception as exc:
        LOG.warning("STT prewarm failed; continuing: %s", exc)


def smoke_test(cfg: Dict[str, Any]) -> int:
    setup_logging(cfg, verbose=True)
    session, input_name, output_name = model_session(cfg["model_path"])
    sample_rate = int(cfg["sample_rate"])
    window_samples = int(float(cfg["window_seconds"]) * sample_rate)
    zeros = np.zeros(window_samples, dtype=np.float32)
    prob = run_wake_inference(session, input_name, output_name, zeros)
    print(json.dumps({
        "ok": True,
        "model_path": cfg["model_path"],
        "zero_audio_probability": prob,
        "threshold": float(cfg["threshold"]),
        "input_name": input_name,
        "output_name": output_name,
    }, indent=2))
    return 0


def list_devices() -> int:
    print(sd.query_devices())
    return 0
