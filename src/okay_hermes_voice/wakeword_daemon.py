#!/usr/bin/env python3
"""Always-on "Okay Hermes" wakeword daemon for Hermes Agent.

Listens locally for the Okay Hermes RepCNN ONNX wakeword. On trigger it:
1. plays an acknowledgement beep,
2. records one spoken command until silence,
3. transcribes using Hermes' configured STT stack,
4. sends the transcript to `hermes chat -Q --source wakeword -q ...`, and
5. speaks the final response using Hermes' configured TTS stack.

Audio stays local except for whatever the configured Hermes LLM/TTS/STT providers do.
The default config uses local faster-whisper STT and Edge TTS.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import logging
import math
import os
import queue
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
HERMES_REPO = Path(os.environ.get("HERMES_REPO", str(HERMES_HOME / "hermes-agent"))).expanduser()
if str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import yaml

from tools.tts_tool import text_to_speech_tool
from tools.voice_mode import is_whisper_hallucination, play_audio_file, play_beep, transcribe_recording

CONFIG_PATH = HERMES_HOME / "wakeword" / "config.yaml"
DEFAULT_CONFIG: Dict[str, Any] = {
    "model_path": str(HERMES_HOME / "wakeword" / "okay-hermes-repcnn-onnx" / "retrained_20260510_165910_folded.onnx"),
    "threshold": 0.4112943708896637,
    "trigger_consecutive_windows": 2,
    "inference_interval_seconds": 0.25,
    "cooldown_seconds": 2.5,
    "sample_rate": 16000,
    "window_seconds": 3.0,
    "block_seconds": 0.1,
    "speech_rms_threshold": 200,
    "speech_start_timeout_seconds": 15.0,
    "speech_silence_duration_seconds": 1.1,
    "max_command_seconds": 120.0,
    "min_command_seconds": 0.45,
    "hermes_bin": "hermes",
    "hermes_timeout_seconds": 900,
    "hermes_source": "wakeword",
    "hermes_inprocess": True,
    # Empty provider/model means: inherit the user's default Hermes config.
    "hermes_provider": "",
    "hermes_model": "",
    # None/blank means: inherit the user's normal CLI toolsets (including skills,
    # terminal, file, memory, session_search, etc.).
    "hermes_toolsets": None,
    "hermes_warm_agent": True,
    "hermes_max_iterations": 90,
    # Keep project context files skipped for a daemon launched from arbitrary
    # directories, but still load ~/.hermes/SOUL.md so voice mode has the same
    # persona and operating discipline as the normal Hermes CLI.
    "hermes_load_soul_identity": True,
    "hermes_prompt_prefix": (
        "You were activated by the local \"Okay Hermes\" wakeword daemon. "
        "The following is a speech-to-text transcript of the user's spoken request. "
        "Use the same Hermes Agent reasoning, tools, skills, memory, and execution discipline as the normal CLI. "
        "Do not treat voice mode as a weaker assistant. The transcript may contain speech-to-text errors; infer the likely intent when obvious, "
        "and mention uncertainty when it matters. Keep spoken answers concise unless the task requires depth."
    ),
    "tts_enabled": True,
    "max_spoken_response_chars": 2500,
    "playback_sink": "@DEFAULT_SINK@",
    "playback_volume": 1.0,
    "beep_enabled": True,
    "prewarm_stt_on_start": True,
    "prewarm_hermes_on_start": True,
    "visualization_enabled": True,
    "visualization_terminal": "auto",
    "visualization_title": "Hermes Voice",
    "visualization_keep_open_seconds": 45.0,
    "visualization_script": str(Path(__file__).with_name("voice_activation_popup.py")),
    "conversation_mode_enabled": True,
    "conversation_followup_start_timeout_seconds": 0.0,
    "conversation_max_turns": 50,
    "conversation_followup_beep_enabled": False,
    "conversation_close_ack": "Closing voice mode.",
    "conversation_close_phrases": [
        "close",
        "please close",
        "close voice",
        "close voice mode",
        "close conversation",
        "close hermes",
        "stop",
        "stop voice",
        "stop listening",
        "end conversation",
        "end voice mode",
        "that's all",
        "that is all",
        "goodbye",
        "bye",
        "cancel",
    ],
    "save_activation_audio": True,
    "activation_archive_dir": str(HERMES_HOME / "wakeword" / "activations"),
    "activation_save_command_audio": True,
    "log_path": str(HERMES_HOME / "logs" / "okay-hermes-voice.log"),
}

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
STOP = threading.Event()
LOG = logging.getLogger("okay-hermes-voice")
_HERMES_AGENT_CACHE: Dict[str, Any] = {}


def deep_merge(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    cfg = deep_merge(DEFAULT_CONFIG, data)
    cfg["model_path"] = str(Path(cfg["model_path"]).expanduser())
    cfg["log_path"] = str(Path(cfg["log_path"]).expanduser())
    cfg["visualization_script"] = str(Path(cfg["visualization_script"]).expanduser())
    cfg["activation_archive_dir"] = str(Path(cfg["activation_archive_dir"]).expanduser())
    return cfg


def setup_logging(cfg: Dict[str, Any], verbose: bool = False) -> None:
    log_path = Path(cfg["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def signal_handler(signum: int, _frame: Any) -> None:
    LOG.info("Received signal %s; stopping", signum)
    STOP.set()


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


def _activation_timestamp(detected_at: float) -> str:
    whole = time.strftime("%Y%m%d_%H%M%S", time.localtime(detected_at))
    millis = int((detected_at - int(detected_at)) * 1000)
    return f"{whole}_{millis:03d}"


def save_activation_archive(cfg: Dict[str, Any], activation: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Save the triggering wakeword audio window and metadata for review."""
    if not cfg.get("save_activation_audio", True):
        return None
    try:
        waveform = activation.get("waveform")
        if waveform is None:
            LOG.warning("Activation archive requested but activation has no waveform")
            return None
        sample_rate = int(activation.get("sample_rate") or cfg.get("sample_rate") or 16000)
        probability = float(activation.get("probability") or 0.0)
        detected_at = float(activation.get("detected_at") or time.time())
        scores = [float(score) for score in (activation.get("scores") or [probability])]
        out_dir = Path(str(cfg.get("activation_archive_dir") or DEFAULT_CONFIG["activation_archive_dir"])).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"activation_{_activation_timestamp(detected_at)}_p{probability:.3f}_{os.getpid()}_{time.monotonic_ns()}"
        wav_path = out_dir / f"{stem}.wav"
        metadata_path = out_dir / f"{stem}.json"
        audio = float_waveform_to_int16(np.asarray(waveform, dtype=np.float32))
        write_wav_int16_to_path(audio, sample_rate, wav_path)
        metadata: Dict[str, Any] = {
            "status": "wake_detected",
            "detected_at": detected_at,
            "detected_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(detected_at)),
            "probability": probability,
            "scores": scores,
            "threshold": float(cfg.get("threshold") or 0.0),
            "trigger_consecutive_windows": int(cfg.get("trigger_consecutive_windows") or len(scores) or 1),
            "sample_rate": sample_rate,
            "window_seconds": float(len(audio) / sample_rate) if sample_rate else 0.0,
            "wake_wav_path": str(wav_path),
            "metadata_path": str(metadata_path),
            "model_path": str(cfg.get("model_path") or ""),
            "command_wav_paths": [],
            "turns": [],
        }
        tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, metadata_path)
        LOG.info("Saved activation archive: %s", wav_path)
        return {"wake_wav_path": str(wav_path), "metadata_path": str(metadata_path), "stem": stem}
    except Exception as exc:
        LOG.warning("Could not save activation archive: %s", exc)
        return None


def update_activation_archive_metadata(archive: Optional[Dict[str, Any]], **updates: Any) -> None:
    """Merge metadata updates into the activation archive JSON."""
    if not archive:
        return
    try:
        raw_metadata_path = archive.get("metadata_path")
        if not raw_metadata_path:
            return
        metadata_path = Path(str(raw_metadata_path)).expanduser()
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with contextlib.suppress(Exception):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(updates)
        metadata["updated_at"] = time.time()
        metadata["updated_at_local"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(metadata["updated_at"]))
        tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, metadata_path)
    except Exception as exc:
        LOG.warning("Could not update activation archive metadata: %s", exc)


def archive_command_audio(cfg: Dict[str, Any], archive: Optional[Dict[str, Any]], command_path: Path, turn_index: int) -> Optional[str]:
    if not archive or not cfg.get("activation_save_command_audio", True):
        return None
    try:
        raw_metadata_path = archive.get("metadata_path")
        if not raw_metadata_path:
            return None
        metadata_path = Path(str(raw_metadata_path)).expanduser()
        stem = str(archive.get("stem") or metadata_path.stem)
        dest = metadata_path.with_name(f"{stem}_turn{turn_index:02d}_command.wav")
        shutil.copy2(str(command_path), str(dest))
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with contextlib.suppress(Exception):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths = metadata.get("command_wav_paths")
        if not isinstance(paths, list):
            paths = []
        paths.append(str(dest))
        update_activation_archive_metadata(archive, command_wav_paths=paths)
        return str(dest)
    except Exception as exc:
        LOG.warning("Could not archive command audio %s: %s", command_path, exc)
        return None


def command_audio_metadata_fields(archived_command_path: Optional[str], command_path: Path, latest: bool = False) -> Dict[str, str]:
    if archived_command_path:
        return {"latest_command_wav_path" if latest else "command_wav_path": archived_command_path}
    return {"temp_command_wav_path": str(command_path)}


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

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=128)
    chunks: List[np.ndarray] = []
    preroll: Deque[np.ndarray] = collections.deque(maxlen=max(1, int(0.4 / float(cfg["block_seconds"]))))
    started = False
    start_time = time.monotonic()
    speech_start_time: Optional[float] = None
    last_voice_time: Optional[float] = None

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
            if has_voice:
                last_voice_time = now
                if not started:
                    started = True
                    speech_start_time = now
                    chunks.extend(list(preroll))
                    LOG.info("Speech started; rms=%.1f", level)
            if started:
                chunks.append(block)
                if last_voice_time is not None and now - last_voice_time >= silence_duration:
                    LOG.info("Speech ended after %.1fs silence", silence_duration)
                    break
            else:
                preroll.append(block)

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


def normalize_voice_command(text: str) -> str:
    """Normalize STT text for exact local voice-control commands."""
    normalized = (text or "").casefold()
    normalized = re.sub(r"[^\w\s']+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for prefix in ("okay hermes ", "ok hermes ", "hey hermes ", "hermes "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
            break
    return normalized


def is_close_transcript(transcript: str, cfg: Dict[str, Any]) -> bool:
    """Return True only for explicit voice-session close commands."""
    normalized = normalize_voice_command(transcript)
    phrases = cfg.get("conversation_close_phrases") or DEFAULT_CONFIG["conversation_close_phrases"]
    normalized_phrases = {normalize_voice_command(str(phrase)) for phrase in phrases if str(phrase).strip()}
    return normalized in normalized_phrases


def command_recording_config_for_turn(cfg: Dict[str, Any], turn_index: int) -> Dict[str, Any]:
    """Return per-turn recording config; follow-ups can wait indefinitely."""
    turn_cfg = dict(cfg)
    if turn_index > 1 and cfg.get("conversation_mode_enabled", True):
        turn_cfg["speech_start_timeout_seconds"] = float(cfg.get("conversation_followup_start_timeout_seconds", 0.0) or 0.0)
    return turn_cfg


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


def prewarm_hermes(cfg: Dict[str, Any]) -> None:
    """Initialize the warm in-process agent at service start."""
    if not (cfg.get("hermes_inprocess", True) and cfg.get("hermes_warm_agent", True) and cfg.get("prewarm_hermes_on_start", True)):
        return
    try:
        provider, model = configured_hermes_runtime_selection(cfg)
        toolsets = configured_hermes_toolsets(cfg)
        started = time.monotonic()
        get_warm_hermes_agent(cfg, provider, model, toolsets)
        LOG.info("Hermes warm-agent prewarm complete in %.2fs", time.monotonic() - started)
    except Exception as exc:
        LOG.warning("Hermes warm-agent prewarm failed; will lazy-init on first request: %s", exc)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def clean_hermes_output(stdout: str) -> str:
    lines = strip_ansi(stdout).splitlines()
    cleaned: List[str] = []
    for line in lines:
        if line.startswith("session_id:"):
            continue
        cleaned.append(line.rstrip())
    return "\n".join(cleaned).strip()


def configured_hermes_runtime_selection(cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve wakeword Hermes provider/model overrides.

    Empty wakeword provider/model values mean "inherit the user's normal Hermes
    defaults".  Resolve the model here so the warm in-process AIAgent is built
    with the same effective model as `hermes chat`, rather than an empty model
    string that can hide the active default in logs and provider routing.
    """
    provider = (cfg.get("hermes_provider") or "").strip() or None
    model = (cfg.get("hermes_model") or "").strip() or None
    if model:
        return provider, model

    try:
        from hermes_cli.config import load_config as load_hermes_config

        hermes_cfg = load_hermes_config()
        model_cfg = hermes_cfg.get("model") or {}
        if isinstance(model_cfg, str):
            model = model_cfg.strip() or None
        elif isinstance(model_cfg, dict):
            model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip() or None
    except Exception as exc:
        LOG.warning("Could not read default Hermes model from config; using provider default: %s", exc)
    return provider, model


def configured_hermes_toolsets(cfg: Dict[str, Any]) -> Optional[List[str]]:
    """Resolve wakeword toolsets.

    A non-empty `hermes_toolsets` value is an explicit wakeword-only override.
    Blank/null means inherit the regular CLI platform toolsets, which is what
    users expect when they say wakeword Hermes should behave like normal Hermes.
    """
    raw_toolsets = cfg.get("hermes_toolsets")
    if isinstance(raw_toolsets, str):
        stripped = raw_toolsets.strip()
        if stripped and stripped.lower() not in {"auto", "config", "default", "inherit"}:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        raw_toolsets = None
    if isinstance(raw_toolsets, (list, tuple, set)):
        return [str(item).strip() for item in raw_toolsets if str(item).strip()]
    if raw_toolsets is not None:
        return raw_toolsets

    try:
        from hermes_cli.config import load_config as load_hermes_config
        from hermes_cli.tools_config import _get_platform_tools

        cli_toolsets = _get_platform_tools(load_hermes_config(), "cli")
        return sorted(str(toolset) for toolset in cli_toolsets)
    except Exception as exc:
        LOG.warning("Could not read CLI toolsets from Hermes config; using all available tools: %s", exc)
        return None


def get_warm_hermes_agent(cfg: Dict[str, Any], provider: Optional[str], model: Optional[str], toolsets: Any):
    """Return a cached in-process AIAgent to avoid `hermes` process startup per wake."""
    key = json.dumps({
        "provider": provider,
        "model": model,
        "toolsets": toolsets,
        "max_iterations": int(cfg.get("hermes_max_iterations") or 8),
    }, sort_keys=True, default=str)
    cached = _HERMES_AGENT_CACHE.get(key)
    if cached is not None:
        return cached

    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.oneshot import _create_session_db_for_oneshot, _oneshot_clarify_callback
    from run_agent import AIAgent

    runtime = resolve_runtime_provider(requested=provider, target_model=model or None)
    agent = AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        api_mode=runtime.get("api_mode"),
        model=model or "",
        max_iterations=int(cfg.get("hermes_max_iterations") or 8),
        enabled_toolsets=toolsets,
        quiet_mode=True,
        platform="cli",
        session_db=_create_session_db_for_oneshot(),
        credential_pool=runtime.get("credential_pool"),
        clarify_callback=_oneshot_clarify_callback,
        skip_context_files=True,
        load_soul_identity=bool(cfg.get("hermes_load_soul_identity", True)),
    )
    agent.suppress_status_output = True
    agent.stream_delta_callback = None
    agent.tool_gen_callback = None
    _HERMES_AGENT_CACHE.clear()
    _HERMES_AGENT_CACHE[key] = agent
    LOG.info("Warm Hermes agent initialized provider=%s model=%s toolsets=%s", provider, model, toolsets)
    return agent


def ask_hermes_turn(
    cfg: Dict[str, Any],
    transcript: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    prompt_prefix = str(cfg.get("hermes_prompt_prefix") or "").strip()
    prompt = f"{prompt_prefix}\n\nTranscript:\n{transcript}" if prompt_prefix else transcript
    provider, model = configured_hermes_runtime_selection(cfg)
    toolsets = configured_hermes_toolsets(cfg)
    history = list(conversation_history or [])

    LOG.info(
        "Invoking Hermes for transcript (%d chars), mode=%s provider=%s model=%s toolsets=%s",
        len(transcript),
        "inprocess" if cfg.get("hermes_inprocess", True) else "subprocess",
        provider or "config",
        model or "config",
        toolsets if toolsets is not None else "config",
    )

    if cfg.get("hermes_inprocess", True):
        started = time.monotonic()
        try:
            if cfg.get("hermes_warm_agent", True):
                agent = get_warm_hermes_agent(cfg, provider, model, toolsets)
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        result = agent.run_conversation(
                            prompt,
                            conversation_history=history,
                            persist_user_message=transcript,
                        )
                        response = result.get("final_response") if isinstance(result, dict) else result
                        if isinstance(result, dict) and isinstance(result.get("messages"), list):
                            history = result["messages"]
            else:
                from hermes_cli.oneshot import _run_agent
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        response = _run_agent(
                            prompt,
                            model=model,
                            provider=provider,
                            toolsets=toolsets,
                            use_config_toolsets=toolsets is None,
                        )
            response = (response or "").strip()
            LOG.info("Hermes in-process latency: %.2fs", time.monotonic() - started)
            LOG.info("Hermes response: %s", response[:1000])
            return response or None, history
        except Exception as exc:
            LOG.exception("In-process Hermes failed; falling back to subprocess: %s", exc)

    hermes_bin = str(cfg["hermes_bin"])
    cmd = [hermes_bin, "chat", "-Q", "--source", str(cfg.get("hermes_source") or "wakeword")]
    if provider:
        cmd.extend(["--provider", provider])
    if model:
        cmd.extend(["-m", model])
    if toolsets:
        cmd.extend(["-t", ",".join(toolsets) if isinstance(toolsets, list) else str(toolsets)])
    cmd.extend(["-q", prompt])

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(cfg["hermes_timeout_seconds"]),
            cwd=str(Path.home()),
            check=False,
        )
    except subprocess.TimeoutExpired:
        LOG.error("Hermes command timed out")
        return "Hermes timed out while handling that request.", history
    except Exception as exc:
        LOG.exception("Failed to invoke Hermes: %s", exc)
        return f"I could not start Hermes: {exc}", history

    LOG.info("Hermes subprocess latency: %.2fs", time.monotonic() - started)
    if proc.stderr.strip():
        LOG.warning("Hermes stderr: %s", strip_ansi(proc.stderr).strip())
    response = clean_hermes_output(proc.stdout)
    if proc.returncode != 0:
        LOG.error("Hermes exited with %s; stdout=%r", proc.returncode, response)
        return response or f"Hermes exited with status {proc.returncode}.", history
    LOG.info("Hermes response: %s", response[:1000])
    if response:
        history.extend([
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": response},
        ])
    return response or None, history


def ask_hermes(cfg: Dict[str, Any], transcript: str) -> Optional[str]:
    """Backward-compatible single-turn wrapper."""
    response, _history = ask_hermes_turn(cfg, transcript)
    return response


def _configured_playback_sinks(cfg: Dict[str, Any]) -> List[str]:
    sink = str(cfg.get("playback_sink") or "@DEFAULT_SINK@").strip()
    if sink.lower() != "all":
        return [sink]
    try:
        proc = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            check=False,
        )
        sinks = []
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                sinks.append(parts[1])
        return sinks or ["@DEFAULT_SINK@"]
    except Exception as exc:
        LOG.warning("Could not enumerate sinks for playback_sink=all: %s", exc)
        return ["@DEFAULT_SINK@"]


def play_tts_file(cfg: Dict[str, Any], file_path: str) -> bool:
    """Play TTS through PipeWire/Pulse first, then fall back to Hermes playback."""
    sinks = _configured_playback_sinks(cfg)
    volume = max(0.0, min(float(cfg.get("playback_volume") or 1.0), 1.5))
    paplay_volume = str(int(min(volume, 1.0) * 65536))
    LOG.info("Playback sinks=%s volume=%.2f", sinks, volume)

    players: List[Tuple[str, List[str]]] = []
    for sink in sinks:
        players.append(("paplay", ["paplay", "-d", sink, "--volume", paplay_volume, file_path]))
    for sink in sinks:
        if sink == "@DEFAULT_SINK@":
            players.append(("pw-play", ["pw-play", "--volume", str(volume), file_path]))
        else:
            players.append(("pw-play", ["pw-play", "--target", sink, "--volume", str(volume), file_path]))

    success = False
    if len(sinks) > 1:
        procs = []
        for label, cmd in players[:len(sinks)]:  # paplay to all sinks concurrently
            try:
                LOG.info("Starting %s playback: %s", label, " ".join(cmd[:4]))
                procs.append((label, cmd, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
            except FileNotFoundError:
                LOG.warning("Playback command missing: %s", label)
            except Exception as exc:
                LOG.warning("Could not start %s: %s", label, exc)
        for label, cmd, proc in procs:
            try:
                out, err = proc.communicate(timeout=300)
                if proc.returncode == 0:
                    success = True
                else:
                    LOG.warning("%s playback exited %s: %s", label, proc.returncode, (err or out or "").strip())
            except subprocess.TimeoutExpired:
                proc.kill()
                LOG.warning("%s playback timed out", label)
        if success:
            return True

    for label, cmd in players:
        try:
            LOG.info("Trying %s playback: %s", label, " ".join(cmd[:5]))
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, check=False)
            if proc.returncode == 0:
                return True
            LOG.warning("%s playback exited %s: %s", label, proc.returncode, (proc.stderr or proc.stdout or "").strip())
        except FileNotFoundError:
            LOG.warning("Playback command missing: %s", label)
        except subprocess.TimeoutExpired:
            LOG.warning("%s playback timed out", label)
        except Exception as exc:
            LOG.warning("%s playback failed: %s", label, exc)

    LOG.info("Falling back to Hermes play_audio_file")
    return bool(play_audio_file(str(file_path)))


def speak_response(cfg: Dict[str, Any], text: str) -> None:
    if not cfg.get("tts_enabled", True):
        LOG.info("TTS disabled; response not spoken")
        return
    max_chars = int(cfg.get("max_spoken_response_chars") or 2500)
    spoken = text.strip()
    if len(spoken) > max_chars:
        spoken = spoken[:max_chars].rstrip() + "… The full response is in the wakeword log."
    LOG.info("Generating TTS (%d chars)", len(spoken))
    result_raw = text_to_speech_tool(spoken)
    try:
        result = json.loads(result_raw)
    except Exception:
        LOG.error("TTS returned non-JSON: %r", result_raw)
        return
    if not result.get("success"):
        LOG.error("TTS failed: %s", result.get("error") or result)
        return
    file_path = result.get("file_path")
    if not file_path:
        LOG.error("TTS response missing file_path: %s", result)
        return
    LOG.info("Playing TTS: %s", file_path)
    ok = play_tts_file(cfg, str(file_path))
    if not ok:
        LOG.error("Audio playback failed: %s", file_path)


def _visualization_state_path() -> Path:
    out_dir = Path(tempfile.gettempdir()) / "hermes_voice_wakeword"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return out_dir / f"voice_visual_{stamp}_{os.getpid()}_{time.monotonic_ns()}.json"


def update_visualization_state(path: Optional[Path], **updates: Any) -> None:
    """Atomically update the state consumed by the popup terminal visualizer."""
    if path is None:
        return
    try:
        state: Dict[str, Any] = {}
        if path.exists():
            with contextlib.suppress(Exception):
                state = json.loads(path.read_text(encoding="utf-8"))
        state.update(updates)
        state["updated_at"] = time.time()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        LOG.warning("Could not update visualization state %s: %s", path, exc)


def read_visualization_state(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        LOG.warning("Could not read visualization state %s: %s", path, exc)
        return {}


def is_visualization_cancel_requested(path: Optional[Path]) -> bool:
    return bool(read_visualization_state(path).get("cancel_requested"))


def visualization_cancel_reason(path: Optional[Path]) -> str:
    state = read_visualization_state(path)
    return str(state.get("cancel_reason") or "terminal_cancel")


def finish_cancelled_voice_session(
    visual_state: Optional[Path],
    activation_archive: Optional[Dict[str, Any]],
    archive_turns: List[Dict[str, Any]],
    reason: str,
) -> None:
    update_visualization_state(
        visual_state,
        status="cancelled",
        message="Voice session cancelled from the Hermes Voice terminal.",
        error="",
        cancel_requested=True,
        cancel_reason=reason,
    )
    update_activation_archive_metadata(
        activation_archive,
        status="cancelled_by_terminal",
        close_reason=reason,
        turns=archive_turns,
    )
    LOG.info("Voice conversation cancelled by terminal request: %s", reason)


def append_visualization_turn(path: Optional[Path], transcript: str, response: str) -> None:
    """Append a completed user/Hermes voice turn to the popup state."""
    if path is None:
        return
    try:
        state: Dict[str, Any] = {}
        if path.exists():
            with contextlib.suppress(Exception):
                state = json.loads(path.read_text(encoding="utf-8"))
        turns = state.get("turns")
        if not isinstance(turns, list):
            turns = []
        turns.append({
            "turn": len(turns) + 1,
            "transcript": transcript,
            "response": response,
            "completed_at": time.time(),
        })
        update_visualization_state(path, turns=turns, transcript=transcript, response=response)
    except Exception as exc:
        LOG.warning("Could not append visualization turn %s: %s", path, exc)


def _visualization_terminal_command(cfg: Dict[str, Any], state_path: Path) -> Optional[List[str]]:
    terminal_name = str(cfg.get("visualization_terminal") or "auto").strip()
    if terminal_name.lower() in {"", "off", "none", "false"}:
        return None

    title = str(cfg.get("visualization_title") or "Hermes Voice")
    script = Path(str(cfg.get("visualization_script") or "")).expanduser()
    if not script.exists():
        LOG.warning("Visualization script missing: %s", script)
        return None

    program = [sys.executable, str(script), "--state", str(state_path)]
    candidates = [terminal_name] if terminal_name.lower() != "auto" else [
        "kitty",
        "konsole",
        "alacritty",
        "wezterm",
        "foot",
        "gnome-terminal",
        "xterm",
    ]

    for candidate in candidates:
        exe = candidate if "/" in candidate else shutil.which(candidate)
        if not exe:
            continue
        name = Path(exe).name
        if name == "kitty":
            return [exe, "--detach", "--title", title, "--class", "hermes-voice", *program]
        if name == "konsole":
            return [exe, "--title", title, "-e", *program]
        if name == "alacritty":
            return [exe, "--title", title, "-e", *program]
        if name == "wezterm":
            return [exe, "start", "--", *program]
        if name == "foot":
            return [exe, "--title", title, *program]
        if name in {"gnome-terminal", "kgx", "xfce4-terminal"}:
            return [exe, "--title", title, "--", *program]
        if name == "xterm":
            return [exe, "-T", title, "-e", *program]
        return [exe, *program]

    LOG.warning("No supported terminal emulator found for visualization candidates=%s", candidates)
    return None


def launch_visualization(cfg: Dict[str, Any], probability: float) -> Optional[Path]:
    """Open a non-blocking terminal window for the current voice activation."""
    if not cfg.get("visualization_enabled", True):
        return None

    state_path = _visualization_state_path()
    update_visualization_state(
        state_path,
        title=str(cfg.get("visualization_title") or "Hermes Voice"),
        status="listening",
        message="Wakeword detected. Listening for your request…",
        probability=float(probability),
        activated_at=time.time(),
        keep_open_seconds=float(cfg.get("visualization_keep_open_seconds") or 45.0),
        transcript="",
        response="",
        turns=[],
        error="",
        cancel_requested=False,
        cancel_reason="",
    )

    cmd = _visualization_terminal_command(cfg, state_path)
    if not cmd:
        return state_path

    try:
        env = os.environ.copy()
        env.setdefault("HERMES_HOME", str(HERMES_HOME))
        env.setdefault("HERMES_REPO", str(HERMES_REPO))
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(Path.home()),
            env=env,
            start_new_session=True,
        )
        LOG.info("Launched voice visualization: %s state=%s", cmd[0], state_path)
    except Exception as exc:
        LOG.warning("Could not launch voice visualization: %s", exc)
    return state_path


def visualization_test(cfg: Dict[str, Any], transcript: str) -> int:
    setup_logging(cfg, verbose=True)
    state_path = launch_visualization(cfg, probability=1.0)
    if state_path is None:
        print(json.dumps({"ok": False, "error": "visualization disabled"}, indent=2))
        return 1
    time.sleep(0.8)
    update_visualization_state(
        state_path,
        status="thinking",
        message="Visualization smoke test. Hermes would now handle the request.",
        transcript=transcript,
    )
    time.sleep(1.0)
    append_visualization_turn(
        state_path,
        transcript=transcript,
        response="Popup rendering is working. The live daemon will show your real spoken request and response here.",
    )
    update_visualization_state(
        state_path,
        status="done",
        message="Visualization smoke test complete.",
    )
    print(json.dumps({"ok": True, "state_path": str(state_path)}, indent=2))
    return 0


def maybe_beep(cfg: Dict[str, Any], frequency: int = 880, count: int = 1) -> None:
    if not cfg.get("beep_enabled", True):
        return
    with contextlib.suppress(Exception):
        play_beep(frequency=frequency, duration=0.10, count=count)


def handle_activation(cfg: Dict[str, Any], activation: Any) -> None:
    if isinstance(activation, dict):
        probability = float(activation.get("probability") or 0.0)
    else:
        probability = float(activation)
        activation = {"probability": probability, "scores": [probability], "detected_at": time.time()}
    LOG.info("Handling wake activation; probability=%.6f", probability)
    activation_archive = save_activation_archive(cfg, activation)
    visual_state = launch_visualization(cfg, probability)
    if activation_archive:
        update_visualization_state(visual_state, activation_archive=activation_archive)
    maybe_beep(cfg, frequency=880, count=1)

    conversation_enabled = bool(cfg.get("conversation_mode_enabled", True))
    max_turns = max(1, int(cfg.get("conversation_max_turns") or 50))
    turn_index = 1
    archive_turns: List[Dict[str, Any]] = []
    hermes_history: List[Dict[str, Any]] = []

    def voice_cancel_requested() -> bool:
        return is_visualization_cancel_requested(visual_state)

    def stop_if_cancelled() -> bool:
        if not voice_cancel_requested():
            return False
        finish_cancelled_voice_session(
            visual_state,
            activation_archive,
            archive_turns,
            visualization_cancel_reason(visual_state),
        )
        return True

    while not STOP.is_set() and turn_index <= max_turns:
        first_turn = turn_index == 1
        listening_message = (
            "Wakeword detected. Listening for your request…"
            if first_turn
            else "Listening for a follow-up. Say “close” to end voice mode."
        )
        update_visualization_state(
            visual_state,
            status="listening",
            message=listening_message,
            current_turn=turn_index,
            error="",
        )
        update_activation_archive_metadata(
            activation_archive,
            status="listening" if first_turn else "listening_followup",
            current_turn=turn_index,
        )
        if not first_turn and cfg.get("conversation_followup_beep_enabled", False):
            maybe_beep(cfg, frequency=660, count=1)

        if stop_if_cancelled():
            return

        command_path = record_command(command_recording_config_for_turn(cfg, turn_index), cancel_check=voice_cancel_requested)
        if stop_if_cancelled():
            return
        if not command_path:
            if first_turn or not conversation_enabled:
                update_visualization_state(
                    visual_state,
                    status="error",
                    message="Wakeword detected, but no spoken request was captured.",
                    error="No speech heard before the start timeout.",
                )
                update_activation_archive_metadata(
                    activation_archive,
                    status="no_first_command",
                    close_reason="no_speech_after_wake",
                    turns=archive_turns,
                )
                maybe_beep(cfg, frequency=330, count=2)
                return
            update_visualization_state(
                visual_state,
                status="listening",
                message="I did not catch that. Still listening; say “close” to end voice mode.",
                error="No usable speech captured for the follow-up turn.",
            )
            update_activation_archive_metadata(
                activation_archive,
                status="followup_no_audio",
                current_turn=turn_index,
                turns=archive_turns,
            )
            maybe_beep(cfg, frequency=330, count=1)
            continue

        archived_command_path = archive_command_audio(cfg, activation_archive, command_path, turn_index)
        update_visualization_state(
            visual_state,
            status="transcribing",
            message="Speech captured. Transcribing now…",
            current_turn=turn_index,
        )
        update_activation_archive_metadata(
            activation_archive,
            status="transcribing",
            current_turn=turn_index,
            **command_audio_metadata_fields(archived_command_path, command_path, latest=True),
        )
        transcript = transcribe_command(command_path)
        if stop_if_cancelled():
            return
        if not transcript:
            if first_turn or not conversation_enabled:
                update_visualization_state(
                    visual_state,
                    status="error",
                    message="Speech was captured, but STT did not produce a usable request.",
                    error="Empty, failed, or hallucinated transcript.",
                )
                update_activation_archive_metadata(
                    activation_archive,
                    status="first_command_not_transcribed",
                    close_reason="stt_empty_or_failed",
                    **command_audio_metadata_fields(archived_command_path, command_path, latest=True),
                    turns=archive_turns,
                )
                maybe_beep(cfg, frequency=330, count=2)
                return
            update_visualization_state(
                visual_state,
                status="listening",
                message="I could not transcribe that follow-up. Still listening; say “close” to end voice mode.",
                error="Empty, failed, or hallucinated transcript.",
            )
            update_activation_archive_metadata(
                activation_archive,
                status="followup_not_transcribed",
                current_turn=turn_index,
                **command_audio_metadata_fields(archived_command_path, command_path, latest=True),
                turns=archive_turns,
            )
            maybe_beep(cfg, frequency=330, count=1)
            continue

        if conversation_enabled and is_close_transcript(transcript, cfg):
            ack = str(cfg.get("conversation_close_ack") or "").strip()
            archive_turns.append({
                "turn": turn_index,
                "transcript": transcript,
                "response": ack,
                **command_audio_metadata_fields(archived_command_path, command_path),
                "closed_session": True,
            })
            if ack:
                append_visualization_turn(visual_state, transcript=transcript, response=ack)
                update_visualization_state(
                    visual_state,
                    status="speaking",
                    message="Closing voice mode…",
                    transcript=transcript,
                    response=ack,
                    error="",
                )
                speak_response(cfg, ack)
            update_visualization_state(
                visual_state,
                status="done",
                message="Voice conversation closed.",
                transcript=transcript,
                response=ack,
                error="",
            )
            update_activation_archive_metadata(
                activation_archive,
                status="closed_by_voice_command",
                close_reason="close_phrase",
                latest_transcript=transcript,
                turns=archive_turns,
            )
            LOG.info("Voice conversation closed by transcript: %r", transcript)
            return

        update_visualization_state(
            visual_state,
            status="thinking",
            message="Request transcribed. Hermes is handling it now…",
            transcript=transcript,
            response="",
            error="",
            current_turn=turn_index,
        )
        update_activation_archive_metadata(
            activation_archive,
            status="thinking",
            current_turn=turn_index,
            latest_transcript=transcript,
            **command_audio_metadata_fields(archived_command_path, command_path, latest=True),
        )
        if stop_if_cancelled():
            return
        response, hermes_history = ask_hermes_turn(cfg, transcript, hermes_history)
        if stop_if_cancelled():
            return
        if response:
            archive_turns.append({
                "turn": turn_index,
                "transcript": transcript,
                "response": response,
                **command_audio_metadata_fields(archived_command_path, command_path),
            })
            update_activation_archive_metadata(
                activation_archive,
                status="speaking",
                latest_transcript=transcript,
                latest_response=response,
                turns=archive_turns,
            )
            append_visualization_turn(visual_state, transcript=transcript, response=response)
            update_visualization_state(
                visual_state,
                status="speaking",
                message=(
                    "Hermes responded. Speaking now; then I’ll keep listening…"
                    if conversation_enabled
                    else "Hermes responded. Generating and playing voice output…"
                ),
                transcript=transcript,
                response=response,
                error="",
                current_turn=turn_index,
            )
            if stop_if_cancelled():
                return
            speak_response(cfg, response)
            if stop_if_cancelled():
                return
            if not conversation_enabled:
                update_visualization_state(
                    visual_state,
                    status="done",
                    message="Voice request complete.",
                    transcript=transcript,
                    response=response,
                    error="",
                )
                update_activation_archive_metadata(
                    activation_archive,
                    status="completed",
                    close_reason="single_turn_complete",
                    turns=archive_turns,
                )
                return
            update_activation_archive_metadata(
                activation_archive,
                status="awaiting_followup",
                turns=archive_turns,
                current_turn=turn_index + 1,
            )
            turn_index += 1
            continue

        update_visualization_state(
            visual_state,
            status="error",
            message="Hermes did not return a response.",
            transcript=transcript,
            error="No response from Hermes.",
        )
        update_activation_archive_metadata(
            activation_archive,
            status="hermes_no_response",
            close_reason="no_response",
            latest_transcript=transcript,
            turns=archive_turns,
        )
        maybe_beep(cfg, frequency=330, count=2)
        return

    if not STOP.is_set():
        update_visualization_state(
            visual_state,
            status="done",
            message=f"Voice conversation hit the safety limit of {max_turns} turns.",
            error="Say the wakeword again to start a new voice session.",
        )
        update_activation_archive_metadata(
            activation_archive,
            status="max_turns_reached",
            close_reason="max_turns",
            turns=archive_turns,
        )


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


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Always-on Okay Hermes wakeword daemon")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to wakeword config YAML")
    parser.add_argument("--smoke-test", action="store_true", help="Load ONNX model and run zero-audio inference, then exit")
    parser.add_argument("--visualization-test", metavar="TEXT", help="Open the popup visualizer with a fake transcript, then exit")
    parser.add_argument("--list-devices", action="store_true", help="Print PortAudio devices, then exit")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(Path(args.config).expanduser())
    if args.list_devices:
        return list_devices()
    if args.smoke_test:
        return smoke_test(cfg)
    if args.visualization_test:
        return visualization_test(cfg, args.visualization_test)

    setup_logging(cfg, verbose=args.verbose)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    LOG.info("Starting Okay Hermes wakeword daemon")
    LOG.info("Config path: %s", args.config)
    session, input_name, output_name = model_session(cfg["model_path"])
    prewarm_stt(cfg)
    prewarm_hermes(cfg)

    while not STOP.is_set():
        try:
            activation = wait_for_wake(cfg, session, input_name, output_name)
            if STOP.is_set() or activation is None:
                break
            handle_activation(cfg, activation)
            cooldown = float(cfg.get("cooldown_seconds") or 2.5)
            if cooldown > 0:
                LOG.info("Cooldown %.1fs", cooldown)
                STOP.wait(cooldown)
        except KeyboardInterrupt:
            STOP.set()
        except Exception as exc:
            LOG.exception("Wakeword loop error: %s", exc)
            STOP.wait(5.0)

    LOG.info("Wakeword daemon stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
