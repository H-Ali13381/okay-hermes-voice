"""Configuration and shared daemon state for Okay Hermes voice."""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

import yaml

HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
HERMES_REPO = Path(os.environ.get("HERMES_REPO", str(HERMES_HOME / "hermes-agent"))).expanduser()
if str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))
CONFIG_PATH = HERMES_HOME / "wakeword" / "config.yaml"
DEFAULT_CONFIG: Dict[str, Any] = {
    "model_path": str(HERMES_HOME / "wakeword" / "okay-hermes-repcnn-onnx" / "wakeword.onnx"),
    "threshold": 0.6973556280136108,
    "trigger_consecutive_windows": 2,
    "inference_interval_seconds": 0.25,
    "cooldown_seconds": 2.5,
    "cancel_cooldown_seconds": 0.0,
    "sample_rate": 16000,
    "window_seconds": 3.0,
    "block_seconds": 0.1,
    "wake_audio_backend": "sounddevice",
    "wake_audio_device": "default",
    "native_listener_bin": str(HERMES_HOME / "wakeword" / "bin" / "okay-hermes-wake-listener"),
    "native_pipewire_target": "",
    "native_listener_verbose": False,
    "native_activation_server_enabled": False,
    "native_activation_socket": str(HERMES_HOME / "wakeword" / "native-handler.sock"),
    "native_activation_server_timeout_seconds": 5.0,
    "speech_rms_threshold": 200,
    "speech_start_consecutive_blocks": 2,
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
    "hermes_cancel_poll_seconds": 0.1,
    "hermes_interrupt_wait_seconds": 5.0,
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
    "interaction_router_enabled": True,
    "interaction_router_provider": "openrouter",
    "interaction_router_model": "google/gemini-2.5-flash-lite",
    "interaction_router_timeout_seconds": 1.5,
    "interaction_router_min_confidence": 0.70,
    "interaction_router_small_model_enabled": False,
    "interaction_router_small_model_provider": "openrouter",
    "interaction_router_small_model_model": "google/gemini-2.5-flash-lite",
    "interaction_router_ack_cache_enabled": True,
    "interaction_router_ack_cache_dir": str(HERMES_HOME / "wakeword" / "ack_cache"),
    "tts_enabled": True,
    "max_spoken_response_chars": 2500,
    "playback_sink": "@DEFAULT_SINK@",
    "playback_volume": 1.0,
    "beep_enabled": True,
    "transcript_only_mode": False,
    # STT provider for captured commands. "hermes" preserves the normal Hermes
    # transcription stack; "nemotron_en_streaming" uses NVIDIA's cache-aware
    # English-only Nemotron streaming ASR as an optional local provider.
    "stt_provider": "hermes",
    "nemotron_model_name": "nvidia/nemotron-speech-streaming-en-0.6b",
    "nemotron_model_path": "",
    "nemotron_device": "auto",
    "nemotron_att_context_size": [70, 13],
    "nemotron_amp": True,
    "nemotron_amp_dtype": "float16",
    "nemotron_cudnn_enabled": False,
    "nemotron_live_streaming": True,
    "nemotron_online_normalization": False,
    "nemotron_pad_and_drop_preencoded": False,
    "parakeet_model_name": "nvidia/parakeet-unified-en-0.6b",
    "parakeet_model_path": "",
    "parakeet_device": "auto",
    "parakeet_left_context_secs": 2.0,
    "parakeet_chunk_secs": 0.56,
    "parakeet_right_context_secs": 0.56,
    "parakeet_amp": True,
    "parakeet_amp_dtype": "float16",
    "parakeet_cudnn_enabled": False,
    "parakeet_live_streaming": True,
    "prewarm_stt_on_start": True,
    "prewarm_hermes_on_start": True,
    "visualization_enabled": True,
    "visualization_terminal": "auto",
    "visualization_title": "Hermes Voice",
    "visualization_keep_open_seconds": 45.0,
    "visualization_launch_grace_seconds": 0.35,
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
