#!/usr/bin/env python3
"""Always-on "Okay Hermes" wakeword daemon for Hermes Agent.

This module keeps the CLI entrypoint and activation orchestration while the
implementation helpers live in focused sibling modules.
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
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import yaml

from tools.tts_tool import text_to_speech_tool
from tools.voice_mode import is_whisper_hallucination, play_audio_file, play_beep, stop_playback, transcribe_recording

from .activation_archive import (
    _activation_timestamp,
    archive_command_audio,
    command_audio_metadata_fields,
    save_activation_archive,
    update_activation_archive_metadata,
)
from .activation_flow import handle_activation
from .audio_io import (
    _cancel_check_requested,
    float_waveform_to_int16,
    list_devices,
    model_session,
    prewarm_stt,
    record_command,
    rms_int16,
    run_wake_inference,
    smoke_test,
    transcribe_command,
    wait_for_wake,
    write_wav_int16,
    write_wav_int16_to_path,
)
from .daemon_config import (
    ANSI_RE,
    CONFIG_PATH,
    DEFAULT_CONFIG,
    HERMES_HOME,
    HERMES_REPO,
    LOG,
    STOP,
    _HERMES_AGENT_CACHE,
    deep_merge,
    load_config,
    setup_logging,
    signal_handler,
)
from .hermes_runtime import (
    _UseSubprocessForCancellation,
    _collect_hermes_process_output,
    _execution_cancel_requested,
    _hermes_cancel_poll_seconds,
    _hermes_interrupt_wait_seconds,
    _interrupt_hermes_agent,
    _run_hermes_subprocess_turn,
    _run_warm_hermes_agent_turn,
    _terminate_hermes_process_group,
    ask_hermes,
    ask_hermes_turn,
    clean_hermes_output,
    configured_hermes_runtime_selection,
    configured_hermes_toolsets,
    get_warm_hermes_agent,
    prewarm_hermes,
    strip_ansi,
)
from .interaction_router import (
    ACK_TEXT,
    AckTemplate,
    AcknowledgementCache,
    InteractionRouterConfig,
    RouteTarget,
    VoiceRequestPlan,
    answer_with_small_model,
    plan_voice_request,
)
from .playback import (
    _collect_playback_output,
    _configured_playback_sinks,
    _play_hermes_audio_file_with_cancel,
    _playback_cancel_requested,
    _terminate_playback_process,
    _wait_playback_process,
    _wait_playback_processes,
    maybe_beep,
    play_tts_file,
    speak_response,
)
from .visualization import (
    _visualization_command_for_terminal,
    _visualization_state_path,
    _visualization_terminal_candidates,
    _visualization_terminal_command,
    _visualization_terminal_commands,
    append_visualization_turn,
    finish_cancelled_voice_session,
    is_visualization_cancel_requested,
    launch_visualization,
    read_visualization_state,
    update_visualization_state,
    visualization_cancel_reason,
    visualization_test,
)
from .voice_routing import (
    _generate_ack_tts,
    _play_interaction_ack_sync,
    answer_routed_request,
    command_recording_config_for_turn,
    interaction_ack_text,
    interaction_router_config_from_daemon_config,
    is_close_transcript,
    normalize_voice_command,
    plan_interaction_route,
    play_interaction_ack,
    route_transcribed_request,
    routed_request_status_message,
)




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
