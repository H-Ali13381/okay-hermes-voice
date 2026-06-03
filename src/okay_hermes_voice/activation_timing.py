"""Timing schema helpers for wake-triggered voice sessions.

The daemon crosses two clock domains:
- the wake detector records ``detected_at`` with wall time before this handler is
  called;
- all work inside the handler should be measured with monotonic time.

Keeping that rule here prevents each activation-flow branch from re-explaining
why some fields are wall-clock deltas and others are monotonic durations.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from .activation_archive import update_activation_archive_metadata
from .visualization import update_visualization_state

TIMING_SCHEMA_VERSION = 1
SPEAK_TIMING_KEYS = (
    "tts_enabled",
    "tts_success",
    "playback_success",
    "tts_seconds",
    "playback_seconds",
    "speak_seconds",
    "tts_file_path",
)


def build_voice_session_timing(activation_detected_at: float, handle_started_at: float) -> Dict[str, float | int]:
    """Build session-level wake latency metadata from detector wall-clock values."""
    return {
        "schema_version": TIMING_SCHEMA_VERSION,
        "activation_detected_at": activation_detected_at,
        "handle_started_at": handle_started_at,
        "wake_to_handle_seconds": max(0.0, handle_started_at - activation_detected_at),
    }


def elapsed_seconds(started: float) -> float:
    """Return a monotonic duration for in-process voice pipeline phases."""
    return max(0.0, time.monotonic() - started)


def wake_to_record_start_seconds(activation_detected_at: float, record_wall_started: float) -> float:
    """Measure detector-to-microphone-start latency across the wall-clock boundary.

    This is intentionally not monotonic: the start value comes from wake detector
    metadata already recorded as ``time.time()``. Once recording begins, the rest
    of the turn uses monotonic deltas.
    """
    return max(0.0, record_wall_started - activation_detected_at)


def merge_speak_timing(turn_timing: Dict[str, Any], speak_result: Any, fallback_seconds: float) -> None:
    """Merge TTS/playback timing into a completed turn record.

    ``speak_response`` is allowed to be ignored by older callers and tests. If a
    test double still returns ``None``, keep the coarse wrapper measurement so the
    archive still shows where time went.
    """
    turn_timing["speak_seconds"] = fallback_seconds
    if not isinstance(speak_result, dict):
        return
    for key in SPEAK_TIMING_KEYS:
        if key in speak_result:
            turn_timing[key] = speak_result[key]


def publish_turn_timing(
    visual_state: Any,
    activation_archive: Any,
    turn_timings: List[Dict[str, Any]],
    turn_timing: Dict[str, Any],
    archive_turns: List[Dict[str, Any]],
) -> None:
    """Publish one completed turn timing snapshot to live state and archive JSON.

    The copy matters: the same dict is assembled over several branches inside
    ``handle_activation``. Publishing a snapshot avoids accidental later mutation
    of the state file or the archive's per-turn record.
    """
    stable_timing = dict(turn_timing)
    turn_timings.append(stable_timing)
    update_visualization_state(
        visual_state,
        latest_turn_timing=stable_timing,
        turn_timings=turn_timings,
    )
    update_activation_archive_metadata(
        activation_archive,
        latest_turn_timing=stable_timing,
        turn_timings=turn_timings,
        turns=archive_turns,
    )
