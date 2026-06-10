"""Public TTS-file playback implementation."""
from __future__ import annotations

import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...daemon_config import LOG
from .cancellation import _playback_cancel_requested
from .fallback import _play_hermes_audio_file_with_cancel
from .sinks import _configured_playback_sinks
from .wait_process import _wait_playback_process
from .wait_processes import _wait_playback_processes


def play_tts_file(
    cfg: Dict[str, Any],
    file_path: str,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> bool:
    """Play TTS through PipeWire/Pulse first, then fall back to Hermes playback."""
    if _playback_cancel_requested(cancel_check):
        LOG.info("Playback skipped because cancellation is already requested")
        return False
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
        for label, cmd in players[:len(sinks)]:
            try:
                LOG.info("Starting %s playback: %s", label, " ".join(cmd[:4]))
                procs.append((label, cmd, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
            except FileNotFoundError:
                LOG.warning("Playback command missing: %s", label)
            except Exception as exc:
                LOG.warning("Could not start %s: %s", label, exc)
        success, cancelled = _wait_playback_processes(procs, cancel_check)
        if cancelled:
            return False
        if success:
            return True

    for label, cmd in players:
        try:
            if _playback_cancel_requested(cancel_check):
                LOG.info("Playback cancelled before starting %s", label)
                return False
            LOG.info("Trying %s playback: %s", label, " ".join(cmd[:5]))
            proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ok, cancelled = _wait_playback_process(label, proc, cancel_check)
            if cancelled:
                return False
            if ok:
                return True
        except FileNotFoundError:
            LOG.warning("Playback command missing: %s", label)
        except Exception as exc:
            LOG.warning("%s playback failed: %s", label, exc)

    if _playback_cancel_requested(cancel_check):
        LOG.info("Playback cancelled before Hermes fallback")
        return False
    LOG.info("Falling back to Hermes play_audio_file")
    return _play_hermes_audio_file_with_cancel(str(file_path), cancel_check)


__all__ = ["play_tts_file"]
