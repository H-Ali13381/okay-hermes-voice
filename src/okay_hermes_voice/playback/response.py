"""TTS generation and cancellable audio playback helpers."""
from __future__ import annotations

import contextlib
import json
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.tts_tool import text_to_speech_tool
from tools.voice_mode import play_audio_file, play_beep, stop_playback

from ..daemon_config import LOG, STOP

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


def _playback_cancel_requested(cancel_check: Optional[Callable[[], bool]]) -> bool:
    if STOP.is_set():
        return True
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception as exc:
        LOG.warning("Playback cancel check failed: %s", exc)
        return False


def _terminate_playback_process(label: str, proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    with contextlib.suppress(Exception):
        proc.terminate()
    try:
        proc.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        LOG.info("%s playback did not stop after terminate; killing", label)
    except Exception:
        return
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=1.0)


def _collect_playback_output(proc: subprocess.Popen[Any]) -> Tuple[str, str]:
    try:
        out, err = proc.communicate(timeout=0.2)
        return str(out or ""), str(err or "")
    except Exception:
        return str(getattr(proc, "stdout", "") or ""), str(getattr(proc, "stderr", "") or "")


def _wait_playback_process(
    label: str,
    proc: subprocess.Popen[Any],
    cancel_check: Optional[Callable[[], bool]],
    timeout: float = 300.0,
) -> Tuple[bool, bool]:
    deadline = time.monotonic() + timeout
    while proc.poll() is None:
        if _playback_cancel_requested(cancel_check):
            _terminate_playback_process(label, proc)
            LOG.info("%s playback cancelled", label)
            return False, True
        if time.monotonic() >= deadline:
            _terminate_playback_process(label, proc)
            LOG.warning("%s playback timed out", label)
            return False, False
        time.sleep(0.05)

    out, err = _collect_playback_output(proc)
    if proc.returncode == 0:
        return True, False
    LOG.warning("%s playback exited %s: %s", label, proc.returncode, (err or out or "").strip())
    return False, False


def _wait_playback_processes(
    procs: List[Tuple[str, List[str], subprocess.Popen[Any]]],
    cancel_check: Optional[Callable[[], bool]],
    timeout: float = 300.0,
) -> Tuple[bool, bool]:
    deadline = time.monotonic() + timeout
    pending = list(procs)
    success = False
    while pending:
        if _playback_cancel_requested(cancel_check):
            for label, _cmd, proc in pending:
                _terminate_playback_process(label, proc)
            LOG.info("Concurrent playback cancelled")
            return False, True

        for item in list(pending):
            label, _cmd, proc = item
            if proc.poll() is None:
                continue
            out, err = _collect_playback_output(proc)
            if proc.returncode == 0:
                success = True
            else:
                LOG.warning("%s playback exited %s: %s", label, proc.returncode, (err or out or "").strip())
            pending.remove(item)

        if pending and time.monotonic() >= deadline:
            for label, _cmd, proc in pending:
                _terminate_playback_process(label, proc)
            LOG.warning("Concurrent playback timed out")
            return success, False
        if pending:
            time.sleep(0.05)
    return success, False


def _play_hermes_audio_file_with_cancel(file_path: str, cancel_check: Optional[Callable[[], bool]]) -> bool:
    if cancel_check is None:
        return bool(play_audio_file(str(file_path)))

    done = threading.Event()
    result: Dict[str, Any] = {"ok": False}

    def _run() -> None:
        try:
            result["ok"] = bool(play_audio_file(str(file_path)))
        except Exception as exc:
            result["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    while not done.wait(0.05):
        if _playback_cancel_requested(cancel_check):
            with contextlib.suppress(Exception):
                stop_playback()
            done.wait(2.0)
            LOG.info("Hermes fallback playback cancelled")
            return False
    if result.get("error"):
        LOG.warning("Hermes fallback playback failed: %s", result["error"])
        return False
    return bool(result.get("ok"))


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
        for label, cmd in players[:len(sinks)]:  # paplay to all sinks concurrently
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


def speak_response(
    cfg: Dict[str, Any],
    text: str,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    stage_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    started = time.monotonic()
    timing: Dict[str, Any] = {
        "tts_enabled": bool(cfg.get("tts_enabled", True)),
        "tts_success": False,
        "playback_success": False,
        "tts_seconds": 0.0,
        "playback_seconds": 0.0,
        "speak_seconds": 0.0,
        "tts_file_path": "",
    }

    def finish() -> Dict[str, Any]:
        timing["speak_seconds"] = max(0.0, time.monotonic() - started)
        return timing

    def notify_stage(stage: str) -> None:
        if stage_callback is None:
            return
        try:
            stage_callback(stage)
        except Exception as exc:
            LOG.warning("Voice popup stage callback failed for %s: %s", stage, exc)

    if not timing["tts_enabled"]:
        LOG.info("TTS disabled; response not spoken")
        return finish()
    max_chars = int(cfg.get("max_spoken_response_chars") or 2500)
    spoken = text.strip()
    if len(spoken) > max_chars:
        spoken = spoken[:max_chars].rstrip() + "… The full response is in the wakeword log."
    LOG.info("Generating TTS (%d chars)", len(spoken))
    notify_stage("tts")
    tts_started = time.monotonic()
    result_raw = text_to_speech_tool(spoken)
    timing["tts_seconds"] = max(0.0, time.monotonic() - tts_started)
    try:
        result = json.loads(result_raw)
    except Exception:
        LOG.error("TTS returned non-JSON: %r", result_raw)
        return finish()
    if not result.get("success"):
        LOG.error("TTS failed: %s", result.get("error") or result)
        return finish()
    timing["tts_success"] = True
    file_path = result.get("file_path")
    if not file_path:
        LOG.error("TTS response missing file_path: %s", result)
        return finish()
    timing["tts_file_path"] = str(file_path)
    LOG.info("Playing TTS: %s", file_path)
    notify_stage("playback")
    playback_started = time.monotonic()
    ok = play_tts_file(cfg, str(file_path), cancel_check=cancel_check)
    timing["playback_seconds"] = max(0.0, time.monotonic() - playback_started)
    timing["playback_success"] = bool(ok)
    if not ok and not _playback_cancel_requested(cancel_check):
        LOG.error("Audio playback failed: %s", file_path)
    return finish()


def maybe_beep(cfg: Dict[str, Any], frequency: int = 880, count: int = 1) -> None:
    if not cfg.get("beep_enabled", True):
        return
    with contextlib.suppress(Exception):
        play_beep(frequency=frequency, duration=0.10, count=count)
