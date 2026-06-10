"""Cancellable single-turn `hermes chat` subprocess execution."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..daemon_config import DEFAULT_CONFIG, LOG
from .cancellation import _execution_cancel_requested
from .config import _hermes_cancel_poll_seconds, _hermes_interrupt_wait_seconds
from .output import clean_hermes_output, strip_ansi
from .process_output import _collect_hermes_process_output
from .termination import _terminate_hermes_process_group


def _run_hermes_subprocess_turn(
    cfg: Dict[str, Any],
    cmd: List[str],
    transcript: str,
    history: List[Dict[str, Any]],
    cancel_check: Optional[Callable[[], bool]],
) -> Tuple[Optional[str], List[Dict[str, Any]], bool]:
    started = time.monotonic()
    try:
        proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(Path.home()), start_new_session=True)
    except Exception as exc:
        LOG.exception("Failed to invoke Hermes: %s", exc)
        return f"I could not start Hermes: {exc}", history, False

    timeout = float(cfg.get("hermes_timeout_seconds") or DEFAULT_CONFIG["hermes_timeout_seconds"])
    deadline = time.monotonic() + timeout
    poll_interval = _hermes_cancel_poll_seconds(cfg)
    while proc.poll() is None:
        if _execution_cancel_requested(cancel_check):
            LOG.info("Voice cancellation requested; terminating Hermes subprocess group")
            _terminate_hermes_process_group(proc, _hermes_interrupt_wait_seconds(cfg))
            _collect_hermes_process_output(proc)
            return None, history, True
        if time.monotonic() >= deadline:
            LOG.error("Hermes command timed out")
            _terminate_hermes_process_group(proc, _hermes_interrupt_wait_seconds(cfg))
            _collect_hermes_process_output(proc)
            return "Hermes timed out while handling that request.", history, False
        time.sleep(poll_interval)

    stdout, stderr = _collect_hermes_process_output(proc)
    LOG.info("Hermes subprocess latency: %.2fs", time.monotonic() - started)
    if stderr.strip():
        LOG.warning("Hermes stderr: %s", strip_ansi(stderr).strip())
    response = clean_hermes_output(stdout)
    if proc.returncode != 0:
        LOG.error("Hermes exited with %s; stdout=%r", proc.returncode, response)
        return response or f"Hermes exited with status {proc.returncode}.", history, False
    LOG.info("Hermes response: %s", response[:1000])
    if response:
        history.extend([
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": response},
        ])
    return response or None, history, False


__all__ = ["_run_hermes_subprocess_turn"]
