"""Launch the popup terminal visualizer."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ...daemon_config import LOG
from .. import state
from ..state import update_visualization_state
from .commands import _visualization_terminal_commands
from .constants import VISUALIZATION_LAUNCH_GRACE_SECONDS
from .environment import _visualization_launch_env
from .process_comm import _communicate_or_wait
from .process_output import _visualization_failure_message
from .process_watch import _watch_visualization_process


def launch_visualization(
    cfg: Dict[str, Any],
    probability: float,
    *,
    on_process_started: Optional[Callable[[], None]] = None,
) -> Optional[Path]:
    """Open a non-blocking terminal window for the current voice activation."""
    if not cfg.get("visualization_enabled", True):
        return None

    state_path = state._visualization_state_path()
    update_visualization_state(
        state_path,
        title=str(cfg.get("visualization_title") or "Hermes Voice"),
        status="wake",
        pipeline_stage="wake",
        message="wake: wakeword detected. Preparing to record your request…",
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

    commands = _visualization_terminal_commands(cfg, state_path)
    if not commands:
        return state_path

    env = _visualization_launch_env(os.environ.copy())
    launch_grace_seconds = float(cfg.get("visualization_launch_grace_seconds") or VISUALIZATION_LAUNCH_GRACE_SECONDS)
    last_error = ""
    process_started_notified = False

    def notify_process_started() -> None:
        nonlocal process_started_notified
        if process_started_notified or on_process_started is None:
            return
        process_started_notified = True
        on_process_started()

    for cmd in commands:
        terminal_label = Path(cmd[0]).name
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(Path.home()), env=env, start_new_session=True, text=True)
            notify_process_started()
            try:
                stdout, stderr, returncode = _communicate_or_wait(proc, timeout=launch_grace_seconds)
            except (subprocess.TimeoutExpired, TimeoutError):
                try:
                    returncode = proc.wait(timeout=0)
                except (subprocess.TimeoutExpired, TimeoutError):
                    update_visualization_state(state_path, visualization_terminal=terminal_label, visualization_launch_error="")
                    LOG.info("Launched voice visualization: %s state=%s", cmd[0], state_path)
                    _watch_visualization_process(proc, terminal_label)
                    return state_path
                stdout, stderr = "", ""
                try:
                    stdout, stderr, _ = _communicate_or_wait(proc)
                except Exception:
                    pass
                last_error = _visualization_failure_message(terminal_label, returncode, stdout, stderr)
                update_visualization_state(state_path, visualization_launch_error=last_error)
                LOG.warning("Voice visualization launch failed after grace timeout: %s", last_error)
                continue
            if returncode not in {0, None}:
                last_error = _visualization_failure_message(terminal_label, returncode, stdout, stderr)
                update_visualization_state(state_path, visualization_launch_error=last_error)
                LOG.warning("Voice visualization launch failed: %s", last_error)
                continue
            update_visualization_state(state_path, visualization_terminal=terminal_label, visualization_launch_error="")
            LOG.info("Launched voice visualization: %s state=%s", cmd[0], state_path)
            return state_path
        except Exception as exc:
            last_error = f"{terminal_label}: {exc}"
            update_visualization_state(state_path, visualization_launch_error=last_error)
            LOG.warning("Could not launch voice visualization with %s: %s", terminal_label, exc)

    if last_error:
        LOG.warning("All voice visualization terminal launch attempts failed: %s", last_error)
    return state_path


__all__ = ["launch_visualization"]
