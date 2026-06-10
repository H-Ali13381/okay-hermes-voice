"""Background reaping for accepted popup terminal processes."""
from __future__ import annotations

import subprocess
import threading
from typing import Any

from ...daemon_config import LOG
from .process_comm import _communicate_or_wait
from .process_output import _short_process_output


def _reap_visualization_process(proc: subprocess.Popen[Any], terminal_label: str) -> None:
    """Wait for an accepted popup terminal process so it cannot become a zombie."""
    try:
        stdout, stderr, returncode = _communicate_or_wait(proc)
        if returncode not in {0, None}:
            LOG.debug("Voice visualization terminal %s exited with status %s: %s", terminal_label, returncode, _short_process_output(stderr) or _short_process_output(stdout))
        else:
            LOG.debug("Voice visualization terminal %s exited with status %s", terminal_label, returncode)
    except Exception as exc:
        LOG.debug("Could not reap voice visualization terminal %s: %s", terminal_label, exc)


def _watch_visualization_process(proc: subprocess.Popen[Any], terminal_label: str) -> None:
    thread = threading.Thread(target=_reap_visualization_process, args=(proc, terminal_label), name=f"hermes-voice-{terminal_label}-reaper", daemon=True)
    thread.start()


__all__ = ["_reap_visualization_process", "_watch_visualization_process"]
