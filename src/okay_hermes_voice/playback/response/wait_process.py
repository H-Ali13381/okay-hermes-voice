"""Wait for one playback subprocess with cancellation."""
from __future__ import annotations

import subprocess
import time
from typing import Any, Callable, Optional, Tuple

from ...daemon_config import LOG
from .cancellation import _playback_cancel_requested
from .output import _collect_playback_output
from .termination import _terminate_playback_process


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


__all__ = ["_wait_playback_process"]
