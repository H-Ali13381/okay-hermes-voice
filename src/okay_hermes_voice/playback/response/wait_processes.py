"""Wait for concurrent playback subprocesses."""
from __future__ import annotations

import subprocess
import time
from typing import Any, Callable, List, Optional, Tuple

from ...daemon_config import LOG
from .cancellation import _playback_cancel_requested
from .output import _collect_playback_output
from .termination import _terminate_playback_process


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


__all__ = ["_wait_playback_processes"]
