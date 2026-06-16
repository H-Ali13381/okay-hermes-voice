"""Playback process termination."""
from __future__ import annotations

import contextlib
import subprocess
from typing import Any

from ...daemon_config import LOG


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


__all__ = ["_terminate_playback_process"]
