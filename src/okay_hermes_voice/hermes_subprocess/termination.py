"""Process-group termination for Hermes subprocess calls."""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from typing import Any

from ..daemon_config import LOG


def _terminate_hermes_process_group(proc: subprocess.Popen[Any], grace_seconds: float) -> None:
    """Terminate a `hermes chat` process group so child tool subprocesses die too."""
    if proc.poll() is not None:
        return
    pid = int(getattr(proc, "pid", 0) or 0)
    if pid:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception as exc:
            LOG.warning("Could not SIGTERM Hermes process group %s: %s; trying process terminate", pid, exc)
            with contextlib.suppress(Exception):
                proc.terminate()
    else:
        with contextlib.suppress(Exception):
            proc.terminate()
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        LOG.warning("Hermes process group did not stop after %.1fs; sending SIGKILL", grace_seconds)
    except Exception:
        return
    if pid:
        with contextlib.suppress(Exception):
            os.killpg(pid, signal.SIGKILL)
    else:
        with contextlib.suppress(Exception):
            proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=1.0)


__all__ = ["_terminate_hermes_process_group"]
