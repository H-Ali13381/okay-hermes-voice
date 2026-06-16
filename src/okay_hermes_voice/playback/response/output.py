"""Playback subprocess output collection."""
from __future__ import annotations

import subprocess
from typing import Any, Tuple


def _collect_playback_output(proc: subprocess.Popen[Any]) -> Tuple[str, str]:
    try:
        out, err = proc.communicate(timeout=0.2)
        return str(out or ""), str(err or "")
    except Exception:
        return str(getattr(proc, "stdout", "") or ""), str(getattr(proc, "stderr", "") or "")


__all__ = ["_collect_playback_output"]
