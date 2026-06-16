"""Subprocess communication compatibility wrapper."""
from __future__ import annotations

import subprocess
from typing import Any, Optional


def _communicate_or_wait(proc: subprocess.Popen[Any], timeout: Optional[float] = None) -> tuple[str, str, Optional[int]]:
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return str(stdout or ""), str(stderr or ""), proc.returncode
    except AttributeError:
        try:
            returncode = proc.wait(timeout=timeout)
        except TypeError:
            returncode = proc.wait()
        return "", "", returncode


__all__ = ["_communicate_or_wait"]
