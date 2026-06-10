"""Safe collection of subprocess stdout/stderr."""
from __future__ import annotations

import subprocess
from typing import Any, Tuple


def _collect_hermes_process_output(proc: subprocess.Popen[Any]) -> Tuple[str, str]:
    try:
        out, err = proc.communicate(timeout=0.2)
    except TypeError:
        try:
            out, err = proc.communicate()
        except Exception:
            out, err = "", ""
    except Exception:
        out, err = getattr(proc, "stdout", "") or "", getattr(proc, "stderr", "") or ""
    return str(out or ""), str(err or "")


__all__ = ["_collect_hermes_process_output"]
