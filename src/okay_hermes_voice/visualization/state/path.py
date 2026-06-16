"""Popup visualization state-file path creation."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def _visualization_state_path() -> Path:
    out_dir = Path(tempfile.gettempdir()) / "hermes_voice_wakeword"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return out_dir / f"voice_visual_{stamp}_{os.getpid()}_{time.monotonic_ns()}.json"


__all__ = ["_visualization_state_path"]
