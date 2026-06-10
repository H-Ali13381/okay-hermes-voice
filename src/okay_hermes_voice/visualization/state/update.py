"""Atomic popup visualization state writes."""
from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ...daemon_config import LOG


def update_visualization_state(path: Optional[Path], **updates: Any) -> None:
    """Atomically update the state consumed by the popup terminal visualizer."""
    if path is None:
        return
    try:
        state: Dict[str, Any] = {}
        if path.exists():
            with contextlib.suppress(Exception):
                state = json.loads(path.read_text(encoding="utf-8"))
        state.update(updates)
        state["updated_at"] = time.time()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        LOG.warning("Could not update visualization state %s: %s", path, exc)


__all__ = ["update_visualization_state"]
