"""Append completed turns to popup visualization state."""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ...daemon_config import LOG
from .update import update_visualization_state


def append_visualization_turn(path: Optional[Path], transcript: str, response: str) -> None:
    """Append a completed user/Hermes voice turn to the popup state."""
    if path is None:
        return
    try:
        state: Dict[str, Any] = {}
        if path.exists():
            with contextlib.suppress(Exception):
                state = json.loads(path.read_text(encoding="utf-8"))
        turns = state.get("turns")
        if not isinstance(turns, list):
            turns = []
        turns.append({"turn": len(turns) + 1, "transcript": transcript, "response": response, "completed_at": time.time()})
        update_visualization_state(path, turns=turns, transcript=transcript, response=response)
    except Exception as exc:
        LOG.warning("Could not append visualization turn %s: %s", path, exc)


__all__ = ["append_visualization_turn"]
