"""Write popup cancellation requests."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict


def request_cancel(path: Path, reason: str = "ctrl_c") -> None:
    """Request cancellation of the active daemon-owned voice session."""
    path = Path(path).expanduser()
    try:
        state: Dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    state = loaded
            except Exception:
                state = {}
        state.update({"status": "cancel_requested", "message": "Ctrl-C pressed in the Hermes Voice window. Stopping this voice session…", "cancel_requested": True, "cancel_reason": reason, "cancel_requested_at": time.time(), "error": ""})
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        print(f"Could not request voice-session cancellation: {exc}", file=sys.stderr)


__all__ = ["request_cancel"]
