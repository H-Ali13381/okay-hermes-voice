"""Popup state-file reads, cancellation writes, and render fingerprints."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

from .constants import RENDER_FINGERPRINT_IGNORED_KEYS


def load_state(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "listening", "message": "Waiting for wakeword state…"}
    except Exception as exc:
        return {"status": "error", "error": f"Could not read state file: {exc}"}


def request_cancel(path: Path, reason: str = "ctrl_c") -> None:
    """Request cancellation of the active daemon-owned voice session.

    The popup is only a visualizer, so Ctrl-C cannot signal the systemd daemon
    directly through process ancestry. Instead it atomically marks the shared
    state file; the daemon polls that file while recording/following up.
    """
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
        state.update(
            {
                "status": "cancel_requested",
                "message": "Ctrl-C pressed in the Hermes Voice window. Stopping this voice session…",
                "cancel_requested": True,
                "cancel_reason": reason,
                "cancel_requested_at": time.time(),
                "error": "",
            }
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        print(f"Could not request voice-session cancellation: {exc}", file=sys.stderr)


def render_fingerprint_state(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: render_fingerprint_state(item)
            for key, item in value.items()
            if key not in RENDER_FINGERPRINT_IGNORED_KEYS
        }
    if isinstance(value, list):
        return [render_fingerprint_state(item) for item in value]
    return value


def state_fingerprint(state: Dict[str, Any]) -> str:
    """Stable identity for deciding whether a popup frame needs repainting."""
    try:
        return json.dumps(
            render_fingerprint_state(state),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        return repr(render_fingerprint_state(state))


__all__ = ["load_state", "render_fingerprint_state", "request_cancel", "state_fingerprint"]
