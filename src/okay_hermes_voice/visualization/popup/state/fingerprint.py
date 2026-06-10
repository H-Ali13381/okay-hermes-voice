"""Render-state fingerprinting for redraw decisions."""
from __future__ import annotations

import json
from typing import Any, Dict

from ..constants import RENDER_FINGERPRINT_IGNORED_KEYS


def render_fingerprint_state(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: render_fingerprint_state(item) for key, item in value.items() if key not in RENDER_FINGERPRINT_IGNORED_KEYS}
    if isinstance(value, list):
        return [render_fingerprint_state(item) for item in value]
    return value


def state_fingerprint(state: Dict[str, Any]) -> str:
    """Stable identity for deciding whether a popup frame needs repainting."""
    try:
        return json.dumps(render_fingerprint_state(state), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return repr(render_fingerprint_state(state))


__all__ = ["render_fingerprint_state", "state_fingerprint"]
