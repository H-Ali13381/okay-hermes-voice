"""Popup visualization state reads."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ...daemon_config import LOG


def read_visualization_state(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        LOG.warning("Could not read visualization state %s: %s", path, exc)
        return {}


__all__ = ["read_visualization_state"]
