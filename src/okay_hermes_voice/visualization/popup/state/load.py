"""Load popup JSON state files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_state(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "listening", "message": "Waiting for wakeword state…"}
    except Exception as exc:
        return {"status": "error", "error": f"Could not read state file: {exc}"}


__all__ = ["load_state"]
