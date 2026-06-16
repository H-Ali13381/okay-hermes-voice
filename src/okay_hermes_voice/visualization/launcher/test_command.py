"""Visualization launcher smoke-test command."""
from __future__ import annotations

import json
import time
from typing import Any, Dict

from ...daemon_config import setup_logging
from ..state import append_visualization_turn, update_visualization_state
from .launch import launch_visualization


def visualization_test(cfg: Dict[str, Any], transcript: str) -> int:
    setup_logging(cfg, verbose=True)
    state_path = launch_visualization(cfg, probability=1.0)
    if state_path is None:
        print(json.dumps({"ok": False, "error": "visualization disabled"}, indent=2))
        return 1
    time.sleep(0.8)
    update_visualization_state(state_path, status="thinking", message="Visualization smoke test. Hermes would now handle the request.", transcript=transcript)
    time.sleep(1.0)
    append_visualization_turn(state_path, transcript=transcript, response="Popup rendering is working. The live daemon will show your real spoken request and response here.")
    update_visualization_state(state_path, status="done", message="Visualization smoke test complete.")
    print(json.dumps({"ok": True, "state_path": str(state_path)}, indent=2))
    return 0


__all__ = ["visualization_test"]
