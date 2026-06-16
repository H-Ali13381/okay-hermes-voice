"""Select the first usable popup terminal command."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .commands import _visualization_terminal_commands


def _visualization_terminal_command(cfg: Dict[str, Any], state_path: Path) -> Optional[List[str]]:
    commands = _visualization_terminal_commands(cfg, state_path)
    return commands[0] if commands else None


__all__ = ["_visualization_terminal_command"]
