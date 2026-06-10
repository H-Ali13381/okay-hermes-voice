"""Popup terminal launch command list assembly."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

from ...daemon_config import LOG
from .command import _visualization_command_for_terminal
from .terminal_candidates import _visualization_terminal_candidates


def _visualization_terminal_commands(cfg: Dict[str, Any], state_path: Path) -> List[List[str]]:
    terminal_name = str(cfg.get("visualization_terminal") or "auto").strip()
    if terminal_name.lower() in {"", "off", "none", "false"}:
        return []

    title = str(cfg.get("visualization_title") or "Hermes Voice")
    script = Path(str(cfg.get("visualization_script") or "")).expanduser()
    if not script.exists():
        LOG.warning("Visualization script missing: %s", script)
        return []

    program = [sys.executable, str(script), "--state", str(state_path)]
    candidates = _visualization_terminal_candidates(terminal_name)
    commands: List[List[str]] = []
    for candidate in candidates:
        exe = candidate if "/" in candidate else shutil.which(candidate)
        if not exe:
            continue
        commands.append(_visualization_command_for_terminal(exe, title, program))

    if not commands:
        LOG.warning("No supported terminal emulator found for visualization candidates=%s", candidates)
    return commands


__all__ = ["_visualization_terminal_commands"]
