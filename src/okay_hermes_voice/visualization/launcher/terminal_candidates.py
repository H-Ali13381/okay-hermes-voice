"""Terminal-emulator candidate selection."""
from __future__ import annotations

from typing import List


def _visualization_terminal_candidates(terminal_name: str) -> List[str]:
    if terminal_name.lower() != "auto":
        return [terminal_name]
    return ["kitty", "konsole", "alacritty", "wezterm", "foot", "gnome-terminal", "kgx", "xfce4-terminal", "xterm"]


__all__ = ["_visualization_terminal_candidates"]
