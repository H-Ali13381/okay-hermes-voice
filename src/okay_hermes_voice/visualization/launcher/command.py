"""Terminal-specific command construction."""
from __future__ import annotations

from pathlib import Path
from typing import List


def _visualization_command_for_terminal(exe: str, title: str, program: List[str]) -> List[str]:
    name = Path(exe).name
    if name == "kitty":
        return [exe, "--title", title, "--class", "hermes-voice", "--override", "scrollback_lines=0", *program]
    if name == "konsole":
        return [exe, "--title", title, "-e", *program]
    if name == "alacritty":
        return [exe, "--title", title, "-e", *program]
    if name == "wezterm":
        return [exe, "start", "--", *program]
    if name == "foot":
        return [exe, "--title", title, *program]
    if name in {"gnome-terminal", "kgx", "xfce4-terminal"}:
        return [exe, "--title", title, "--", *program]
    if name == "xterm":
        return [exe, "-T", title, "-e", *program]
    return [exe, *program]


__all__ = ["_visualization_command_for_terminal"]
