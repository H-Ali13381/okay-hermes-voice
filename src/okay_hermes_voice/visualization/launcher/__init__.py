"""Terminal launch boundary facade for the popup visualizer."""
from __future__ import annotations

from . import commands as _commands_mod
from . import launch as _launch_mod
from .command import _visualization_command_for_terminal
from .commands import _visualization_terminal_commands
from .constants import VISUALIZATION_LAUNCH_GRACE_SECONDS
from .environment import _infer_wayland_display, _visualization_launch_env
from .launch import launch_visualization
from .process_comm import _communicate_or_wait
from .process_output import _short_process_output, _visualization_failure_message
from .process_watch import _reap_visualization_process, _watch_visualization_process
from .selected_command import _visualization_terminal_command
from .terminal_candidates import _visualization_terminal_candidates
from .test_command import visualization_test

shutil = _commands_mod.shutil
subprocess = _launch_mod.subprocess

__all__ = [
    "VISUALIZATION_LAUNCH_GRACE_SECONDS",
    "launch_visualization",
    "visualization_test",
]
