"""Keyboard, mouse-wheel, and terminal-control facade for the popup."""
from __future__ import annotations

import select as select
import sys as sys

from .escape import normalized_escape_key
from .read import read_keypress
from .scroll import apply_scroll_key
from .terminal_control import write_terminal_control

__all__ = ["apply_scroll_key", "normalized_escape_key", "read_keypress", "write_terminal_control"]
