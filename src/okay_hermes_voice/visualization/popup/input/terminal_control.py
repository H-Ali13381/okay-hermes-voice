"""Terminal control sequence writes."""
from __future__ import annotations

import sys


def write_terminal_control(sequence: str) -> None:
    sys.stdout.write(sequence)
    sys.stdout.flush()


__all__ = ["write_terminal_control"]
