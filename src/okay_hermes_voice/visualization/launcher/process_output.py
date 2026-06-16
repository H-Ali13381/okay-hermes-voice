"""Short launch-failure output formatting."""
from __future__ import annotations

from typing import Optional


def _short_process_output(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines)[-500:]


def _visualization_failure_message(terminal_label: str, returncode: Optional[int], stdout: str, stderr: str) -> str:
    output = _short_process_output(stderr) or _short_process_output(stdout)
    message = f"{terminal_label} exited immediately with status {returncode}"
    return f"{message}: {output}" if output else message


__all__ = ["_short_process_output", "_visualization_failure_message"]
