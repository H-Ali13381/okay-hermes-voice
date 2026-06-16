"""Cancellable Hermes subprocess execution facade."""
from __future__ import annotations

from .cancellation import _execution_cancel_requested
from .config import _hermes_cancel_poll_seconds, _hermes_interrupt_wait_seconds
from .exceptions import _UseSubprocessForCancellation
from .output import clean_hermes_output, strip_ansi
from .process_output import _collect_hermes_process_output
from .run_turn import _run_hermes_subprocess_turn
from .termination import _terminate_hermes_process_group

__all__ = [
    "_UseSubprocessForCancellation",
    "_collect_hermes_process_output",
    "_execution_cancel_requested",
    "_hermes_cancel_poll_seconds",
    "_hermes_interrupt_wait_seconds",
    "_run_hermes_subprocess_turn",
    "_terminate_hermes_process_group",
    "clean_hermes_output",
    "strip_ansi",
]
