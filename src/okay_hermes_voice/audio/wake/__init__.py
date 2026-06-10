"""Wakeword model loading and inference facade."""
from __future__ import annotations

from .inference import run_wake_inference
from .session import model_session
from .wait import wait_for_wake

__all__ = ["model_session", "run_wake_inference", "wait_for_wake"]
