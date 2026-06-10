"""Hermes subprocess output cleanup."""
from __future__ import annotations

from typing import List

from ..daemon_config import ANSI_RE


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def clean_hermes_output(stdout: str) -> str:
    lines = strip_ansi(stdout).splitlines()
    cleaned: List[str] = []
    for line in lines:
        if line.startswith("session_id:"):
            continue
        cleaned.append(line.rstrip())
    return "\n".join(cleaned).strip()


__all__ = ["clean_hermes_output", "strip_ansi"]
