"""Value objects returned by command recording."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandRecording:
    """Recorded command WAV plus optional transcript produced during recording."""

    path: Path
    live_transcript: str = ""


__all__ = ["CommandRecording"]
