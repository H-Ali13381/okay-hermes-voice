"""Choose archive metadata fields for a recorded command clip."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def command_audio_metadata_fields(
    archived_command_path: Optional[str],
    command_path: Path,
    latest: bool = False,
) -> Dict[str, str]:
    if archived_command_path:
        return {"latest_command_wav_path" if latest else "command_wav_path": archived_command_path}
    return {"temp_command_wav_path": str(command_path)}


__all__ = ["command_audio_metadata_fields"]
