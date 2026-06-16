"""Copy recorded command audio beside its activation archive."""
from __future__ import annotations

import contextlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from ..daemon_config import LOG
from .metadata_update import update_activation_archive_metadata


def archive_command_audio(
    cfg: Dict[str, Any],
    archive: Optional[Dict[str, Any]],
    command_path: Path,
    turn_index: int,
) -> Optional[str]:
    if not archive or not cfg.get("activation_save_command_audio", True):
        return None
    try:
        raw_metadata_path = archive.get("metadata_path")
        if not raw_metadata_path:
            return None
        metadata_path = Path(str(raw_metadata_path)).expanduser()
        stem = str(archive.get("stem") or metadata_path.stem)
        dest = metadata_path.with_name(f"{stem}_turn{turn_index:02d}_command.wav")
        shutil.copy2(str(command_path), str(dest))
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with contextlib.suppress(Exception):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths = metadata.get("command_wav_paths")
        if not isinstance(paths, list):
            paths = []
        paths.append(str(dest))
        update_activation_archive_metadata(archive, command_wav_paths=paths)
        return str(dest)
    except Exception as exc:
        LOG.warning("Could not archive command audio %s: %s", command_path, exc)
        return None


__all__ = ["archive_command_audio"]
