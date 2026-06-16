"""Merge status, timing, and transcript updates into archive metadata."""
from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..daemon_config import LOG


def update_activation_archive_metadata(archive: Optional[Dict[str, Any]], **updates: Any) -> None:
    """Merge metadata updates into the activation archive JSON."""
    if not archive:
        return
    try:
        raw_metadata_path = archive.get("metadata_path")
        if not raw_metadata_path:
            return
        metadata_path = Path(str(raw_metadata_path)).expanduser()
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with contextlib.suppress(Exception):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(updates)
        metadata["updated_at"] = time.time()
        metadata["updated_at_local"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(metadata["updated_at"]))
        tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, metadata_path)
    except Exception as exc:
        LOG.warning("Could not update activation archive metadata: %s", exc)


__all__ = ["update_activation_archive_metadata"]
