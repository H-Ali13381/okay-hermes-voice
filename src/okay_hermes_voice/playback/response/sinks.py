"""Playback sink selection."""
from __future__ import annotations

import subprocess
from typing import Any, Dict, List

from ...daemon_config import LOG


def _configured_playback_sinks(cfg: Dict[str, Any]) -> List[str]:
    sink = str(cfg.get("playback_sink") or "@DEFAULT_SINK@").strip()
    if sink.lower() != "all":
        return [sink]
    try:
        proc = subprocess.run(["pactl", "list", "short", "sinks"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3, check=False)
        sinks = []
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                sinks.append(parts[1])
        return sinks or ["@DEFAULT_SINK@"]
    except Exception as exc:
        LOG.warning("Could not enumerate sinks for playback_sink=all: %s", exc)
        return ["@DEFAULT_SINK@"]


__all__ = ["_configured_playback_sinks"]
