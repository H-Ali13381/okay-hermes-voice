"""Per-turn command recording config."""
from __future__ import annotations

from typing import Any, Dict


def command_recording_config_for_turn(cfg: Dict[str, Any], turn_index: int) -> Dict[str, Any]:
    """Return per-turn recording config; follow-ups can wait indefinitely."""
    turn_cfg = dict(cfg)
    if turn_index > 1 and cfg.get("conversation_mode_enabled", True):
        turn_cfg["speech_start_timeout_seconds"] = float(cfg.get("conversation_followup_start_timeout_seconds", 0.0) or 0.0)
    return turn_cfg


__all__ = ["command_recording_config_for_turn"]
