"""Per-turn command recording config."""
from __future__ import annotations

from typing import Any, Dict


def command_recording_config_for_turn(cfg: Dict[str, Any], turn_index: int) -> Dict[str, Any]:
    """Return per-turn recording config; follow-ups use a configurable start timeout; 0.0 means indefinite."""
    turn_cfg = dict(cfg)
    if turn_index > 1 and cfg.get("conversation_mode_enabled", True):
        if cfg.get("_heavy_agent_delegation_pending", False):
            turn_cfg["speech_start_timeout_seconds"] = float(
                cfg.get("heavy_agent_delegation_followup_start_timeout_seconds", 2.0) or 2.0
            )
        else:
            turn_cfg["speech_start_timeout_seconds"] = float(cfg.get("conversation_followup_start_timeout_seconds", 0.0) or 0.0)
    return turn_cfg


__all__ = ["command_recording_config_for_turn"]
