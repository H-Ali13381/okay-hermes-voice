"""Publish terminal-cancelled voice-session state."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ...activation_archive import update_activation_archive_metadata
from ...daemon_config import LOG
from .update import update_visualization_state


def finish_cancelled_voice_session(
    visual_state: Optional[Path],
    activation_archive: Optional[Dict[str, Any]],
    archive_turns: List[Dict[str, Any]],
    reason: str,
) -> None:
    update_visualization_state(
        visual_state,
        status="cancelled",
        message="Voice session cancelled from the Hermes Voice terminal.",
        error="",
        cancel_requested=True,
        cancel_reason=reason,
    )
    update_activation_archive_metadata(
        activation_archive,
        status="cancelled_by_terminal",
        close_reason=reason,
        cancel_reason=reason,
        turns=archive_turns,
    )
    LOG.info("Voice conversation cancelled by terminal request: %s", reason)


__all__ = ["finish_cancelled_voice_session"]
