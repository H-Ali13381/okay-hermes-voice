"""Acknowledgement playback scheduling."""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

from ..daemon_config import LOG
from ..interaction_router import AckTemplate
from .ack_playback_sync import _play_interaction_ack_sync


def play_interaction_ack(
    cfg: Dict[str, Any],
    template_id: AckTemplate,
    cancel_check: Optional[Callable[[], bool]] = None,
    *,
    block: bool = True,
    loop_until_cancelled: bool = False,
) -> bool:
    """Play a short acknowledgement, optionally looping until cancellation."""
    if block:
        return _play_interaction_ack_sync(cfg, template_id, cancel_check=cancel_check)
    if template_id is AckTemplate.NONE or not cfg.get("tts_enabled", True):
        return False

    def _run_ack() -> None:
        while True:
            if _cancel_requested(cancel_check):
                return
            played = _play_interaction_ack_sync(cfg, template_id, cancel_check=cancel_check)
            if not loop_until_cancelled or not played:
                return
            if _cancel_requested(cancel_check):
                return

    thread = threading.Thread(target=_run_ack, name=f"okay-hermes-ack-{template_id.value}", daemon=True)
    thread.start()
    LOG.info("Scheduled interaction acknowledgement: %s", template_id.value)
    return True


def _cancel_requested(cancel_check: Optional[Callable[[], bool]]) -> bool:
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception as exc:
        LOG.warning("Acknowledgement cancel check failed: %s", exc)
        return False


__all__ = ["play_interaction_ack"]
