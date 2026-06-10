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
) -> bool:
    """Play a short receipt-only acknowledgement before longer work starts."""
    if block:
        return _play_interaction_ack_sync(cfg, template_id, cancel_check=cancel_check)
    if template_id is AckTemplate.NONE or not cfg.get("tts_enabled", True):
        return False

    def _run_ack() -> None:
        _play_interaction_ack_sync(cfg, template_id, cancel_check=cancel_check)

    thread = threading.Thread(target=_run_ack, name=f"okay-hermes-ack-{template_id.value}", daemon=True)
    thread.start()
    LOG.info("Scheduled interaction acknowledgement: %s", template_id.value)
    return True


__all__ = ["play_interaction_ack"]
