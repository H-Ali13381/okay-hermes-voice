"""Synchronous acknowledgement playback."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..daemon_config import LOG
from ..interaction_router import AckTemplate
from .router_config import interaction_router_config_from_daemon_config


def _play_interaction_ack_sync(
    cfg: Dict[str, Any],
    template_id: AckTemplate,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> bool:
    if template_id is AckTemplate.NONE or not cfg.get("tts_enabled", True):
        return False
    router_cfg = interaction_router_config_from_daemon_config(cfg)
    from . import ACK_TEXT, AcknowledgementCache, _generate_ack_tts, play_tts_file, speak_response
    if not router_cfg.ack_cache_enabled:
        speak_response(cfg, ACK_TEXT[template_id], cancel_check=cancel_check)
        return True
    cache = AcknowledgementCache(
        router_cfg.ack_cache_path,
        tts_generator=_generate_ack_tts,
        audio_player=lambda path: play_tts_file(cfg, str(path), cancel_check=cancel_check),
    )
    try:
        LOG.info("Playing interaction acknowledgement: %s", template_id.value)
        return cache.play(template_id)
    except Exception as exc:
        LOG.warning("Could not play cached interaction acknowledgement: %s", exc)
        return False


__all__ = ["_play_interaction_ack_sync"]
