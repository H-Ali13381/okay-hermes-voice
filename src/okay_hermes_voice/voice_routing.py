"""Voice command routing, acknowledgements, and response dispatch."""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.tts_tool import text_to_speech_tool

from .daemon_config import DEFAULT_CONFIG, HERMES_HOME, LOG
from .hermes_runtime import ask_hermes_turn
from .interaction_router import (
    ACK_TEXT,
    AckTemplate,
    AcknowledgementCache,
    InteractionRouterConfig,
    RouteTarget,
    VoiceRequestPlan,
    answer_with_small_model,
    plan_voice_request,
)
from .playback import play_tts_file, speak_response

def normalize_voice_command(text: str) -> str:
    """Normalize STT text for exact local voice-control commands."""
    normalized = (text or "").casefold()
    normalized = re.sub(r"[^\w\s']+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for prefix in ("okay hermes ", "ok hermes ", "hey hermes ", "hermes "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
            break
    return normalized


def is_close_transcript(transcript: str, cfg: Dict[str, Any]) -> bool:
    """Return True only for explicit voice-session close commands."""
    normalized = normalize_voice_command(transcript)
    phrases = cfg.get("conversation_close_phrases") or DEFAULT_CONFIG["conversation_close_phrases"]
    normalized_phrases = {normalize_voice_command(str(phrase)) for phrase in phrases if str(phrase).strip()}
    return normalized in normalized_phrases


def command_recording_config_for_turn(cfg: Dict[str, Any], turn_index: int) -> Dict[str, Any]:
    """Return per-turn recording config; follow-ups can wait indefinitely."""
    turn_cfg = dict(cfg)
    if turn_index > 1 and cfg.get("conversation_mode_enabled", True):
        turn_cfg["speech_start_timeout_seconds"] = float(cfg.get("conversation_followup_start_timeout_seconds", 0.0) or 0.0)
    return turn_cfg


def interaction_router_config_from_daemon_config(cfg: Dict[str, Any]) -> InteractionRouterConfig:
    """Translate daemon config keys into the standalone router config."""
    return InteractionRouterConfig.from_mapping({
        "router_enabled": cfg.get("interaction_router_enabled", True),
        "router_provider": cfg.get("interaction_router_provider", "openrouter"),
        "router_model": cfg.get("interaction_router_model", "google/gemini-2.5-flash-lite"),
        "router_timeout_seconds": cfg.get("interaction_router_timeout_seconds", 1.5),
        "router_min_confidence": cfg.get("interaction_router_min_confidence", 0.70),
        "small_model_enabled": cfg.get("interaction_router_small_model_enabled", False),
        "small_model_provider": cfg.get("interaction_router_small_model_provider", "openrouter"),
        "small_model_model": cfg.get("interaction_router_small_model_model", "google/gemini-2.5-flash-lite"),
        "ack_cache_enabled": cfg.get("interaction_router_ack_cache_enabled", True),
        "ack_cache_dir": cfg.get("interaction_router_ack_cache_dir", str(HERMES_HOME / "wakeword" / "ack_cache")),
    })


def plan_interaction_route(cfg: Dict[str, Any], transcript: str) -> Optional[VoiceRequestPlan]:
    """Classify a transcript and choose the deterministic voice route."""
    router_cfg = interaction_router_config_from_daemon_config(cfg)
    if not router_cfg.router_enabled:
        return None
    started = time.monotonic()
    plan = plan_voice_request(transcript, router_cfg)
    LOG.info(
        "Interaction router target=%s ack=%s reason=%s confidence=%.2f latency=%.3fs router_reason=%s",
        plan.route.target.value,
        plan.route.ack_template_id.value,
        plan.route.reason,
        plan.decision.confidence,
        time.monotonic() - started,
        plan.decision.brief_reason,
    )
    return plan


def _generate_ack_tts(text: str, out_path: Path) -> Path:
    """Generate one acknowledgement clip and copy it into the ack cache."""
    result_raw = text_to_speech_tool(text)
    try:
        result = json.loads(result_raw)
    except Exception as exc:
        raise RuntimeError(f"TTS returned non-JSON for acknowledgement: {result_raw!r}") from exc
    if not result.get("success") or not result.get("file_path"):
        raise RuntimeError(f"TTS failed for acknowledgement: {result}")
    source = Path(str(result["file_path"])).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target = out_path.with_suffix(source.suffix or out_path.suffix)
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    return target


def _play_interaction_ack_sync(
    cfg: Dict[str, Any],
    template_id: AckTemplate,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> bool:
    if template_id is AckTemplate.NONE or not cfg.get("tts_enabled", True):
        return False
    router_cfg = interaction_router_config_from_daemon_config(cfg)
    if not router_cfg.ack_cache_enabled:
        from .interaction_router import ACK_TEXT
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

    thread = threading.Thread(
        target=_run_ack,
        name=f"okay-hermes-ack-{template_id.value}",
        daemon=True,
    )
    thread.start()
    LOG.info("Scheduled interaction acknowledgement: %s", template_id.value)
    return True


def route_transcribed_request(
    cfg: Dict[str, Any],
    transcript: str,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[VoiceRequestPlan]:
    """Plan routing for a final STT transcript and play any immediate acknowledgement."""
    plan = plan_interaction_route(cfg, transcript)
    if plan is None:
        return None
    if plan.route.ack_template_id is not AckTemplate.NONE:
        play_interaction_ack(cfg, plan.route.ack_template_id, cancel_check=cancel_check, block=False)
    return plan


def interaction_ack_text(plan: Optional[VoiceRequestPlan]) -> str:
    if plan is None or plan.route.ack_template_id is AckTemplate.NONE:
        return ""
    return ACK_TEXT.get(plan.route.ack_template_id, "")


def routed_request_status_message(plan: Optional[VoiceRequestPlan]) -> str:
    route_target = plan.route.target.value if plan else "heavy_agent"
    route_label = route_target.replace("_", " ")
    ack_text = interaction_ack_text(plan)
    if ack_text:
        return f"{ack_text} Request routed to {route_label}. Handling it now…"
    return f"Request routed to {route_label}. Handling it now…"


def answer_routed_request(
    cfg: Dict[str, Any],
    transcript: str,
    plan: Optional[VoiceRequestPlan],
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]], str]:
    """Execute the selected route, falling back to heavy Hermes when needed."""
    history = list(conversation_history or [])
    if plan and plan.route.target is RouteTarget.SAFETY_FLOW:
        return "I can’t help with that request.", history, "safety_flow"
    if plan and plan.route.target is RouteTarget.ASK_CLARIFICATION:
        return "Could you clarify what you want me to do?", history, "ask_clarification"
    if plan and plan.route.target is RouteTarget.SMALL_MODEL:
        router_cfg = interaction_router_config_from_daemon_config(cfg)
        response = answer_with_small_model(transcript, router_cfg)
        if response:
            history.extend([
                {"role": "user", "content": transcript},
                {"role": "assistant", "content": response},
            ])
            return response, history, "small_model"
        LOG.info("Small-model route produced no response; falling back to heavy Hermes")
    response, history = ask_hermes_turn(cfg, transcript, history, cancel_check=cancel_check)
    return response, history, "heavy_agent"
