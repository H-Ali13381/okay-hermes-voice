"""Activation session setup and wake metadata normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .services import ActivationFlowServices


@dataclass(frozen=True)
class ActivationSession:
    activation: Dict[str, Any]
    probability: float
    activation_detected_at: float
    archive: Any
    visual_state: Any


def normalize_activation(deps: ActivationFlowServices, activation: Any) -> tuple[Dict[str, Any], float, float]:
    if isinstance(activation, dict):
        normalized = activation
        probability = float(normalized.get("probability") or 0.0)
    else:
        probability = float(activation)
        normalized = {
            "probability": probability,
            "scores": [probability],
            "detected_at": deps.time.time(),
        }
    activation_detected_at = float(normalized.get("detected_at") or deps.time.time())
    return normalized, probability, activation_detected_at


def start_activation_session(
    deps: ActivationFlowServices,
    cfg: Dict[str, Any],
    activation: Any,
) -> ActivationSession:
    normalized, probability, activation_detected_at = normalize_activation(deps, activation)
    handle_started_at = deps.time.time()
    voice_session_timing = {
        "schema_version": 1,
        "activation_detected_at": activation_detected_at,
        "handle_started_at": handle_started_at,
        "wake_to_handle_seconds": max(0.0, handle_started_at - activation_detected_at),
    }
    deps.log.info("Handling wake activation; probability=%.6f", probability)
    activation_archive = deps.save_activation_archive(cfg, normalized)
    visual_state = deps.launch_visualization(cfg, probability)
    deps.update_visualization_state(visual_state, voice_session_timing=voice_session_timing)
    deps.update_activation_archive_metadata(activation_archive, voice_session_timing=voice_session_timing)
    if activation_archive:
        deps.update_visualization_state(visual_state, activation_archive=activation_archive)
    deps.maybe_beep(cfg, frequency=880, count=1)
    return ActivationSession(
        activation=normalized,
        probability=probability,
        activation_detected_at=activation_detected_at,
        archive=activation_archive,
        visual_state=visual_state,
    )
