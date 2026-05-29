"""Types and parsing helpers for the voice interaction router."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

class RequestComplexity(str, Enum):
    IMMEDIATE = "immediate"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    UNSAFE = "unsafe"
    UNCLEAR = "unclear"


class RouteTarget(str, Enum):
    IMMEDIATE_ONLY = "immediate_only"
    SMALL_MODEL = "small_model"
    HEAVY_AGENT = "heavy_agent"
    ASK_CLARIFICATION = "ask_clarification"
    SAFETY_FLOW = "safety_flow"


class AckTemplate(str, Enum):
    NONE = "none"
    GOT_IT = "got_it"
    CHECKING = "checking"
    THINKING = "thinking"
    LOOKING_THAT_UP = "looking_that_up"
    WORKING = "working"


class ToolRisk(str, Enum):
    NONE = "none"
    READ_ONLY = "read_only"
    SIDE_EFFECT = "side_effect"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class InteractionRouterConfig:
    """Runtime config for the low-latency voice interaction router."""

    router_enabled: bool = True
    router_provider: str = "openrouter"
    router_model: str = "google/gemini-2.5-flash-lite"
    router_timeout_seconds: float = 1.5
    router_min_confidence: float = 0.70

    small_model_enabled: bool = False
    small_model_provider: str = "openrouter"
    small_model_model: str = "google/gemini-2.5-flash-lite"

    ack_cache_enabled: bool = True
    ack_cache_dir: str = "~/.cache/okay-hermes-voice/acks"
    default_ack_template_id: str = "got_it"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "InteractionRouterConfig":
        raw = dict(data or {})
        allowed = {field.name: field for field in fields(cls)}
        defaults = cls()
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in allowed:
                continue
            default = getattr(defaults, key)
            if isinstance(default, bool):
                kwargs[key] = _as_bool(value)
            elif isinstance(default, float):
                kwargs[key] = float(value)
            elif isinstance(default, int):
                kwargs[key] = int(value)
            elif isinstance(default, str):
                kwargs[key] = str(value).strip()
            else:
                kwargs[key] = value
        cfg = cls(**kwargs)
        cfg.router_min_confidence = max(0.0, min(float(cfg.router_min_confidence), 1.0))
        cfg.router_timeout_seconds = max(0.1, float(cfg.router_timeout_seconds))
        return cfg

    @property
    def ack_cache_path(self) -> Path:
        return Path(self.ack_cache_dir).expanduser()


@dataclass(slots=True)
class RouterDecision:
    schema_version: str = "1.0"
    request_complexity: RequestComplexity = RequestComplexity.UNCLEAR
    route_target: RouteTarget = RouteTarget.HEAVY_AGENT
    ack_template_id: AckTemplate = AckTemplate.GOT_IT
    requires_tools: bool = False
    requires_memory: bool = False
    requires_external_data: bool = False
    tool_risk: ToolRisk = ToolRisk.UNKNOWN
    confidence: float = 0.0
    brief_reason: str = "fallback"

    @classmethod
    def parse(cls, raw: str) -> "RouterDecision":
        try:
            data = json.loads(raw)
        except Exception:
            return cls(brief_reason="invalid_json")
        if not isinstance(data, dict):
            return cls(brief_reason="json_not_object")
        return cls(
            schema_version=str(data.get("schema_version") or "1.0"),
            request_complexity=_enum_value(
                RequestComplexity,
                data.get("request_complexity"),
                RequestComplexity.UNCLEAR,
            ),
            route_target=_enum_value(
                RouteTarget,
                data.get("route_target"),
                RouteTarget.HEAVY_AGENT,
            ),
            ack_template_id=_enum_value(
                AckTemplate,
                data.get("ack_template_id"),
                AckTemplate.GOT_IT,
            ),
            requires_tools=_as_bool(data.get("requires_tools", False)),
            requires_memory=_as_bool(data.get("requires_memory", False)),
            requires_external_data=_as_bool(data.get("requires_external_data", False)),
            tool_risk=_enum_value(ToolRisk, data.get("tool_risk"), ToolRisk.UNKNOWN),
            confidence=_clamp_confidence(data.get("confidence", 0.0)),
            brief_reason=str(data.get("brief_reason") or ""),
        )


@dataclass(slots=True)
class VoiceRoute:
    target: RouteTarget
    ack_template_id: AckTemplate
    reason: str


@dataclass(slots=True)
class VoiceRequestPlan:
    transcript: str
    decision: RouterDecision
    route: VoiceRoute


def _enum_value(enum_type: type[Enum], value: Any, fallback: Enum) -> Any:
    try:
        return enum_type(str(value))
    except Exception:
        return fallback


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(confidence):
        return 0.0
    return max(0.0, min(confidence, 1.0))
