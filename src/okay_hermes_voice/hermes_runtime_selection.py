"""Resolve Hermes provider/model/toolset configuration for voice mode."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .daemon_config import LOG

def configured_hermes_runtime_selection(cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve wakeword Hermes provider/model overrides.

    Empty wakeword provider/model values mean "inherit the user's normal Hermes
    defaults".  Resolve the model here so the warm in-process AIAgent is built
    with the same effective model as `hermes chat`, rather than an empty model
    string that can hide the active default in logs and provider routing.
    """
    provider = (cfg.get("hermes_provider") or "").strip() or None
    model = (cfg.get("hermes_model") or "").strip() or None
    if model:
        return provider, model

    try:
        from hermes_cli.config import load_config as load_hermes_config

        hermes_cfg = load_hermes_config()
        model_cfg = hermes_cfg.get("model") or {}
        if isinstance(model_cfg, str):
            model = model_cfg.strip() or None
        elif isinstance(model_cfg, dict):
            model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip() or None
    except Exception as exc:
        LOG.warning("Could not read default Hermes model from config; using provider default: %s", exc)
    return provider, model


def configured_hermes_toolsets(cfg: Dict[str, Any]) -> Optional[List[str]]:
    """Resolve wakeword toolsets.

    A non-empty `hermes_toolsets` value is an explicit wakeword-only override.
    Blank/null means inherit the regular CLI platform toolsets, which is what
    users expect when they say wakeword Hermes should behave like normal Hermes.
    """
    raw_toolsets = cfg.get("hermes_toolsets")
    if isinstance(raw_toolsets, str):
        stripped = raw_toolsets.strip()
        if stripped and stripped.lower() not in {"auto", "config", "default", "inherit"}:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        raw_toolsets = None
    if isinstance(raw_toolsets, (list, tuple, set)):
        return [str(item).strip() for item in raw_toolsets if str(item).strip()]
    if raw_toolsets is not None:
        return raw_toolsets

    try:
        from hermes_cli.config import load_config as load_hermes_config
        from hermes_cli.tools_config import _get_platform_tools

        cli_toolsets = _get_platform_tools(load_hermes_config(), "cli")
        return sorted(str(toolset) for toolset in cli_toolsets)
    except Exception as exc:
        LOG.warning("Could not read CLI toolsets from Hermes config; using all available tools: %s", exc)
        return None
