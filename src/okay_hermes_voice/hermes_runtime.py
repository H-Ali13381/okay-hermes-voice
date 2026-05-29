"""Hermes Agent runtime facade for voice mode."""
from __future__ import annotations

import contextlib
import os
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .daemon_config import DEFAULT_CONFIG, LOG, _HERMES_AGENT_CACHE
from .hermes_agent_cache import get_warm_hermes_agent, _interrupt_hermes_agent, _run_warm_hermes_agent_turn
from .hermes_runtime_selection import configured_hermes_runtime_selection, configured_hermes_toolsets
from .hermes_subprocess import (
    _UseSubprocessForCancellation,
    _collect_hermes_process_output,
    _execution_cancel_requested,
    _hermes_cancel_poll_seconds,
    _hermes_interrupt_wait_seconds,
    _run_hermes_subprocess_turn,
    _terminate_hermes_process_group,
    clean_hermes_output,
    strip_ansi,
)


def prewarm_hermes(cfg: Dict[str, Any]) -> None:
    """Initialize the warm in-process agent at service start."""
    if not (cfg.get("hermes_inprocess", True) and cfg.get("hermes_warm_agent", True) and cfg.get("prewarm_hermes_on_start", True)):
        return
    try:
        provider, model = configured_hermes_runtime_selection(cfg)
        toolsets = configured_hermes_toolsets(cfg)
        started = time.monotonic()
        get_warm_hermes_agent(cfg, provider, model, toolsets)
        LOG.info("Hermes warm-agent prewarm complete in %.2fs", time.monotonic() - started)
    except Exception as exc:
        LOG.warning("Hermes warm-agent prewarm failed; will lazy-init on first request: %s", exc)


def ask_hermes_turn(
    cfg: Dict[str, Any],
    transcript: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    prompt_prefix = str(cfg.get("hermes_prompt_prefix") or "").strip()
    prompt = f"{prompt_prefix}\n\nTranscript:\n{transcript}" if prompt_prefix else transcript
    provider, model = configured_hermes_runtime_selection(cfg)
    toolsets = configured_hermes_toolsets(cfg)
    history = list(conversation_history or [])

    LOG.info(
        "Invoking Hermes for transcript (%d chars), mode=%s provider=%s model=%s toolsets=%s",
        len(transcript),
        "inprocess" if cfg.get("hermes_inprocess", True) else "subprocess",
        provider or "config",
        model or "config",
        toolsets if toolsets is not None else "config",
    )

    if cfg.get("hermes_inprocess", True):
        started = time.monotonic()
        try:
            if cfg.get("hermes_warm_agent", True):
                agent = get_warm_hermes_agent(cfg, provider, model, toolsets)
                response, history, cancelled = _run_warm_hermes_agent_turn(
                    cfg,
                    agent,
                    prompt,
                    transcript,
                    history,
                    cancel_check,
                )
                if cancelled:
                    return None, history
            elif cancel_check is not None:
                LOG.info("Non-warm in-process Hermes is not interruptible; using cancellable subprocess path")
                raise _UseSubprocessForCancellation()
            else:
                from hermes_cli.oneshot import _run_agent
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        response = _run_agent(
                            prompt,
                            model=model,
                            provider=provider,
                            toolsets=toolsets,
                            use_config_toolsets=toolsets is None,
                        )
            response = (response or "").strip()
            LOG.info("Hermes in-process latency: %.2fs", time.monotonic() - started)
            LOG.info("Hermes response: %s", response[:1000])
            return response or None, history
        except _UseSubprocessForCancellation:
            pass
        except Exception as exc:
            if _execution_cancel_requested(cancel_check):
                LOG.info("Hermes in-process execution cancelled")
                return None, history
            LOG.exception("In-process Hermes failed; falling back to subprocess: %s", exc)

    hermes_bin = str(cfg.get("hermes_bin") or DEFAULT_CONFIG["hermes_bin"])
    cmd = [hermes_bin, "chat", "-Q", "--source", str(cfg.get("hermes_source") or "wakeword")]
    if provider:
        cmd.extend(["--provider", provider])
    if model:
        cmd.extend(["-m", model])
    if toolsets:
        cmd.extend(["-t", ",".join(toolsets) if isinstance(toolsets, list) else str(toolsets)])
    cmd.extend(["-q", prompt])

    response, history, _cancelled = _run_hermes_subprocess_turn(cfg, cmd, transcript, history, cancel_check)
    return response, history


def ask_hermes(cfg: Dict[str, Any], transcript: str) -> Optional[str]:
    """Backward-compatible single-turn wrapper."""
    response, _history = ask_hermes_turn(cfg, transcript)
    return response
