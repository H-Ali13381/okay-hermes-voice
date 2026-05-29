"""Warm in-process Hermes Agent cache and cancellable warm-agent turns."""
from __future__ import annotations

import contextlib
import json
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from .daemon_config import LOG, STOP, _HERMES_AGENT_CACHE
from .hermes_subprocess import (
    _execution_cancel_requested,
    _hermes_cancel_poll_seconds,
    _hermes_interrupt_wait_seconds,
)

def get_warm_hermes_agent(cfg: Dict[str, Any], provider: Optional[str], model: Optional[str], toolsets: Any):
    """Return a cached in-process AIAgent to avoid `hermes` process startup per wake."""
    key = json.dumps({
        "provider": provider,
        "model": model,
        "toolsets": toolsets,
        "max_iterations": int(cfg.get("hermes_max_iterations") or 8),
    }, sort_keys=True, default=str)
    cached = _HERMES_AGENT_CACHE.get(key)
    if cached is not None:
        return cached

    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.oneshot import _create_session_db_for_oneshot, _oneshot_clarify_callback
    from run_agent import AIAgent

    runtime = resolve_runtime_provider(requested=provider, target_model=model or None)
    agent = AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        api_mode=runtime.get("api_mode"),
        model=model or "",
        max_iterations=int(cfg.get("hermes_max_iterations") or 8),
        enabled_toolsets=toolsets,
        quiet_mode=True,
        platform="cli",
        session_db=_create_session_db_for_oneshot(),
        credential_pool=runtime.get("credential_pool"),
        clarify_callback=_oneshot_clarify_callback,
        skip_context_files=True,
        load_soul_identity=bool(cfg.get("hermes_load_soul_identity", True)),
    )
    agent.suppress_status_output = True
    agent.stream_delta_callback = None
    agent.tool_gen_callback = None
    _HERMES_AGENT_CACHE.clear()
    _HERMES_AGENT_CACHE[key] = agent
    LOG.info("Warm Hermes agent initialized provider=%s model=%s toolsets=%s", provider, model, toolsets)
    return agent


def _interrupt_hermes_agent(agent: Any, message: str = "Voice session cancelled") -> None:
    interrupt = getattr(agent, "interrupt", None)
    if not callable(interrupt):
        LOG.warning("Warm Hermes agent has no interrupt() method; cannot signal graceful cancellation")
        return
    try:
        interrupt(message)
    except Exception as exc:
        LOG.warning("Failed to interrupt warm Hermes agent: %s", exc)


def _run_warm_hermes_agent_turn(
    cfg: Dict[str, Any],
    agent: Any,
    prompt: str,
    transcript: str,
    history: List[Dict[str, Any]],
    cancel_check: Optional[Callable[[], bool]],
) -> Tuple[Optional[str], List[Dict[str, Any]], bool]:
    """Run warm AIAgent on a worker thread so popup cancellation can call agent.interrupt()."""
    if cancel_check is None and not STOP.is_set():
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                result = agent.run_conversation(
                    prompt,
                    conversation_history=history,
                    persist_user_message=transcript,
                )
        response = result.get("final_response") if isinstance(result, dict) else result
        if isinstance(result, dict) and isinstance(result.get("messages"), list):
            history = result["messages"]
        return (response or None), history, False

    done = threading.Event()
    result_holder: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    result_holder["result"] = agent.run_conversation(
                        prompt,
                        conversation_history=history,
                        persist_user_message=transcript,
                    )
        except BaseException as exc:  # noqa: BLE001 - propagate from worker after join
            result_holder["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_runner, name="okay-hermes-agent-turn", daemon=True)
    thread.start()
    poll_interval = _hermes_cancel_poll_seconds(cfg)
    interrupted_for_cancel = False
    while not done.wait(poll_interval):
        if _execution_cancel_requested(cancel_check):
            interrupted_for_cancel = True
            LOG.info("Voice cancellation requested; interrupting warm Hermes agent")
            _interrupt_hermes_agent(agent)
            done.wait(_hermes_interrupt_wait_seconds(cfg))
            if not done.is_set():
                LOG.warning("Warm Hermes agent did not finish after interruption; dropping cached agent")
            _HERMES_AGENT_CACHE.clear()
            return None, history, True

    if result_holder.get("error") is not None:
        raise result_holder["error"]
    result = result_holder.get("result")
    if _execution_cancel_requested(cancel_check):
        interrupted_for_cancel = True
        _HERMES_AGENT_CACHE.clear()
    if interrupted_for_cancel or (isinstance(result, dict) and result.get("interrupted") and _execution_cancel_requested(cancel_check)):
        return None, history, True
    response = result.get("final_response") if isinstance(result, dict) else result
    if isinstance(result, dict) and isinstance(result.get("messages"), list):
        history = result["messages"]
    return (response or None), history, False
