"""Non-blocking heavy-agent dispatch for routed voice requests."""
from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..daemon_config import LOG
from ..hermes_runtime import ask_hermes_turn


@dataclass(slots=True)
class HeavyDelegationResult:
    delegation_id: str
    response: str
    history: List[Dict[str, Any]]
    source: str = "heavy_agent_delegation"


class HeavyDelegationManager:
    def __init__(self) -> None:
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def enabled(self, cfg: Dict[str, Any]) -> bool:
        return bool(cfg.get("heavy_agent_delegation_enabled", False))

    def has_pending(self) -> bool:
        with self._lock:
            return any(task.get("status") == "running" for task in self._tasks.values())

    def dispatch(
        self,
        cfg: Dict[str, Any],
        transcript: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Dispatch a heavy Hermes turn through Hermes' async delegation rail."""
        if not self.enabled(cfg):
            return None
        if self.has_pending():
            LOG.info("Heavy-agent delegation already running; using inline heavy path")
            return None

        id_ready = threading.Event()
        holder: Dict[str, str] = {}
        cancel_event = threading.Event()
        history = list(conversation_history or [])
        heavy_cfg = self._heavy_agent_cfg(cfg)

        def runner() -> Dict[str, Any]:
            id_ready.wait(timeout=5.0)
            delegation_id = holder.get("delegation_id", "")
            started = time.monotonic()
            try:
                response, updated_history = ask_hermes_turn(
                    heavy_cfg,
                    transcript,
                    history,
                    cancel_check=cancel_event.is_set,
                )
                response_text = (response or "").strip()
                if delegation_id:
                    self._complete(delegation_id, response_text, updated_history)
                return {
                    "status": "completed" if response_text else "error",
                    "summary": response_text or None,
                    "error": None if response_text else "Heavy agent returned no response.",
                    "api_calls": 0,
                    "duration_seconds": round(time.monotonic() - started, 2),
                }
            except BaseException as exc:  # noqa: BLE001 - keep background worker contained
                if delegation_id:
                    self._fail(delegation_id, f"{type(exc).__name__}: {exc}")
                return {
                    "status": "error",
                    "summary": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "api_calls": 0,
                    "duration_seconds": round(time.monotonic() - started, 2),
                }

        try:
            async_delegation = importlib.import_module("tools.async_delegation")
            dispatch_async_delegation = async_delegation.dispatch_async_delegation
        except Exception as exc:
            LOG.warning("Hermes async delegation unavailable; using inline heavy path: %s", exc)
            return None

        dispatch = dispatch_async_delegation(
            goal=self._goal(),
            context=self._context(history),
            toolsets=None,
            role="leaf",
            model=str(heavy_cfg.get("hermes_model") or "") or None,
            session_key="okay-hermes-voice",
            runner=runner,
            interrupt_fn=cancel_event.set,
            max_async_children=max(1, int(cfg.get("heavy_agent_delegation_max_async_children") or 1)),
        )
        if dispatch.get("status") != "dispatched":
            LOG.info("Heavy-agent delegation rejected: %s", dispatch.get("error") or dispatch)
            return None

        delegation_id = str(dispatch["delegation_id"])
        holder["delegation_id"] = delegation_id
        with self._lock:
            self._tasks[delegation_id] = {
                "status": "running",
                "transcript": transcript,
                "history": history,
                "dispatched_at": time.time(),
                "cancel": cancel_event.set,
            }
        id_ready.set()
        LOG.info("Dispatched heavy-agent voice delegation %s", delegation_id)
        return delegation_id

    def pop_completed(self) -> Optional[HeavyDelegationResult]:
        """Return and remove the oldest completed heavy delegation, if any."""
        with self._lock:
            completed = [
                (delegation_id, task)
                for delegation_id, task in self._tasks.items()
                if task.get("status") in {"completed", "error"}
            ]
            if not completed:
                return None
            completed.sort(key=lambda item: item[1].get("completed_at") or item[1].get("dispatched_at") or 0)
            delegation_id, task = completed[0]
            self._tasks.pop(delegation_id, None)

        response = str(task.get("response") or task.get("error") or "The heavy agent finished without a response.").strip()
        history = list(task.get("history") or [])
        return HeavyDelegationResult(delegation_id=delegation_id, response=response, history=history)

    def cancel_pending(self) -> int:
        count = 0
        with self._lock:
            tasks = list(self._tasks.values())
        for task in tasks:
            cancel = task.get("cancel")
            if callable(cancel) and task.get("status") == "running":
                cancel()
                count += 1
        return count

    def reset_for_tests(self) -> None:
        with self._lock:
            self._tasks.clear()

    def _complete(self, delegation_id: str, response: str, history: List[Dict[str, Any]]) -> None:
        with self._lock:
            task = self._tasks.get(delegation_id)
            if task is None:
                return
            task.update(
                {
                    "status": "completed" if response else "error",
                    "response": response,
                    "history": list(history),
                    "error": None if response else "Heavy agent returned no response.",
                    "completed_at": time.time(),
                }
            )

    def _fail(self, delegation_id: str, error: str) -> None:
        with self._lock:
            task = self._tasks.get(delegation_id)
            if task is None:
                return
            task.update({"status": "error", "error": error, "completed_at": time.time()})

    @staticmethod
    def _heavy_agent_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        heavy_cfg = dict(cfg)
        provider = str(cfg.get("heavy_agent_delegation_provider") or "").strip()
        model = str(cfg.get("heavy_agent_delegation_model") or "").strip()
        toolsets = cfg.get("heavy_agent_delegation_toolsets", None)
        max_iterations = cfg.get("heavy_agent_delegation_max_iterations", None)
        prompt_suffix = str(cfg.get("heavy_agent_delegation_prompt_suffix") or "").strip()

        if provider:
            heavy_cfg["hermes_provider"] = provider
        if model:
            heavy_cfg["hermes_model"] = model
        if toolsets is not None:
            heavy_cfg["hermes_toolsets"] = toolsets
        if max_iterations is not None:
            heavy_cfg["hermes_max_iterations"] = int(max_iterations)
        if prompt_suffix:
            base_prefix = str(heavy_cfg.get("hermes_prompt_prefix") or "").strip()
            heavy_cfg["hermes_prompt_prefix"] = f"{base_prefix}\n\n{prompt_suffix}" if base_prefix else prompt_suffix
        return heavy_cfg

    @staticmethod
    def _goal() -> str:
        return "Handle this Okay Hermes Voice heavy request and produce the final user-facing answer."

    @staticmethod
    def _context(history: List[Dict[str, Any]]) -> str:
        if not history:
            return "No prior voice conversation history. Keep the final answer concise enough for TTS unless depth is required."
        return (
            "Prior voice conversation history is available to the runner. "
            "Keep the final answer concise enough for TTS unless depth is required. "
            f"History messages: {len(history)}."
        )


_MANAGER = HeavyDelegationManager()

heavy_agent_delegation_enabled = _MANAGER.enabled
has_pending_heavy_delegation = _MANAGER.has_pending
dispatch_heavy_agent_delegation = _MANAGER.dispatch
pop_completed_heavy_delegation = _MANAGER.pop_completed
cancel_pending_heavy_delegations = _MANAGER.cancel_pending
_reset_heavy_delegation_for_tests = _MANAGER.reset_for_tests


__all__ = [
    "HeavyDelegationResult",
    "cancel_pending_heavy_delegations",
    "dispatch_heavy_agent_delegation",
    "has_pending_heavy_delegation",
    "heavy_agent_delegation_enabled",
    "pop_completed_heavy_delegation",
]
