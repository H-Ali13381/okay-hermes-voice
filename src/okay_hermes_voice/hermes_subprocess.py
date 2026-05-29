"""Cancellable Hermes subprocess execution helpers."""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .daemon_config import ANSI_RE, DEFAULT_CONFIG, LOG, STOP

def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def clean_hermes_output(stdout: str) -> str:
    lines = strip_ansi(stdout).splitlines()
    cleaned: List[str] = []
    for line in lines:
        if line.startswith("session_id:"):
            continue
        cleaned.append(line.rstrip())
    return "\n".join(cleaned).strip()


class _UseSubprocessForCancellation(RuntimeError):
    """Internal sentinel for falling through to process-group cancellable Hermes."""


def _execution_cancel_requested(cancel_check: Optional[Callable[[], bool]]) -> bool:
    """Return True when the voice session has requested cancellation."""
    if STOP.is_set():
        return True
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception as exc:
        LOG.warning("Hermes execution cancel check failed: %s", exc)
        return False


def _hermes_cancel_poll_seconds(cfg: Dict[str, Any]) -> float:
    return max(0.01, min(float(cfg.get("hermes_cancel_poll_seconds") or 0.1), 1.0))


def _hermes_interrupt_wait_seconds(cfg: Dict[str, Any]) -> float:
    return max(0.1, min(float(cfg.get("hermes_interrupt_wait_seconds") or 5.0), 60.0))


def _collect_hermes_process_output(proc: subprocess.Popen[Any]) -> Tuple[str, str]:
    try:
        out, err = proc.communicate(timeout=0.2)
    except TypeError:
        try:
            out, err = proc.communicate()
        except Exception:
            out, err = "", ""
    except Exception:
        out, err = getattr(proc, "stdout", "") or "", getattr(proc, "stderr", "") or ""
    return str(out or ""), str(err or "")


def _terminate_hermes_process_group(proc: subprocess.Popen[Any], grace_seconds: float) -> None:
    """Terminate a `hermes chat` process group so child tool subprocesses die too."""
    if proc.poll() is not None:
        return
    pid = int(getattr(proc, "pid", 0) or 0)
    if pid:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception as exc:
            LOG.warning("Could not SIGTERM Hermes process group %s: %s; trying process terminate", pid, exc)
            with contextlib.suppress(Exception):
                proc.terminate()
    else:
        with contextlib.suppress(Exception):
            proc.terminate()
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        LOG.warning("Hermes process group did not stop after %.1fs; sending SIGKILL", grace_seconds)
    except Exception:
        return
    if pid:
        with contextlib.suppress(Exception):
            os.killpg(pid, signal.SIGKILL)
    else:
        with contextlib.suppress(Exception):
            proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=1.0)


def _run_hermes_subprocess_turn(
    cfg: Dict[str, Any],
    cmd: List[str],
    transcript: str,
    history: List[Dict[str, Any]],
    cancel_check: Optional[Callable[[], bool]],
) -> Tuple[Optional[str], List[Dict[str, Any]], bool]:
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path.home()),
            start_new_session=True,
        )
    except Exception as exc:
        LOG.exception("Failed to invoke Hermes: %s", exc)
        return f"I could not start Hermes: {exc}", history, False

    timeout = float(cfg.get("hermes_timeout_seconds") or DEFAULT_CONFIG["hermes_timeout_seconds"])
    deadline = time.monotonic() + timeout
    poll_interval = _hermes_cancel_poll_seconds(cfg)
    while proc.poll() is None:
        if _execution_cancel_requested(cancel_check):
            LOG.info("Voice cancellation requested; terminating Hermes subprocess group")
            _terminate_hermes_process_group(proc, _hermes_interrupt_wait_seconds(cfg))
            _collect_hermes_process_output(proc)
            return None, history, True
        if time.monotonic() >= deadline:
            LOG.error("Hermes command timed out")
            _terminate_hermes_process_group(proc, _hermes_interrupt_wait_seconds(cfg))
            _collect_hermes_process_output(proc)
            return "Hermes timed out while handling that request.", history, False
        time.sleep(poll_interval)

    stdout, stderr = _collect_hermes_process_output(proc)
    LOG.info("Hermes subprocess latency: %.2fs", time.monotonic() - started)
    if stderr.strip():
        LOG.warning("Hermes stderr: %s", strip_ansi(stderr).strip())
    response = clean_hermes_output(stdout)
    if proc.returncode != 0:
        LOG.error("Hermes exited with %s; stdout=%r", proc.returncode, response)
        return response or f"Hermes exited with status {proc.returncode}.", history, False
    LOG.info("Hermes response: %s", response[:1000])
    if response:
        history.extend([
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": response},
        ])
    return response or None, history, False
