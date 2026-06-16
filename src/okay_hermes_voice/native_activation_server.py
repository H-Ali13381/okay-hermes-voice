"""Warm Unix-socket activation server for the native wake listener."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .activation_flow import handle_activation
from .daemon_config import CONFIG_PATH, LOG, STOP, load_config, setup_logging, signal_handler
from .audio import prewarm_stt
from .hermes_runtime import prewarm_hermes
from .interaction_clients import prewarm_interaction_router
from .voice_routing.router_config import interaction_router_config_from_daemon_config

DEFAULT_SOCKET_NAME = "native-handler.sock"


@dataclass
class ActivationServerState:
    """Mutable debounce state for activations queued while a session is active."""

    last_session_finished_at: float = 0.0


def native_activation_socket_path(cfg: Dict[str, Any]) -> Path:
    raw = cfg.get("native_activation_socket") or Path.home() / ".hermes" / "wakeword" / DEFAULT_SOCKET_NAME
    return Path(str(raw)).expanduser()


def native_activation_ready_path(cfg: Dict[str, Any]) -> Path:
    return native_activation_socket_path(cfg).with_suffix(".ready")


def mark_ready(cfg: Dict[str, Any]) -> None:
    ready_path = native_activation_ready_path(cfg)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text("ready\n", encoding="utf-8")


def clear_ready(cfg: Dict[str, Any]) -> None:
    try:
        native_activation_ready_path(cfg).unlink()
    except FileNotFoundError:
        pass


def prewarm_runtime(cfg: Dict[str, Any]) -> None:
    """Load the expensive post-wake runtimes in this long-lived process."""
    prewarm_stt(cfg)
    prewarm_hermes(cfg)
    if not cfg.get("prewarm_router_on_start", True):
        return
    router_cfg = interaction_router_config_from_daemon_config(cfg)
    if not router_cfg.router_enabled:
        return
    started = time.monotonic()
    try:
        router_ready = prewarm_interaction_router(router_cfg)
    except Exception as exc:  # pragma: no cover - depends on provider/auth environment
        LOG.warning(
            "Interaction router client prewarm failed in %.2fs provider=%s model=%s error=%s: %s",
            time.monotonic() - started,
            router_cfg.router_provider,
            router_cfg.router_model,
            type(exc).__name__,
            exc,
        )
        return
    if router_ready:
        LOG.info("Interaction router client prewarm complete in %.2fs", time.monotonic() - started)
    else:
        LOG.warning(
            "Interaction router client prewarm skipped in %.2fs provider=%s model=%s",
            time.monotonic() - started,
            router_cfg.router_provider,
            router_cfg.router_model,
        )


def _activation_detected_at(activation: Dict[str, Any]) -> float:
    try:
        return float(activation.get("detected_at") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_stale_activation(cfg: Dict[str, Any], activation: Dict[str, Any], state: ActivationServerState) -> bool:
    detected_at = _activation_detected_at(activation)
    if detected_at <= 0.0 or state.last_session_finished_at <= 0.0:
        return False
    cooldown = max(0.0, float(cfg.get("cooldown_seconds") or 0.0))
    return detected_at <= state.last_session_finished_at + cooldown


def handle_activation_payload(
    cfg: Dict[str, Any],
    payload: bytes,
    *,
    state: Optional[ActivationServerState] = None,
) -> Dict[str, Any]:
    text = payload.decode("utf-8").strip()
    if not text:
        raise ValueError("warm activation server expected JSON payload")
    activation = json.loads(text)
    if not isinstance(activation, dict):
        raise ValueError("native activation JSON must be an object")
    if state is not None and _is_stale_activation(cfg, activation, state):
        LOG.info(
            "Ignoring queued native activation detected during cooldown; probability=%s detected_at=%s last_finished_at=%.6f",
            activation.get("probability"),
            activation.get("detected_at"),
            state.last_session_finished_at,
        )
        return {"ok": True, "ignored": True, "reason": "stale_activation"}
    if STOP.is_set():
        STOP.clear()
    try:
        result = handle_activation(cfg, activation)
    finally:
        if state is not None:
            state.last_session_finished_at = time.time()
    return {"ok": True, "result": result}


def _serve_once(server: socket.socket, cfg: Dict[str, Any], state: ActivationServerState) -> None:
    conn, _addr = server.accept()
    with conn:
        chunks: list[bytes] = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        try:
            response = handle_activation_payload(cfg, b"".join(chunks), state=state)
        except Exception as exc:  # pragma: no cover - exercised via integration/logs
            LOG.exception("Warm native activation handling failed: %s", exc)
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            conn.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
        except OSError as exc:
            LOG.info("Warm native activation client disconnected before response: %s", exc)


def serve(cfg: Dict[str, Any]) -> int:
    socket_path = native_activation_socket_path(cfg)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    clear_ready(cfg)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass

    state = ActivationServerState()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            os.chmod(socket_path, 0o600)
            server.listen(1)
            server.settimeout(0.5)
            LOG.info("Warm native activation server listening on %s", socket_path)
            prewarm_runtime(cfg)
            mark_ready(cfg)
            while not STOP.is_set():
                try:
                    _serve_once(server, cfg, state)
                except socket.timeout:
                    continue
    finally:
        clear_ready(cfg)
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Warm native Okay Hermes activation handler server")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to wakeword config YAML")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(Path(args.config).expanduser())
    setup_logging(cfg, verbose=args.verbose)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    if STOP.is_set():
        STOP.clear()
    return serve(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
