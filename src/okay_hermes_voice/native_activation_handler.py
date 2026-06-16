"""Short-lived activation handler for the native wakeword listener."""
from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .daemon_config import CONFIG_PATH, STOP, load_config, setup_logging, signal_handler

__all__ = ["main", "read_activation", "send_activation_to_server", "should_fallback_to_local_handler"]


def read_activation() -> dict[str, object]:
    payload = sys.stdin.read().strip()
    if not payload:
        raise ValueError("native activation handler expected JSON on stdin")
    activation = json.loads(payload)
    if not isinstance(activation, dict):
        raise ValueError("native activation JSON must be an object")
    return activation


def native_activation_socket_path(cfg: Dict[str, Any]) -> Path:
    raw = cfg.get("native_activation_socket") or Path.home() / ".hermes" / "wakeword" / "native-handler.sock"
    return Path(str(raw)).expanduser()


def handle_activation(cfg: Dict[str, Any], activation: Dict[str, Any]) -> object:
    from .activation_flow import handle_activation as run

    return run(cfg, activation)


def send_activation_to_server(activation: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    socket_path = native_activation_socket_path(cfg)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(float(cfg.get("native_activation_server_timeout_seconds", 5.0)))
        client.connect(str(socket_path))
        client.sendall(json.dumps(activation, ensure_ascii=False).encode("utf-8"))
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    response_text = b"".join(chunks).decode("utf-8").strip()
    response = json.loads(response_text) if response_text else {"ok": False, "error": "empty response"}
    if response.get("ok"):
        return 0
    print(f"warm native activation server failed: {response.get('error') or response}", file=sys.stderr)
    return 1


def should_fallback_to_local_handler(exc: Exception) -> bool:
    """Return whether a warm-server failure should open a local voice session.

    A timeout means the warm server is usually busy with an active voice session.
    Falling back locally in that case creates a second popup/window for the same
    wakephrase. Only fall back when the server is unreachable, not merely busy.
    """
    return not isinstance(exc, socket.timeout)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Handle one native Okay Hermes wake activation")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to wakeword config YAML")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(Path(args.config).expanduser())
    setup_logging(cfg, verbose=args.verbose)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    if STOP.is_set():
        STOP.clear()
    activation = read_activation()
    if cfg.get("native_activation_server_enabled", False):
        try:
            return send_activation_to_server(activation, cfg)
        except Exception as exc:
            if not should_fallback_to_local_handler(exc):
                print(f"warm native activation server busy; not falling back locally: {exc}", file=sys.stderr)
                return 0
            print(f"warm native activation server unavailable; falling back locally: {exc}", file=sys.stderr)
    handle_activation(cfg, activation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
