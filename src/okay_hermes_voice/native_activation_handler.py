"""Short-lived activation handler for the native wakeword listener."""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Iterable, Optional

from .activation_flow import handle_activation
from .daemon_config import CONFIG_PATH, STOP, load_config, setup_logging, signal_handler

__all__ = ["main", "read_activation"]


def read_activation() -> dict[str, object]:
    payload = sys.stdin.read().strip()
    if not payload:
        raise ValueError("native activation handler expected JSON on stdin")
    activation = json.loads(payload)
    if not isinstance(activation, dict):
        raise ValueError("native activation JSON must be an object")
    return activation


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
    handle_activation(cfg, activation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
