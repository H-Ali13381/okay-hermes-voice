#!/usr/bin/env python3
"""CLI entrypoint for the always-on "Okay Hermes" wakeword daemon.

This root module owns the daemon story: parse CLI args, load config, warm
runtime dependencies, wait for wake activations, hand each activation to the
conversation flow, and re-arm. Audio, popup, archive, routing, playback, and
Hermes runtime mechanics live in deeper semantic modules.
"""
from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .activation_archive import format_activation_latency_summary, summarize_activation_archives
from .activation_flow import VOICE_SESSION_CANCELLED, handle_activation
from .audio import list_devices, model_session, prewarm_stt, smoke_test, wait_for_wake
from .daemon_config import CONFIG_PATH, DEFAULT_CONFIG, LOG, STOP, load_config, setup_logging, signal_handler
from .hermes_runtime import prewarm_hermes
from .interaction_router import AckTemplate, RouteTarget, VoiceRequestPlan
from .visualization import visualization_test

__all__ = ["main", "post_activation_cooldown_seconds"]


def post_activation_cooldown_seconds(cfg: Dict[str, Any], session_result: Any) -> float:
    """Return the re-arm delay after a voice activation completes."""
    if session_result == VOICE_SESSION_CANCELLED:
        raw = cfg.get("cancel_cooldown_seconds")
        if raw is None or raw == "":
            raw = DEFAULT_CONFIG.get("cancel_cooldown_seconds", 0.0)
    else:
        raw = cfg.get("cooldown_seconds")
        if raw is None or raw == "":
            raw = DEFAULT_CONFIG.get("cooldown_seconds", 2.5)
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0 if session_result == VOICE_SESSION_CANCELLED else float(DEFAULT_CONFIG.get("cooldown_seconds", 2.5))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Always-on Okay Hermes wakeword daemon")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to wakeword config YAML")
    parser.add_argument("--smoke-test", action="store_true", help="Load ONNX model and run zero-audio inference, then exit")
    parser.add_argument("--visualization-test", metavar="TEXT", help="Open the popup visualizer with a fake transcript, then exit")
    parser.add_argument("--activation-summary", nargs="?", const="", metavar="DIR", help="Print Phase 0 latency summary from activation archive JSON files, then exit")
    parser.add_argument("--summary-json", action="store_true", help="Print activation summary as JSON instead of terminal text")
    parser.add_argument("--list-devices", action="store_true", help="Print PortAudio devices, then exit")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.activation_summary is not None and args.activation_summary:
        summary = summarize_activation_archives(Path(args.activation_summary).expanduser())
        if args.summary_json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(format_activation_latency_summary(summary), end="")
        return 0

    cfg = load_config(Path(args.config).expanduser())
    if args.activation_summary is not None:
        summary_dir = Path(str(cfg.get("activation_archive_dir") or DEFAULT_CONFIG["activation_archive_dir"])).expanduser()
        summary = summarize_activation_archives(summary_dir)
        if args.summary_json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(format_activation_latency_summary(summary), end="")
        return 0
    if args.list_devices:
        return list_devices()
    if args.smoke_test:
        return smoke_test(cfg)
    if args.visualization_test:
        return visualization_test(cfg, args.visualization_test)

    setup_logging(cfg, verbose=args.verbose)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    LOG.info("Starting Okay Hermes wakeword daemon")
    LOG.info("Config path: %s", args.config)
    session, input_name, output_name = model_session(cfg["model_path"])
    prewarm_stt(cfg)
    prewarm_hermes(cfg)

    while not STOP.is_set():
        try:
            activation = wait_for_wake(cfg, session, input_name, output_name)
            if STOP.is_set() or activation is None:
                break
            session_result = handle_activation(cfg, activation)
            cooldown = post_activation_cooldown_seconds(cfg, session_result)
            if cooldown > 0:
                LOG.info("Cooldown %.1fs", cooldown)
                STOP.wait(cooldown)
        except KeyboardInterrupt:
            STOP.set()
        except Exception as exc:
            LOG.exception("Wakeword loop error: %s", exc)
            STOP.wait(5.0)

    LOG.info("Wakeword daemon stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
