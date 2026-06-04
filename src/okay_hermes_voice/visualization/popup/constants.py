"""Shared terminal-popup constants and status vocabulary."""
from __future__ import annotations

FINAL_STATUSES = {"done", "error", "cancelled"}
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
PIPELINE_STAGES = (
    ("wake", "wake"),
    ("record", "record"),
    ("transcript", "transcript"),
    ("route", "route"),
    ("answer", "answer"),
    ("tts", "TTS"),
    ("playback", "playback"),
)
PIPELINE_STAGE_INDEX = {stage: idx for idx, (stage, _label) in enumerate(PIPELINE_STAGES)}
STATUS_LABELS = {
    "wake": "wake · wakeword detected",
    "record": "record · recording request",
    "transcript": "transcript · speech to text",
    "route": "route · choosing handler",
    "answer": "answer · Hermes responding",
    "tts": "TTS · generating speech",
    "playback": "playback · playing response",
    "listening": "Listening for your request",
    "transcribing": "Transcribing speech",
    "thinking": "Hermes is thinking",
    "speaking": "Speaking response",
    "cancel_requested": "Stopping voice session",
    "done": "Done",
    "error": "Needs attention",
    "cancelled": "Cancelled",
}
STATUS_COLORS = {
    "wake": "\033[36m",
    "record": "\033[36m",
    "transcript": "\033[35m",
    "route": "\033[33m",
    "answer": "\033[33m",
    "tts": "\033[32m",
    "playback": "\033[32m",
    "listening": "\033[36m",
    "transcribing": "\033[35m",
    "thinking": "\033[33m",
    "speaking": "\033[32m",
    "cancel_requested": "\033[31m",
    "done": "\033[32m",
    "error": "\033[31m",
    "cancelled": "\033[31m",
}
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
ALT_SCREEN_ENTER = "\033[?1049h\033[?25l"
ALT_SCREEN_EXIT = "\033[?25h\033[?1049l"
MOUSE_CAPTURE_ENTER = "\033[?1007h\033[?1000h\033[?1006h"
MOUSE_CAPTURE_EXIT = "\033[?1006l\033[?1000l\033[?1007l"
POLL_INTERVAL_SECONDS = 0.2
SCROLL_KEY_SEQUENCES = {
    "\033[A": "up",
    "\033[B": "down",
    "\033[5~": "page_up",
    "\033[6~": "page_down",
    "\033[H": "home",
    "\033[F": "end",
    "\033OH": "home",
    "\033OF": "end",
}
SCROLL_KEY_CHARS = {
    "k": "up",
    "j": "down",
    "g": "home",
    "G": "end",
}
RENDER_FINGERPRINT_IGNORED_KEYS = {
    # These fields are useful for logs/daemon bookkeeping, but repainting only
    # because they changed makes the terminal scrollback fill with duplicate
    # visible frames.
    "activation_archive",
    "cancel_reason",
    "cancel_requested",
    "cancel_requested_at",
    "completed_at",
    "current_turn",
    "interaction_ack_text",
    "updated_at",
    "visualization_launch_error",
    "visualization_terminal",
}
