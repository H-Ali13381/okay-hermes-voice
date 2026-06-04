"""JSON state boundary for the terminal popup visualizer."""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..activation_archive import update_activation_archive_metadata
from ..daemon_config import LOG


def _visualization_state_path() -> Path:
    out_dir = Path(tempfile.gettempdir()) / "hermes_voice_wakeword"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return out_dir / f"voice_visual_{stamp}_{os.getpid()}_{time.monotonic_ns()}.json"


def update_visualization_state(path: Optional[Path], **updates: Any) -> None:
    """Atomically update the state consumed by the popup terminal visualizer."""
    if path is None:
        return
    try:
        state: Dict[str, Any] = {}
        if path.exists():
            with contextlib.suppress(Exception):
                state = json.loads(path.read_text(encoding="utf-8"))
        state.update(updates)
        state["updated_at"] = time.time()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:
        LOG.warning("Could not update visualization state %s: %s", path, exc)


def read_visualization_state(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        LOG.warning("Could not read visualization state %s: %s", path, exc)
        return {}


def is_visualization_cancel_requested(path: Optional[Path]) -> bool:
    return bool(read_visualization_state(path).get("cancel_requested"))


def visualization_cancel_reason(path: Optional[Path]) -> str:
    state = read_visualization_state(path)
    return str(state.get("cancel_reason") or "terminal_cancel")


def finish_cancelled_voice_session(
    visual_state: Optional[Path],
    activation_archive: Optional[Dict[str, Any]],
    archive_turns: List[Dict[str, Any]],
    reason: str,
) -> None:
    update_visualization_state(
        visual_state,
        status="cancelled",
        message="Voice session cancelled from the Hermes Voice terminal.",
        error="",
        cancel_requested=True,
        cancel_reason=reason,
    )
    update_activation_archive_metadata(
        activation_archive,
        status="cancelled_by_terminal",
        close_reason=reason,
        cancel_reason=reason,
        turns=archive_turns,
    )
    LOG.info("Voice conversation cancelled by terminal request: %s", reason)


def append_visualization_turn(path: Optional[Path], transcript: str, response: str) -> None:
    """Append a completed user/Hermes voice turn to the popup state."""
    if path is None:
        return
    try:
        state: Dict[str, Any] = {}
        if path.exists():
            with contextlib.suppress(Exception):
                state = json.loads(path.read_text(encoding="utf-8"))
        turns = state.get("turns")
        if not isinstance(turns, list):
            turns = []
        turns.append({
            "turn": len(turns) + 1,
            "transcript": transcript,
            "response": response,
            "completed_at": time.time(),
        })
        update_visualization_state(path, turns=turns, transcript=transcript, response=response)
    except Exception as exc:
        LOG.warning("Could not append visualization turn %s: %s", path, exc)
