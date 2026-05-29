"""Terminal popup visualization state and launcher helpers."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .activation_archive import update_activation_archive_metadata
from .daemon_config import HERMES_HOME, HERMES_REPO, LOG, setup_logging

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


def _visualization_terminal_candidates(terminal_name: str) -> List[str]:
    if terminal_name.lower() != "auto":
        return [terminal_name]

    return [
        "kitty",
        "konsole",
        "alacritty",
        "wezterm",
        "foot",
        "gnome-terminal",
        "kgx",
        "xfce4-terminal",
        "xterm",
    ]


def _visualization_command_for_terminal(exe: str, title: str, program: List[str]) -> List[str]:
    name = Path(exe).name
    if name == "kitty":
        # Popen below is already non-blocking. Avoid kitty --detach so launch
        # failures stay attached to the process we start instead of being hidden
        # behind a daemonizing parent that exits before the window is usable.
        return [exe, "--title", title, "--class", "hermes-voice", *program]
    if name == "konsole":
        return [exe, "--title", title, "-e", *program]
    if name == "alacritty":
        return [exe, "--title", title, "-e", *program]
    if name == "wezterm":
        return [exe, "start", "--", *program]
    if name == "foot":
        return [exe, "--title", title, *program]
    if name in {"gnome-terminal", "kgx", "xfce4-terminal"}:
        return [exe, "--title", title, "--", *program]
    if name == "xterm":
        return [exe, "-T", title, "-e", *program]
    return [exe, *program]


def _visualization_terminal_commands(cfg: Dict[str, Any], state_path: Path) -> List[List[str]]:
    terminal_name = str(cfg.get("visualization_terminal") or "auto").strip()
    if terminal_name.lower() in {"", "off", "none", "false"}:
        return []

    title = str(cfg.get("visualization_title") or "Hermes Voice")
    script = Path(str(cfg.get("visualization_script") or "")).expanduser()
    if not script.exists():
        LOG.warning("Visualization script missing: %s", script)
        return []

    program = [sys.executable, str(script), "--state", str(state_path)]
    candidates = _visualization_terminal_candidates(terminal_name)
    commands: List[List[str]] = []
    for candidate in candidates:
        exe = candidate if "/" in candidate else shutil.which(candidate)
        if not exe:
            continue
        commands.append(_visualization_command_for_terminal(exe, title, program))

    if not commands:
        LOG.warning("No supported terminal emulator found for visualization candidates=%s", candidates)
    return commands


def _visualization_terminal_command(cfg: Dict[str, Any], state_path: Path) -> Optional[List[str]]:
    commands = _visualization_terminal_commands(cfg, state_path)
    return commands[0] if commands else None


def _reap_visualization_process(proc: subprocess.Popen[Any], terminal_label: str) -> None:
    """Wait for an accepted popup terminal process so it cannot become a zombie."""
    try:
        returncode = proc.wait()
        LOG.debug("Voice visualization terminal %s exited with status %s", terminal_label, returncode)
    except Exception as exc:
        LOG.debug("Could not reap voice visualization terminal %s: %s", terminal_label, exc)


def _watch_visualization_process(proc: subprocess.Popen[Any], terminal_label: str) -> None:
    thread = threading.Thread(
        target=_reap_visualization_process,
        args=(proc, terminal_label),
        name=f"hermes-voice-{terminal_label}-reaper",
        daemon=True,
    )
    thread.start()


def launch_visualization(cfg: Dict[str, Any], probability: float) -> Optional[Path]:
    """Open a non-blocking terminal window for the current voice activation."""
    if not cfg.get("visualization_enabled", True):
        return None

    state_path = _visualization_state_path()
    update_visualization_state(
        state_path,
        title=str(cfg.get("visualization_title") or "Hermes Voice"),
        status="listening",
        message="Wakeword detected. Listening for your request…",
        probability=float(probability),
        activated_at=time.time(),
        keep_open_seconds=float(cfg.get("visualization_keep_open_seconds") or 45.0),
        transcript="",
        response="",
        turns=[],
        error="",
        cancel_requested=False,
        cancel_reason="",
    )

    commands = _visualization_terminal_commands(cfg, state_path)
    if not commands:
        return state_path

    env = os.environ.copy()
    env.setdefault("HERMES_HOME", str(HERMES_HOME))
    env.setdefault("HERMES_REPO", str(HERMES_REPO))
    last_error = ""
    for cmd in commands:
        terminal_label = Path(cmd[0]).name
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(Path.home()),
                env=env,
                start_new_session=True,
            )
            update_visualization_state(
                state_path,
                visualization_terminal=terminal_label,
                visualization_launch_error="",
            )
            LOG.info("Launched voice visualization: %s state=%s", cmd[0], state_path)
            # Catch terminals that fail immediately, while accepting terminals
            # that return zero after delegating to an existing server process.
            time.sleep(0.05)
            status = proc.poll()
            if status is not None and status != 0:
                _reap_visualization_process(proc, terminal_label)
                last_error = f"{terminal_label} exited immediately with status {proc.returncode}"
                update_visualization_state(state_path, visualization_launch_error=last_error)
                LOG.warning("Voice visualization launch failed: %s", last_error)
                continue
            if status == 0:
                _reap_visualization_process(proc, terminal_label)
            else:
                _watch_visualization_process(proc, terminal_label)
            return state_path
        except Exception as exc:
            last_error = f"{terminal_label}: {exc}"
            update_visualization_state(state_path, visualization_launch_error=last_error)
            LOG.warning("Could not launch voice visualization with %s: %s", terminal_label, exc)

    if last_error:
        LOG.warning("All voice visualization terminal launch attempts failed: %s", last_error)
    return state_path


def visualization_test(cfg: Dict[str, Any], transcript: str) -> int:
    setup_logging(cfg, verbose=True)
    state_path = launch_visualization(cfg, probability=1.0)
    if state_path is None:
        print(json.dumps({"ok": False, "error": "visualization disabled"}, indent=2))
        return 1
    time.sleep(0.8)
    update_visualization_state(
        state_path,
        status="thinking",
        message="Visualization smoke test. Hermes would now handle the request.",
        transcript=transcript,
    )
    time.sleep(1.0)
    append_visualization_turn(
        state_path,
        transcript=transcript,
        response="Popup rendering is working. The live daemon will show your real spoken request and response here.",
    )
    update_visualization_state(
        state_path,
        status="done",
        message="Visualization smoke test complete.",
    )
    print(json.dumps({"ok": True, "state_path": str(state_path)}, indent=2))
    return 0
