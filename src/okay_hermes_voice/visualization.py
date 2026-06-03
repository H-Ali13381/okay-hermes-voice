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

VISUALIZATION_LAUNCH_GRACE_SECONDS = 0.35


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


def _infer_wayland_display(env: Dict[str, str]) -> Optional[str]:
    runtime_dir = env.get("XDG_RUNTIME_DIR")
    if not runtime_dir:
        return None
    try:
        candidates = sorted(
            path.name
            for path in Path(runtime_dir).iterdir()
            if path.name.startswith("wayland-") and not path.name.endswith(".lock") and path.is_socket()
        )
    except Exception:
        return None
    return candidates[0] if candidates else None


def _visualization_launch_env(base_env: Dict[str, str]) -> Dict[str, str]:
    env = dict(base_env)
    env.setdefault("HERMES_HOME", str(HERMES_HOME))
    env.setdefault("HERMES_REPO", str(HERMES_REPO))
    if not env.get("WAYLAND_DISPLAY"):
        wayland_display = _infer_wayland_display(env)
        if wayland_display:
            env["WAYLAND_DISPLAY"] = wayland_display
            if not env.get("XDG_SESSION_TYPE"):
                env["XDG_SESSION_TYPE"] = "wayland"
    return env


def _visualization_command_for_terminal(exe: str, title: str, program: List[str]) -> List[str]:
    name = Path(exe).name
    if name == "kitty":
        # Popen below is already non-blocking. Avoid kitty --detach so launch
        # failures stay attached to the process we start instead of being hidden
        # behind a daemonizing parent that exits before the window is usable.
        # The popup owns an internal TUI viewport, so disable kitty scrollback to
        # stop mouse-wheel gestures from escaping into an expanding buffer.
        return [
            exe,
            "--title",
            title,
            "--class",
            "hermes-voice",
            "--override",
            "scrollback_lines=0",
            *program,
        ]
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


def _communicate_or_wait(
    proc: subprocess.Popen[Any],
    timeout: Optional[float] = None,
) -> tuple[str, str, Optional[int]]:
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return str(stdout or ""), str(stderr or ""), proc.returncode
    except AttributeError:
        try:
            returncode = proc.wait(timeout=timeout)
        except TypeError:
            returncode = proc.wait()
        return "", "", returncode


def _short_process_output(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines)[-500:]


def _visualization_failure_message(
    terminal_label: str,
    returncode: Optional[int],
    stdout: str,
    stderr: str,
) -> str:
    output = _short_process_output(stderr) or _short_process_output(stdout)
    message = f"{terminal_label} exited immediately with status {returncode}"
    return f"{message}: {output}" if output else message


def _reap_visualization_process(proc: subprocess.Popen[Any], terminal_label: str) -> None:
    """Wait for an accepted popup terminal process so it cannot become a zombie."""
    try:
        stdout, stderr, returncode = _communicate_or_wait(proc)
        if returncode not in {0, None}:
            LOG.debug(
                "Voice visualization terminal %s exited with status %s: %s",
                terminal_label,
                returncode,
                _short_process_output(stderr) or _short_process_output(stdout),
            )
        else:
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
        status="wake",
        pipeline_stage="wake",
        message="wake: wakeword detected. Preparing to record your request…",
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

    env = _visualization_launch_env(os.environ.copy())
    launch_grace_seconds = float(cfg.get("visualization_launch_grace_seconds") or VISUALIZATION_LAUNCH_GRACE_SECONDS)
    last_error = ""
    for cmd in commands:
        terminal_label = Path(cmd[0]).name
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path.home()),
                env=env,
                start_new_session=True,
                text=True,
            )
            try:
                stdout, stderr, returncode = _communicate_or_wait(proc, timeout=launch_grace_seconds)
            except (subprocess.TimeoutExpired, TimeoutError):
                update_visualization_state(
                    state_path,
                    visualization_terminal=terminal_label,
                    visualization_launch_error="",
                )
                LOG.info("Launched voice visualization: %s state=%s", cmd[0], state_path)
                _watch_visualization_process(proc, terminal_label)
                return state_path
            if returncode not in {0, None}:
                last_error = _visualization_failure_message(terminal_label, returncode, stdout, stderr)
                update_visualization_state(state_path, visualization_launch_error=last_error)
                LOG.warning("Voice visualization launch failed: %s", last_error)
                continue
            update_visualization_state(
                state_path,
                visualization_terminal=terminal_label,
                visualization_launch_error="",
            )
            LOG.info("Launched voice visualization: %s state=%s", cmd[0], state_path)
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
