"""Environment construction for launched popup terminals."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ...daemon_config import HERMES_HOME, HERMES_REPO


def _infer_wayland_display(env: Dict[str, str]) -> Optional[str]:
    runtime_dir = env.get("XDG_RUNTIME_DIR")
    if not runtime_dir:
        return None
    try:
        candidates = sorted(path.name for path in Path(runtime_dir).iterdir() if path.name.startswith("wayland-") and not path.name.endswith(".lock") and path.is_socket())
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


__all__ = ["_infer_wayland_display", "_visualization_launch_env"]
