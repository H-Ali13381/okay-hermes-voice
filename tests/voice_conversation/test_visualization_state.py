from __future__ import annotations

import json
import os
import socket
from pathlib import Path

from okay_hermes_voice import voice_activation_popup as popup
from okay_hermes_voice import visualization as viz


def test_append_visualization_turn_preserves_conversation_history(tmp_path):
    state_path = tmp_path / "voice_state.json"
    viz.update_visualization_state(
        state_path,
        status="thinking",
        message="first turn",
        transcript="",
        response="",
        turns=[],
    )

    viz.append_visualization_turn(state_path, transcript="First request", response="First response")
    viz.append_visualization_turn(state_path, transcript="Second request", response="Second response")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["transcript"] == "Second request"
    assert state["response"] == "Second response"
    assert [turn["transcript"] for turn in state["turns"]] == ["First request", "Second request"]
    assert [turn["response"] for turn in state["turns"]] == ["First response", "Second response"]

def test_popup_render_shows_multiple_conversation_turns():
    rendered = popup.render(
        {
            "title": "Hermes Voice",
            "status": "listening",
            "message": "Listening for follow-up",
            "activated_at": 1,
            "updated_at": 2,
            "turns": [
                {"transcript": "First request", "response": "First response"},
                {"transcript": "Second request", "response": "Second response"},
            ],
        },
        tick=0,
        final_seen_at=None,
    )

    assert "Conversation" in rendered
    assert "You: First request" in rendered
    assert "Hermes: First response" in rendered
    assert "You: Second request" in rendered
    assert "Hermes: Second response" in rendered

def test_popup_ctrl_c_request_marks_state_for_daemon_cancel(tmp_path):
    state_path = tmp_path / "voice_state.json"
    state_path.write_text(json.dumps({"status": "listening", "message": "Listening"}), encoding="utf-8")

    popup.request_cancel(state_path, reason="ctrl_c")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["cancel_requested"] is True
    assert state["cancel_reason"] == "ctrl_c"
    assert state["status"] == "cancel_requested"
    assert "cancel_requested_at" in state


def test_launch_visualization_initializes_wake_pipeline_stage(monkeypatch, tmp_path):
    state_path = tmp_path / "voice_state.json"
    monkeypatch.setattr(viz, "_visualization_state_path", lambda: state_path)

    launched = viz.launch_visualization(
        {
            "visualization_enabled": True,
            "visualization_terminal": "off",
            "visualization_title": "Hermes Voice",
            "visualization_keep_open_seconds": 45.0,
        },
        probability=0.91,
    )

    assert launched == state_path
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "wake"
    assert state["pipeline_stage"] == "wake"
    assert "wake" in state["message"].casefold()


def test_popup_render_shows_voice_pipeline_progress():
    rendered = popup.render(
        {
            "title": "Hermes Voice",
            "status": "answer",
            "pipeline_stage": "answer",
            "message": "Hermes is producing the answer.",
            "activated_at": 1,
            "updated_at": 2,
        },
        tick=0,
        final_seen_at=None,
    )

    assert "Pipeline" in rendered
    for label in ("wake", "record", "transcript", "route", "answer", "TTS", "playback"):
        assert label in rendered


def test_popup_state_fingerprint_ignores_metadata_that_is_not_rendered():
    visible_state = {
        "title": "Hermes Voice",
        "status": "listening",
        "message": "Listening for a follow-up. Say “close” to end voice mode.",
        "activated_at": 1,
        "updated_at": 2,
        "keep_open_seconds": 45.0,
        "current_turn": 1,
        "activation_archive": {"path": "/tmp/activation-a"},
        "visualization_terminal": "kitty",
        "visualization_launch_error": "",
        "interaction_ack_text": "Okay, I’m on it.",
        "cancel_requested": False,
        "cancel_reason": "",
        "turns": [
            {
                "turn": 1,
                "transcript": "First request",
                "response": "First response",
                "completed_at": 10,
            }
        ],
    }
    metadata_only_update = {
        **visible_state,
        "updated_at": 99,
        "current_turn": 2,
        "activation_archive": {"path": "/tmp/activation-b"},
        "visualization_terminal": "konsole",
        "visualization_launch_error": "launcher metadata changed",
        "interaction_ack_text": "Different non-rendered ack metadata",
        "turns": [{**visible_state["turns"][0], "completed_at": 999}],
    }
    visible_update = {**visible_state, "message": "Hermes is thinking now."}

    assert popup.state_fingerprint(visible_state) == popup.state_fingerprint(metadata_only_update)
    assert popup.state_fingerprint(visible_state) != popup.state_fingerprint(visible_update)


def test_popup_render_uses_internal_scroll_viewport_for_long_output(monkeypatch):
    monkeypatch.setattr(popup.shutil, "get_terminal_size", lambda fallback: os.terminal_size((72, 12)))
    response = "\n".join(f"response line {idx:02d}" for idx in range(1, 31))
    state = {
        "title": "Hermes Voice",
        "status": "done",
        "message": "Voice request complete.",
        "activated_at": 1,
        "updated_at": 2,
        "response": response,
        "keep_open_seconds": 0,
    }

    top = popup.render(state, tick=0, final_seen_at=0.0, scroll_offset=0)
    bottom = popup.render(state, tick=0, final_seen_at=0.0, scroll_offset=999)

    assert "response line 01" in top
    assert "response line 30" not in top
    assert "response line 30" in bottom
    assert "response line 01" not in bottom
    assert "Scroll:" in top


def test_popup_read_keypress_maps_sgr_mouse_wheel_events_to_scroll_actions(monkeypatch):
    class FakeStdin:
        def __init__(self, text):
            self.chars = list(text)

        def isatty(self):
            return True

        def read(self, _size):
            return self.chars.pop(0)

    def read_action(sequence):
        fake_stdin = FakeStdin(sequence)

        def fake_select(readers, _writers, _errors, _timeout):
            return (readers, [], []) if fake_stdin.chars else ([], [], [])

        monkeypatch.setattr(popup.sys, "stdin", fake_stdin)
        monkeypatch.setattr(popup.select, "select", fake_select)
        return popup.read_keypress()

    assert read_action("\033[<64;12;4M") == "up"
    assert read_action("\033[<65;12;4M") == "down"


def test_popup_run_uses_alternate_screen_so_full_redraws_do_not_expand_scrollback(monkeypatch, tmp_path):
    state_path = tmp_path / "voice_state.json"
    final_state = {
        "title": "Hermes Voice",
        "status": "done",
        "message": "Voice request complete.",
        "activated_at": 1,
        "updated_at": 2,
        "keep_open_seconds": 0.1,
    }

    class CapturedStdout:
        def __init__(self):
            self.writes = []

        def write(self, text):
            self.writes.append(text)
            return len(text)

        def flush(self):
            pass

    clock = {"now": 0.0}

    def fake_sleep(seconds):
        clock["now"] += seconds

    captured_stdout = CapturedStdout()
    monkeypatch.setattr(popup, "load_state", lambda _path: final_state)
    monkeypatch.setattr(popup.sys, "stdout", captured_stdout)
    monkeypatch.setattr(popup.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(popup.time, "sleep", fake_sleep)

    assert popup.run(state_path) == 0

    output = "".join(captured_stdout.writes)
    assert output.startswith("\033[?1049h\033[?25l")
    assert "Voice request complete." in output
    assert output.endswith("\033[?25h\033[?1049l")


def test_popup_run_enables_mouse_capture_so_wheel_scroll_stays_inside_tui(monkeypatch, tmp_path):
    state_path = tmp_path / "voice_state.json"
    final_state = {
        "title": "Hermes Voice",
        "status": "done",
        "message": "Voice request complete.",
        "activated_at": 1,
        "updated_at": 2,
        "keep_open_seconds": 0.1,
    }

    class CapturedStdout:
        def __init__(self):
            self.writes = []

        def write(self, text):
            self.writes.append(text)
            return len(text)

        def flush(self):
            pass

    clock = {"now": 0.0}

    def fake_sleep(seconds):
        clock["now"] += seconds

    captured_stdout = CapturedStdout()
    monkeypatch.setattr(popup, "load_state", lambda _path: final_state)
    monkeypatch.setattr(popup.sys, "stdout", captured_stdout)
    monkeypatch.setattr(popup.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(popup.time, "sleep", fake_sleep)

    assert popup.run(state_path) == 0

    output = "".join(captured_stdout.writes)
    first_frame_at = output.index("\033[2J\033[H")
    assert "\033[?1007h" in output[:first_frame_at]
    assert "\033[?1000h" in output[:first_frame_at]
    assert "\033[?1006h" in output[:first_frame_at]
    assert "\033[?1006l" in output[first_frame_at:]
    assert "\033[?1000l" in output[first_frame_at:]
    assert "\033[?1007l" in output[first_frame_at:]


def test_popup_run_redraws_when_scroll_key_changes_viewport_without_state_change(monkeypatch, tmp_path):
    state_path = tmp_path / "voice_state.json"
    done_state = {
        "title": "Hermes Voice",
        "status": "done",
        "message": "Voice request complete.",
        "activated_at": 1,
        "updated_at": 2,
        "response": "\n".join(f"response line {idx:02d}" for idx in range(1, 31)),
        "keep_open_seconds": 0.4,
    }

    class CapturedStdout:
        def __init__(self):
            self.writes = []

        def write(self, text):
            self.writes.append(text)
            return len(text)

        def flush(self):
            pass

    clock = {"now": 0.0}

    def fake_sleep(seconds):
        clock["now"] += seconds

    keys = iter([None, "end", None, None])

    def fake_read_keypress():
        return next(keys, None)

    captured_stdout = CapturedStdout()
    monkeypatch.setattr(popup.shutil, "get_terminal_size", lambda fallback: os.terminal_size((72, 12)))
    monkeypatch.setattr(popup, "load_state", lambda _path: done_state)
    monkeypatch.setattr(popup, "read_keypress", fake_read_keypress, raising=False)
    monkeypatch.setattr(popup.sys, "stdout", captured_stdout)
    monkeypatch.setattr(popup.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(popup.time, "sleep", fake_sleep)

    assert popup.run(state_path) == 0

    frame_writes = [write for write in captured_stdout.writes if write.startswith("\033[2J\033[H")]
    assert len(frame_writes) >= 2
    assert "response line 01" in frame_writes[0]
    assert "response line 30" not in frame_writes[0]
    assert "response line 30" in frame_writes[1]


def test_popup_run_redraws_active_spinner_without_waiting_for_state_change(monkeypatch, tmp_path):
    state_path = tmp_path / "voice_state.json"
    active_state = {
        "title": "Hermes Voice",
        "status": "answer",
        "pipeline_stage": "answer",
        "message": "answer: Hermes is producing the response…",
        "activated_at": 1,
        "updated_at": 2,
        "keep_open_seconds": 0.1,
    }
    done_state = {
        **active_state,
        "status": "done",
        "message": "Voice request complete.",
        "turns": [{"transcript": "Summarize", "response": "Long response " * 80}],
        "updated_at": 3,
    }
    states = [active_state, active_state, active_state, done_state]
    load_calls = {"count": 0}

    def fake_load_state(_path):
        idx = min(load_calls["count"], len(states) - 1)
        load_calls["count"] += 1
        return states[idx]

    class CapturedStdout:
        def __init__(self):
            self.writes = []

        def write(self, text):
            self.writes.append(text)
            return len(text)

        def flush(self):
            pass

    clock = {"now": 0.0}

    def fake_sleep(seconds):
        clock["now"] += seconds

    captured_stdout = CapturedStdout()
    monkeypatch.setattr(popup, "load_state", fake_load_state)
    monkeypatch.setattr(popup.sys, "stdout", captured_stdout)
    monkeypatch.setattr(popup.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(popup.time, "sleep", fake_sleep)

    assert popup.run(state_path) == 0

    frame_writes = [write for write in captured_stdout.writes if write.startswith("\033[2J\033[H")]
    active_status_lines = [
        next(line for line in frame.splitlines() if "Hermes responding" in line)
        for frame in frame_writes
        if "Hermes responding" in frame
    ]
    assert len(active_status_lines) >= 3
    assert "⠋" in active_status_lines[0]
    assert "⠙" in active_status_lines[1]
    assert "⠹" in active_status_lines[2]
    assert "Long response" in frame_writes[-1]


def test_visualization_terminal_auto_prefers_kitty_even_on_kde(monkeypatch, tmp_path):
    script = tmp_path / "popup.py"
    script.write_text("print('popup')", encoding="utf-8")
    state_path = tmp_path / "state.json"

    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setattr(
        viz.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"kitty", "konsole"} else None,
    )

    cmd = viz._visualization_terminal_command(
        {
            "visualization_terminal": "auto",
            "visualization_title": "Hermes Voice",
            "visualization_script": str(script),
        },
        state_path,
    )

    assert cmd is not None
    assert cmd[0] == "/usr/bin/kitty"

def test_visualization_terminal_kitty_does_not_use_detach(monkeypatch, tmp_path):
    script = tmp_path / "popup.py"
    script.write_text("print('popup')", encoding="utf-8")
    state_path = tmp_path / "state.json"

    monkeypatch.setattr(viz.shutil, "which", lambda name: "/usr/bin/kitty" if name == "kitty" else None)

    cmd = viz._visualization_terminal_command(
        {
            "visualization_terminal": "kitty",
            "visualization_title": "Hermes Voice",
            "visualization_script": str(script),
        },
        state_path,
    )

    assert cmd is not None
    assert cmd[0] == "/usr/bin/kitty"
    assert "--detach" not in cmd


def test_visualization_terminal_kitty_disables_scrollback_for_tui_popup(monkeypatch, tmp_path):
    script = tmp_path / "popup.py"
    script.write_text("print('popup')", encoding="utf-8")
    state_path = tmp_path / "state.json"

    monkeypatch.setattr(viz.shutil, "which", lambda name: "/usr/bin/kitty" if name == "kitty" else None)

    cmd = viz._visualization_terminal_command(
        {
            "visualization_terminal": "kitty",
            "visualization_title": "Hermes Voice",
            "visualization_script": str(script),
        },
        state_path,
    )

    assert cmd is not None
    assert cmd[0] == "/usr/bin/kitty"
    assert "--override" in cmd
    override_idx = cmd.index("--override")
    assert cmd[override_idx + 1] == "scrollback_lines=0"
    assert str(script) in cmd
    assert override_idx < cmd.index(str(script))


def test_launch_visualization_falls_back_when_first_terminal_fails(monkeypatch, tmp_path):
    script = tmp_path / "popup.py"
    script.write_text("print('popup')", encoding="utf-8")
    state_path = tmp_path / "voice_state.json"
    launched = []

    monkeypatch.setattr(viz, "_visualization_state_path", lambda: state_path)
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "")
    monkeypatch.setattr(
        viz.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"kitty", "konsole"} else None,
    )

    class FakeProc:
        returncode = None

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            return "", ""

    def fake_popen(cmd, **_kwargs):
        launched.append(cmd[0])
        if Path(cmd[0]).name == "kitty":
            raise OSError("simulated kitty launch failure")
        return FakeProc()

    monkeypatch.setattr(viz.subprocess, "Popen", fake_popen)

    result = viz.launch_visualization(
        {
            "visualization_enabled": True,
            "visualization_terminal": "auto",
            "visualization_title": "Hermes Voice",
            "visualization_script": str(script),
            "visualization_keep_open_seconds": 45.0,
        },
        probability=0.9,
    )

    assert result == state_path
    assert launched == ["/usr/bin/kitty", "/usr/bin/konsole"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["visualization_terminal"] == "konsole"
    assert state["visualization_launch_error"] == ""


def test_launch_visualization_infers_wayland_display_for_systemd_service_env(monkeypatch, tmp_path):
    script = tmp_path / "popup.py"
    script.write_text("print('popup')", encoding="utf-8")
    state_path = tmp_path / "voice_state.json"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    wayland_socket = runtime_dir / "wayland-0"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(wayland_socket))
    captured_env = {}

    monkeypatch.setattr(viz, "_visualization_state_path", lambda: state_path)
    monkeypatch.setattr(viz.shutil, "which", lambda name: "/usr/bin/kitty" if name == "kitty" else None)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)

    class FakeProc:
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            raise TimeoutError

    def fake_popen(_cmd, **kwargs):
        captured_env.update(kwargs["env"])
        return FakeProc()

    try:
        monkeypatch.setattr(viz.subprocess, "Popen", fake_popen)

        result = viz.launch_visualization(
            {
                "visualization_enabled": True,
                "visualization_terminal": "kitty",
                "visualization_title": "Hermes Voice",
                "visualization_script": str(script),
                "visualization_keep_open_seconds": 45.0,
            },
            probability=0.9,
        )
    finally:
        sock.close()

    assert result == state_path
    assert captured_env["WAYLAND_DISPLAY"] == "wayland-0"
    assert captured_env["XDG_SESSION_TYPE"] == "wayland"


def test_launch_visualization_falls_back_when_terminal_exits_after_grace(monkeypatch, tmp_path):
    script = tmp_path / "popup.py"
    script.write_text("print('popup')", encoding="utf-8")
    state_path = tmp_path / "voice_state.json"
    launched = []

    monkeypatch.setattr(viz, "_visualization_state_path", lambda: state_path)
    monkeypatch.setattr(
        viz.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"kitty", "konsole"} else None,
    )

    class DelayedFailProc:
        returncode = 1

        def poll(self):
            return None

        def wait(self, timeout=None):
            return self.returncode

        def communicate(self, timeout=None):
            return "", "GLFW initialization failed"

    class AliveProc:
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            raise TimeoutError

    def fake_popen(cmd, **_kwargs):
        launched.append(Path(cmd[0]).name)
        if Path(cmd[0]).name == "kitty":
            return DelayedFailProc()
        return AliveProc()

    monkeypatch.setattr(viz.subprocess, "Popen", fake_popen)

    result = viz.launch_visualization(
        {
            "visualization_enabled": True,
            "visualization_terminal": "auto",
            "visualization_title": "Hermes Voice",
            "visualization_script": str(script),
            "visualization_keep_open_seconds": 45.0,
        },
        probability=0.9,
    )

    assert result == state_path
    assert launched == ["kitty", "konsole"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["visualization_terminal"] == "konsole"
    assert state["visualization_launch_error"] == ""


def test_launch_visualization_reaps_terminal_that_exits_after_delegating(monkeypatch, tmp_path):
    script = tmp_path / "popup.py"
    script.write_text("print('popup')", encoding="utf-8")
    state_path = tmp_path / "voice_state.json"

    monkeypatch.setattr(viz, "_visualization_state_path", lambda: state_path)
    monkeypatch.setattr(viz.shutil, "which", lambda name: "/usr/bin/kitty" if name == "kitty" else None)

    class FakeProc:
        returncode = 0

        def __init__(self):
            self.waited = False

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.waited = True
            return self.returncode

    fake_proc = FakeProc()
    monkeypatch.setattr(viz.subprocess, "Popen", lambda *_args, **_kwargs: fake_proc)

    result = viz.launch_visualization(
        {
            "visualization_enabled": True,
            "visualization_terminal": "kitty",
            "visualization_title": "Hermes Voice",
            "visualization_script": str(script),
            "visualization_keep_open_seconds": 45.0,
        },
        probability=0.9,
    )

    assert result == state_path
    assert fake_proc.waited is True


def test_daemon_detects_visualization_cancel_request(tmp_path):
    state_path = tmp_path / "voice_state.json"
    viz.update_visualization_state(state_path, status="listening")

    assert not viz.is_visualization_cancel_requested(state_path)

    popup.request_cancel(state_path, reason="ctrl_c")

    assert viz.is_visualization_cancel_requested(state_path)
