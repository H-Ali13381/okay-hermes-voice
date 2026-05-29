from __future__ import annotations

import json
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
