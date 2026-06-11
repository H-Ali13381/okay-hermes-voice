from __future__ import annotations

import json
import sys
from pathlib import Path

from okay_hermes_voice import native_activation_handler

REPO_ROOT = Path(__file__).parents[2]


def test_user_service_execs_native_binary_not_python_launcher():
    service = (REPO_ROOT / "systemd" / "hermes-wakeword.service").read_text(encoding="utf-8")
    exec_start = next(line for line in service.splitlines() if line.startswith("ExecStart="))

    assert "okay-hermes-wake-listener" in exec_start
    assert "--activation-config" in exec_start
    assert "python" not in exec_start.lower()
    assert "native_wakeword_launcher" not in exec_start
    assert "--handler-command" not in exec_start


def test_native_activation_handler_reads_activation_json(monkeypatch):
    monkeypatch.setattr(sys, "stdin", type("FakeStdin", (), {"read": lambda self: json.dumps({"probability": 0.8})})())

    assert native_activation_handler.read_activation() == {"probability": 0.8}
