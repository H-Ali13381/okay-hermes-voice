from __future__ import annotations

import json
import socket
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


def test_native_activation_handler_proxies_to_warm_server_when_enabled(monkeypatch, tmp_path):
    calls = []
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "native_activation_server_enabled: true\n"
        f"native_activation_socket: {tmp_path / 'handler.sock'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "stdin", type("FakeStdin", (), {"read": lambda self: json.dumps({"probability": 0.9})})())
    monkeypatch.setattr(native_activation_handler, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(native_activation_handler, "handle_activation", lambda *_args, **_kwargs: calls.append("local"))
    monkeypatch.setattr(
        native_activation_handler,
        "send_activation_to_server",
        lambda activation, cfg: calls.append((activation, cfg["native_activation_socket"])) or 0,
    )

    assert native_activation_handler.main(["--config", str(config_path)]) == 0
    assert calls == [({"probability": 0.9}, str(tmp_path / "handler.sock"))]


def test_native_activation_handler_does_not_fallback_locally_when_warm_server_is_busy(monkeypatch, tmp_path):
    calls = []
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "native_activation_server_enabled: true\n"
        f"native_activation_socket: {tmp_path / 'handler.sock'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "stdin", type("FakeStdin", (), {"read": lambda self: json.dumps({"probability": 0.9})})())
    monkeypatch.setattr(native_activation_handler, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(native_activation_handler, "handle_activation", lambda *_args, **_kwargs: calls.append("local"))
    monkeypatch.setattr(
        native_activation_handler,
        "send_activation_to_server",
        lambda _activation, _cfg: (_ for _ in ()).throw(socket.timeout("server busy")),
    )

    assert native_activation_handler.main(["--config", str(config_path)]) == 0
    assert calls == []


def test_native_activation_server_prewarms_before_accepting_connections(monkeypatch, tmp_path):
    from okay_hermes_voice import native_activation_server

    calls = []
    cfg = {
        "native_activation_socket": str(tmp_path / "handler.sock"),
        "prewarm_stt_on_start": True,
        "prewarm_hermes_on_start": True,
    }
    monkeypatch.setattr(native_activation_server, "prewarm_stt", lambda config: calls.append(("stt", config)))
    monkeypatch.setattr(native_activation_server, "prewarm_hermes", lambda config: calls.append(("hermes", config)))

    native_activation_server.prewarm_runtime(cfg)

    assert calls == [("stt", cfg), ("hermes", cfg)]


def test_native_activation_server_readiness_marker_is_socket_scoped(tmp_path):
    from okay_hermes_voice import native_activation_server

    cfg = {"native_activation_socket": str(tmp_path / "handler.sock")}

    assert native_activation_server.native_activation_ready_path(cfg) == tmp_path / "handler.ready"


def test_native_activation_server_writes_and_removes_readiness_marker(tmp_path):
    from okay_hermes_voice import native_activation_server

    cfg = {"native_activation_socket": str(tmp_path / "handler.sock")}
    ready_path = tmp_path / "handler.ready"

    native_activation_server.mark_ready(cfg)
    assert ready_path.read_text(encoding="utf-8").strip() == "ready"

    native_activation_server.clear_ready(cfg)
    assert not ready_path.exists()


def test_native_activation_server_drops_queued_activation_from_active_session(monkeypatch):
    from okay_hermes_voice import native_activation_server

    calls = []
    state = native_activation_server.ActivationServerState()
    monkeypatch.setattr(
        native_activation_server,
        "handle_activation",
        lambda _cfg, activation: calls.append(activation["detected_at"]) or "completed",
    )

    first = native_activation_server.handle_activation_payload(
        {"cooldown_seconds": 0.0},
        json.dumps({"probability": 0.9, "detected_at": 100.0}).encode("utf-8"),
        state=state,
    )
    queued = native_activation_server.handle_activation_payload(
        {"cooldown_seconds": 0.0},
        json.dumps({"probability": 0.8, "detected_at": 100.5}).encode("utf-8"),
        state=state,
    )

    assert first == {"ok": True, "result": "completed"}
    assert queued == {"ok": True, "ignored": True, "reason": "stale_activation"}
    assert calls == [100.0]
