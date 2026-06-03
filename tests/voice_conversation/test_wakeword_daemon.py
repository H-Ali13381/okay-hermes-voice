from __future__ import annotations

import json

from okay_hermes_voice import activation_flow as flow
from okay_hermes_voice import wakeword_daemon as wake


def test_main_prints_activation_latency_summary_json_without_starting_daemon(tmp_path, capsys):
    (tmp_path / "activation_one.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "benchmark_preset": "simple_chat",
                "turns": [
                    {
                        "turn": 1,
                        "response_source": "heavy_agent",
                        "timings": {"turn": 1, "answer_seconds": 1.25, "turn_seconds": 2.5},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert wake.main(["--activation-summary", str(tmp_path), "--summary-json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["archive_count"] == 1
    assert payload["turn_count"] == 1
    assert payload["timing_fields"]["answer_seconds"]["mean"] == 1.25


def test_main_rearms_immediately_after_terminal_cancel(monkeypatch):
    wait_calls = []
    activations = iter([{"probability": 0.9}, None])

    wake.STOP.clear()
    monkeypatch.setattr(wake.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        wake,
        "load_config",
        lambda _path: {
            "model_path": "/tmp/fake.onnx",
            "cooldown_seconds": 2.5,
            "cancel_cooldown_seconds": 0.0,
        },
    )
    monkeypatch.setattr(wake, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wake, "model_session", lambda _model_path: (object(), "input", "output"))
    monkeypatch.setattr(wake, "prewarm_stt", lambda _cfg: None)
    monkeypatch.setattr(wake, "prewarm_hermes", lambda _cfg: None)
    monkeypatch.setattr(wake, "wait_for_wake", lambda *_args, **_kwargs: next(activations))
    monkeypatch.setattr(wake, "handle_activation", lambda *_args, **_kwargs: flow.VOICE_SESSION_CANCELLED)
    monkeypatch.setattr(wake.STOP, "wait", lambda seconds: wait_calls.append(seconds) or False)

    assert wake.main([]) == 0

    assert wait_calls == []


def test_post_activation_cooldown_uses_cancel_specific_rearm_delay():
    assert wake.post_activation_cooldown_seconds(
        {"cooldown_seconds": 2.5},
        flow.VOICE_SESSION_CANCELLED,
    ) == 0.0
    assert wake.post_activation_cooldown_seconds(
        {"cooldown_seconds": 2.5, "cancel_cooldown_seconds": 0.2},
        flow.VOICE_SESSION_CANCELLED,
    ) == 0.2
    assert wake.post_activation_cooldown_seconds(
        {"cooldown_seconds": 2.5},
        flow.VOICE_SESSION_COMPLETED,
    ) == 2.5
