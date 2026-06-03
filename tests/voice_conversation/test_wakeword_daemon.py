from __future__ import annotations

import json

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
