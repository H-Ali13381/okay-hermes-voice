from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from okay_hermes_voice import activation_archive as archive_mod
from okay_hermes_voice import daemon_config


def _write_archive_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_default_wakeword_model_matches_public_artifact():
    assert daemon_config.DEFAULT_CONFIG["model_path"].endswith(
        "okay-hermes-repcnn-onnx/wakeword.onnx"
    )
    assert daemon_config.DEFAULT_CONFIG["threshold"] == 0.6973556280136108

def test_save_activation_archive_writes_wake_clip_and_metadata(tmp_path):
    cfg = {
        "activation_archive_dir": str(tmp_path / "activations"),
        "sample_rate": 16000,
        "window_seconds": 3.0,
        "threshold": 0.6973556280136108,
        "model_path": "/models/okay-hermes.onnx",
    }
    waveform = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    activation = {
        "probability": 0.92,
        "scores": [0.88, 0.92],
        "waveform": waveform,
        "sample_rate": 16000,
        "detected_at": 1234.5,
    }

    archive = archive_mod.save_activation_archive(cfg, activation)

    assert archive is not None
    wav_path = Path(archive["wake_wav_path"])
    meta_path = Path(archive["metadata_path"])
    assert wav_path.parent == tmp_path / "activations"
    assert wav_path.exists()
    assert meta_path.exists()
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 16000
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["probability"] == 0.92
    assert metadata["scores"] == [0.88, 0.92]
    assert metadata["sample_rate"] == 16000
    assert metadata["status"] == "wake_detected"
    assert metadata["model_path"] == "/models/okay-hermes.onnx"

def test_update_activation_archive_metadata_merges_turn_details(tmp_path):
    meta_path = tmp_path / "activation.json"
    meta_path.write_text(json.dumps({"status": "wake_detected", "turns": []}), encoding="utf-8")
    archive = {"metadata_path": str(meta_path)}

    archive_mod.update_activation_archive_metadata(
        archive,
        status="completed",
        turns=[{"turn": 1, "transcript": "hello", "response": "hi"}],
        close_reason="close_phrase",
    )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["turns"] == [{"turn": 1, "transcript": "hello", "response": "hi"}]
    assert metadata["close_reason"] == "close_phrase"


def test_summarize_activation_archives_groups_phase_zero_latency_by_preset(tmp_path):
    _write_archive_metadata(
        tmp_path / "activation_simple_1.json",
        {
            "status": "completed",
            "benchmark_preset": "simple_chat",
            "benchmark_category": "Simple chat",
            "voice_session_timing": {"wake_to_handle_seconds": 0.3},
            "turns": [
                {
                    "turn": 1,
                    "response_source": "heavy_agent",
                    "timings": {
                        "turn": 1,
                        "wake_to_record_start_seconds": 0.4,
                        "record_seconds": 1.0,
                        "transcribe_seconds": 0.5,
                        "route_seconds": 0.1,
                        "answer_seconds": 2.0,
                        "tts_seconds": 0.4,
                        "playback_seconds": 0.8,
                        "speak_seconds": 1.2,
                        "turn_seconds": 4.8,
                    },
                }
            ],
        },
    )
    _write_archive_metadata(
        tmp_path / "activation_simple_2.json",
        {
            "status": "completed",
            "benchmark_preset": "simple_chat",
            "benchmark_category": "Simple chat",
            "turns": [
                {
                    "turn": 1,
                    "response_source": "small_model",
                    "timings": {
                        "turn": 1,
                        "record_seconds": 1.4,
                        "transcribe_seconds": 0.7,
                        "route_seconds": 0.2,
                        "answer_seconds": 1.8,
                        "tts_seconds": 0.6,
                        "playback_seconds": 1.0,
                        "speak_seconds": 1.6,
                        "turn_seconds": 5.7,
                    },
                }
            ],
        },
    )
    _write_archive_metadata(
        tmp_path / "activation_cancelled.json",
        {
            "status": "cancelled",
            "benchmark_preset": "cancel_rearm",
            "cancel_reason": "ctrl_c",
            "turns": [],
        },
    )
    (tmp_path / "not-json.txt").write_text("ignored", encoding="utf-8")

    summary = archive_mod.summarize_activation_archives(tmp_path)

    assert summary["schema_version"] == 1
    assert summary["archive_count"] == 3
    assert summary["turn_count"] == 2
    assert summary["status_counts"] == {"cancelled": 1, "completed": 2}
    assert summary["response_source_counts"] == {"heavy_agent": 1, "small_model": 1}
    assert summary["timing_fields"]["turn_seconds"] == {
        "count": 2,
        "min": 4.8,
        "mean": 5.25,
        "p50": 5.25,
        "p95": 5.655,
        "max": 5.7,
    }
    assert summary["timing_fields"]["answer_seconds"]["mean"] == 1.9

    simple = summary["by_preset"]["simple_chat"]
    assert simple["archive_count"] == 2
    assert simple["turn_count"] == 2
    assert simple["benchmark_category"] == "Simple chat"
    assert simple["timing_fields"]["turn_seconds"]["mean"] == 5.25

    cancelled = summary["by_preset"]["cancel_rearm"]
    assert cancelled["archive_count"] == 1
    assert cancelled["turn_count"] == 0
    assert cancelled["status_counts"] == {"cancelled": 1}
