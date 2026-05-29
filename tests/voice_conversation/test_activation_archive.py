from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from okay_hermes_voice import activation_archive as archive_mod
from okay_hermes_voice import daemon_config


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
