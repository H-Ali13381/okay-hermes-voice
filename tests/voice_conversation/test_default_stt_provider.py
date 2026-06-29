from __future__ import annotations

from pathlib import Path
import types

import numpy as np

from okay_hermes_voice import daemon_config
from okay_hermes_voice.audio import recording as audio_recording
from okay_hermes_voice.audio import parakeet_stt


def test_default_config_engages_parakeet_live_streaming(monkeypatch):
    block_samples = 1600
    blocks = [
        np.zeros((block_samples, 1), dtype=np.int16),
        *(np.full((block_samples, 1), 900, dtype=np.int16) for _ in range(3)),
        *(np.zeros((block_samples, 1), dtype=np.int16) for _ in range(14)),
    ]

    class FakeInputStream:
        def __init__(self, *args, callback, **kwargs):
            self.callback = callback

        def __enter__(self):
            for block in blocks:
                self.callback(block, len(block), None, None)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeLiveSession:
        def __init__(self):
            self.blocks = []

        def accept_int16(self, block):
            self.blocks.append(np.asarray(block).copy())
            return "partial default parakeet"

        def finalize(self):
            return "final default parakeet transcript"

    live = FakeLiveSession()
    live_start_cfgs = []
    captured = {}
    clock = {"now": 0.0}

    def fake_monotonic():
        clock["now"] += 0.1
        return clock["now"]

    def fake_write_wav(audio_data, sample_rate):
        captured["audio"] = np.asarray(audio_data, dtype=np.int16).copy()
        captured["sample_rate"] = sample_rate
        return Path("/tmp/default-parakeet-command.wav")

    cfg = dict(daemon_config.DEFAULT_CONFIG)
    cfg.update(
        {
            "block_seconds": 0.1,
            "speech_rms_threshold": 200,
            "speech_silence_duration_seconds": 0.3,
            "streamed_text_idle_duration_seconds": 0.0,
            "speech_start_timeout_seconds": 10.0,
        }
    )

    monkeypatch.setattr(audio_recording, "_SD", types.SimpleNamespace(InputStream=FakeInputStream))
    monkeypatch.setattr(audio_recording.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(audio_recording, "write_wav_int16", fake_write_wav)
    monkeypatch.setattr(
        audio_recording,
        "start_parakeet_live_streaming",
        lambda started_cfg: live_start_cfgs.append(started_cfg) or live,
    )

    assert parakeet_stt.is_parakeet_provider(cfg["stt_provider"])

    result = audio_recording.record_command(cfg)

    assert live_start_cfgs == [cfg]
    assert isinstance(result, audio_recording.CommandRecording)
    assert result.path == Path("/tmp/default-parakeet-command.wav")
    assert result.live_transcript == "final default parakeet transcript"
    assert captured["sample_rate"] == 16000
    assert len(live.blocks) > 3
