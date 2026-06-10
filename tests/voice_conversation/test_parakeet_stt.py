from __future__ import annotations

from pathlib import Path
import types

import numpy as np

from okay_hermes_voice import daemon_config
from okay_hermes_voice.audio import parakeet_stt
from okay_hermes_voice.audio import recording as audio_recording
from okay_hermes_voice.audio import transcription


def test_parakeet_provider_dispatches_post_wav_stt_without_hermes(monkeypatch, tmp_path):
    command = tmp_path / "command.wav"
    command.write_bytes(b"fake wav")
    calls = []

    monkeypatch.setattr(transcription, "transcribe_recording", lambda *_args: (_ for _ in ()).throw(AssertionError("Hermes STT should not run")))
    monkeypatch.setattr(transcription, "is_whisper_hallucination", lambda text: False)
    monkeypatch.setattr(
        transcription,
        "transcribe_parakeet_streaming",
        lambda path, cfg: calls.append((path, cfg)) or {"success": True, "transcript": "parakeet heard this"},
    )

    cfg = {"stt_provider": "parakeet_unified_streaming"}
    assert transcription.transcribe_command(command, cfg) == "parakeet heard this"
    assert calls == [(command, cfg)]


def test_prewarm_uses_parakeet_without_dummy_wav(monkeypatch):
    calls = []
    monkeypatch.setattr(transcription, "write_wav_int16", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dummy wav should not be written")))
    monkeypatch.setattr(transcription, "prewarm_parakeet_streaming", lambda cfg: calls.append(cfg))

    cfg = {"stt_provider": "parakeet_unified_streaming", "prewarm_stt_on_start": True}
    transcription.prewarm_stt(cfg)

    assert calls == [cfg]


def test_record_command_starts_parakeet_live_streaming(monkeypatch):
    block_samples = 1600
    blocks = [
        np.zeros((block_samples, 1), dtype=np.int16),
        *(np.full((block_samples, 1), 900, dtype=np.int16) for _ in range(3)),
        *(np.zeros((block_samples, 1), dtype=np.int16) for _ in range(4)),
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
            return "partial parakeet"

        def finalize(self):
            return "final parakeet transcript"

    live = FakeLiveSession()
    captured = {}

    def fake_write_wav(audio_data, sample_rate):
        captured["audio"] = np.asarray(audio_data, dtype=np.int16).copy()
        captured["sample_rate"] = sample_rate
        return Path("/tmp/parakeet-command.wav")

    clock = {"now": 0.0}

    def fake_monotonic():
        clock["now"] += 0.1
        return clock["now"]

    cfg = dict(daemon_config.DEFAULT_CONFIG)
    cfg.update(
        {
            "stt_provider": "parakeet_unified_streaming",
            "parakeet_live_streaming": True,
            "block_seconds": 0.1,
            "speech_rms_threshold": 200,
            "speech_silence_duration_seconds": 0.3,
            "speech_start_timeout_seconds": 10.0,
        }
    )
    monkeypatch.setattr(audio_recording, "_SD", types.SimpleNamespace(InputStream=FakeInputStream))
    monkeypatch.setattr(audio_recording.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(audio_recording, "write_wav_int16", fake_write_wav)
    monkeypatch.setattr(audio_recording, "start_parakeet_live_streaming", lambda _cfg: live)

    result = audio_recording.record_command(cfg)

    assert isinstance(result, audio_recording.CommandRecording)
    assert result.path == Path("/tmp/parakeet-command.wav")
    assert result.live_transcript == "final parakeet transcript"
    assert captured["sample_rate"] == 16000
    assert len(live.blocks) > 3


def test_default_config_documents_parakeet_streaming_alternative():
    assert daemon_config.DEFAULT_CONFIG["parakeet_model_name"] == "nvidia/parakeet-unified-en-0.6b"
    assert daemon_config.DEFAULT_CONFIG["parakeet_live_streaming"] is True
    assert parakeet_stt.is_parakeet_provider("parakeet_unified_streaming") is True
