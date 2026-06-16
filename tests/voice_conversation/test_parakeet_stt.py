from __future__ import annotations

from pathlib import Path
import types

from omegaconf import OmegaConf  # type: ignore[import-not-found]

import numpy as np

from okay_hermes_voice import daemon_config
from okay_hermes_voice.audio import parakeet_stt
from okay_hermes_voice.audio import recording as audio_recording
from okay_hermes_voice.audio import transcription
from okay_hermes_voice.audio.parakeet.transcriber import ParakeetStreamingTranscriber
from okay_hermes_voice.audio.parakeet_config import ParakeetStreamingConfig


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


def test_record_command_ends_after_streamed_text_stops_advancing(monkeypatch):
    block_samples = 1600
    blocks = [np.full((block_samples, 1), 900, dtype=np.int16) for _ in range(20)]

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
            self.transcripts = ["turn", "turn on", "turn on", "turn on", "turn on", "turn on"]

        def accept_int16(self, block):
            self.blocks.append(np.asarray(block).copy())
            if len(self.blocks) > 6:
                raise AssertionError("recording continued after streamed text stopped advancing")
            if self.transcripts:
                return self.transcripts.pop(0)
            return "turn on"

        def finalize(self):
            return "turn on"

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
            "speech_silence_duration_seconds": 99.0,
            "streamed_text_idle_duration_seconds": 0.3,
            "min_command_seconds": 0.0,
            "speech_start_timeout_seconds": 10.0,
        }
    )
    monkeypatch.setattr(audio_recording, "_SD", types.SimpleNamespace(InputStream=FakeInputStream))
    monkeypatch.setattr(audio_recording.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(audio_recording, "write_wav_int16", fake_write_wav)
    monkeypatch.setattr(audio_recording, "start_parakeet_live_streaming", lambda _cfg: live)

    result = audio_recording.record_command(cfg)

    assert isinstance(result, audio_recording.CommandRecording)
    assert result.live_transcript == "turn on"
    assert 3 <= len(live.blocks) < len(blocks)
    assert captured["sample_rate"] == 16000


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
            "streamed_text_idle_duration_seconds": 0.0,
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


def test_parakeet_tdt_switches_to_streaming_label_looping_decoder():
    seen = []

    class FakeModel:
        def __init__(self):
            self.cfg = types.SimpleNamespace(
                decoding=OmegaConf.create(
                    {
                        "model_type": "tdt",
                        "strategy": "greedy",
                        "fused_batch_size": None,
                        "tdt_include_token_duration": None,
                        "greedy": {"loop_labels": False, "preserve_alignments": True},
                    }
                )
            )
            self.decoding = types.SimpleNamespace(decoding=types.SimpleNamespace())

        def change_decoding_strategy(self, cfg):
            seen.append(cfg)
            if cfg.strategy == "greedy_batch":
                self.decoding = types.SimpleNamespace(
                    decoding=types.SimpleNamespace(decoding_computer=object())
                )

    model = FakeModel()
    transcriber = ParakeetStreamingTranscriber(ParakeetStreamingConfig(model_name="nvidia/parakeet-tdt-1.1b"))

    transcriber._ensure_streaming_decoding_strategy(model)

    assert seen[-1].strategy == "greedy_batch"
    assert seen[-1].greedy.loop_labels is True
    assert transcriber._model_has_live_decoding(model) is True


def test_parakeet_batch_fallback_handles_tdt_decoders_without_live_computer(monkeypatch, tmp_path):
    command = tmp_path / "command.wav"
    command.write_bytes(b"fake wav")
    calls = []

    class FakeModel:
        decoding = types.SimpleNamespace(decoding=types.SimpleNamespace())

        def transcribe(self, paths, batch_size=1, return_hypotheses=False):
            calls.append((paths, batch_size, return_hypotheses))
            return [types.SimpleNamespace(text="batch tdt transcript")]

    transcriber = ParakeetStreamingTranscriber(ParakeetStreamingConfig(model_name="nvidia/parakeet-tdt-1.1b"))
    transcriber._loaded = True
    transcriber._model = FakeModel()
    monkeypatch.setattr(transcriber, "load", lambda: None)

    assert transcriber.transcribe_file(command) == "batch tdt transcript"
    assert calls == [([str(command)], 1, False)]


def test_default_config_documents_parakeet_streaming_alternative():
    assert daemon_config.DEFAULT_CONFIG["parakeet_model_name"] == "nvidia/parakeet-unified-en-0.6b"
    assert daemon_config.DEFAULT_CONFIG["parakeet_live_streaming"] is True
    assert daemon_config.DEFAULT_CONFIG["streamed_text_idle_duration_seconds"] == 1.0
    assert parakeet_stt.is_parakeet_provider("parakeet_unified_streaming") is True
