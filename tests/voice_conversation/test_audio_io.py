from __future__ import annotations

from pathlib import Path
import json
import types

import numpy as np

from okay_hermes_voice import audio_io as audio
from okay_hermes_voice import daemon_config as daemon_config
from okay_hermes_voice import playback
from okay_hermes_voice.audio import recording as audio_recording


def test_audio_io_facade_exports_semantic_audio_modules():
    from okay_hermes_voice.audio import devices as devices_mod
    from okay_hermes_voice.audio import recording as recording_mod
    from okay_hermes_voice.audio import smoke as smoke_mod
    from okay_hermes_voice.audio import transcription as transcription_mod
    from okay_hermes_voice.audio import wake as wake_mod
    from okay_hermes_voice.audio import waveform as waveform_mod
    from okay_hermes_voice.audio import wav as wav_mod

    assert audio.model_session is wake_mod.model_session
    assert audio.run_wake_inference is wake_mod.run_wake_inference
    assert audio.wait_for_wake is wake_mod.wait_for_wake
    assert audio.record_command is recording_mod.record_command
    assert audio.transcribe_command is transcription_mod.transcribe_command
    assert audio.prewarm_stt is transcription_mod.prewarm_stt
    assert audio.float_waveform_to_int16 is waveform_mod.float_waveform_to_int16
    assert audio.rms_int16 is waveform_mod.rms_int16
    assert audio.write_wav_int16 is wav_mod.write_wav_int16
    assert audio.write_wav_int16_to_path is wav_mod.write_wav_int16_to_path
    assert audio.list_devices is devices_mod.list_devices
    assert audio.smoke_test is smoke_mod.smoke_test
    assert all(not name.startswith("_") for name in audio.__all__)
    assert not {"collections", "contextlib", "json", "math", "queue", "tempfile", "time", "wave"} & set(audio.__all__)


def test_playback_package_facade_exports_response_helpers():
    from okay_hermes_voice.playback import response as response_mod

    assert hasattr(playback, "__path__")
    assert playback.speak_response is response_mod.speak_response
    assert playback.play_tts_file is response_mod.play_tts_file
    assert playback.maybe_beep is response_mod.maybe_beep
    assert all(not name.startswith("_") for name in playback.__all__)
    assert not {"contextlib", "json", "subprocess", "threading", "time"} & set(playback.__all__)


def test_record_command_returns_none_without_opening_stream_when_cancelled(monkeypatch):
    def fail_if_opened(*_args, **_kwargs):
        raise AssertionError("InputStream should not open after cancellation")

    cfg = dict(daemon_config.DEFAULT_CONFIG)
    cfg.update({"speech_start_timeout_seconds": 1.0, "block_seconds": 0.1})
    monkeypatch.setattr(audio_recording.sd, "InputStream", fail_if_opened)

    assert audio.record_command(cfg, cancel_check=lambda: True) is None


def test_record_command_ignores_isolated_spike_before_real_speech(monkeypatch):
    block_samples = 1600
    blocks = [
        np.full((block_samples, 1), 500, dtype=np.int16),
        *(np.zeros((block_samples, 1), dtype=np.int16) for _ in range(5)),
        *(np.full((block_samples, 1), 900, dtype=np.int16) for _ in range(3)),
        *(np.zeros((block_samples, 1), dtype=np.int16) for _ in range(5)),
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

    captured = {}

    def fake_write_wav(audio_data, sample_rate):
        captured["audio"] = np.asarray(audio_data, dtype=np.int16).copy()
        captured["sample_rate"] = sample_rate
        return Path("/tmp/captured-command.wav")

    clock = {"now": 0.0}

    def fake_monotonic():
        clock["now"] += 0.1
        return clock["now"]

    cfg = dict(daemon_config.DEFAULT_CONFIG)
    cfg.update(
        {
            "block_seconds": 0.1,
            "speech_rms_threshold": 200,
            "speech_silence_duration_seconds": 0.3,
            "speech_start_timeout_seconds": 10.0,
        }
    )
    monkeypatch.setattr(audio_recording.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(audio_recording.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(audio_recording, "write_wav_int16", fake_write_wav)

    assert audio.record_command(cfg) == Path("/tmp/captured-command.wav")
    assert captured["sample_rate"] == 16000
    assert int(np.max(captured["audio"])) == 900
    assert not np.any(captured["audio"] == 500)


def test_record_command_streams_nemotron_live_during_capture(monkeypatch):
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

        def finalize(self):
            return "live nemotron transcript"

    live = FakeLiveSession()
    captured = {}

    def fake_write_wav(audio_data, sample_rate):
        captured["audio"] = np.asarray(audio_data, dtype=np.int16).copy()
        captured["sample_rate"] = sample_rate
        return Path("/tmp/live-captured-command.wav")

    clock = {"now": 0.0}

    def fake_monotonic():
        clock["now"] += 0.1
        return clock["now"]

    cfg = dict(daemon_config.DEFAULT_CONFIG)
    cfg.update(
        {
            "stt_provider": "nemotron_en_streaming",
            "nemotron_live_streaming": True,
            "block_seconds": 0.1,
            "speech_rms_threshold": 200,
            "speech_silence_duration_seconds": 0.3,
            "speech_start_timeout_seconds": 10.0,
        }
    )
    monkeypatch.setattr(audio_recording.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(audio_recording.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(audio_recording, "write_wav_int16", fake_write_wav)
    monkeypatch.setattr(audio_recording, "start_nemotron_live_streaming", lambda _cfg: live)

    result = audio.record_command(cfg)

    assert isinstance(result, audio_recording.CommandRecording)
    assert result.path == Path("/tmp/live-captured-command.wav")
    assert result.live_transcript == "live nemotron transcript"
    assert captured["sample_rate"] == 16000
    streamed_maxima = [int(np.max(block)) for block in live.blocks]
    assert streamed_maxima.count(900) >= 3
    assert len(live.blocks) > 3


def test_play_tts_file_terminates_player_when_cancel_requested(monkeypatch):
    fake_proc = types.SimpleNamespace(
        returncode=None,
        terminated=False,
        killed=False,
        stderr="",
        stdout="",
    )

    def poll():
        return fake_proc.returncode

    def terminate():
        fake_proc.terminated = True
        fake_proc.returncode = -15

    def kill():
        fake_proc.killed = True
        fake_proc.returncode = -9

    def wait(timeout=None):
        return fake_proc.returncode

    fake_proc.poll = poll
    fake_proc.terminate = terminate
    fake_proc.kill = kill
    fake_proc.wait = wait

    monkeypatch.setattr(playback.response.subprocess, "Popen", lambda *_args, **_kwargs: fake_proc)
    monkeypatch.setattr(
        playback.response.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("playback should use cancellable Popen")),
    )
    monkeypatch.setattr(playback.response.time, "sleep", lambda *_args, **_kwargs: None)

    cancel_checks = iter([False, False, True])

    ok = playback.play_tts_file(
        {"playback_sink": "@DEFAULT_SINK@", "playback_volume": 1.0},
        "/tmp/response.wav",
        cancel_check=lambda: next(cancel_checks, True),
    )

    assert ok is False
    assert fake_proc.terminated is True


def test_speak_response_returns_generation_and_playback_timing(monkeypatch):
    clock = {"now": 10.0}
    calls = []

    def fake_monotonic():
        return clock["now"]

    def fake_text_to_speech_tool(text):
        calls.append(("tts", text))
        clock["now"] += 0.25
        return json.dumps({"success": True, "file_path": "/tmp/response.wav"})

    def fake_play_tts_file(cfg, file_path, cancel_check=None):
        calls.append(("play", file_path, bool(cancel_check and cancel_check())))
        clock["now"] += 0.75
        return True

    monkeypatch.setattr(playback.response.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(playback.response, "text_to_speech_tool", fake_text_to_speech_tool)
    monkeypatch.setattr(playback.response, "play_tts_file", fake_play_tts_file)

    timing = playback.speak_response(
        {"tts_enabled": True, "max_spoken_response_chars": 2500},
        "spoken answer",
        cancel_check=lambda: False,
    )

    assert calls == [("tts", "spoken answer"), ("play", "/tmp/response.wav", False)]
    assert timing["tts_success"] is True
    assert timing["playback_success"] is True
    assert timing["tts_seconds"] == 0.25
    assert timing["playback_seconds"] == 0.75
    assert timing["speak_seconds"] == 1.0
    assert timing["tts_file_path"] == "/tmp/response.wav"


def test_speak_response_notifies_tts_and_playback_stage_callbacks(monkeypatch):
    monkeypatch.setattr(playback.response, "text_to_speech_tool", lambda _text: json.dumps({"success": True, "file_path": "/tmp/response.wav"}))
    monkeypatch.setattr(playback.response, "play_tts_file", lambda *_args, **_kwargs: True)

    stages = []

    timing = playback.speak_response(
        {"tts_enabled": True, "max_spoken_response_chars": 2500},
        "spoken answer",
        cancel_check=lambda: False,
        stage_callback=stages.append,
    )

    assert stages == ["tts", "playback"]
    assert timing["tts_success"] is True
    assert timing["playback_success"] is True
