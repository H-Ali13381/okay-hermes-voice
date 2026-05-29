from __future__ import annotations

from pathlib import Path
import types

import numpy as np

from okay_hermes_voice import audio_io as audio
from okay_hermes_voice import daemon_config as daemon_config
from okay_hermes_voice import playback


def test_record_command_returns_none_without_opening_stream_when_cancelled(monkeypatch):
    def fail_if_opened(*_args, **_kwargs):
        raise AssertionError("InputStream should not open after cancellation")

    cfg = dict(daemon_config.DEFAULT_CONFIG)
    cfg.update({"speech_start_timeout_seconds": 1.0, "block_seconds": 0.1})
    monkeypatch.setattr(audio.sd, "InputStream", fail_if_opened)

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
    monkeypatch.setattr(audio.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(audio.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(audio, "write_wav_int16", fake_write_wav)

    assert audio.record_command(cfg) == Path("/tmp/captured-command.wav")
    assert captured["sample_rate"] == 16000
    assert int(np.max(captured["audio"])) == 900
    assert not np.any(captured["audio"] == 500)


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

    monkeypatch.setattr(playback.subprocess, "Popen", lambda *_args, **_kwargs: fake_proc)
    monkeypatch.setattr(
        playback.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("playback should use cancellable Popen")),
    )
    monkeypatch.setattr(playback.time, "sleep", lambda *_args, **_kwargs: None)

    cancel_checks = iter([False, False, True])

    ok = playback.play_tts_file(
        {"playback_sink": "@DEFAULT_SINK@", "playback_volume": 1.0},
        "/tmp/response.wav",
        cancel_check=lambda: next(cancel_checks, True),
    )

    assert ok is False
    assert fake_proc.terminated is True
