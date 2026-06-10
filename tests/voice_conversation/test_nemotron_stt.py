from __future__ import annotations

import contextlib
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np

from okay_hermes_voice.audio import nemotron_stt
from okay_hermes_voice.audio import transcription
from okay_hermes_voice import daemon_config
from okay_hermes_voice.activation.flow import VOICE_SESSION_COMPLETED, ActivationFlowServices, handle_activation_impl
from okay_hermes_voice.activation_archive import command_audio_metadata_fields, update_activation_archive_metadata
from okay_hermes_voice.audio.nemotron import transcriber as nemotron_transcriber
from okay_hermes_voice.visualization import (
    append_visualization_turn,
    finish_cancelled_voice_session,
    is_visualization_cancel_requested,
    update_visualization_state,
    visualization_cancel_reason,
)
from okay_hermes_voice.voice_routing.close_detection import is_close_transcript
from okay_hermes_voice.voice_routing.status import interaction_ack_text, routed_request_status_message


class FakeTensor:
    def to(self, _dtype):
        return self


class FakeTorch:
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

    class cuda:
        @staticmethod
        def is_available():
            return False

    backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(enabled=True))

    class amp:
        @staticmethod
        def autocast(*_args, **_kwargs):
            return contextlib.nullcontext()

    @staticmethod
    def device(name):
        return types.SimpleNamespace(type=name)

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()

    @staticmethod
    def no_grad():
        return contextlib.nullcontext()


class FakeEncoder:
    streaming_cfg = types.SimpleNamespace(drop_extra_pre_encoded=13)

    def __init__(self):
        self.att_context_size = None

    def set_default_att_context_size(self, att_context_size):
        self.att_context_size = att_context_size

    def get_initial_cache_state(self, *, batch_size):
        return f"channel-{batch_size}", "time", "length"


class FakeModel:
    def __init__(self):
        self.encoder = FakeEncoder()
        self.cfg = types.SimpleNamespace(decoding="fake-decoding")
        self.decoding = object()
        self.calls = []
        self.decoding_strategy = None
        self.device = None

    def change_decoding_strategy(self, strategy):
        self.decoding_strategy = strategy

    def to(self, *, device):
        self.device = device
        return self

    def eval(self):
        return None

    def conformer_stream_step(self, **kwargs):
        self.calls.append(kwargs)
        step = len(self.calls)
        text = "partial command" if step == 1 else "final command"
        return (
            f"pred-{step}",
            [types.SimpleNamespace(text=text)],
            f"channel-next-{step}",
            f"time-next-{step}",
            f"length-next-{step}",
            f"hyp-{step}",
        )


class FakeASRModel:
    created_model = None
    loaded_by = None

    @classmethod
    def from_pretrained(cls, *, model_name, map_location):
        cls.loaded_by = ("name", model_name, map_location.type)
        cls.created_model = FakeModel()
        return cls.created_model

    @classmethod
    def restore_from(cls, *, restore_path, map_location):
        cls.loaded_by = ("path", restore_path, map_location.type)
        cls.created_model = FakeModel()
        return cls.created_model


class FakeNemo:
    models = types.SimpleNamespace(ASRModel=FakeASRModel)


class FakeStreamingBuffer:
    instances = []

    def __init__(self, *, model, online_normalization, pad_and_drop_preencoded):
        self.model = model
        self.online_normalization = online_normalization
        self.pad_and_drop_preencoded = pad_and_drop_preencoded
        self.streams_length = [2]
        self.appended = None
        self.append_calls = []
        self._step_index = -1
        self.steps = [(FakeTensor(), "length-1"), (FakeTensor(), "length-2")]
        FakeStreamingBuffer.instances.append(self)

    def append_audio(self, audio, stream_id):
        self.appended = (np.asarray(audio), stream_id)
        self.append_calls.append((np.asarray(audio), stream_id))
        return None, None, stream_id

    def __iter__(self):
        for index, step in enumerate(self.steps):
            self._step_index = index
            yield step

    def is_buffer_empty(self):
        return self._step_index >= len(self.steps) - 1


class _TestLog:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def test_nemotron_transcriber_uses_cache_aware_streaming_loop(monkeypatch, tmp_path):
    FakeASRModel.created_model = None
    FakeASRModel.loaded_by = None
    FakeStreamingBuffer.instances.clear()
    monkeypatch.setattr(
        nemotron_transcriber,
        "load_nemotron_dependencies",
        lambda: (FakeTorch, FakeNemo, FakeStreamingBuffer),
    )

    monkeypatch.setattr(
        nemotron_transcriber,
        "load_mono_audio_samples",
        lambda _path: np.array([[1, 3, 5], [3, 5, 7]], dtype=np.float32).mean(axis=0),
    )

    cfg = nemotron_stt.NemotronStreamingConfig(
        model_name="nvidia/nemotron-speech-streaming-en-0.6b",
        att_context_size=(70, 13),
        amp=True,
    )
    transcriber = nemotron_stt.NemotronStreamingTranscriber(cfg)
    audio_path = tmp_path / "command.wav"
    audio_path.write_bytes(b"fake wav")

    assert transcriber.transcribe_file(audio_path) == "final command"

    model = FakeASRModel.created_model
    assert isinstance(model, FakeModel)
    assert FakeASRModel.loaded_by == ("name", "nvidia/nemotron-speech-streaming-en-0.6b", "cpu")
    assert model.encoder.att_context_size == [70, 13]
    assert FakeTorch.backends.cudnn.enabled is False
    assert model.decoding_strategy == "fake-decoding"
    appended_audio, stream_id = FakeStreamingBuffer.instances[0].appended
    assert stream_id == -1
    assert appended_audio.ndim == 1
    assert appended_audio.dtype == np.float32
    assert appended_audio.tolist() == [2.0, 4.0, 6.0]
    assert len(model.calls) == 2
    assert model.calls[0]["previous_hypotheses"] is None
    assert model.calls[0]["previous_pred_out"] is None
    assert model.calls[0]["drop_extra_pre_encoded"] == 0
    assert model.calls[1]["previous_hypotheses"] == "hyp-1"
    assert model.calls[1]["previous_pred_out"] == "pred-1"
    assert model.calls[1]["drop_extra_pre_encoded"] == 13
    assert model.calls[1]["keep_all_outputs"] is True


def test_nemotron_live_session_reuses_first_stream_after_negative_initial_stream_id(monkeypatch):
    FakeASRModel.created_model = None
    FakeASRModel.loaded_by = None
    FakeStreamingBuffer.instances.clear()
    monkeypatch.setattr(
        nemotron_transcriber,
        "load_nemotron_dependencies",
        lambda: (FakeTorch, FakeNemo, FakeStreamingBuffer),
    )

    transcriber = nemotron_stt.NemotronStreamingTranscriber(nemotron_stt.NemotronStreamingConfig())
    session = transcriber.start_live_session()

    session.accept_audio(np.array([0.1, 0.2], dtype=np.float32))
    session.accept_audio(np.array([0.3, 0.4], dtype=np.float32))

    buffer = FakeStreamingBuffer.instances[0]
    assert [stream_id for _audio, stream_id in buffer.append_calls] == [-1, 0]


def test_transcribe_command_dispatches_to_nemotron_provider(monkeypatch, tmp_path):
    calls = []
    command = tmp_path / "command.wav"
    command.write_bytes(b"fake wav")

    monkeypatch.setattr(transcription, "transcribe_recording", lambda *_args: (_ for _ in ()).throw(AssertionError("Hermes STT should not run")))
    monkeypatch.setattr(transcription, "is_whisper_hallucination", lambda text: False)
    monkeypatch.setattr(
        transcription,
        "transcribe_nemotron_streaming",
        lambda path, cfg: calls.append((path, cfg)) or {"success": True, "transcript": "turn on the lights"},
    )

    cfg = {"stt_provider": "nemotron_en_streaming"}
    assert transcription.transcribe_command(command, cfg) == "turn on the lights"
    assert calls == [(command, cfg)]


def test_default_stt_provider_remains_hermes(monkeypatch, tmp_path):
    command = tmp_path / "command.wav"
    command.write_bytes(b"fake wav")

    monkeypatch.setattr(transcription, "is_whisper_hallucination", lambda text: False)
    monkeypatch.setattr(transcription, "transcribe_recording", lambda path: {"success": True, "transcript": f"hermes:{Path(path).name}"})

    assert transcription.transcribe_command(command) == "hermes:command.wav"


def test_prewarm_uses_nemotron_without_dummy_wav(monkeypatch):
    calls = []
    monkeypatch.setattr(transcription, "write_wav_int16", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dummy wav should not be written")))
    monkeypatch.setattr(transcription, "prewarm_nemotron_streaming", lambda cfg: calls.append(cfg))

    cfg = {"stt_provider": "nemotron_en_streaming", "prewarm_stt_on_start": True}
    transcription.prewarm_stt(cfg)

    assert calls == [cfg]


def test_default_config_documents_nemotron_alternative():
    assert daemon_config.DEFAULT_CONFIG["stt_provider"] == "hermes"
    assert daemon_config.DEFAULT_CONFIG["transcript_only_mode"] is False
    assert daemon_config.DEFAULT_CONFIG["nemotron_model_name"] == "nvidia/nemotron-speech-streaming-en-0.6b"
    assert daemon_config.DEFAULT_CONFIG["nemotron_att_context_size"] == [70, 13]
    assert daemon_config.DEFAULT_CONFIG["nemotron_cudnn_enabled"] is False
    assert daemon_config.DEFAULT_CONFIG["nemotron_live_streaming"] is True


def test_activation_flow_passes_daemon_config_to_stt(monkeypatch, tmp_path):
    state_path = tmp_path / "voice_state.json"
    command_path = tmp_path / "command.wav"
    command_path.write_bytes(b"fake wav")
    seen = []

    def fake_launch_visualization(_cfg, probability):
        update_visualization_state(
            state_path,
            status="listening",
            probability=probability,
            cancel_requested=False,
            cancel_reason="",
        )
        return state_path

    def fake_transcribe(path, cfg):
        seen.append((path, cfg["stt_provider"]))
        return "test nemotron"

    services = ActivationFlowServices(
        archive_command_audio=lambda *_args, **_kwargs: None,
        command_audio_metadata_fields=command_audio_metadata_fields,
        save_activation_archive=lambda *_args, **_kwargs: None,
        update_activation_archive_metadata=update_activation_archive_metadata,
        record_command=lambda *_args, **_kwargs: command_path,
        transcribe_command=fake_transcribe,
        log=_TestLog(),
        stop=threading.Event(),
        maybe_beep=lambda *_args, **_kwargs: None,
        speak_response=lambda *_args, **_kwargs: None,
        append_visualization_turn=append_visualization_turn,
        finish_cancelled_voice_session=finish_cancelled_voice_session,
        is_visualization_cancel_requested=is_visualization_cancel_requested,
        launch_visualization=fake_launch_visualization,
        update_visualization_state=update_visualization_state,
        visualization_cancel_reason=visualization_cancel_reason,
        answer_routed_request=lambda *_args, **_kwargs: ("spoken answer", [], "heavy_agent"),
        command_recording_config_for_turn=lambda cfg, _turn: cfg,
        interaction_ack_text=interaction_ack_text,
        is_close_transcript=is_close_transcript,
        route_transcribed_request=lambda *_args, **_kwargs: None,
        routed_request_status_message=routed_request_status_message,
        time=__import__("time"),
    )

    result = handle_activation_impl(
        services,
        {"conversation_mode_enabled": False, "stt_provider": "nemotron_en_streaming"},
        {"probability": 0.9},
    )

    assert result == VOICE_SESSION_COMPLETED
    assert seen == [(command_path, "nemotron_en_streaming")]
