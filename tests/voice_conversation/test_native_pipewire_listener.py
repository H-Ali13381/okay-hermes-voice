from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
SOURCE = REPO_ROOT / "native" / "okay-hermes-wake-listener.c"


def _static_function_body(source: str, name: str) -> str:
    name_index = source.index(name)
    start = source.rfind("static", 0, name_index)
    if start < 0:
        raise AssertionError(f"Could not find static function {name}")
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : index]
    raise AssertionError(f"Could not find complete body for {name}")


def test_native_pipewire_listener_keeps_model_work_out_of_process_callback():
    source = SOURCE.read_text(encoding="utf-8")
    body = _static_function_body(source, "on_process")

    assert "ring_write" in body
    assert "pw_stream_queue_buffer" in body
    assert "run_wakeword_worker" not in body
    assert "OrtRun" not in body
    assert "run_model" not in body
    assert "onnx" not in body.lower()
    assert "malloc" not in body
    assert "printf" not in body


def test_native_listener_stores_only_16khz_mono_model_rate_audio():
    source = SOURCE.read_text(encoding="utf-8")
    worker = _static_function_body(source, "run_wakeword_worker")

    assert "MAX_GRAPH_RATE" not in source
    assert "MAX_GRAPH_CHANNELS" not in source
    assert "ring_init(&data.ring, RING_SECONDS * MODEL_RATE)" in source
    assert "calloc(data->ring.capacity" not in worker
    assert "MODEL_SAMPLES" in worker


def test_native_pipewire_listener_uses_onnx_runtime_c_api_in_worker_not_python():
    source = SOURCE.read_text(encoding="utf-8")
    worker = _static_function_body(source, "run_wakeword_worker")

    assert "#include \"onnxruntime_c_api.h\"" in source
    assert "OrtGetApiBase" in source
    assert "CreateSession" in source
    assert "CreateTensorWithDataAsOrtValue" in source
    assert "OrtRun" in source or "->Run" in source
    assert "python" not in worker.lower()
    assert "popen" not in worker.lower()
    assert "system(" not in worker.lower()


def test_native_listener_disables_onnx_arena_features_for_tiny_fixed_shape_model():
    source = SOURCE.read_text(encoding="utf-8")
    init = _static_function_body(source, "wake_model_init")

    assert "DisableCpuMemArena" in init
    assert "DisableMemPattern" in init
    assert "CreateCpuMemoryInfo(OrtDeviceAllocator" in init


def test_native_listener_reports_capture_health_from_worker_not_rt_callback():
    source = SOURCE.read_text(encoding="utf-8")
    callback = _static_function_body(source, "on_process")
    worker = _static_function_body(source, "run_wakeword_worker")

    assert "write_capture_status" in source
    assert '"starting"' in source
    assert '"healthy"' in worker
    assert '"unhealthy"' in worker
    assert "CAPTURE_HEALTH_TIMEOUT_SECONDS 10.0" in source
    assert "write_capture_status" not in callback
    assert "fopen" not in callback
    assert "rename(" not in callback


def test_native_listener_uses_one_worker_timer_for_inference_and_capture_health():
    source = SOURCE.read_text(encoding="utf-8")
    worker = _static_function_body(source, "run_wakeword_worker")
    sleeper = _static_function_body(source, "sleep_worker_until_next_deadline")

    assert "WORKER_POLL_NS" not in source
    assert "sleep_worker_tick" not in source
    assert "sleep_worker_until_next_deadline" in worker
    assert "inference_interval" in sleeper
    assert "CAPTURE_HEALTH_TIMEOUT_SECONDS" in sleeper
    assert "nanosleep" in sleeper


def test_native_listener_derives_paths_from_hermes_home_root():
    source = SOURCE.read_text(encoding="utf-8")
    parser = _static_function_body(source, "parse_options")
    resolver = _static_function_body(source, "resolve_hermes_paths")
    status_resolver = _static_function_body(source, "resolve_capture_status_path")
    handler = _static_function_body(source, "run_handler_command")

    assert "--hermes-home" in source
    assert "hermes_home" in parser
    assert "wakeword/okay-hermes-repcnn-onnx/wakeword.fixed-1x48000.onnx" in resolver
    assert "wakeword/config.yaml" in resolver
    assert "CAPTURE_STATUS_FILENAME" in status_resolver
    assert "hermes_home" in status_resolver
    assert "activation_config_path" not in status_resolver
    assert "hermes-agent/venv/bin/python" in handler
    assert "HOME" not in handler


def test_native_pipewire_listener_compiles_and_exposes_help(tmp_path):
    if shutil.which("cc") is None or shutil.which("pkg-config") is None:
        raise AssertionError("cc and pkg-config are required to build the native PipeWire listener")

    pkg = subprocess.run(
        ["pkg-config", "--cflags", "--libs", "libpipewire-0.3"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    binary = tmp_path / "okay-hermes-wake-listener"
    command = [
        str(REPO_ROOT / "native" / "build_wake_listener.sh"),
        "--output",
        str(binary),
    ]
    env = {**os.environ, "LC_ALL": "C"}
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)

    help_result = subprocess.run([str(binary), "--help"], check=True, text=True, stdout=subprocess.PIPE)
    assert "Okay Hermes native PipeWire wake listener" in help_result.stdout
    assert "--duration-seconds" in help_result.stdout
    assert "--threshold" in help_result.stdout
    assert "--model" in help_result.stdout
    assert "--handler-command" in help_result.stdout
