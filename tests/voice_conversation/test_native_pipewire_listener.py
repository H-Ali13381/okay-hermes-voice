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
