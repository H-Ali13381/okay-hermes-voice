"""Wakeword ONNX session construction."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import onnxruntime as ort

from ...daemon_config import LOG


def model_session(model_path: str) -> Tuple[ort.InferenceSession, str, str]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Wakeword ONNX model not found: {path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    shape = session.get_inputs()[0].shape
    LOG.info("Loaded wakeword model: %s input=%s shape=%s output=%s", path, input_name, shape, output_name)
    return session, input_name, output_name


__all__ = ["model_session"]
