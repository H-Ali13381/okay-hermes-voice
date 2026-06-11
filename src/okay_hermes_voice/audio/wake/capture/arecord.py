"""ALSA arecord wakeword capture backend."""
from __future__ import annotations

import contextlib
import subprocess
from typing import Iterator

import numpy as np

from ....daemon_config import STOP


def iter_arecord_blocks(*, sample_rate: int, block_samples: int, device: str = "default") -> Iterator[np.ndarray]:
    """Yield mono float32 blocks from ``arecord`` raw S16_LE audio."""
    cmd = [
        "arecord",
        "-q",
        "-D",
        str(device or "default"),
        "-t",
        "raw",
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        "1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    block_bytes = block_samples * 2
    try:
        while not STOP.is_set():
            raw = proc.stdout.read(block_bytes)
            if len(raw) != block_bytes:
                break
            int16 = np.frombuffer(raw, dtype="<i2", count=block_samples)
            yield int16.astype(np.float32) / 32768.0
    finally:
        stop_arecord_process(proc)


def stop_arecord_process(proc: subprocess.Popen[bytes]) -> None:
    """Terminate the arecord subprocess without surfacing shutdown races."""
    with contextlib.suppress(Exception):
        proc.terminate()
    with contextlib.suppress(Exception):
        proc.wait(timeout=1)
    poll = getattr(proc, "poll", None)
    if poll is not None and poll() is None:
        with contextlib.suppress(Exception):
            proc.kill()
