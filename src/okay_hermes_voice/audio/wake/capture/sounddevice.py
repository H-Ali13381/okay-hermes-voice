"""sounddevice wakeword capture backend."""
from __future__ import annotations

import contextlib
import queue
from typing import Any, Iterator

import numpy as np

from ....daemon_config import LOG, STOP


def iter_sounddevice_blocks(*, sample_rate: int, block_samples: int) -> Iterator[np.ndarray]:
    """Yield mono float32 blocks from sounddevice."""
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)

    import sounddevice as sd  # type: ignore[import-not-found]

    def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        if status:
            LOG.debug("Wake audio callback status: %s", status)
        block = np.asarray(indata[:, 0], dtype=np.float32).copy()
        with contextlib.suppress(queue.Full):
            audio_q.put_nowait(block)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=block_samples, callback=callback):
        while not STOP.is_set():
            try:
                yield audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
