"""PortAudio device listing for the daemon CLI."""
from __future__ import annotations

import sounddevice as sd


def list_devices() -> int:
    print(sd.query_devices())
    return 0


__all__ = ["list_devices"]
