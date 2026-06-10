"""PortAudio device listing for the daemon CLI."""
from __future__ import annotations


def list_devices() -> int:
    import sounddevice as sd  # type: ignore[import-not-found]

    print(sd.query_devices())
    return 0


__all__ = ["list_devices"]
