#!/usr/bin/env python3
"""Generate soft loading-motif acknowledgement clips for Okay Hermes Voice."""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

SR = 48_000
DURATION = 4.25
ACK_NAMES = ("got_it", "checking", "thinking", "looking_that_up", "working")


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _smoothstep(x: float) -> float:
    x = _clamp(x)
    return x * x * (3.0 - 2.0 * x)


def _perc_env(t: float, dur: float, attack: float, release: float) -> float:
    if t < 0.0 or t >= dur:
        return 0.0
    a = _smoothstep(t / attack) if attack > 0 else 1.0
    r = _smoothstep((dur - t) / release) if release > 0 else 1.0
    body = math.exp(-t / max(release * 1.7, 0.03))
    return a * min(1.0, r) * body


def _add_sample(left: list[float], right: list[float], idx: int, val: float, pan: float) -> None:
    if 0 <= idx < len(left):
        gl = math.cos((pan + 1.0) * math.pi / 4.0)
        gr = math.sin((pan + 1.0) * math.pi / 4.0)
        left[idx] += val * gl
        right[idx] += val * gr


def _add_tone(
    left: list[float],
    right: list[float],
    start: float,
    dur: float,
    freq: float,
    amp: float,
    pan: float = 0.0,
    attack: float = 0.012,
    release: float = 0.16,
    bend: float = -0.035,
) -> None:
    s0 = int(start * SR)
    count = int(dur * SR)
    phase = 0.0
    lp = 0.0
    for i in range(count):
        idx = s0 + i
        if idx >= len(left):
            break
        t = i / SR
        e = _perc_env(t, dur, attack, release)
        if e <= 0.0:
            continue
        f = freq * (1.0 + bend * math.exp(-t / 0.07))
        phase += 2.0 * math.pi * f / SR
        sig = math.sin(phase) + 0.22 * math.sin(2.0 * phase + 0.15) + 0.055 * math.sin(3.0 * phase + 0.5)
        lp = 0.90 * lp + 0.10 * sig
        _add_sample(left, right, idx, amp * e * lp, pan)


def _add_tick(left: list[float], right: list[float], start: float, amp: float = 0.035, pan: float = 0.0) -> None:
    dur = 0.090
    s0 = int(start * SR)
    count = int(dur * SR)
    p1 = 0.0
    p2 = 0.0
    for i in range(count):
        idx = s0 + i
        if idx >= len(left):
            break
        t = i / SR
        e = _perc_env(t, dur, 0.004, 0.030)
        if e <= 0.0:
            continue
        p1 += 2.0 * math.pi * 980.0 / SR
        p2 += 2.0 * math.pi * 1560.0 / SR
        sig = 0.70 * math.sin(p1) + 0.30 * math.sin(p2 + 0.3)
        _add_sample(left, right, idx, amp * e * sig, pan)


def _render_motif() -> bytes:
    n = int(SR * DURATION)
    left = [0.0] * n
    right = [0.0] * n

    base = 0.18
    _add_tone(left, right, base + 0.00, 0.42, 196, 0.25, -0.10, release=0.17)
    _add_tick(left, right, base + 0.006, 0.020, -0.28)
    _add_tone(left, right, base + 0.25, 0.50, 147, 0.28, 0.08, release=0.23)
    _add_tick(left, right, base + 0.256, 0.018, 0.20)

    _add_tone(left, right, base + 0.84, 0.28, 247, 0.16, -0.16, release=0.12)
    _add_tone(left, right, base + 1.02, 0.32, 185, 0.18, 0.16, release=0.14)
    _add_tone(left, right, base + 1.55, 0.29, 220, 0.13, -0.06, release=0.12)
    _add_tone(left, right, base + 1.77, 0.30, 165, 0.14, 0.08, release=0.13)

    _add_tone(left, right, base + 2.32, 0.22, 330, 0.075, -0.22, release=0.08)
    _add_tick(left, right, base + 2.323, 0.014, -0.25)
    _add_tone(left, right, base + 2.49, 0.22, 294, 0.070, 0.18, release=0.09)
    _add_tick(left, right, base + 2.493, 0.012, 0.22)
    for k, off in enumerate([3.14, 3.53, 3.92]):
        _add_tick(left, right, base + off, 0.0065 - 0.0010 * k, pan=(-0.16 + 0.16 * k))

    dry_l = left[:]
    dry_r = right[:]
    for i in range(n):
        wet_l = 0.0
        wet_r = 0.0
        for delay, gain in [(0.145, 0.10), (0.310, 0.045), (0.515, 0.020)]:
            d = int(delay * SR)
            if i >= d:
                wet_l += gain * dry_r[i - d]
                wet_r += gain * dry_l[i - d]
        left[i] = dry_l[i] + wet_l
        right[i] = dry_r[i] + wet_r

    fade = int(0.50 * SR)
    for i in range(fade):
        j = n - 1 - i
        g = _smoothstep(i / fade)
        left[j] *= g
        right[j] *= g
    for i in range(int(0.035 * SR)):
        g = _smoothstep(i / int(0.035 * SR))
        left[i] *= g
        right[i] *= g

    for i in range(n):
        left[i] = math.tanh(left[i] * 1.15) / math.tanh(1.15)
        right[i] = math.tanh(right[i] * 1.15) / math.tanh(1.15)

    peak = max(max(abs(x) for x in left), max(abs(x) for x in right), 1e-9)
    gain = 0.56 / peak
    frames = bytearray()
    for l, r in zip(left, right):
        frames += struct.pack("<hh", int(_clamp(l * gain) * 32767), int(_clamp(r * gain) * 32767))
    return bytes(frames)


def _write_wav(path: Path, frames: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("~/.hermes/wakeword/ack_cache"))
    parser.add_argument("--force", action="store_true", help="overwrite existing acknowledgement files")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser()
    frames = _render_motif()
    for name in ACK_NAMES:
        path = output_dir / f"{name}.wav"
        if path.exists() and not args.force:
            print(f"keeping existing {path}")
            continue
        _write_wav(path, frames)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
