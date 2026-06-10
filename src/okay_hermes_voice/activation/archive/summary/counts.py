"""Counter serialization for activation archive summaries."""
from __future__ import annotations

from collections import Counter
from typing import Dict


def counter_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


__all__ = ["counter_dict"]
